"""Decompose the baseline-native DiFR gap into (architecture) + (integerization).

float64 PIPELINE-ARCHITECTURE REPLICA: the exact function the witness chain
integerizes, computed in real arithmetic with NO rounding:
  - weights w.float() (not regridded), embedding fp32
  - rmsnorm: x * 1/sqrt(mean(x^2)+eps) * g          (the R bracket's target)
  - q/k/v proj, RoPE (theta=10000 float64), NO o_proj
  - softmax temperature 128 over the causal mask    (m68-pipeline /2^20 quirk)
  - line-157 pi permutation of the head concat
  - swiglu silu(g)*u, final norm + lm_head
Measured per prompt (same harness prompts/Gumbel protocol as difr_baseline.py):
  (b) replica vs FP8 teacher  -> architecture-only gap
  (c) integer chain vs replica -> pure integerization drift (the
      ROPE_ATTENTION_DESIGN section 9.6 / ORCHESTRATOR_REPORT caveat-1 number,
      'new witness vs pipeline output')

Run: IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python \
       decompose_difr.py --seed 20260611
(teacher logits must already be cached by difr_baseline.py)
"""
import argparse
import fcntl
import json
import os
import statistics
import sys

import numpy as np

MEASURE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MEASURE)
sys.path.insert(0, "/workspace/projects/int-model-approximation/results/llama_pareto")

from difr_baseline import heldout_prompts, GpuLock, SCRATCH, MODEL_ID, SEQ_LEN, SF

SEQ, EMBED, INTER, HD, NH = 1024, 768, 3072, 64, 12
EPS = 1e-6


def rmsnorm(x, g):
    return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + EPS) * g[None, :]


def float_replica_logits(model, ids):
    import torch
    with torch.no_grad():
        x = model.model.embed_tokens(ids[0]).detach().double().numpy()        # (SEQ, EMBED)
    L = {}
    for l in range(2):
        layer = model.model.layers[l]
        L[l] = {
            "g_in": layer.input_layernorm.weight.detach().double().numpy(),
            "g_post": layer.post_attention_layernorm.weight.detach().double().numpy(),
            "Wq": layer.self_attn.q_proj.weight.detach().double().numpy().T,
            "Wk": layer.self_attn.k_proj.weight.detach().double().numpy().T,
            "Wv": layer.self_attn.v_proj.weight.detach().double().numpy().T,
            "Wg": layer.mlp.gate_proj.weight.detach().double().numpy().T,
            "Wu": layer.mlp.up_proj.weight.detach().double().numpy().T,
            "Wd": layer.mlp.down_proj.weight.detach().double().numpy().T,
        }
    g_f = model.model.norm.weight.detach().double().numpy()
    W_lm = model.lm_head.weight.detach().double().numpy().T

    # RoPE tables (theta=10000, default rope, positions 0..SEQ)
    half = HD // 2
    inv_freq = 10000.0 ** (-np.arange(half) / half)
    ang = np.arange(SEQ)[:, None] * inv_freq[None, :]
    ang = np.concatenate([ang, ang], axis=1)                          # (SEQ, HD)
    e = np.arange(EMBED)
    cosW = np.cos(ang)[:, e % HD]
    sigma = np.where((e & 32) != 0, 1.0, -1.0)
    sinW = sigma[None, :] * np.sin(ang)[:, e % HD]
    flip = e ^ 32
    mask = np.tril(np.ones((SEQ, SEQ)))

    for l in range(2):
        w = L[l]
        a_in = rmsnorm(x, w["g_in"])
        Q, K, V = a_in @ w["Wq"], a_in @ w["Wk"], a_in @ w["Wv"]
        Qr = Q * cosW + Q[:, flip] * sinW
        Kr = K * cosW + K[:, flip] * sinW
        heads = []
        for h in range(NH):
            sl = slice(64 * h, 64 * h + 64)
            s = Qr[:, sl] @ Kr[:, sl].T                               # raw QK^T
            E = np.exp(s / 128.0) * mask                              # temp 128 (pipeline quirk)
            P = E / E.sum(axis=1, keepdims=True)
            heads.append(P @ V[:, sl])
        M = np.concatenate(heads, axis=1)
        attn_out = M.T.reshape(SEQ, EMBED)                            # line-157 pi
        z1 = x + attn_out
        a_post = rmsnorm(z1, w["g_post"])
        gate = a_post @ w["Wg"]
        up = a_post @ w["Wu"]
        hidden = gate / (1.0 + np.exp(-gate)) * up                    # silu(g)*u
        x = z1 + hidden @ w["Wd"]
    return rmsnorm(x, g_f) @ W_lm                                     # (SEQ, V) float64


def metric_block(z_ref_t, z_t, gumbel_seed):
    import torch
    from int_model_approximation import metrics as M
    g = torch.empty(z_ref_t.shape, device=z_ref_t.device, dtype=torch.float32)
    gen = torch.Generator(device=z_ref_t.device)
    gen.manual_seed(gumbel_seed)
    g.exponential_(generator=gen).log_().neg_()
    margin = M.post_gumbel_margin(z_ref_t, z_t, g)
    d_abs = (z_ref_t.float() - z_t.float()).abs()
    return {
        "difr_mean": float(margin.mean()),
        "difr_p99": float(margin.flatten().quantile(0.99)),
        "logit_l2_mean": float(M.logit_l2(z_ref_t, z_t).mean()),
        "top1": float(M.top1_match(z_ref_t, z_t).float().mean()),
        "argmax_flips": int((~M.top1_match(z_ref_t, z_t)).sum()),
        "max_abs_logit_delta": float(d_abs.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from int_chain import IntChain

    prompts = heldout_prompts(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    chain = IntChain("/root/zkorch/stage2-official1/registration")
    chain.verify_weights_against_model(model)
    chain.set_head(model)

    rep_vs_teacher, chain_vs_rep = [], []
    z_rep_all, z_chain_all = [], []
    emb = model.model.embed_tokens
    for pi, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        reps = -(-SEQ_LEN // ids.shape[1])
        ids = ids.repeat(1, reps)[:, :SEQ_LEN]
        z_rep = float_replica_logits(model, ids).astype(np.float32)
        with torch.no_grad():
            x0 = torch.round(emb(ids[0]).float() * SF).to(torch.int32).numpy()
        z_chain = chain.logits(chain.forward(x0)).astype(np.float32)
        z_rep_all.append(z_rep)
        z_chain_all.append(z_chain)
        print(f"prompt {pi}: replica+chain forwards done")

    with GpuLock():
        for pi in range(len(prompts)):
            z_ref = torch.from_numpy(np.load(os.path.join(SCRATCH, f"z_ref_{a.seed}_{pi}.npy"))).to("cuda")
            z_rep = torch.from_numpy(z_rep_all[pi]).to("cuda")
            z_ch = torch.from_numpy(z_chain_all[pi]).to("cuda")
            rep_vs_teacher.append(metric_block(z_ref, z_rep, a.seed + 1 + pi))
            chain_vs_rep.append(metric_block(z_rep, z_ch, a.seed + 1 + pi))
            print(f"prompt {pi}: rep_vs_teacher {rep_vs_teacher[-1]}")
            print(f"prompt {pi}: chain_vs_rep  {chain_vs_rep[-1]}")

    def agg(rows):
        out = {k: statistics.mean(r[k] for r in rows) for k in rows[0] if k != "argmax_flips"}
        out["argmax_flips_total"] = sum(r["argmax_flips"] for r in rows)
        return out

    out = {
        "seed": a.seed,
        "replica_vs_teacher (architecture-only gap)": {"per_prompt": rep_vs_teacher,
                                                       "aggregate": agg(rep_vs_teacher)},
        "chain_vs_replica (pure integerization drift)": {"per_prompt": chain_vs_rep,
                                                         "aggregate": agg(chain_vs_rep)},
    }
    res = os.path.join(MEASURE, f"difr_decomposition_seed{a.seed}.json")
    with open(res, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v["aggregate"] for k, v in out.items() if isinstance(v, dict)}, indent=2))
    print(f"wrote {res}")


if __name__ == "__main__":
    main()
