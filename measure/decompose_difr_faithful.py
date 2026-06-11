"""Decompose the faithful-arch-v1 DiFR into (architecture + weight grid) +
(integerization) — the decompose_difr.py pattern with the replica now being
the FAITHFUL architecture (STAGE3 §5.2 leg 1 / leg 2).

float64 FAITHFUL-ARCHITECTURE REPLICA: the exact function faithful-arch-v1
integerizes, computed in real arithmetic with NO rounding:
  - weights w.float() (not regridded), embedding fp32
  - rmsnorm: x * 1/sqrt(mean(x^2)+eps) * g
  - q/k/v proj, RoPE (theta=10000 float64), softmax TEMPERATURE 8 over the
    causal mask (allowed-max shift — shift-invariant, exact in real arith),
  - PLAIN head concat (no line-157 permutation), o_proj APPLIED,
  - swiglu silu(g)*u, final norm + lm_head
Measured per prompt (same harness prompts/Gumbel protocol):
  (b) replica vs FP8 teacher   -> architecture + 2^-16-weight-grid-free gap
  (c) integer chain vs replica -> pure integerization drift (leg 1)

Run: IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python \
       decompose_difr_faithful.py --seed 20260611
(teacher logits must already be cached by difr_baseline.py / difr_faithful.py)
"""
import argparse
import json
import os
import statistics
import sys

import numpy as np

MEASURE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MEASURE)
sys.path.insert(0, "/workspace/projects/int-model-approximation/results/llama_pareto")

from difr_baseline import heldout_prompts, GpuLock, SCRATCH, MODEL_ID, SEQ_LEN, SF
from decompose_difr import rmsnorm, metric_block

SEQ, EMBED, INTER, HD, NH = 1024, 768, 3072, 64, 12
REG = "/root/zkorch/stage3v2-fa/registration"


def float_replica_logits(model, ids):
    """Faithful llama-68m in float64: temp 8, plain concat, o_proj applied."""
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
            "Wo": layer.self_attn.o_proj.weight.detach().double().numpy().T,
            "Wg": layer.mlp.gate_proj.weight.detach().double().numpy().T,
            "Wu": layer.mlp.up_proj.weight.detach().double().numpy().T,
            "Wd": layer.mlp.down_proj.weight.detach().double().numpy().T,
        }
    g_f = model.model.norm.weight.detach().double().numpy()
    W_lm = model.lm_head.weight.detach().double().numpy().T

    half = HD // 2
    inv_freq = 10000.0 ** (-np.arange(half) / half)
    ang = np.arange(SEQ)[:, None] * inv_freq[None, :]
    ang = np.concatenate([ang, ang], axis=1)                          # (SEQ, HD)
    e = np.arange(EMBED)
    cosW = np.cos(ang)[:, e % HD]
    sigma = np.where((e & 32) != 0, 1.0, -1.0)
    sinW = sigma[None, :] * np.sin(ang)[:, e % HD]
    flip = e ^ 32
    mask = np.tril(np.ones((SEQ, SEQ), dtype=bool))

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
            sm = np.where(mask, s, -np.inf).max(axis=1, keepdims=True)
            E = np.where(mask, np.exp((s - sm) / 8.0), 0.0)           # temp 8, max-shift
            P = E / E.sum(axis=1, keepdims=True)
            heads.append(P @ V[:, sl])
        M = np.concatenate(heads, axis=1)                             # plain concat
        attn_out = M @ w["Wo"]                                        # o_proj applied
        z1 = x + attn_out
        a_post = rmsnorm(z1, w["g_post"])
        gate = a_post @ w["Wg"]
        up = a_post @ w["Wu"]
        hidden = gate / (1.0 + np.exp(-gate)) * up                    # silu(g)*u
        x = z1 + hidden @ w["Wd"]
    return rmsnorm(x, g_f) @ W_lm                                     # (SEQ, V) float64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from int_chain import FaithfulChain

    prompts = heldout_prompts(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    chain = FaithfulChain(REG)
    chain.verify_weights_against_model(model)

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
        chain.stats = {}
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
        "replica": "float64 faithful llama-68m (temp 8, plain concat, o_proj; "
                   "original float weights, no grid)",
        "replica_vs_teacher (architecture-only gap)": {"per_prompt": rep_vs_teacher,
                                                       "aggregate": agg(rep_vs_teacher)},
        "chain_vs_replica (integerization + weight-grid drift)": {"per_prompt": chain_vs_rep,
                                                                  "aggregate": agg(chain_vs_rep)},
    }
    res = os.path.join(MEASURE, f"difr_decomposition_faithful_seed{a.seed}.json")
    with open(res, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v["aggregate"] for k, v in out.items() if isinstance(v, dict)}, indent=2))
    print(f"wrote {res}")


if __name__ == "__main__":
    main()
