"""Per-position covert-channel dump for the C(b) capacity sweep.

Reuses the EXACT DiFR machinery (cached FP8-teacher logits in /root/zkorch-difr,
metrics.post_gumbel_margin construction, dolly seed draw, tiling, Gumbel seeding
seed+1+pi) from difr_baseline.py / difr_faithful.py. Rather than collapsing to an
aggregate DiFR mean, it dumps, per token position t (8 prompts x 1024 = 8192
positions), the quantities the capacity formula needs:

  margin[t]    = (teacher preferred post-Gumbel score)
                 - (post-Gumbel score, under the teacher's z_ref, of the token the
                   student M_int serves)          [ >= 0; the post_gumbel_margin, UNCLAMPED ]
  cand_rank[t] = rank of the student-served token in the teacher's post-Gumbel
                 ordering (0 == teacher's preferred token); used for the top-K refinement.
  N_b[t, j]    = #{ vocab tokens v : (teacher preferred post-Gumbel score)
                                     - (post-Gumbel score of v) <= bgrid[j] }
                 ("within-margin" token count at threshold bgrid[j]; >= 1 always).

N_b depends ONLY on the teacher's post-Gumbel score distribution, so it is identical
across schemes for a fixed (seed, prompt); we recompute it per scheme anyway so each
dump is self-contained.

The Gumbel sampling temperature stays at the DiFR default 1.0 (the metric's
temperature, NOT the in-chain softmax temperature 8/128 that names the schemes).
Margins are kept UNCLAMPED here (difr_baseline clamps at delta_max=50 for the
aggregate mean); for the b-sweep we want the true exceed-fraction p(b), and any
position with margin>50 violates for every b<50 either way.

Schemes:
  baseline  - int_chain.IntChain      on /root/zkorch/stage2-official1/registration  (CPU)
  faithful  - int_chain.FaithfulChain on /root/zkorch/stage3v2-fa/registration       (CPU)
  codebook  - int_model_approximation CodebookLinear x14 linears-swap, forced_logits (GPU);
              keeps lm_head float, exactly like the FP8 teacher's own construction.

Run (seed must match the cached teacher logits, 20260611):
  IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python \
      capacity_dump.py --scheme faithful --seed 20260611
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

MEASURE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MEASURE)
PARETO = "/workspace/projects/int-model-approximation/results/llama_pareto"
sys.path.insert(0, PARETO)

from difr_baseline import (heldout_prompts, GpuLock, SCRATCH, MODEL_ID,
                           SEQ_LEN, N_HELDOUT, SF, DOLLY)

REG = {
    "baseline": "/root/zkorch/stage2-official1/registration",
    "faithful": "/root/zkorch/stage3v2-fa/registration",
}
VOCAB = 32000


def build_bgrid():
    """Threshold grid for b (nats). Dense near 0 (faithful margins ~1e-2..0.4) and
    out to ~55 (baseline margins, mean 9, p99 24). Saved with the dump so the
    analyzer uses the identical grid."""
    parts = [
        np.linspace(0.0, 0.1, 51),     # 0.002 steps
        np.linspace(0.1, 0.5, 41),     # 0.01  steps
        np.linspace(0.5, 2.0, 31),     # 0.05  steps
        np.linspace(2.0, 6.0, 41),     # 0.1   steps
        np.linspace(6.0, 12.0, 31),    # 0.2   steps
        np.linspace(12.0, 26.0, 29),   # 0.5   steps
        np.linspace(26.0, 55.0, 30),   # ~1    step
    ]
    return np.unique(np.concatenate(parts).round(6)).astype(np.float64)


def student_logits(scheme, seed, ids_list):
    """Return list of (SEQ, VOCAB) float32 student logit arrays, one per prompt."""
    import torch
    from transformers import AutoModelForCausalLM

    if scheme in ("baseline", "faithful"):
        from int_chain import IntChain, FaithfulChain
        model_cpu = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32).eval()
        if scheme == "baseline":
            chain = IntChain(REG["baseline"])
            chain.verify_weights_against_model(model_cpu)
            chain.set_head(model_cpu)
        else:
            chain = FaithfulChain(REG["faithful"])
            chain.verify_weights_against_model(model_cpu)
        print(f"[{scheme}] registered-weight provenance guard: OK")
        z_stu, stats = [], []
        emb = model_cpu.model.embed_tokens
        with torch.no_grad():
            for pi, ids in enumerate(ids_list):
                t0 = time.time()
                x = emb(ids[0])
                x0 = torch.round(x.float() * SF).to(torch.int32).numpy()
                chain.stats = {}
                final = chain.forward(x0)
                z = chain.logits(final)                 # float64 (SEQ, V)
                z_stu.append(z.astype(np.float32))
                stats.append({k: v for k, v in chain.stats.items()})
                print(f"[{scheme}] student prompt {pi}: chain {time.time()-t0:.1f}s")
        return z_stu, stats

    if scheme == "codebook":
        from int_model_approximation.__main__ import CodebookLinear
        from llama_difr import replace_linears, forced_logits
        z_stu, stats = [], []
        with GpuLock():
            student = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()
            n = replace_linears(student, lambda w, s, b: CodebookLinear(w, s, b))
            print(f"[codebook] {n} linears -> CodebookLinear (lm_head stays float)")
            with torch.no_grad():
                for pi, ids in enumerate(ids_list):
                    t0 = time.time()
                    z = forced_logits(student, ids.to("cuda"))     # (SEQ, V) float32
                    z_stu.append(z.cpu().numpy().astype(np.float32))
                    stats.append({"n_linears": n})
                    print(f"[codebook] student prompt {pi}: {time.time()-t0:.2f}s")
            del student
            torch.cuda.empty_cache()
        return z_stu, stats

    raise ValueError(scheme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["baseline", "faithful", "codebook"])
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    assert os.environ.get("IMA_TEACHER_KERNEL") == "fp8_scaled_mm", \
        "run with IMA_TEACHER_KERNEL=fp8_scaled_mm (harness teacher kernel)"

    import torch
    from transformers import AutoTokenizer

    os.makedirs(SCRATCH, exist_ok=True)
    prompts = heldout_prompts(a.seed)
    dolly_sha = hashlib.sha256(open(DOLLY, "rb").read()).hexdigest()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids_list = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        reps = -(-SEQ_LEN // ids.shape[1])
        ids_list.append(ids.repeat(1, reps)[:, :SEQ_LEN])

    # teacher logits MUST already be cached by difr_baseline.py (same seed).
    z_ref_paths = [os.path.join(SCRATCH, f"z_ref_{a.seed}_{pi}.npy")
                   for pi in range(len(ids_list))]
    for p in z_ref_paths:
        assert os.path.exists(p), f"missing cached teacher logits {p} (run difr_baseline first)"

    z_stu, stats = student_logits(a.scheme, a.seed, ids_list)

    bgrid = build_bgrid()
    bt = torch.from_numpy(bgrid).float()
    nb = len(bgrid)

    margins, cand_ranks, Nb = [], [], []
    agg_top1 = 0
    with GpuLock():
        bt = bt.to("cuda")
        for pi in range(len(ids_list)):
            z_ref = torch.from_numpy(np.load(z_ref_paths[pi])).to("cuda").float()
            z = torch.from_numpy(z_stu[pi]).to("cuda").float()
            g = torch.empty(z_ref.shape, device="cuda", dtype=torch.float32)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(a.seed + 1 + pi)      # IDENTICAL Gumbel seeding to difr_baseline
            g.exponential_(generator=gen).log_().neg_()

            z_ref_full = z_ref + g                # post-Gumbel teacher scores
            z_cand_full = z + g                   # post-Gumbel student scores
            ref_pref = z_ref_full.max(dim=-1).values            # (SEQ,)
            cand_tok = z_cand_full.argmax(dim=-1)               # (SEQ,) served token
            delta = ref_pref[:, None] - z_ref_full              # (SEQ, V) >= 0
            margin = delta.gather(-1, cand_tok[:, None]).squeeze(-1)   # (SEQ,)

            delta_sorted, _ = torch.sort(delta, dim=-1)         # ascending, (SEQ,V)
            # N_b[t,j] = #{v : delta[t,v] <= bgrid[j]}
            nb_t = torch.searchsorted(
                delta_sorted, bt[None, :].expand(delta.shape[0], nb).contiguous(), right=True)
            # rank of served token = #{v : delta_v < margin_t}
            rank = torch.searchsorted(
                delta_sorted, margin[:, None], right=False).squeeze(-1)

            margins.append(margin.cpu().numpy().astype(np.float64))
            cand_ranks.append(rank.cpu().numpy().astype(np.int32))
            Nb.append(nb_t.cpu().numpy().astype(np.int32))
            agg_top1 += int((margin == 0).sum().item())
            print(f"[{a.scheme}] prompt {pi}: margin_mean={float(margin.mean()):.4g} "
                  f"exact_agree={float((margin==0).float().mean()):.4f} "
                  f"max_Nb={int(nb_t[:, -1].max())}")
            del z_ref, z, g, z_ref_full, z_cand_full, delta, delta_sorted, nb_t
            torch.cuda.empty_cache()

    margins = np.concatenate(margins)            # (8192,)
    cand_ranks = np.concatenate(cand_ranks)      # (8192,)
    Nb = np.concatenate(Nb, axis=0)              # (8192, nb)
    out = os.path.join(MEASURE, f"capacity_dump_{a.scheme}_seed{a.seed}.npz")
    np.savez_compressed(
        out, scheme=a.scheme, seed=a.seed, vocab=VOCAB, bgrid=bgrid,
        margins=margins, cand_ranks=cand_ranks, Nb=Nb,
        dolly_sha256=dolly_sha)
    meta = {
        "scheme": a.scheme, "seed": a.seed, "vocab": VOCAB,
        "n_positions": int(margins.size),
        "exact_postgumbel_agreement": agg_top1 / margins.size,
        "p_at_b0(=1-agreement)": 1 - agg_top1 / margins.size,
        "margin_mean_unclamped": float(margins.mean()),
        "margin_p99_unclamped": float(np.percentile(margins, 99)),
        "n_bgrid": nb, "bgrid_max": float(bgrid[-1]),
        "dolly_sha256": dolly_sha, "npz": out,
        "stats": stats,
    }
    with open(os.path.join(MEASURE, f"capacity_dump_{a.scheme}_seed{a.seed}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in (
        "scheme", "exact_postgumbel_agreement", "p_at_b0(=1-agreement)",
        "margin_mean_unclamped", "margin_p99_unclamped", "n_bgrid")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
