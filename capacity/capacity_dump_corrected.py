"""Per-position covert-channel dump — CORRECTED threat-model orientation.

The original measure/capacity_dump.py computed margins with the roles SWAPPED
relative to the real threat model: it used the FP8 teacher as the verifier's
reference and the M_int argmax as the served token. In the actual protocol the
datacenter SERVES tokens from the fast quantized model (FP8) — that is what a
network tap observes — while the ZKP proves the INTEGER model M_int. The
verifier therefore checks the served FP8 tokens against the PROVEN M_int logits
within margin b. Corrected roles:

  REFERENCE (verifier's logits)   = M_int   (chain / codebook student)   [was: FP8 teacher]
  SERVED token (honest behaviour) = argmax of the FP8 fast model's
                                    post-Gumbel scores                    [was: M_int argmax]

Per token position t (8 prompts x 1024 = 8192 positions):

  margin[t]    = (M_int preferred post-Gumbel score)
                 - (M_int post-Gumbel score of the FP8-served token)   [ >= 0, UNCLAMPED ]
  cand_rank[t] = rank of the FP8-served token in M_int's post-Gumbel ordering
                 (0 == M_int's own preferred token); for the top-K refinement.
  N_b[t, j]    = #{ vocab v : (M_int preferred score)
                              - (M_int post-Gumbel score of v) <= bgrid[j] }

Unlike the swapped run, N_b now depends on the SCHEME (it is computed under each
scheme's own M_int logits), and the served token stream is the SAME across
schemes (always the FP8 argmax under the shared Gumbel draw).

Everything else is identical to the swapped run: same 8 dolly prompts
(heldout_prompts(seed)), same cached FP8 logits /root/zkorch-difr/z_ref_*.npy,
same Gumbel seeding seed+1+pi at metric temperature 1.0, same b grid (extended
upward only if a scheme's corrected margins exceed the old 55-nat cap), same
schemes. The M_int logits are recomputed via measure/capacity_dump.student_logits
(chains on CPU, codebook on GPU) — training/proving is NOT re-run.

Run (seed must match the cached teacher logits, 20260611):
  IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python \
      capacity_dump_corrected.py --scheme faithful --seed 20260611
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURE = os.path.join(os.path.dirname(HERE), "measure")
sys.path.insert(0, MEASURE)

from difr_baseline import heldout_prompts, GpuLock, SCRATCH, MODEL_ID, SEQ_LEN, DOLLY
from capacity_dump import build_bgrid, student_logits, VOCAB


def extend_bgrid(bgrid, margin_max):
    """Same 248-point grid as the swapped run; if corrected margins exceed its
    55-nat cap (possible — margins now live on the M_int logit scale), append
    ~1-nat steps up to just past the observed max so p(b) still reaches 0 and
    the b->inf self-check stays meaningful."""
    if margin_max <= bgrid[-1]:
        return bgrid
    extra = np.arange(np.ceil(bgrid[-1]) + 1.0, margin_max + 2.0, 1.0)
    extra = np.append(extra, margin_max * 1.02)
    return np.unique(np.concatenate([bgrid, extra]).round(6)).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["baseline", "faithful", "codebook"])
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    assert os.environ.get("IMA_TEACHER_KERNEL") == "fp8_scaled_mm", \
        "run with IMA_TEACHER_KERNEL=fp8_scaled_mm (harness teacher kernel)"

    import torch
    from transformers import AutoTokenizer

    prompts = heldout_prompts(a.seed)
    dolly_sha = hashlib.sha256(open(DOLLY, "rb").read()).hexdigest()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids_list = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        reps = -(-SEQ_LEN // ids.shape[1])
        ids_list.append(ids.repeat(1, reps)[:, :SEQ_LEN])

    z_fp8_paths = [os.path.join(SCRATCH, f"z_ref_{a.seed}_{pi}.npy")
                   for pi in range(len(ids_list))]
    for p in z_fp8_paths:
        assert os.path.exists(p), f"missing cached FP8 logits {p} (run difr_baseline first)"

    # M_int logits — the verifier's REFERENCE under the corrected orientation
    z_int, stats = student_logits(a.scheme, a.seed, ids_list)

    # ---- phase 1: margins / ranks / sorted deltas under M_int's post-Gumbel scores
    margins, cand_ranks, ds_cpu = [], [], []
    agg_top1 = 0
    with GpuLock():
        for pi in range(len(ids_list)):
            z_fp8 = torch.from_numpy(np.load(z_fp8_paths[pi])).to("cuda").float()
            z_ref = torch.from_numpy(z_int[pi]).to("cuda").float()   # M_int = reference
            g = torch.empty(z_fp8.shape, device="cuda", dtype=torch.float32)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(a.seed + 1 + pi)      # IDENTICAL Gumbel seeding to difr_baseline
            g.exponential_(generator=gen).log_().neg_()

            z_ref_full = z_ref + g                # post-Gumbel M_int scores (reference)
            z_srv_full = z_fp8 + g                # post-Gumbel FP8 scores
            ref_pref = z_ref_full.max(dim=-1).values            # (SEQ,)
            srv_tok = z_srv_full.argmax(dim=-1)                 # (SEQ,) FP8-SERVED token
            delta = ref_pref[:, None] - z_ref_full              # (SEQ, V) >= 0, under M_int
            margin = delta.gather(-1, srv_tok[:, None]).squeeze(-1)     # (SEQ,)

            delta_sorted, _ = torch.sort(delta, dim=-1)         # ascending, (SEQ,V)
            rank = torch.searchsorted(
                delta_sorted, margin[:, None], right=False).squeeze(-1)

            margins.append(margin.cpu().numpy().astype(np.float64))
            cand_ranks.append(rank.cpu().numpy().astype(np.int32))
            ds_cpu.append(delta_sorted.cpu())     # keep for phase-2 N_b (1 GB total)
            agg_top1 += int((margin == 0).sum().item())
            print(f"[{a.scheme}] prompt {pi}: margin_mean={float(margin.mean()):.4g} "
                  f"exact_agree={float((margin==0).float().mean()):.4f} "
                  f"margin_max={float(margin.max()):.4g}")
            del z_fp8, z_ref, g, z_ref_full, z_srv_full, delta, delta_sorted
            torch.cuda.empty_cache()

        # ---- phase 2: N_b on the (possibly extended) shared grid
        margins_all = np.concatenate(margins)
        bgrid = extend_bgrid(build_bgrid(), float(margins_all.max()))
        bt = torch.from_numpy(bgrid).float().to("cuda")
        nb = len(bgrid)
        Nb = []
        for pi in range(len(ids_list)):
            ds = ds_cpu[pi].to("cuda")
            nb_t = torch.searchsorted(
                ds, bt[None, :].expand(ds.shape[0], nb).contiguous(), right=True)
            Nb.append(nb_t.cpu().numpy().astype(np.int32))
            print(f"[{a.scheme}] prompt {pi}: max_Nb={int(nb_t[:, -1].max())}")
            del ds, nb_t
            torch.cuda.empty_cache()

    margins = margins_all                          # (8192,)
    cand_ranks = np.concatenate(cand_ranks)        # (8192,)
    Nb = np.concatenate(Nb, axis=0)                # (8192, nb)
    out = os.path.join(HERE, f"capacity_dump_corrected_{a.scheme}_seed{a.seed}.npz")
    np.savez_compressed(
        out, scheme=a.scheme, seed=a.seed, vocab=VOCAB, bgrid=bgrid,
        margins=margins, cand_ranks=cand_ranks, Nb=Nb,
        dolly_sha256=dolly_sha, orientation="corrected: ref=M_int, served=FP8 argmax")
    meta = {
        "scheme": a.scheme, "seed": a.seed, "vocab": VOCAB,
        "orientation": "corrected: reference=M_int (proven), served=FP8 fast-model argmax",
        "n_positions": int(margins.size),
        "exact_postgumbel_agreement": agg_top1 / margins.size,
        "p_at_b0(=1-agreement)": 1 - agg_top1 / margins.size,
        "margin_mean_unclamped": float(margins.mean()),
        "margin_p99_unclamped": float(np.percentile(margins, 99)),
        "margin_max_unclamped": float(margins.max()),
        "n_bgrid": int(nb), "bgrid_max": float(bgrid[-1]),
        "bgrid_extended_beyond_55": bool(bgrid[-1] > 55.0),
        "dolly_sha256": dolly_sha, "npz": out,
        "stats": stats,
    }
    with open(os.path.join(HERE, f"capacity_dump_corrected_{a.scheme}_seed{a.seed}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in (
        "scheme", "orientation", "exact_postgumbel_agreement", "p_at_b0(=1-agreement)",
        "margin_mean_unclamped", "margin_p99_unclamped", "margin_max_unclamped",
        "n_bgrid", "bgrid_max")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
