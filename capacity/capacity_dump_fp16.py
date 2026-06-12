"""Per-position covert-channel dump — bf16 FULL-PRECISION reference ("FP16 ref").

Sanity-check variant of the corrected-orientation sweep (CAPACITY_CORRECTED.md):
the published runs always had the FP8 teacher on one side, but FP8 is itself a
lossy quantization. Here the verifier's reference is the GENUINE full-precision
model and the served stream is each scheme's quantized M_int:

  REFERENCE (verifier's logits)   = plain bf16 JackFram/llama-68m, NO FP8Linear
                                    swap (z_fp16ref cache via fp16_ref_logits.py)
  SERVED token                    = argmax of the scheme's M_int post-Gumbel scores

This keeps the corrected threat-model orientation — the reference is the model
the verifier checks against, the served stream is what the datacenter actually
emits — instantiated for the "true-model verifier": there is no separate FP8
fast model in this variant; M_int IS the served model. Per token position t
(8 prompts x 1024 = 8192):

  margin[t]    = (bf16-ref preferred post-Gumbel score)
                 - (bf16-ref post-Gumbel score of the M_int-served token)  [>=0, UNCLAMPED]
  cand_rank[t] = rank of the M_int-served token in the bf16 ref's post-Gumbel ordering
  N_b[t, j]    = #{ vocab v : (bf16 pref) - (bf16 post-Gumbel score of v) <= bgrid[j] }

N_b and the top-K set are SCHEME-INDEPENDENT here (shared bf16 reference; we
recompute them per scheme anyway so each dump stays self-contained); the served
stream varies by scheme. Everything else is identical to the published runs:
same 8 dolly prompts, same Gumbel seeding seed+1+pi at metric temperature 1.0,
same 248-point b grid (extended past 55 only if margins demand it), same scheme
logits via measure/capacity_dump.student_logits (no re-training/re-proving).

REGISTRATION RELOCATION (baseline): /root/zkorch/stage2-official1 no longer
exists on this box; the baseline IntChain is pointed at
/root/zkorch/stage3-official1/registration, which carries the same
baseline-architecture registration (no o_proj weights, no softmax8 table; all
4 nonlinear tables byte-identical to the validated stage3v2-fa copies). Two
guards make this safe: (1) IntChain.verify_weights_against_model byte-checks
every registered weight against round(w.float().T*2^16) of the HF checkpoint —
the same derivation stage2-official1 passed; (2) --xcheck (default ON) requires
the swapped-orientation margins recomputed from THESE M_int logits against the
cached FP8 logits to reproduce measure/capacity_dump_{scheme}_seed{seed}.npz
(produced while stage2-official1 still existed) to float32 precision.

Run (no IMA_TEACHER_KERNEL — the bf16 reference has no FP8Linear and the
codebook student never reads the teacher-kernel switch):
  /root/int-model-env/bin/python capacity_dump_fp16.py --scheme faithful --seed 20260611
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
import capacity_dump as cd
from capacity_dump import build_bgrid, student_logits, VOCAB
from capacity_dump_corrected import extend_bgrid

if not os.path.isdir(cd.REG["baseline"]):
    cd.REG["baseline"] = "/root/zkorch/stage3-official1/registration"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["baseline", "faithful", "codebook"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--no-xcheck", action="store_true",
                    help="skip reproduction of the swapped-run dump margins")
    a = ap.parse_args()
    assert "IMA_TEACHER_KERNEL" not in os.environ, \
        "run WITHOUT IMA_TEACHER_KERNEL (nothing in this run uses the FP8 teacher GEMM)"

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

    z16_paths = [os.path.join(SCRATCH, f"z_fp16ref_{a.seed}_{pi}.npy")
                 for pi in range(len(ids_list))]
    for p in z16_paths:
        assert os.path.exists(p), f"missing bf16 reference logits {p} (run fp16_ref_logits first)"
    z_fp8_paths = [os.path.join(SCRATCH, f"z_ref_{a.seed}_{pi}.npy")
                   for pi in range(len(ids_list))]

    old_margins = None
    if not a.no_xcheck:
        old_npz = os.path.join(MEASURE, f"capacity_dump_{a.scheme}_seed{a.seed}.npz")
        assert os.path.exists(old_npz), f"xcheck needs {old_npz}"
        old_margins = np.load(old_npz)["margins"]          # swapped run: ref=FP8, served=M_int
        for p in z_fp8_paths:
            assert os.path.exists(p), f"xcheck needs cached FP8 logits {p}"

    # M_int logits — the SERVED model under this orientation
    z_int, stats = student_logits(a.scheme, a.seed, ids_list)

    # ---- phase 1: margins / ranks / sorted deltas under the bf16 ref's post-Gumbel scores
    margins, cand_ranks, ds_cpu = [], [], []
    agg_top1 = 0
    xcheck_maxdiff = 0.0
    with GpuLock():
        for pi in range(len(ids_list)):
            z16 = torch.from_numpy(np.load(z16_paths[pi])).to("cuda").float()  # reference
            z_srv = torch.from_numpy(z_int[pi]).to("cuda").float()             # served M_int
            g = torch.empty(z16.shape, device="cuda", dtype=torch.float32)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(a.seed + 1 + pi)      # IDENTICAL Gumbel seeding to difr_baseline
            g.exponential_(generator=gen).log_().neg_()

            z_ref_full = z16 + g                  # post-Gumbel bf16 reference scores
            z_srv_full = z_srv + g                # post-Gumbel M_int scores
            ref_pref = z_ref_full.max(dim=-1).values            # (SEQ,)
            srv_tok = z_srv_full.argmax(dim=-1)                 # (SEQ,) M_int-SERVED token
            delta = ref_pref[:, None] - z_ref_full              # (SEQ, V) >= 0, under bf16 ref
            margin = delta.gather(-1, srv_tok[:, None]).squeeze(-1)     # (SEQ,)

            delta_sorted, _ = torch.sort(delta, dim=-1)         # ascending, (SEQ,V)
            rank = torch.searchsorted(
                delta_sorted, margin[:, None], right=False).squeeze(-1)

            if old_margins is not None:
                # reproduce the swapped-run margin (ref=FP8, served=THIS srv_tok)
                z8 = torch.from_numpy(np.load(z_fp8_paths[pi])).to("cuda").float()
                s8 = z8 + g
                m8 = (s8.max(dim=-1).values
                      - s8.gather(-1, srv_tok[:, None]).squeeze(-1)).cpu().numpy()
                dmax = float(np.abs(m8 - old_margins[pi * SEQ_LEN:(pi + 1) * SEQ_LEN]).max())
                xcheck_maxdiff = max(xcheck_maxdiff, dmax)
                assert dmax < 1e-4, \
                    f"xcheck FAILED prompt {pi}: served stream does not reproduce the " \
                    f"swapped-run dump (max margin diff {dmax})"
                del z8, s8

            margins.append(margin.cpu().numpy().astype(np.float64))
            cand_ranks.append(rank.cpu().numpy().astype(np.int32))
            ds_cpu.append(delta_sorted.cpu())     # keep for phase-2 N_b (~1 GB total)
            agg_top1 += int((margin == 0).sum().item())
            print(f"[{a.scheme}] prompt {pi}: margin_mean={float(margin.mean()):.4g} "
                  f"exact_agree={float((margin==0).float().mean()):.4f} "
                  f"margin_max={float(margin.max()):.4g}")
            del z16, z_srv, g, z_ref_full, z_srv_full, delta, delta_sorted
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
    out = os.path.join(HERE, f"capacity_dump_fp16_{a.scheme}_seed{a.seed}.npz")
    np.savez_compressed(
        out, scheme=a.scheme, seed=a.seed, vocab=VOCAB, bgrid=bgrid,
        margins=margins, cand_ranks=cand_ranks, Nb=Nb,
        dolly_sha256=dolly_sha,
        orientation="fp16-ref: ref=bf16 full-precision (no FP8 swap), served=M_int argmax")
    meta = {
        "scheme": a.scheme, "seed": a.seed, "vocab": VOCAB,
        "orientation": "fp16-ref: reference=plain bf16 model (no FP8Linear), "
                       "served=M_int post-Gumbel argmax",
        "baseline_registration": cd.REG["baseline"] if a.scheme == "baseline" else None,
        "n_positions": int(margins.size),
        "exact_postgumbel_agreement": agg_top1 / margins.size,
        "p_at_b0(=1-agreement)": 1 - agg_top1 / margins.size,
        "margin_mean_unclamped": float(margins.mean()),
        "margin_p99_unclamped": float(np.percentile(margins, 99)),
        "margin_max_unclamped": float(margins.max()),
        "n_bgrid": int(nb), "bgrid_max": float(bgrid[-1]),
        "bgrid_extended_beyond_55": bool(bgrid[-1] > 55.0),
        "xcheck_vs_swapped_dump_max_margin_diff": (
            None if old_margins is None else xcheck_maxdiff),
        "dolly_sha256": dolly_sha, "npz": out,
        "stats": stats,
    }
    with open(os.path.join(HERE, f"capacity_dump_fp16_{a.scheme}_seed{a.seed}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in (
        "scheme", "orientation", "exact_postgumbel_agreement", "p_at_b0(=1-agreement)",
        "margin_mean_unclamped", "margin_p99_unclamped", "margin_max_unclamped",
        "n_bgrid", "bgrid_max", "xcheck_vs_swapped_dump_max_margin_diff")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
