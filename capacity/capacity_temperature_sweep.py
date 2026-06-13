"""Temperature-sensitivity sweep for the FAITHFUL covert-channel capacity.

CORRECTED orientation (reference = M_int post-Gumbel, served = FP8 argmax of the
post-Gumbel scores), identical to capacity_dump_corrected.py, but the Gumbel
term is rescaled by a sampling temperature T:

    post-Gumbel score(v) = logits(v) + T * g_sigma(v)

with g_sigma the SAME public Gumbel draw used everywhere (seed seed+1+pi). The
served token is argmax_v(z_fp8 + T*g); the verifier's reference is z_int + T*g;
margins / cand_ranks / N_b are all re-derived under z_int + T*g. T=1 reproduces
the headline corrected-faithful run byte-for-byte (same seed, same prompts, same
b grid). This quantifies how temperature-dependent the headline number is.

The DiFR metric margin is logits + T*g; here T IS that sampling temperature and
it is applied CONSISTENTLY to both the served-token sampling (z_fp8 + T*g) and
the reference margin / N_b (z_int + T*g). Greedy decoding is the T->0 limit.

The expensive M_int (FaithfulChain, CPU) logits are recomputed once and cached
to z_int_faithful_seed{seed}.npz; the per-T sweep is pure GPU arithmetic on the
cached z_int / z_fp8 logits. Capacity is computed with the UNCHANGED
capacity_analyze.analyze() (we stage each T as a temp npz and call analyze on it)
so the math is byte-identical to the headline pipeline.

Run:
  IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python \
      capacity_temperature_sweep.py --seed 20260611
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURE = os.path.join(os.path.dirname(HERE), "measure")
sys.path.insert(0, MEASURE)

from difr_baseline import heldout_prompts, GpuLock, SCRATCH, MODEL_ID, SEQ_LEN, DOLLY
from capacity_dump import build_bgrid, student_logits, VOCAB
from capacity_dump_corrected import extend_bgrid
import capacity_analyze as ca

SCHEME = "faithful"
# Task-requested sweep set: {0.3,0.5,0.7,1.0,1.3,2.0}. We also probe the near-greedy
# limit (0.05,0.1,0.2) to characterise T->0 explicitly -- it does NOT vanish, see
# CAPACITY_TEMPERATURE.md (the served FP8 / proven M_int argmax gap is deterministic).
TEMPS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 2.0]
TEMPS_REQUESTED = [0.3, 0.5, 0.7, 1.0, 1.3, 2.0]


def get_z_int(seed, ids_list):
    """Recompute (and cache) the M_int faithful logits, one (SEQ,V) array / prompt."""
    cache = os.path.join(HERE, f"z_int_{SCHEME}_seed{seed}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        print(f"[z_int] loaded cache {cache}")
        return [d[f"p{pi}"] for pi in range(len(ids_list))]
    z_int, _ = student_logits(SCHEME, seed, ids_list)
    np.savez_compressed(cache, **{f"p{pi}": z_int[pi] for pi in range(len(z_int))})
    print(f"[z_int] cached {cache}")
    return z_int


def sweep_one_T(T, z_int, z_fp8_paths, seed, bgrid_base):
    """Return a capacity-analyze result dict for sampling temperature T."""
    import torch
    margins, cand_ranks, ds_cpu = [], [], []
    agg_top1 = 0
    n_pos = 0
    with GpuLock():
        for pi in range(len(z_int)):
            z_fp8 = torch.from_numpy(np.load(z_fp8_paths[pi])).to("cuda").float()
            z_ref = torch.from_numpy(z_int[pi]).to("cuda").float()    # M_int = reference
            g = torch.empty(z_fp8.shape, device="cuda", dtype=torch.float32)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seed + 1 + pi)        # IDENTICAL Gumbel seeding to difr_baseline
            g.exponential_(generator=gen).log_().neg_()
            g.mul_(T)                             # <-- sampling temperature scales the Gumbel term

            z_ref_full = z_ref + g                # post-Gumbel M_int scores (reference)
            z_srv_full = z_fp8 + g                # post-Gumbel FP8 scores
            ref_pref = z_ref_full.max(dim=-1).values
            srv_tok = z_srv_full.argmax(dim=-1)                 # FP8-SERVED token
            delta = ref_pref[:, None] - z_ref_full              # (SEQ,V) >= 0, under M_int
            margin = delta.gather(-1, srv_tok[:, None]).squeeze(-1)

            delta_sorted, _ = torch.sort(delta, dim=-1)
            rank = torch.searchsorted(
                delta_sorted, margin[:, None], right=False).squeeze(-1)

            margins.append(margin.cpu().numpy().astype(np.float64))
            cand_ranks.append(rank.cpu().numpy().astype(np.int32))
            ds_cpu.append(delta_sorted.cpu())
            agg_top1 += int((margin == 0).sum().item())
            n_pos += margin.numel()
            del z_fp8, z_ref, g, z_ref_full, z_srv_full, delta, delta_sorted
            torch.cuda.empty_cache()

        margins_all = np.concatenate(margins)
        bgrid = extend_bgrid(bgrid_base, float(margins_all.max()))
        bt = torch.from_numpy(bgrid).float().to("cuda")
        nb = len(bgrid)
        Nb = []
        for pi in range(len(z_int)):
            ds = ds_cpu[pi].to("cuda")
            nb_t = torch.searchsorted(
                ds, bt[None, :].expand(ds.shape[0], nb).contiguous(), right=True)
            Nb.append(nb_t.cpu().numpy().astype(np.int32))
            del ds, nb_t
            torch.cuda.empty_cache()

    margins_all = np.concatenate(margins)
    cand_ranks = np.concatenate(cand_ranks)
    Nb = np.concatenate(Nb, axis=0)
    agreement = agg_top1 / n_pos

    # stage as a temp npz and reuse the UNCHANGED analyze() for byte-identical math
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        tmp = tf.name
    np.savez_compressed(tmp, scheme=SCHEME, seed=seed, vocab=VOCAB, bgrid=bgrid,
                        margins=margins_all, cand_ranks=cand_ranks, Nb=Nb)
    res = ca.analyze(tmp)
    os.unlink(tmp)
    res["T"] = T
    res["exact_postgumbel_agreement"] = agreement
    res["p_at_b0"] = 1.0 - agreement
    res["margin_mean_unclamped"] = float(margins_all.mean())
    res["margin_p99_unclamped"] = float(np.percentile(margins_all, 99))
    res["margin_max_unclamped"] = float(margins_all.max())
    res["bgrid_max"] = float(bgrid[-1])
    return res


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ts = [r["T"] for r in results]
    c_simple = [r["min_simple"]["C_min_bits_per_token"] for r in results]
    c_topk = [r["min_topK"]["C_min_bits_per_token"] for r in results]
    p0 = [r["p_at_b0"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(Ts, c_simple, "o-", color="#2471a3", label="min$_b$ C(b) simple")
    ax1.plot(Ts, c_topk, "s--", color="#1e8449", label="min$_{b,K}$ C(b) top-K (K=16)")
    for x, y in zip(Ts, c_simple):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), fontsize=8, ha="center")
    ax1.axvline(1.0, color="gray", ls=":", lw=1, label="T=1 (headline)")
    ax1.set_xlabel("sampling temperature T")
    ax1.set_ylabel("worst-case covert capacity  min$_b$ C(b)  (bits / token)")
    ax1.set_title("Faithful-arch-v1 covert capacity vs sampling temperature\n"
                  "(corrected orientation: ref=M_int post-Gumbel, served=FP8 argmax(logits+T·g))")
    ax1.legend(fontsize=9, loc="lower left"); ax1.grid(alpha=0.3); ax1.set_ylim(0, 0.55)

    ax2.plot(Ts, p0, "o-", color="#c0392b", label="p(b=0) = 1 - exact agreement")
    for x, y in zip(Ts, p0):
        ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), fontsize=8, ha="center")
    ax2.axvline(1.0, color="gray", ls=":", lw=1)
    ax2.set_xlabel("sampling temperature T")
    ax2.set_ylabel("exact-disagreement rate p(b=0)")
    ax2.set_title("Post-Gumbel disagreement rate vs T\n(T->0: deterministic FP8-vs-M_int argmax gap)")
    ax2.legend(fontsize=9, loc="lower left"); ax2.grid(alpha=0.3); ax2.set_ylim(0, 0.092)

    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    assert os.environ.get("IMA_TEACHER_KERNEL") == "fp8_scaled_mm", \
        "run with IMA_TEACHER_KERNEL=fp8_scaled_mm (harness teacher kernel)"

    from transformers import AutoTokenizer
    prompts = heldout_prompts(a.seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids_list = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        reps = -(-SEQ_LEN // ids.shape[1])
        ids_list.append(ids.repeat(1, reps)[:, :SEQ_LEN])

    z_fp8_paths = [os.path.join(SCRATCH, f"z_ref_{a.seed}_{pi}.npy")
                   for pi in range(len(ids_list))]
    for p in z_fp8_paths:
        assert os.path.exists(p), f"missing cached FP8 logits {p}"

    z_int = get_z_int(a.seed, ids_list)
    bgrid_base = build_bgrid()

    results = []
    for T in TEMPS:
        r = sweep_one_T(T, z_int, z_fp8_paths, a.seed, bgrid_base)
        results.append(r)
        print(f"[T={T}] min_simple C = {r['min_simple']['C_min_bits_per_token']:.4f} "
              f"b/tok @ b*={r['min_simple']['b_star']:.4g} (p*={r['min_simple']['p_star']:.4g}); "
              f"min_topK = {r['min_topK']['C_min_bits_per_token']:.4f}; "
              f"p(b=0)={r['p_at_b0']:.4f}; margin_mean={r['margin_mean_unclamped']:.4g} "
              f"p99={r['margin_p99_unclamped']:.4g} max={r['margin_max_unclamped']:.4g}")

    # keep the result JSON compact: drop the long per-grid arrays
    slim = []
    for r in results:
        slim.append({k: v for k, v in r.items()
                     if k not in ("b", "p", "C_simple", "C_topK", "q", "e_logNb")})
    out = os.path.join(HERE, f"capacity_temperature_results_seed{a.seed}.json")
    with open(out, "w") as f:
        json.dump({"scheme": SCHEME, "seed": a.seed, "temps": TEMPS,
                   "orientation": "corrected: ref=M_int post-Gumbel, served=FP8 argmax(logits+T*g)",
                   "results": slim}, f, indent=2)
    print(f"wrote {out}")
    plot(results, os.path.join(os.path.dirname(HERE), "capacity_vs_T.png"))

    print("\n=== T-sweep summary (faithful, corrected orientation) ===")
    print(f"{'T':>5} {'min_simple':>12} {'min_topK':>10} {'b*_simple':>10} "
          f"{'p(b=0)':>9} {'margin_mean':>12} {'margin_max':>11}")
    for r in results:
        print(f"{r['T']:>5} {r['min_simple']['C_min_bits_per_token']:>12.4f} "
              f"{r['min_topK']['C_min_bits_per_token']:>10.4f} "
              f"{r['min_simple']['b_star']:>10.4g} {r['p_at_b0']:>9.4f} "
              f"{r['margin_mean_unclamped']:>12.4g} {r['margin_max_unclamped']:>11.4g}")


if __name__ == "__main__":
    main()
