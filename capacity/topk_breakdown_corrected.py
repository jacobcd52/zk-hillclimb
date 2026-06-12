"""Top-K (Rinberg) capacity rule on the CORRECTED-orientation dumps.

Reuses topk_breakdown.py's Dump class (exact-grid sweep + dense refinement,
five-component breakdown, K sweep) unchanged, pointed at
capacity_dump_corrected_{scheme}_seed{seed}.npz (reference = M_int, served =
FP8 argmax). Under the corrected orientation cand_rank is the FP8-served
token's rank in M_INT's post-Gumbel ordering, so "top-K" now means the
verifier's (proven model's) top-K — the semantically right set for the
chain-rule refinement.

Outputs:
  capacity/topk_corrected_results_seed{seed}.json
  ../capacity_corrected_topk_ksweep.png

Run:
  /root/int-model-env/bin/python topk_breakdown_corrected.py --seed 20260611
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from topk_breakdown import Dump, KSWEEP, K_MAIN, N_DENSE, COLORS

OUTDIR = os.path.dirname(HERE)
SCHEMES = ["baseline", "faithful", "codebook"]
LABELS = {
    "baseline": "baseline-native (ref=M_int, served=FP8)",
    "faithful": "faithful-arch-v1 (ref=M_int, served=FP8)",
    "codebook": "codebook (ref=M_int, served=FP8)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260611)
    a = ap.parse_args()

    results = {}
    for sc in SCHEMES:
        dump = Dump(os.path.join(HERE, f"capacity_dump_corrected_{sc}_seed{a.seed}.npz"))
        res = {"scheme": sc, "V": dump.V, "n_positions": dump.P,
               "orientation": "corrected: ref=M_int, served=FP8 argmax"}

        # ---- 1+2: K=16 optimum + five-component breakdown ----
        best = dump.minimize(K_MAIN)
        comp_sum = best["a"] + best["bb"] + best["c"] + best["d"] + best["e"]
        res["K16"] = {
            "K": K_MAIN, "b_star": float(best["b"]), "C_min": float(best["C"]),
            "p": float(best["p"]), "q": float(best["q"]),
            "e_logNb": float(best["e_logNb"]),
            "n_viol": int(best["n_viol"]), "n_tail_viol": int(best["n_tail"]),
            "components": {
                "a_H(p)": float(best["a"]),
                "b_(1-p)ElogNb": float(best["bb"]),
                "c_pH(q)": float(best["c"]),
                "d_p(1-q)log2K": float(best["d"]),
                "e_pq_log2(V-K)": float(best["e"]),
            },
            "component_sum": float(comp_sum),
            "sum_matches_C_3dp": bool(round(comp_sum, 3) == round(best["C"], 3)),
            "n_sweep_points": best["n_sweep_points"],
            "refined_off_grid": best["refined_off_grid"],
        }
        print(f"[{sc}] K=16: C_min={best['C']:.4f} @ b*={best['b']:.4g} "
              f"p={best['p']:.5f} q={best['q']:.4f} ElogNb={best['e_logNb']:.4f} "
              f"(n_viol={int(best['n_viol'])}, sum-check "
              f"{comp_sum:.6f} vs {best['C']:.6f})")

        # ---- 3: K sweep ----
        res["Ksweep"] = []
        for K in KSWEEP:
            bk = dump.minimize(K)
            res["Ksweep"].append({
                "K": K, "C_min": float(bk["C"]), "b_star": float(bk["b"]),
                "p": float(bk["p"]), "q": float(bk["q"]),
                "n_viol": int(bk["n_viol"]), "n_tail_viol": int(bk["n_tail"]),
            })
            print(f"      K={K:>4}: min C_topK={bk['C']:.4f} @ b*={bk['b']:.4g} "
                  f"(p={bk['p']:.5f}, q={bk['q']:.4f}, "
                  f"tail {int(bk['n_tail'])}/{int(bk['n_viol'])})")
        kbest = min(res["Ksweep"], key=lambda r: r["C_min"])
        res["K_lowest"] = kbest["K"]

        # ---- 4: simple-rule optimum for comparison (same sweep machinery) ----
        cs = dump.c_simple(dump.bgrid)
        j = int(np.argmin(cs))
        dense_b = np.linspace(dump.bgrid[max(j - 1, 0)],
                              dump.bgrid[min(j + 1, len(dump.bgrid) - 1)], N_DENSE)
        cd = dump.c_simple(dense_b)
        if cd.min() < cs[j]:
            jj = int(np.argmin(cd)); b_s, c_s = float(dense_b[jj]), float(cd[jj])
        else:
            b_s, c_s = float(dump.bgrid[j]), float(cs[j])
        res["simple_min"] = {"C_min": c_s, "b_star": b_s}
        print(f"      simple rule: min C={c_s:.4f} @ b*={b_s:.4g} "
              f"-> top-K(16) buys {c_s - res['K16']['C_min']:.4f} bits "
              f"({100*(c_s-res['K16']['C_min'])/c_s:.1f} %)")
        # max served-token rank over ALL positions (supports small-sample q=0 claims)
        res["max_rank_all_positions"] = int(dump.ranks.max())
        res["median_rank_violations_at_K16_bstar"] = float(np.median(
            dump.ranks[dump.margins > res["K16"]["b_star"]])) if res["K16"]["n_viol"] else None
        results[sc] = res

    # ---- plot: min-over-b C_topK vs K ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for sc in SCHEMES:
        ks = [r["K"] for r in results[sc]["Ksweep"]]
        cs = [r["C_min"] for r in results[sc]["Ksweep"]]
        ax.plot(ks, cs, "o-", color=COLORS[sc], label=LABELS[sc])
        kb = results[sc]["K_lowest"]
        cb = next(r["C_min"] for r in results[sc]["Ksweep"] if r["K"] == kb)
        ax.plot([kb], [cb], "*", color="black", ms=14, zorder=5)
        ax.annotate(f"K={kb}: {cb:.3f}", (kb, cb), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)
        sm = results[sc]["simple_min"]["C_min"]
        ax.axhline(sm, color=COLORS[sc], ls=":", lw=1, alpha=0.6)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(KSWEEP); ax.set_xticklabels([str(k) for k in KSWEEP])
    ax.set_xlabel("top-K parameter K"); ax.set_ylabel("min over b of C_topK (bits/token)")
    ax.set_title("Worst-case covert capacity vs top-K — CORRECTED orientation\n"
                 "(ref = M_int, served = FP8; stars = lowest-capacity K; "
                 "dotted = simple-rule min)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    plot_path = os.path.join(OUTDIR, "capacity_corrected_topk_ksweep.png")
    fig.savefig(plot_path, dpi=120); plt.close(fig)

    out = os.path.join(HERE, f"topk_corrected_results_seed{a.seed}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}\nplot  {plot_path}")


if __name__ == "__main__":
    main()
