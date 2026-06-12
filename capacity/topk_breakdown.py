"""Top-K (Rinberg) capacity rule: optimal b*, full term breakdown, K sweep.

Consumes the existing per-position dumps
  measure/capacity_dump_{scheme}_seed{seed}.npz   (margins, cand_ranks, Nb grid)
(NO model re-run) and computes, for each scheme, the top-K refined capacity

  C_topK(b) = H(p) + (1-p)*E_t[log2 N_b | margin<=b]
              + p*( H(q) + (1-q)*log2 K + q*log2(V-K) )

  p(b) = frac positions with margin > b                       (exact at any b)
  q(b) = frac of VIOLATING positions whose served token is outside the
         teacher's post-Gumbel top-K, i.e. cand_rank >= K     (exact at any b)
  E[log2 N_b] = conditional mean over compliant positions     (exact on the
         dump's 248-point b grid; LINEARLY INTERPOLATED per position between
         bracketing grid points for the dense refinement around the argmin)

Five-component breakdown at b*:
  (a) H(p)                  which positions violate
  (b) (1-p)*E[log2 N_b]     within-margin multiplicity (compliant)
  (c) p*H(q)                violate: tail-vs-topK choice
  (d) p*(1-q)*log2 K        violate into top-K
  (e) p*q*log2(V-K)         violate into tail

Sweep: the dump's own 248 exact grid points + 200 dense points around the
coarse argmin (>=448 points total per scheme/K).  K sweep over
{1,2,4,8,16,32,64,256,1024}; K=1 reduces (q->1) to ~the simple rule.

Run:
  /root/int-model-env/bin/python topk_breakdown.py --seed 20260611
Outputs:
  capacity/topk_results_seed{seed}.json
  ../capacity_topk_ksweep.png   (min-over-b C_topK vs K, per scheme)
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURE = os.path.join(os.path.dirname(HERE), "measure")
OUTDIR = os.path.dirname(HERE) if os.path.basename(
    os.path.dirname(HERE)) == "zk-hillclimb" else "/workspace/projects/zk-hillclimb"
SCHEMES = ["baseline", "faithful", "codebook"]
KSWEEP = [1, 2, 4, 8, 16, 32, 64, 256, 1024]
K_MAIN = 16
N_DENSE = 200
COLORS = {"baseline": "#c0392b", "faithful": "#2471a3", "codebook": "#1e8449"}
LABELS = {
    "baseline": "baseline-native (DiFR 8.99)",
    "faithful": "faithful-arch-v1 (DiFR 0.0156)",
    "codebook": "codebook (DiFR 0.0060)",
}


def H(x):
    """Binary Shannon entropy in bits, elementwise; H(0)=H(1)=0."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    out = np.zeros_like(x)
    m = (x > 0) & (x < 1)
    out[m] = -x[m] * np.log2(x[m]) - (1 - x[m]) * np.log2(1 - x[m])
    return out


class Dump:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.bgrid = d["bgrid"].astype(np.float64)        # (nb,) exact grid
        self.margins = d["margins"].astype(np.float64)    # (P,)
        self.ranks = d["cand_ranks"].astype(np.int64)     # (P,)
        self.logNb = np.log2(d["Nb"].astype(np.float64))  # (P, nb)
        self.V = int(d["vocab"])
        self.P = self.margins.size

    def e_logNb(self, b, compliant):
        """E[log2 N_b | compliant] at thresholds b (m,), interpolating per-position
        log2 N_b between bracketing grid points (exact when b is on the grid).
        compliant: (P, m) bool mask (computed exactly from margins)."""
        hi = np.searchsorted(self.bgrid, b, side="left").clip(0, len(self.bgrid) - 1)
        lo = np.maximum(hi - 1, 0)
        span = self.bgrid[hi] - self.bgrid[lo]
        w = np.where(span > 0, (b - self.bgrid[lo]) / np.where(span > 0, span, 1), 0.0)
        on_grid = np.isclose(self.bgrid[hi], b)
        w = np.where(on_grid, 1.0, w)                     # exact at grid points
        ln = self.logNb[:, lo] * (1 - w)[None, :] + self.logNb[:, hi] * w[None, :]
        n_comp = compliant.sum(axis=0)
        s = (ln * compliant).sum(axis=0)
        return np.where(n_comp > 0, s / np.maximum(n_comp, 1), 0.0)

    def curve(self, b, K):
        """C_topK and its five components at thresholds b (m,) for top-K size K.
        Returns dict of arrays (m,)."""
        b = np.asarray(b, dtype=np.float64)
        compliant = self.margins[:, None] <= b[None, :]   # (P, m) exact
        n_comp = compliant.sum(axis=0)
        p = 1.0 - n_comp / self.P
        e_ln = self.e_logNb(b, compliant)

        viol = ~compliant
        n_viol = viol.sum(axis=0)
        tail = (self.ranks >= K)[:, None]
        n_tail = (viol & tail).sum(axis=0)
        q = np.where(n_viol > 0, n_tail / np.maximum(n_viol, 1), 0.0)

        a_ = H(p)
        b_ = (1.0 - p) * e_ln
        c_ = p * H(q)
        d_ = p * (1.0 - q) * np.log2(K) if K > 1 else np.zeros_like(p)
        e_ = p * q * np.log2(self.V - K)
        return {"b": b, "p": p, "q": q, "e_logNb": e_ln,
                "n_viol": n_viol, "n_tail": n_tail,
                "a": a_, "bb": b_, "c": c_, "d": d_, "e": e_,
                "C": a_ + b_ + c_ + d_ + e_}

    def c_simple(self, b):
        b = np.asarray(b, dtype=np.float64)
        compliant = self.margins[:, None] <= b[None, :]
        p = 1.0 - compliant.sum(axis=0) / self.P
        return H(p) + (1 - p) * self.e_logNb(b, compliant) + p * np.log2(self.V)

    def minimize(self, K):
        """Coarse min on the exact 248-point grid, then N_DENSE-point refinement
        between the bracketing grid points. Returns (curve-dict at b*, n_sweep)."""
        coarse = self.curve(self.bgrid, K)
        j = int(np.argmin(coarse["C"]))
        lo = self.bgrid[max(j - 1, 0)]
        hi = self.bgrid[min(j + 1, len(self.bgrid) - 1)]
        dense_b = np.linspace(lo, hi, N_DENSE)
        dense = self.curve(dense_b, K)
        # global argmin over coarse + dense
        if dense["C"].min() < coarse["C"][j]:
            jj = int(np.argmin(dense["C"]))
            src, jbest = dense, jj
        else:
            src, jbest = coarse, j
        best = {k: (v[jbest] if isinstance(v, np.ndarray) else v)
                for k, v in src.items()}
        best["n_sweep_points"] = len(self.bgrid) + N_DENSE
        best["refined_off_grid"] = bool(src is dense and
                                        not np.any(np.isclose(self.bgrid, best["b"])))
        return best


def fmt_row(name, val, total):
    return f"| {name} | {val:.4f} | {100*val/total:.1f} % |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260611)
    a = ap.parse_args()

    results = {}
    for sc in SCHEMES:
        dump = Dump(os.path.join(MEASURE, f"capacity_dump_{sc}_seed{a.seed}.npz"))
        res = {"scheme": sc, "V": dump.V, "n_positions": dump.P}

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
    ax.set_title("Worst-case covert capacity vs top-K modeling choice\n"
                 "(stars = lowest-capacity K; dotted = simple-rule min, the K=1 limit)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    plot_path = os.path.join(OUTDIR, "capacity_topk_ksweep.png")
    fig.savefig(plot_path, dpi=120); plt.close(fig)

    out = os.path.join(HERE, f"topk_results_seed{a.seed}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}\nplot  {plot_path}")


if __name__ == "__main__":
    main()
