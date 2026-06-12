"""Covert-channel capacity sweep C(b) from the per-position dumps.

Consumes capacity_dump_{scheme}_seed{seed}.npz (margins, cand_ranks, N_b grid)
and computes, on the shared b grid, the per-token covert capacity an adversary
holding the exact model could exploit while staying within the honest DiFR profile:

  p(b)         = fraction of positions with margin_t > b           (the violation rate)
  N_b(t)       = #{vocab v : teacher_pref - postGumbel(v) <= b}    (within-margin count)
  H(x)         = binary Shannon entropy in bits

  SIMPLE:
    C(b) = H(p) + (1-p)*E_t[log2 N_b(t) | margin_t <= b] + p*log2(V)
      term1 H(p)      : which positions the adversary violates (must match honest rate)
      term2           : compliant tokens may pick any within-margin token (mostly log2 1 = 0)
      term3 p*log2 V  : violating tokens pick freely from the whole vocab

  TOP-K REFINEMENT (K=16, Rinberg double chain rule):
    C_topK(b) = H(p) + (1-p)*E_t[log2 N_b] + p*( H(q) + (1-q)*log2 K + q*log2(V-K) )
      q = fraction of VIOLATIONS whose served token is outside the teacher top-K
          (cand_rank >= K). Common (in-top-K) violations pay log2 K instead of log2 V.

Self-checks reported: C(0) ~ H(p0)+p0*log2 V (N_0~1); C(b->inf) -> log2 V (N_b->V, p->0).
Headline per scheme: min_b C(b) and argmin b* (and p* there), for both formulas.

Run:
  /root/int-model-env/bin/python capacity_analyze.py --seed 20260611
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEASURE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = "/workspace/projects/zk-hillclimb"
K = 16
SEQ_LEN = 1024
SCHEMES = ["baseline", "faithful", "codebook"]
LABELS = {
    "baseline": "baseline-native (zkLLM-style, DiFR 8.99)",
    "faithful": "faithful-arch-v1 (DiFR 0.0156)",
    "codebook": "codebook (DiFR 0.0060)",
}
COLORS = {"baseline": "#c0392b", "faithful": "#2471a3", "codebook": "#1e8449"}


def H(x):
    """Binary Shannon entropy in bits, elementwise; H(0)=H(1)=0."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    out = np.zeros_like(x)
    m = (x > 0) & (x < 1)
    out[m] = -x[m] * np.log2(x[m]) - (1 - x[m]) * np.log2(1 - x[m])
    return out


def analyze(npz_path):
    d = np.load(npz_path)
    b = d["bgrid"].astype(np.float64)                 # (nb,)
    margins = d["margins"].astype(np.float64)         # (P,)
    cand_ranks = d["cand_ranks"].astype(np.int64)     # (P,)
    Nb = d["Nb"].astype(np.float64)                   # (P, nb)
    V = int(d["vocab"])
    P = margins.size
    log2V = np.log2(V)
    logNb = np.log2(Nb)                               # (P, nb)

    compliant = margins[:, None] <= b[None, :]        # (P, nb) True = within margin
    n_comp = compliant.sum(axis=0)                    # (nb,)
    p = 1.0 - n_comp / P                              # violation fraction (nb,)

    # term2 = E over compliant positions of log2 N_b
    sum_comp = (logNb * compliant).sum(axis=0)
    e_logNb = np.where(n_comp > 0, sum_comp / np.maximum(n_comp, 1), 0.0)
    term2 = (1.0 - p) * e_logNb

    # SIMPLE
    C_simple = H(p) + term2 + p * log2V

    # TOP-K: q = fraction of violations with served token outside teacher top-K
    viol = ~compliant                                 # (P, nb)
    n_viol = viol.sum(axis=0)
    tail = (cand_ranks >= K)[:, None]                 # (P,1) served token outside top-K
    n_tail_viol = (viol & tail).sum(axis=0)
    q = np.where(n_viol > 0, n_tail_viol / np.maximum(n_viol, 1), 0.0)
    viol_cost = H(q) + (1.0 - q) * np.log2(K) + q * np.log2(V - K)
    C_topK = H(p) + term2 + p * viol_cost

    # endpoints
    j0 = int(np.argmin(b))                            # b == 0
    p0 = p[j0]
    C0_simple_check = H(np.array([p0]))[0] + p0 * log2V     # N_0 ~ 1 -> term2 ~ 0
    jinf = int(np.argmax(b))
    res = {
        "scheme": str(d["scheme"]), "V": V, "n_positions": P, "log2V": log2V,
        "b": b.tolist(), "p": p.tolist(),
        "C_simple": C_simple.tolist(), "C_topK": C_topK.tolist(),
        "q": q.tolist(), "e_logNb": e_logNb.tolist(),
        "p0_postgumbel": float(p0),
        "endpoint_b0": {
            "b": float(b[j0]), "p": float(p0),
            "C_simple": float(C_simple[j0]), "C_topK": float(C_topK[j0]),
            "C0_formula_check_H(p0)+p0*log2V": float(C0_simple_check),
            "mean_Nb_at_b0": float(Nb[:, j0].mean()),
        },
        "endpoint_binf": {
            "b": float(b[jinf]), "p": float(p[jinf]),
            "C_simple": float(C_simple[jinf]), "C_topK": float(C_topK[jinf]),
            "mean_Nb_at_binf": float(Nb[:, jinf].mean()),
            "log2V_target": log2V,
        },
    }
    for name, C in (("simple", C_simple), ("topK", C_topK)):
        jmin = int(np.argmin(C))
        res[f"min_{name}"] = {
            "C_min_bits_per_token": float(C[jmin]),
            "b_star": float(b[jmin]),
            "p_star": float(p[jmin]),
            "bits_per_forward_pass(x1024)": float(C[jmin] * SEQ_LEN),
        }
    return res


def plot_scheme(res, path):
    b = np.array(res["b"]); p = np.array(res["p"])
    Cs = np.array(res["C_simple"]); Ck = np.array(res["C_topK"])
    sc = res["scheme"]; col = COLORS[sc]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    pos = b > 0
    ax1.plot(b[pos], Cs[pos], "-", color=col, label="C(b) simple")
    ax1.plot(b[pos], Ck[pos], "--", color=col, alpha=0.8, label="C(b) top-K (K=16)")
    for name, C, mk in (("simple", Cs, "o"), ("topK", Ck, "s")):
        m = res[f"min_{name}"]
        ax1.plot(m["b_star"], m["C_min_bits_per_token"], mk, color="black", ms=8,
                 label=f"min {name}: {m['C_min_bits_per_token']:.3f} b/tok @ b*={m['b_star']:.3g}")
    ax1.axhline(res["log2V"], color="gray", ls=":", lw=1, label=f"log2 V = {res['log2V']:.2f}")
    ax1.set_xscale("log"); ax1.set_xlabel("acceptance threshold b (nats)")
    ax1.set_ylabel("capacity C (bits / token)")
    ax1.set_title(f"{LABELS[sc]}\nC(b) vs b"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    order = np.argsort(p)
    ax2.plot(p[order], Cs[order], "-", color=col, label="C vs p simple")
    ax2.plot(p[order], Ck[order], "--", color=col, alpha=0.8, label="C vs p top-K")
    for name, C, mk in (("simple", Cs, "o"), ("topK", Ck, "s")):
        m = res[f"min_{name}"]
        ax2.plot(m["p_star"], m["C_min_bits_per_token"], mk, color="black", ms=8,
                 label=f"min {name} @ p*={m['p_star']:.3g}")
    ax2.set_xlabel("violation fraction p(b)"); ax2.set_ylabel("capacity C (bits / token)")
    ax2.set_title("C vs p (parametric in b)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_combined(results, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for sc in SCHEMES:
        r = results[sc]; b = np.array(r["b"]); pos = b > 0
        Cs = np.array(r["C_simple"]); Ck = np.array(r["C_topK"])
        ax.plot(b[pos], Cs[pos], "-", color=COLORS[sc], label=f"{LABELS[sc]} (simple)")
        ax.plot(b[pos], Ck[pos], "--", color=COLORS[sc], alpha=0.7,
                label=f"{LABELS[sc]} (top-K)")
        m = r["min_simple"]
        ax.plot(m["b_star"], m["C_min_bits_per_token"], "o", color=COLORS[sc],
                ms=9, mec="black")
    ax.axhline(np.log2(32000), color="gray", ls=":", lw=1, label="log2 V = 14.97")
    ax.set_xscale("log"); ax.set_xlabel("acceptance threshold b (nats)")
    ax.set_ylabel("covert capacity C (bits / token)")
    ax.set_title("Covert-channel capacity vs acceptance threshold\n"
                 "(circles = min_b C(b), the worst-case capacity per scheme)")
    ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    results = {}
    for sc in SCHEMES:
        npz = os.path.join(MEASURE, f"capacity_dump_{sc}_seed{a.seed}.npz")
        if not os.path.exists(npz):
            print(f"skip {sc}: no dump at {npz}"); continue
        results[sc] = analyze(npz)
        plot_scheme(results[sc], os.path.join(OUTDIR, f"capacity_{sc}.png"))
        print(f"[{sc}] min simple C = {results[sc]['min_simple']['C_min_bits_per_token']:.4f} "
              f"b/tok @ b*={results[sc]['min_simple']['b_star']:.4g} "
              f"(p*={results[sc]['min_simple']['p_star']:.4g}); "
              f"top-K min = {results[sc]['min_topK']['C_min_bits_per_token']:.4f}")
        e0 = results[sc]["endpoint_b0"]; ei = results[sc]["endpoint_binf"]
        print(f"      b=0: C_simple={e0['C_simple']:.4f} (formula check "
              f"{e0['C0_formula_check_H(p0)+p0*log2V']:.4f}, Nb={e0['mean_Nb_at_b0']:.2f}); "
              f"b={ei['b']:.0f}: C_simple={ei['C_simple']:.4f} (->log2V {ei['log2V_target']:.4f}, "
              f"Nb={ei['mean_Nb_at_binf']:.0f})")
    if set(results) >= set(SCHEMES):
        plot_combined(results, os.path.join(OUTDIR, "capacity_combined.png"))
    with open(os.path.join(MEASURE, f"capacity_results_seed{a.seed}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {os.path.join(MEASURE, f'capacity_results_seed{a.seed}.json')}")
    print("plots ->", OUTDIR)


if __name__ == "__main__":
    main()
