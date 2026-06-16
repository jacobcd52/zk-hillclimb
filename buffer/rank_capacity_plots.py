"""Two requested capacity plots, reusing the validated three_measure machinery.

  PLOT 1  rank_capacity_schemes.png
     RANK-ENTROPY capacity bound vs audit size, one line per scheme
     (FIXED-POINT [formerly "faithful"] and CODEBOOK).
     y = benign_rate + FPR<=1e-10 buffer(N) + counting-slack(N)  [bits/token]
       = upper (1 - 1e-10) confidence bound on the per-token channel capacity.
     solid = observed region (<=131072-token data limit), dashed = extrapolated.

  PLOT 2  codebook_fiveterm_vs_rank.png
     CODEBOOK only: optimal FIVE-TERM bound vs RANK-ENTROPY bound.
     Same y axis.  Rank-entropy starts high (V*log n counting slack) but the slack
     washes out and it ends up below the five-term bound at large N.

Run: /root/int-model-env/bin/python rank_capacity_plots.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import analyze_threshold as at
import analyze_buffer as ab
import three_measure_curve as tm

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.dirname(HERE)
EPS = 1e-10
REPS = 1_000_000
DATA_LIMIT = tm.DATA_LIMIT

SCHEME_META = {
    "faithful": {"name": "fixed-point", "color": "#1f77b4"},
    "codebook": {"name": "codebook",    "color": "#d62728"},
}


def scheme_curves(sname, rng):
    """Return five-term and full-rank-entropy y-curves + benign rates for a scheme."""
    S = at.SCHEMES[sname]
    margins, ranks, Nb, bgrid, V, block = at.load_scheme(S["orig"], S["extra"])
    b, K = S["optimum"]["b"], S["optimum"]["K"]

    # ---- five-term (margin/afforded), optimum b,K ----
    pt = ab.per_token(margins, ranks, Nb, bgrid, V, b, K)
    mu5 = pt["mu"]
    Y5 = ab.session_sums(pt["s"], block)
    obs5, c5, _ = tm.buffer_curve_sumlike(Y5, rng, reps=REPS)
    curve5 = tm.y_curve(mu5, obs5, c5, slack_coeff=2.0)        # 2 params (p,q)

    # ---- full rank-entropy ----
    support = np.unique(ranks)
    C = tm.session_count_matrix(ranks, block, support)
    P = C.sum(axis=0) / C.sum()
    H_full = float(np.where(P > 0, -P * np.log2(P), 0.0).sum())
    obsR, cR, _ = tm.buffer_curve_functional(C, support, H_full, tm.entropy_rows,
                                             rng, reps=REPS)
    curveR = tm.y_curve(H_full, obsR, cR, slack_coeff=float(V - 1))   # (V-1) params

    print(f"[{sname:8s}] five-term mu={mu5:.4f} (c={c5:.2f})   "
          f"rank-entropy H={H_full:.4f} (c={cR:.2f})   V={V}")
    return {"mu5": mu5, "curve5": curve5, "Hrank": H_full, "curveR": curveR,
            "color": SCHEME_META[sname]["color"], "name": SCHEME_META[sname]["name"]}


def plot_line(ax, curve, color, label, lw=2.5):
    """Solid where observed, dashed where extrapolated; single legend entry (solid)."""
    Ns = np.array(curve["N"], float)
    ys = np.array(curve["y"], float)
    obs = np.array(curve["observed"], bool)
    ax.plot(Ns[obs], ys[obs], "-", color=color, lw=lw, label=label, zorder=3)
    if obs.any() and (~obs).any():
        j = np.where(obs)[0][-1]
        ax.plot(Ns[j:], ys[j:], "--", color=color, lw=lw - 0.5, zorder=3)
    elif (~obs).all():
        ax.plot(Ns, ys, "--", color=color, lw=lw - 0.5, label=label, zorder=3)


def datalimit_marker(ax):
    ax.axvline(DATA_LIMIT, color="0.35", ls="-", lw=1.1, alpha=0.8, zorder=1)
    ax.text(DATA_LIMIT * 1.15, ax.get_ylim()[1] * 0.6,
            "data limit\n(131,072 tok)\n→ dashed = extrapolated",
            color="0.25", va="top", ha="left", fontsize=8)


def benign_asymptote(ax, y, color, xtext):
    ax.axhline(y, color=color, ls=":", lw=1.2, alpha=0.85, zorder=1)
    ax.text(xtext, y, f" {y:.3f}", color=color, va="bottom", ha="right",
            fontsize=8.5, fontweight="bold")


def main():
    rng = np.random.default_rng(20260616)
    info = {s: scheme_curves(s, rng) for s in ("faithful", "codebook")}

    XMIN, XMAX = 1e4, 1e12
    extrap_handle = Line2D([0], [0], color="0.4", ls="--", lw=2.0,
                           label="dashed = extrapolated from data")

    # ============================ PLOT 1 ============================ #
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for s in ("faithful", "codebook"):
        d = info[s]
        plot_line(ax, d["curveR"], d["color"], d["name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(XMIN, XMAX)
    for s in ("faithful", "codebook"):
        benign_asymptote(ax, info[s]["Hrank"], info[s]["color"], XMAX)
    datalimit_marker(ax)
    ax.set_xlabel("tokens audited,  N")
    ax.set_ylabel("$R_{\\mathrm{rank}}$ — upper (1 − 10$^{-10}$) confidence bound   "
                  "(bits / token)")
    ax.set_title("Rank-entropy capacity bound vs audit size", fontsize=13,
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [extrap_handle], l + [extrap_handle.get_label()],
              fontsize=9.5, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    p1 = os.path.join(OUTDIR, "rank_capacity_schemes.png")
    fig.savefig(p1, dpi=140); plt.close(fig)

    # ============================ PLOT 2 ============================ #
    d = info["codebook"]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    plot_line(ax, d["curve5"], "#2ca02c", "$R$ (five-term)")
    plot_line(ax, d["curveR"], "#9467bd", "$R_{\\mathrm{rank}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(XMIN, XMAX)
    benign_asymptote(ax, d["mu5"], "#2ca02c", XMAX)
    benign_asymptote(ax, d["Hrank"], "#9467bd", XMAX)
    datalimit_marker(ax)

    # crossover where rank-entropy drops below five-term
    Nf = np.unique(np.round(10.0 ** np.arange(4.0, 12.001, 0.002)).astype(np.int64))
    def yf(cv):
        return np.interp(np.log10(Nf), np.log10(cv["N"]), cv["y"])
    diff = yf(d["curveR"]) - yf(d["curve5"])
    below = np.where(diff < 0)[0]
    xover = int(Nf[below[0]]) if below.size else None
    if xover:
        ax.axvline(xover, color="0.5", ls=":", lw=1.0, alpha=0.7)
        ax.text(xover * 1.2, ax.get_ylim()[0] * 1.5,
                f"rank-entropy wins\nN ≈ {xover:.0e}", fontsize=8, color="0.3")

    ax.set_xlabel("tokens audited,  N")
    ax.set_ylabel("$R$ — upper (1 − 10$^{-10}$) confidence bound   (bits / token)")
    ax.set_title("Codebook: five-term vs rank-entropy bound", fontsize=13,
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [extrap_handle], l + [extrap_handle.get_label()],
              fontsize=9.5, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    p2 = os.path.join(OUTDIR, "codebook_fiveterm_vs_rank.png")
    fig.savefig(p2, dpi=140); plt.close(fig)

    out = {"crossover_rank_below_fiveterm_codebook": xover,
           "benign": {s: {"five_term": info[s]["mu5"], "rank_entropy": info[s]["Hrank"]}
                      for s in info}}
    with open(os.path.join(HERE, "rank_capacity_plots.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\ncodebook crossover (rank<5term): N≈{xover}")
    print(f"wrote {p1}\n      {p2}")


if __name__ == "__main__":
    main()
