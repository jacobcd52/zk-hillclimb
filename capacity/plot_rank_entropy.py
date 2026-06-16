"""Plot R_rank covert capacity (rank entropy, bits/token) vs model scale.

Scatter, x = params (log), y = R_rank. Colored by family; FP8-native models drawn
with hollow markers in a darker shade of the family colour. Within-family points
connected by a line (sorted by params, bf16 line + fp8 line separately). No lines
across families.

Run: /root/int-model-env/bin/python capacity/plot_rank_entropy.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "rank_vs_scale.png")

FAMILY_COLOR = {
    "qwen2.5":    "#1f77b4",
    "llama":      "#d62728",
    "llama-open": "#ff9896",
    "gemma":      "#2ca02c",
    "smollm2":    "#9467bd",
    "phi3":       "#8c564b",
    "olmo":       "#e377c2",
}

res = [r for r in json.load(open(os.path.join(HERE, "rank_entropy_results.json")))
       if "R_rank" in r]
res.sort(key=lambda r: r["n_params"])

fig, ax = plt.subplots(figsize=(10, 7))

families = sorted({r["family"] for r in res})
for fam in families:
    color = FAMILY_COLOR.get(fam, "#555555")
    for fp8 in (False, True):
        grp = [r for r in res if r["family"] == fam and r["fp8_native"] == fp8]
        if not grp:
            continue
        grp.sort(key=lambda r: r["n_params"])
        xs = [r["n_params"] for r in grp]
        ys = [r["R_rank"] for r in grp]
        if fp8:
            ax.plot(xs, ys, "--", color=color, lw=1.6, alpha=0.8, zorder=2)
            ax.scatter(xs, ys, s=110, facecolors="none", edgecolors=color,
                       linewidths=2.0, marker="o", zorder=3)
        else:
            ax.plot(xs, ys, "-", color=color, lw=1.8, alpha=0.9, zorder=2)
            ax.scatter(xs, ys, s=70, color=color, marker="o", zorder=3)

# point labels
for r in res:
    ax.annotate(f"  {r['label']}", (r["n_params"], r["R_rank"]),
                fontsize=6.5, va="center", alpha=0.8)

ax.set_xscale("log")
ax.set_xlabel("model parameters (log scale)")
ax.set_ylabel("R_rank  =  rank-distribution entropy  (bits / token)")
ax.set_title("R_rank covert capacity (rank entropy) vs model scale\n"
             "codebook integerization, on-policy T=1 tokens")
ax.grid(alpha=0.3, which="both")

# legend: families + fp8 marker convention
handles = [Line2D([0], [0], color=FAMILY_COLOR.get(f, "#555555"), marker="o",
                  lw=1.8, label=f) for f in families]
handles += [
    Line2D([0], [0], color="k", marker="o", lw=1.8, label="bf16 (filled marker)"),
    Line2D([0], [0], color="k", marker="o", lw=1.6, ls="--",
           markerfacecolor="none", markeredgewidth=2.0,
           label="FP8-native (hollow marker, dashed)"),
]
ax.legend(handles=handles, fontsize=8, loc="best", ncol=2)

fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
print("\nR_rank by scale:")
for r in res:
    tag = "FP8" if r["fp8_native"] else "bf16"
    print(f"  {r['label']:20s} {r['family']:11s} {tag:4s} "
          f"{r['n_params']:>13,} params  R_rank={r['R_rank']:.4f}  "
          f"vocab={r['vocab']:>6}  ntok={r['n_completion_tokens']}")
