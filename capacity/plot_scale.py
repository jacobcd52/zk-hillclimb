"""Plot five-term codebook capacity mu* vs model parameter count.
Run: /root/int-model-env/bin/python capacity/plot_scale.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "mu_vs_scale.png")

res = [r for r in json.load(open(os.path.join(HERE, "scale_sweep_results.json")))
       if "mu_star" in r]
res.sort(key=lambda r: r["n_params"])
xs = [r["n_params"] for r in res]
ys = [r["mu_star"] for r in res]

fig, ax = plt.subplots(figsize=(8.4, 6))
ax.plot(xs, ys, "o-", color="#d62728", lw=2.2, ms=8)
for r in res:
    ax.annotate(f"  {r['label']}\n  μ*={r['mu_star']:.3f}", (r["n_params"], r["mu_star"]),
                fontsize=8, va="bottom")
ax.set_xscale("log")
ax.set_xlabel("model parameters")
ax.set_ylabel("five-term covert capacity  μ*  (bits / token)")
ax.set_title("Codebook covert-channel capacity vs model scale\n"
             f"converged five-term benign rate (K=4), {res[0]['n_tokens']} tokens/model")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
print("\nmu* by scale:")
for r in res:
    print(f"  {r['label']:16s} {r['n_params']:>12,} params  mu*={r['mu_star']:.4f}  b*={r['b_star']}")
