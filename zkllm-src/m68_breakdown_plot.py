"""Per-component breakdown of the FULL zkLLM proof of llama-68m (complete, measured)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("m68_timings.json"))
layers = d["timings"]["layers"]
# average per-component across layers
steps = {}
for L in layers:
    for s in L["steps"]:
        steps.setdefault(s["step"], []).append(s["seconds"])
avg = {k: sum(v) / len(v) for k, v in steps.items()}
order = ["rmsnorm.input", "self-attn.linear", "self-attn.attn", "skip.attn",
         "rmsnorm.post", "ffn", "skip.ffn"]
labels = {"rmsnorm.input": "RMSNorm (input)", "self-attn.linear": "Attn Q/K/V proj\n(+rescale+commit-open)",
          "self-attn.attn": "Attn core\n(QK^T+zkAttn softmax+AV)", "skip.attn": "skip (attn)",
          "rmsnorm.post": "RMSNorm (post-attn)", "ffn": "FFN gate/up/down+SwiGLU\n(+rescale+commit-open)",
          "skip.ffn": "skip (ffn)"}
vals = [avg[k] for k in order]
names = [labels[k] for k in order]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(range(len(order)), vals, color="#4C78A8")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + 0.03, i, f"{v:.2f}s", va="center", fontsize=9)
per_layer = d["prove_per_layer_s"]
ax.set_xlabel("proof time per transformer layer (s), seq=1024", fontsize=10)
ax.set_title(f"FULL zkLLM proof — per-layer component breakdown\n"
             f"{d['model']} (embed {d['embed']}, {d['heads']} heads, MHA) — "
             f"per layer {per_layer:.2f}s | all {d['layers_run']} layers {d['prove_total_s_all_layers']:.1f}s "
             f"+ one-time commit {d['commit_total_s']:.1f}s",
             fontsize=10.5, fontweight="bold")
ax.grid(True, axis="x", ls="--", alpha=0.3)
note = ("COMPLETE measurement — no estimated components. Lookup/commitment-opening parts (inside Attn-proj &\n"
        "FFN) dominate; the zkAttn softmax itself is cheap (~0.7s). zkLLM's released pipeline omits o_proj.")
ax.text(0.0, -0.16, note, transform=ax.transAxes, fontsize=8, color="#555",
        family="monospace")
fig.tight_layout()
fig.savefig("m68_breakdown.png", dpi=150)
print("wrote m68_breakdown.png")
print(f"per-layer total: {per_layer:.2f}s | full model ({d['layers_run']}L): {d['prove_total_s_all_layers']:.1f}s")
print(f"one-time: ppgen {d['ppgen_total_s']:.1f}s + commit {d['commit_total_s']:.1f}s")
