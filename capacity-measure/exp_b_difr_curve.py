"""Experiment B: the DiFR logit-tolerance channel (the delta-curve).

The harness token check (harness/score.py + acceptance_bits.py methodology) is NOT
a raw "logit within delta of argmax" test -- it is a POST-GUMBEL margin test:

    margin_t = (z_ref + g)[argmax] - (z_ref + g)[t]            (shared Gumbel g, seed 0)
    accept token t  iff  margin_t <= tau

where z_ref are the verifier/teacher (FP8-dynamic) logits.  An adversary free to serve
any accepted token encodes  log2 |A_tau|  bits at that position, so the channel capacity
in bits/token at tolerance tau is

    C(tau) = E_position[ log2 |{ t : margin_t <= tau }| ].

This script computes C(tau) for tau in [1e-4, 1.0] (log-spaced) for the three real-text
contexts (wiki / lorem / code), per input and pooled, and marks the harness's actual
operating point (the honest-student p99/max post-Gumbel margin floors from
acceptance_bits.json).  For completeness it ALSO computes the naive RAW-logit form
(no Gumbel) that the task's B.1 literally describes, so the two can be compared.

Reference logits = the FP8-dynamic teacher (the DiFR "verifier"), identical construction
to acceptance_bits.py / llama_difr.py, so the marked operating point is on the same footing.

Run:
  IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python exp_b_difr_curve.py
"""
import json
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

PARETO = "/workspace/projects/int-model-approximation/results/llama_pareto"
sys.path.insert(0, PARETO)
from int_model_approximation.__main__ import FP8Linear            # noqa: E402
from int_model_approximation import metrics as M                  # noqa: E402
from llama_difr import replace_linears, forced_logits             # noqa: E402

import capacity_lib as L                                          # noqa: E402

DEV = "cuda"
GUMBEL_SEED = 0
# Honest-student post-Gumbel margin floors (acceptance_bits.json) -> harness operating tau.
FLOORS = {
    "zkllm_native p99": 0.21338552236557007,
    "codebook p99":     0.2333327978849411,
    "zkllm_native max": 0.5899658203125,
    "codebook max":     0.5703535079956055,
}
OP_P99 = 0.21338552236557007   # the tightest p99 floor -> the harness's effective operating tau
OP_MAX = 0.5899658203125


def fp8_teacher():
    m = AutoModelForCausalLM.from_pretrained(L.MODEL_CARD, cache_dir=L.CACHE_DIR,
                                             torch_dtype=torch.bfloat16).to(DEV).eval()
    replace_linears(m, lambda w, s, b: FP8Linear(w, s, b))
    return m


def gumbel_like(z):
    g = torch.empty(z.shape, device=z.device, dtype=torch.float32)
    gen = torch.Generator(device=z.device)
    gen.manual_seed(GUMBEL_SEED)
    g.exponential_(generator=gen).log_().neg_()
    return g


def capacity_curve(margins, taus):
    """margins: [N, V] >=0, 0 at the argmax. Returns mean log2|A_tau| over the N positions."""
    out = []
    for tau in taus:
        sizes = (margins <= tau).sum(dim=-1).float()      # |A_tau| per position, >=1
        out.append(float(torch.log2(sizes).mean()))
    return out


def frac_below(margins, taus):
    """Fraction of positions whose acceptance set has >1 token at tolerance tau,
    i.e. positions where the 2nd-best token is within tau of the top (a free choice)."""
    out = []
    for tau in taus:
        multi = ((margins <= tau).sum(dim=-1) > 1).float().mean()
        out.append(float(multi))
    return out


def main():
    assert os.environ.get("IMA_TEACHER_KERNEL") == "fp8_scaled_mm"
    tok = AutoTokenizer.from_pretrained(L.MODEL_CARD, cache_dir=L.CACHE_DIR)
    inputs = L.load_inputs(tok)                 # same wiki/lorem/code contexts as Exp A
    teacher = fp8_teacher()

    taus = torch.logspace(-4, 0, 80, device=DEV)   # 1e-4 .. 1.0, log-spaced
    taus_l = taus.tolist()

    per_input = {}
    pooled_pg, pooled_raw = [], []
    for name, ids in inputs.items():
        z = forced_logits(teacher, ids)              # [seq, V] FP8 teacher logits
        g = gumbel_like(z)
        zn = z + g
        pg_margin = zn.max(dim=-1, keepdim=True).values - zn      # post-Gumbel margins [seq,V]
        raw_margin = z.max(dim=-1, keepdim=True).values - z       # raw-logit margins  [seq,V]
        pooled_pg.append(pg_margin)
        pooled_raw.append(raw_margin)

        # top-2 raw margin per position (for the "fraction with top-2 margin < delta" curve)
        top2 = z.topk(2, dim=-1).values
        raw_top2 = (top2[:, 0] - top2[:, 1])                      # [seq]

        per_input[name] = {
            "capacity_postgumbel": capacity_curve(pg_margin, taus),
            "capacity_raw": capacity_curve(raw_margin, taus),
            "frac_multi_postgumbel": frac_below(pg_margin, taus),
            "frac_top2margin_below_raw": [float((raw_top2 < t).float().mean()) for t in taus_l],
            "raw_top2_margin_min": float(raw_top2.min()),
            "raw_top2_margin_median": float(raw_top2.median()),
        }
        print(f"[{name}] C(op_p99={OP_P99:.3g})="
              f"{capacity_curve(pg_margin, torch.tensor([OP_P99], device=DEV))[0]:.4f} bits/tok "
              f"C(op_max={OP_MAX:.3g})="
              f"{capacity_curve(pg_margin, torch.tensor([OP_MAX], device=DEV))[0]:.4f} bits/tok")

    pg_all = torch.cat(pooled_pg, dim=0)
    raw_all = torch.cat(pooled_raw, dim=0)
    pooled = {
        "capacity_postgumbel": capacity_curve(pg_all, taus),
        "capacity_raw": capacity_curve(raw_all, taus),
        "frac_multi_postgumbel": frac_below(pg_all, taus),
    }

    def at(margins, tau):
        return capacity_curve(margins, torch.tensor([tau], device=DEV))[0]

    op = {
        "operating_tau_p99": OP_P99,
        "operating_tau_max": OP_MAX,
        "pooled_bits_at_p99": at(pg_all, OP_P99),
        "pooled_bits_at_max": at(pg_all, OP_MAX),
        "per_input_bits_at_p99": {n: at(c, OP_P99) for n, c in
                                  zip(inputs, pooled_pg)},
        "per_input_bits_at_max": {n: at(c, OP_MAX) for n, c in
                                  zip(inputs, pooled_pg)},
    }

    out = {"taus": taus_l, "per_input": per_input, "pooled": pooled,
           "operating_point": op, "floors": FLOORS, "gumbel_seed": GUMBEL_SEED,
           "vocab": int(pg_all.shape[-1]), "n_positions_pooled": int(pg_all.shape[0])}
    with open("exp_b_results.json", "w") as f:
        json.dump(out, f, indent=1)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = {"real_wiki": "#4C78A8", "real_lorem": "#F58518", "real_code": "#54A24B"}
    for name in inputs:
        ax.plot(taus_l, per_input[name]["capacity_postgumbel"],
                color=colors.get(name, "gray"), lw=1.6, label=f"{name} (post-Gumbel)")
    ax.plot(taus_l, pooled["capacity_postgumbel"], color="black", lw=2.4,
            label="pooled (post-Gumbel)")
    ax.plot(taus_l, pooled["capacity_raw"], color="black", lw=1.2, ls=":",
            label="pooled (raw logits, no Gumbel)")
    ax.axvline(OP_P99, color="#C62828", ls="--", lw=1.4,
               label=f"harness op. point tau_p99={OP_P99:.3g} -> {op['pooled_bits_at_p99']:.3f} bits/tok")
    ax.axvline(OP_MAX, color="#8E24AA", ls="--", lw=1.2,
               label=f"honest-student tau_max={OP_MAX:.3g} -> {op['pooled_bits_at_max']:.3f} bits/tok")
    ax.set_xscale("log")
    ax.set_xlabel("DiFR verifier tolerance  delta = tau  (nats, post-Gumbel margin)")
    ax.set_ylabel("covert capacity  E[log2 |acceptance set|]  (bits / served token)")
    ax.set_title("DiFR logit-tolerance channel — llama-68m, seq 1024, 3 real-text contexts\n"
                 "verifier = FP8-dynamic teacher; acceptance = post-Gumbel margin <= tau (harness form)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig("difr_delta_curve.png", dpi=150)
    print("wrote exp_b_results.json / difr_delta_curve.png")
    print(json.dumps(op, indent=2))


if __name__ == "__main__":
    main()
