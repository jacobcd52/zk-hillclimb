"""C(b) capacity sweep on the CORRECTED-orientation dumps.

Identical formulas/plots to measure/capacity_analyze.py — we import its
analyze() / plot_scheme() / plot_combined() unchanged so the math is
byte-identical — pointed at capacity_dump_corrected_{scheme}_seed{seed}.npz
(reference = M_int, served = FP8 argmax). Only labels and output names differ:

  plots   -> ../capacity_corrected_{scheme}.png, ../capacity_corrected_combined.png
  results -> capacity_corrected_results_seed{seed}.json

Run:
  /root/int-model-env/bin/python capacity_analyze_corrected.py --seed 20260611
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURE = os.path.join(os.path.dirname(HERE), "measure")
sys.path.insert(0, MEASURE)

import capacity_analyze as ca

OUTDIR = os.path.dirname(HERE)
SCHEMES = ["baseline", "faithful", "codebook"]

# corrected-orientation labels (the old ones quoted the swapped-direction DiFR)
ca.LABELS = {
    "baseline": "baseline-native (ref=M_int, served=FP8)",
    "faithful": "faithful-arch-v1 (ref=M_int, served=FP8)",
    "codebook": "codebook (ref=M_int, served=FP8)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    results = {}
    for sc in SCHEMES:
        npz = os.path.join(HERE, f"capacity_dump_corrected_{sc}_seed{a.seed}.npz")
        if not os.path.exists(npz):
            print(f"skip {sc}: no dump at {npz}"); continue
        results[sc] = ca.analyze(npz)
        ca.plot_scheme(results[sc], os.path.join(OUTDIR, f"capacity_corrected_{sc}.png"))
        r = results[sc]
        print(f"[{sc}] min simple C = {r['min_simple']['C_min_bits_per_token']:.4f} "
              f"b/tok @ b*={r['min_simple']['b_star']:.4g} "
              f"(p*={r['min_simple']['p_star']:.4g}); "
              f"top-K min = {r['min_topK']['C_min_bits_per_token']:.4f}")
        e0 = r["endpoint_b0"]; ei = r["endpoint_binf"]
        print(f"      b=0: C_simple={e0['C_simple']:.4f} (formula check "
              f"{e0['C0_formula_check_H(p0)+p0*log2V']:.4f}, Nb={e0['mean_Nb_at_b0']:.2f}); "
              f"b={ei['b']:.0f}: C_simple={ei['C_simple']:.4f} (->log2V {ei['log2V_target']:.4f}, "
              f"Nb={ei['mean_Nb_at_binf']:.0f})")
    if set(results) >= set(SCHEMES):
        ca.plot_combined(results, os.path.join(OUTDIR, "capacity_corrected_combined.png"))
    out = os.path.join(HERE, f"capacity_corrected_results_seed{a.seed}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
    print("plots ->", OUTDIR)


if __name__ == "__main__":
    main()
