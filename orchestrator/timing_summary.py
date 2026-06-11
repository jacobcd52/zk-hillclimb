"""Aggregate per-obligation prove/verify timings + proof bytes for the report.

Usage: python timing_summary.py <run_dir>

Groups the prove_manifest.json runs and transcript.json timing entries by
manifest id (layer index folded into layer{0,1} pairs like the stage-1 report
table) and prints a markdown table plus totals.
"""
import json
import os
import re
import sys
from collections import defaultdict

run_dir = sys.argv[1]
pm = json.load(open(os.path.join(run_dir, "prove_manifest.json")))
tr = json.load(open(os.path.join(run_dir, "transcript.json")))


def fold(mid):
    return re.sub(r"^layer[01]\.", "layer{0,1}.", mid)


prove = defaultdict(lambda: [0.0, 0])      # folded id -> [sum_s, n_runs]
for r in pm["runs"]:
    f = fold(r["manifest_id"])
    prove[f][0] += r["seconds"]
    prove[f][1] += 1

verify = defaultdict(lambda: [0.0, 0])
for key, s in tr["timing"].items():
    if key == "total_verify_wall_s":
        continue
    m = re.match(r"(?:edge:)?([^\[]+)(?:\[.*\])?$", key)
    f = fold(m.group(1))
    verify[f][0] += s
    verify[f][1] += 1

order = []
for mid in [r["manifest_id"] for r in pm["runs"]]:
    f = fold(mid)
    if f not in order:
        order.append(f)

print("| obligation (manifest id, both layers) | prove runs | prove s | verify runs | verify s |")
print("|---|--:|--:|--:|--:|")
tp = tv = 0.0
for f in order:
    ps, pn = prove[f]
    vs, vn = verify.get(f, [0.0, 0])
    tp += ps
    tv += vs
    print(f"| {f} | {pn} | {ps:.1f} | {vn} | {vs:.1f} |")
print(f"| **TOTAL (driver time)** | {sum(n for _, n in prove.values())} | **{tp:.1f}** | "
      f"{sum(n for _, n in verify.values())} | **{tv:.1f}** |")
print()
print(f"prove wall (incl. witness/python): {pm['totals']['prove_wall_s']} s")
print(f"verify wall: {tr['timing']['total_verify_wall_s']} s")
print(f"proof bytes: {pm['totals']['proof_bytes']:,}")

# attention-only subtotal (the stage-2 delta)
attn_p = sum(s for r in pm["runs"] if ".attn" in r["manifest_id"] for s in [r["seconds"]])
attn_v = sum(s for k, s in tr["timing"].items()
             if ".attn" in k and k != "total_verify_wall_s")
print(f"attention-segment driver time: prove {attn_p:.1f} s / verify {attn_v:.1f} s")
