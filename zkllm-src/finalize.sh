#!/bin/bash
cd /root/zkllm
until [ -f sweep_results.csv ] && grep -q WROTE sweep_log.txt 2>/dev/null; do sleep 10; done
/workspace/envs/aqlm/bin/python plot.py
python3 - <<'PY'
import csv
rows=list(csv.DictReader(open("sweep_results.csv")))
hdr="| config | B | IN | OUT | forward (us) | prove (ms) | verify (ms) | proof (KB) | overhead |"
print(hdr); print("|"+"---|"*9)
for r in rows:
    print(f"| {r['label']} | {r['B']} | {r['IN']} | {r['OUT']} | {r['fwd_us']} | {r['prove_ms']} | {r['verify_ms']} | {r['proof_kb']} | {int(float(r['overhead_x'])):,}x |")
PY
