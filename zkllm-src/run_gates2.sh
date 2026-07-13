#!/bin/bash
# Final gates for the scaling-study source: full battery rebuild+run, compact
# teeth, forced-stream identity pairs (new levers at defaults), guard reruns.
set -u
cd /root/zkllm
OUT=/root/gates2_result.log
: > $OUT
echo "=== battery (rebuild + run) ===" | tee -a $OUT
bash run_battery.sh >> $OUT 2>&1 && echo "BATTERY: ALL GREEN" | tee -a $OUT || echo "BATTERY: FAILURES" | tee -a $OUT
echo "=== compact teeth ===" | tee -a $OUT
bash run_compact_teeth.sh >> $OUT 2>&1 && echo "COMPACT: OK" | tee -a $OUT || echo "COMPACT: FAIL" | tee -a $OUT
echo "=== identity pairs (p3_tb_s2, levers at defaults) ===" | tee -a $OUT
for env in "" "P3_SC5ZG_CAP=800000000" "P3_SBLIND_MIN=10" "P3_SBLIND_MIN=10 P3_SC5ZG_CAP=800000000" "P3_CPK_DEV=0" "P3_PK_SPILL=/workspace/p3_spill"; do
  line=$(env $env /root/p3_tb_s2 64 64 2 32 128 1 1 tables_ld6.bin 2>/dev/null | grep -a BENCH)
  echo "[$env] $line" | tee -a $OUT
done
echo "=== guard reruns ===" | tee -a $OUT
P3_MEMLOG=1 timeout 3600 /root/p3_tb_s2 256 64 2 32 128 1 1 tables_ld6.bin 2>/dev/null | grep -a BENCH | tee -a $OUT
P3_MEMLOG=1 timeout 3600 /root/p3_tb_s2 128 256 4 64 1024 1 1 tables_ld8.bin 2>/dev/null | grep -a BENCH | tee -a $OUT
echo "GATES2 DONE" | tee -a $OUT
