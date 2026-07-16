#!/bin/bash
# Stretch sweep: push seq len and model size on both lines. Sequential, one GPU
# job at a time, OOM-shielded so the bench dies (not the pod) if it breaches 41 GB.
echo 1000 > /proc/self/oom_score_adj
cd /root/zkllm
ST=/root/run_stretch_status.log; : > $ST
run() {  # tag  binary  args...
  local tag=$1; shift
  echo "[$(date +%H:%M:%S)] START $tag: $*" >> $ST
  P3_MEMLOG=1 timeout 7200 "$@" > /root/zkrun_stretch_$tag.log 2>&1
  echo "[$(date +%H:%M:%S)] END $tag rc=$?" >> $ST
  grep -aE "BENCH" /root/zkrun_stretch_$tag.log | tail -1 >> $ST
}
IB=/root/p3_int_layer_bench
FB=/root/p3_tb_i2d
# --- INTEGER line: push seq (d=64) then model (seq=64), lots of headroom ---
run int_s2048  $IB 2048 64 2 32 128 1 1 tables_ld6.bin
run int_s4096  $IB 4096 64 2 32 128 1 1 tables_ld6.bin
run int_d1024  $IB 64 1024 16 64 4096 1 1 tables_ld10.bin
run int_d2048  $IB 64 2048 16 128 8192 1 1 tables_ld10.bin
run int_b256   $IB 128 64 2 32 128 256 1 tables_ld6.bin
# --- FP8 line: the two plausible stretches (both may OOM; that IS the answer) ---
run fp8_s2048  $FB 2048 64 2 32 128 1 1 tables_ld6.bin
run fp8_d1024  $FB 64 1024 16 64 4096 1 1 tables_ld10.bin
echo "[$(date +%H:%M:%S)] STRETCH DONE" >> $ST
