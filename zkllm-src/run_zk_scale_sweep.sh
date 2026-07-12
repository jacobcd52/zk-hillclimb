#!/bin/bash
# ZK-at-scale sweep (design doc section 22): one config per process (VmHWM is
# per-point), sequential (one GPU job at a time), results appended to
# /root/zk_scale_results.log as BENCH/MBENCH lines.
set -u
cd /root/zkllm
OUT=/root/zk_scale_results.log
TB=/root/p3_tb_c22
MB=/root/p3_model_bench
run() {
  echo "### $*" | tee -a $OUT
  P3_MEMLOG=1 P3_ZKPROF=1 timeout 7200 "$@" 2>&1 | grep -E "BENCH|MBENCH|STAGES|FATAL|witness" | tee -a $OUT
  echo "exit=$?" | tee -a $OUT
}
# d=64 seq scaling, zk=1
run $TB 256 64 2 32 128 1 1 tables_ld6.bin
run $TB 512 64 2 32 128 1 1 tables_ld6.bin
run $TB 1024 64 2 32 128 1 1 tables_ld6.bin
# batch scaling (well-utilized forward), zk=1
run $TB 128 64 2 32 128 4 1 tables_ld6.bin
run $TB 128 64 2 32 128 16 1 tables_ld6.bin
run $TB 256 64 2 32 128 16 1 tables_ld6.bin
run $TB 128 64 2 32 128 64 1 tables_ld6.bin
# d=256, zk=1
run $TB 128 256 4 64 1024 1 1 tables_ld8.bin
run $TB 256 256 4 64 1024 1 1 tables_ld8.bin
# full model, zk=1
run $MB 1 128 64 2 32 128 256 1 tables_ld6.bin
run $MB 2 128 64 2 32 128 256 1 tables_ld6.bin
run $MB 4 128 64 2 32 128 256 1 tables_ld6.bin
run $MB 2 256 64 2 32 128 256 1 tables_ld6.bin
echo "SWEEP DONE" | tee -a $OUT
