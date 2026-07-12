#!/bin/bash
# Final ZK-at-scale configs with the c24 binaries (per-column GPU bind +
# streamed sumcheck prefix + mixed batch-open parking).  Sequential, one at a
# time; REAL exit codes; full per-run logs in /root/zkrun_<tag>.log.
set -u
cd /root/zkllm
OUT=/root/zk_scale_results3.log
TB=/root/p3_tb_c24
run() {
  local tag=$1; shift
  echo "### $tag: $*" | tee -a $OUT
  P3_MEMLOG=1 P3_ZKPROF=1 timeout 7200 "$@" > /root/zkrun_$tag.log 2>&1
  local rc=$?
  echo "exit=$rc" | tee -a $OUT
  grep -aE "BENCH|MBENCH|STAGES|FATAL|terminate|witness [0-9]" /root/zkrun_$tag.log | tee -a $OUT
}
# the two configs unlocked by the new fixes
run s1024c24    $TB 1024 64 2 32 128 1 1 tables_ld6.bin
run d256s256c24 $TB 256 256 4 64 1024 1 1 tables_ld8.bin
# big-token attempts (may hit the 41 GB container cap on witness alone)
run s256b16c24  $TB 256 64 2 32 128 16 1 tables_ld6.bin
run s128b64c24  $TB 128 64 2 32 128 64 1 tables_ld6.bin
# zk=0 reference points for the overhead ratios
run s1024z0     $TB 1024 64 2 32 128 1 0 tables_ld6.bin
run s512z0      $TB 512 64 2 32 128 1 0 tables_ld6.bin
run d256s128z0  $TB 128 256 4 64 1024 1 0 tables_ld8.bin
run d256s256z0  $TB 256 256 4 64 1024 1 0 tables_ld8.bin
echo "FINAL SWEEP DONE" | tee -a $OUT
