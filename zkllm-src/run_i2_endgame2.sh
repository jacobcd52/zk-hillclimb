#!/bin/bash
# Iteration-2 endgame restart: gates3 then the token-scale reruns, sequential.
echo 1000 > /proc/self/oom_score_adj
cd /root/zkllm
R=/root/zk_scale_results6.log
ST=/root/run_i2_endgame2_status.log
: > $ST
grep -a -E '^BENCH|^STAGES' /root/zkrun_i2d_d256s256.log | sed 's/^/i2d_d256s256 /' >> $R

echo "[$(date +%H:%M:%S)] gates3 start" >> $ST
bash run_gates3.sh > /root/zkrun_i2_gates3.log 2>&1
echo "[$(date +%H:%M:%S)] gates3 done rc=$?" >> $ST

for cfg in "s256b16 256 64 2 32 128 16" "s128b64 128 64 2 32 128 64" "s256b64 256 64 2 32 128 64"; do
  set -- $cfg
  tag=$1; shift
  echo "[$(date +%H:%M:%S)] $tag start" >> $ST
  env P3_MEMLOG=1 P3_ZKPROF=1 timeout 7200 /root/p3_tb_i2d "$@" 1 tables_ld6.bin > /root/zkrun_i2d_$tag.log 2>&1
  echo "[$(date +%H:%M:%S)] $tag done rc=$?" >> $ST
  grep -a -E '^BENCH|^STAGES' /root/zkrun_i2d_$tag.log | sed "s/^/i2d_$tag /" >> $R
done
echo "[$(date +%H:%M:%S)] ENDGAME2 DONE" >> $ST
