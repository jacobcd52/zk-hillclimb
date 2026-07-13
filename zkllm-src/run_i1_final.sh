#!/bin/bash
# Iteration-1 final chained runner: gates + headline reruns + 16384-token attempt.
# Runs sequentially (ONE GPU job at a time), OOM-shielded, survives the agent session.
echo 1000 > /proc/self/oom_score_adj
cd /root/zkllm
R=/root/zk_scale_results6.log
S=/root/run_i1_final_status.log
: > $S

echo "[$(date +%H:%M:%S)] STEP gates start" >> $S
bash run_gates2.sh > /root/zkrun_i1_gates2.log 2>&1
echo "[$(date +%H:%M:%S)] STEP gates done rc=$?" >> $S

echo "[$(date +%H:%M:%S)] STEP d256s256 start" >> $S
env P3_MEMLOG=1 P3_ZKPROF=1 timeout 7200 /root/p3_tb_i1c 256 256 4 64 1024 1 1 tables_ld8.bin > /root/zkrun_i1c_d256s256.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] STEP d256s256 done rc=$rc" >> $S
grep -a -E '^BENCH|^STAGES' /root/zkrun_i1c_d256s256.log >> $R

echo "[$(date +%H:%M:%S)] STEP s256b16 start" >> $S
env P3_MEMLOG=1 P3_ZKPROF=1 timeout 7200 /root/p3_tb_i1c 256 64 2 32 128 16 1 tables_ld6.bin > /root/zkrun_i1c_s256b16.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] STEP s256b16 done rc=$rc" >> $S
grep -a -E '^BENCH|^STAGES' /root/zkrun_i1c_s256b16.log >> $R

echo "[$(date +%H:%M:%S)] STEP s256b64 (16384 tok) start" >> $S
env P3_MEMLOG=1 P3_ZKPROF=1 timeout 7200 /root/p3_tb_i1c 256 64 2 32 128 64 1 tables_ld6.bin > /root/zkrun_i1c_s256b64.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] STEP s256b64 done rc=$rc" >> $S
grep -a -E '^BENCH|^STAGES' /root/zkrun_i1c_s256b64.log >> $R

echo "[$(date +%H:%M:%S)] RUNNER DONE" >> $S
