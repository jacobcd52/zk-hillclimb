#!/bin/bash
# Run ONE bench config with FULL output captured (MEMLOG + ZKPROF + stderr)
# to /root/zkrun_<tag>.log, and append the summary lines to the results log.
# usage: run_one_zk.sh <tag> <binary> <args...>
set -u
cd /root/zkllm
TAG=$1; shift
LOG=/root/zkrun_$TAG.log
OUT=/root/zk_scale_results.log
echo "### $*" | tee -a $OUT
P3_MEMLOG=1 P3_ZKPROF=1 stdbuf -oL -eL timeout 7200 "$@" > $LOG 2>&1
rc=$?
echo "exit=$rc" | tee -a $OUT
grep -E "BENCH|MBENCH|STAGES|FATAL|witness" $LOG | tee -a $OUT
exit $rc
