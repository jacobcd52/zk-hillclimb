#!/bin/bash
# Post-fix acceptance gates, sequential. Battery binaries (*.bat) are already
# built by the debug agent; run them, then compact teeth, forced-stream pairs,
# and the two real reruns with /root/p3_tb_c24r.
set -u
cd /root/zkllm
OUT=/root/gates_result.log
: > $OUT
TESTS="p3_matmul_selftest p3_rmsnorm_test p3_swiglu_test p3_quant_test p3_bfadd_test p3_rope_test p3_softmax_test p3_gkr_selftest p3_logup_test p3_hawkeye_zk_test zk_gadget_smoke p3_transformer_test zk_layer_smoke p3_transformer_zk_test p3_model_test zk_model_smoke p3_model_zk_test p3_binius_field_test p3_binius_ntt_test p3_binius_pcs_test p3_binius_sumcheck_test p3_binius_e2e_test p3_binius_hawkeye_test p3_binius_logup_test p3_binius_acc_test p3_binius_trans_test"
echo "=== battery run ===" | tee -a $OUT
fail=0
for t in $TESTS; do
  out=$(/root/$t.bat 2>/dev/null | grep -E "passed, [0-9]+ failed" | tail -1)
  echo "$t: $out" | tee -a $OUT
  echo "$out" | grep -q "ALL PASS" || { echo "  ^^^ FAIL" | tee -a $OUT; fail=1; }
done
[ $fail -eq 0 ] && echo "BATTERY: ALL GREEN" | tee -a $OUT || echo "BATTERY: FAILURES" | tee -a $OUT
echo "=== compact teeth ===" | tee -a $OUT
bash run_compact_teeth.sh >> $OUT 2>&1 && echo "COMPACT: OK" | tee -a $OUT || echo "COMPACT: FAIL" | tee -a $OUT
echo "=== forced-stream pairs (c24r) ===" | tee -a $OUT
for env in "" "P3_SC5ZG_CAP=800000000" "P3_SBLIND_MIN=10" "P3_SBLIND_MIN=10 P3_SC5ZG_CAP=800000000"; do
  line=$(env $env /root/p3_tb_c24r 64 64 2 32 128 1 1 tables_ld6.bin 2>/dev/null | grep -a BENCH)
  echo "[$env] $line" | tee -a $OUT
done
echo "=== rerun s1024 zk1 ===" | tee -a $OUT
P3_MEMLOG=1 P3_ZKPROF=1 timeout 7200 /root/p3_tb_c24r 1024 64 2 32 128 1 1 tables_ld6.bin > /root/zkrun_s1024r.log 2>&1
echo "exit=$?" | tee -a $OUT
grep -aE "BENCH|STAGES|FATAL|terminate" /root/zkrun_s1024r.log | tee -a $OUT
echo "=== rerun d256s256 zk1 ===" | tee -a $OUT
P3_MEMLOG=1 P3_ZKPROF=1 timeout 7200 /root/p3_tb_c24r 256 256 4 64 1024 1 1 tables_ld8.bin > /root/zkrun_d256s256r.log 2>&1
echo "exit=$?" | tee -a $OUT
grep -aE "BENCH|STAGES|FATAL|terminate" /root/zkrun_d256s256r.log | tee -a $OUT
echo "GATES DONE" | tee -a $OUT
