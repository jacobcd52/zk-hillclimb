#!/bin/bash
# Full regression battery: builds all 23 test binaries (parallel) and runs them
# sequentially, printing each suite's summary line.  Exit 0 iff every suite
# reports ALL PASS (or its expected pass-count line).
# Suites 18-22 are the Binius substrate (design doc section 21); suite 23 is
# the Binius Hawkeye per-product gadget migration (section 21.8).
set -u
cd /root/zkllm
NV="nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp"
TESTS="p3_matmul_selftest p3_rmsnorm_test p3_swiglu_test p3_quant_test p3_bfadd_test p3_rope_test p3_softmax_test p3_gkr_selftest p3_logup_test p3_hawkeye_zk_test zk_gadget_smoke p3_transformer_test zk_layer_smoke p3_transformer_zk_test p3_model_test zk_model_smoke p3_model_zk_test p3_binius_field_test p3_binius_ntt_test p3_binius_pcs_test p3_binius_sumcheck_test p3_binius_e2e_test p3_binius_hawkeye_test"
echo "=== build ==="
pids=""
for t in $TESTS; do
  ( $NV $t.cu -o /root/$t.new 2>/root/$t.buildlog && mv /root/$t.new /root/$t.bat ) &
  pids="$pids $!"
done
bfail=0
for p in $pids; do wait $p || bfail=1; done
if [ $bfail -ne 0 ]; then
  echo "BUILD FAILURE:"; grep -l "error" /root/*.buildlog 2>/dev/null; exit 1
fi
echo "=== run ==="
fail=0
for t in $TESTS; do
  out=$(/root/$t.bat 2>/dev/null | grep -E "passed, [0-9]+ failed" | tail -1)
  echo "$t: $out"
  echo "$out" | grep -q "ALL PASS" || { echo "  ^^^ FAIL"; fail=1; }
done
if [ $fail -eq 0 ]; then echo "BATTERY: ALL GREEN"; else echo "BATTERY: FAILURES"; fi
exit $fail
