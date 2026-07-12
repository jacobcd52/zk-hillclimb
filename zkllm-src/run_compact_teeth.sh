#!/bin/bash
# Section-22 compaction teeth: re-run the composed + hawkeye + logup suites
# with P3_COMPACT_MIN=0 so EVERY committed column takes the compact/resolver
# path even at test dims (the default 2^20 threshold leaves tiny-dim tests on
# the raw path).  All suites must stay ALL PASS.
set -u
cd /root/zkllm
export P3_COMPACT_MIN=0
fail=0
for t in p3_hawkeye_zk_test p3_logup_test zk_gadget_smoke p3_transformer_test \
         zk_layer_smoke p3_transformer_zk_test p3_model_test zk_model_smoke \
         p3_model_zk_test; do
  out=$(/root/$t.bat 2>/dev/null | grep -E "passed, [0-9]+ failed" | tail -1)
  echo "$t (P3_COMPACT_MIN=0): $out"
  echo "$out" | grep -q "ALL PASS" || { echo "  ^^^ FAIL"; fail=1; }
done
if [ $fail -eq 0 ]; then echo "COMPACT TEETH: ALL GREEN"; else echo "COMPACT TEETH: FAILURES"; fi
exit $fail
