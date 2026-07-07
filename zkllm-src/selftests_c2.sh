#!/bin/bash
# Stage C2 header-edit re-validation: every TU including zkob_claims.cuh was
# rebuilt after the batch_prove streaming edit; per the PHASE0 §13 / Stage-B
# header-edit rule, every includer's FULL selftest re-runs and must pass.
cd /root/zkllm
FAIL=0
run() {
  local name=$1; shift
  echo "=== selftest: $name ==="
  local t0=$(date +%s)
  if "$@" > /tmp/st_$name.log 2>&1; then
    local marker=$(grep -E "ALL PASS|TOY-BATCHOPEN" /tmp/st_$name.log | tail -1)
    echo "    PASS ($(($(date +%s)-t0)) s): $marker"
  else
    echo "    FAIL ($(($(date +%s)-t0)) s) — tail:"
    tail -5 /tmp/st_$name.log | sed 's/^/    /'
    FAIL=1
  fi
}
run vrf_toy_batchopen ./vrf_toy_batchopen
run zkob_batchopen ./zkob_batchopen selftest
run zkob_fc ./zkob_fc selftest
run zkob_rescale ./zkob_rescale selftest
run zkob_skip ./zkob_skip selftest
run zkob_glu ./zkob_glu selftest
run zkob_rope ./zkob_rope selftest
run zkob_headmerge ./zkob_headmerge selftest
run zkob_headslice ./zkob_headslice selftest
run zkob_rmsnorm ./zkob_rmsnorm selftest
run zkob_softmax ./zkob_softmax selftest
run zkob_softmax8 ./zkob_softmax8 selftest
run zkob_rowmax ./zkob_rowmax selftest
if [ "$FAIL" = "0" ]; then echo "C2 HEADER-EDIT REVALIDATION: ALL 13 PASS"; else echo "C2 REVALIDATION: FAILURES"; exit 1; fi
