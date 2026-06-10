#!/bin/bash
# Orchestrator selftest (ORCHESTRATOR_DESIGN.md §7):
#  (a) register -> prove_walk -> verify_walk on real llama-68m data: ACCEPT,
#      all covered ids checked; check_transcript vs stage-1 manifest passes
#      (vs the FULL frozen manifest the stage-1 gap is printed, not hidden).
#  (b) tamper one chained commitment file -> verify_walk REJECTS the right id.
#  (c) tamper one registered-weight hash in public.json -> REJECT at registration.
set -u
PY=/root/int-model-env/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
HARNESS=/workspace/projects/zk-hillclimb/harness
RUN_ID="${1:-selftest-$(date +%Y%m%d-%H%M%S)}"
RUN=/root/zkorch/$RUN_ID
PASS=0; FAIL=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

echo "=== selftest run: $RUN ==="

# ---------- (a) honest end-to-end ----------
$PY "$HERE/register.py" --run-id "$RUN_ID" || { bad "register"; echo "ABORT"; exit 1; }
ok "register"
$PY "$HERE/prove_walk.py" "$RUN" || { bad "prove_walk"; echo "ABORT"; exit 1; }
ok "prove_walk"
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript.json"; then
  ok "verify_walk honest ACCEPT"
else
  bad "verify_walk honest ACCEPT"
fi
N_CHECKED=$($PY -c "import json;print(len(json.load(open('$RUN/transcript.json'))['checked']))")
N_EXPECT=$($PY -c "import sys;sys.path.insert(0,'$HERE');import common;print(len(common.covered_ids()))")
if [ "$N_CHECKED" = "$N_EXPECT" ]; then ok "all $N_EXPECT covered ids checked"; else bad "checked=$N_CHECKED expected=$N_EXPECT"; fi

$PY "$HERE/make_stage1_manifest.py" "$HARNESS/manifest_llama68m.json" "$RUN/manifest_stage1.json"
if $PY "$HARNESS/check_transcript.py" "$RUN/manifest_stage1.json" "$RUN/transcript.json"; then
  ok "check_transcript vs stage-1 scope manifest"
else
  bad "check_transcript vs stage-1 scope manifest"
fi
echo "--- check_transcript vs FULL frozen manifest (stage-1 gap, reported honestly) ---"
if $PY "$HARNESS/check_transcript.py" "$HARNESS/manifest_llama68m.json" "$RUN/transcript.json"; then
  bad "full-manifest check unexpectedly passed (attention not proven — should have a gap!)"
else
  ok "full-manifest gap correctly reported (attention/softmax/lm_head pending)"
fi

# ---------- (b) tamper a chained commitment file ----------
TGT="$RUN/proofs/layer0.mlp.swiglu/glu/com_H.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[24] ^= 0xFF   # first point x-limbs (rmsnorm tamper-offset convention)
open(p, "wb").write(bytes(b))
EOF
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_tamper_chain.json"; then
  bad "tampered com_H accepted"
else
  HIT=$($PY -c "import json;t=json.load(open('$RUN/transcript_tamper_chain.json'));print('layer0.mlp.swiglu' in t['rejected'] and t['verdict']=='REJECT')")
  if [ "$HIT" = "True" ]; then ok "tampered chained commitment REJECTED at layer0.mlp.swiglu"; else bad "REJECT but wrong id"; fi
fi
mv "$TGT.bak" "$TGT"

# ---------- (c) tamper a registered hash in public.json ----------
cp "$RUN/public.json" "$RUN/public.json.bak"
$PY - "$RUN/public.json" <<'EOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
k = sorted(d["registered_weight_commitments"])[0]
h = d["registered_weight_commitments"][k]
d["registered_weight_commitments"][k] = ("0" if h[0] != "0" else "1") + h[1:]
json.dump(d, open(p, "w"), indent=2, sort_keys=True)
EOF
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_tamper_reg.json"; then
  bad "tampered registration accepted"
else
  HIT=$($PY -c "import json;t=json.load(open('$RUN/transcript_tamper_reg.json'));d=t['details'].get('statement.registered_weight_hash',{});print(t['verdict']=='REJECT' and d.get('ok')==False)")
  if [ "$HIT" = "True" ]; then ok "tampered registered hash REJECTED at registration check"; else bad "REJECT but wrong locus"; fi
fi
mv "$RUN/public.json.bak" "$RUN/public.json"

# ---------- re-verify restored run still ACCEPTs ----------
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_restored.json"; then
  ok "restored run re-ACCEPTs"
else
  bad "restored run re-ACCEPTs"
fi

echo "=== $PASS PASS / $FAIL FAIL ==="
if [ "$FAIL" = "0" ]; then echo "ALL PASS"; exit 0; else echo "SELFTEST FAILED"; exit 1; fi
