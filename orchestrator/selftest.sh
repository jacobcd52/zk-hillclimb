#!/bin/bash
# Orchestrator selftest, STAGE 3 (full statement: MLP + rmsnorm + skips +
# complete attention chain + final_norm + lm_head + served-token argmax
# binding; manifest CLOSED per STAGE3_FAITHFUL_DESIGN §3.5 — 56/56 non-waived
# checked + 3 covered-waived; only embedding.lookup stays waived-uncovered):
#  (a) register -> prove_walk -> verify_walk on real llama-68m data: ACCEPT,
#      checked = 56 non-waived + 3 covered-waived = 59 [arithmetic recounted
#      from the manifest and printed]; skipped = {} (nothing stage-waived);
#      check_transcript vs the FULL FROZEN manifest passes (exit 0) with the
#      covered-waived NOTE listing exactly the 3 ids.
#  (b) tamper a slice commitment (proofs/.../slice/com_KhT05.bin) ->
#      verify_walk REJECTs, localized to layer0.attn.scores_matmul.  [stage 2]
#  (c) tamper the registered rope-cos table hash in public.json -> REJECT at
#      the registration check, NO drivers run.                       [stage 2]
#  (d) tamper one byte of statement.logit_binding/rowmax/com_S.bin -> REJECT
#      localized to statement.logit_binding (transcript divergence — com_S is
#      absorbed before any challenge).                               [stage 3]
#  (e) flip one token id in registration/tstar.i32.bin (tamper t*) -> REJECT
#      at the REGISTRATION HASH check, no drivers run. §3.4's ordering
#      dictates this locus: t* is sha256-pinned inside public.json and the
#      registration phase re-hashes fail-closed BEFORE any driver, so the
#      tamper never reaches statement.logit_binding's rowmax verify.
#  (f) tamper the registered lm_head commitment hash in public.json ->
#      registration REJECT, no drivers run.                          [stage 3]
#  (g) restore everything -> full re-ACCEPT with checked = 59.
set -u
PY=/root/int-model-env/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
HARNESS=/workspace/projects/zk-hillclimb/harness
RUN_ID="${1:-selftest3-$(date +%Y%m%d-%H%M%S)}"
RUN=/root/zkorch/$RUN_ID
PASS=0; FAIL=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

echo "=== stage-3 selftest run: $RUN ==="

# ---------- expected-count arithmetic, recounted from the manifest ----------
read -r N_NONWAIVED N_COVWAIVED N_EXPECT <<<"$($PY - "$HERE" "$HARNESS/manifest_llama68m.json" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1])
import common
man = json.load(open(sys.argv[2]))
nonwaived = {o["id"] for o in man["obligations"] if not o["waived"]}
waived = {o["id"] for o in man["obligations"] if o["waived"]}
assert common.skipped_ids() == {}, "stage 3 must skip nothing"
cov = set(common.covered_ids())
missing = sorted(nonwaived - cov)
assert not missing, f"non-waived ids not covered: {missing}"
covwaived = sorted(cov - nonwaived)
assert set(covwaived) <= waived, f"covered ids outside the manifest: {sorted(set(covwaived)-waived)}"
assert covwaived == ["final_norm.rmsnorm", "lm_head.matmul", "lm_head.rescaling"], covwaived
print(len(nonwaived), len(covwaived), len(cov))
EOF
)"
echo "manifest arithmetic: $N_NONWAIVED non-waived + $N_COVWAIVED covered-waived (final_norm.rmsnorm, lm_head.matmul, lm_head.rescaling) = $N_EXPECT expected checked; 0 stage-skipped"

# ---------- (a) honest end-to-end ----------
$PY "$HERE/register.py" --run-id "$RUN_ID" || { bad "register"; echo "ABORT"; exit 1; }
ok "register"
$PY "$HERE/prove_walk.py" "$RUN" || { bad "prove_walk"; echo "ABORT"; exit 1; }
ok "prove_walk"
TIE_BITS=$($PY -c "import json;print(json.load(open('$RUN/prove_manifest.json'))['rowmax_selector_ties']['total_bits'])")
echo "measured rowmax selector-tie duty: $TIE_BITS bits (sum log2(#maximizers), logit grid)"
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript.json"; then
  ok "verify_walk honest ACCEPT"
else
  bad "verify_walk honest ACCEPT"
fi
N_CHECKED=$($PY -c "import json;print(len(json.load(open('$RUN/transcript.json'))['checked']))")
N_SKIPPED=$($PY -c "import json;print(len(json.load(open('$RUN/transcript.json'))['skipped']))")
if [ "$N_CHECKED" = "$N_EXPECT" ]; then ok "checked = $N_EXPECT (= $N_NONWAIVED non-waived + $N_COVWAIVED covered-waived)"; else bad "checked=$N_CHECKED expected=$N_EXPECT"; fi
if [ "$N_SKIPPED" = "0" ]; then ok "skipped = {} (nothing stage-waived)"; else bad "skipped not empty (n=$N_SKIPPED)"; fi

echo "--- check_transcript vs the FULL FROZEN manifest (stage 3: must PASS with the covered-waived NOTE) ---"
FULL_OUT=$($PY "$HARNESS/check_transcript.py" "$HARNESS/manifest_llama68m.json" "$RUN/transcript.json" 2>&1)
FULL_RC=$?
echo "$FULL_OUT"
NOTE_OK=$($PY - "$FULL_OUT" <<'EOF'
import sys
out = sys.argv[1]
counts_ok = "required: 56  checked: 59  missing: 0  unknown: 0" in out
note = [l for l in out.splitlines() if l.startswith("NOTE:")]
note_ok = (len(note) == 1 and "3 waived obligations now covered" in note[0]
           and all(i in note[0] for i in ("final_norm.rmsnorm", "lm_head.matmul", "lm_head.rescaling")))
print(counts_ok and note_ok)
EOF
)
if [ "$FULL_RC" = "0" ] && [ "$NOTE_OK" = "True" ]; then
  ok "check_transcript vs FROZEN manifest: PASS (56 required, 59 checked, 0 missing) + covered-waived NOTE (3 ids)"
else
  bad "check_transcript vs FROZEN manifest (rc=$FULL_RC, counts+NOTE ok=$NOTE_OK)"
fi

# ---------- (b) tamper a slice commitment [stage-2 phase, kept] ----------
TGT="$RUN/proofs/layer0.attn.scores_matmul/slice/com_KhT05.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[24] ^= 0xFF   # first point x-limbs (pinned tamper-offset convention)
open(p, "wb").write(bytes(b))
EOF
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_tamper_slice.json"; then
  bad "tampered com_KhT05 accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$RUN/transcript_tamper_slice.json'))
layer_rej = [m for m in t['rejected'] if m.startswith('layer')]
print(t['verdict'] == 'REJECT' and layer_rej == ['layer0.attn.scores_matmul'])")
  if [ "$HIT" = "True" ]; then
    ok "tampered slice commitment REJECTED, localized to layer0.attn.scores_matmul"
  else
    bad "REJECT but wrong localization: $($PY -c "import json;print(json.load(open('$RUN/transcript_tamper_slice.json'))['rejected'])")"
  fi
fi
mv "$TGT.bak" "$TGT"

# ---------- (c) tamper the registered rope-cos table hash [stage-2 phase, kept] ----------
cp "$RUN/public.json" "$RUN/public.json.bak"
$PY - "$RUN/public.json" <<'EOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
h = d["tables"]["rope-cos-table.bin"]
d["tables"]["rope-cos-table.bin"] = ("0" if h[0] != "0" else "1") + h[1:]
json.dump(d, open(p, "w"), indent=2, sort_keys=True)
EOF
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_tamper_ropetab.json"; then
  bad "tampered rope-cos registration accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$RUN/transcript_tamper_ropetab.json'))
d = t['details'].get('statement.registered_weight_hash', {})
no_drivers = t.get('checked') == [] and 'not run' in t.get('note', '')
print(t['verdict'] == 'REJECT' and d.get('ok') is False and no_drivers)")
  if [ "$HIT" = "True" ]; then
    ok "tampered rope-cos hash REJECTED at registration, no drivers run"
  else
    bad "REJECT but wrong locus / drivers ran"
  fi
fi
mv "$RUN/public.json.bak" "$RUN/public.json"

# ---------- (d) tamper the logit-binding selector commitment [stage 3] ----------
TGT="$RUN/proofs/statement.logit_binding/rowmax/com_S.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[24] ^= 0xFF   # pinned tamper-offset convention (com_* at 24)
open(p, "wb").write(bytes(b))
EOF
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_tamper_comS.json"; then
  bad "tampered logit-binding com_S accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$RUN/transcript_tamper_comS.json'))
print(t['verdict'] == 'REJECT' and t['rejected'] == ['statement.logit_binding'])")
  if [ "$HIT" = "True" ]; then
    ok "tampered rowmax com_S REJECTED, localized to statement.logit_binding"
  else
    bad "REJECT but wrong localization: $($PY -c "import json;print(json.load(open('$RUN/transcript_tamper_comS.json'))['rejected'])")"
  fi
fi
mv "$TGT.bak" "$TGT"

# ---------- (e) tamper t* (flip one served token id) [stage 3] ----------
# §3.4 ordering: t* is sha256-pinned INSIDE public.json (run_seed binds it);
# the verifier re-hashes registration fail-closed BEFORE any driver runs. So
# a t* tamper must reject at the REGISTRATION HASH check — never reaching
# statement.logit_binding's rowmax/T-BIND. The assertions below pin that.
TGT="$RUN/registration/tstar.i32.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
import numpy as np
p = sys.argv[1]
t = np.fromfile(p, dtype=np.int32)
t[7] = (t[7] + 1) % 32000   # one token id changed, still in [0, V)
t.tofile(p)
EOF
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_tamper_tstar.json"; then
  bad "tampered t* accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$RUN/transcript_tamper_tstar.json'))
d = t['details'].get('statement.registered_weight_hash', {})
no_drivers = t.get('checked') == [] and 'not run' in t.get('note', '')
at_tstar = 'served_tokens.file' in d.get('reason', '')
print(t['verdict'] == 'REJECT' and d.get('ok') is False and no_drivers and at_tstar)")
  if [ "$HIT" = "True" ]; then
    ok "tampered t* REJECTED at the registration hash (served_tokens.file), no drivers run — §3.4 ordering: registration locus, NOT statement.logit_binding"
  else
    bad "t* tamper: wrong locus / drivers ran"
  fi
fi
mv "$TGT.bak" "$TGT"

# ---------- (f) tamper the registered lm_head commitment hash [stage 3] ----------
cp "$RUN/public.json" "$RUN/public.json.bak"
$PY - "$RUN/public.json" <<'EOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
h = d["registered_weight_commitments"]["lm_head"]
d["registered_weight_commitments"]["lm_head"] = ("0" if h[0] != "0" else "1") + h[1:]
json.dump(d, open(p, "w"), indent=2, sort_keys=True)
EOF
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_tamper_lmcom.json"; then
  bad "tampered lm_head com hash accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$RUN/transcript_tamper_lmcom.json'))
d = t['details'].get('statement.registered_weight_hash', {})
no_drivers = t.get('checked') == [] and 'not run' in t.get('note', '')
print(t['verdict'] == 'REJECT' and d.get('ok') is False and no_drivers and
      'weights.lm_head' in d.get('reason', ''))")
  if [ "$HIT" = "True" ]; then
    ok "tampered lm_head com hash REJECTED at registration, no drivers run"
  else
    bad "lm_head com tamper: wrong locus / drivers ran"
  fi
fi
mv "$RUN/public.json.bak" "$RUN/public.json"

# ---------- (g) restored run re-ACCEPTs ----------
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_restored.json"; then
  N_RESTORED=$($PY -c "import json;print(len(json.load(open('$RUN/transcript_restored.json'))['checked']))")
  if [ "$N_RESTORED" = "$N_EXPECT" ]; then
    ok "restored run re-ACCEPTs with checked = $N_EXPECT"
  else
    bad "restored run accepted but checked=$N_RESTORED != $N_EXPECT"
  fi
else
  bad "restored run re-ACCEPTs"
fi

echo "=== $PASS PASS / $FAIL FAIL ==="
if [ "$FAIL" = "0" ]; then echo "ALL PASS"; exit 0; else echo "SELFTEST FAILED"; exit 1; fi
