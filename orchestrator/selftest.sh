#!/bin/bash
# Orchestrator selftest, STAGE 2 (full forward pass: MLP + rmsnorm + skips +
# complete attention chain; only lm_head.commitment_opening +
# statement.logit_binding waived):
#  (a) register -> prove_walk -> verify_walk on real llama-68m data: ACCEPT,
#      checked = (non-waived in FROZEN manifest) - (stage-2 waived) [arithmetic
#      recounted from the manifest and printed], skipped = the 2 waived only;
#      check_transcript vs the stage-2 scope manifest passes; vs the FULL
#      frozen manifest exactly the 2 waived ids are missing (printed, not hidden).
#  (b) tamper a slice commitment (proofs/.../slice/com_KhT05.bin) ->
#      verify_walk REJECTs, localized to layer0.attn.scores_matmul.
#  (c) tamper the registered rope-cos table hash in public.json -> REJECT at
#      the registration check, NO drivers run; restore -> full re-ACCEPT.
set -u
PY=/root/int-model-env/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
HARNESS=/workspace/projects/zk-hillclimb/harness
RUN_ID="${1:-selftest2-$(date +%Y%m%d-%H%M%S)}"
RUN=/root/zkorch/$RUN_ID
PASS=0; FAIL=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

echo "=== stage-2 selftest run: $RUN ==="

# ---------- expected-count arithmetic, recounted from the manifest ----------
read -r N_NONWAIVED N_STAGE2_WAIVED N_EXPECT <<<"$($PY - "$HERE" "$HARNESS/manifest_llama68m.json" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1])
import common
man = json.load(open(sys.argv[2]))
nonwaived = [o["id"] for o in man["obligations"] if not o["waived"]]
sk = common.skipped_ids()
missing_from_manifest = [k for k in sk if k not in nonwaived]
assert not missing_from_manifest, f"waived ids not in manifest: {missing_from_manifest}"
n_cov = len(common.covered_ids())
assert n_cov == len(nonwaived) - len(sk), (n_cov, len(nonwaived), len(sk))
print(len(nonwaived), len(sk), n_cov)
EOF
)"
echo "manifest arithmetic: $N_NONWAIVED non-waived - $N_STAGE2_WAIVED stage-2 waived (lm_head.commitment_opening, statement.logit_binding) = $N_EXPECT expected checked"

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
N_SKIPPED=$($PY -c "import json;print(len(json.load(open('$RUN/transcript.json'))['skipped']))")
if [ "$N_CHECKED" = "$N_EXPECT" ]; then ok "checked = $N_EXPECT (= $N_NONWAIVED - $N_STAGE2_WAIVED)"; else bad "checked=$N_CHECKED expected=$N_EXPECT"; fi
SKIP_OK=$($PY -c "import json;s=json.load(open('$RUN/transcript.json'))['skipped'];print(sorted(s)==['lm_head.commitment_opening','statement.logit_binding'])")
if [ "$SKIP_OK" = "True" ]; then ok "skipped = exactly the 2 waived ids"; else bad "skipped list wrong (n=$N_SKIPPED)"; fi

$PY "$HERE/make_stage1_manifest.py" "$HARNESS/manifest_llama68m.json" "$RUN/manifest_stage2.json"
if $PY "$HARNESS/check_transcript.py" "$RUN/manifest_stage2.json" "$RUN/transcript.json"; then
  ok "check_transcript vs stage-2 scope manifest"
else
  bad "check_transcript vs stage-2 scope manifest"
fi
echo "--- check_transcript vs FULL frozen manifest (stage-2 gap, reported honestly) ---"
FULL_OUT=$($PY "$HARNESS/check_transcript.py" "$HARNESS/manifest_llama68m.json" "$RUN/transcript.json" 2>&1)
FULL_RC=$?
echo "$FULL_OUT"
MISS_OK=$($PY - "$FULL_OUT" <<'EOF'
import sys
out = sys.argv[1]
miss = sorted(l.split(": ", 1)[1] for l in out.splitlines() if l.startswith("MISSING OBLIGATION:"))
print(miss == ["lm_head.commitment_opening", "statement.logit_binding"])
EOF
)
if [ "$FULL_RC" != "0" ] && [ "$MISS_OK" = "True" ]; then
  ok "full-manifest gap = exactly the 2 waived ids (lm_head opening + logit binding)"
else
  bad "full-manifest gap wrong (rc=$FULL_RC, exact-2-missing=$MISS_OK)"
fi

# ---------- (b) tamper a slice commitment ----------
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

# ---------- (c) tamper the registered rope-cos table hash in public.json ----------
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

# ---------- restored run re-ACCEPTs ----------
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_restored.json"; then
  ok "restored run re-ACCEPTs"
else
  bad "restored run re-ACCEPTs"
fi

echo "=== $PASS PASS / $FAIL FAIL ==="
if [ "$FAIL" = "0" ]; then echo "ALL PASS"; exit 0; else echo "SELFTEST FAILED"; exit 1; fi
