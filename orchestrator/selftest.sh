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
#
# SUBMISSION faithful-arch-v1 (STAGE3_FAITHFUL_DESIGN §4, Part C — a SECOND
# full walk on its own run id; both chains coexist behind public.json's
# "submission" field, the baseline phases above must stay green):
#  (h) register --submission faithful-arch-v1 -> prove_walk -> verify_walk:
#      ACCEPT, checked = 56 non-waived + 9 covered-waived (6 o_proj +
#      final_norm.rmsnorm + lm_head.matmul + lm_head.rescaling) = 65
#      [arithmetic recounted from the manifest and printed]; skipped = {};
#      check_transcript vs the FULL FROZEN manifest: exit 0 with
#      `required: 56 checked: 65 missing: 0 unknown: 0` + the covered-waived
#      NOTE listing exactly the 9 ids; check_transcript vs the generated
#      manifest_faithful_scope.json (9 waivers removed, harness untouched):
#      `required: 65 checked: 65 missing: 0 unknown: 0`. Selector-tie duty
#      (§2.4: 24 causal + 1 vpad rowmax instances) printed.
#  (i) tamper one byte of proofs/layer0.attn.softmax/rowmax.h05/com_mx.bin ->
#      REJECT localized to layer0.attn.softmax (the rowmax.h05 transcript
#      diverges — com_mx is absorbed before any challenge — AND edge RM2.h05
#      fails byte-equality: two independent detections); restore.
#  (j) restored faithful run re-ACCEPTs with checked = 65.
# STAGE C2 (TRANSPORT_REBUILD): a batched-transport section (k)-(n) follows
# (j): a THIRD full faithful-arch-v1 walk registered with transport=batched —
# every driver in claim mode, ONE zkob_batchopen discharge per sub-batch —
# honest ACCEPT + the same tamper loci + batch-specific forgeries (vfin /
# claims_match / batched-IPA tampers) + the at-scale fold cross-check.
# Run `selftest.sh <run-id> --batched-only` to run only that section.
set -u
PY=/root/int-model-env/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
HARNESS=/workspace/projects/zk-hillclimb/harness
RUN_ID="${1:-selftest3-$(date +%Y%m%d-%H%M%S)}"
RUN=/root/zkorch/$RUN_ID
SECTIONS="${2:-all}"
PASS=0; FAIL=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

if [ "$SECTIONS" = "--batched-only" ]; then
echo "=== stage-3 selftest: BATCHED SECTION ONLY (run id base: $RUN_ID) ==="
else
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

# ============================================================================
# SUBMISSION faithful-arch-v1 (STAGE3 §4 Part C): a SECOND full walk under a
# new registration (new public.json -> new run_seed — that is the point).
# ============================================================================
FRUN_ID="${RUN_ID}-fa"
FRUN=/root/zkorch/$FRUN_ID
echo ""
echo "=== faithful-arch-v1 submission run: $FRUN ==="

# ---------- expected-count arithmetic, recounted from the manifest ----------
read -r F_NONWAIVED F_COVWAIVED F_EXPECT <<<"$($PY - "$HERE" "$HARNESS/manifest_llama68m.json" <<'EOF'
import json, sys
sys.path.insert(0, sys.argv[1])
import common
man = json.load(open(sys.argv[2]))
nonwaived = {o["id"] for o in man["obligations"] if not o["waived"]}
waived = {o["id"] for o in man["obligations"] if o["waived"]}
assert common.skipped_ids() == {}, "faithful-arch must skip nothing"
cov = set(common.covered_ids("faithful-arch-v1"))
missing = sorted(nonwaived - cov)
assert not missing, f"non-waived ids not covered: {missing}"
covwaived = sorted(cov - nonwaived)
assert set(covwaived) <= waived, f"covered ids outside the manifest: {sorted(set(covwaived)-waived)}"
assert covwaived == ["final_norm.rmsnorm",
                     "layer0.attn.o_proj.commitment_opening",
                     "layer0.attn.o_proj.matmul", "layer0.attn.o_proj.rescaling",
                     "layer1.attn.o_proj.commitment_opening",
                     "layer1.attn.o_proj.matmul", "layer1.attn.o_proj.rescaling",
                     "lm_head.matmul", "lm_head.rescaling"], covwaived
print(len(nonwaived), len(covwaived), len(cov))
EOF
)"
echo "manifest arithmetic (faithful-arch-v1): $F_NONWAIVED non-waived + $F_COVWAIVED covered-waived (6 o_proj.* + final_norm.rmsnorm + lm_head.matmul + lm_head.rescaling) = $F_EXPECT expected checked; 0 stage-skipped"

# ---------- (h) faithful honest end-to-end ----------
$PY "$HERE/register.py" --run-id "$FRUN_ID" --submission faithful-arch-v1 || { bad "faithful register"; echo "ABORT"; exit 1; }
ok "faithful register"
$PY "$HERE/prove_walk.py" "$FRUN" || { bad "faithful prove_walk"; echo "ABORT"; exit 1; }
ok "faithful prove_walk"
$PY -c "
import json
t = json.load(open('$FRUN/prove_manifest.json'))['rowmax_selector_ties']
print(f\"measured rowmax selector-tie duty: {t['total_bits']} bits over {len(t['instances'])} instances \"
      f\"({t['rows_with_ties']} tied rows total) — STAGE3 §2.4 duty\")"
if $PY "$HERE/verify_walk.py" "$FRUN" --out "$FRUN/transcript.json"; then
  ok "faithful verify_walk honest ACCEPT"
else
  bad "faithful verify_walk honest ACCEPT"
fi
FN_CHECKED=$($PY -c "import json;print(len(json.load(open('$FRUN/transcript.json'))['checked']))")
FN_SKIPPED=$($PY -c "import json;print(len(json.load(open('$FRUN/transcript.json'))['skipped']))")
if [ "$FN_CHECKED" = "$F_EXPECT" ]; then ok "faithful checked = $F_EXPECT (= $F_NONWAIVED non-waived + $F_COVWAIVED covered-waived)"; else bad "faithful checked=$FN_CHECKED expected=$F_EXPECT"; fi
if [ "$FN_SKIPPED" = "0" ]; then ok "faithful skipped = {}"; else bad "faithful skipped not empty (n=$FN_SKIPPED)"; fi

echo "--- check_transcript vs the FULL FROZEN manifest (must PASS: 56 required, 65 checked + 9-id NOTE) ---"
FFULL_OUT=$($PY "$HARNESS/check_transcript.py" "$HARNESS/manifest_llama68m.json" "$FRUN/transcript.json" 2>&1)
FFULL_RC=$?
echo "$FFULL_OUT"
FNOTE_OK=$($PY - "$FFULL_OUT" <<'EOF'
import sys
out = sys.argv[1]
counts_ok = "required: 56  checked: 65  missing: 0  unknown: 0" in out
note = [l for l in out.splitlines() if l.startswith("NOTE:")]
ids = ("final_norm.rmsnorm", "lm_head.matmul", "lm_head.rescaling",
       "layer0.attn.o_proj.commitment_opening", "layer0.attn.o_proj.matmul",
       "layer0.attn.o_proj.rescaling", "layer1.attn.o_proj.commitment_opening",
       "layer1.attn.o_proj.matmul", "layer1.attn.o_proj.rescaling")
note_ok = (len(note) == 1 and "9 waived obligations now covered" in note[0]
           and all(i in note[0] for i in ids))
print(counts_ok and note_ok)
EOF
)
if [ "$FFULL_RC" = "0" ] && [ "$FNOTE_OK" = "True" ]; then
  ok "check_transcript vs FROZEN manifest: PASS (56 required, 65 checked, 0 missing) + covered-waived NOTE (9 ids)"
else
  bad "faithful check_transcript vs FROZEN manifest (rc=$FFULL_RC, counts+NOTE ok=$FNOTE_OK)"
fi

echo "--- check_transcript vs the generated faithful SCOPE manifest (65/65) ---"
$PY "$HERE/make_faithful_scope_manifest.py" "$HARNESS/manifest_llama68m.json" "$FRUN/manifest_faithful_scope.json"
FSCOPE_OUT=$($PY "$HARNESS/check_transcript.py" "$FRUN/manifest_faithful_scope.json" "$FRUN/transcript.json" 2>&1)
FSCOPE_RC=$?
echo "$FSCOPE_OUT"
if [ "$FSCOPE_RC" = "0" ] && echo "$FSCOPE_OUT" | grep -q "required: 65  checked: 65  missing: 0  unknown: 0"; then
  ok "check_transcript vs faithful scope manifest: required 65, checked 65, missing 0, unknown 0"
else
  bad "faithful scope-manifest check (rc=$FSCOPE_RC)"
fi

# ---------- (i) tamper a chained rowmax com_mx [faithful] ----------
# com_mx is absorbed into the rowmax FS transcript before any challenge AND
# byte-chained to softmax8's com_mx by edge RM2.h05 — two independent
# detections, both under manifest id layer0.attn.softmax.
TGT="$FRUN/proofs/layer0.attn.softmax/rowmax.h05/com_mx.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[24] ^= 0xFF   # pinned tamper-offset convention (com_* at 24)
open(p, "wb").write(bytes(b))
EOF
if $PY "$HERE/verify_walk.py" "$FRUN" --out "$FRUN/transcript_tamper_commx.json"; then
  bad "tampered rowmax com_mx accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$FRUN/transcript_tamper_commx.json'))
layer_rej = [m for m in t['rejected'] if m.startswith('layer')]
d = t['details']['layer0.attn.softmax']['reason']
two_routes = ('rowmax.h05: driver verify REJECT' in d) and ('RM2.05' in d)
print(t['verdict'] == 'REJECT' and layer_rej == ['layer0.attn.softmax'] and two_routes)")
  if [ "$HIT" = "True" ]; then
    ok "tampered rowmax com_mx REJECTED, localized to layer0.attn.softmax (rowmax.h05 transcript divergence + edge RM2.h05 — two independent detections)"
  else
    bad "rowmax com_mx tamper: wrong localization: $($PY -c "import json;print(json.load(open('$FRUN/transcript_tamper_commx.json'))['rejected'])")"
  fi
fi
mv "$TGT.bak" "$TGT"

# ---------- (j) restored faithful run re-ACCEPTs ----------
if $PY "$HERE/verify_walk.py" "$FRUN" --out "$FRUN/transcript_restored.json"; then
  FN_RESTORED=$($PY -c "import json;print(len(json.load(open('$FRUN/transcript_restored.json'))['checked']))")
  if [ "$FN_RESTORED" = "$F_EXPECT" ]; then
    ok "restored faithful run re-ACCEPTs with checked = $F_EXPECT"
  else
    bad "restored faithful run accepted but checked=$FN_RESTORED != $F_EXPECT"
  fi
else
  bad "restored faithful run re-ACCEPTs"
fi

fi   # end of SECTIONS != --batched-only

# ============================================================================
# BATCHED TRANSPORT (TRANSPORT_REBUILD_DESIGN Stage C2, gates T4/T5): a full
# faithful-arch-v1 walk registered with transport=batched. Every driver runs
# in claim mode (--claims), all claims discharge through zkob_batchopen
# sub-batches; per-driver verdicts are ACCEPT-conditional, the orchestrator
# verdict gates on opening_batch (F12).
# ============================================================================
BRUN_ID="${RUN_ID}-fab"
BRUN=/root/zkorch/$BRUN_ID
ZB=/root/zkllm/zkob_batchopen
echo ""
echo "=== batched-transport faithful-arch-v1 run: $BRUN ==="

# ---------- (k) batched honest end-to-end ----------
$PY "$HERE/register.py" --run-id "$BRUN_ID" --submission faithful-arch-v1 --transport batched \
  || { bad "batched register"; echo "ABORT"; exit 1; }
ok "batched register (transport pinned in public.json -> run_seed)"
$PY "$HERE/prove_walk.py" "$BRUN" || { bad "batched prove_walk"; echo "ABORT"; exit 1; }
ok "batched prove_walk (incl. per-sub-batch zkob_batchopen prove + witness cleanup)"
$PY -c "
import json
m = json.load(open('$BRUN/prove_manifest.json'))
obs = m['opening_batch']
print(f\"opening_batch prove: {obs['n_batches']} sub-batches, \"
      f\"{sum(b['claims'] for b in obs['batches'])} claims, \"
      f\"{sum(b['elements'] for b in obs['batches'])/1e6:.0f}M elements, \"
      f\"{sum(b['witness_bytes_freed'] for b in obs['batches'])/2**30:.1f} GiB witness freed\")
print(f\"prove totals: {m['totals']}\")"
if $PY "$HERE/verify_walk.py" "$BRUN" --out "$BRUN/transcript.json"; then
  ok "batched verify_walk honest ACCEPT"
else
  bad "batched verify_walk honest ACCEPT"
fi
BN_CHECKED=$($PY -c "import json;print(len(json.load(open('$BRUN/transcript.json'))['checked']))")
BOB_OK=$($PY -c "import json;t=json.load(open('$BRUN/transcript.json'));print(t['opening_batch']['ok'] and t['transport']=='batched')")
if [ "$BN_CHECKED" = "65" ]; then ok "batched checked = 65"; else bad "batched checked=$BN_CHECKED expected=65"; fi
if [ "$BOB_OK" = "True" ]; then ok "opening_batch ACCEPT (all sub-batches; registered-comref discharge pin)"; else bad "opening_batch not ok in honest transcript"; fi
$PY -c "
import json
t = json.load(open('$BRUN/transcript.json'))
print(f\"verify wall: {t['timing']['total_verify_wall_s']} s; \"
      f\"opening_batch: {t['opening_batch']['n_batches']} sub-batches, \"
      f\"{t['opening_batch']['claims']} claims\")"

echo "--- check_transcript vs the FULL FROZEN manifest (batched run) ---"
BFULL_OUT=$($PY "$HARNESS/check_transcript.py" "$HARNESS/manifest_llama68m.json" "$BRUN/transcript.json" 2>&1)
BFULL_RC=$?
echo "$BFULL_OUT"
if [ "$BFULL_RC" = "0" ] && echo "$BFULL_OUT" | grep -q "required: 56  checked: 65  missing: 0  unknown: 0"; then
  ok "batched check_transcript vs FROZEN manifest: PASS (56 required, 65 checked)"
else
  bad "batched check_transcript vs FROZEN manifest (rc=$BFULL_RC)"
fi
$PY "$HERE/make_faithful_scope_manifest.py" "$HARNESS/manifest_llama68m.json" "$BRUN/manifest_faithful_scope.json"
BSCOPE_OUT=$($PY "$HARNESS/check_transcript.py" "$BRUN/manifest_faithful_scope.json" "$BRUN/transcript.json" 2>&1)
if [ $? = 0 ] && echo "$BSCOPE_OUT" | grep -q "required: 65  checked: 65  missing: 0  unknown: 0"; then
  ok "batched check_transcript vs faithful scope manifest: 65/65"
else
  bad "batched scope-manifest check"
fi

# ---------- (l1) tamper a slice commitment [batched analog of (b)] ----------
TGT="$BRUN/proofs/layer0.attn.scores_matmul/slice/com_KhT05.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[24] ^= 0xFF
open(p, "wb").write(bytes(b))
EOF
if $PY "$HERE/verify_walk.py" "$BRUN" --out "$BRUN/transcript_tamper_slice.json"; then
  bad "batched: tampered com_KhT05 accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$BRUN/transcript_tamper_slice.json'))
layer_rej = [m for m in t['rejected'] if m.startswith('layer')]
print(t['verdict'] == 'REJECT' and layer_rej == ['layer0.attn.scores_matmul'])")
  if [ "$HIT" = "True" ]; then
    ok "batched: tampered slice commitment REJECTED, localized to layer0.attn.scores_matmul (transcript divergence; sub-batch claims_match fires too)"
  else
    bad "batched: slice tamper wrong localization: $($PY -c "import json;print(json.load(open('$BRUN/transcript_tamper_slice.json'))['rejected'])")"
  fi
fi
mv "$TGT.bak" "$TGT"

# ---------- (l2) tamper t* [batched analog of (e): registration locus] ------
TGT="$BRUN/registration/tstar.i32.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
import numpy as np
p = sys.argv[1]
t = np.fromfile(p, dtype=np.int32)
t[7] = (t[7] + 1) % 32000
t.tofile(p)
EOF
if $PY "$HERE/verify_walk.py" "$BRUN" --out "$BRUN/transcript_tamper_tstar.json"; then
  bad "batched: tampered t* accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$BRUN/transcript_tamper_tstar.json'))
d = t['details'].get('statement.registered_weight_hash', {})
no_drivers = t.get('checked') == [] and 'not run' in t.get('note', '')
print(t['verdict'] == 'REJECT' and d.get('ok') is False and no_drivers and
      'served_tokens.file' in d.get('reason', ''))")
  if [ "$HIT" = "True" ]; then
    ok "batched: tampered t* REJECTED at the registration hash, no drivers run"
  else
    bad "batched: t* tamper wrong locus / drivers ran"
  fi
fi
mv "$TGT.bak" "$TGT"

# ---------- (l3) tamper a chained rowmax com_mx [batched analog of (i)] -----
TGT="$BRUN/proofs/layer0.attn.softmax/rowmax.h05/com_mx.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[24] ^= 0xFF
open(p, "wb").write(bytes(b))
EOF
if $PY "$HERE/verify_walk.py" "$BRUN" --out "$BRUN/transcript_tamper_commx.json"; then
  bad "batched: tampered rowmax com_mx accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$BRUN/transcript_tamper_commx.json'))
layer_rej = [m for m in t['rejected'] if m.startswith('layer')]
d = t['details']['layer0.attn.softmax']['reason']
two_routes = ('rowmax.h05: driver verify REJECT' in d) and ('RM2.05' in d)
print(t['verdict'] == 'REJECT' and layer_rej == ['layer0.attn.softmax'] and two_routes)")
  if [ "$HIT" = "True" ]; then
    ok "batched: tampered rowmax com_mx REJECTED, localized (driver transcript divergence + edge RM2.h05 — both detections live in batched mode)"
  else
    bad "batched: com_mx tamper wrong localization"
  fi
fi
mv "$TGT.bak" "$TGT"

# ---------- (m1) BATCH-SPECIFIC: tamper the opening_batch terminal ----------
# All 230 driver verifies stay ACCEPT-conditional; ONLY the batch dies (F12
# gating: conditional verdicts + opening_batch REJECT => overall REJECT).
TGT="$BRUN/proofs/opening_batch/b0/batch_vfin.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[-32] ^= 0x01   # last tensor's v'_j, low byte
open(p, "wb").write(bytes(b))
EOF
if $PY "$HERE/verify_walk.py" "$BRUN" --out "$BRUN/transcript_tamper_vfin.json"; then
  bad "batched: tampered batch_vfin accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$BRUN/transcript_tamper_vfin.json'))
ob = t['opening_batch']
b0 = [b for b in ob['batches'] if b['batch'] == 0][0]
no_layer_rej = not [m for m in t['rejected'] if m.startswith(('layer', 'final', 'lm_head', 'statement.logit'))]
print(t['verdict'] == 'REJECT' and ob['ok'] is False and not b0['ok']
      and 'terminal' in (b0['locus'] or '') and no_layer_rej)")
  if [ "$HIT" = "True" ]; then
    ok "batched: batch_vfin tamper -> opening_batch.terminal REJECT while every driver stays conditional-ACCEPT (the F12 gating case)"
  else
    bad "batched: vfin tamper wrong locus/gating: $($PY -c "import json;t=json.load(open('$BRUN/transcript_tamper_vfin.json'));print(t['opening_batch'])")"
  fi
fi
mv "$TGT.bak" "$TGT"

# ---------- (m2) targeted batch tampers against the persisted vacc ----------
RUN_SEED=$($PY -c "import hashlib;print(hashlib.sha256(open('$BRUN/public.json','rb').read()).hexdigest())")
GENSPEC="64=registration/gen64.bin 1024=registration/gen1024.bin 4096=registration/gen4096.bin 32768=registration/gen32768.bin"
bo_verify() {  # bo_verify <k> -> exit code of the direct batch verify
  (cd "$BRUN" && ZKOB_REQUIRE_RELATIVE_COMREF=1 $ZB verify \
     proofs/opening_batch/b$1 vacc/b$1 "$RUN_SEED:b$1" registration/q.bin $GENSPEC)
}
TGT="$BRUN/proofs/opening_batch/b0/claims.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[-1] ^= 0x01   # last claim's eval, last byte
open(p, "wb").write(bytes(b))
EOF
OUT=$(bo_verify 0 2>&1); RC=$?
if [ "$RC" != "0" ] && echo "$OUT" | grep -q "REJECT\[opening_batch.claims_match\]"; then
  ok "batched: prover claims.bin eval tamper -> opening_batch.claims_match"
else
  bad "batched: claims.bin tamper wrong locus (rc=$RC): $(echo "$OUT" | tail -1)"
fi
mv "$TGT.bak" "$TGT"
TGT="$BRUN/proofs/opening_batch/b0/ipa_batch_1024.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[-32] ^= 0x01   # a_final
open(p, "wb").write(bytes(b))
EOF
OUT=$(bo_verify 0 2>&1); RC=$?
if [ "$RC" != "0" ] && echo "$OUT" | grep -q "REJECT\[opening_batch.ipa1024\]"; then
  ok "batched: batched-IPA a_final tamper -> opening_batch.ipa1024"
else
  bad "batched: ipa tamper wrong locus (rc=$RC): $(echo "$OUT" | tail -1)"
fi
mv "$TGT.bak" "$TGT"

# ---------- (m3) at-scale fold cross-check (convention pin, all batches) ----
XOK=1
NB=$($PY -c "import json;print(json.load(open('$BRUN/transcript.json'))['opening_batch']['n_batches'])")
for k in $(seq 0 $((NB-1))); do
  OUT=$(cd "$BRUN" && ZKOB_REQUIRE_RELATIVE_COMREF=1 ZKOB_FOLD_CROSSCHECK=1 $ZB verify \
     proofs/opening_batch/b$k vacc/b$k "$RUN_SEED:b$k" registration/q.bin $GENSPEC 2>&1)
  if [ $? != 0 ] || ! echo "$OUT" | grep -q "opening_batch ACCEPT"; then
    XOK=0; echo "  fold cross-check FAILED on b$k: $(echo "$OUT" | tail -1)"
  fi
done
if [ "$XOK" = "1" ]; then
  ok "batched: ZKOB_FOLD_CROSSCHECK=1 re-verify of all $NB sub-batches ACCEPTs (batched fold == per-tensor fold_chain element-exact at full-walk scale)"
else
  bad "batched: fold cross-check"
fi

# ---------- (n) restored batched run re-ACCEPTs ----------
if $PY "$HERE/verify_walk.py" "$BRUN" --out "$BRUN/transcript_restored.json"; then
  BN_RESTORED=$($PY -c "import json;print(len(json.load(open('$BRUN/transcript_restored.json'))['checked']))")
  BOB_RESTORED=$($PY -c "import json;print(json.load(open('$BRUN/transcript_restored.json'))['opening_batch']['ok'])")
  if [ "$BN_RESTORED" = "65" ] && [ "$BOB_RESTORED" = "True" ]; then
    ok "restored batched run re-ACCEPTs with checked = 65 + opening_batch ACCEPT"
  else
    bad "restored batched run: checked=$BN_RESTORED opening_batch=$BOB_RESTORED"
  fi
else
  bad "restored batched run re-ACCEPTs"
fi

echo "=== $PASS PASS / $FAIL FAIL ==="
if [ "$FAIL" = "0" ]; then echo "ALL PASS"; exit 0; else echo "SELFTEST FAILED"; exit 1; fi
