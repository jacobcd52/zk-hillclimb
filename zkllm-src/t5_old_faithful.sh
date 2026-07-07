#!/bin/bash
# Stage C2 / T5 "before" leg: a fresh OLD-transport (inline-IPA) faithful-arch-v1
# walk with the CURRENT binaries + edited orchestrator (inline path), measured
# on the same box/day as the batched run; plus the selftest (i)/(j) tamper +
# restore phases in old mode (evidence the inline path of the refactored
# verify_walk still rejects/accepts exactly as before).
set -u
PY=/root/int-model-env/bin/python
HERE=/workspace/projects/zk-hillclimb/orchestrator
RUN_ID="${1:-c2old}"
RUN=/root/zkorch/$RUN_ID
PASS=0; FAIL=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

echo "=== T5 OLD-transport faithful run: $RUN ==="
$PY "$HERE/register.py" --run-id "$RUN_ID" --submission faithful-arch-v1 || { bad register; exit 1; }
ok "old register (transport absent from public.json = inline)"
$PY "$HERE/prove_walk.py" "$RUN" || { bad prove; exit 1; }
ok "old prove_walk"
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript.json"; then
  ok "old verify_walk honest ACCEPT"
else
  bad "old verify_walk honest ACCEPT"
fi
$PY -c "
import json
t = json.load(open('$RUN/transcript.json'))
m = json.load(open('$RUN/prove_manifest.json'))
print(f\"OLD checked={len(t['checked'])} verify_wall={t['timing']['total_verify_wall_s']}s \"
      f\"prove_wall={m['totals']['prove_wall_s']}s proof_bytes={m['totals']['proof_bytes']}\")"
N=$($PY -c "import json;print(len(json.load(open('$RUN/transcript.json'))['checked']))")
[ "$N" = "65" ] && ok "old checked = 65" || bad "old checked=$N"

# (i)-analog: tamper rowmax com_mx -> REJECT with both detections; restore -> ACCEPT
TGT="$RUN/proofs/layer0.attn.softmax/rowmax.h05/com_mx.bin"
cp "$TGT" "$TGT.bak"
$PY - "$TGT" <<'EOF'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[24] ^= 0xFF
open(p, "wb").write(bytes(b))
EOF
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_tamper_commx.json"; then
  bad "old: tampered com_mx accepted"
else
  HIT=$($PY -c "
import json
t = json.load(open('$RUN/transcript_tamper_commx.json'))
layer_rej = [m for m in t['rejected'] if m.startswith('layer')]
d = t['details']['layer0.attn.softmax']['reason']
print(t['verdict'] == 'REJECT' and layer_rej == ['layer0.attn.softmax']
      and 'rowmax.h05: driver verify REJECT' in d and 'RM2.05' in d)")
  [ "$HIT" = "True" ] && ok "old: com_mx tamper REJECTED, two detections (driver + edge RM2.h05)" \
                      || bad "old: com_mx tamper wrong localization"
fi
mv "$TGT.bak" "$TGT"
if $PY "$HERE/verify_walk.py" "$RUN" --out "$RUN/transcript_restored.json"; then
  N=$($PY -c "import json;print(len(json.load(open('$RUN/transcript_restored.json'))['checked']))")
  [ "$N" = "65" ] && ok "old: restored re-ACCEPT 65" || bad "old: restored checked=$N"
else
  bad "old: restored re-ACCEPT"
fi
echo "=== T5-OLD: $PASS PASS / $FAIL FAIL ==="
[ "$FAIL" = "0" ] && echo ALL PASS || exit 1
