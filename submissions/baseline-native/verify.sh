#!/usr/bin/env bash
# Verifier entrypoint: read ONLY the serialized proof + public statement, emit a
# transcript JSON listing checked obligation ids and ACCEPT/REJECT.
# Usage: verify.sh <proof_dir> <transcript_out.json>
set -euo pipefail
ZK=/root/zkllm
PROOF="${1:?proof_dir}"
TR_OUT="${2:-$PROOF/transcript.json}"
SEED=$(cat "$PROOF/seed.hex")

cd "$ZK"
RAW=$(./zkverify "$PROOF" "$SEED" \
        layer0.attn.q_proj.matmul \
        layer0.attn.q_proj.commitment_opening \
        layer0.attn.q_proj.rescaling || true)
echo "$RAW"

# turn the driver's "<id> OK/FAIL" lines into the harness transcript JSON
/root/int-model-env/bin/python - "$TR_OUT" "$PROOF" <<PYEOF
import sys, json, hashlib, os
raw = """$RAW"""
proof_dir = sys.argv[2]
checked=[]; details={}; verdict="ACCEPT"
for line in raw.splitlines():
    parts=line.split()
    if len(parts)>=2 and parts[1]=="OK":
        checked.append(parts[0]); details[parts[0]]={"ok":True}
    elif len(parts)>=2 and parts[1]=="FAIL":
        details[parts[0]]={"ok":False,"reason":" ".join(parts[2:])}
        verdict="REJECT"
    elif parts and parts[0]=="VERDICT":
        if parts[1]=="REJECT": verdict="REJECT"

# Additional witness-free, REGISTERED-hash binding (forgeries B1/B2/B4):
# the committed point shipped in the proof MUST hash to the registered commitment
# recorded in public.json. The verifier checks against the REGISTERED hash, not
# against whatever the proof claims internally.
pub = json.load(open(os.path.join(proof_dir, "public.json")))
reg = pub.get("registered_weight_commitments", {}).get("layer0.attn.q_proj")
com_id = "layer0.attn.q_proj.commitment_opening"
com_path = os.path.join(proof_dir, com_id, "commitment.bin")
if reg is not None and os.path.exists(com_path):
    got = hashlib.sha256(open(com_path, "rb").read()).hexdigest()
    if got != reg:
        details.setdefault(com_id, {})
        details[com_id] = {"ok": False, "reason": "commitment_ne_registered_hash"}
        if com_id in checked: checked.remove(com_id)
        verdict = "REJECT"
    else:
        details.setdefault(com_id, {}).setdefault("registered_hash_ok", True)
else:
    details.setdefault(com_id, {})
    details[com_id] = {"ok": False, "reason": "missing_registered_commitment_or_file"}
    if com_id in checked: checked.remove(com_id)
    verdict = "REJECT"

json.dump({"verdict":verdict,"checked":checked,"details":details}, open(sys.argv[1],"w"), indent=2)
print("wrote", sys.argv[1], "verdict", verdict)
PYEOF
