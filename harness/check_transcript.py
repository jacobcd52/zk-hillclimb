"""Compare a verifier transcript against the obligation manifest.

A transcript is JSON: {"checked": ["layer0.attn.q_proj.matmul", ...],
"verdict": "ACCEPT"|"REJECT", ...}. Pass criteria:
  - every non-waived obligation id appears in `checked`
  - no unknown ids (an id not in the manifest suggests the verifier is
    checking against a DIFFERENT architecture -> forgery D1 territory)
Exit 0 = coverage OK; exit 1 = gap (printed).

Usage: python check_transcript.py <manifest.json> <transcript.json>
"""
import json
import sys


def check(manifest_path, transcript_path):
    man = json.load(open(manifest_path))
    tr = json.load(open(transcript_path))
    checked = set(tr["checked"])
    required = {o["id"] for o in man["obligations"] if not o["waived"]}
    waived = {o["id"] for o in man["obligations"] if o["waived"]}
    known = required | waived
    missing = sorted(required - checked)
    unknown = sorted(checked - known)
    covered_waived = sorted(checked & waived)
    ok = not missing and not unknown
    print(f"required: {len(required)}  checked: {len(checked)}  "
          f"missing: {len(missing)}  unknown: {len(unknown)}")
    if covered_waived:
        print(f"NOTE: {len(covered_waived)} waived obligations now covered "
              f"(candidates for waiver removal): {covered_waived}")
    for m in missing:
        print(f"MISSING OBLIGATION: {m}")
    for u in unknown:
        print(f"UNKNOWN ID (wrong architecture?): {u}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1], sys.argv[2]) else 1)
