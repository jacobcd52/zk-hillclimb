"""Derive a stage-scope manifest from the FROZEN harness manifest: ids the
current stage skips (common.skipped_ids()) are marked waived with an explicit
reason. The harness file is never touched. STAGE 3 NOTE: skipped_ids() is now
EMPTY (the manifest is closed), so this script emits a plain copy and the
selftest no longer uses it — check_transcript runs against the frozen manifest
directly and must PASS. Kept for stage-1/2 reproducibility.

Usage: python make_stage1_manifest.py <harness_manifest.json> <out.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

man = json.load(open(sys.argv[1]))
sk = C.skipped_ids()
for o in man["obligations"]:
    if o["id"] in sk and not o["waived"]:
        o["waived"] = sk[o["id"]]
man["note"] = ("STAGE-SCOPE MANIFEST derived from the frozen harness manifest; "
               "stage-skipped ids marked waived with reasons. NOT a harness file.")
json.dump(man, open(sys.argv[2], "w"), indent=2)
print(f"wrote {sys.argv[2]}")
