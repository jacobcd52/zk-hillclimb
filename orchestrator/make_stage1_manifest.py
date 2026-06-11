"""Derive a stage-scope manifest from the FROZEN harness manifest: ids the
current stage skips (common.skipped_ids() — for stage 2 ONLY
lm_head.commitment_opening + statement.logit_binding; embedding has no
non-waived manifest id, embedding.lookup is waived in the frozen manifest
itself) are marked waived with an explicit reason. The harness file is never
touched. Used to demonstrate exact format/coverage compliance of the declared
stage scope; the full-manifest gap is reported separately and loudly.

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
