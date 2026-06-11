"""Derive the faithful-arch-v1 SCOPE manifest from the FROZEN harness manifest
(STAGE3_FAITHFUL_DESIGN §4.4, the make_stage1_manifest.py precedent): the 9
waived ids the submission genuinely covers (six layer{0,1}.attn.o_proj.* +
final_norm.rmsnorm + lm_head.matmul + lm_head.rescaling) have their waivers
REMOVED, so check_transcript against it must show
`required: 65 checked: 65 missing: 0 unknown: 0`. The harness file is never
touched; only embedding.lookup remains waived, with the unchanged reason.

Usage: python make_faithful_scope_manifest.py <harness_manifest.json> <out.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

man = json.load(open(sys.argv[1]))
nonwaived = {o["id"] for o in man["obligations"] if not o["waived"]}
unwaive = sorted(set(C.covered_ids("faithful-arch-v1")) - nonwaived
                 - {"statement.registered_weight_hash", "statement.prompt_binding"})
hit = []
for o in man["obligations"]:
    if o["id"] in unwaive:
        o["waived"] = False
        hit.append(o["id"])
assert sorted(hit) == unwaive, (hit, unwaive)
man["note"] = ("FAITHFUL-ARCH-V1 SCOPE MANIFEST derived from the frozen harness "
               "manifest; the 9 waived ids the submission covers are un-waived: "
               + ", ".join(unwaive) + ". NOT a harness file.")
json.dump(man, open(sys.argv[2], "w"), indent=2)
print(f"wrote {sys.argv[2]} ({len(unwaive)} waivers removed)")
