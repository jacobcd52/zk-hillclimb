# Task: implement zkob_softmax8 + the zkob_headmerge perm flag (faithful-arch Part C drivers)

Two deliverables from the FINAL design /workspace/projects/zk-hillclimb/STAGE3_FAITHFUL_DESIGN.md
Part C. Follow it EXACTLY.

## Deliverable 1 — zkob_softmax8.cu (NEW driver; §4.3 is normative)
Temperature-8 softmax consuming the exact row-max from zkob_rowmax: inputs z_ (scores
int32 @2^9, chained) and mx (int32 @2^9, chained from rowmax's mx-out; com_mx byte-edge),
exponent table E8(v) = rint(2^16·e^{v/4096}) over the §4.3 domain with the pinned masked-
sentinel row, the same zero-advice rounding-bracket P binding as zkob_softmax (redone
bounds per §4.3: S ≤ 2^26, 2×14-bit limbs — implement the §4.3 arithmetic exactly),
FS schedule per §4.3's schedule block, CLI per §4.3, chain files per §4.3.
Base your structure on /root/zkllm/zkob_softmax.cu (the validated temp-128 driver) —
copy its discipline (constant-claim handling, broadcast bindings, selftest style).
Also write gen_softmax8_table.py per §4.3's pinned generator block.

## Deliverable 2 — zkob_headmerge perm flag (§4.2 is normative; EDITING A VALIDATED DRIVER)
Add the pinned `<perm>` CLI argument (`pi157` | `concat`) to /root/zkllm/zkob_headmerge.cu:
the gather formula switches between the §1.3 π⁻¹ and plain concat; the mode is absorbed
into the transcript exactly as §4.2 pins (absorb label and position per the design).
THE DIFF MUST BE MINIMAL: the gather formula + the one absorb + CLI parsing + selftest
additions. Required selftest extensions (§4.2): both modes honest ACCEPT (toy + real
scale); the CROSS-MODE SPLICE test (a pi157 proof presented to a concat verify and vice
versa must REJECT via transcript divergence); the existing pi157 evil/tamper suite still
passes unchanged.

## Validation
1. zkob_softmax8 selftest: full plan per §4.3 (toy cases, every evil mode on exactly the
   named check — including the new mx-edge cases the design lists, byte tampers on every
   file, real-scale B=NCOL=1024 with a real mx from a zkob_rowmax causal run — invoke
   zkob_rowmax to produce it, chain-file style). ALL PASS line discipline.
2. Rebuilt zkob_headmerge: BOTH-mode selftest ALL PASS incl. the splice test.
3. Because headmerge changed: re-run the FULL selftests of ALL drivers that share its
   machinery patterns is NOT required (header untouched) — but re-run zkob_headmerge's
   own full suite and confirm the header is bit-untouched (sha256 before/after — print it).
4. /root/zkllm/SOFTMAX8_REPORT.md: both deliverables, selftest summaries, real-scale
   timings, the headmerge diff description (line count; what changed), deviations, concerns.

Rules: pinned build commands; FrTensors PLAIN; no header edits (sha256-prove it); no
other driver edits; no GitHub. GPU may be shared with an audit job — retry once on
transient CUDA failures. Honest reporting; STOP on genuine design blockers.
