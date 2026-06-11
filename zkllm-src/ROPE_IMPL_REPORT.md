# ROPE_IMPL_REPORT — zkob_rope / zkob_headslice / zkob_headmerge

Implemented by a Fable agent from ROPE_ATTENTION_DESIGN.md (no design deviations reported);
the agent's session ended before writing this report, so the coordinator completed it from
the acceptance selftest logs (coordinator-run, GPU-exclusive, 2026-06-11). Measured-numbers
mapping confirmed by the independent audit (ROPE_REVIEW.md §9/MINOR-1: the originally
drafted table had the rope and headmerge rows swapped; corrected below from the retained
logs — rope_selftest_accept.log:119, headslice…:226, headmerge…:134 — and re-derived from
the design's §7.2 file inventories: rope = 2 × 144 KB gen-1024 commitments dominate
≈ §7.2's ≈295 KB; headmerge = 13 × 144 KB commitments = 1.87 MB ≈ §7.2's ≈1.9 MB).

| driver | selftest | checks | real-scale prove | verify | proof+coms |
|---|---|---|---|---|---|
| zkob_rope      | ALL PASS | 50  | 1.99 s  | 1.40 s  | 309,040 B (309 KB) |
| zkob_headslice | ALL PASS | 105 | 29.25 s | 24.98 s | 4,275,660 B (4.28 MB) |
| zkob_headmerge | ALL PASS | 58  | 3.49 s  | 2.20 s  | 1,966,884 B (1.97 MB) |

(The audit's independent re-runs reproduced these within noise: 1.99/1.34, 29.00/24.37,
3.48/2.19.)

§9.1 gate: headslice real-scale = 29.25 s prove / 24.98 s verify < 30 s — PASSES without
the sanctioned fallback; **no me_weights memoization is applied** (audit §10 confirmed
from source). **The prove-side margin is < 1 s** (audit MINOR-5): any regression (slower
GPU, driver change, contention) trips the gate; the sanctioned host-only me_weights
memoization remains available (the dominant cost is 72 IPA verifies each rebuilding
me_weights; the 36 full-side openings share u_row = v_t). Performance only, not soundness.
gen_rope_tables.py written per §2.1 and run (tables 256 KB each, cos(0)=65536); the audit
regenerated both tables independently and they match bit-exactly.
Agent cost $17.56, 44 turns. Audit complete: ROPE_REVIEW.md, **VERDICT: SOUND** (all three
drivers; no critical or major findings).
