# ROPE_IMPL_REPORT — zkob_rope / zkob_headslice / zkob_headmerge

Implemented by a Fable agent from ROPE_ATTENTION_DESIGN.md (no design deviations reported);
the agent's session ended before writing this report, so the coordinator completed it from
the acceptance selftest logs (coordinator-run, GPU-exclusive, 2026-06-11).

| driver | selftest | checks | real-scale prove | verify | proof+coms |
|---|---|---|---|---|---|
| zkob_rope      | ALL PASS | 50  | 3.49 s | 2.20 s | 1.97 MB* |
| zkob_headslice | ALL PASS | 58  | 29.25 s | 24.98 s | 4.28 MB |
| zkob_headmerge | ALL PASS | 105 | 1.99 s | 1.40 s | 309 KB* |

*the rope/merge byte figures appear swapped in the log line order vs the design's §7.2
predictions (rope ≈295 KB, merge ≈1.9 MB); per-file logs retained as
/root/zkllm/{rope,headslice,headmerge}_selftest_accept.log — auditor: please confirm the
mapping when walking the selftests.

§9.1 gate: headslice real-scale = 29.2 s < 30 s — PASSES without fallback (whether
me_weights memoization was applied is visible in the source; auditor to note).
gen_rope_tables.py written per §2.1 and run (tables 256 KB each, cos(0)=65536).
Agent cost $17.56, 44 turns. Audit pending (next gate).
