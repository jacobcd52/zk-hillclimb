# Task: wire and run submission faithful-arch-v1 (STAGE3_FAITHFUL_DESIGN Part C)

Stage 3 is DONE and ACCEPTED (orchestrator selftest stage3-official1: ALL PASS, 59 ids,
logit binding live). All Part C drivers exist, are audited and hardened: zkob_rowmax
(causal mode), zkob_softmax8, zkob_headmerge --concat, and o_proj uses the existing
fc+rescale. Your job: the Part C orchestrator wiring + the full submission run.

Normative: STAGE3_FAITHFUL_DESIGN.md §4 (esp. §4.0-4.4 composition/wiring and the §4.4
re-registration mechanics) + §2.7 chain edges (RM1/RM2 per head). Also read
ORCHESTRATOR_DESIGN.md + report (stages 1-3), PHASE0_NOTES §19-21 (the new drivers'
registers incl. pinned orchestrator obligations and the tie-count duty).

Work:
1. register.py: a `--submission faithful-arch-v1` mode per §4.4 — registers o_proj
   weights (export per the standard convention; note the pipeline never dumps o_proj —
   re-export guard only), the softmax8 exp table (gen_softmax8_table.py), and writes the
   revised public.json (new statement; run_seed changes — that is the point).
2. prove_walk.py / common.py: a submission flag switching the attention chain per §4.3:
   rowmax causal per head (chained from rescale10, mx chains into softmax8 — edges
   RM1/RM2), zkob_softmax8 instead of zkob_softmax, headmerge concat mode, o_proj
   fc+rescale between headmerge and attn_skip (edges per §4.1), everything else
   unchanged. Witness semantics: §4.3's integer chain (the orchestrator computes mx
   per §2.1 exactly — but PREFER consuming zkob_rowmax's mx-out chain file, the pinned
   authority).
3. verify_walk.py: the corresponding verifies/edges under the submission flag;
   manifest accounting per §4.0 (o_proj ids become covered; scores_matmul composes
   rowmax; softmax id composes softmax8).
4. selftest.sh: a faithful-arch phase — full run ACCEPT with the §4.0 accounting
   (print the id arithmetic), one tamper (rowmax com_mx) REJECTed at the right place,
   restore + re-ACCEPT. Keep ALL stage-2/3 phases green (the baseline path must still
   work — both chains coexist behind the flag).
5. Report: append to ORCHESTRATOR_REPORT.md — per-obligation timings for the new
   instances vs §5.1 predictions, TOTAL prove/verify wall + bytes for the submission,
   measured selector-tie counts (causal instances now — report the §2.4 duty), honest
   caveats. DO NOT run the DiFR measurement (coordinator runs it separately after
   acceptance).
No git commits; no GitHub; never modify int-model-approximation. GPU free; full
selftest likely ~90-120 min (two full walks). The 900 s timeout should hold per §5.1;
document any near-misses.
