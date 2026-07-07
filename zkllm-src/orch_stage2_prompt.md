# Task: orchestrator STAGE 2 — wire the attention chain, close every open boundary, full-manifest run

Stage 1 of the orchestrator (/workspace/projects/zk-hillclimb/orchestrator/, hardened and
audited) covers the MLP subgraph: 30 manifest ids checked, 26 SKIPPED. Your job: extend it
to the FULL frozen manifest minus only the lm_head/logit ids — i.e. wire the complete
attention chain so both layers verify end-to-end and edge S1's `com_attn_out` OPEN BOUNDARY
is closed.

## Read first
1. orchestrator/ORCHESTRATOR_DESIGN.md + ORCHESTRATOR_REPORT.md (stage 1, incl. hardening
   round) — extend, do not rewrite; the §5 verifier-independence invariants and the
   hardening fixes (timeouts, hash-pinning assertion, hygiene walk) must keep holding.
2. /workspace/projects/zk-hillclimb/ROPE_ATTENTION_DESIGN.md — §4.0 (manifest composition),
   §7.3 (chain wiring per head, verbatim), §7.4 (the FULL edge table A1..A15 — every edge
   becomes a verify_walk check), §2.1 (rope tables), §9.5 (transcript details must list
   every sub-run; a manifest id is ok only if ALL composed sub-runs and ALL its edges pass).
3. SOFTMAX_DESIGN.md §7 (softmax CLI + the scores-rescale 13/10 chain + widening shim) and
   PHASE0_NOTES.md §15 (exp table registration).
4. The driver binaries in /root/zkllm: zkob_rope, zkob_headslice, zkob_headmerge (NEW —
   CLIs in ROPE_ATTENTION_DESIGN §7.1), zkob_softmax, plus the stage-1 five. Do not edit
   or rebuild any driver. An audit job may run driver selftests concurrently — the GPU
   lock in common.py already serializes; keep using it.

## The work
1. register.py: add `ppgen 64` → gen64.bin; generate + register rope-cos/sin tables
   (gen_rope_tables.py, run from /root/zkllm) and softmax-exp-table.bin
   (gen_softmax_exp_table.py); hash-pin all four new artifacts in public.json (the
   MINOR-5 structural assertion must see them).
2. prove_walk.py: per layer, replace the python-float attention segment with the §7.3
   integer chain: q/k/v_proj fc+rescale (already exist) → rope.q/rope.k (+rescales) →
   headslice → per-head scores fc → rescale13 → widen → rescale10 → softmax → values fc
   → values rescale → headmerge → attn_out. The witness-authority rule: attn_out now
   comes from the integer chain (headmerge O2 output), feeding attn_skip.add and
   everything downstream. Composition under the frozen manifest ids per
   ROPE_ATTENTION_DESIGN §4.0 table; sub-obdir layout per §4.0; seeds per §7.3.
3. verify_walk.py: add the new driver verify invocations (com_W path args = the slice
   commitment files per §4.4/4.5) and EVERY §7.4 edge (A1..A15) including closing S1's
   open boundary (com_attn_out ≡ headmerge com_O2). Registered-table args come from
   registration/ (hash-pinned). The per-id ok-rule of §9.5.
4. make_stage1_manifest.py → generalize: stage2 waives ONLY lm_head.commitment_opening +
   statement.logit_binding (embedding has no manifest id — confirm against the manifest;
   if it has one, waive it too with the documented reason).
5. selftest: extend selftest.sh — (a) full stage-2 run: expect ACCEPT with checked = 54
   (= 56 − 2 waived; recount from the manifest yourself and print the arithmetic),
   skipped = the 2 waived only; check_transcript vs the stage2 manifest PASS, and vs the
   FULL manifest report exactly the 2 missing; (b) tamper a slice commitment
   (proofs/…/slice/com_KhT05.bin) → REJECT localized to scores_matmul (driver transcript
   divergence and/or edge); (c) tamper the registered rope-cos table hash in public.json
   → REJECT at registration, no drivers run; restore + re-ACCEPT. PASS/FAIL lines,
   final ALL PASS.
6. ORCHESTRATOR_REPORT.md: append a Stage 2 section — per-obligation timings table
   (prove and verify), TOTAL wall-clock prove and verify for the full forward pass and
   total proof bytes (these are headline numbers for the writeup — measure them cleanly),
   what changed, honest caveats. Expect roughly: stage-1 ~3.5 min + attention ~8.3 min
   prove / ~9.5 min verify (ROPE_ATTENTION_DESIGN §8.6) ⟹ full selftest possibly ~45-60
   min wall; the 900 s per-driver timeout in common.py should be ample — do NOT raise it
   without measuring an actual near-timeout.
7. Do NOT git-commit (coordinator commits). Never push to GitHub; never modify
   int-model-approximation. Honest reporting throughout — a REJECT or a wrong checked-
   count is reported, not papered over.
