# Task: orchestrator STAGE 3 — close the manifest (STAGE3_FAITHFUL_DESIGN Part B)

Extend /workspace/projects/zk-hillclimb/orchestrator/ (stage 2: 54/56 ids, hardened,
audited) to FULL manifest coverage per STAGE3_FAITHFUL_DESIGN.md §3 (Part B is normative:
§3.1 final norm, §3.2 lm_head/gen32768, §3.3 logit binding, §3.4 wiring/ordering, §3.5
accounting). Part C (faithful-arch) is NOT this task — stage 3 binds the SAME function.

Read: STAGE3_FAITHFUL_DESIGN §1-3 + §6; ORCHESTRATOR_DESIGN.md + report (incl. stage-2
section); PHASE0_NOTES §19 (zkob_rowmax register: CLI, edges, the tie-count reporting
duty — implement it in prove_walk per §2.4(ii): report Σ log2(#maximizers) in
prove_manifest.json); the driver binaries incl. zkob_rowmax (do not edit/rebuild any).

Work items:
1. register.py: ppgen 32768 → gen32768.bin; export + commit lm_head weight (§3.2 incl.
   the tie_word_embeddings flag recording); final-norm gain g_final export+commit;
   ALL hash-pinned in public.json (the MINOR-5 assertion must cover them).
2. prove_walk.py: final_norm trio (exact-R authority per §3.1) → lm_head fc + rescale
   (§3.2) → t* derivation (argmax of logits per §3.3) → t* written + sha256-pinned into
   public.json BEFORE sealing per §3.4's pinned ordering (re-derive run_seed correctly —
   follow §3.4 exactly) → rowmax vpad instance with t*.
3. verify_walk.py: the new driver verifies (fc with registered lm_head com; rescale;
   rowmax vpad with the registered t* path), edges L1 + the §3.4 list, statement.
   logit_binding semantics per §3.3; manifest accounting per §3.5 (56/56 non-waived
   checked + 3 covered-waived; only embedding.lookup stays waived-uncovered).
4. selftest.sh: extend — (a) full stage-3 run ACCEPT with the §3.5 counts (print the
   arithmetic); (b) tamper t* in public.json (one token id) → REJECT at
   statement.logit_binding (or registration hash — whichever §3.4's ordering dictates;
   assert which); (c) tamper the registered lm_head com hash → registration REJECT, no
   drivers run; restore + re-ACCEPT. Keep all stage-2 phases passing.
5. ORCHESTRATOR_REPORT.md: append Stage 3 — per-obligation timings (lm_head fc/rescale
   and rowmax-vpad rows separately; compare vs §5.1 predictions), new TOTAL prove/verify
   wall + proof bytes, the measured selector-tie count, honest caveats.
No git commits; no GitHub; never modify int-model-approximation; GPU is free (expect the
full selftest ~60-90 min; the 900 s per-driver timeout should hold — measure, don't raise
blindly; if a single driver genuinely needs more, raise to exactly what §5.1's worst
prediction + 50% implies and document).
