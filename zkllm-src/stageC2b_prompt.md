# Task: finish Stage C2 — run the batched full-walk end-to-end, capture T4/T5 before/after

A prior agent (Stage C2) WIRED the orchestrator for batched mode but hit its turn limit
before running the measurement. The wiring is on disk: prove_walk.py runs drivers in
--claims mode into a global accumulator + one zkob_batchopen prove; verify_walk.py does the
batched verify (per-driver claim recompute + one zkob_batchopen verify + edges). All 11
drivers are claim-mode-validated (Stage C1, committed). An orphaned profiling probe was
killed; GPU is clear. Your job: VALIDATE + MEASURE, finishing /root/zkllm/STAGE_C2_REPORT.md.

Read first: /root/zkllm/STAGE_C2_REPORT.md (skeleton + whatever the prior agent noted),
STAGE_C1_REPORT.md + STAGE_B_REPORT.md (the claim/accumulator/batchopen interface),
orchestrator/{prove_walk.py,verify_walk.py,common.py,selftest.sh}, TRANSPORT_REBUILD_DESIGN
§6 Stage C + §2, TRANSPORT_REVIEW.md (F3-F6 + BO-1..BO-12). FIRST: run a small/2-driver
batched orchestrator flow to confirm the wiring actually works before the long full run
(the prior agent may have left bugs — find and fix them; you may edit orchestrator/*.py and
the .cu drivers' selftest setup, but NOT the 3 protected headers vrf_common.cuh/
zkob_lookup.cuh, and if you touch zkob_claims.cuh rebuild+retest all includers).

## Gates to actually run and record
- T4: full llama-68m forward pass (faithful-arch-v1, 65 ids) ACCEPTs in BATCHED mode;
  check_transcript passes; the existing forgery/tamper phases still REJECT; PLUS a
  batch-specific forgery (corrupt opening_batch / drop a claim) REJECTs at its named check.
- T5 (THE HEADLINE): on the full faithful-arch-v1 walk, measure NEW batched vs the OLD
  inline numbers (old = prove ~1062s, verify ~999s, 176MB — from prior reports). Report
  NEW total verify wall, proof+commitment bytes, prove wall. The verify projection is ~2-3s;
  confirm or refute on the REAL full walk.
- If the 38GB prove-side batch residency (Stage B flag) OOMs at lm_head/full scale,
  implement the sub-batching/streaming to make it fit the 24GB card, or document the exact
  blocker with numbers. Do NOT silently shrink scope.

## Deliverable
Fill /root/zkllm/STAGE_C2_REPORT.md: T4/T5 with the before/after table, bugs fixed in the
wiring, any streaming work, the batch-specific forgery, deviations, concerns. RUN
selftest.sh (batched) yourself and paste the verdict + the measured numbers. Copy changed
orchestrator files + any .cu to /workspace/projects/zk-hillclimb/zkllm-src/ and
orchestrator/ when passing. NO git commits. No GitHub; int-model-approximation untouched.
GPU is free and clear. The full run is ~15-20 min prove — be patient, don't kill early; if
your turn limit looms, leave the run going in the background (nohup) and write what's
measured so far so the next agent can pick up.
