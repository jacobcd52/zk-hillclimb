# Task: Stage C-part-2 — orchestrator batch integration + single-process verifier + the full-walk before/after measurement

Stages A/B/C1 DONE: all 11 drivers emit batch claims in claim mode; zkob_batchopen +
zkob_claims.cuh (flattened fold + fast IPA) validated; verify projects ~2-3s full-walk.
NOW wire it into the orchestrator so a WHOLE llama-68m forward pass proves and verifies
through ONE batched opening, and MEASURE the real before/after.

Read: /root/zkllm/STAGE_C1_REPORT.md (the claim-emit interface: each driver in --claims
mode writes claims.bin + witrefs + drvstate into an accumulator dir; verifier-side recompute
was DEFERRED to here), STAGE_B_REPORT.md (zkob_batchopen prove/verify, the BO battery,
comref canonicalization, one-gen-file-per-domain assertion), TRANSPORT_REBUILD_DESIGN §6
Stage C + §2, TRANSPORT_REVIEW.md (F3-F6 pins, BO-1..BO-12), and the orchestrator
(/workspace/projects/zk-hillclimb/orchestrator/: common.py, prove_walk.py, verify_walk.py,
selftest.sh, ORCHESTRATOR_DESIGN.md).

## Work
1. **prove_walk.py**: run every driver in --claims mode emitting into ONE global accumulator
   dir per run; after the walk, run zkob_batchopen prove ONCE over the accumulator (the
   batch-eval sumcheck + one IPA per domain). Keep the per-driver round/lookup/link work
   and chain edges exactly as today. Add the `opening_batch` artifact.
2. **verify_walk.py** (or a new single-process zkverify_walk): instead of ~235 driver verify
   subprocesses each doing inline IPAs, each driver-verify recomputes its claims into the
   verifier accumulator (no opening), then ONE zkob_batchopen verify discharges all claims;
   plus the F3 size checks, claims_match, all chain edges + homomorphic links + statement
   obligations exactly as today. Conditional verdict: ACCEPT iff every driver's non-opening
   checks pass AND the batch verifies AND every edge holds.
3. **selftest.sh**: a batched-mode full run (reuse the stage3 + faithful-arch flows). Gates:
   - T4: full llama-68m forward pass (faithful-arch-v1, 65 ids) ACCEPTs in batched mode;
     check_transcript still passes; the SAME forgery/tamper phases still REJECT (incl. a
     batch-specific one: corrupt the opening_batch -> REJECT).
   - T5: MEASURE before/after on the full faithful-arch-v1 walk: OLD transport (inline,
     ~999s verify / 176MB) vs NEW (batched). Report total verify wall, proof+commitment
     bytes, and prove wall (should be ~unchanged + small batch overhead). This is THE
     headline number — measure it cleanly (exclusive GPU, the run is ~15-20 min prove).
4. Honest accounting: if the 38GB prove-streaming item (Stage B flag) blocks the full
   batch at lm_head scale, implement the sub-batching/streaming needed or document the
   exact blocker; do NOT silently shrink scope.

## Deliverable
/root/zkllm/STAGE_C2_REPORT.md: T4/T5 results with the before/after table (verify wall +
proof size + prove wall, old vs new), what changed in the orchestrator, the batch-specific
forgery test, any streaming work, deviations, concerns. Run selftest.sh yourself and paste
the verdict. Copy changed orchestrator files + any .cu to zkllm-src/ when passing. NO git
commits (coordinator commits). Don't edit the 3 protected headers. No GitHub;
int-model-approximation untouched. GPU free. The full run is long — be patient, don't kill
early; if an agent-turn limit looms, leave the run going and write what's measured so far.
