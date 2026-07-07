# Task: finish Stage C2 — fix tamper localization + single-process verifier, re-measure

The batched full-walk orchestrator WORKS: faithful-arch-v1 (65 ids) ACCEPTs in batched
mode, opening_batch ACCEPT over 7 sub-batches / 1535 claims (log:
/root/zkorch/selftest_c2_batched.log). Measured: prove 636 s (down from 1062), verify
383 s (down from 999), proof 176 MB. TWO things remain before Stage C2 is done:

## Problem 1: tamper LOCALIZATION (3 selftest FAILs — soundness OK, attribution wrong)
The 3 failing phases all REJECT correctly (forgeries ARE caught) but blame the wrong/too-many
ids, because a tampered claim fails its whole sub-batch so ALL ids sharing that sub-batch
show as rejected:
- slice tamper -> rejects, but flags all 15 commitment_opening ids + statement.prompt_binding
  (expected: localized to layer0.attn.scores_matmul). (log line ~835)
- com_mx tamper -> the chain-edge RM2.05 byte-check DID localize it to layer0.attn.softmax,
  but the test still flagged "wrong localization" (the batch-fail also blamed others). (~1085)
- vfin tamper -> correctly REJECTs at opening_batch.terminal (batch 0), test flagged "wrong
  locus/gating" (~1341).
FIX (preserve the project's "reject at exactly the named check" discipline): when a
sub-batch fails, the verifier must PINPOINT the offending id rather than blaming the whole
sub-batch. Cleanest approach (your call): on batch failure, fall back to per-claim / per-id
re-checking (or use the chain-edge + per-driver non-opening checks, which already localize
com_mx) so the transcript's `rejected` set is exactly the tampered id(s). Update selftest
expectations ONLY if the batched attribution is genuinely as tight as the inline one;
do NOT loosen a test to hide a real attribution loss. All honest ACCEPT + every forgery
REJECT must hold, at the correct named locus.

## Problem 2: verify is 383 s, not the projected ~2-3 s
The batching killed the IPA-opening cost, but verify_walk still spawns ~235 driver-verify
SUBPROCESSES (each a CUDA-init + round/lookup recompute). The design (TRANSPORT_REBUILD_DESIGN
§6 Stage C) calls for a SINGLE-PROCESS zkverify_walk that does all per-driver claim
recompute + the one batched verify + edges in one process, amortizing the CUDA init.
Implement it; measure the new verify wall. (If single-process is a big lift, at minimum
quantify how much of the 383 s is subprocess CUDA-init floor vs real compute, so we know the
achievable floor.)

## Gates / deliverable
Re-run selftest.sh batched: T4 = honest ACCEPT + EVERY tamper rejects at its correct named
locus (the 3 above + the existing battery). T5 = the before/after table (prove/verify/proof,
old vs new) with the single-process verify number. Fill /root/zkllm/STAGE_C2_REPORT.md;
RUN the selftest yourself and paste the verdict + numbers. Copy changed orchestrator/*.py
(+ any .cu) to /workspace/projects/zk-hillclimb/{orchestrator,zkllm-src}/ when ALL pass.
Read STAGE_C1/B_REPORT.md + TRANSPORT_REBUILD_DESIGN §6 + TRANSPORT_REVIEW.md F3-F6 first.
Don't edit the 3 protected headers. NO git commits. No GitHub; int-model-approximation
untouched. GPU free. Full run ~15-20 min prove — be patient; if turn limit looms, leave the
run going (nohup) and write what's measured + what's left.
