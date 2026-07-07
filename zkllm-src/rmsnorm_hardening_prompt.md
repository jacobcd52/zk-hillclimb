# Task: extend zkob_rmsnorm selftest coverage (3 new evil modes) + document

`/root/zkllm/zkob_rmsnorm.cu` passed implementation (RMSNORM_REPORT.md) and an independent
soundness audit (RMSNORM_REVIEW.md, VERDICT: SOUND). The audit's MINOR-1 and MINOR-2 ask for
three additional semantic evil modes in the selftest. Your job: add exactly those, re-validate,
and write the documentation section. Read RMSNORM_REVIEW.md MINOR-1/2/5/7 first, plus the
selftest section of zkob_rmsnorm.cu (the existing evil modes 1–5 are the pattern to copy).

## Changes (ONLY in the selftest/prover-evil sections of zkob_rmsnorm.cu — the verify()
function must not change by a single character; same for prove()'s honest path):
1. evil=6: R[idx] −= 2, P2 recomputed mod p (negative wraps), limbs = low 80 bits of the
   field value → expect rejection by "affine link P2" (mirror of evil=1).
2. evil=7: R[idx] −= 2, limbs honest-truncated, P2 = limb reconstruction → expect rejection
   by "q2 round 0" (mirror of evil=2; strict=false only for q2).
3. evil=8: L[0, s] += 65536 for one in-range s, with P1 recomputed from the modified limbs
   (so the affine link and quartics stay consistent) → expect rejection by the limb lookup
   (round 0 or terminal — match the actual reject string).
Use the existing reject-reason-string matching discipline: the selftest must require the
EXACT named check to fire, like evils 1–5 do.

## Then
4. Rebuild (pinned commands in RMSNORM_REPORT.md) and run `./zkob_rmsnorm selftest` —
   everything must pass including the new modes (small + real-scale). If a new evil mode is
   rejected by a DIFFERENT check than expected, STOP and write what happened to the report —
   do not loosen the match.
5. Append to /root/zkllm/RMSNORM_REPORT.md: a short "Hardening round" section (new evil
   modes, new selftest totals).
6. Write the documentation: append a new section "## 14. zkob_rmsnorm" to
   /workspace/projects/zk-hillclimb/submissions/baseline-native/PHASE0_NOTES.md, in the same
   style as the existing driver sections (read §11–13 for the register): what the obligation
   proves, the seven sub-obligations, FS schedule summary, CLI signatures, file list
   (NOTE: com_W_ is saved as com_Wr.bin — orchestrator must use that name), chain interface
   (W.i64 / Y.i64; com_W==rescale com_X, com_Wr==rescale com_Xr), real-scale numbers, and
   TWO pinned orchestrator obligations from the audit: (a) com_X here MUST be chained
   byte-identical to the upstream activation commitment — a standalone ACCEPT proves much
   less (audit MINOR-5); (b) the com_Wr.bin naming (MINOR-7).
7. Copy the final zkob_rmsnorm.cu to /workspace/projects/zk-hillclimb/zkllm-src/ (overwrite).
   Do NOT git-commit anything; the coordinator commits.

Rules: do not touch any other source file; do not push anything to GitHub; the shared header
is untouchable. GPU is an RTX 4090, may be shared with another measurement job — if you hit
an OOM, wait a minute and retry.
