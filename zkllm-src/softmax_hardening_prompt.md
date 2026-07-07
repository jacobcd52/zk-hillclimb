# Task: harden zkob_softmax selftest per audit + document

`/root/zkllm/zkob_softmax.cu` passed implementation (SOFTMAX_REPORT.md) and independent
audit (SOFTMAX_REVIEW.md, VERDICT: SOUND). Apply the audit's MINOR recommendations and
write the documentation. Read SOFTMAX_REVIEW.md (all MINOR findings) and the selftest
section of zkob_softmax.cu first; zkob_rmsnorm.cu's hardening round is the precedent.

## Changes — ONLY in selftest/evil-mode code; verify() and prove()'s honest path must not
change by a single character, with ONE exception noted in (3):
1. MINOR-1: add a semantic out-of-range-limb evil mode (a limb value ≥ LEN_R with
   compensating adjustments keeping everything else consistent) → must be rejected by the
   limb lookup; require the exact reject string.
2. MINOR-2: add the missing evil==0 convention sanity checks for the four L-plane openings
   (cross-check each v** against multi_dim_me of L at the corresponding point, honest runs
   only), matching the audit's suggestion.
3. MINOR-4: if (and only if) the audit's suggested fix is to convert missing-proof-file
   throws into clean REJECT lines in verify(), apply exactly that suggested fix (it is
   fail-closed either way; follow the audit's text). If the audit merely notes it, leave
   the code alone and document the behavior.
4. Rebuild (pinned commands in SOFTMAX_REPORT.md) and run the FULL selftest — everything
   must pass; new evil modes must be rejected by exactly the named checks. If a new mode
   trips a different check, STOP and report honestly; do not loosen the match.
5. Append a "Hardening round" section to SOFTMAX_REPORT.md (new modes, new totals).
6. Append "## 15. zkob_softmax" to
   /workspace/projects/zk-hillclimb/submissions/baseline-native/PHASE0_NOTES.md in the
   register style of §11–14: what it proves (zero-advice softmax, 0-bit covert capacity),
   the obligations, FS schedule summary, CLI, file list, chain interface (com_z ==
   rescale-stage-2 com_Xr; com_P == values-matmul com_X; int32 P chain file at scale 2^16),
   real-scale numbers, the exp-table registration rule (sha256, generation script
   gen_softmax_exp_table.py), and the pinned orchestrator obligations from the audit
   (MINOR-5 chain byte-equalities; MINOR-4 behavior).
7. Copy final zkob_softmax.cu + gen_softmax_exp_table.py + SOFTMAX_REPORT.md +
   SOFTMAX_REVIEW.md to /workspace/projects/zk-hillclimb/zkllm-src/. Do NOT git-commit.

Rules: no edits to any other source; never push to GitHub; shared header untouchable.
GPU shared with another job (orchestrator runs driver binaries) — serialize, retry on OOM.
