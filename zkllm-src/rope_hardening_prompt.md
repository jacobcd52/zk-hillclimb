# Task: harden the three attention drivers per audit + document (final driver-layer pass)

/root/zkllm/{zkob_rope,zkob_headslice,zkob_headmerge}.cu passed their independent audit
(/root/zkllm/ROPE_REVIEW.md — VERDICT: SOUND per driver). Apply the MINOR findings and
write the documentation. Precedents: the rmsnorm and softmax hardening rounds.

## Changes — ONLY selftest/evil-mode code; no verify() may change by a character, and
prove() honest paths only as the audit's fixes prescribe:
1. MINOR-1: rewrite the measured-numbers table in /root/zkllm/ROPE_IMPL_REPORT.md with
   the audit's resolved mapping (and remove the open question).
2. MINOR-2: add a headmerge toy case with B ≠ C_pad (e.g. B=16, C=6, HD=2 ⟹ C_pad=8) —
   the π⁻¹ gather must be exercised where row/col confusion is NOT vacuous; honest
   ACCEPT + one targeted evil (a π-variant gather, e.g. the stage-1 "plain concat"
   M instead of π(M)) rejected by the Σ c_h == ev check.
3. MINOR-3: add a rope evil mode that corrupts the SIN-side (hadamard-2 / W2 path)
   semantically — e.g. Y64 built with σ sign flipped on one entry, everything else
   honest-consistent → must be rejected by the named hadamard-2 / final c1+c2 check
   (whichever the audit's text names — follow it).
4. MINOR-4: extend headslice evil coverage so each error family (wrong head, transposed
   wrong, off-by-one column) hits each tensor family (Q/K/V) at least once across cases.
5. MINOR-5/-7: no code change — document (gate margin <1s on prove; missing-file throw
   is fail-closed) in the report and PHASE0 sections below.
6. MINOR-8: apply only if zero-risk (dead-symbol cleanup in selftest code); else skip.

## Then
7. Rebuild the three binaries (pinned commands) and re-run ALL THREE full selftests —
   ALL PASS required; new evils must hit exactly the named checks (no loosening).
   NOTE: a heavy orchestrator job shares the GPU (lock at /tmp/zkorch.gpu.lock is NOT
   used by raw selftests — just run sequentially and retry once on transient CUDA
   failures; expect slowness).
8. Append "Hardening round" to ROPE_IMPL_REPORT.md (modes added, new totals, gate note).
9. Append PHASE0_NOTES.md sections "## 16. zkob_rope", "## 17. zkob_headslice",
   "## 18. zkob_headmerge" in the §11–15 register style (obligations, FS summary, CLI,
   files, chain edges by reference to ROPE_ATTENTION_DESIGN §7.4, real-scale numbers,
   pinned orchestrator obligations incl. MINOR-6's standalone-ACCEPT caveat and the
   §9.1 gate status).
10. Sync the three .cu files + ROPE_IMPL_REPORT.md + ROPE_REVIEW.md to
    /workspace/projects/zk-hillclimb/zkllm-src/. Do NOT git-commit.

Rules: no edits to any other source; shared header untouchable; no GitHub; honest
reporting (a FAIL is a FAIL).
