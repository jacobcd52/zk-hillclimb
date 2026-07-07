# Task: harden zkob_rowmax per audit + document

/root/zkllm/zkob_rowmax.cu passed audit (/root/zkllm/ROWMAX_REVIEW.md, VERDICT: SOUND,
9 MINORs). Apply ALL findings exactly as the audit's fix texts prescribe. Precedents:
the prior hardening rounds (rmsnorm/softmax/rope registers).

Scope rules: verify() may change ONLY if a finding's fix explicitly says so (MINOR-8's
fail-closed REJECT conversion, if prescribed — follow the audit text); prove()'s honest
path only as prescribed (MINOR-6's chunked-commit byte test is selftest-side); everything
else is selftest/report/doc edits. Findings MINOR-1/2/9 are doc/report corrections —
fix ROWMAX_REPORT.md and note the design-doc errata in it (do NOT edit
STAGE3_FAITHFUL_DESIGN.md; coordinator owns it — list the errata in the report).

Then:
1. Rebuild, full selftest — ALL PASS, new negative tests on exactly the named checks
   (MINOR-3 serialized-claim_H forgery, MINOR-4 v1 tamper, MINOR-5 guard throws,
   MINOR-6 chunked-vs-unchunked byte test).
2. Append "Hardening round" to ROWMAX_REPORT.md (per finding: what changed, new totals).
3. Append "## 19. zkob_rowmax" to PHASE0_NOTES.md in the register style (§11-18):
   what it proves (exact row-max, causal+vpad, the selector tie channel + its measured
   gate), obligations, FS summary, CLI, files, chain edges (STAGE3 §2.7), real-scale
   numbers both modes incl. the 10.99 GiB peak, pinned orchestrator obligations
   (MINOR-7 standalone-ACCEPT caveat; the tie-count reporting duty).
4. Sync zkob_rowmax.cu + ROWMAX_REPORT.md + ROWMAX_REVIEW.md to
   /workspace/projects/zk-hillclimb/zkllm-src/. No git commits.

No header edits; no other drivers; no GitHub. GPU may be shared with an audit job.
Honest reporting.
