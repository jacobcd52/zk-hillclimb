# Task: finish the zkob_rowmax hardening paperwork (edits are DONE and validated)

A prior session applied all 9 ROWMAX_REVIEW.md findings to /root/zkllm/zkob_rowmax.cu,
rebuilt, and the coordinator re-ran the full selftest: ALL PASS, 170 checks
(/root/zkllm/rowmax_selftest_hardened.log). The session ended before the docs. Your job
is ONLY the paperwork (no code edits, no rebuilds):
1. Read ROWMAX_REVIEW.md + the current zkob_rowmax.cu (diff against
   /workspace/projects/zk-hillclimb/zkllm-src/zkob_rowmax.cu = the pre-hardening version)
   to see exactly what the hardening changed.
2. Append "Hardening round" to /root/zkllm/ROWMAX_REPORT.md: per finding what changed
   (file:line), the doc-errata list for MINOR-1/2/9 (design-doc corrections for the
   coordinator), new selftest totals (170).
3. Append "## 19. zkob_rowmax" to
   /workspace/projects/zk-hillclimb/submissions/baseline-native/PHASE0_NOTES.md in the
   §11-18 register style: what it proves (exact row-max causal+vpad, selector tie channel
   + measured-gate duty), obligations, FS summary, CLI, files, chain edges (STAGE3 §2.7),
   real-scale numbers (causal and vpad incl. 10.99 GiB peak), pinned orchestrator
   obligations (standalone-ACCEPT caveat MINOR-7; tie-count reporting).
4. Copy zkob_rowmax.cu, ROWMAX_REPORT.md, ROWMAX_REVIEW.md to
   /workspace/projects/zk-hillclimb/zkllm-src/. No git commits; no GitHub.
