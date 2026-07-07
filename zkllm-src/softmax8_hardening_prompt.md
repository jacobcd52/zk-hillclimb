# Task: harden zkob_softmax8 + headmerge per audit + document (final cycle-2 pass)

/root/zkllm/SOFTMAX8_REVIEW.md: VERDICT SOUND for both zkob_softmax8.cu and the
zkob_headmerge perm-flag diff; 7 MINOR findings. Apply ALL exactly as the audit's fix
texts prescribe (stricter option where offered). Standard scope rules from prior
hardening rounds: verify()/honest-prove() changes only where a finding explicitly
prescribes them; otherwise selftest/report/doc only. Doc-errata findings: fix in
SOFTMAX8_REPORT.md (never edit the design doc — list errata for the coordinator).

Then:
1. Rebuild both, full selftests — ALL PASS, new negative tests rejecting on exactly the
   named checks. NOTE another hardening job may briefly share the GPU; retry transient
   CUDA failures once.
2. Append "Hardening round" to /root/zkllm/SOFTMAX8_REPORT.md (per finding + new totals).
3. Append PHASE0_NOTES.md sections "## 20. zkob_softmax8" and "## 21. zkob_headmerge
   perm flag" in the register style (§11-19): obligations, FS summaries, CLI, files,
   chain edges (STAGE3 §4.2/§4.3), real-scale numbers, pinned orchestrator obligations
   (incl. which mode the faithful-arch manifest uses and the cross-mode splice defense).
4. Sync zkob_softmax8.cu, zkob_headmerge.cu, gen_softmax8_table.py, SOFTMAX8_REPORT.md,
   SOFTMAX8_REVIEW.md to /workspace/projects/zk-hillclimb/zkllm-src/. No git commits.

No header edits (sha256-prove untouched); no other drivers; no GitHub. Honest reporting.
