# Task: apply verifier-audit hardening to the orchestrator + re-validate

The orchestrator verifier passed its independent audit
(/workspace/projects/zk-hillclimb/orchestrator/VERIFIER_REVIEW.md, VERDICT: SOUND) with
six MINOR findings. Apply ALL six exactly as the audit's suggested fixes prescribe
(read each finding's "fix" text carefully; where the audit offers options, take the
stricter one). Scope: orchestrator python only (verify_walk.py, common.py, possibly
register.py/prove_walk.py if a fix touches shared helpers). Do not touch any .cu/.cuh,
any driver binary, the harness/ directory, or the manifest.

Then re-validate:
1. Run `bash selftest.sh` end-to-end in /workspace/projects/zk-hillclimb/orchestrator/ —
   must finish ALL PASS (9/9). NOTE: the GPU is shared with a heavy CUDA implementation
   job right now; driver invocations are serialized by the existing lock, just be patient
   (expect the run to take longer than the ~20 min baseline; do not kill it early).
2. Append a "Hardening round" section to ORCHESTRATOR_REPORT.md: each finding → what
   changed (file:line), selftest re-run result.
3. Do NOT git-commit (coordinator commits).

If a fix breaks the selftest, fix YOUR change (the audit's intent is binding, its exact
code suggestion is not) — never weaken a check to make the test pass. Honest reporting.
