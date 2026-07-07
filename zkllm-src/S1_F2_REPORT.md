# S1 + F-2 + prove-pooling report (IN PROGRESS — being filled as gates pass)

> Status: DRAFT. A previous session implemented F-2 + S1a + S1b(pack) + the
> ZKORCH_RUN_ROOT override and was killed mid-rebuild (empty s1_f2_result.json,
> stale binaries vs zkob_claims.cuh). This session reviewed that work, completed
> the rebuild, and is re-running the full validation battery. Nothing below is
> final until its gate line says PASS.

## Item 1 — F-2 verify hardening: (pending gates)
## Item 2 — S1 proof-size shrink: (pending gates)
## Item 3 — prove-transport pooling: (staged, pending S1 gates)

## Baseline (before) numbers, measured from existing runs
- inline faithful-arch-v1 (stage3v2-fa): proof_bytes = 175,581,196 (~167.4 MiB), prove_wall 1062.5 s
- batched wpriv (wprivconfirm-fawp): proof_bytes = 176,411,934 (~168.2 MiB), prove_wall 526.3 s, verify_wall 30.45 s
- batched non-wpriv C2 baseline (selftest.sh pinned): proof 176,326,580 B, prove 521.96 s, verify 27.12 s
