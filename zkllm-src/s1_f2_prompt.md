# Task: S1 proof-size shrink + F-2 verify hardening + the prove-transport leftover

The transport rebuild (Stages A-D) is complete, audited SOUND (REBUILD_AUDIT.md), pushed.
Three remaining items, in THIS order (do the quick low-risk ones first so they land even if
S1 runs long):

## IMPORTANT environment facts
- BUILD + RUN BINARIES from /root/zkllm (local disk). NEVER execute binaries from /workspace
  (it's a FUSE mount — large binaries segfault; the documented gotcha).
- BUT direct all RUN OUTPUT (orchestrator run dirs: witnesses, proofs) to **/workspace/zkorch**
  (the 671TB persistent volume) NOT /root/zkorch (the small 50GB container disk). The
  orchestrator's run root is `RUN_ROOT` in orchestrator/common.py (currently /root/zkorch) —
  override it to /workspace/zkorch for your runs (env var, arg, or a local edit you DON'T
  commit — just for running; the committed default can stay). Data on FUSE is fine; only
  exec is not. This removes the disk constraint entirely.
- The deployed /root/zkllm tree is the buildable one (matches the committed repo after the
  F-1 sync). Don't edit the 3 protected headers (vrf_common.cuh, zkob_lookup.cuh); if you
  touch zkob_claims.cuh, rebuild + re-run ALL includer selftests.

## 1. F-2 (quick hardening, do first) — REBUILD_AUDIT.md F-2
Production verify currently runs neither the batched-fold cross-check nor the -dlto kernel
probe (they're selftest-only). Enable them at production verify startup in zkob_batchopen
verify: run bo_probe_kernels() (the -dlto runtime probe) and the fold cross-check
(ZKOB_FOLD_CROSSCHECK behavior) ALWAYS in verify, fail-closed (throw/REJECT on mismatch).
The -dlto miscompile is a real project gotcha; verify should catch it, not just selftest.
Keep it cheap (probe once at startup, not per-tensor). Re-run zkob_batchopen selftest +
the batched orchestrator selftest to confirm still-green.

## 2. S1 (the headline) — proof-size shrink 176MB -> ~45MB, TRANSPORT_REBUILD_DESIGN §2.6
Implement the S1 commitment-storage optimization: canonical-affine commitment encoding
(store G1 points in compressed/affine form instead of the bloated Jacobian serialization)
+ content-dedupe (the same committed tensor appears in multiple obdirs via byte-equal
chaining — store once, reference). Per the design this targets ~45MB. This changes how
commitments are SERIALIZED, not the protocol — the transcript/challenge bytes and the
verifier's checks must be IDENTICAL (the proof is the same; only on-disk encoding shrinks).
Validate: the full faithful-arch-v1 batched walk still ACCEPTs, every forgery still REJECTs
at its named locus, and MEASURE proof+commitment bytes before/after. If full canonical-affine
is a big lift, content-dedupe alone is a worthwhile partial win — land what works, document
the rest.

## 3. Prove-transport pooling (if time) — the Stage-C2 leftover
prove_walk still uses per-process transport (~70s recoverable by pooling the prove side too,
like the single-process verifier). Cheap-ish; do it if S1 leaves time, else document.

## Gates / deliverable
Run the batched orchestrator selftest (--wpriv and non-wpriv) yourself after EACH item;
honest ACCEPT + all forgeries reject at named loci must hold throughout. /root/zkllm/
S1_F2_REPORT.md: per item — done Y/N, the before/after proof-size + verify-time numbers,
what changed (file:line), the selftest verdicts you ran, deviations, what's left. Copy
changed sources to /workspace/projects/zk-hillclimb/zkllm-src/ and orchestrator/ when an item
passes. NO git commits (coordinator commits + pushes after acceptance). No GitHub;
int-model-approximation untouched. GPU free. Run output on /workspace/zkorch. If turn limit
looms, land what's validated + write a clean handoff.
