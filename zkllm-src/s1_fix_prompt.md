# Task: fix the S1 canonical-G1 encoding bug (zkob_fc crash), then re-validate S1 + F-2 end-to-end

A prior agent implemented F-2 (kernel probe in production verify — VALIDATED GOOD, leave it)
and S1 (proof-size shrink via canonical-affine G1 commitment encoding + content-dedupe). F-2
works; **S1 is BUGGY**: `zkob_fc selftest` ABORTS at the end with
  `terminate called: std::runtime_error: zkg1: invalid canonical G1 encoding (point 0, code 3)`
The inline honest+forgery cases all pass first; the crash is in the canonical-G1
encode/decode path (the S1 serialization change). zkob_batchopen selftest passes 33/33, so
the bug is in a code path fc exercises that batchopen doesn't (likely an INFINITY/identity
commitment point, or a flag-bit / y-parity / field-range edge case in the compressed
BLS12-381 G1 format — 48-byte compressed, 3 top flag bits: compression, infinity, sort).

Read: /root/zkllm/S1_F2_REPORT.md (what S1a/S1b/the "pack" encoding did), the canonical-G1
encode/decode code the agent added (grep for "canonical", "zkg1", "code 3", the pack/compress
helper — likely in zkserial.cuh / zkob_claims.cuh / zkob_batchopen.cu), and the BLS12-381
compressed-G1 convention in the upstream g1-tensor.cuh / bls12-381.cuh (match it EXACTLY —
do not invent a format). The 3 protected headers (vrf_common.cuh, zkob_lookup.cuh) must
stay byte-identical.

## Do
1. Find the bug. Most likely: the identity/infinity point (a zero commitment row — fc can
   have one) isn't encoded with the infinity flag, OR the y-parity/sort bit is wrong, OR a
   field element isn't reduced before packing. Reproduce with `./zkob_fc selftest`, get the
   failing point, fix the encoder AND decoder so they round-trip ALL points incl. infinity.
   Add an encode->decode round-trip self-check over random + infinity + generator points.
2. CRITICAL invariant: the canonical encoding must be a pure ON-DISK SERIALIZATION change.
   The transcript/Fiat-Shamir bytes, challenges, and the VERIFIER'S checks must be
   byte-identical to before S1 (the proof is the same; only commitment storage shrinks).
   Confirm: a batched walk's transcript/challenges are unchanged vs the pre-S1 path.
3. Rebuild ALL affected binaries (pinned -dlto commands). Re-run: zkob_fc, zkob_rescale,
   zkob_batchopen selftests (ALL PASS), then the FULL batched orchestrator selftest
   (selftest.sh, both --wpriv and non-wpriv) with RUN OUTPUT ON /workspace/zkorch (set
   ZKORCH_RUN_ROOT or RUN_ROOT=/workspace/zkorch; binaries stay on /root/zkllm — FUSE can't
   exec). Honest ACCEPT + every forgery REJECT at its named locus must hold.
4. MEASURE the proof+commitment bytes before (176 MB) vs after S1, and confirm verify is
   still ~27-30 s. Report the real shrink (target ~45 MB; content-dedupe alone is a fine
   partial win if full canonical-affine fights you — but it must not CRASH).

## Deliverable
Finish /root/zkllm/S1_F2_REPORT.md: F-2 done (confirm), S1 fixed (the bug + fix file:line,
the round-trip check, before/after proof size + verify time), prove-pooling if done, the
selftest verdicts you ran. Copy changed sources to /workspace/projects/zk-hillclimb/
zkllm-src/ + orchestrator/ ONLY when ALL selftests pass. If S1 canonical-affine proves too
fragile, FALL BACK to content-dedupe-only (smaller but safe) and say so. NO git commits.
Don't edit protected headers. No GitHub; int-model-approximation untouched. Honest reporting.
