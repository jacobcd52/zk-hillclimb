# Free prover speedups implemented (from the gpt-5.5-pro + Opus brainstorm)

All four are "free": no change to the proof format, soundness, or privacy posture.
Validated after every change: **FRI selftest 11/11**, **soundness battery 9/9** (honest
accept + all tamper/forge cases reject). The battery runs the GPU prover against the host
verifier, so it also confirms the host/device hashes stay byte-identical.

## What changed (and which bottleneck it attacked)

1. **GPU twiddle tables (NTT).**  `P3Ntt` rebuilt its twiddle tables on the *host* every
   single encode — a sequential ~2^(logM0-1)-length prefix-product (≈2^25 CPU multiplies for
   the W matrix) plus a ~256 MB host→device upload — AND it built the inverse table that
   forward RS-encoding never uses. Now both tables are built on the GPU with a parallel
   pow-kernel (`p3_twiddle_kernel`). This was the single biggest commit cost.
   *(p3_ntt.cuh)*

2. **No codeword round-trip in commit.**  `commit_gpu` used to compute the codeword on the
   GPU, copy it to host, then copy it *back* to the GPU to build the Merkle tree. Now it
   builds the tree directly from the device codeword (`rs_encode_gpu_dev` + `DeviceMerkle::
   build_dev`), copying to host only once (for the later opening). *(p3_basefold.cuh)*

3. **Single-block Merkle node hash.**  Internal nodes hash 64 bytes, but standard SHA-256
   pads that into a **second, all-padding block** — i.e. every internal node paid two
   compression functions. Replaced with `p3_sha256_compress64`: one fixed-input SHA-256
   compression from the IV, no length padding (still collision-resistant — the compression
   IS the CR primitive). Leaves keep full SHA-256 over the 8-byte value (distinct domain).
   Host `node_hash` and the device kernel share the *exact same* `__host__ __device__`
   function, so roots/paths stay identical. This roughly halved internal-node hashing AND
   the host verifier's path checking. *(p3_merkle.cuh, p3_fri.cuh)*

4. **GPU prep contractions.**  The host loop building AX0/AX1/BW0/BW1 was an O(IN·OUT)
   matrix-vector contraction over the full weight matrix (~18% of prove). Moved to two GPU
   kernels (`p3pfc_ax_kernel`, `p3pfc_bw_kernel`). *(p3_private_fc.cuh)*

## Measured result (RTX 4090, R=1, Q=64)

| layer | prove: before → after | verify: before → after |
|---|---|---|
| llama-68m  B=4   | 567.9 → **296.4 ms**  (1.92×) | 97.6  → **31.7 ms** (3.1×) |
| llama-68m  B=16  | 584.2 → **305.3 ms**  (1.91×) | 104.1 → **34.6 ms** (3.0×) |
| llama-68m  B=64  | 595.1 → **322.3 ms**  (1.85×) | 116.0 → **37.5 ms** (3.1×) |
| llama-68m  B=256 | 736.0 → **406.2 ms**  (1.81×) | 125.4 → **42.6 ms** (2.9×) |
| gpt2-large (2048×8192) | 2179.4 → **1111.5 ms** (1.96×) | 120.5 → **39.6 ms** (3.0×) |
| 3B-class   (4096×8192) | 4636.2 → **2561.1 ms** (1.81×) | 163.6 → **41.4 ms** (4.0×) |
| wide       (2048×16384)| 4648.5 → **2620.1 ms** (1.77×) | 160.9 → **41.1 ms** (3.9×) |

**~1.8–2.0× faster prove, ~3–4× faster verify.** Proof size unchanged.

Post-optimization phase breakdown (3B-class, ms): commit 1603, prep 47 (was 557),
sumcheck 4, open 805. The remaining cost is now almost entirely **Merkle hashing inside
commit + the openings** — the next free win is a faster hash (Blake3 / Poseidon2) and/or
packed Merkle leaves, then chunking away the power-of-two padding. These were brainstormed
but not yet implemented (they are larger, coordinated prover+verifier changes).

Note: this is still the **integrity** prover (zero-knowledge not yet achieved — the opening
sumcheck leak is the separate, #1 crypto task). These speedups carry over to the eventual ZK
version.

## Plot: context length 1024, with an estimated generation (decode) bar

The plot now uses a realistic **context length of 1024 tokens** and shows only the current
prover (no "before" bars). Prove times grow modestly with token count because the X/Y
openings and prep scale with B while the weight (W) commitment/opening dominate and are
~B-independent — so a single proof still covers all 1024 tokens.

**Generation (decode) estimate — a literature ratio, not a measurement.** Prefill is
compute-bound (one weight-read serves many tokens); decode is memory-bandwidth bound (one
weight-read per token). So the per-token decode:prefill time ratio is essentially the GPU's
compute-to-bandwidth (ops:byte) ratio when prefill saturates: ~156 (A100), ~295 (H100), ~165
(RTX 4090) as a **ceiling**. In practice (short prompts, imperfect saturation, batched decode)
the observed single-stream ratio is lower, **typically ~10×–100×**. We pick a mid-range,
conservative **50×** and estimate `generate(N) = 50 × (prefill per-token) × N`, using only the
measured prefill numbers (no decode measurement on our model). This keeps linear token scaling
and is well inside the literature range.

At context = 1024 tokens (RTX 4090): proof overhead vs **realistic generation** is far smaller
than vs prefill, e.g. 3B-class — prove 3484 ms vs **generate ~64 ms (~54×)** vs prefill ~1.3 ms
(~2700×). (Estimate; ratio from vLLM / LLM-inference roofline analyses, typical range 10–100×.)
