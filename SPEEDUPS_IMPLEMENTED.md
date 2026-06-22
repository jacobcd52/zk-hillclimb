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

**Generation (decode) estimate.** Autoregressive decode produces one token at a time and is
**memory-bandwidth bound**: each token must read the layer's weights from HBM. The standard
single-stream decode model is `t_token = weight_bytes / HBM_bandwidth` — the same roofline
that gives e.g. ~24 tok/s for a 70B model on an H100 (3.35 TB/s). We estimate generation of
N tokens as `N × t_token`. We take `t_token` = the **measured single-token forward (B=1)** of
each layer, which on the RTX 4090 already sits right at the fp16 bandwidth roofline
(llama up_proj ~8.8 µs, gpt2-large ~30 µs, 3B-class ~148 µs) — so this is grounded, not a
launch-overhead-inflated number. Prefill (the measured `forward(B)`) amortizes one weight
pass across all B tokens, so it is much faster than decode for the same tokens.

Measured at B=1024 (RTX 4090): the proof overhead vs **realistic generation** is far smaller
than vs prefill, e.g. 3B-class — prove 3484 ms vs **generate ~151 ms (~23×)** vs prefill
~1.3 ms (~2700×). (Estimate, single-stream decode; sources: vLLM / LLM-inference roofline
analyses.)
