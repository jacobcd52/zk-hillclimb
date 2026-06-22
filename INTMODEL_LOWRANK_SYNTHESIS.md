# Constructing a low-R_rank, ZK-provable integer model — synthesis (Opus + gpt-5.5-pro)

Context: served model M_q = fixed fp8 (`_scaled_mm`, fp32 accumulation). We want an INTEGER model
M_int that is Freivalds/sumcheck-provable AND minimizes R_rank vs M_q. Measured baseline: exact-int
codebook M_int gives R_rank = 0.355 (frac_rank0 0.94); matching accumulation *precision* (fp32) gives
0.361 (no help); the residual is near-tie floating-point nondeterminism. Tie convention: token-ID
(recheck unchanged at 0.355).

## Both agents converged on the same partition of the design space

The residual ~0.35 bits is **near-tie disagreement between two distinct accumulation ORDERS**, not a
systematic bias. So:

- **Class I — "prove a different exact-integer scheme"** (exact codebook, fp32 replica, pairwise/Kahan,
  block-tiled int, even a *distilled integer twin*): each is just a THIRD distinct order, so it stays at
  the ~0.33–0.36 floor. The measurement already shows this (exact-int vs fp32 disagree by 0.328).
- **Class II — "served == proven, bit-for-bit"**: the only way R_rank → 0.

Key numeric (gpt): to move the needle you must kill most of the 6% near-tie flips —
R_rank 0.10 needs frac_rank0 0.987; R_rank 0.05 needs 0.994. Choosing a cleverer integer accumulator
can't do that.

## The winner (both agents, #1): Scenario B — serve the exact integer kernel you prove

Stop serving fp8-and-proving-an-int-twin; **serve the deterministic integer model itself** and prove
that. Exact integer reduction has a UNIQUE value (no order chaos), so honest R_rank = 0 AND it's the
cheapest possible proof (a bare exact int matmul — Freivalds- and sumcheck-native). Two instantiations:
- **B-codebook (minimal model change):** represent each e4m3 value as integer phi(c)=2^9*e4m3(c)
  (|.|<=229376, ~int19); S = sum phi(A)phi(W) exact in int64 (Qwen 0.5B: |S|<2^49, safe in a 61-bit
  field); deterministic fixed-point rescale by the committed scales. Closest to the current checkpoint;
  slower (operands aren't native int8 tensor-core width).
- **B-int8 (fast/production):** int8 activations + int8 weights, exact int32 accumulation
  (4864*127^2 < 2^27), integer multiply-shift rescale. Cleanest Freivalds object; needs a short
  QAT/distill to preserve quality.
- Plus: deterministic fixed-point/LUT RMSNorm/softmax/RoPE/SiLU, and **committed (public-seed) Gumbel
  sampling** so the sampler's choice isn't itself a channel.
Cost: ~1x (the cheapest real proof). Tradeoff: changes the served arithmetic (int8 vs fp8 quality gap,
small at this granularity; codebook variant keeps fp8 values exactly).

## Scenario A (M_q fixed) — the honest options

- **A2: rank-distilled integer twin (cheap).** Train a Freivalds-friendly integer model (+ optional
  low-rank logit-correction head, r=32-128, ~5-15% extra) to mimic M_q's logits with a rank-aware loss.
  This is capability-free (M_q untouched). **The agents DIVERGE on how much it helps:** Opus calls it a
  trap (stuck at the floor); gpt expects a modest 0.355 -> ~0.25-0.32 (maybe 0.20-0.28 domain-specific),
  NOT near zero. Reconciliation with our data: M_q sits slightly further from the clean computations
  (0.355/0.361) than they do from each other (0.328), so there IS a small *systematic* component
  (~0.03 bits) distillation can capture; the bulk is irreducible per-token order chaos it cannot. Net:
  expect a small win, a floor around ~0.3, never ~0.
- **A3: blockwise fp8 tile-replay (near-zero, expensive).** Reproduce M_q's exact tile/accumulation
  order as an integer + fp32-accumulator-state circuit. R_rank -> ~0 if the tiling is exact, but
  ~50-100x more lookup-heavy (tile replay) up to 10^3-10^5x (per-FMA / Hawkeye). Fragile: cuBLASLt
  SM89 order is undocumented, batch-variant, and drifts with driver/shape. Only path to ~0 without
  changing M_q.

## Two warnings both agents stressed
- **Pinning a deterministic fp8 CUDA kernel is NOT enough by itself:** rounded fp32 accumulation isn't
  bilinear, so Freivalds doesn't apply. You must EITHER prove the exact fp arithmetic (expensive, A3)
  OR change serving to exact integer accumulation (B). Determinism enables a low τ but doesn't give a
  cheap proof on its own.
- **Verifier tolerance / top-K acceptance is NOT capacity reduction** — the accepted set becomes the
  covert channel. It's policy accounting, not a lower-R_rank model.

## Recommendation
If we may constrain serving (Scenario B): **serve = prove an exact deterministic integer kernel**
(codebook variant first for minimal risk; QAT-int8 if throughput matters) + committed Gumbel sampling
-> R_rank = 0 at the cheapest proof cost. This is the free-lunch quadrant and reframes the project:
the lowest-R_rank provable integer model is simply the one you actually serve.
If serving must stay fp8 (Scenario A): a distilled integer twin buys a small reduction (~0.3 floor);
only exact tile-replay reaches ~0, at 50-100x+ proof cost.
