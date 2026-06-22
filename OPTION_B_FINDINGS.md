# Option B (calibrate the integerization) — findings on an fp8-native model

**Model:** RedHatAI/Qwen2.5-0.5B-FP8-dynamic (base), fp8-native served weights.
**Setup:** served M_q = `FP8Linear` (`torch._scaled_mm`, fp32 accumulation). Reference
M_int candidates all use the IDENTICAL per-token fp8 activations and per-row fp8 weights
as M_q (so weight/activation/non-matmul are already at parity). R_rank = entropy of the
served token's post-Gumbel rank under M_int (= benign covert-channel capacity, bits/token).
On-policy, T=1, 4967 completion tokens. Script: `capacity/optionb_decompose.py`,
data: `capacity/optionb_decompose.json`.

## Result (decomposition)

| served M_q | proof reference M_int | R_rank | frac_rank0 |
|---|---|---|---|
| `_scaled_mm` fp8 | exact codebook int (current) | **0.355** | 0.942 |
| `_scaled_mm` fp8 | fp32 accumulation            | **0.361** | 0.941 |
| codebook int     | fp32 accumulation            | 0.328     | 0.948 |

Per-token logit L2 between any two of the three computations is ~90–97 over a 152k vocab
(tiny per-logit; ~94% exact top-1 agreement).

## Interpretation — the free calibration lever does NOT help

- Changing the proof reference's accumulation from exact-int to fp32 (i.e. matching the
  served kernel's *precision*) leaves capacity unchanged (0.355 → 0.361).
- The two "clean" computations (exact-int and fp32) disagree with EACH OTHER by ~0.33 bits.
- So the residual ~0.35 bits is **near-tie floating-point nondeterminism**: any two DISTINCT
  accumulation orders disagree about equally, concentrated at the ~6% of positions where the
  top-1/top-2 logits are near-ties (max_rank ≤ 9, mean_rank ~0.06). There is no systematic
  precision/order offset for a different integer scheme to cancel — just chaos at ties.

**Conclusion:** "calibrate which integer scheme you prove" (the free Option-B knob) has
essentially no effect on this gap. The capacity is set by arithmetic nondeterminism at ties.

## The lever that DOES work, still free in capability — determinism

The only thing that collapses this gap is making the served arithmetic **deterministic and
identical to the proven arithmetic** (served == proven, bit for bit). Then:
- the honest served logits equal the proven logits → benign R_rank → 0;
- the verifier's acceptance tolerance τ can go to ~0 (today τ must be loose to avoid rejecting
  honest fp32-accumulation noise — that looseness is exactly what gives an adversarial server
  its acceptance-set slack);
- with τ → 0 the adversarial acceptance set collapses to ~the single proven token → adversarial
  capacity → 0 too.
- **Zero capability cost:** identical fp8 weights and values; we only PIN the accumulation order.

So Option B's real form is not "pick a better integer scheme" but **"require the datacenter to
serve the exact committed integer kernel"** — i.e. arithmetic determinism / batch-invariant
inference, then prove that exact kernel (Hawkeye replays the served order; cost is the existing
DiFR-vs-proof-cost Pareto), plus verifiable/committed sampling for the residual
sampling-choice channel.

## Practical obstacles / next steps
- Build/serve a **deterministic** fp8-speed kernel whose accumulation order is fixed and
  cheaply provable (Hawkeye-on-the-served-order). Connects to deterministic / batch-invariant
  inference (e.g. Thinking Machines' batch-invariant kernels).
- Confirm empirically: R_rank(served==proven) = 0, and that `_scaled_mm` is run-to-run
  deterministic on fixed batch/shape (batch-invariance is the catch).
- The sampling-choice channel (adversary picking among τ-acceptable tokens) is separate and is
  closed by τ→0 (enabled by determinism) + committed-seed sampling.
