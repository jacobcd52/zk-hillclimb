## Bottom line

Your measurements imply a strong constraint: **for Scenario A, cheap exact-integer matmul variants are already at the floor**.  
`frac_rank0 ≈ 0.94` means the `0.35 bits/token` is mostly a 6% near-tie flip process. If the nonzero ranks are mostly rank 1, then:

- `R_rank ≈ 0.35` ↔ `frac_rank0 ≈ 0.94`
- `R_rank ≈ 0.10` would require `frac_rank0 ≈ 0.987`
- `R_rank ≈ 0.05` would require `frac_rank0 ≈ 0.994`

So to get near zero you must eliminate most of those 6% near-tie disagreements. Merely choosing a different exact-int accumulation, fp32 accumulation order, Kahan/pairwise sum, etc. will not do that; your fp32-vs-exact-int measurements already show this.

The clean answer is:

> **Scenario B best construction:** make the served kernel be the exact deterministic integer kernel that is proved. Then `R_rank = 0` at normal Freivalds/sumcheck cost.  
> **Scenario A best cheap construction:** train a rank-distilled exact-integer twin, possibly with a low-rank logit correction head. Expect modest improvement, e.g. `0.35 → 0.25–0.32`, not near zero.  
> **Scenario A near-zero construction:** replay the exact fp8/bf16 kernel accumulation order as an integer circuit / tile-replay proof. This attacks the near-tie nondeterminism, but costs tens to thousands of times more than the Freivalds-friendly exact-int model.

---

# 1. Single best constructions first

## 1. Best overall, Scenario B: serve the exact integer kernel you prove

This is the dominant construction if you are allowed to constrain serving.

### Exact integer object proved

Use a fully specified deterministic integer Transformer.

Two concrete instantiations:

### B1a. No-retrain / minimal model delta: exact fp8-codebook integer kernel

Represent each e4m3 value by an integer codebook value

\[
\phi(c) = 2^9 \cdot \mathrm{e4m3}(c) \in \mathbb{Z},
\]

so `|φ(c)| ≤ 229376`, about signed 19 bits. For each matmul:

\[
S_{ij} = \sum_{k=1}^K \phi(A_{ik}) \phi(W_{kj}) \in \mathbb{Z}.
\]

For Qwen2.5-0.5B dimensions, `K ≤ 4864`, so

\[
|S_{ij}| \lesssim 4864 \cdot 229376^2 < 2^{49},
\]

safe in int64 and in a 61-bit field if you range-check before/after.

Then apply activation/weight scales using deterministic fixed-point rational arithmetic:

\[
Y_{ij} =
\operatorname{Round}_{\mathrm{spec}}
\left(
S_{ij} \cdot \widehat{s}_A \cdot \widehat{s}_W / 2^{18+f}
\right).
\]

Activation quantization is also integer-specified. For dynamic fp8 activation scale `scale = absmax/448`, avoid division in the proof by proving the e4m3 bin with cross-multiplied inequalities:

\[
\mathrm{lo}_q \cdot m
\le
448 x_i
<
\mathrm{hi}_q \cdot m,
\]

where `m = absmax`, and `lo_q, hi_q` are integer bin boundaries for code `q`.

Non-matmul ops should also be served and proved with the same fixed-point/LUT spec:

- RMSNorm: exact int64 sum of squares, reciprocal-sqrt LUT.
- RoPE: fixed-point sin/cos tables.
- SiLU/GELU: LUT or piecewise polynomial.
- Attention softmax: fixed-point exp LUT plus reciprocal LUT.
- Residuals: exact int add with deterministic rescale.
- Final sampling: public fixed-point Gumbel seed, prove `argmax(logit + gumbel)`.

### B1b. Fast production version: int8/int16 activations and int8 weights, int32 accum

Use standard ZK-native quantization:

\[
S = X W,\quad X,W \in [-127,127],\quad S \in \mathbb{Z}^{m \times n}.
\]

Worst Qwen MLP dot:

\[
4864 \cdot 127^2 \approx 7.85 \times 10^7 < 2^{27},
\]

so int32 is safe. Rescale by deterministic integer multiply-shift:

\[
Y_{ij} =
\operatorname{sat}
\left(
\left\lfloor
\frac{S_{ij} \mu_j + b_j + 2^{s-1}}{2^s}
\right\rfloor
\right).
\]

This is the cleanest Freivalds object: exact bilinear matmul plus small lookup/circuit rescale.

### Why `R_rank` drops

If the datacenter serves exactly this kernel and the proof checks the same public Gumbel-max sampler, then the served token is always rank 0 under `M_int`.

\[
R_{\mathrm{rank}} = 0,\qquad \mathrm{frac\_rank0}=1.
\]

This does not “predict” the fp8 near-tie nondeterminism. It removes the mismatch by making the served arithmetic equal to the proven arithmetic.

Compared to the old fixed `torch._scaled_mm` fp8 service, expect token changes at roughly the current disagreement rate, around 6%, concentrated in near-ties. Capability loss is likely small, but this must be evaluated.

### ZK proving cost

This is the cheapest real proof path.

- Matmuls: Freivalds or GKR/sumcheck over exact integer matrices.
- Sumcheck rounds: roughly `log m + log n + log k` per batched matmul relation.
- Extra lookups: one per rescale/round/activation element, not one per MAC.
- No per-FMA floating-point rounding circuit.
- Weight privacy: commit to integer weights/scales; use standard zkLLM-style polynomial/hash commitment openings.

### Scenario

Requires **Scenario B**: served kernel is constrained to the same deterministic integer spec.

### Risks / tradeoffs

- Real tradeoff: changes served arithmetic.
- B1a preserves the fp8 checkpoint most closely but may be slower because e4m3-codebook values are effectively int19/int20, not native int8 tensor-core operands.
- B1b is faster and cleaner but needs QAT/distillation to preserve model quality.
- You must use public/committed sampling randomness. If the server can choose the Gumbel seed after seeing tokens, the seed itself becomes a channel.
- Pinning a deterministic cuBLAS fp8 algorithm is **not enough** unless the proof replays that exact fp arithmetic.

---

## 2. Best Scenario A cheap construction: rank-distilled exact-integer twin

If `M_q` is fixed and cannot be changed, the best Freivalds-friendly construction is not another hand-chosen accumulator. It is a trained integer student whose objective is directly `R_rank`.

### Exact integer object proved

Use the same ZK-native integer architecture as above:

- int8 or int16 activations,
- int8 weights or e4m3-codebook integer weights,
- exact int32/int64 accumulation,
- deterministic integer rescale,
- LUT RMSNorm/softmax/SiLU/RoPE,
- final fixed-point logits.

Then train all private integer weights, scales, and optionally a small logit correction head to mimic the fixed served fp8 model.

A useful final correction head:

\[
\ell_{\mathrm{int}}(v)
=
\ell_{\mathrm{base}}(v)
+
b_v
+
\bigl(U (V h)\bigr)_v,
\]

with `V ∈ Z^{r × d}`, `U ∈ Z^{Vocab × r}`, `r = 32–128`. For Qwen2.5-0.5B, `d ≈ 896`, `Vocab ≈ 152k`; rank 64 adds about

\[
896 \cdot 64 + 64 \cdot 152000 \approx 9.8\text{M}
\]

MACs/token, around 7% of the LM head cost.

Train with a coupled-rank objective. For teacher logits `ℓ_q`, integer-student logits `ℓ_int`, and shared Gumbels `g`:

\[
y_g = \arg\max_v \{\ell_q(v) + g_v\}.
\]

Use a loss like

\[
L =
\mathrm{KL}_{\mathrm{topK}}(\ell_q, \ell_{\mathrm{int}})
+
\lambda
\mathbb{E}_g
\log
\left(
1 +
\sum_{v \ne y_g}
\exp(
\ell_{\mathrm{int}}(v)+g_v
-
\ell_{\mathrm{int}}(y_g)-g_{y_g}
)
\right).
\]

Overweight examples where the teacher top gap is small or where the baseline exact-int model has nonzero rank.

### Why `R_rank` drops

This can learn systematic biases:

- vocabulary bias,
- row/tile bias,
- activation-scale bias,
- systematic fp8 epilogue bias,
- domain-specific near-tie tendencies.

But it does **not** truly attack the accumulation-order nondeterminism unless that residual is learnable from the integer hidden state. Your measurements suggest much of the remaining gap behaves like order-specific near-tie noise.

Reasonable expectations:

- simple temperature/vocab calibration: `0.355 → 0.33–0.35`
- full rank-distilled integer twin: `0.355 → 0.25–0.32`
- heavily domain-specific distillation with low-rank correction: maybe `0.20–0.28`

I would not expect `<0.10` unless the logit residual L2 falls from about `90` to around `20–30` or less on held-out traces.

### ZK proving cost

Very good.

- Same Freivalds/sumcheck matmul proof as the base integer model.
- Full retraining of weights has zero extra proof cost.
- Rank-64 correction head adds one small hidden-to-rank matmul and one rank-to-vocab matmul, both Freivalds-friendly.
- Additional cost roughly 5–15% depending on rank.
- Nonlinearities remain lookup-sized, not MAC-sized.

### Scenario

Works in **Scenario A**. `M_q` is unchanged.

Also usable in Scenario B, in which case serving the distilled integer model gives `R_rank = 0`.

### Risks / failure modes

- Distribution shift: it may learn benchmark-specific tie biases.
- If you train on the exact evaluation transcripts, you can fake low `R_rank`; commitment must precede evaluation/generation.
- If integerized non-matmul ops are coarser than the measured bf16 path, they may add new rank noise.
- It may improve logit L2 without improving near-tie rank unless the loss directly targets coupled rank.

This is the main “free lunch” in Scenario A: it costs nothing to the served fp8 model. But it is not a free lunch for covert capacity; it only removes learnable mismatch.

---

## 3. Best Scenario A absolute construction: blockwise fp8 tensor-core replay

If Scenario A is fixed and you need `R_rank` near zero, you must reproduce the served fp8 arithmetic, including accumulation order. The cheapest nontrivial version is not per-FMA replay; it is block/tile replay.

### Exact integer object proved

Reverse-engineer and freeze the actual `torch._scaled_mm` / cuBLASLt SM89 algorithm:

- tile shape,
- K-block size,
- split-K behavior,
- MMA accumulation tree,
- scale epilogue order,
- bf16 output rounding.

For each GEMM, use e4m3 codebook integers `φ(A), φ(W)` and chunk K into blocks of size `B`, usually `B = 16` or `32` if matching MMA structure.

For each output and chunk:

\[
P^{(c)}_{ij}
=
\sum_{k=cB}^{(c+1)B-1}
\phi(A_{ik}) \phi(W_{kj}).
\]

Prove all `P^{(c)}` by a batched sumcheck/Freivalds relation over `(i,j,c,k)`.

Then replay the fp32 accumulator as integer bitfields:

\[
F^{(0)}_{ij} = 0,
\]

\[
F^{(c+1)}_{ij}
=
\operatorname{FP32AddRN}
\left(
F^{(c)}_{ij},
\operatorname{FP32Round}
\left(
P^{(c)}_{ij} \cdot \mathrm{scale}
\right)
\right),
\]

or move scale to the final epilogue if that is what cuBLASLt does:

\[
Y_{ij}
=
\operatorname{BF16RoundRN}
\left(
F^{(C)}_{ij}
\cdot s_A s_W
\right).
\]

The fp32/bf16 operations are proven as integer circuits/lookups over sign/exponent/mantissa fields.

### Why `R_rank` drops

This directly attacks the measured residual. The mismatch is near-tie sensitivity to a specific floating-point accumulation perturbation. If you replay that perturbation, the tie flips disappear.

Expected:

- exact replay of GEMMs plus exact bf16 non-matmul replay: `R_rank ≈ 0`
- exact GEMM tile replay but approximate non-matmul integerization: maybe `0.02–0.10`
- plausible but wrong tiling/order: can fall back to `0.3+`

### ZK proving cost

Much higher than the ZK-native integer model.

For a matvec-like linear layer with `K × N` weights and chunk size `B`, the number of accumulator transitions is:

\[
N \cdot \lceil K/B \rceil.
\]

For Qwen2.5-0.5B decode with `B = 16`, rough count:

- per layer: about `~1M` fp32-add/round transitions,
- 24 layers: `~24M`,
- LM head: `152k × 56 ≈ 8.5M`,
- total: `~30M` fp32 add/round lookup/circuit rows per token.

Plain exact-int proof has closer to `~0.5M` output/rescale rows per token. So this is plausibly `50–100×` more lookup-heavy, though still far cheaper than per-FMA replay.

### Scenario

Works in **Scenario A** if the served kernel is truly fixed and reproducible.

### Risks

- cuBLASLt/tensor-core microarchitecture is not fully specified.
- Driver, PyTorch, shape, workspace, or algorithm changes can break equality.
- If the fp8 kernel has run-to-run nondeterminism, the realized schedule becomes a witness; allowing that witness can itself create a channel unless it is public/committed.
- Exact bf16 softmax/RMSNorm/residual replay may be needed for true `R_rank = 0`.
- This is a real proof-cost tradeoff, not a Freivalds-free construction.

---

# 2. Fuller ranked list

Ranked mainly by `R_rank` reduction per proving cost.

| Rank | Construction | Scenario | Expected `R_rank` | Attacks near-tie fp nondeterminism? | ZK cost | Main risks |
|---:|---|---|---:|---|---|---|
| 1 | **Serve the proven exact integer kernel**: int8/int32 or e4m3-codebook/int64, deterministic fixed-point nonlinearities, public Gumbel sampler | B | `0` | Yes, by eliminating mismatch | `1×`, Freivalds/GKR-friendly | Changes served arithmetic; quality/latency tradeoff |
| 2 | **Serve exact codebook integer version of the fp8 checkpoint** | B | `0` vs new served kernel; old fp8 disagreement still ~6% | Yes | `1–1.5×`; int64/codebook ranges | Slower than tensor fp8; non-matmul spec must also be integer |
| 3 | **Rank-distilled integer twin with optional low-rank logit correction** | A | `0.25–0.32`, optimistic domain-specific `0.20–0.28` | Partly; learns systematic bias only | `1.05–1.15×` with rank-64 head | Overfit, distribution shift, cannot remove chaotic order residual |
| 4 | **Cheap affine calibration**: global temperature, per-vocab bias, per-output-row/tile bias | A | `0.33–0.35` | Barely | Almost free | Likely tiny improvement; can flip non-ties |
| 5 | **Larger verifier-only integer student**: 2–4× width/rank, trained only to mimic `M_q` | A | maybe `0.15–0.28` if domain is narrow | Partly | `2–4×` | Proof cost grows; can become memorization; no guarantee |
| 6 | **Blockwise fp8 tile replay**: exact integer partial sums plus fp32 accumulator-state circuit | A | `0–0.10` if tiling exact; otherwise `0.3+` | Yes | `50–100×` lookup-heavy | Reverse-engineering, driver drift, huge memory |
| 7 | **Full exact fp8/bf16 replay / zkVM / Hawkeye path** | A | `0` if bitwise exact | Yes | `10^3–10^5×` vs Freivalds model | Per-FMA fp circuits; exact softmax/RMSNorm kernels too |
| 8 | **Other exact-int accumulator choices**: pairwise, Kahan, fp32 different order, exponent-binned surrogate | A | likely `0.33–0.36` | No, unless equal to served order | `1–20×` | Your measurements already show no gain |
| 9 | **Verifier tolerance / top-K acceptance** | A/B | Does not reduce true `R_rank` | No | Maybe cheap | Not a free lunch: accepted set becomes covert channel |

---

## Important non-recommendations

### 1. Do not spend effort on “better” exact-int accumulation unless it matches the served order

You already measured:

- exact codebook integer: `R_rank = 0.355`
- fp32 same operands, different order: `R_rank = 0.361`
- exact-int vs fp32 disagree by `0.328`

That is the signature of near-tie sensitivity, not a bad integer calibration. A new deterministic order is just another perturbation unless it is the actual `torch._scaled_mm` order.

### 2. Pinning a deterministic fp8 CUDA kernel is insufficient

It may make serving reproducible, but the proof still cannot use Freivalds because rounded fp32 accumulation is not bilinear. You either:

- prove the exact fp arithmetic, expensive, or
- change serving to exact integer accumulation.

### 3. Tolerance is not capacity reduction

Accepting “rank ≤ K” or “within ε logits” may reduce rejection, but it explicitly leaves a choice set. The covert channel is then approximately the entropy of that choice set. It is policy accounting, not a lower `R_rank` model construction.

---

# Practical recommendation

If Scenario B is allowed:

1. First deploy the **canonical exact integer codebook kernel** corresponding to your current fp8 checkpoint. This gives `R_rank = 0` with minimal model-change risk.
2. If throughput is poor, train a **QAT int8/int32 kernel** and serve/prove that. Also `R_rank = 0`.
3. Use public Gumbel-max sampling and prove the sampled argmax.

If Scenario A is mandatory:

1. Keep the simplest exact-int Freivalds-friendly model as baseline.
2. Add rank-aware distillation and a small low-rank/vocab correction head.
3. Expect modest improvement, not zero.
4. If the target is `<0.1 bits/token`, build the blockwise fp8 tile-replay proof prototype; anything cheaper is unlikely to overcome the measured near-tie accumulation-order gap.