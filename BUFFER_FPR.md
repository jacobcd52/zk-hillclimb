> **See also `THRESHOLD_CURVE.md`** — extends this analysis to the operational
> **THRESHOLD(N) = μ + buffer(N)/N** per-token treaty-ceiling curve for **BOTH schemes
> overlaid** (faithful μ≈0.31 + codebook μ≈0.19, each relaxing toward its own μ), adds
> the **√N variance-scaling fit** and the **"N to reach within 10 %/1 % of μ at FPR
> 1e-10"** numbers (≈1.8×10⁵ / ≈1.8×10⁷ tokens, near scheme-independent), and generates
> the **120 codebook benign prompts** that this faithful-only doc lacked.

# Per-token afforded capacity, autocorrelation, and the buffer-vs-FPR-vs-audit-size curve

**Goal.** For the capacity-budget treaty (a ZKP proves the integer model M_int's
post-Gumbel logits; network taps observe the *served* FP8 tokens; the verifier bounds the
worst-case covert capacity **afforded** per token and tears up the treaty when the
cumulative budget exceeds a threshold), determine how large a **buffer** above the benign
mean is needed so an **honest** datacenter is wrongly shut down with probability
**< 1e-10**, as a function of how many tokens have been audited.

All numbers use the **CORRECTED faithful orientation** (REFERENCE = faithful M_int's
post-Gumbel logits; SERVED token = FP8 fast-model post-Gumbel argmax; the realistic
deployed case — same construction as `CAPACITY_CORRECTED.md`). Seed 20260611 for the
original 8 prompts; **120 additional distinct dolly prompts were generated** through the
identical pipeline to push the analysis from 8 192 → **131 072 tokens / 128 prompt-blocks**
(flagged per the brief; see §6 *Honest caveats* and *Reproduce*).

> **Decoding regime + temperature (clarification, 2026-06-13).** Capacity is measured in the
> **verifiable sampled-decoding regime** (served token = `argmax_v(M_int/FP8 logits + T·g_σ)`,
> `g_σ` a public *committed*-seed Gumbel draw; committing the seed closes the sampling channel).
> **All buffer numbers here are at T = 1.** Greedy is the T→0 limit; per `CAPACITY_TEMPERATURE.md`
> the faithful worst-case capacity is ~T-insensitive (0.38–0.45 bits/tok over T ∈ [0.05, 2.0]) and
> does not vanish at greedy, so the benign-mean μ and buffer curve below are not greedy-only and
> change little with T. (See `THREAT_MODEL_NOTES.md §0` for the regime reconciliation.)

---

## Bottom line (plain language)

At the **calibrated optimum** (top-K **K\*=4**, threshold **b\*=0.347**; benign mean
**μ = 0.306 bits/token**):

- **To hold FPR < 1e-10 you need a buffer of:**
  - **rigorous (Bernstein, independent-prompt):** ≈ **3.9 k bits** after auditing 16 k
    tokens, **4.9 k** after 65 k, **12.3 k** after 1 M tokens;
  - **validated tail-fit estimate (tighter):** ≈ **1.9 k bits** after 16 k tokens,
    **3.3 k** after 65 k tokens.
- **Per-token buffer** (buffer ÷ N) falls like **~1/√N**: from > μ at small audits down to
  **≈ 0.05 bits/token at 65 k tokens** (tail-fit) and **≈ 0.012 bits/token at 1 M tokens**
  (rigorous Bernstein).
- **Per-token buffer drops below the benign mean μ once N ≳ 4 k tokens (tail-fit) /
  N ≳ 16 k tokens (rigorous Bernstein)** — i.e. after roughly **4–16 distinct
  prompts/conversations**. Below that the buffer dwarfs the signal and *a single-prompt
  audit cannot certify anything at 1e-10*.
- The **conservative K=16** setting (μ = 0.330 bits/tok) gives essentially the same curve
  (buffers ~5–10 % larger; same crossover N).

The single most important correction to "assume i.i.d." is **between-prompt
heterogeneity**: the variance of the cumulative budget grows ≈ **7.5× (K=4) / 9× (K=16)**
faster than an i.i.d.-token model predicts, so the effective sample size is **N_eff ≈ N/7.5**,
not N. The buffers above already incorporate this.

---

## 1. Setup, the optimum, and the per-token decomposition

The swept worst-case per-token capacity is

```
C(b,K) = H(p) + (1-p)·E[log2 N_b]  +  p·( H(q) + (1-q)·log2 K + q·log2(V-K) )
```

with p = fraction of served tokens outside margin b under M_int (violations),
q = fraction of violations whose served token is outside M_int's top-K, N_b(t) = #vocab
within margin b of M_int's preferred post-Gumbel score, V = 32000.

**Calibrated optimum (confirmed from the data).** Jointly minimising over (b, K) on the
128-prompt set gives **K\*=4, b\*=0.347, C = 0.306 bits/token** (the K-sweep is U-shaped
with its floor at K=4, exactly as in `CAPACITY_CORRECTED.md`; the "b\*~0.6" in the brief is
the K=1/simple-rule optimum). We also report the **conservative K=16, b\*=0.465,
C = 0.330 bits/token** carried forward as the headline in the capacity reports. (These
128-prompt means sit just below the published 8-prompt optima 0.337 / 0.356 — within
prompt-sampling variation, a cross-check that the extra prompts are statistically the same
benign population.)

**Per-token afforded capacity r_t.** Decompose C into a per-token contribution:

```
r_t = overhead  +  s_t ,        overhead = H(p) + p·H(q)   (a per-token CONSTANT)

s_t = log2 N_b(t)     if margin_t ≤ b            (compliant: within-margin multiplicity)
    = log2 K          if violation and cand_rank_t <  K   (violate into M_int's top-K)
    = log2(V-K)       if violation and cand_rank_t ≥  K   (violate into the tail)
```

`mean_t(r_t) = C(b,K)` exactly. **Amortization of the "which-positions/which-bucket
violate" message entropy:** the H(p) term (and the small p·H(q) term) is distributed as an
equal per-token **constant** `overhead`, computed from the message's *empirical* p (and q).
Because it is a constant it shifts the mean to exactly C but **contributes nothing to the
variance or the buffer** — the buffer is driven entirely by the stochastic part `s_t`.
`overhead = 0.106 bits/tok` (K=4) and `0.067` (K=16). `log2 N_b(t)` is taken at b\* by the
same per-position linear interpolation between grid points used by the capacity sweep.

`R(N) = Σ_{t=1}^{N} r_t` is the cumulative afforded budget; the treaty tears up when
`R(N) > R*(N) = N·μ + buffer(N)`.

---

## 2. The per-token r_t distribution

(left panel of **`buffer_rt_dist_acf.png`**; `s_t` is r_t minus the constant overhead)

| quantity | **optimum K=4 (b\*=0.347)** | **conservative K=16 (b\*=0.465)** |
|---|---|---|
| mean μ_s = C − overhead | 0.200 | 0.262 |
| std σ | 0.546 | 0.596 |
| variance σ² | 0.298 | 0.355 |
| **fraction exactly 0** | **83.7 %** | **79.6 %** |
| max | **14.97** (= log2(V−4)) | **4.00** (= log2 16) |
| skewness | 8.7 | 3.0 |
| excess kurtosis | 197 | 12 |
| p99 / p99.9 / p99.99 | 2.00 / 2.32 / 14.97 | 2.32 / 4.00 / 4.00 |
| violations / tail-violations (of 131072) | 1771 / **48** | 1049 / **0** |

**Heavily zero-inflated:** ~80 % of tokens afford **0 bits** (the served FP8 token *is*
M_int's preferred token and is alone within margin, N_b=1 → log2 1 = 0). The nonzero mass is
small multiplicities (log2 2 = 1, log2 3 = 1.58, …) plus the violation payloads.

**The tails differ sharply between the two settings — this drives everything downstream:**

- **K=4 (optimum)** has a **heavy discrete tail**: 48 of the 131 072 tokens are
  *tail-violations* (served token at M_int rank ≥ 4) worth **log2(V−4) ≈ 14.97 bits each**
  — a ~15-bit spike at rate **3.7e-4 per token**. These spikes (skew 8.7, excess kurtosis
  **197**) dominate the upper tail of R(N).
- **K=16 (conservative)** has **no tail-violations at all** (max served rank over all
  131 072 positions is 9 < 16, so q = 0): every violation pays exactly **log2 16 = 4 bits**,
  and s_t is **bounded by 4**. Lighter tail (excess kurtosis 12), but a higher mean. So K=16
  is conservative *in the mean* yet *better-behaved in the tail* — a genuine trade-off the
  buffer analysis must weigh, not a strict ordering.

---

## 3. Autocorrelation along sequences, τ and N_eff

(right panel of **`buffer_rt_dist_acf.png`**)

Computed **within prompt-blocks only** (each tiled dolly prompt is one block; pairs never
cross block boundaries). Two distinct quantities:

- **Token-level integrated autocorrelation time** (Sokal adaptive window):
  **τ_acf ≈ 1.15 (K=4) / 1.20 (K=16)**. Consecutive tokens' afforded-capacity contributions
  are **nearly uncorrelated at short lag** — the ACF drops from 1 to ≈ 0.01 at lag 1 and
  stays at a **small positive plateau (~0.01) that does not decay**. The capacity signal is
  sparse and event-driven, so the prompt's token-level periodicity (the prompts are tiled,
  §6, tiling) does **not** translate into strong short-lag autocorrelation.

- **Session-level variance inflation** (the operative correction):
  **τ_var = Var(Y_session) / (L·σ²) = 7.47 (K=4) / 9.09 (K=16)**, where Y_session is the
  sum of s_t over a full 1024-token prompt-block (L = 1024). This is the real non-i.i.d.
  effect and it is **large**.

**Why they disagree, and which one matters.** The non-decaying ~0.01 ACF plateau is the
signature of **between-prompt heterogeneity**: a variance decomposition gives per-prompt
mean afforded-rate ranging from **0.091 to 0.422** (K=4; ~4.6× spread), and
`Var(Y_session) = L²·Var(rate)` almost exactly — i.e. essentially **all** of the
session-sum variance is the prompt-to-prompt difference in difficulty, not within-sequence
lag structure. A short-window estimator (τ_acf) misses the tiny plateau; summed over a full
1024-token session it inflates the variance 7–9×. **For the buffer the operative
correction is τ_var**, giving

> **N_eff ≈ N / τ_var ≈ N / 7.5 (K=4), N / 9 (K=16).**

(Caveat — this τ_var is partly a **tiling artifact** and is therefore *conservative* for a
per-prompt audit; see §6.3.)

---

## 4. Var(R(N)) vs N — i.i.d. vs correlation-corrected

(**`buffer_varRN.png`**) Empirical Var(R(N)) of length-N partial sums (within-block windows
for N ≤ L) vs the i.i.d. line N·σ²:

| N | Var(R(N)) K=4 | N·σ² (i.i.d.) | ratio | Var(R(N)) K=16 | N·σ² | ratio |
|---|---|---|---|---|---|---|
| 64 | 31.5 | 19.1 | 1.65 | 44.0 | 22.7 | 1.93 |
| 256 | 259 | 76 | 3.40 | 395 | 91 | 4.34 |
| 1024 | **2279** | 305 | **7.47** | **3307** | 364 | **9.09** |

Var(R(N)) grows **super-linearly within a prompt** — the inflation ratio climbs monotonically
1.65 → 3.40 → 7.47 (K=4) as the window lengthens, because each added token accumulates the
prompt's systematic difficulty offset — reaching the **7.5× / 9× inflation** at the full
session length. Across **independent sessions** (distinct prompts/conversations) variance is
linear again with the corrected slope:

> **Var(R(N)) = (N/L)·Var(Y_session) = N · σ² · τ_var**  (the model used for all buffers below).

This is the realistic operational model: an audit stream is a concatenation of
near-independent sessions, with strong correlation *inside* each session and independence
*across* them.

---

## 5. Buffer(N) for FPR ≤ 1e-10 — two methods + bootstrap validation

We want `buffer(N)` with `P(R(N) − N·μ > buffer) ≤ ε`, ε = 1e-10, for honest traffic.
With only ~131 k real tokens we **cannot observe 1e-10 events directly**, so we use both a
rigorous bound and a validated extrapolation, and cross-check both against a block-bootstrap
where the FPR is observable.

### (a) Rigorous concentration bound (Bernstein, independent prompts)

Write `R(N) − N·μ = Σ_{b=1}^{B}(Y_b − μ_Y)` over **B = N/L independent sessions**. Bernstein
for independent bounded summands:

```
P(R(N) − N·μ > t) ≤ exp( − t² / ( 2·(B·Var(Y) + M·t/3) ) )      ⇒
buffer = (ln(1/ε)·M/3) + sqrt( (ln(1/ε)·M/3)² + 2·ln(1/ε)·B·Var(Y) )
```

with empirical session variance `Var(Y)` (= 2279 / 3307 bits², K=4/K=16) and a bound `M` on
the session-sum deviation `|Y_b − μ_Y|`. **Rigorous given (i) sessions independent (distinct
prompts), and (ii) `|Y_b − μ_Y| ≤ M`.** We set `M` = the **observed** max session deviation
(`M ≈ 227 / 210` bits) — the data never exceeds it; this is an *empirical-support* bound, not
an a-priori one. The fully a-priori bound (an honest session could in principle be 1024 tail
violations, `M = L·log2(V−K) ≈ 15 000`) is **vacuous (~2.3e5 bits, flat in N)** — nothing a
priori forbids a pathological-but-honest session, which is exactly why an audit must span
many prompts. We therefore quote the **empirical-support Bernstein** as the practical
rigorous buffer.

### (b) Parametric tail fit (Gaussian and sub-exponential), extrapolated to 1e-10

Fit the **block-bootstrap** distribution of R(N) (resample sessions with replacement, sum).
- **Gaussian:** buffer = z(ε)·√(B·Var(Y)), z(1e-10) = 6.36.
- **Sub-exponential:** fit `ln S(t) ≈ c − t/λ` to the empirical bootstrap survival curve in
  the observable window [1e-5, 1e-2], extrapolate to S = ε. λ ≈ 131 (K=4) / 145 (K=16) bits
  at N = 65 k — the heavier-than-Gaussian decay scale.

### Validation against the block-bootstrap (the FPRs we *can* see)

(**`buffer_tail_validation.png`**) Deep bootstrap (4×10⁷ resamples) at **N = 4096**, buffer
in bits:

| ε (observable) | empirical | sub-exp fit | Gaussian |
|---|---|---|---|
| 1e-4 | 494 | 485 (−2 %) | 355 (**−28 %**) |
| 1e-5 | 580 | 592 (+2 %) | 407 (**−30 %**) |
| 1e-6 | 683 | 699 (+2 %) | 454 (**−34 %**) |
| 1e-7 | 730 | 807 (+11 %) | 496 (**−32 %**) |

**The sub-exponential fit tracks the empirical tail to within a few % out to 1e-6 and is
slightly conservative; the Gaussian underestimates the buffer by ~30 % and *worsens* deeper
into the tail** — using a Gaussian buffer would silently raise the true FPR above target.
**We therefore take the sub-exponential fit as the trustworthy estimate, the Gaussian as a
known-unsafe lower bound, and Bernstein as the rigorous upper bound.** The gap between the
rigorous Bernstein and the validated sub-exp estimate (≈ 1.3–1.5× at N ≥ 65 k) is the price
of demanding rigor over trusting the validated tail model.

### Buffer table (bits) for ε = 1e-10

(**`buffer_vs_N.png`**, left) — optimum **K=4** (μ = 0.306):

| N (tokens) | Bernstein (rigorous) | sub-exp (estimate) | Gaussian (unsafe) |
|---|---|---|---|
| 1 024 | 3 516 | 1 497 † | 304 |
| 4 096 | 3 603 | 1 115 | 607 |
| 16 384 | 3 915 | 1 879 | 1 215 |
| 65 536 | 4 867 | 3 347 | 2 429 |
| 262 144 | 7 212 | 6 224 | 4 859 |
| 1 048 576 | 12 255 | — ‡ | 9 717 |

conservative **K=16** (μ = 0.330): 985 † / 3 271 (1 k), 1 103 / 3 403 (4 k), 2 036 / 3 856
(16 k), 3 735 / 5 126 (65 k), 7 527 / 8 061 (262 k), — / 14 204 (1 M) [sub-exp / Bernstein].

† sub-exp at N ≤ 2 048 (≤ 2 sessions) is unreliable — the bootstrap has only 128 discrete
session values, no genuine continuum; use Bernstein there. ‡ sub-exp not fit beyond
N = 262 144 (bootstrap cost); the Bernstein/Gaussian envelope brackets it.

### Per-token buffer (buffer ÷ N) and the crossover

(**`buffer_vs_N.png`**, right) — falls as ~1/√N:

| N | K=4 sub-exp /N | K=4 Bernstein /N | μ |
|---|---|---|---|
| 4 096 | 0.272 | 0.880 | 0.306 |
| 16 384 | 0.115 | 0.239 | 0.306 |
| 65 536 | **0.051** | 0.074 | 0.306 |
| 1 048 576 | — | 0.012 | 0.306 |

**Per-token buffer drops below the benign mean μ once N ≳ 4 096 tokens (sub-exp estimate) /
N ≳ 16 384 tokens (rigorous Bernstein)** — about **4–16 distinct prompts**. K=16 crosses at
the same N (4 096 / 16 384). For N below ~4 k the buffer *exceeds* μ·N many-fold: a treaty
that audits only a handful of prompts cannot distinguish honest from adversarial at 1e-10.

---

## 6. Honest caveats

1. **Afforded ≠ detected ≠ transmitted.** r_t is the *worst-case* capacity an adversary
   holding the exact model *could* embed while perfectly mimicking the honest margin/rank
   profile — a **noiseless-channel ceiling**, not an expected or realized leak (Rinberg:
   realized ≪ theoretical, < 0.5 %). The buffer protects an **honest** datacenter from false
   teardown; it does **not** by itself bound a real adversary's throughput.

2. **How much real data we have vs would want.** 131 072 tokens across **128 distinct dolly
   prompts** (8 original + 120 generated here). Because within-prompt tokens are highly
   correlated (§3), the **effective independent sample size is ~128 prompt-blocks**, not
   131 k. That suffices to estimate the bulk and the tail to ~1e-6 by bootstrap, but
   **everything past 1e-6 is extrapolation**. The sub-exp fit is validated *to* 1e-6 and
   assumed to continue log-linearly to 1e-10; if the true honest tail is heavier than
   exponential past 1e-6 the estimate is low (the rigorous Bernstein still holds). More
   prompts (cheap: ~5–9 s each — we generated 120 in ~15 min) would push the observable FPR
   lower and tighten λ and Var(Y); the leverage is in **more distinct prompts**, not more
   tokens per prompt.

3. **Tiling artifact inflates τ_var (conservative).** Each prompt is one short dolly example
   **repeated** to 1024 tokens (periods 18–1116). So a session's 1024 tokens are ~one
   prompt's difficulty repeated, not 1024 fresh contexts — this *amplifies* the
   between-session heterogeneity and makes **τ_var ≈ 7.5–9 an upper bound** on what
   non-repeated natural traffic would show within a 1024-token window. The buffers are
   therefore **conservative** on the autocorrelation axis. The genuine, deployment-relevant
   component is **conversation-to-conversation** difficulty variation, which is real and
   persists.

4. **Bernstein rigor rests on two assumptions:** session independence (true for our
   distinct-prompt draw; ~true for distinct conversations in deployment) and the
   session-deviation support bound M. We use the **empirical** M (observed max); a single
   honest session heavier than anything in 128 observed prompts would need a larger M. The
   a-priori bound that assumes nothing is vacuous (§5a) — meaning *the protocol's safety
   genuinely depends on auditing enough independent prompts that no single session dominates
   the budget*.

5. **q = 0 / small-sample tail at K=16.** q = 0 (no tail-violations) now rests on 1049
   violations with max served rank 9 < 16 (vs 59 violations in the published 8-prompt run) —
   much firmer. At K=4 there are 48 genuine tail-violations (rank ≥ 4), correctly priced at
   log2(V−4); these are the K=4 heavy tail and are fully included.

6. **Inherited:** fixed public seed (not an official round), Gumbel temperature 1.0,
   unclamped margins, V = 32000; bf16/FP8 construction byte-identical to
   `CAPACITY_CORRECTED.md`. `int-model-approximation` used **read-only**; nothing committed
   or pushed.

---

## Reproduce

```bash
cd /workspace/projects/zk-hillclimb/buffer
# 1. generate 120 additional DISTINCT benign prompts (faithful corrected orientation)
IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python gen_more_faithful.py \
    --nprompts 120 --rowseed 20260612 --gseed-base 30000000 \
    --out faithful_extra_corrected.npz
# 2. full buffer / FPR / autocorrelation analysis + plots + buffer_results.json
/root/int-model-env/bin/python analyze_buffer.py
```

Inputs: `../capacity/capacity_dump_corrected_faithful_seed20260611.npz` (the published
8-prompt dump) + `faithful_extra_corrected.npz` (the 120 new prompts, rows disjoint from the
original 8). Outputs: `buffer_results.json` (all numbers), and in the parent dir
`buffer_rt_dist_acf.png`, `buffer_varRN.png`, `buffer_vs_N.png`,
`buffer_tail_validation.png`. The capacity formula / optimum machinery is imported
byte-identically from `capacity/topk_breakdown.py`; the new code only adds the per-token
decomposition, autocorrelation, variance, and buffer estimators.
