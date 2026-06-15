# The treaty-threshold curve: per-token capacity ceiling vs audit size, FPR ≤ 1e-10

**The operational plot.** For the capacity-budget treaty (a ZKP proves the integer
model M_int's post-Gumbel logits; a network tap observes the *served* FP8 tokens; the
verifier bounds the worst-case covert capacity **afforded** per token and tears up the
treaty when the cumulative budget exceeds a ceiling), this is the curve an auditor
actually reads off:

> **THRESHOLD(N) = μ + buffer(N)/N**  — the per-token average-capacity ceiling
> (bits/token) after auditing **N** tokens. A datacenter whose **measured mean
> afforded capacity** over those N tokens exceeds THRESHOLD(N) is torn up. buffer(N) is
> sized so an **honest** datacenter trips it with probability **≤ 1e-10**. As N grows,
> buffer(N)/N falls ≈ 1/√N and **THRESHOLD(N) relaxes down toward the scheme's benign
> mean μ** — audit more, and you can hold a tighter line without false teardowns.

**`threshold_curve.png`** — the deliverable: a **single panel** at the **K=4 optimum**,
overlaying **FAITHFUL** (μ≈0.31) and **CODEBOOK** (μ≈0.19), each curve approaching its
own μ. Each scheme's curve is drawn **solid in the OBSERVED region** (N ≤ N_data, where
the variance is measured directly from the real tokens) and **dashed in the
EXTRAPOLATED region** (N > N_data, where the variance is projected by the fitted 1/N
law), with a vertical line at the **data limit N_data = 131 072 tokens = 128 prompts**.

This extends `BUFFER_FPR.md` (faithful only) to **both schemes** and to **more benign
tokens**: **131 072 tokens / 128 distinct dolly prompt-blocks per scheme** (8 original +
**120 newly generated**). The 120 codebook prompts were generated for this analysis
(`gen_more_codebook.py`, identical workload to the faithful extra: same 120 dolly rows,
same tiling, same per-prompt Gumbel seed, so the FP8-served token stream is
byte-identical and only the reference M_int differs).

> **Decoding regime (per `THREAT_MODEL_NOTES.md` §0/§1).** This is the **benign-workload
> capacity arising from the M_int-vs-served-model gap** in the verifiable **sampled**
> regime: reference = M_int post-Gumbel logits, served token = the fast FP8 model's
> post-Gumbel argmax under a **shared committed-seed** Gumbel draw, metric temperature
> **T = 1**. It is **not** a greedy number; per `CAPACITY_TEMPERATURE.md` the faithful
> worst-case capacity is ~T-insensitive, so μ and the curve change little with T.

---

## Bottom line (plain language)

- **Each scheme has its own benign mean μ** (128-prompt sample, K=4 optimum):
  - **FAITHFUL:** μ = **0.306** bits/tok (b*=0.347).
  - **CODEBOOK:** μ = **0.191** bits/tok (b*=0.220).
  - Codebook's μ is **~37 % lower** than faithful's — codebook tracks the FP8 teacher
    more tightly post-Gumbel (96.3 % exact agreement vs 94.2 %), so the honest
    datacenter affords less covert capacity and the treaty can hold a lower line.
- **The threshold relaxes toward μ as you audit more** (per-token, FPR ≤ 1e-10). The
  numbers below are **consistent with the figure**: solid where the variance is
  **observed** (N ≤ 131 072), dashed where it is **extrapolated** (N > 131 072):

  | N (tokens) | FAITHFUL K=4 | CODEBOOK K=4 | region |
  |---|---|---|---|
  | 1e4   | 0.435 | 0.269 | observed |
  | 1e5   | 0.347 | 0.215 | observed |
  | 1e6   | 0.319 | 0.198 | extrapolated |
  | 1e7   | 0.310 | 0.193 | extrapolated |
  | 1e8   | 0.307 | 0.191 | extrapolated |
  | →∞    | → μ = 0.306 | → μ = 0.191 | (asymptote) |

- **To bring the threshold within X % of μ at FPR 1e-10** (from the √N variance-scaling
  fit; tokens, rounded):
  - **within 10 % of μ:** ≈ **1.7–1.8×10⁵ tokens** (~170 prompts) — *both* schemes.
  - **within 1 % of μ:**  ≈ **1.6–1.8×10⁷ tokens** (~16 000 prompts) — *both* schemes.
  - It is **nearly scheme-independent** because the buffer coefficient scales with μ:
    `c_subexp/μ ≈ 42` (faithful) ≈ `41` (codebook), and
    `N_within(frac) = (c/(frac·μ))²`. So "within 10 %" ≈ (420)² ≈ 1.8×10⁵ and
    "within 1 %" ≈ (4200)² ≈ 1.8×10⁷ for **either** scheme.
- **A single-prompt audit certifies nothing at 1e-10.** Below N ≈ 4 k tokens the buffer
  dwarfs μ (THRESHOLD ≫ μ); the operational regime is many distinct prompts.

---

## 1. Per-token afforded capacity r_t (definition — same as `BUFFER_FPR.md`)

Per token, at the scheme's calibrated (b*, K*):

```
r_t = overhead + s_t ,   overhead = H(p) + p·H(q)   (per-token CONSTANT → shifts mean to μ, no variance)
s_t = log2 N_b(t)   if margin_t ≤ b*           (compliant: #vocab within margin b* of M_int's preferred score)
    = log2 K*       if violation, served rank < K*  (violate into M_int's top-K)
    = log2(V−K*)    if violation, served rank ≥ K*  (violate into the tail),  V = 32000
```

`mean_t(r_t) = C(b*,K*) = μ` exactly. The buffer is driven entirely by the stochastic
`s_t`; the `overhead` constant contributes nothing to it. **(b*, K*)** are each scheme's
own published K=4 optimum (the U-shaped K-sweep floor) from
`capacity/topk_corrected_results_seed20260611.json`. Machinery imported byte-identically
from `analyze_buffer.py` / `capacity/topk_breakdown.py`; **verified to reproduce the
published 8-prompt optima exactly** (faithful K4 0.3367, codebook K4 0.2203).

**Why the 128-prompt μ sits below the 8-prompt headline** (faithful 0.337→0.306,
codebook 0.220→0.191): the 120 additional prompts are, on average, slightly *easier*
(higher post-Gumbel agreement) than the original 8 — the **same downward drift in both
schemes**, a cross-check that the extra prompts are the same benign population sampled
deeper, not a code change. These larger-sample μ are the honest operating means.

---

## 2. The per-token / per-session distribution (128 prompts/scheme, K=4)

| quantity | FAITHFUL K=4 | CODEBOOK K=4 |
|---|---|---|
| μ (bits/tok) | 0.306 | 0.191 |
| σ²(s_t) per-token | 0.298 | 0.158 |
| violations / 131072 | 1771 | 991 |
| **tail-violations** (rank ≥ K) | **48** | **12** |
| session-sum Var(Y), L=1024 | 2279 | 933 |
| max session deviation M (bits) | 227 | 93 |

**K=4 carries a heavy ~15-bit discrete tail** (tail-violations priced at
log2(V−4)≈14.97) — 48 in faithful, **12 in codebook**. Codebook's tail is lighter than
faithful's (12 vs 48 spikes, lower σ²) — its variance and thus its buffer are smaller in
absolute bits. *(A conservative K=16 variant — q=0, no tail-violations, s_t bounded by 4
bits, μ slightly higher — is retained in `threshold_results.json`; it is dropped from
the figure here, which shows the K=4 optimum only.)*

---

## 3. The buffer, decomposed: observed variance × a single validated tail multiplier

This is the **key methodological framing** of the new figure. The FPR-1e-10 buffer is
written as **variance × a single tail multiplier**, so the part we can *measure* and the
part we must *model* are cleanly separated:

> **buffer_per_token(N) = z_eff · √Var(R̄_N)** ,  **THRESHOLD(N) = μ + z_eff · √Var(R̄_N)**

where **R̄_N = R(N)/N** is the N-token *average* capacity and **Var(R̄_N) = Var(R(N))/N²**.

### 3a. The variance Var(R̄_N) — OBSERVED, then EXTRAPOLATED (the solid/dashed split)

The operative non-i.i.d. effect is **between-prompt heterogeneity**: across independent
sessions (distinct prompts) the cumulative-budget variance is linear with an
autocorrelation/heterogeneity-**inflated** slope,

> **Var(R(N)) = (N/L)·Var(Y),   N_eff = N/τ_var,   τ_var = Var(Y)/(L·σ²).**

| | FAITHFUL K=4 | CODEBOOK K=4 |
|---|---|---|
| **τ_var** (= N/N_eff) | 7.47 | **5.76** |

- **OBSERVED region — N ≤ N_data = 131 072 tokens (128 prompts).** Var(R̄_N) is
  **measured directly from the real tokens**: within a prompt (N ≤ L) from the empirical
  variance of length-N within-block window sums; across prompts from the empirical
  between-session variance Var(Y). The directly-bootstrapped between-session cumulative
  variance **tracks the (N/L)·Var(Y) law to ~1 %** out to all 128 sessions (measured /
  law ratio = 0.993 for *both* schemes — see the run printout), so the line is anchored
  in data across the whole solid segment.
- **EXTRAPOLATED region — N > N_data.** The **same** law Var(R(N)) = (N/L)·Var(Y) is
  continued to **more sessions than were measured**, using the established
  autocorrelation-corrected slope Var(Y)/L = σ²·τ_var. This is the dashed segment.

**Codebook's heterogeneity is milder** (τ_var ≈ 5.8 vs faithful 7.5) — its per-prompt
difficulty varies less, so its effective sample size is larger per token. (τ_var is
partly a **tiling artifact** — each prompt is one short dolly example repeated to 1024
tokens — and is therefore a *conservative* upper bound on what non-repeated natural
traffic would show within a session; the genuine, persistent component is
conversation-to-conversation difficulty variation. See `BUFFER_FPR.md` §3/§6.3.)

### 3b. The tail multiplier z_eff — the single validated sub-exp model (ALWAYS a model)

Converting a variance into an FPR-1e-10 buffer needs a tail shape. We use **one**
multiplier — the **bootstrap-validated sub-exponential tail** — in **both** regions:

> **z_eff = c_sub / √(Var(Y)/L)** , with **c_sub** the sub-exp √N coefficient (bits·√tok).

| | FAITHFUL K=4 | CODEBOOK K=4 |
|---|---|---|
| **z_eff (sub-exp, validated)** | **8.65** | **8.21** |
| z (Gaussian, known-unsafe floor) | 6.36 | 6.36 |
| c_sub (bits·√tok) | 12.9 | 7.8 |

The validated heavy tail gives **z_eff ≈ 8.2–8.7**, well above the Gaussian z=6.36 — the
honest tail is heavier than Gaussian. With this, **buffer_per_token = z_eff·√Var(R̄_N) =
c_sub/√N** exactly, reproducing the validated √N curve.

**Validation at observable FPRs (both schemes).** Deep block-bootstrap (4×10⁷ resamples)
at N=4096, sub-exp vs empirical at the FPR the bootstrap can see:

| scheme (K=4) | ε | empirical | sub-exp | Gaussian |
|---|---|---|---|---|
| FAITHFUL | 1e-6 | 683 | 702 (**+3 %**) | 454 (**−34 %**) |
| CODEBOOK | 1e-6 | 317 | 346 (**+9 %**) | 290 (**−9 %**) |

For **both** schemes the **sub-exp fit is slightly conservative** (tracks the empirical
honest tail to within +3…+9 %) while the **Gaussian underestimates** (a known-unsafe
floor). Codebook's Gaussian error is milder (−9 % vs faithful's −34 %) because codebook's
tail is **lighter** (12 vs 48 tail-violations) — closer to Gaussian, so the sub-exp fit
is if anything *more* reliable for codebook. This is the fit that, in `BUFFER_FPR.md`,
was validated against a 4×10⁷-resample bootstrap to 1e-6; we carry that validation here.

### 3c. Honesty: "observed" means the VARIANCE, not the 1e-10 event

**1e-10 is never directly observed** — seeing it would need ~10¹⁰ honest samples; we
have ~10⁵. What *is* observed is the **variance** Var(R̄_N), out to N_data. The 1e-10
buffer multiplier z_eff is **always** the validated sub-exp tail model, in the solid
region as much as the dashed one. So the solid/dashed split is purely *"was the variance
measured directly (solid) or projected by the law (dashed)"* — it is **not** a claim
that 1e-10-rare events were observed anywhere on the curve.

---

## 4. Bernstein — the rigorous conservative cross-check (text only, not plotted)

The earlier two-line figure overlaid a **rigorous Bernstein** buffer alongside the
sub-exp fit; the second line was found confusing, so it is **moved here as a text
cross-check** and **dropped from the plot**. Bernstein on B = N/L **independent
sessions**, with empirical session variance Var(Y) and the **empirical-support**
deviation bound M (observed max session deviation):

> `buffer = a + √[a² + 2·ln(1/ε)·(N/L)·Var(Y)]` ,  `a = ln(1/ε)·M/3` ,  ε = 1e-10.

It is rigorous **given** session independence (true for distinct prompts) and
`|Y−μ_Y| ≤ M`. The constant-in-N `a`-term (≈1743 bits faithful, ≈714 codebook)
**dominates at moderate N** and makes Bernstein the most conservative there; its
asymptotic √N coefficient (10.1 faithful, 6.5 codebook) is *below* the validated sub-exp
coefficient (12.9 / 7.8), so the validated tail is heavier asymptotically. THRESHOLD(N)
under each model:

| N | FAITHFUL sub-exp (plotted) | FAITHFUL Bernstein (rigorous) | CODEBOOK sub-exp (plotted) | CODEBOOK Bernstein (rigorous) |
|---|---|---|---|---|
| 1e4 | 0.435 | 0.682 | 0.269 | 0.358 |
| 1e5 | 0.347 | 0.360 | 0.215 | 0.220 |
| 1e6 | 0.319 | 0.318 | 0.198 | 0.198 |
| 1e7 | 0.310 | 0.309 | 0.193 | 0.193 |
| 1e8 | 0.307 | 0.307 | 0.191 | 0.191 |

The two honest models **bracket** the truth: Bernstein is the conservative ceiling at
moderate N (its `a`-term), the validated sub-exp is the trustworthy operating estimate
and the heavier asymptotic tail. The plotted curve is the **sub-exp** line; the Gaussian
(z=6.36) would sit below both and is a **known-unsafe floor**, not shown.

---

## 5. N to bring the threshold within X % of μ at FPR 1e-10

From the §3 √N law, `N_within(frac) = (c/(frac·μ))²`:

| target | model | FAITHFUL K=4 | CODEBOOK K=4 |
|---|---|---|---|
| **within 10 % of μ** | sub-exp (validated) | 1.78×10⁵ | 1.69×10⁵ |
| | Bernstein (rigorous) | 2.25×10⁵ | 1.91×10⁵ |
| **within 1 % of μ** | sub-exp (validated) | 1.78×10⁷ | 1.69×10⁷ |
| | Bernstein (rigorous) | 1.22×10⁷ | 1.24×10⁷ |

(At **within-10 %** (N ~ 2×10⁵) Bernstein needs *more* tokens than sub-exp because its
constant-in-N `a`-term is still active. At **within-1 %** (N ~ 1.5×10⁷) the `a`-term has
washed out and Bernstein's asymptotic √N coefficient is *below* sub-exp's, so Bernstein
needs *fewer* — the validated sub-exp tail is heavier asymptotically. Take the **larger**
of the two as the safe answer at each target.)

**Reading:** to hold an honest-FPR-1e-10 treaty line that sits within **10 % above the
benign mean**, audit **~1.8×10⁵ tokens ≈ 170 distinct prompts/conversations**; to hold
it within **1 %**, audit **~1.8×10⁷ tokens ≈ 16 k prompts** — **for either scheme**, since
the budget scales with μ. At the **131 k tokens / 128 prompts we actually have** (the end
of the solid region), the threshold is already within **~11 % (codebook) / ~12 %
(faithful)** of μ — and everything to the right of N_data on the plot is the dashed
extrapolation of the validated √N law, not new measurement.

---

## 6. Plain-language reading of the plot

- **Each datacenter is graded against a moving line.** Early in an audit (small N) the
  honest line THRESHOLD(N) is far above μ — random benign variation alone can push a
  short window's mean capacity well above μ, so the treaty must allow a wide margin or it
  would tear up honest datacenters. As the audit accumulates tokens, that margin shrinks
  like 1/√N and the line **descends toward μ**: with enough evidence, a datacenter
  running even slightly above its honest mean is distinguishable at 1e-10.
- **Solid = measured variance, dashed = extrapolated variance.** The solid part of each
  curve (up to 131 072 tokens) rests on a directly-measured variance; the dashed part
  projects the same √N law to longer audits than we have data for. The 1e-10 tail
  multiplier is the validated sub-exp model throughout — we never *observe* a 1e-10 event.
- **Codebook runs a lower, tighter line than faithful.** Because codebook matches the
  served FP8 model more closely, its honest mean (μ≈0.19) and its variance are both lower
  — its whole threshold curve sits **below** faithful's (μ≈0.31). A covert channel has
  less room under the codebook treaty.
- **But the *relative* audit cost is the same.** Reaching within 10 %/1 % of μ takes
  ~1.8×10⁵ / ~1.8×10⁷ tokens for **both** schemes, because the buffer scales with μ. The
  choice between schemes is about the *absolute* capacity ceiling (codebook lower), not
  about how long you must audit to certify it.
- **The honest caveat that matters most** (inherited from `BUFFER_FPR.md` §6):
  **afforded ≠ detected ≠ transmitted.** r_t is a worst-case *noiseless-channel ceiling*
  an adversary holding the exact model *could* embed while perfectly mimicking the honest
  margin/rank profile — realized leak is far smaller (Rinberg: < 0.5 %). This curve
  protects an **honest** datacenter from false teardown; it does not by itself bound a
  real adversary's throughput. And the rigor rests on **session independence + the
  empirical support bound M** — i.e. on auditing enough distinct prompts that no single
  honest session dominates the budget.

---

## Reproduce

```bash
cd /workspace/projects/zk-hillclimb/buffer
# 1. faithful extra (120 prompts) — pre-existing from BUFFER_FPR.md:
#    gen_more_faithful.py --nprompts 120 --rowseed 20260612 --gseed-base 30000000
# 2. codebook extra (120 prompts, SAME workload, generated for THIS analysis):
IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python gen_more_codebook.py \
    --nprompts 120 --rowseed 20260612 --gseed-base 30000000 \
    --out codebook_extra_corrected.npz
# 3. both-scheme buffer/variance/tail machinery + threshold_results.json (all models,
#    incl. K=16 + Bernstein cross-check):
/root/int-model-env/bin/python analyze_threshold.py
# 4. the single-panel observed/extrapolated K=4 figure (this deliverable):
/root/int-model-env/bin/python plot_threshold_observed.py
```

Inputs per scheme: `../capacity/capacity_dump_corrected_<scheme>_seed20260611.npz`
(8 prompts) + `<scheme>_extra_corrected.npz` (120 prompts, rows disjoint from the
original 8, identical 248-pt bgrid). Outputs: `threshold_results.json` (all numbers,
both schemes, K=4 + K=16, sub-exp + Bernstein + Gaussian), `threshold_observed_table.json`
(the K=4 decade table + z_eff + observed/law variance ratios for the figure), and
`../threshold_curve.png` (the single-panel observed/extrapolated figure). Buffer/variance/
tail machinery imported byte-identically from `analyze_buffer.py`; the figure's only
reframing is the variance × single-tail-multiplier decomposition and the
observed-vs-extrapolated split at N_data. `int-model-approximation` used **read-only**;
nothing committed or pushed.

**Tokens generated for this deliverable:** 120 new codebook prompt-blocks =
**122 880 tokens** (the 120 faithful prompts pre-existed). Total analyzed:
**262 144 tokens = 256 prompt-blocks** across both schemes (131 072 / 128 each). The plot
rework is a **plotting + buffer-methodology change only — no re-measurement.**
