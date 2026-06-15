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

**`threshold_curve.png`** — the deliverable: two panels (optimum K=4, conservative
K=16), each overlaying **FAITHFUL** (μ≈0.31) and **CODEBOOK** (μ≈0.19), each curve
approaching its own μ, with the rigorous (Bernstein) and validated (sub-exp tail-fit)
variants shown.

This extends `BUFFER_FPR.md` (faithful only) to **both schemes** and to **more benign
tokens**: **131 072 tokens / 128 distinct dolly prompt-blocks per scheme** (8 original +
**120 newly generated here** for *each* scheme; the faithful extra pre-existed, the
**120 codebook prompts were generated for this analysis** — `gen_more_codebook.py`,
identical workload to the faithful extra: same 120 dolly rows, same tiling, same
per-prompt Gumbel seed, so the FP8-served token stream is byte-identical and only the
reference M_int differs).

> **Decoding regime (per `THREAT_MODEL_NOTES.md` §0/§1).** This is the **benign-workload
> capacity arising from the M_int-vs-served-model gap** in the verifiable **sampled**
> regime: reference = M_int post-Gumbel logits, served token = the fast FP8 model's
> post-Gumbel argmax under a **shared committed-seed** Gumbel draw, metric temperature
> **T = 1**. It is **not** a greedy number; per `CAPACITY_TEMPERATURE.md` the faithful
> worst-case capacity is ~T-insensitive, so μ and the curve change little with T.

---

## Bottom line (plain language)

- **Each scheme has its own benign mean μ** (128-prompt, this larger sample):
  - **FAITHFUL:** μ = **0.306** bits/tok (optimum K=4, b*=0.347); **0.330** (conservative K=16).
  - **CODEBOOK:** μ = **0.191** bits/tok (optimum K=4, b*=0.220); **0.201** (conservative K=16).
  - Codebook's μ is **~37 % lower** than faithful's — codebook tracks the FP8 teacher
    more tightly post-Gumbel (96.3 % exact agreement vs 94.2 %), so the honest
    datacenter affords less covert capacity and the treaty can hold a lower line.
- **The threshold relaxes toward μ as you audit more** (per-token, FPR ≤ 1e-10,
  **validated sub-exp** estimate):

  | N (tokens) | FAITHFUL K=4 THRESHOLD | CODEBOOK K=4 THRESHOLD |
  |---|---|---|
  | 16 384  | 0.420 | 0.253 |
  | 65 536  | 0.356 | 0.221 |
  | 262 144 | 0.330 | 0.205 |
  | →∞      | → μ = 0.306 | → μ = 0.191 |

- **To bring the threshold within X % of μ at FPR 1e-10** (the user's "how many tokens
  to reach the 1e-10 number" — from the √N variance-scaling fit; tokens, rounded):
  - **within 10 % of μ:** ≈ **1.8×10⁵ tokens** (~170 prompts) — *both* schemes
    (1.6–1.8×10⁵ tail-fit, 1.9–2.3×10⁵ rigorous Bernstein).
  - **within 1 % of μ:**  ≈ **1.6–1.8×10⁷ tokens** (~16 000 prompts) — *both* schemes
    (1.2–1.5×10⁷ rigorous, 1.6–2.0×10⁷ tail-fit).
  - It is **nearly scheme-independent** because the buffer coefficient scales with μ:
    `c_subexp/μ ≈ 42` (faithful) ≈ `40` (codebook), and
    `N_within(frac) = (c/(frac·μ))² ≈ ((c/μ)/frac)²`. So "within 10 %" ≈ (420)² ≈
    1.8×10⁵ and "within 1 %" ≈ (4200)² ≈ 1.8×10⁷ for **either** scheme.
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
own published optima from `capacity/topk_corrected_results_seed20260611.json` (K=4 is the
U-shaped K-sweep floor = optimum; K=16 = conservative). Machinery imported byte-identically
from `analyze_buffer.py` / `capacity/topk_breakdown.py`; **verified to reproduce the
published 8-prompt optima exactly** (faithful K4 0.3367, codebook K4 0.2203, K16 0.2283).

**Why the 128-prompt μ sits below the 8-prompt headline** (faithful 0.337→0.306,
codebook 0.220→0.191): the 120 additional prompts are, on average, slightly *easier*
(higher post-Gumbel agreement) than the original 8 — the **same downward drift in both
schemes**, a cross-check that the extra prompts are the same benign population sampled
deeper, not a code change. These larger-sample μ are the honest operating means.

---

## 2. The per-token / per-session distribution (128 prompts/scheme)

| quantity | FAITHFUL K=4 | FAITHFUL K=16 | CODEBOOK K=4 | CODEBOOK K=16 |
|---|---|---|---|---|
| μ (bits/tok) | 0.306 | 0.330 | 0.191 | 0.201 |
| σ²(s_t) per-token | 0.298 | 0.355 | 0.158 | 0.214 |
| violations / 131072 | 1771 | 1049 | 991 | 709 |
| **tail-violations** (rank ≥ K) | **48** | **0** | **12** | **0** |
| session-sum Var(Y), L=1024 | 2279 | 3308 | 933 | 1360 |
| max session deviation M (bits) | 227 | 210 | 93 | 108 |

As in faithful, **K=4 carries a heavy ~15-bit discrete tail** (tail-violations priced at
log2(V−4)≈14.97) — 48 in faithful, **12 in codebook** — while **K=16 has q=0 (no
tail-violations, s_t bounded by 4 bits)** for both schemes: conservative in the mean,
better-behaved in the tail. Codebook's tail is lighter than faithful's (12 vs 48 spikes,
lower σ²) — its variance and thus its buffer are smaller in absolute bits.

---

## 3. Variance scaling — the √N fit that grounds the 1e-10 extrapolation

The operative non-i.i.d. effect is **between-prompt heterogeneity**: across independent
sessions (distinct prompts) the cumulative-budget variance is linear with an **inflated**
slope,

> **Var(R(N)) = (N/L)·Var(Y) = N · σ² · τ_var,  N_eff = N / τ_var.**

| | FAITHFUL K=4 | FAITHFUL K=16 | CODEBOOK K=4 | CODEBOOK K=16 |
|---|---|---|---|---|
| **τ_var** (= N/N_eff) | 7.47 | 9.09 | **5.76** | **6.21** |

**Codebook's heterogeneity is milder** (τ_var ≈ 5.8 vs faithful 7.5) — its per-prompt
difficulty varies less, so its effective sample size is larger per token. (τ_var is
partly a **tiling artifact** — each prompt is one short dolly example repeated to 1024
tokens — and is therefore a *conservative* upper bound on what non-repeated natural
traffic would show within a session; the genuine, persistent component is
conversation-to-conversation difficulty variation. See `BUFFER_FPR.md` §3/§6.3.)

**The √N law (the "how many tokens to reach 1e-10" fit).** With the independent-session
variance model, the **per-token buffer falls as c/√N**. The **asymptotic** coefficient
**c (bits·√tok)** is closed-form for Gaussian (`c = z·√(Var(Y)/L)`, z=6.36) and Bernstein
(`c = √(2·ln(1/ε)·Var(Y)/L)`); for the validated sub-exp model we fit c from the buffer
curve (N ≥ 16 384, no a-term so the fit is clean):

| model | FAITHFUL K=4 | CODEBOOK K=4 | FAITHFUL K=16 | CODEBOOK K=16 |
|---|---|---|---|---|
| Bernstein asymptotic c | 10.1 | 6.5 | 12.2 | 7.8 |
| **sub-exp (validated) c** | **12.9** | **7.7** | **14.9** | **9.5** |
| Gaussian (unsafe) c | 9.5 | 6.1 | 11.4 | 7.3 |

Asymptotically **Gaussian < Bernstein < sub-exp** — the validated heavy tail is heavier
than even the rigorous Bernstein √N term. **But Bernstein also carries a constant-in-N
term** `a = ln(1/ε)·M/3` (≈1742 bits faithful K=4, ≈714 codebook K=4) that **dominates at
moderate N** and makes Bernstein the most conservative there; the exact numeric inversion
in §5 keeps it. Inverting `c/√N = frac·μ` gives **N_within(frac) = (c/(frac·μ))²**. The
ratio **c_subexp/μ ≈ 42 (faithful) ≈ 40 (codebook)** is what makes the token budget to
reach within a *relative* fraction of μ nearly scheme-independent.

---

## 4. buffer(N) for FPR ≤ 1e-10 — rigorous vs validated tail-fit (kept honest)

Identical methodology to `BUFFER_FPR.md` §5; with only ~131 k real tokens we **cannot
observe 1e-10 events directly**, so we quote a bracket:

- **(a) Rigorous Bernstein** on B = N/L **independent sessions**, empirical session
  variance Var(Y) and **empirical-support** deviation bound M (observed max — the data
  never exceeds it):
  `buffer = (ln(1/ε)·M/3) + √[(ln(1/ε)·M/3)² + 2·ln(1/ε)·B·Var(Y)]`, ε=1e-10.
  Rigorous **given** session independence (true for distinct prompts) and `|Y−μ_Y| ≤ M`.
  The fully a-priori M (a pathological honest session of 1024 tail-violations) is vacuous
  — which is exactly why **the protocol's safety depends on auditing enough independent
  prompts that no single session dominates the budget.**
- **(b) Validated sub-exponential tail-fit**: fit `ln S(t) ≈ c − t/λ` to the
  block-bootstrap survival curve in the **observable** window [1e-5, 1e-2], extrapolate
  to S = ε. In `BUFFER_FPR.md` this fit was **validated against a 4×10⁷-resample
  bootstrap to 1e-6** (tracks empirical to a few %, slightly conservative; the Gaussian
  underestimates by ~30 % and is a **known-unsafe** lower bound). We carry that
  validation here.

**Validation at observable FPRs (this analysis, both schemes).** Deep block-bootstrap
(4×10⁷ resamples) at N=4096, buffer in bits, sub-exp/Gaussian vs empirical at the FPR
the bootstrap can see:

| scheme (K=4) | ε | empirical | sub-exp | Gaussian |
|---|---|---|---|---|
| FAITHFUL | 1e-6 | 683 | 702 (**+3 %**) | 454 (**−34 %**) |
| CODEBOOK | 1e-6 | 317 | 346 (**+9 %**) | 290 (**−9 %**) |

For **both** schemes the **sub-exp fit is slightly conservative** (tracks the empirical
honest tail to within +3…+9 %) while the **Gaussian underestimates** (a known-unsafe
floor). Codebook's Gaussian error is milder (−9 % vs faithful's −34 %) because codebook's
tail is **lighter** (12 vs 48 tail-violations) — closer to Gaussian, so the sub-exp fit is
if anything *more* reliable for codebook.

**The trustworthy estimate is the sub-exp fit; Bernstein is the rigorous upper bound;
Gaussian is a known-unsafe floor.** The two honest curves (Bernstein, sub-exp) are both
plotted; the gap between them (≈1.1–1.3× at N ≥ 65 k) is the price of rigor over
trusting the validated tail model.

---

## 5. THRESHOLD(N) and the N-to-reach-μ numbers

**THRESHOLD(N) = μ + buffer(N)/N (bits/token), FPR ≤ 1e-10** — `threshold_curve.png`:

**FAITHFUL** (optimum K=4, μ=0.306):

| N | THRESHOLD sub-exp | THRESHOLD Bernstein |
|---|---|---|
| 16 384  | 0.420 | 0.545 |
| 65 536  | 0.356 | 0.380 |
| 262 144 | 0.330 | 0.333 |
| 1 048 576 | — † | 0.318 |
| 16 777 216 | — † | 0.308 |

**CODEBOOK** (optimum K=4, μ=0.191):

| N | THRESHOLD sub-exp | THRESHOLD Bernstein |
|---|---|---|
| 16 384  | 0.253 | 0.301 |
| 65 536  | 0.221 | 0.229 |
| 262 144 | 0.205 | 0.206 |
| 1 048 576 | — † | 0.198 |
| 16 777 216 | — † | 0.192 |

† sub-exp not fit beyond N = 262 144 (bootstrap cost; m > 256 sessions); the
Bernstein/Gaussian envelope brackets it. Conservative-K=16 curves (faithful μ=0.330,
codebook μ=0.201) are in `threshold_results.json` and the right panel of the PNG — same
shape, buffers ~5–15 % larger.

**N needed to bring the threshold within X % of μ at FPR 1e-10** (tokens; from the §3
√N fit, `N=(c/(frac·μ))²`):

| target | model | FAITHFUL K=4 | CODEBOOK K=4 |
|---|---|---|---|
| **within 10 % of μ** | sub-exp (validated) | 1.78×10⁵ | 1.63×10⁵ |
| | Bernstein (rigorous) | 2.25×10⁵ | 1.91×10⁵ |
| **within 1 % of μ** | sub-exp (validated) | 1.78×10⁷ | 1.63×10⁷ |
| | Bernstein (rigorous) | 1.22×10⁷ | 1.24×10⁷ |

(At **within-10 %** (N ~ 2×10⁵) Bernstein needs *more* tokens than sub-exp because its
constant-in-N `a`-term is still active. At **within-1 %** (N ~ 1.5×10⁷) the `a`-term has
washed out and Bernstein's asymptotic √N coefficient (10.1) is *below* sub-exp's (12.9),
so Bernstein needs *fewer* — the validated sub-exp tail is heavier asymptotically. Take
the **larger** of the two as the safe answer at each target.)

**Reading:** to hold an honest-FPR-1e-10 treaty line that sits within **10 % above the
benign mean**, audit **~1.8×10⁵ tokens ≈ 170 distinct prompts/conversations**; to hold
it within **1 %**, audit **~1.8×10⁷ tokens ≈ 16 k prompts** — **for either scheme**, since
the budget scales with μ. At the **131 k tokens / 128 prompts we actually have**, the
threshold is already within **~8 % (codebook) / ~16 % (faithful)** of μ on the validated
fit at N=65 k, and converging.

---

## 6. Plain-language reading of the plot

- **Each datacenter is graded against a moving line.** Early in an audit (small N) the
  honest line THRESHOLD(N) is far above μ — random benign variation alone can push a
  short window's mean capacity well above μ, so the treaty must allow a wide margin or it
  would tear up honest datacenters. As the audit accumulates tokens, that margin shrinks
  like 1/√N and the line **descends toward μ**: with enough evidence, a datacenter
  running even slightly above its honest mean is distinguishable at 1e-10.
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
  real adversary's throughput. And Bernstein rigor rests on **session independence + the
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
# 3. both-scheme threshold curve + N-to-within-μ + plot + threshold_results.json
/root/int-model-env/bin/python analyze_threshold.py
```

Inputs per scheme: `../capacity/capacity_dump_corrected_<scheme>_seed20260611.npz`
(8 prompts) + `<scheme>_extra_corrected.npz` (120 prompts, rows disjoint from the
original 8, identical 248-pt bgrid). Outputs: `threshold_results.json` (all numbers) and
`../threshold_curve.png`. Buffer/variance/tail machinery imported byte-identically from
`analyze_buffer.py`; only the threshold framing, the √N coefficient fit, and the
N-to-within-μ inversion are added. `int-model-approximation` used **read-only**; nothing
committed or pushed.

**Tokens generated for this deliverable:** 120 new codebook prompt-blocks =
**122 880 tokens** (the 120 faithful prompts pre-existed). Total analyzed:
**262 144 tokens = 256 prompt-blocks** across both schemes (131 072 / 128 each).
