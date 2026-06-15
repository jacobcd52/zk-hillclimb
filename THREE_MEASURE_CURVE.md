# Three capacity measures' treaty-threshold curves vs #tokens — FAITHFUL scheme

**Goal.** Hold the scheme fixed (faithful corrected orientation: REFERENCE = M_int
post-Gumbel logits, SERVED = FP8 argmax; the ~131 072-token / 128-prompt benign dump
reused verbatim from `buffer/` + `capacity/`, **model never re-run**) and ask: of three
candidate per-token covert-capacity *meters* the treaty could enforce, which gives the
**tightest** ceiling at a given audit size? For each meter we plot

```
y(N) = benign_rate  +  FPR_buffer(N)  +  counting_slack(N)        [bits / token]
```

the per-token leakage capacity the treaty must tolerate so an **honest** datacentre is
torn up with probability **≤ 1e-10** after auditing **N** tokens. **Lower y = tighter =
better.** x-axis is N = tokens audited, log scale, 1e4 → 1e12 (the "billions of tokens"
regime is exactly where the full-rank meter's `V·log n` term finally washes out).

> Reuses the validated machinery in `buffer/analyze_buffer.py` (per-token `s_t`, whole-session
> block-bootstrap, sub-exponential tail-fit to 1e-10, `z_upper`) and the established
> autocorrelation result (session-level variance inflation → `N_eff ≈ N/τ`). All three meters
> are computed from the **same** benign faithful dump. `int-model-approximation` untouched
> (read-only); nothing committed or pushed.

**Figure:** `three_measure_curve.png`. **Numbers:** `buffer/three_measure_results.json`.
**Code:** `buffer/three_measure_curve.py`.

---

## The three meters and their benign rates

| # | meter | benign rate (bits/tok) | counting-slack coeff (#params) | buffer law |
|---|---|---|---|---|
| 1 | **FIVE-TERM R5** (margin/afforded, K=4, b\*=0.347) | **0.30583** | `2·log₂N/N` (p,q) | sum-like `c/√N` |
| 2 | **FULL RANK-ENTROPY** `H(served rank)` | **0.36173** | `(V−1)·log₂N/N`, **V−1 = 31 999** | entropy fluct. `c/√N` |
| 3a | **TOP-K rank-entropy, K=4** | **0.36742** | `(4−1)·log₂N/N` = 3 | `c/√N` |
| 3b | **TOP-K rank-entropy, K=16** | **0.36173** | `(16−1)·log₂N/N` = 15 | entropy fluct. `c/√N` |

The benign served-token rank distribution (over all 131 072 tokens) has support **only on
ranks 0–9** (A_obs = 10 categories): `P(rank 0)=0.9418, P(1)=0.0503, P(2)=0.0062, …,
P(9)=8e-6`. Hence `H_full = 0.36173 bits`, `σ_H = 1.1554`. Because the deepest served rank
is 9, **no token is outside the top-16** (`p_out=0` at K=16) — so the **K=16 top-K meter has
exactly the same benign rate and statistical buffer as the full-rank meter** (it is the same
entropy of the same 10-category histogram), differing *only* in its counting slack (15 vs
31 999 params). At K=4 the 56 out-of-top-4 tokens (`p_out=4.27e-4`) are lumped at
`log₂(V−4)=14.97` bits, nudging the benign rate up to 0.36742 and erasing the rank-4…9
resolution from the conditional entropy S.

The five-term and the rank-entropy meters are **not nested** — they price different
channels. The five-term charges compliant tokens for *within-margin multiplicity*
`log₂ N_b` (zero for 83.7 % of tokens at b\*), giving the lowest benign mean (0.306); the
rank meters charge a single histogram-entropy reflecting the 5 % rank-1 rate (0.362). Lower
is not "more correct," it is a tighter treaty ceiling — that is what we are comparing.

---

## The three y(N) terms

**benign_rate** — the meter evaluated on the full benign dump (its asymptotic value); the
floor the treaty must always tolerate.

**FPR_buffer(N)** — the statistical margin with `P(measured(N) > benign_rate + buffer) ≤
1e-10` for honest traffic, sized by **whole-session block-bootstrap** (resample the 128
1024-token prompt-blocks with replacement → this automatically carries the within-prompt
autocorrelation, so the buffer already lives in the `N_eff = N/τ` world) and a
**sub-exponential upper-tail fit** to 1e-10 — the exact estimator validated to ~1e-6 against
deep bootstrap in `BUFFER_FPR.md`. Buffer is **observed (solid)** for **N ≤ 131 072** (the
data limit), **extrapolated (dashed)** for larger N by each meter's fitted `c/√N` law:

| meter | fitted c (bits·√tok) | law / cross-check |
|---|---|---|
| five-term | **13.24** (spread 5 %) | sum-like; matches `BUFFER_FPR` (per-tok 0.051 @65k → c≈13.1) |
| full-rank | **18.44** (spread 6 %) | entropy fluctuation |
| top-4 | **21.40** (spread 6 %) | heavier (rare 15-bit out-of-top-4 spikes) |
| top-16 | **18.31** (spread 6 %) | ≡ full-rank (same histogram) |

*Entropy-fluctuation cross-check (measures 2 & 3b).* The analytic law is `buffer =
z(1e-10)·σ_H/√N_eff`. The entropy-estimator variance inflation measured from the bootstrap
is **τ_entropy ≈ 3.18** (smaller than the five-term's τ=7.47, because entropy is governed by
the bulk rank-1 multinomial, not the heavy 15-bit violation tail). With `z(1e-10)=6.36`,
`σ_H=1.155`, this gives the **Gaussian** coefficient `c = z·σ_H·√τ = 13.1`. The
bootstrap **sub-exp** fit returns **18.4** — ~40 % larger, i.e. the honest entropy tail is
heavier than Gaussian (the same direction and magnitude as the Gaussian-underestimates-by-30 %
result in `BUFFER_FPR.md`). We plot the **sub-exp (larger, safer)** coefficient.

**counting_slack(N)** = `(#params)·log₂N / N` — the method-of-types description cost: the
number of empirical types of denominator N over an A-ary alphabet is `~(N+1)^(A−1)`, so
identifying which type the audit realised costs `(A−1)·log₂N` bits, `(A−1)·log₂N/N` per
token. The slack uses the **theoretical** alphabet the meter must be robust to (V for
full-rank, K for top-K, 2 for the five-term's (p,q) macro-type), **not** the 10 observed
ranks — an adversary's type could be any of them. This is the term that separates the
meters.

---

## y(N) table (bits/token)

| meter | N = 1e4 | N = 1e6 | N = 1e9 | N = 1e12 |
|---|---|---|---|---|
| **FIVE-TERM R5** | **0.4611** | **0.3191** | **0.3062** | **0.3058** |
| FULL RANK-ENTROPY | 43.09 | 1.018 | 0.3633 | 0.3617 |
| TOP-K (K=4) | 0.6189 | 0.3889 | 0.3681 | 0.3674 |
| TOP-K (K=16) | 0.5928 | 0.3803 | 0.3623 | 0.3617 |

(At **N=1e4** the buffer is **observed** — N < the 131 072-token data limit. At N=1e6, 1e9,
1e12 the buffer is **extrapolated** `c/√N`. The slack term is analytic at all N.)

---

## Which term dominates, in which regime

- **FIVE-TERM (blue):** slack is negligible at all N (2 params). **Buffer-dominated for
  N ≲ 1e5** (`c/√N` ≈ 0.13 @1e4), **benign-dominated (→0.306) for N ≳ 3e5**. Lowest benign
  rate *and* lowest buffer coefficient → lowest curve essentially everywhere.
- **FULL RANK-ENTROPY (red):** utterly **slack-dominated** until the `V·log n` term washes
  out — slack = **42.5 @1e4, 0.638 @1e6, 9.6e-4 @1e9, 1.3e-6 @1e12**. Only past **N ≈ 1e8–1e9**
  does it fall to its benign floor 0.362. The buffer (18.4/√N) and the Miller-Madow bias are
  irrelevant against this term until washout.
- **TOP-K K=4 (green):** slack negligible (3 params). Buffer-dominated N ≲ 1e5, benign floor
  0.367 thereafter.
- **TOP-K K=16 (purple):** slack negligible (15 params; 0.02 @1e4, 3e-4 @1e6). Same benign
  floor and buffer as full-rank → tracks full-rank's **asymptote** but with **2 133× smaller
  slack**, so it sits at the floor from N ≈ 1e5 instead of N ≈ 1e9.

### Crossovers (the V·log n washout)

- **Full-rank drops below TOP-4 at N ≈ 1.5e8** — the only genuine crossover. Above this the
  `(V−1)log₂N/N` slack has shrunk below the 0.0057-bit benign gap (0.36742 − 0.36173).
- **Full-rank never drops below TOP-16.** They share the same benign rate (0.36173) and
  buffer; full-rank carries strictly more slack, so it approaches top-16 **from above** and
  converges but never crosses. **Top-16 dominates full-rank at every N.**
- **Full-rank never drops below the FIVE-TERM.** Its floor 0.36173 > the five-term's 0.30583,
  so no washout can make the realized rank-entropy beat the margin-based meter.

---

## Verdict

**The FIVE-TERM (margin/afforded) meter gives the tightest treaty bound at *both* small and
realistic audit sizes** — 0.461 bits/tok at 1e4 and 0.306 at 1e9–1e12, below all three
rank-entropy meters at every N (lowest benign rate 0.306 *and* lowest buffer coefficient
13.2).

The **full rank-entropy** — the "information-theoretically natural" realized meter — is the
**worst** choice at every realistic audit size: its `(V−1)·log₂N/N` counting slack makes
y(1e4) ≈ 43 bits/tok and y(1e6) ≈ 1.0, and even after the `V·log n` washout (N ≳ 1e8–1e9) it
only converges to 0.362 — tying top-16 and beating top-4, but **never** beating the
five-term. **The top-K trick (K=16) recovers the full-rank asymptote with 2 133× less slack
and so dominates the full-rank meter everywhere**; it is the meter to use *if* one insists on
a rank-entropy formulation, but it still loses to the margin-based five-term because
rank-entropy ignores the within-margin multiplicity channel.

---

## Honest caveats

1. **Buffer beyond 131 072 tokens is extrapolation.** The sub-exp tail fit is validated to
   ~1e-6 by deep bootstrap (`BUFFER_FPR.md`) and assumed log-linear to 1e-10; the dashed
   segments additionally assume the fitted `c/√N` law holds out to 1e12. If the honest tail
   is heavier than exponential past 1e-6 the buffers are low — but for N ≳ 1e6 the buffer is
   already sub-dominant to the benign floor for every meter, so the curve shapes and the
   verdict are insensitive to it. The data limit (solid→dashed) is marked on the figure.
2. **Entropy-estimator bias is real but tiny and subdominant.** The plug-in entropy is biased
   **down** by Miller-Madow `−(A_obs−1)/(2N ln2) = −6.49/N` (A_obs=10) — opposite sign to the
   buffer and ≤ 6.5e-4 bits/tok already at N=1e4, ≪ the slack there. It is a *sampling* bias
   of the estimator and is **distinct** from the counting slack (a worst-case
   type-*description* cost over the theoretical V-/K-ary alphabet, not the 10 observed ranks);
   we keep them separate and do not net them.
3. **τ_entropy ≈ 3.18 vs τ_var = 7.47.** The entropy meters fluctuate less per session than
   the five-term sum because they are dominated by the bulk rank-1 multinomial rather than the
   heavy log₂(V−K) violation spikes; both τ are partly inflated by the prompt-tiling artifact
   (see `BUFFER_FPR.md §6.3`) and are therefore conservative.
4. **Meters price different channels.** The five-term's lower benign rate reflects that
   83.7 % of compliant tokens have N_b=1 (0 bits) at b\*; it is not strictly an upper bound on
   the rank-entropy and vice-versa. The comparison is "which per-token ceiling is tighter,"
   not "which is correct."
5. **Inherited:** 128 distinct dolly prompts (8 + 120), each tiled to 1024 tokens; fixed
   public Gumbel seed (T=1, not an official round); V=32000; bf16/FP8 construction identical
   to `CAPACITY_CORRECTED.md`. Afforded ≠ realized ≠ transmitted (noiseless-channel ceiling).

## Reproduce

```bash
cd /workspace/projects/zk-hillclimb/buffer
/root/int-model-env/bin/python three_measure_curve.py
# inputs (read-only): ../capacity/capacity_dump_corrected_faithful_seed20260611.npz
#                     + faithful_extra_corrected.npz   (120 extra prompts)
# outputs: ../three_measure_curve.png, three_measure_results.json
```
