# Top-K (Rinberg) capacity rule — optimum b\*, full term breakdown, K sweep

Companion to `CAPACITY_SWEEP.md`. That report gave only the headline minima; this one
opens up the **top-K refined formula at its own optimum** — the five-component breakdown
at `b*_topK` — and sweeps the top-K parameter `K` to show how the bound depends on the
modeling choice. Everything is recomputed from the existing per-position dumps
(`measure/capacity_dump_{scheme}_seed20260611.npz`); **the model was not re-run**.

```
C_topK(b) = H(p) + (1-p)·E_t[log2 N_b]  +  p·( H(q) + (1-q)·log2 K + q·log2(V-K) )

  (a) H(p)              which positions violate (must match the honest exceed-rate)
  (b) (1-p)·E[log2 N_b] within-margin multiplicity on compliant tokens
  (c) p·H(q)            violate: the tail-vs-topK choice itself
  (d) p·(1-q)·log2 K    violations served from inside the teacher's top-K
  (e) p·q·log2(V-K)     violations served from the tail
```

`p(b)` = fraction of positions with post-Gumbel margin > b; `q(b)` = fraction of
*violating* positions whose served token sits outside the teacher's post-Gumbel top-K
(`cand_rank >= K`, measured from the dump); `V = 32000`. Seed 20260611, 8 dolly prompts
× 1024 = 8192 positions, FP8 teacher — identical setup to `CAPACITY_SWEEP.md`.

**Sweep resolution.** Per scheme and per K: the dump's full 248-point exact `b` grid
**plus** a 200-point dense refinement bracketing the coarse argmin (448 points total).
`p` and `q` are exact at every swept `b` (they depend only on `margins`/`cand_ranks`);
only `E[log2 N_b]` is linearly interpolated between grid points off-grid (see Caveats —
the refined minima sit ≤ 0.003 bits below their bracketing *exact* grid values, so the
interpolation cannot be hiding anything material).

Script: `capacity/topk_breakdown.py` → `capacity/topk_results_seed20260611.json`.

---

## 4-line summary

1. **New best (K=16) worst-case entropies:** baseline **12.070**, faithful **0.361**,
   codebook **0.227** bits/token (×1024 ≈ 12 360 / 370 / 232 bits per forward pass).
2. **faithful & codebook after refinement:** the dominant term is now the *compliant*
   within-margin multiplicity (b) — 60 % and 72 % of the total — not violations; the
   violation payload collapsed from `log2 V ≈ 15` to `log2 K = 4` bits each because
   `q = 0` (every violation is a swap inside the teacher's top-5/top-3).
3. **baseline:** still dominated by tail violations (e) at 58 % + multiplicity (b) at
   33 %; top-K buys only 0.2 % because `q ≈ 0.98` at the optimum.
4. **K sweep:** faithful/codebook are **U-shaped in K with the minimum at K=4**
   (0.343 / 0.212 bits/tok — the smallest K that still covers the observed violation
   ranks); baseline is monotone *decreasing* in K over the sweep (11.28 at K=1024).

---

## 1–2. K=16 optimum and the five-component breakdown

The percentages are of the total `C_topK(b*)`. In every scheme the five components sum
to the swept capacity **exactly** (to machine precision, so certainly to 3 decimals —
`sum_matches_C_3dp: true` in the results JSON; the sweep value is computed independently
from the formula expression and compared against the component sum).

### baseline-native — b\*_topK = 8.737, C_topK = **12.0704** bits/tok

p(b\*) = 0.47839 (3919 / 8192 violations) · q(b\*) = 0.98392 (3856 / 3919 in the tail) ·
E[log2 N_b](b\*) = 7.5537

| component | bits | % of total |
|---|---|---|
| (a) H(p) — which positions violate | 0.9987 | 8.3 % |
| (b) (1−p)·E[log2 N_b] — within-margin multiplicity | 3.9401 | 32.6 % |
| (c) p·H(q) — tail-vs-topK choice | 0.0568 | 0.5 % |
| (d) p·(1−q)·log2 K — violate into top-K | 0.0308 | 0.3 % |
| (e) p·q·log2(V−K) — violate into tail | 7.0441 | 58.4 % |
| **sum (= C_topK, checked)** | **12.0704** | 100 % |

The channel is wide open and it is the **tail** that does it: at the optimum nearly half
of all positions violate and 98 % of those violations land outside the teacher's top-16,
each worth `log2(V−16) ≈ 14.97` bits. The huge `E[log2 N_b] = 7.55` reflects the
baseline's enormous margins — at b≈8.7 the within-margin set on *compliant* tokens
already holds ~2⁷·⁶ ≈ 190 tokens on average.

### faithful-arch-v1 — b\*_topK = 0.3767, C_topK = **0.3612** bits/tok

p(b\*) = 0.01221 (100 / 8192 violations) · q(b\*) = 0.0000 (0 / 100 in the tail) ·
E[log2 N_b](b\*) = 0.2199

| component | bits | % of total |
|---|---|---|
| (a) H(p) — which positions violate | 0.0951 | 26.3 % |
| (b) (1−p)·E[log2 N_b] — within-margin multiplicity | 0.2172 | 60.1 % |
| (c) p·H(q) — tail-vs-topK choice | 0.0000 | 0.0 % |
| (d) p·(1−q)·log2 K — violate into top-K | 0.0488 | 13.5 % |
| (e) p·q·log2(V−K) — violate into tail | 0.0000 | 0.0 % |
| **sum (= C_topK, checked)** | **0.3612** | 100 % |

`q = 0` kills components (c) and (e) outright: all 100 violations at b\* serve a token of
teacher rank 1–5 (the dump's max rank is 5 across **all** 8192 positions). The binding
term is now (b), the near-ties *inside* the honest margin — i.e. the residual channel is
mostly "which of the teacher's own near-tied top tokens do you pick", not "violate".

### codebook — b\*_topK = 0.2832, C_topK = **0.2267** bits/tok

p(b\*) = 0.00476 (39 / 8192 violations) · q(b\*) = 0.0000 (0 / 39 in the tail) ·
E[log2 N_b](b\*) = 0.1649

| component | bits | % of total |
|---|---|---|
| (a) H(p) — which positions violate | 0.0436 | 19.2 % |
| (b) (1−p)·E[log2 N_b] — within-margin multiplicity | 0.1641 | 72.4 % |
| (c) p·H(q) — tail-vs-topK choice | 0.0000 | 0.0 % |
| (d) p·(1−q)·log2 K — violate into top-K | 0.0190 | 8.4 % |
| (e) p·q·log2(V−K) — violate into tail | 0.0000 | 0.0 % |
| **sum (= C_topK, checked)** | **0.2267** | 100 % |

Same structure as faithful, tighter still (max observed violation rank: 3). Nearly
three-quarters of the remaining channel is within-margin multiplicity.

---

## 3. K sweep — how the bound depends on the top-K modeling choice

`min_b C_topK` and its argmin per K. K=1 means "a violation must serve the teacher's
argmax" — but a violation by definition doesn't (margin > b ≥ 0 ⇒ rank ≥ 1), so q → 1
and K=1 reproduces the simple rule up to `log2(V−1)` vs `log2 V` (≈ 5×10⁻⁵ bits).

| K | baseline C / b\* (q) | faithful C / b\* (q) | codebook C / b\* (q) |
|---|---|---|---|
| 1 | 12.0977 / 8.79 (1.000) | 0.4500 / 0.529 (1.000) | 0.2785 / 0.288 (1.000) |
| 2 | 12.0977 / 8.79 (1.000) | 0.4028 / 0.377 (0.390) | 0.2308 / 0.233 (0.200) |
| 4 | 12.0956 / 8.79 (0.998) | **0.3433** / 0.303 (0.036) | **0.2124** / 0.233 (0.000) |
| 8 | 12.0871 / 8.74 (0.993) | 0.3487 / 0.303 (0.000) | 0.2197 / 0.233 (0.000) |
| 16 | 12.0704 / 8.74 (0.984) | 0.3612 / 0.377 (0.000) | 0.2267 / 0.283 (0.000) |
| 32 | 12.0362 / 8.74 (0.965) | 0.3734 / 0.377 (0.000) | 0.2315 / 0.283 (0.000) |
| 64 | 11.9663 / 2.37 (0.678) | 0.3856 / 0.377 (0.000) | 0.2363 / 0.283 (0.000) |
| 256 | 11.4968 / 4.18 (0.557) | 0.4084 / 0.422 (0.000) | 0.2458 / 0.283 (0.000) |
| 1024 | **11.2805** / 5.79 (0.410) | 0.4278 / 0.495 (0.000) | 0.2553 / 0.283 (0.000) |

![ksweep](capacity_topk_ksweep.png)

**Lowest-capacity K and monotonicity.**

- **faithful and codebook: NOT monotone — U-shaped in K, minimum at K=4**
  (0.3433 and 0.2124 bits/tok). The mechanism: at K=1 every violation is a "tail"
  violation paying ~`log2 V`; growing K moves violations into the cheap `log2 K` bucket
  (q falls fast because the observed violation ranks max out at 5 / 3); but once K
  exceeds the rank support, q is pinned at 0 and the only effect of larger K is a more
  expensive `log2 K` per violation — capacity climbs again. The minimum lands at the
  **smallest K that (nearly) covers the observed violation ranks**: K=4 covers ranks
  1–3 outright (codebook, q=0) and 96 % of faithful's (q=0.036; its ranks reach 5).
- **baseline: monotone decreasing across the whole sweep**, lowest at K=1024 (11.2805).
  Its violations are spread over the entire vocab (median rank ≈ 185, max 31 854), so q
  never collapses; each increase in K just converts some ~15-bit tail picks into
  `log2 K ≤ 10`-bit top-K picks. K=1024 is not a distinguished optimum — the curve would
  keep creeping down toward the degenerate K→V limit. The refinement can't rescue a
  scheme whose disagreements aren't concentrated near the teacher's preference.

**Adversary-vs-defender reading.** Each K yields a valid capacity bound *under the
assumption that the adversary must match the honest rank profile at that granularity*
(violation rate p **and** tail fraction q_K, else the served-rank histogram is itself a
detectable signature). Smaller K constrains the adversary's rank profile more finely —
a tighter, more realistic bound — at the price of a larger measured q. If the verifier
monitors served-token ranks as well as margins, the defender may quote the min over K:
**faithful 0.343 (K=4), codebook 0.212 (K=4)**; if only margins are checked, the honest
ceiling is the simple rule (the K=1 column). K=16 is the conventional middle ground and
the headline we carry forward.

---

## 4. Top-K vs the simple rule at the optimum

Simple-rule optima recomputed with the identical 448-point sweep (so the comparison is
like-for-like; the published `CAPACITY_SWEEP.md` values 0.459 / 0.280 / 12.099 are the
same minima on the coarser 248-point grid, reproduced here on-grid to 4 decimals).

| Scheme | simple min (b\*) | top-K=16 min (b\*) | saved | violation payload: simple → top-K |
|---|---|---|---|---|
| baseline | 12.0977 (8.79) | 12.0704 (8.74) | 0.027 bits (**0.2 %**) | 7.11 → 7.13 bits (c+d+e; unchanged — q≈0.98) |
| faithful | 0.4500 (0.529) | 0.3612 (0.377) | 0.089 bits (**19.7 %**) | 0.0895 → 0.0488 bits |
| codebook | 0.2785 (0.288) | 0.2267 (0.283) | 0.052 bits (**18.6 %**) | 0.0695 → 0.0190 bits |

**Which term shrank:** the violation payload. Under the simple rule every violation pays
`log2 V = 14.97` bits (term `p·log2 V`); with q = 0 the top-K rule re-prices it at
`log2 K = 4` bits (term (d), with (c) and (e) vanishing). For faithful that cuts the
payload from 0.0895 to 0.0488 bits even though p *doubled* (0.0060 → 0.0122) — violations
got ~4× cheaper, so the optimal b\* slid down (0.529 → 0.377) to buy a smaller
within-margin term (b): 0.31 → 0.22 bits. Codebook behaves identically (payload
0.0695 → 0.0190). For baseline the refinement is nearly inert: 98 % of its violations
are genuine tail picks, so re-pricing the top-K bucket touches almost nothing.

---

## Caveats (honest)

1. **How q is estimated.** q(b) is the tail fraction of the *scheme's own natural*
   disagreements (the student's served tokens), used as the honest profile the adversary
   must mimic. At the K=16 optima q = 0 is estimated from only **100** (faithful) and
   **39** (codebook) violating positions: rule-of-three 95 % upper bounds are q ≤ 0.030
   and q ≤ 0.077. Plugging those worst cases in adds ≤ 0.006 bits (≤ ~2–3 % of the
   minimum) — the conclusions are insensitive, but with 8 192 positions the q = 0 cells
   are genuinely small-sample. (Supporting structure: max served-token rank over **all**
   8 192 positions is 5 / 3, not just over violations.)
2. **Ties.** `cand_rank` counts tokens with post-Gumbel deficit *strictly* less than the
   served token's, so on an exact float tie the served token takes the **lowest** rank of
   the tied group — biasing q (and hence capacity) slightly *down*. Post-Gumbel scores
   are continuous, so exact float32 ties beyond rank 0 are rare; margin = 0 (rank 0) is
   "agreement", never a violation.
3. **Off-grid interpolation.** p and q are exact at every swept b. `E[log2 N_b]` is
   exact on the 248-point dump grid and linearly interpolated (per position) on the
   200-point dense refinement; every refined minimum is bracketed by exact grid
   evaluations no more than 0.003 bits above it, which bounds the interpolation error.
   `N_b` is a step function of b, so the interpolated value always lies between the two
   bracketing exact values.
4. **K=1 semantics.** "Must serve teacher argmax on a violation" is self-contradictory
   (a violation has rank ≥ 1 by construction), so q(K=1) ≡ 1 and the K=1 row is the
   simple rule, as the table shows — a consistency check, not new information.
5. **Inherited from `CAPACITY_SWEEP.md`:** fixed public seed 20260611 (not an official
   round), 8 dolly prompts × 1024 positions, Gumbel temperature 1.0 (the DiFR metric's),
   unclamped margins, and the Rinberg "realized ≪ theoretical (< 0.5 %)" caveat — these
   are noiseless-channel ceilings, not expected leak rates.

---

## Reproduce

```bash
cd /workspace/projects/zk-hillclimb/capacity
/root/int-model-env/bin/python topk_breakdown.py --seed 20260611
# -> capacity/topk_results_seed20260611.json
# -> ../capacity_topk_ksweep.png
```

Inputs are the existing dumps `measure/capacity_dump_{baseline,faithful,codebook}_seed20260611.npz`
(produced by `measure/capacity_dump.py`; not re-generated here). On-grid spot checks
reproduce the published `CAPACITY_SWEEP.md` table values to the printed precision:
C_simple(0.55) = 0.4585 / C_topK(0.38) = 0.3633 (faithful), C_simple(0.29) = 0.2795 /
C_topK(0.28) = 0.2281 (codebook), C_simple(8.8) = 12.0991 / C_topK(8.8) = 12.0730 (baseline).
