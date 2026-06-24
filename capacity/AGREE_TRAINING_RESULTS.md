# Can training make the integer model agree more with the fp8 model? (empirical)

Setup: served M_q = fp8 (FP8Linear, `_scaled_mm`) on Qwen2.5-0.5B; reference M_int = an
integer model (int8 W8A8, or the fp8-codebook). Metric = held-out **R_rank** (entropy of
M_q's served token's rank under M_int, shared Gumbel, token-ID tie-break) on a pre-generated
M_q-on-policy corpus (~3980 seqs / 420k completion tokens, single pass, no multi-epoch).
We train M_int to match M_q and measure whether R_rank drops. Lower = better.

## Untrained baselines (this harness, top-K-served eval)
- fp8 / codebook base: **R_rank 0.641**
- int8 W8A8 base:      **R_rank 0.784**  (int8 adds its own quantization error on top of the
  accumulation gap, so it disagrees with M_q more than the fp8 codebook does)

## Training sweep — every configuration makes R_rank WORSE (best = untrained init)
Bases {int8, fp8} x modes {full-param QAT (STE), frozen-base logit-correction head} x
losses {top-K KL, hard-CE on M_q argmax, Gumbel-coupled CE} x LR {1e-6 ... 1e-3}:

| run | base | mode | loss | lr | init | final |
|---|---|---|---|---|---|---|
| full_qat | int8 | full_qat | topk_kl | 1e-4 | 0.798 | 2.765 |
| full_qat | int8 | full_qat | topk_kl | 1e-5 | 0.798 | 1.058 |
| full_qat | int8 | full_qat | topk_kl | 1e-6 | 0.798 | 0.817 |
| full_qat | int8 | full_qat | hard_ce | 1e-5 | 0.798 | 1.621 |
| head     | int8 | head     | topk_kl | 1e-3 | 0.784 | 0.868 |
| head     | int8 | head     | hard_ce | 1e-3 | 0.784 | 1.424 |
| head     | fp8  | head     | topk_kl | 1e-3 | 0.641 | 0.740 |
| head     | fp8  | head     | gumbel  | 1e-3 | 0.641 | 0.984 |
| head     | int8 | head     | gumbel  | 1e-3 | 0.784 | 1.061 |

In every case `best_rrank == init_rrank`: training
never beat the untrained model at any checkpoint. Loss went DOWN while R_rank went UP — the
surrogate losses are flat-to-anti-correlated with R_rank.

## Why (consistent with the measured cause of the gap)
R_rank is dominated by **near-tie nondeterminism**: at the ~6-15% of positions where M_q's
top tokens are nearly tied, M_q's accumulation-rounding picks a winner that M_int (a different
arithmetic) can't predict. The losses optimize M_q's *soft distribution* / *expected argmax*,
which (a) does not address the per-token, position-specific tie flips, and (b) any logit/weight
movement that lowers the loss perturbs the delicate near-tie orderings that were *already*
aligned in the untrained model, flipping some of them -> R_rank rises. The untrained,
operand-matched model is effectively a local optimum for R_rank; these losses only move away
from it. This held for the frozen-base **head** too (which cannot destabilize the base), so it
is a loss-vs-metric misalignment, not mere training instability.

## Takeaways
- **No cheap or trained win found for "train M_int to match M_q".** The lowest R_rank is the
  untrained best-operand-matched model (the fp8 codebook). A learned per-vocab-bias + low-rank
  logit head (a superset of closed-form bias-correction) did not help, so closed-form
  calibration is not expected to either.
- The other direction (train the *served* M_q toward M_int) was not run here: it modifies the
  deployed model and, per the near-tie analysis, would only lower R_rank by sharpening the
  model's distribution (overconfidence) -> a capability/calibration cost.
- Plot: `agree_training_plot.png`. Corpus: `corpus/mq_corpus.npz`. Harness: `train_agree.py`.

## Update: MSE-on-top-logits loss (the loss that finally moves R_rank the right way)

Earlier losses (top-K KL, hard-CE, gumbel-CE) all *raised* R_rank. Switching to **MSE on
M_q's top-m logit VALUES, computed in fp32** (so small `z_int-z_q` diffs don't round to zero
in bf16) at **very low LR** finally reduces it. Diagnostic added: `R_topK` = rank restricted
to M_q's top-K candidate set (isolates the tail).

| base | loss | lr | R_rank init→best | R_topK init→best | notes |
|---|---|---|---|---|---|
| int8 | mse top-5  | 1e-6 | 0.791 → 0.762 | 0.553 → 0.509 | best ~step50, noisy |
| int8 | mse top-16 | 1e-6 | 0.795 → 0.784 | 0.555 → 0.511 | best ~step500 then OVERSHOOTS back to 0.80 |
| int8 | mse top-* | ≥3e-6 | worsens | worsens | LR window razor-thin |
| fp8/codebook | mse top-16 | 3e-7 | 0.644 → 0.638 | 0.372 → 0.355 | first time below the codebook floor |
| fp8/codebook | mse top-16 | 1e-6 | 0.644 → 0.674 | — | too high |

Findings:
- **MSE-on-logit-values + fp32 + tiny LR genuinely lowers R_rank** (reverses the earlier
  "training always hurts"). The earlier failures were the wrong loss (KL/CE optimize
  distribution shape/argmax, not the logit values that set the ranking) + too-high LR.
- But the gains are **small, fragile, and tail-limited**: R_topK (the top-K ordering MSE
  directly targets) improves cleanly (~0.55→0.51, ~0.37→0.355), while full-vocab R_rank
  improves only marginally because the unconstrained ~152k-token tail dominates it (each tail
  token gets its own Gumbel; full-param training lets the tail drift up and overtake).
- LR window is razor-thin (int8 ≤1e-6, fp8 ≤3e-7) and runs **overshoot** with more steps →
  early-stopping required. top-16 ≈ top-5 on full R_rank (both tail-limited).
- **No config beats the untrained codebook floor (~0.64 here)** by a meaningful margin; the
  best is the codebook itself nudged to ~0.638.

**The bottleneck is the tail.** To convert the clean R_topK gain into a full-R_rank gain, the
next lever is tail control: a logsumexp/tail-suppression term, or re-caching FULL M_q logits
for a full-distribution match (top-m MSE only constrains m tokens).

## CORRECTION: the harness R_rank was inflated by a served-token bug (found via verification)

The eval approximated M_q's "served" token as its argmax over the cached **top-64** logits
instead of the **full vocab**. For the ~5% of positions where M_q's true Gumbel-argmax is a
lower-probability token, this breaks the shared-Gumbel coupling and inflates R_rank.

Verification (`verify_rrank.py`, exact codebook, full logits):
- R_rank, **clean** (full-vocab served) = **0.392**   (matches the original 0.355)
- R_rank, harness (top-64 served)        = 0.648  (the inflated number)
- competitors outside M_q top-64 = **0.011/token** -> the "tail" is negligible; R_rank is
  near-tie-dominated *within* the top. (My earlier "tail is the bottleneck" was wrong.)

Fixed the eval to use a cached full-vocab served token (`cache_served.py` ->
`corpus/served_full.npz`). **Clean frozen baselines:** int8 **0.568**, fp8/codebook **0.384**
(R_topK 0.519 / 0.349; R_rank now ~= R_topK, confirming no tail).

**Clean training (MSE-top-m, fp32, full-vocab eval):**
- int8, top-16, lr 3e-7: R_rank **0.580 -> 0.528** (R_topK 0.530 -> 0.470). Real, modest.
- int8, lr 1e-6: no improvement (window even tighter on the clean metric).
- fp8/codebook: (runs in progress) — testing whether 0.384 can go toward/below the 0.355 floor.

Net so far: with the corrected metric, MSE-on-top-logit-values at tiny LR gives a real but
modest R_rank reduction (int8 ~0.58->0.53); the codebook is already near the ~0.38 floor.

## Long run from codebook init (lr 1e-7, 12k steps, data cycled): hits the floor

To test whether a very stable low-LR descent could break the codebook's floor, ran
full_qat fp8/codebook base, MSE-top64, lr 1e-7, bs4, 12000 steps (data cycled ~7x).
Result: R_rank **oscillates in 0.376-0.398** the whole run; init 0.389, best 0.376 (R_topK
0.333, ~the original 0.355), final 0.390. **No real descent** — the codebook sits at its
near-tie floor and training just wanders there.

## Definitive conclusion (training to lower R_rank)
- The operand-matched **codebook is the floor (~0.38, R_topK ~0.335 ~= original 0.355)** and
  CANNOT be trained lower (any loss / LR / length just oscillates).
- A coarser integer model (**int8, 0.58**) CAN be trained DOWN toward that floor (~0.525 via
  MSE-on-top-logit-values, fp32, lr 3e-7), recovering int8's extra quantization error, but
  not past the codebook.
- => Training-for-agreement = "let a cheap integer model catch up to the codebook," NOT "beat
  the codebook." The near-tie nondeterminism floor is irreducible by training.
- (The earlier "training always worsens R_rank" and "tail is the bottleneck" were artifacts of
  the top-64 served-token eval bug; fixed via full-vocab served caching.)
