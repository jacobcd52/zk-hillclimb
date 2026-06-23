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
