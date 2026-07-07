# Task: plot three capacity measures' treaty-threshold curves vs #tokens (faithful scheme)

Reuse the benign dumps + machinery in capacity/ and buffer/ (faithful: reference = M_int
post-Gumbel, served = FP8 argmax; ranks/N_b/margins already dumped; ~131k benign tokens; the
autocorrelation/N_eff~N/7.5 fit and the validated sub-exp tail are established). DO NOT
re-run the model. V = 32000.

## Plot: y(N) vs N, three curves, one figure (faithful)
y(N) = [benign per-token rate of the measure] + [FPR buffer(N)] + [counting-slack(N)]
  i.e. the effective per-token leakage capacity the treaty must tolerate to hold a
  false-tear-up rate <= 1e-10 after auditing N tokens. Lower = tighter = better.
x-axis: N = tokens audited, LOG scale, powers of 10 from 1e4 to 1e12 (show the large-N /
  "billions of tokens" regime, since that is where the V log n term washes out).

The THREE measures (compute each measure's benign rate + variance from the faithful dump):
1. FIVE-TERM bound: R5 = H(p) + (1-p) E[log2 N_b | within-margin]
                          + p( H(q) + (1-q) log2 K + q log2(V-K) )   [the afforded/margin one;
   benign mean already ~0.306]. counting-slack_5(N) = 2*log2(N)/N (macro-type is ~(p,q),
   2 params; negligible).
2. RANK-ENTROPY (full, incl. V log N correction): R_rank = H(empirical distribution of the
   SERVED token's post-Gumbel rank over the N tokens) [realized; mostly rank 0]. 
   counting-slack_full(N) = (V-1)*log2(N)/N   [V=32000; this is the big n-dependent term].
3. RANK-ENTROPY (top-K trick): R_topK = H(p) + (1-p)*S + p*log2(V-K), where p = empirical
   fraction of served tokens OUTSIDE the top-K, S = entropy of the served-rank histogram
   RESTRICTED to (conditional on) the in-top-K positions. counting-slack_topK(N) =
   (K-1)*log2(N)/N. Use K = 4 (state it; optionally also show K=16). 

For EACH measure compute, from the benign faithful data:
- benign rate (the mean value of the measure on the benign workload).
- FPR buffer(N): the statistical margin so that P(benign measured value > benign_rate +
  buffer) <= 1e-10. Use a BLOCK-BOOTSTRAP over CONTIGUOUS token windows of size N (respects
  the within-prompt autocorrelation / N_eff) for N up to the ~131k data limit; fit the upper
  tail (validated sub-exp, as in BUFFER_FPR) to reach 1e-10; for N beyond the data,
  EXTRAPOLATE the buffer by the fitted law (~1/sqrt(N) for the sum-like measures; for the
  entropy functional use the empirical-entropy fluctuation sigma_H/sqrt(N_eff),
  sigma_H^2 = sum_r P(r) (log2 P(r))^2 - H^2, plus note the -(A-1)/(2 N ln2) entropy bias).
  Be explicit which law each measure uses.
- counting-slack(N): the analytic term above (2/V-1/K-1 times log2(N)/N).
- y(N) = benign + buffer(N) + slack(N).

## Deliverable
/workspace/projects/zk-hillclimb/THREE_MEASURE_CURVE.md + three_measure_curve.png:
- the single figure (3 curves; solid where N <= data limit i.e. buffer is observed, dashed
  where extrapolated; mark the data-limit boundary).
- a short table of y(N) at N = 1e4, 1e6, 1e9, 1e12 for all three.
- the benign rates + which term dominates each curve in which N regime, the CROSSOVER N where
  the full-rank curve drops below the five-term and below the top-K curve (the V log n washout).
- a one-line verdict: which measure gives the tightest treaty bound at realistic audit sizes
  (~1e9-1e12), and at small sizes.
Honest about the bias/extrapolation. /root/int-model-env/bin/python. No git commits; no
pushes; int-model-approximation READ ONLY.
