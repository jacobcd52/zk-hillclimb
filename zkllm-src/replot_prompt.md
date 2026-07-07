# Task: regenerate the treaty-threshold plot (K=4 only) with observed-vs-extrapolated buffer

Rework the plot in THRESHOLD_CURVE.md / threshold_curve.png using the EXISTING data + scripts
(buffer/threshold_results.json, buffer/analyze_threshold.py, the corrected + extra dumps;
~131k tokens/scheme: faithful + codebook). This is a plotting + buffer-methodology rework,
NOT a re-measurement. Keep the K=4 (top-K optimum) capacity definition we already use.

## The plot (single panel, replaces the current 2-panel figure)
- ONE panel, K=4 only (drop the K=16 conservative panel).
- Two curves overlaid: FAITHFUL and CODEBOOK.
- x-axis: N = tokens audited, LOG scale, ticks at POWERS OF 10 STARTING AT 1e4
  (1e4, 1e5, 1e6, 1e7, ...). Do not start below 1e4.
- y-axis: THRESHOLD(N) = mu + buffer_{FPR<=1e-10}(N), decreasing toward each scheme's mu.

## The buffer: observed where we have it, extrapolated above (the key fix)
The buffer depends on the cumulative variance Var(Rbar_N) of the N-token AVERAGE capacity.
- For N up to the data limit N_data (~131k tokens), compute Var(Rbar_N) DIRECTLY from the
  real tokens (empirical variance of length-N block/window averages over the ~131k-token
  series, respecting within-prompt structure). This is the OBSERVED region.
- For N > N_data, EXTRAPOLATE Var(Rbar_N) via the fitted law (the N_eff / ~1/N with the
  autocorrelation-corrected slope already established). This is the EXTRAPOLATED region.
- Convert variance -> buffer at FPR 1e-10 with the SINGLE validated sub-exponential tail
  (the one already bootstrap-validated). DROP the separate Bernstein line from the figure
  (the user found two lines confusing) — mention Bernstein only in the doc text as a
  rigorous conservative cross-check, not on the plot.
- Draw each scheme's curve SOLID in the observed-N region and DASHED in the extrapolated
  region, with a clear marker/vertical line at N_data where observed -> extrapolated.

## Honesty (state on the figure caption + in the doc)
1e-10 is never directly observed (needs ~1e10 samples); what is observed is the VARIANCE up
to N_data. So "observed region" = variance measured directly; the 1e-10 buffer multiplier is
always the validated sub-exp tail model. Make this explicit so the solid/dashed distinction
isn't misread as "we observed 1e-10 events."

## Deliverable
Overwrite threshold_curve.png with the single-panel figure; update THRESHOLD_CURVE.md text +
caption to match (single panel, observed/extrapolated split, Bernstein moved to a text
cross-check, the observed-means-variance clarification). Keep the numeric threshold table
(faithful/codebook at 1e4/1e5/1e6...) consistent with the new figure. /root/int-model-env/bin/python.
No git commits; no pushes; int-model-approximation READ ONLY.
