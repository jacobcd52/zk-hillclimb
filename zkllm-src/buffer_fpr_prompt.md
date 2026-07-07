# Task: measure the per-token afforded-capacity r, its variance/autocorrelation, and the buffer-vs-FPR-vs-audit-size curve for the capacity-budget treaty

CONTEXT (read CAPACITY_CORRECTED.md, CAPACITY_TOPK.md, CAPACITY_FP16.md + their scripts in
capacity/ and measure/ first). We are designing a treaty-enforcement protocol: a ZKP gives
the INTEGER model's logits; network taps give the SERVED tokens (from the fast quantized/FP8
model); the verifier bounds the worst-case covert capacity AFFORDED per token and tears up
the treaty when cumulative budget exceeds a threshold. We need the false-positive analysis:
how big a buffer above the benign mean is needed so an HONEST datacenter is wrongly shut
down with probability < 1e-10, as a function of how many tokens have been audited.

## Definitions (pin these exactly)
- Benign/honest scenario = the CORRECTED orientation already dumped: REFERENCE = M_int (the
  proven integer model's post-Gumbel logits), SERVED token = the FP8 fast model's argmax.
  Use the FAITHFUL scheme (the realistic deployed case). The dumps with per-position margins
  + N_b grids exist (capacity/capacity_dump_corrected_faithful_*.npz or regenerate via the
  corrected scripts). Same 8 dolly prompts / seed 20260611.
- Per-token afforded-capacity r_t at threshold b (and top-K parameter k): the per-position
  contribution to the capacity sum C = H(p) + (1-p)E[log2 N_b] + p*(H(q)+(1-q)log2 K + q log2(V-K)).
  Decompose per token:
    * if the served token is WITHIN margin b (compliant): r_t = log2(N_b(t))  [the
      multiplicity it afforded] ;
    * if it VIOLATES (served outside margin b): r_t = log2(K) if served in the integer
      model's top-K else log2(V-K)  [the violation alphabet] ;
    * plus amortize the H(p) "which-positions-violate" term: add the per-token binary-entropy
      share, i.e. distribute H(p_message) across tokens (state your exact amortization; the
      simplest is to add H(p)/1 as a per-token constant computed from the message's empirical
      p — be explicit).
  Cumulative R(N) = sum of r_t over N audited tokens. Use the CALIBRATED optimum (b*, k*) from
  the faithful corrected sweep (b* ~0.6, k*=4 region — confirm from the data); also report at
  k fixed conservatively (larger k / worst-case) so the bound isn't optimistic.

## Measure (benign workload)
1. Per-token r_t distribution: histogram, mean mu, variance sigma^2, range [min,max], skew/tail
   shape. Note it's heavily zero-inflated (most tokens N_b=1 -> 0 bits, no violation).
2. AUTOCORRELATION of r_t ALONG sequences (within prompt): the autocorrelation function and the
   integrated autocorrelation time tau (effective sample size N_eff = N/tau). This is the key
   non-i.i.d. correction — do NOT assume independence.
3. Cumulative R(N) statistics: for window sizes N in {64,256,1024,4096, and as large as the
   data allows by concatenating prompts}, measure Var(R(N)) empirically. Confirm whether
   Var(R(N)) ~ N*sigma^2 (i.i.d.) or grows differently (correlation) -> get the correlation-
   corrected variance slope.

## Derive the buffer-vs-FPR-vs-N curve
For target FPR eps = 1e-10 and audit size N tokens, the buffer = R*(N) - N*mu such that
P(R(N) - N*mu > buffer) <= eps for honest traffic. Because we only have ~8k real tokens we
CANNOT observe 1e-10 events directly -> use BOTH:
  (a) a CONCENTRATION bound: r_t is bounded (r_t <= log2 V per token); use a Bernstein /
      sub-exponential tail with the correlation-corrected variance (N_eff) -> a RIGOROUS
      (conservative) buffer. State the bound and constants.
  (b) a parametric tail fit (Gaussian and/or a heavier sub-exponential) to the empirical
      cumulative-R distribution at the largest observable N, extrapolated to eps=1e-10 ->
      a tighter ESTIMATE. Report both; the gap between them is the cost of not having more data.
  Cross-check both against a block-bootstrap of the real data at the FPRs we CAN observe (e.g.
  1e-3..1e-4) to validate the tail model.
Output the buffer(N) curve and the buffer-PER-TOKEN (buffer/N) curve (should fall ~1/sqrt(N))
for eps=1e-10. If generating more benign tokens is cheap (more dolly prompts through the
existing pipeline), do it to push the observable FPR lower and tighten the fit -- flag if you do.

## Deliverable
/workspace/projects/zk-hillclimb/BUFFER_FPR.md: the r_t distribution + autocorrelation (with
tau / N_eff), Var(R(N)) vs N (i.i.d. vs corrected), the buffer-vs-N and buffer-per-token-vs-N
curves for FPR 1e-10 (concentration bound AND tail-fit, with the block-bootstrap validation),
a plain-language bottom line ("to hold FPR<1e-10 you need buffer X bits/token after auditing N
tokens; per-token buffer drops below the benign mean once N > ..."), and HONEST caveats
(extrapolation assumptions, how many real tokens we have vs would want, autocorrelation impact,
the afforded-vs-detected distinction). Plots as PNG referenced from the md. Scripts under
capacity/ or a new buffer/ dir. /root/int-model-env/bin/python. No git commits; no pushes;
int-model-approximation READ ONLY.
