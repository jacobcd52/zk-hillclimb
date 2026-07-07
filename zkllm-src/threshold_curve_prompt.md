# Task: the treaty-threshold curve — (benign capacity mu + buffer) vs audit size N, for FPR 1e-10, faithful AND codebook, with more benign tokens

We have the buffer analysis (BUFFER_FPR.md) for FAITHFUL only, on ~8 prompts (~8k tokens).
The user wants THE operational plot and it computed for BOTH schemes with more tokens:

## The plot (the deliverable)
- x-axis: N = number of tokens over which the treaty averages the per-token channel capacity.
- y-axis: THRESHOLD(N) = benign mean capacity mu + buffer(N), where buffer(N) is chosen so
  the false-positive rate (honest datacenter wrongly exceeding threshold -> treaty torn up)
  is <= 1e-10. THRESHOLD(N) is a DECREASING curve approaching mu as N grows (more tokens ->
  tighter variance -> smaller buffer).
- Two curves overlaid: FAITHFUL (mu~0.31) and CODEBOOK (mu~0.23), each approaching its own mu.
- Interpretation to state: at audit size N, a datacenter whose measured average per-token
  capacity exceeds THRESHOLD(N) is torn up; the threshold relaxes toward mu as you audit more.

## Method (reuse BUFFER_FPR.md's machinery + scripts in capacity/ and buffer/)
- Per-token afforded capacity r_t = SAME definition as BUFFER_FPR (verifiable sampled regime,
  T=1, shared-seed Gumbel; reference = M_int post-Gumbel, served = the fast model's argmax;
  at each scheme's calibrated optimum b*,K*). Note in the doc that this is the benign-workload
  capacity arising from the M_int-vs-served-model gap (NOT a greedy number; per the corrected
  regime in THREAT_MODEL_NOTES.md §0/§1).
- GENERATE MORE BENIGN TOKENS: run the existing pipeline on MORE dolly prompts (target ~128
  prompts ~= 130k tokens, or as many as run cheaply on llama-68m) for BOTH faithful and
  codebook, same sampled/T=1/seed construction. This (a) lets the tail be observed to lower
  FPR (~1e-4..1e-5) for a more reliable extrapolation, and (b) pins the cumulative-variance
  scaling (the within-prompt autocorrelation / N_eff ~ N/7.5 finding) more tightly so the
  1e-10 extrapolation is grounded.
- buffer(N) for FPR=1e-10: report BOTH the rigorous concentration bound (Bernstein/sub-exp,
  bounded r_t, correlation-corrected variance) AND the validated tail-fit, as BUFFER_FPR did;
  cross-check via block-bootstrap at observable FPRs. The user specifically wants: "using a
  reasonably small number of tokens, estimate how many tokens we'd need for the 1e-10 number"
  -> so SHOW the variance-scaling fit and the implied N-to-reach-within-X%-of-mu.

## Deliverable
Update BUFFER_FPR.md (or a new THRESHOLD_CURVE.md) with: the THRESHOLD(N)-vs-N plot (both
schemes, saved PNG), the per-scheme mu, the variance-scaling fit (with the larger token set),
the "N needed to get within 10%/1% of mu at FPR 1e-10" numbers, and a plain-language reading.
Keep the rigorous-vs-tail-fit honesty. Note how many benign tokens were generated.
/root/int-model-env/bin/python. Run output / token dumps on /workspace if large (the
671TB volume; binaries stay on local disk but this is pure python/data so /workspace is fine).
No git commits; no pushes; int-model-approximation READ ONLY.
