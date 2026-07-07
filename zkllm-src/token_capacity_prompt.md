# Task: measure TOKEN-LEVEL covert capacity of the rmsnorm ±1 channel + the DiFR-margin channel

Context (read /workspace/projects/zk-hillclimb/THREAT_MODEL_NOTES.md first — this task
fills in its TBD cells): our ZK verifier for an integerized llama-68m forward pass tolerates
exactly one freedom on the prover side: the per-row inverse-RMS integer R may be any value
satisfying (R−1)²·M ≤ 2^64·C ≤ (R+1)²·M — i.e. R_true and usually its ±1 neighbors
(~1.6 bits/row of TENSOR freedom; rows = sequence positions; one R vector per rmsnorm site).
Everything else is exact. The question this task answers: **how many bits per GREEDY token
can an adversary actually push through that freedom?** Plus the same question for the
verifier's logit-tolerance δ (the DiFR-margin channel).

## Setup
- Pipeline: /workspace/projects/int-model-approximation, file m68-pipeline.py (READ ONLY —
  copy what you need to /tmp/token-capacity/ and work there; never modify or push that repo).
  Model JackFram/llama-68m (HF cache is warm). Identify every rmsnorm site in the forward
  pass (per layer: input norm + post-attention norm; plus the final norm — confirm the count
  in the code) and how its inverse advice enters (rms_inv_temp or equivalent).
- Inputs: the three real-text 1024-token contexts used in
  /workspace/projects/zk-hillclimb/SCORES_RANGE.md (wiki, lorem, code) — reproduce that
  loading approach. Random-Gaussian inputs understate real behavior; use real text.
- GPU: RTX 4090, shared with another job — keep batch sizes modest, retry on OOM.

## Experiment A — rmsnorm ±1 channel → token flips
1. For each rmsnorm site and row, compute the ACCEPTED SET of R values (integer bracket —
   compute exactly with integer math: all R with (R−1)²M ≤ 2^64C ≤ (R+1)²M; report the
   distribution of accepted-set sizes — typically 2-3).
2. Baseline: run the pipeline forward pass with R_true everywhere; record the greedy
   (argmax) token and the top-2 logit margin at every position.
3. Perturbation runs (the adversary's reach):
   a. Coarse: all-rows R_true+1 (where accepted) per site, all-rows −1, all sites combined
      both ways — does ANY argmax flip at ANY position? Record logit-delta distribution
      (max and per-position) per variant.
   b. Targeted: for the positions with the SMALLEST baseline top-2 margins (say the 50
      smallest across the 3 inputs), search greedily over per-row choices to flip that
      position's argmax: rank rows by single-row influence on the target logit gap (probe:
      flip one row at a time among, say, the 64 most-attended rows), then accumulate the
      helpful flips. Report: how many positions of the 50 are flippable, and the minimum
      margin that survived all attacks.
4. Capacity estimates, stated honestly with method:
   - LOWER bound (constructive): bits/token = log2(#distinct reachable token sequences) /
     #positions from the flips you actually achieved (if k independent positions are
     flippable per 1024, that's ~k/1024 bits/token; account for the adversary needing the
     SAME proof to stay valid — all flips come from ONE choice of R vectors, so check which
     flips are simultaneously achievable).
   - UPPER bound: tensor freedom (1.6 bits/row × rows × sites) is the trivial ceiling;
     tighten it if you can argue a better information-theoretic bound from the measured
     logit sensitivities (e.g., if max |logit delta| << min achievable margin quantile,
     the channel is provably near-0 for most positions). Be rigorous about what is proven
     vs estimated.

## Experiment B — DiFR logit-tolerance channel (the δ-curve)
Independent of ZK: if the token check accepts any served token whose logit is within δ of
the argmax logit, the adversary gets a choice wherever the top-2 margin < δ.
1. From the baseline margins (A.2): plot/curve the fraction of positions with top-2 margin
   < δ, and more generally E[log2(#tokens within δ of top-1)] as a function of δ — that
   expectation IS the channel capacity in bits/token at tolerance δ.
2. Produce the curve for δ from 1e-4 to 1.0 (log-spaced), per input and pooled.
3. Mark on the curve: the δ that DiFR actually uses in the harness
   (/workspace/projects/zk-hillclimb/harness/ — find the margin threshold in score.py or
   HARNESS.md; if the harness uses a different acceptance form, e.g. relative margin or
   rank-based, compute the analogous curve for THAT form and say so explicitly).

## Deliverable
/workspace/projects/zk-hillclimb/TOKEN_CAPACITY.md:
- Accepted-set size distribution; logit-delta distributions for coarse and targeted attacks;
  flippable-position count; the two capacity bounds for the rmsnorm channel with methods.
- The δ-curve (bits/token vs δ) with the harness's actual operating point marked, as BOTH
  a small table and a saved matplotlib PNG (same directory, referenced from the .md).
- A 5-line executive summary at the top: rmsnorm-channel token-level capacity (bounds),
  DiFR-channel capacity at the operating point, and WHICH CHANNEL DOMINATES.
- Exact commands/scripts used (save scripts under /workspace/projects/zk-hillclimb/
  capacity-measure/), seeds, and honest caveats. If something cannot be measured, say what
  and why rather than guessing.
