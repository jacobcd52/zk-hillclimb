# Task: measure covert-channel capacity C(b) for llama-68m under a few integerization schemes, with the b-sweep U-curve

GOAL: for a given integerization M_int of llama-68m, compute the per-token covert-channel
capacity an adversary (the datacenter, which has the exact model) could exploit, as a
function of an acceptance threshold b, and find its worst-case (min over b). Produce plots.

## The capacity formula (USE THIS EXACTLY; it was derived/corrected with the user — see
## the DiFR paper arXiv:2511.20621 and Rinberg's weight-exfiltration post for grounding)
Reference model = M_true = the FP8 teacher (what SHOULD be served). Student = M_int (the
proven integerized model). Per token position t, using the SAME post-Gumbel margin the
existing DiFR harness already computes against the teacher:
  margin_t = (teacher's preferred-token post-Gumbel score) - (post-Gumbel score of the
             token M_int serves)   [>= 0; this is exactly measure/'s post_gumbel_margin
             of M_int vs the FP8 teacher — REUSE that machinery, do not reinvent]
Define p(b) = fraction of tokens with margin_t > b   (the "violation"/exceed fraction;
DECREASING in b; at b=0, p = 1 - argmax_agreement). Define
  N_b(t) = #{ vocab tokens v : (teacher preferred score) - (post-Gumbel score of v) <= b }
           (count of tokens "within the margin" at position t; mostly 1).
Per-token capacity:
  C(b) = H(p)  +  (1-p) * E_t[ log2 N_b(t) | margin_t <= b ]  +  p * log2(V)
where H is binary Shannon entropy (bits), V = vocab = 32000.
  - term 1 H(p): which positions the adversary violates (must match honest rate).
  - term 2: compliant tokens pick any of N_b within-margin tokens (mostly 0 bits).
  - term 3: violating tokens pick freely from vocab.
ALSO produce the Rinberg refinement variant (chain rule applied twice): split the
violation choice into "violate within top-K (K=16)" vs "violate into the tail", so the
common violation pays log2(K) instead of log2(V):
  C_topK(b) = H(p) + (1-p)*E[log2 N_b] + p*( H(q) + (1-q)*log2(K) + q*log2(V-K) )
  where q = fraction of violations that go into the tail (measure it from the data:
  among violating tokens, how often is the served token outside the teacher's top-K).
SELF-CHECK the formula at limits before trusting it: b=0 (p = 1-argmax_agree, N_0~1, so
C(0) ~ H(p0) + p0*log2 V); b->inf (p->0, N_b->V, C-> log2 V). Report these endpoints and
confirm the curve is U-shaped (or report honestly if it isn't). Report min_b C(b) and the
argmin b* — THIS is the headline capacity for the integerization.

## Schemes to measure (do as many as are readily available; AT MINIMUM the first two)
1. faithful-arch-v1 (the faithful fixed-point integerization, temp-8, o_proj, plain
   concat) — DiFR 0.0156 vs teacher. Dumps/scripts in
   /workspace/projects/zk-hillclimb/measure/ and the run /root/zkorch/stage3v2-fa.
2. baseline-native (the zkLLM-style pipeline integerization: no o_proj, line-157 scramble,
   temp-128) — DiFR 8.988 vs teacher. Same measure/ infra; run /root/zkorch/stage3v2.
3. codebook integerization, IF available: check /workspace/projects/int-model-approximation
   (READ ONLY, never modify/push) for a codebook integerization path of llama-68m — the
   results/llama_pareto/llama_difr_results.json referenced a 'codebook' scheme (difr_mean
   ~0.0075). If you can produce per-token teacher-margin data under codebook with modest
   effort, include it; if not, say so and skip (don't force it).
The teacher logits + the per-position post-Gumbel margins for schemes 1-2 are essentially
already computed by measure/difr_baseline.py / int_chain.py — extend those to dump the
FULL per-position margin arrays AND the within-margin token counts N_b across a grid of b,
rather than just the aggregate DiFR mean. Same 8 dolly prompts, same seed 20260611, same
protocol (cite the seed-is-fixed caveat like DIFR_BASELINE_NATIVE does).

## Deliverable: /workspace/projects/zk-hillclimb/CAPACITY_SWEEP.md  +  plots in the same dir
- For each scheme: a table and PLOTS of C(b) vs b AND C vs p (both, the user wants all
  plots), for both the simple and top-K-refined formulas, with min_b C(b) and b*/p* marked.
- A combined plot overlaying the schemes' C(b) curves (capacity vs threshold) so the user
  can compare integerizations at a glance.
- The headline number per scheme: min-over-b capacity in bits/token, and bits/forward-pass
  (×1024) for context, with the realized-<<-capacity caveat (Rinberg: realized ~<0.5% of
  theoretical) stated.
- Exact commands/scripts (save under measure/ or a new capacity/ dir), honest caveats.
Save plots as PNG and reference them from the .md. Use /root/int-model-env/bin/python.
No git commits; no pushes; int-model-approximation READ ONLY.
