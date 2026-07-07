# Task: redo the covert-capacity sweep with the CORRECT threat-model orientation

The existing capacity sweep (/workspace/projects/zk-hillclimb/CAPACITY_SWEEP.md,
CAPACITY_TOPK.md, scripts in capacity/) computed margins with the reference/served roles
SWAPPED relative to the real threat model. Fix it.

CORRECT threat model: the datacenter SERVES tokens from the fast quantized model (FP8) —
that is what network taps observe. The ZKP proves the INTEGER model M_int. The verifier
checks served-FP8-tokens against the proven M_int logits within margin b. So:
- REFERENCE (verifier's logits) = M_int (the proven integer model)  [was: FP8 teacher]
- SERVED token (honest) = argmax of the FP8 fast model               [was: M_int argmax]
- margin_t = (M_int's post-Gumbel preferred-token score) − (M_int's post-Gumbel score of
  the FP8-served token), i.e. computed UNDER M_int's logits, for the FP8 argmax token.
- p(b) = fraction with margin_t > b (the honest violation rate the protocol must tolerate
  because FP8 legitimately disagrees with M_int).
- N_b(t) = #{ vocab v : (M_int preferred score) − (M_int post-Gumbel score of v) ≤ b }
  (within-margin count under M_int's logits).
Everything else (the C(b) and C_topK five-term formula, the b-sweep, the K-sweep, the
limit self-checks, the plots) stays the SAME structure — only the two models' roles flip.

Both models' per-position logits are already produced by the measure/ pipeline
(int_chain.py = M_int logits; the FP8 teacher dumps z_ref_*.npy). Reuse them; do NOT
re-run training/proving. Schemes: faithful, codebook (and baseline for completeness).
Same 8 dolly prompts, seed 20260611.

Deliverable: /workspace/projects/zk-hillclimb/CAPACITY_CORRECTED.md —
1. The corrected headline table (min_b C and min_b C_topK per scheme) SIDE BY SIDE with
   the old swapped numbers, so we see how much the orientation mattered.
2. The five-component breakdown at the corrected top-K optimum per scheme.
3. The corrected U-curve plots (C vs b) + the K-sweep, same as before.
4. A short note: did the orientation change the conclusion (codebook < faithful << baseline;
   term-b dominance)? Quantify any qualitative change. Confirm limit self-checks still pass.
Honest about ties/approximations. Scripts under capacity/. /root/int-model-env/bin/python.
No git commits; no pushes; int-model-approximation READ ONLY.
