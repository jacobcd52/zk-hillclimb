# Task: measure end-to-end DiFR of the integer witness chain (post witness-authority switch)

The ZK orchestrator's integer chain now REPLACES the float pipeline path in three places
(pinned witness-authority rule): rmsnorm advice R (exact integer bracket), softmax P
(integer spec, SOFTMAX_DESIGN §2), and RoPE (integer spec, ROPE_ATTENTION_DESIGN §2.2,
registered int cos/sin tables) — plus driver-semantics rescale rounding (round-half-up)
throughout. Nobody has measured the END-TO-END DiFR of this integerized model against the
float teacher since the switch. That number is owed before any approximation claim
(ROPE_ATTENTION_DESIGN §9.6) and it seeds the Pareto chart's first point ("baseline-native").

## How
1. Read /workspace/projects/zk-hillclimb/harness/HARNESS.md + score.py — the FROZEN DiFR
   protocol (which inputs, which teacher, the score definition, held-out discipline). Use
   THAT protocol exactly; do not invent a variant. Read THREAT_MODEL_NOTES.md for framing.
2. The integer-chain activations already exist: the stage-2 orchestrator run wrote every
   chain tensor under /root/zkorch/<latest stage-2 run>/data/ (and ORCHESTRATOR_DESIGN §1
   documents the layout). Reuse them if the harness inputs match; otherwise regenerate
   the witness chain via orchestrator/prove_walk.py's witness functions (import them —
   do NOT re-prove anything; proving is not needed for DiFR) on the harness's prescribed
   inputs. NOTE a 60+ min GPU selftest is running — your work is light (one 68M forward
   pass per input); use the GPU lock /tmp/zkorch.gpu.lock for any GPU step, or CPU.
3. Complete the chain to logits the way m68-pipeline.py does (final norm + lm_head — read
   the pipeline, READ ONLY repo, copy to /tmp if instrumenting). Where the proof chain has
   integer-spec semantics, use the integer-spec values; where it has no semantics yet
   (final norm advice + lm_head), follow the pipeline's existing integer path (note in the
   report exactly which segments are pipeline-float vs integer-spec authority).
4. Score DiFR per the harness against the float teacher on the harness's eval inputs.
   Also record max|logit delta| and argmax-flip count vs the float teacher (connects to
   TOKEN_CAPACITY.md's margins).

## Deliverable
/workspace/projects/zk-hillclimb/DIFR_BASELINE_NATIVE.md: the DiFR score (the protocol's
exact metric + any auxiliary stats), logit-delta distribution, argmax flips, which chain
segments carried which authority, exact commands/scripts (save under capacity-measure/ or
a new measure/ dir), comparison against any pre-switch DiFR number found in
results/LEDGER.md or harness history (cite file), honest caveats. If the harness protocol
cannot be followed exactly, say precisely why and stop rather than approximating.
No git commits; no pushes; never modify int-model-approximation.
