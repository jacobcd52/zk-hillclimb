# Task: FINAL design — exact-max primitive + lm_head/logit binding (stage 3) + faithful-architecture revision (design only)

Two consumers need one new primitive; design all of it in one document. Quality bar:
SOFTMAX_DESIGN.md / ROPE_ATTENTION_DESIGN.md (read both as registers). Implementer must
make zero design decisions.

## Context (read all)
1. /workspace/projects/zk-hillclimb/DIFR_BASELINE_NATIVE.md — the measurement driving
   this: integerization floor 2.4e-6 nats; the 9-nat gap = three pipeline quirks
   (no o_proj, line-157 scramble, temperature 128). §7 caveat 6 = the strategy.
2. SOFTMAX_DESIGN.md (esp. §3.1 why max-shift was eliminated and §9.7 the temp-8
   warning), ROPE_ATTENTION_DESIGN.md (§1.3 the π permutation, §9.4 the fix note),
   PHASE0_NOTES.md §1-18, ORCHESTRATOR_DESIGN.md (+stage-2 section of the report),
   THREAT_MODEL_NOTES.md (greedy argmax binding is the pinned final statement),
   harness/manifest_llama68m.json (the 2 remaining ids + what o_proj waiving says),
   m68-pipeline.py READ-ONLY (final norm lines ~124-127, lm_head path, o_proj absence).
3. Machinery: zkob_lookup.cuh + vrf_common.cuh as-is (no edits); all 9 validated drivers.

## Part A — the exact-max primitive (new driver, zkob_rowmax or similar)
Prove, with ZERO advice freedom, that a committed per-row scalar mx[i] equals
max_{j ∈ allowed(i)} z[i,j] over a committed grid — via a committed one-hot selector:
S ∈ {0,1} (range/lookup), Σ_j S[i,j] = 1 per row (sumcheck against an eq-weight),
⟨S[i,·], z[i,·]⟩ = mx[i] (hadamard-style sumcheck), and ∀ allowed j: mx[i] − z[i,j] ≥ 0
(range lookup; bound the width). Handle the two masking regimes its consumers need:
(a) causal mask (softmax rows: allowed = j ≤ i — the MK machinery), (b) column padding
(vocab 32000 in 32768: pad columns must not win the max — pin exactly how, e.g. weights
zero + range proof only over real columns, or pad with -inf sentinel committed as a
public constant; decide and justify). Selector S must also be proven 0 ON masked/pad
positions (else a masked column could "win"). Full FS schedule, bounds, CLI, selftest
plan with an evil mode per check (wrong max too high, too low, selector at masked pos,
selector not one-hot, ...). Cost analysis at both shapes: 24x (1024x1024 causal) and
1x (1024x32768 vocab).

## Part B — stage 3: close the manifest (lm_head + final statement)
- final_norm.rmsnorm: the existing zkob_rmsnorm trio at the final-norm site (advice R
  switches to the exact bracket — same authority rule; check the pipeline's final-norm
  eps/semantics and pin C_eps).
- lm_head: registered weight (vocab 32000 → OUT_pad 32768, NEW gen32768 via ppgen —
  size/feasibility check: 32768-gen MSM cost, commitment sizes, predicted prove/verify
  time for one 1024x768x32000 zkob_fc + rescale; if gen32768 is infeasible on 24GB,
  design the split (e.g. 4x8192 column blocks via the headslice double-opening pattern)
  — decide ONE way and pin it).
- statement.logit_binding: the greedy pinned statement (THREAT_MODEL_NOTES §1): served
  token t* = argmax_v logits[pos, v] — Part A's primitive at the vocab shape, applied at
  which positions (pin: the harness's scored positions), plus how t* enters the public
  statement (absorbed into transcript; recorded in public.json/transcript.json).
- Orchestrator wiring: new edges, registration additions, manifest accounting to
  56/56 with 0 skipped (or state exactly what remains waived and why).

## Part C — the faithful-architecture revision (the first hill-climb submission)
Design the pipeline-authority changes as ONE coherent re-registration ("submission:
faithful-arch-v1"), each with its proof-side delta:
1. o_proj: add per-layer output projection (registered weight, fc + rescale; slot
   between headmerge and attn_skip — new edges; manifest ids exist but are waived —
   how does the submission's manifest accounting work vs the frozen harness manifest?
   Pin it per HARNESS.md's submission rules).
2. line-157 fix: headmerge gather becomes plain concat (pin the exact formula change,
   driver flag vs constant — NO new driver if avoidable; re-registration implications).
3. temperature 8: softmax exponent = score_real/8. With max-shift mx from Part A bound
   exactly (covert capacity stays 0), exponents land in (-inf, 0] — redesign the exp
   table/domain (what width covers (z − mx) at scale 2^9? table E(v) for v ∈ [−LEN, 0];
   what happens to S bounds, P bracket, limb widths — redo the SOFTMAX_DESIGN §6
   arithmetic for temp 8). Decide whether this is a revision of zkob_softmax (flag:
   editing a validated driver = full revalidation) or a new zkob_softmax8 driver
   (preferred if the diff is structural); pin selftest deltas.
4. Predicted end state: DiFR expectation (~the 0.0077-0.008 linears-only floor — argue
   it), added prove/verify cost (Part A instances + o_proj + table changes), and the
   before/after Pareto points table.

## Deliverable
/workspace/projects/zk-hillclimb/STAGE3_FAITHFUL_DESIGN.md, sections: 1 pipeline
semantics (quoted); 2 Part A spec (statement, obligations, FS, bounds, CLI, selftest);
3 Part B (same rigor); 4 Part C (same rigor + the submission/re-registration mechanics);
5 cost & Pareto predictions; 6 open questions (honest). Design only; no code. READ-ONLY
except that one file. GPU not needed.
