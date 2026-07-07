# Task: Stage D — weight privacy (the differentiator nobody else built)

The transport rebuild core is DONE (Stage C2 committed: full llama-68m forward pass proves
in 636s, VERIFIES in 27s, full re-val 36/0 ALL PASS). Now add the property NO other zkML
stack actually implements (BACKEND_DECISION.md: DeepProve/JOLT/zkGPT all leak weight-MLE
evals at FS points): FORMAL WEIGHT PRIVACY — the registered weights must be hidden, with
proofs that reveal nothing about them beyond the committed hash.

Read first: TRANSPORT_REBUILD_DESIGN.md §4 (the Stage-D design: hiding Pedersen rows +
masked weight-claim sub-batch + ZK final opening; the accumulator carries the EvalVar/H-slot
variant tag from Stage A specifically so D drops in without a format change), §4.1 (the
current leakage accounting), §4.4 (forward-compat obligations already in place);
TRANSPORT_REVIEW.md F7 (the Stage-D underspecification the audit flagged — RESOLVE it);
Artemis (arXiv 2409.12055) as the commit-and-prove reference; PHASE0_NOTES §10 (commitment
conventions); zkob_claims.cuh + zkob_batchopen.cu (where the batch opening lives) + zkob_fc.cu
+ zkob_rmsnorm.cu (the two drivers whose openings touch REGISTERED weights — q/k/v/gate/up/
down/o_proj/lm_head W, and rmsnorm g).

## Scope (weights only; activations/inputs are the PUBLIC statement, stay non-hiding)
Per TRANSPORT_REBUILD_DESIGN §4.2, in staged sub-steps each independently validatable:
1. D1 — blinded weight registration: registered weight commitments become HIDING Pedersen
   (com = <g, w> + r·h, fresh blind r per weight, h an independent generator). Registration
   stores the blinds privately; public.json keeps the (now hiding) commitment hash. Verify
   the hash-pin + chain still work (the blind is part of the committed value).
2. D2 — masks on the weight claims: the batch-eval sumcheck rounds that involve a weight
   polynomial get ZK masking (a random mask polynomial added, its commitment absorbed, so
   the revealed evaluations are blinded). fc + rmsnorm weight claims only.
3. D3 — ZK final opening: the single batched IPA over the weight sub-batch becomes a
   hiding/ZK opening (blinded so the final a_final + L/R reveal nothing about w). Activations'
   sub-batch stays the fast non-hiding opening.
4. D4 — confirm the leakage is closed: a regression that checks NO weight-MLE evaluation
   appears in plaintext in any proof artifact for the weight sub-batch (grep the transcript/
   ipa files for the known weight evals; they must be absent/blinded).

## Hard rules
- Keep honest ACCEPT + every forgery REJECT (run the batched selftest after each sub-step).
- Don't edit the 3 protected headers (vrf_common.cuh, zkob_lookup.cuh); zkob_claims.cuh edits
  allowed with the rebuild-all-includers rule. Probe new kernel shapes; the hiding commitment
  needs an independent generator h — generate it deterministically (hash-to-curve or a pinned
  ppgen extra point) and register it. NO new G1 kernel unless unavoidable (flag loudly; the
  -dlto miscompile risk).
- Soundness: blinding must not weaken binding (Pedersen hiding+binding both hold under DLOG).
  Write the ZK/soundness argument for each sub-step to be audited later.
- This is genuinely hard + new crypto. If a sub-step proves infeasible in the time, land the
  ones that work, and DOCUMENT precisely where you stopped and why — partial weight privacy
  (e.g. D1+D2 done, D3 pending) is real progress. Do NOT fake it.

## Deliverable
/root/zkllm/STAGE_D_REPORT.md: per sub-step D1-D4 — implemented Y/N, the ZK/soundness
argument, the selftest result after it, the D4 leakage-regression outcome (does a weight eval
still leak? quantify), overhead (prove/verify/proof delta), deviations, what's left. Copy
changed sources to /workspace/projects/zk-hillclimb/zkllm-src/. RUN the batched selftest
yourself and paste verdicts. NO git commits. No GitHub; int-model-approximation untouched.
GPU free. If turn limit looms, land + document partial progress and leave a clean handoff.
