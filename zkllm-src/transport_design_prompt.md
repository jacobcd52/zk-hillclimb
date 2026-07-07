# Task: design the efficient, weight-private ZKP rebuild — transport architecture + weight privacy + test harness (DESIGN doc, then we implement)

CONTEXT: We have a WORKING, audited, sound ZK system for integerized llama-68m (11 CUDA
drivers, 3-stage orchestrator, separate verifier, forgery test suites — all in
/workspace/projects/zk-hillclimb/, see ORCHESTRATOR_DESIGN.md, PHASE0_NOTES.md §11-21,
and BACKEND_DECISION.md which you MUST read first). Measured: llama-68m seq 1024 proves
1062 s, VERIFIES 1999 s, 176 MB proof. The decision (BACKEND_DECISION §2): KEEP our stack,
rebuild the proof-TRANSPORT layer along DeepProve's architecture (claim accumulation + ONE
batched commitment opening), targeting verify 10-60 s and proof <=30 MB, WITHOUT changing
what each obligation proves and WITHOUT losing byte-equality chaining, homomorphic links,
the registered integerized model, or the zero-advice covert-channel closures. Plus the
weight-privacy endgame nobody else has built (Artemis arXiv:2409.12055 is the reference).

Your job: produce /workspace/projects/zk-hillclimb/TRANSPORT_REBUILD_DESIGN.md — a design
complete enough to implement from, in the style/rigor of our existing design docs
(STAGE3_FAITHFUL_DESIGN.md, ROPE_ATTENTION_DESIGN.md are the bar). Sections:

1. CURRENT TRANSPORT, PRECISELY. Read vrf_common.cuh + zkob_lookup.cuh: how openings work
   today (open_prove/open_verify, the IPA, the per-driver 12-17 openings, the FS schedule
   tail of each driver). Quantify where the 1999 s verify goes (use the profiling already
   in BACKEND_DECISION / transcript.json timing; if absent, estimate from opening counts).

2. THE BATCHED-OPENING PROTOCOL. Specify: each driver, instead of discharging IPA openings
   inline, EMITS terminal claims (commitment-id, evaluation point, claimed value) into a
   global accumulator; one batched opening (random-linear-combination over all claims at a
   shared challenge, single MSM / multi-open) discharges them. Give the exact protocol, the
   FS schedule change (one global accumulation phase after all obligations), and a WRITTEN
   SOUNDNESS ARGUMENT (RLC batching soundness; why binding survives; Schwartz-Zippel over
   the RLC). Decide: keep Pedersen/BLS12-381 + batched-IPA (one O(N) MSM verify), OR add a
   KZG-class PCS (HyperKZG via MIT arkworks/jolt) for log-verify of the single batch — give
   both with the verify/proof tradeoff, and RECOMMEND one. Confirm byte-equality chaining
   and homomorphic links (affine limb links, skip-connection point checks) are UNTOUCHED
   (they must be — that's why we didn't fork Basefold).

3. PER-DRIVER CHANGE LIST. For each of the 11 drivers + orchestrator: what changes (the
   opening tail only), what is invariant (sumcheck rounds, lookups, commitments). The
   re-validation cost: every driver's FS tail changes => every selftest + both audit FS
   walkthroughs re-run (the PHASE0 §13 rule). Be explicit this is the real cost.

4. WEIGHT-PRIVACY ENDGAME. The thing NO candidate (DeepProve/JOLT/zkGPT all leak weight-MLE
   evals at FS points — BACKEND_DECISION §1) actually built. Design blinded weight
   commitments + masked/ZK openings (hiding Pedersen + ZK-sumcheck masking, or the Artemis
   CP-SNARK commitment-consistency trick) for the REGISTERED WEIGHT commitments specifically
   (activations can stay non-hiding — the threat model hides weights, inputs/activations are
   the public statement). Quantify the prover/verifier/proof overhead. This can be a LATER
   stage but must be designed now so the transport rebuild doesn't preclude it.

5. THE TEST HARNESS (the user explicitly wants this). We ALREADY have: per-driver selftests
   with ~40 semantic forgeries + byte-tampers, the orchestrator selftest.sh with tamper
   phases, and the harness/ forgery suite. Specify how the rebuild is validated: (a) every
   existing selftest must still pass (honest ACCEPT + every forgery REJECT); (b) NEW
   forgery cases targeting the batched-opening protocol itself (a prover who accumulates an
   inconsistent claim, forges the RLC, omits a claim, swaps a commitment-id — each must be
   rejected by exactly the new check); (c) an end-to-end before/after: same llama-68m run,
   OLD transport vs NEW transport, both ACCEPT, verify-time and proof-size measured. Pin the
   pass/fail matrix.

6. STAGED IMPLEMENTATION PLAN. Order the work so each stage is independently validatable:
   e.g. Stage A = batched opening on ONE driver (zkob_fc) + its selftest; Stage B = the
   global accumulator across a 2-driver chain; Stage C = full orchestrator; Stage D =
   weight-privacy blinding. Estimate effort per stage.

7. HONEST COMPARISON TO OTHER STACKS (the user explicitly wants this, ongoing): a table —
   ours-after-rebuild vs DeepProve vs JOLT Atlas vs Artemis on: prove, verify, proof size,
   weight privacy (real?), integerization fidelity, covert-channel closure, license,
   extensibility. What we'd be BETTER at (weight privacy, zero-advice, registered integer
   model, auditable chaining) and WORSE at (raw verify if we skip KZG, GPU maturity).

READ-ONLY except writing that one design file. No code yet. No git commits; no pushes;
int-model-approximation untouched. Flag every uncertainty honestly.
