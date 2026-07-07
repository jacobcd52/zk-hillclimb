# Task: Stage A of the transport rebuild — the batched-opening primitive + zkob_fc claim mode, validated

Implement Stage A of /workspace/projects/zk-hillclimb/TRANSPORT_REBUILD_DESIGN.md (§6
Stage A), incorporating the audit's REQUIRED PINS from TRANSPORT_REVIEW.md. Read BOTH
fully first, plus PHASE0_NOTES.md §10-13 (IPA/commitment/FS conventions) and
vrf_common.cuh + zkob_lookup.cuh + zkob_fc.cu (the current open primitive + fc driver).
This is trust-critical, soundness-gated work — match the rigor of our existing drivers.

## Deliverables
1. **vrf_toy_batchopen.cu** — a standalone toy that pins the batched-opening protocol
   (§2.1/§2.2) at small scale against BRUTE FORCE: build a handful of small committed
   tensors across 2 generator domains, emit claims (commitment-id, point, eval, domain)
   into the accumulator, run the batch-evaluation sumcheck (RLC-of-eq reduction) + one
   IPA per domain, and CHECK the batched result equals the sum of the individual opens
   computed directly. Probe every new kernel shape at runtime (the -dlto miscompile rule).
   Pin the eq-embedding/orientation/RLC-challenge derivation against an independent
   brute-force recomputation. Must print PASS/FAIL per case.
2. **Claim + accumulator serialization** (claims.bin format per §2.1) — with the EvalVar
   variant tag and the H-slot registration touch (§4.4 forward-compat for Stage D), so
   the format does not need changing later.
3. **zkob_batchopen.cu** — prove/verify of the batch over a real accumulator. The verify
   MUST implement the audit's required pins:
   - **F3 (CRITICAL covert-channel pin):** per distinct tensor, check
     `com_file_point_count == n_rows == 2^{vars_j - logG_j}` BEFORE fold_chain — restoring
     open_verify's size check at the new location. Without this, trailing prover-chosen
     commitment rows are silently accepted = an unconstrained covert channel. Do NOT skip.
   - **F4-F6 (transcript/plumbing pins):** every claim absorbed before the RLC/batch
     challenge is squeezed; the verifier RECOMPUTES the claim list (claims_match) rather
     than trusting prover order; W-claims that discharge *.commitment_opening ids carry
     the REGISTERED file path as comref (per F5/F6 in the review). Follow the review's
     exact wording for each.
4. **zkob_fc claim mode** behind a flag: instead of its inline IPA openings, fc emits its
   terminal claims into the accumulator; the OLD inline-opening tail stays compilable
   (flag-selected) until Stage C flips the default. Update zkob_fc's selftest to run BOTH
   modes: old-tail honest+forgeries still pass; new claim-mode honest ACCEPT + the same
   forgeries REJECT (now via the batch).
5. **Per-phase verify instrumentation** (closing the profiling gap §1.3): a separate
   *_prof binary or env-guarded timers splitting verify into round-checks / claims_match /
   fold / IPA, so Stage B can measure the speedup.

## Gates (must pass before Stage A is "done")
- T1: vrf_toy_batchopen ALL PASS (batched == sum-of-individual, brute-force pins hold);
  zkob_batchopen selftest ALL PASS including F3/F4-F6 negative tests (a proof with extra
  trailing commitment rows REJECTS; a dropped/reordered/misattributed claim REJECTS; an
  RLC-cancellation attempt REJECTS) — these are BO-battery items from §5.2, fold in the
  F8 locus correction and F10 (the F3 trailing-rows test).
- T2: on ONE zkob_fc instance at real scale (1024x768x1024 or the lm_head 768x32000),
  measure verify time old-tail vs claim-mode+batch — report the speedup (the design
  predicts the inline IPAs are ~90% of verify; confirm or refute on this one driver).

## Deliverable doc
/root/zkllm/STAGE_A_REPORT.md: what was built, T1/T2 results with numbers, the F3-F6 pins
as-implemented (file:line), any deviations from the design (flag honestly), and whether
the T2 speedup supports the §2.7 verify projection. Copy the new .cu files to
/workspace/projects/zk-hillclimb/zkllm-src/ when passing. NO git commits (coordinator
commits). No header edits unless the design explicitly requires one (if so, flag it
loudly — it triggers the all-driver re-validation rule). Build with the pinned -dlto
commands. No GitHub; int-model-approximation untouched. GPU is free. Iterate until T1/T2
pass; if a design step proves unsound or infeasible, STOP and write the blocker.
