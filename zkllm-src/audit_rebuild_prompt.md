# Task: external-style independent soundness audit of the AS-BUILT batched-opening + weight-privacy code

OUR OWN defensive ZK codebase. The transport rebuild (Stages A-D) is complete: the
inline-IPA openings were replaced by a batched claim-opening protocol (zkob_batchopen +
zkob_claims.cuh), and weight privacy was added (hiding commitments + hidden weight claims +
ZK opening). The DESIGN was audited SOUND earlier (TRANSPORT_REVIEW.md). NOW audit the
AS-BUILT IMPLEMENTATION for soundness — does the code actually match the audited design, are
the required pins really present, and does the ZK/weight-privacy property hold? This is the
trust gate before we rely on the faster system.

Read: TRANSPORT_REBUILD_DESIGN.md (§2 protocol, §4 weight privacy), TRANSPORT_REVIEW.md
(the prior DESIGN audit + required pins F3-F6 + BO-1..BO-12 battery), STAGE_B/C1/C2/D reports;
the code: zkob_claims.cuh (claim/accumulator + batched fold + fast IPA + the hiding/blinding
wp_* helpers + zk_ipa), zkob_batchopen.cu (the batch-eval sumcheck + per-domain IPA + wbatch),
the claim-mode tails of zkob_fc.cu + zkob_rmsnorm.cu (registered-weight claims), and the
orchestrator verify_walk.py (the single-process batched verifier + edges + gate). Compare to
the 3 protected headers vrf_common.cuh/zkob_lookup.cuh (trusted) — confirm they're unmodified.

## Hunt (adversarial — find a cheating prover the AS-BUILT verifier accepts)
1. F3 pin AS-BUILT: does batchopen verify ACTUALLY check com_file_point_count == n_rows ==
   2^(vars-logG) BEFORE fold for EVERY tensor? (the covert-channel-critical pin). Trace it.
2. F4-F6 AS-BUILT: claims absorbed before the RLC challenge; verifier RECOMPUTES the claim
   list (claims_match) not trusting prover order; registered-weight claims carry the
   REGISTERED file path as comref. Verify in code, not just the report.
3. The batch-eval sumcheck reduction (§2.2) as-coded: RLC soundness, eq construction, no
   claim droppable/aliasable; the per-domain IPA binds. Does the fast me_weights/fold kernel
   match the slow path (the cross-check is real and on in production verify, not just selftest)?
4. Weight privacy (§4) as-built: is the hiding commitment actually hiding (blind included,
   independent generator H registered)? Do the masked weight-claim sumcheck rounds + the ZK
   IPA actually prevent weight-eval leakage — re-run/inspect the D4 leak scan logic: could a
   weight-MLE eval still be recoverable from any artifact (transcript, ipa, claims, blinds)?
   Is binding preserved under blinding (hiding+binding both hold)? Check the documented
   residual (rmsnorm g via public W=Rxg) is the ONLY leak, not a symptom of a broader hole.
5. Localization (the C2 fix): does a single tampered claim now reject at its CORRECT named
   id, or can a prover exploit the batch attribution to hide which obligation is corrupt?
6. Single-process verifier: any state leakage between obligations, any check that ran in the
   old per-subprocess path but is missing/weakened in the consolidated one? Fail-closed on
   crash/timeout/unexpected-rc?

You MAY build + run selftests (zkob_batchopen selftest / wselftest, the per-driver wpriv
selftests, the orchestrator batched selftest) to confirm claims, and construct your own
adversarial test under /tmp/audit2/. Deliverable: /workspace/projects/zk-hillclimb/
REBUILD_AUDIT.md — VERDICT (SOUND / ISSUES / BROKEN) for (a) batching, (b) weight privacy,
(c) localization; per finding file:line + a concrete cheating-prover sketch if one exists +
fix; for clean categories say what you checked. READ-ONLY except the review file + /tmp/audit2.
No git commits; no pushes; int-model-approximation untouched. Don't edit the protected headers.
