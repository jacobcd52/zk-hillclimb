# TRANSPORT_REVIEW — adversarial soundness audit of the batched-opening protocol design

Status: REVIEW, 2026-06-11. Audits TRANSPORT_REBUILD_DESIGN.md (the protocol in §2.1–2.4,
the privacy plan §4, the battery §5.2) BEFORE any code exists. Inputs read in full:
TRANSPORT_REBUILD_DESIGN.md; PHASE0_NOTES §7–§13 (pinned IPA/commitment/lookup/FS
conventions); vrf_common.cuh (fold_chain, ipa_prove/ipa_verify, me_weights,
fs_challenge_fr); zkob_lookup.cuh (open_prove/open_verify — the primitive the batch
replaces); fs_transcript.hpp; zkob_fc.cu verify path (lines 226–300, as the
representative driver); zkob_rmsnorm.cu (affine-link and com-size check sites);
ORCHESTRATOR_DESIGN run_seed/seed conventions. Method: for each of the eight audit
questions I attempted to construct a cheating prover the batched protocol accepts but
the current per-opening protocol would reject, then verified the algebra of the §2.2
reduction independently (completeness identity, degree bounds, terminal check,
per-group RLC/IPA binding).

---

## VERDICT: **SOUND-AS-DESIGNED**

No protocol-level soundness hole was found. I could not construct a cheating prover
that the batched protocol accepts where the per-opening protocol rejects: every attack
I tried dies on one of (a) the verifier-recomputed claim list (claims_match), (b) the
G0≺G1 / G3≺G4 absorb-before-squeeze ordering, (c) Schwartz–Zippel at the 2^-240 class,
or (d) the already-audited §10 IPA + Pedersen binding. The reduction in §2.2 is
algebraically correct (checked independently, below) and Lemmas 1–5 compose as claimed.

That said, the verdict is conditional on a short list of **required pins** — checks the
current `open_verify` performs that the design text does not explicitly restate for the
batch, and which MUST appear in Stage A or the implemented protocol is weaker than the
designed one (F3 is the sharpest: a dropped structural check would reopen a covert
channel). Plus: one locus error in the §5.2 battery (F8), several battery gaps
(F9–F12), and one genuine underspecification in the Stage-D sketch (F7) that does not
affect Stages A–C and is severable, exactly as the design claims. None of these
requires redesigning the protocol.

Findings index: F1–F2 clean-category notes with minor corrections; F3 required pin
(structural/covert-channel); F4–F6 required pins (transcript/plumbing hygiene); F7
Stage-D design gap; F8 battery locus correction; F9–F12 battery additions.

---

## 0. Independent check of the §2.2 reduction (algebra)

Done before attacking, since a forgery hunt against a wrong-as-written protocol is
meaningless. All four pieces check out:

1. **Completeness identity.** P̂_j(x_low, x_high) = P_j(x_low)·Π_h(1−x_high_h) is the
   MLE of "P_j on the embedded subcube, 0 elsewhere" (on the hypercube, Π(1−x_h) is the
   indicator of all-high-bits-zero). For multilinear P̂_j,
   Σ_{x∈{0,1}^m} eq(û,x)·P̂_j(x) = P̂_j(û), so
   Σ_x Σ_j M_j(x)P̂_j(x) = Σ_i ρ_i·P̂_{tensor(i)}(û_i) = Σ_i ρ_i·P_{tensor(i)}(point_i).
   The G2 identity holds for honest claims. Note M_j is supported only on tensor j's
   2^{vars_j} subcube (the high-bit eq factors are (1−x_h)), which is what makes the
   §2.2 streaming cost Σ_j 2^{vars_j} and not 2^25 per tensor — consistent.
2. **Degree bound.** M_j and P̂_j are each multilinear; per-variable round degree is 2;
   three evaluations per round encode the round polynomial with degree ≤ 2 *by
   construction* (the verifier never needs to enforce a degree cap separately). ✓
3. **Terminal.** P̂_j(r) = κ_j·P_j(r[0..vars_j)) with κ_j = Π_{h≥vars_j}(1−r_h); G3's
   check `cur == Σ_j M_j(r)·κ_j·v'_j` with M_j(r), κ_j verifier-computed is the correct
   evaluation of the verifier-DEFINED polynomial F(x) = Σ_j M_j(x)P̂_j^{committed}(x)
   once G5 forces v'_j = P_j^{committed}(r_low). This is the standard "sumcheck for a
   polynomial the verifier can evaluate at r via one oracle call, where G5 is the
   oracle" composition — sound.
4. **G5 consistency.** A_g := Σ_{j∈g} ρ'_j κ_j·rowfold(P_j, r[logG..vars_j)) satisfies
   ⟨A_g, me_weights(r[0..logG))⟩ = Σ_j ρ'_j κ_j P_j(r[0..vars_j)), matching v*_g for
   honest v'_j, and C*_g = ⟨gen_g, A_g⟩ by Pedersen linearity. The κ_j on BOTH sides is
   consistent (redundant for soundness but harmless). The column-bits-first flat layout
   is what makes one shared b-vector per domain correct; verified against fc's actual
   point layouts (W: u_output‖u_input, X: u_input‖u_batch, Y: u_output‖u_batch —
   zkob_fc.cu:271-300 matches the §3 table).

**κ_j = 0 degenerate case (checked, harmless):** if some r_h = 1 exactly for
h ≥ vars_j, tensor j's coefficient is 0 in BOTH G3 and G5, so its v'_j is unconstrained
— but its true contribution is also 0, so an unconstrained v'_j buys the prover
nothing: a false cur still needs some j with M_j(r)·κ_j ≠ 0 to absorb the discrepancy,
and that j's v'_j is then bound by G5. The prover cannot induce κ_j = 0 (r is squeezed
after everything the prover controls). Probability over honest r ≈ 25·m'/|C| —
negligible, and inside the stated union bound's class.

---

## 1. RLC batching soundness — CLEAN (with one cosmetic bound correction)

**Checked.** ρ is squeezed at G1 strictly after G0 absorbs: n, every claim tuple
(id‖domain‖n_rows‖point‖eval), every distinct comref hash, and every driver's final
transcript state. So every claim's (commitment-id, point, eval) is bound before the
batch challenge — in fact **double-bound**: each eval is absorbed into its own driver
transcript before the driver tail ends (verified at the source: fc absorbs `claim` at
zkob_fc.cu:139/249 and `claim_X, claim_W` at :197/269 before any IPA; the §11 lookup
schedule absorbs A_f/S_f/m_f likewise), and that transcript's final state is absorbed
as drvstate in G0.

**Cancellation attack (the question-1 forgery).** To craft per-claim evals whose δ_i
cancel in Σ δ_i ρ^i, the prover must know ρ before fixing at least one eval. But every
eval feeds ρ through two independent SHA-256 paths (G0 directly; driver absorb →
drvstate → G0). Choosing an eval as a function of ρ where ρ is a function of that eval
is a SHA-256 fixed-point/grinding problem; grinding is additionally useless because the
accepted-proof freedom elsewhere in the system is ~0 bits (the zero-advice closures),
so the prover cannot even cheaply re-randomize ρ. With ρ fixed first, Schwartz–Zippel
applies: g(Z) = Σ_i δ_i Z^i is nonzero of degree ≤ 1,535, and powers-of-ρ (vs n
independent challenges) is fine — the SZ bound is degree-driven.

**One correction to Lemma 3's constant (cosmetic).** fs_challenge_fr's distribution is
not uniform over a 2^254.86 set: the top limb is uint32 mod 1944954707, so residues
< 405,057,882 carry mass 3/2^32 and the rest 2/2^32. SZ over a non-uniform challenge
should be stated via the **max point mass** (≤ 3·2^-256 here): Pr[g(ρ)=0] ≤
1535·3·2^-256 ≈ 2^-243.8. Same class as the stated 2^-243.6; restate the lemma in
max-point-mass form so the argument is airtight as written. Same note applies to
Lemmas 4–5 and the round challenges.

---

## 2. Batch-evaluation sumcheck binding — CLEAN

**Checked.** Per-claim binding survives the reduction because (a) the claim multiset
entering G2 is verifier-recomputed (the prover's claims.bin is only ever byte-compared,
never consumed — see F5 for the pin that keeps it that way); (b) M_j is built per
tensor by the VERIFIER from (ρ, û_i), so no claim can be silently dropped (count and
order pinned, BO-2/BO-10), aliased to another tensor (M_j is keyed by the
verifier-derived tensor identity), or moved to a different point (û_i comes from the
verifier's own FS replay); (c) the terminal's only prover input is v'_j, absorbed at G3
before ρ', then forced to P_j^{committed}(r_low) by the group IPA. Claims sharing a
tensor (rope's two T-claims, rowmax's six S-claims) are summed inside one M_j with
distinct ρ powers — distinct false evals cannot mutually cancel except on the SZ event.

Chain-edge byte-duplicate commitments (com_Y ≡ next com_X as two files) become two
distinct tensors with separate M_j over the same underlying group element — redundant
but sound; pin the tensor identity key (canonical claim id/comref as recomputed) so
"distinct tensor" is never ambiguous (folded into F4).

**Lemma 4's prose should be tightened (editorial, the math is right).** The trailing
"OR all v'_j are correct, in which case ... formally:" sentence trails off. The correct
chain, which the design's checks do implement: if the initial sum is false then except
w.p. ≤ 2·m_max·(max point mass), cur ≠ Σ_j M_j(r)κ_j·P_j^{true}(r_low); hence either
G3 rejects, or Σ_j M_j(r)κ_j(v'_j − true_j) ≠ 0, which implies **∃j with κ_j ≠ 0 and
v'_j ≠ true_j** — exactly the precondition Lemma 5 needs (this also disposes of the
κ_j = 0 hideout, §0 above). Rewrite the lemma in that form during the T6 audit.

---

## 3. Cross-domain partition — CLEAN, with one REQUIRED PIN (F3)

**Checked.** Domain misattribution is not available to the prover: the claim's domain
and n_rows are verifier-recomputed (the driver knows its own generator file), gens are
loaded by the verifier from registration, and a claim forged into the wrong group in
claims.bin alone dies at claims_match. Structural confusion between the four IPAs is
also closed: ipa_verify requires n == 1<<rounds and |u_b| == log n (vrf_common.cuh:
342-344), and the four domains have four distinct sizes, so pasting ipa_batch_64.bin
into the gen1024 slot fails the round-count check deterministically. The shared-prefix
column points across groups (r[0..6) ⊂ r[0..10) ⊂ …) are independent public b-vectors
— no interaction.

**F3 (REQUIRED PIN — structural check, covert-channel relevant).** Today,
`open_verify` enforces `com.size == 1u << (u_pt.size() − logG)` (zkob_lookup.cuh:132)
before folding. The batch design **removes open_verify and does not restate this
check** in G5. Without it, `fold_chain` (vrf_common.cuh:144-163) consumes exactly
|us| challenges and returns d_a[0] regardless of remaining size — i.e., a commitment
file with **extra trailing rows is silently ignored by the fold** (the partial fold's
element 0 combines only the first 2^logR rows). Consequences if the check is lost:

- *Not a statement forgery*: the opened value still binds the first 2^logR rows, and
  per-driver pre-tail size checks (fc: zkob_fc.cu:235 "commitment row counts"; rmsnorm:
  zkob_rmsnorm.cu:478; both invariant under the rebuild) catch oversize files for the
  drivers that have them.
- *But a covert channel*: any commitment whose size is checked ONLY inside today's
  open_verify would, post-rebuild, accept unconstrained trailing rows — prover-chosen
  bytes in an ACCEPTED proof, size-unbounded. For a project whose whole point is
  closing covert channels, this is the one place the rebuild could silently reopen one.

**Fix (cheap, two lines of spec):** (i) zkob_batchopen verify checks, per distinct
tensor, `com_file_point_count == n_rows == 2^{vars_j − logG_j}` before fold_chain —
restoring open_verify's check at the new location; (ii) Stage C explicitly audits each
driver for any commitment whose only size check was the open_verify instance (fc and
rmsnorm are clean; the audit must cover all 11). Add the battery test (F10).

---

## 4. Homomorphic-link & byte-equality anchoring — CLEAN

**Checked.** The links and edges are point checks / byte checks on com files; what
makes them *meaningful* is that the tensors behind those files are pinned by opened
claims. The batch opens the SAME tuples against the SAME files: comref (path) +
sha256(file bytes) absorbed in G0, the file absorbed into the producing driver's
transcript (drvstate-bound), and G5's fold_chain runs over those bytes. Specifically
re-derived for the three classes named in §2.4:

- **rescale affine limb link** `com_X == sf·com_X̂ + com_rem`: rem stays anchored by
  the S_f-vs-com_rem claim (same tuple as today's IPA), A/m likewise; X̂ remains
  determined homomorphically — unchanged trust structure.
- **rmsnorm links** operate on com_L rows (zkob_rmsnorm.cu:499-510), and its tensors
  keep their 17 claims; the registered com_g is anchored by the val_g claim (:698's
  IPA becomes a batch claim against the same registered file).
- **skip ⊕ point checks**: zkob_skip has no openings; its operands are anchored by
  the NEIGHBOR drivers' claims, which all route through the batch against the same
  files. Anchor preserved.
- **headslice pair-equality** (the only driver consuming two claims' evals jointly,
  design §8.6): the driver checks eval_full == eval_slice on prover-supplied evals
  BEFORE the batch validates either. Jointly forging both by the same δ passes the
  equality and yields two false claims — caught by SZ exactly like BO-1. Sound; the
  design's choice to keep the equality driver-side is correct.

Two pins, folded into F5/F6 below: the W-claims that discharge the 15
`*.commitment_opening` manifest ids must carry the REGISTERED file path as comref (the
fc driver already opens W against the registered copy — zkob_fc.cu:233 — so the
recomputed claim will, but the orchestrator's claims_match should ASSERT the registered
comrefs appear, making the discharge explicit rather than emergent); and the
ACCEPT-conditional flow must be covered by a gating test (no consumer may treat a
conditional driver verdict as final — §5.2's BO-2 exercises this path implicitly; name
it, F12).

---

## 5. FS transcript completeness — CLEAN, with hygiene pins (F4–F6)

**Inventory of everything the prover supplies in phase G, with its binding point:**

| prover artifact | binding |
|---|---|
| claims.bin | byte-compare vs verifier-recomputed list ONLY; never enters the transcript or the batch computation |
| batch_sumcheck.bin round evals | absorbed per round before that round's challenge (G2) |
| batch_vfin.bin v'_j | absorbed at G3, before ρ' (G4) |
| ipa_batch_*.bin L,R | absorbed inside ipa_verify before each x (the §10 discipline) |
| ipa_batch_*.bin a_final | NOT absorbed — terminal message of its IPA, checked by the final point equation; nothing is derived after it within that IPA, and later groups' challenges need not bind it. Same convention as today's sequential per-driver IPAs. Sound. |

Challenges: ρ binds claims+comrefs+drvstates (G0≺G1 ✓); each r_t binds its round
message ✓; ρ' binds the v'_j (G3≺G4 ✓); group IPAs run sequentially on the same
transcript in pinned ascending order, structure derived from the (verifier's) claim
list ✓. Cross-run replay dies on the run_seed:opening_batch seed + drvstate absorbs
(BO-9) ✓. Nothing prover-supplied is left unabsorbed. The §13 rule holds as designed.

**F4 (pin — absorb encoding).** G0's per-claim absorb concatenates a variable-length
id with fixed-width fields inside ONE absorb call. fs::Transcript length-prefixes the
whole blob but not internal fields, so two different (id, fields) pairs could in
principle serialize identically. Not exploitable as designed — the verifier absorbs its
OWN recomputed list, so the prover never chooses these bytes — but it is exactly the
kind of latent ambiguity that becomes a hole if a refactor ever consumes the prover's
list. Pin: length-prefix the id inside the claim encoding, and absorb the EvalVar tag
explicitly (a v0 Plain eval and a v1 Committed eval must be domain-separated by tag,
not just by byte-length). Also pin the canonical tensor-identity key (see §2).

**F5 (pin — claims_match is load-bearing and must be inseparable).** If zkob_batchopen
verify ever consumed the prover's claims.bin with claims_match skipped or reordered, a
prover could simply omit its one false claim and batch the rest honestly — the batch
would accept, and ONLY claims_match catches it. The design already states the right
discipline ("the accumulator is verifier-recomputed content, never trusted"); pin two
implementation consequences: (i) zkob_batchopen verify takes the verifier-recomputed
list as its input, with the prover file used for the byte-compare only; (ii) the
claims_match result is a component of the `opening_batch` verdict itself (as the check
id `opening_batch.claims_match` already suggests), not a separate orchestrator check
that could be waived independently.

**F6 (pin — drvstate provenance).** The batch verifier must absorb drvstates produced
by the VERIFIER's own per-driver runs. In the multi-process Stage A–B world these
travel through files between verify processes; that channel must be verifier-internal
(written and read inside one verify invocation), never a prover artifact. The
single-process zkverify_walk (§2.7) makes this structural; until then it is a
convention the T6 audit must check. Also: cross-check or drop batch_sumcheck.bin's
redundant claim-count field, and pin the n_claims = 0 / empty-group / single-claim
structural edges (REJECT or trivially-pass semantics stated, not implied).

---

## 6. Zero-new-advice — HOLDS (conditional on F3)

**Checked.** Accepted-proof prover freedom added by phase G: round evals p(0..2) are
forced (a deviation changes cur off the honest polynomial and survives only on the SZ
event — same argument class as every existing sumcheck); v'_j are forced to
P_j(r_low) by G5; IPA L/R/a_final are forced by the final point equation (the §10
adaptive-forgery rejection); claims.bin carries 0 bits (verifier-recomputed, order
pinned); m_max, group set, tensor order are all derived from the claim list. The batch
adds 0 bits to the capacity layer, as §2.3 claims — **provided F3's structural check
lands**, since unconstrained trailing commitment rows would otherwise be a
size-unbounded channel through the accepted artifact set. (The CAPACITY_SWEEP
methodology should re-run over the new artifact set in Stage C as a regression — the
design's T8 only does this for Stage D; cheap to include in T5.)

---

## 7. Weight-privacy forward-compat (§4) — FORMAT OK; ONE STAGE-D DESIGN GAP (F7)

**Format check (what Stages A–C must not preclude): adequate.** EvalVar tag from day
one with the Committed path explicitly rejected until D; dual accumulators routed by
comref from Stage B; H slot in registration from Stage A; G5 grouping keyed on
(domain, comref-class) so weight and activation tensors never share an RLC. Nothing in
the A–C accumulator format blocks D1–D4. Spot-checks supporting D1: the registered
weight commitments appear in no affine link in rescale (link is com_X/com_X̂/com_rem)
or rmsnorm (links are on com_L rows), consistent with §4.2's claim; Stage D should
still sweep the full edge map as the design says. D3's sigma terminals are sound in
shape (claim_X public, linear-or-product-with-public relations). Pin: the weight
sub-batch needs its own transcript seed suffix and its own claims_match.

**F7 (Stage-D design gap — fixable, severable, no A–C impact).** D4 says the weight
sub-batch runs "the SAME reduction with two changes" (D2 masking + ZK final IPA). That
is one change short. With Committed evals, the batch's initial claim
Σ_i ρ_i v_i **exists only inside commitments** (the verifier can form
V* = Σ ρ_i C_{v_i} homomorphically but cannot see the scalar). Libra-style D2 masking
as described still runs its round checks in the clear against a PUBLIC running claim
(`p(0)+p(1) == cur`); with a hidden initial claim there is no public cur to check
against. The missing component: committed round messages with homomorphic round checks
— prover sends Pedersen commitments C_{p(0)}, C_{p(1)}, C_{p(2)}, verifier checks
C_{p(0)} + C_{p(1)} == C_cur and folds C_cur' = lagrange3(C_{p0},C_{p1},C_{p2})(r_t)
homomorphically (lagrange3 is LINEAR in the p's, so this works with blind tracking),
with one Schnorr proof for the terminal's H-component against the RLC of the C_v's.
This is implementable entirely with the existing Pedersen + sigma machinery — no new
assumptions — but it changes D4's cost line (≈ +3 G1 absorbs per round × ~25 rounds
plus blind bookkeeping) and it needs its own simulator write-up, which §8.8 already
demands for the masking. **Action: amend §4.2 D4 to name the committed-round-message
sumcheck before Stage D is scheduled.** The design's own framing ("designed now ... so
the rebuild cannot preclude it") survives: the accumulator format carries everything
needed; only the D4 protocol sketch was optimistic.

Also worth one honest line in §4.1: pre-D, the public batch's per-tensor terminal
v'_j for a weight tensor is a DIRECT weight functional in the clear (same class as
today's claim_W — no new leakage class, but "the batched a_final ... strictly less
informative" should not be read as the batch leaking less overall pre-D).

---

## 8. The BO battery — GOOD COVERAGE, ONE LOCUS ERROR + FOUR ADDITIONS

Mapping the attacks of §§1–6 above onto BO-1..13: claim omission/extra/reorder/dup →
BO-2/BO-10; comref swap → BO-3; ρ-binding violation via doctored list → BO-4; round
tamper → BO-5; vfin tamper → BO-6; padding/vars lie → BO-7; field-level byte tampers →
BO-8; cross-run replay → BO-9; substituted com file → BO-11; group omission/reorder →
BO-12; Stage-D routing → BO-13. The named-locus discipline is the right harness. Gaps:

**F8 (locus error in BO-1 — must be corrected before expectations are recorded).**
BO-1's expected locus is "G3 terminal or final IPA". Wrong for the natural cheating
strategy: a prover that emits a false eval (driver-locally compensated) and then runs
the batch prover HONESTLY over the committed tensors produces round-0 evals summing to
the TRUE total, while the verifier's cur_0 = Σ ρ_i v_i is computed from the FALSE
claims — it dies at the **G2 round-0 check `p(0)+p(1) != cur`**, deterministically.
Only a fully adaptive prover that lies coherently through all 25 rounds reaches G3/IPA.
Split: **BO-1a** honest-procedure batch prover over a false claim → locus = batch
sumcheck round 0; **BO-1b** adaptive prover (recompute every round poly from the false
running claim) → locus = G3 terminal or the group IPA. Both are cheap to implement at
toy scale and they exercise different code paths (round loop vs terminal/G5).

**F9 (add: ρ-sensitivity unit pin).** The one implementation bug that would enable the
question-1 cancellation attack — ρ squeezed before the last claim/drvstate is absorbed
(an off-by-one in the G0 loop) — is invisible to every BO test, because forged-list
tests die at claims_match first and honest runs don't probe binding. Add a unit-level
pin in vrf_toy_batchopen: two claim lists differing in the LAST claim's last byte (and
in the last drvstate) must yield different ρ. This is the transcript analog of the
INV2 startup self-check, and it is the only automated guard for the absorb-order rule
short of the human T6 walkthrough.

**F10 (add: structural-shape battery, pairs with F3).** Com file with extra trailing
rows; com file truncated; n_rows inconsistent with vars (claim forged consistently at
the structural level) → each must die at the NAMED structural check
(`opening_batch.shape` or per-driver "commitment row counts"), never reach fold_chain,
and never segfault. Include the n_claims=0 / missing-domain-group / single-row-tensor
(u_row empty) edges.

**F11 (add: single-artifact cross-run replay).** BO-9 replays the whole batch
directory; add the variant replaying ONE ipa_batch_g.bin from another honest run of the
same model (expected: that group's IPA final equation fails on transcript divergence).
Distinct code path from BO-8's byte tamper (a fully self-consistent foreign proof vs a
corrupted one).

**F12 (name the gating test).** One explicit case: all 65 driver verdicts
ACCEPT-conditional + opening_batch REJECT ⇒ orchestrator overall REJECT, and
transcript.json shows the conditional verdicts as conditional. Guards the new
orchestration logic (the conditional-verdict plumbing is new attack surface even though
it is not cryptography).

With F8 corrected and F9–F12 added, every attack constructed in this review is covered
by a named test.

---

## 9. Inherited assumptions, restated (unchanged by the rebuild — for the record)

- **Generator provenance:** pp generators are known-dlog to whoever runs setup (§10
  side finding: Commitment::random = G·r; the auditing side runs setup). Lemma 5's
  "two openings ⇒ DLOG break" therefore means: binding holds against provers who do
  not know the gen/Q mutual dlogs. The batch leans on exactly the same assumption as
  today's 1,535 inline IPAs — no weakening, no strengthening. Already documented and
  accepted; restate it in §2.3's setting paragraph so the batch's soundness statement
  is self-contained.
- **FS in the ROM:** Lemmas 3–5 state interactive-style bounds; the FS-compiled bounds
  carry the usual RO-query factor. At the 2^-240 class with SHA-256, immaterial;
  one sentence in §2.3 suffices.
- **Convention risk (bit orders):** the design's own flag is correct and the T1 toy
  pin is the right mitigation. The specific orientations T1 must nail, collected here:
  k_fr_fold binds the current MSB ⇒ G2 round t's challenge is the coordinate of
  variable m_max−1−t (the r[·] indexing in G0–G5 is by VARIABLE, not by round);
  me_weights is LSB-first (bit i ↔ u[i]); fold_chain's pair-fold consumes row bits
  LSB-first; κ_j's index set is h ≥ vars_j in variable order; û padding is in the high
  variables. The toy pin must include: multi-claim tensors, two domains, a single-row
  tensor, a max-vars tensor, and the headslice shape (two claims, same tensor,
  different points, equality checked outside the batch — §8.6's ask is right).

---

## 10. Summary of required actions

| # | severity | action | stage |
|---|---|---|---|
| F3 | REQUIRED (covert channel) | restore open_verify's row-count check in zkob_batchopen (+ Stage-C audit of per-driver size checks) | A spec, C audit |
| F5 | REQUIRED (load-bearing check) | batch verify consumes verifier-recomputed list only; claims_match inseparable from opening_batch verdict | A spec |
| F6 | REQUIRED (plumbing) | drvstate provenance verifier-internal; pin structural edges | A/B |
| F4 | recommended (hygiene) | field-level length prefixes + EvalVar tag absorb + tensor-key pin | A |
| F7 | REQUIRED before Stage D | amend §4.2 D4: committed-round-message sumcheck for the hidden initial claim | D design |
| F8 | REQUIRED (battery) | split BO-1 into BO-1a (round-0 locus) / BO-1b (G3/IPA locus) | B |
| F9–F12 | recommended (battery) | ρ-sensitivity pin; structural-shape cases; single-file replay; gating test | A–C |
| §1/§2/§9 | editorial | max-point-mass SZ statement; tighten Lemma 4; restate inherited assumptions in §2.3 | T6 |

None of these changes the protocol of §2.2. The reduction is the right construction,
the Fiat–Shamir schedule obeys the §13 rule at every squeeze, the claim transfer is
double-bound, the per-domain grouping is exactly what Pedersen linearity licenses, and
the zero-advice property survives. **Proceed to Stage A**, with F3/F5/F6 written into
the Stage-A spec and F8 into the battery expectations before any selftest expectation
lines are recorded.
