# RMSNORM_REVIEW — independent soundness audit of zkob_rmsnorm.cu

Reviewer: second engineer (independent review). Date: 2026-06-10.
Scope: `/root/zkllm/zkob_rmsnorm.cu` (918 lines) against the FINAL design in
`HANDOFF.md` §"IMMEDIATE NEXT STEP", the pinned conventions in
`PHASE0_NOTES.md` (§7–§13), and the trusted shared machinery
(`vrf_common.cuh`, `zkob_lookup.cuh`, `fs_transcript.hpp`).
Selftest independently re-run: `ZKOB-RMSNORM SELFTEST: ALL PASS` reproduced
(45 PASS, real-scale prove 9.07 s / verify 7.10 s / 678,876 bytes — matches the
report). The selftest was NOT taken on trust; every verifier check was walked
in the source.

## VERDICT: SOUND

No critical or major soundness gap found. The verifier enforces exactly the
relation the spec claims (R bound to X within ±1 integer tolerance), every
prover-supplied value the verifier consumes is anchored, and the Fiat-Shamir
schedules of prove() and verify() are absorb-for-absorb identical. Minor
notes below (selftest coverage gaps, scoped assumptions the orchestrator must
uphold, naming) — none lets a cheating prover pass this verifier.

---

## CRITICAL findings

**None.** The two automatic-CRITICAL categories are clean:

- **New G1 CUDA kernels: none.** The driver defines zero kernels
  (`grep KERNEL/__global__ zkob_rmsnorm.cu` is empty). All G1 work goes
  through the proven 1-thread helpers (`h_mul`/`h_add`/`g1_eq`,
  zkob_rmsnorm.cu:477–482) and the pinned `fold_chain`/`dev_msm` paths inside
  `open_verify`/`ipa_verify` — per the -dlto miscompilation rule.
- **Verifier accepting an unanchored prover value: none found** (full
  enumeration in "Verifier independence" below).

## MAJOR findings

**None.** Categories checked and why each is clean:

### 1. Fiat-Shamir ordering — clean

Compared prove (zkob_rmsnorm.cu:230–408) and verify (485–698) absorb/squeeze
schedules event by event. They are identical:

```
absorb B, C, C_eps_lo, C_eps_hi, com_X, com_g, com_R, com_M, com_P1, com_P2,
       com_L, com_m_L, com_W, com_W_, com_Y                       → β
absorb com_A_L                                                    → α → u_L(logDL)
[lookup: per round absorb p0..p3 → w_k] → absorb A_f, S_f, m_f
→ IPA(A_L), IPA(L), IPA(m_L) (each: per round absorb L,R → x)
→ u_b(logB) → absorb ev_M → [SS rounds] → absorb S_f2, U_f2 → IPA(X), IPA(M)
→ u_b2(logB) → absorb ev_P1, ev_P2 → [q1 rounds "q1p0..4"] → [q2 rounds]
→ absorb q1A_f,q1B_f,q1C_f,q2A_f,q2B_f,q2C_f → 6 IPAs
→ u_b3, u_c3 → absorb val_R, val_g, val_W → 3 IPAs
→ u_h(logD) → absorb claim_Y → [hadamard rounds] → absorb S_f2h, U_f2h → 3 IPAs
```

Every challenge is squeezed only after the message it binds: β after com_L
and com_m_L (logUp requirement: witness + multiplicities committed before the
denominator shift); α after com_A_L; every sumcheck round challenge after that
round's evaluations; every IPA round challenge after that round's L,R. Every
IPA statement is transcript-bound before its rounds: the commitment is in the
header absorb, the opening point is either itself transcript-derived (sumcheck
challenges) or a fresh squeeze, and the expected eval is absorbed (A_f, S_f,
m_f, ev_M, S_f2, ev_P1/P2, q*A_f/C_f, val_*, claim_Y, S_f2h, U_f2h) before the
corresponding `open_verify` runs. Labels match exactly on both sides
(including `"com_W_"` for com_Wr, and the `"q1p"+i`/`"q2p"+i` quartic tags).
No absorb present on one side and missing on the other; no order difference.

### 2. Verifier independence — every disk value anchored

Everything verify() reads, and its anchor:

| value (file) | anchor |
|---|---|
| dims.bin (B, C, C_eps) | checked == CLI args (435); CLI args absorbed into transcript |
| com_X | absorbed; opened twice (SS terminal :590, hadamard terminal :698); chained to upstream obligation by orchestrator byte-equality |
| com_g | **read from the registered path** (com_g_path, :438), not from the proof dir; absorbed; opened :670 |
| com_R | absorbed; opened at pt1 (+1 offset), pt2 (−1 offset), u_b3 |
| com_M | absorbed; opened at u_b (=ev_M), pt1 (=q1.C_f), pt2 (=q2.C_f) |
| com_P1 / com_P2 | absorbed; opened at u_b2 (=ev_P1/ev_P2); rows 0–4 / 5–9 of com_L bound homomorphically (:477–482) |
| com_L | absorbed; opened at u_ptL (=lookup S_f, :562); the SAME object feeds the affine links |
| com_m_L | absorbed (before β); opened at u_mL (=m_f) |
| com_A_L | absorbed (after β, before α); opened at u_ptL (=A_f) |
| com_W | absorbed; opened at u_pt3 (=val_W); chain anchor for zkob_rescale (com_W == rescale com_X) |
| com_Wr ("com_W_") | absorbed; opened at u_pth (=hadamard S_f2); chain anchor (== rescale com_Xr) |
| com_Y | absorbed; opened at u_h (=claim_Y); chains downstream |
| lookup.bin: ev | round-by-round p(0)+p(1)==claim chain from the RECOMPUTED anchor α+α² (:508) — never trusts a prover claim0 |
| lookup.bin: A_f, S_f, m_f | IPA openings :560–565; B_f, T_f recomputed from the public table (:524–548), not deserialized |
| hpss.bin: claim_H (ev_M) | IPA opening vs com_M at u_b (:592) |
| hpss.bin: S_f2, U_f2 | S_f2 opened vs com_X (:590); U_f2 forced == S_f2 (:585) |
| qp1/qp2.bin: claim0 (ev_P1/ev_P2) | IPA openings vs com_P1/com_P2 at u_b2 (:643–646) |
| qp*.bin: A_f, B_f, C_f | A_f anchored as R̃(pt)∓... via R openings with the ±1 offsets (:648, :653); B_f forced == A_f (:615, :633); C_f anchored as M̃(pt) (:651, :656) |
| outer.bin: val_R/val_g/val_W | three openings (:668–673) + product identity (:664) |
| hp.bin: claim_H, S_f2, U_f2 | openings vs com_Y, com_Wr, com_X (:694–698) |
| ipa_*.bin (17 files) | each consumed by exactly one `ipa_verify`, whose result is required |

Round-count guards (:466–469) pin ev array lengths to 4·logDL / 4·logD /
5·logB / 4·logD; `ipa_verify` internally pins L/R counts to log n.
Commitment row counts pinned (:449–453). No early-accept path; ACCEPT is the
last line after all 17 openings.

### 3. Openings — all 17 verified, right points, right commitments, right gens

Counted and traced (3 lookup + 2 SS + 6 quartic + 3 outer + 3 hadamard = 17;
matches the 17 ipa_* files, all of which are byte-tampered in the selftest).
All 17 calls are of the form `if (!open_verify(...)) RJ(...)` — none ignored.
Generator counts: gen_B (B_pad) for R/M/P1/P2/L/m_L/A_L row tensors, gen_C
(C_pad) for X/g/W/W_/Y — matches the commit side. `open_verify` cross-checks
`com.size == 2^(|u|−logG)` per call, so a wrong-shape commitment cannot pair
with a wrong-length point. Point reuse is correct where it occurs: A_f and S_f
both open at u_ptL (the lookup terminal point — required by the terminal
identity), X and W_ both open at u_pth (the hadamard terminal point), M opens
at three independent points (u_b, pt1, pt2) — all consistent because the
commitment binds one vector, hence one MLE. Bit-order conventions
(LSB-first column bits, `u_col = u_pt[0..logG)`) are the pinned §10
conventions and are additionally cross-checked prover-side against
`multi_dim_me` in the evil==0 sanity blocks (:277–283, :307–313, :346–354,
:399–405), which run in the honest selftest.

### 4. Bracket logic — the algebra closes; R off by ≥2 cannot pass

Walked end to end:

- **Quartic q1** (claim_q1 = 2⁶⁴C − ev_P1): by sumcheck soundness over the
  Lagrange-5 round chain, the terminal must equal
  eq̃(u_b2,pt1)·ã(pt1)·b̃(pt1)·M̃(pt1). The verifier computes eq1 =
  Π_k my_eq(u_b2[logB−1−k], w_k) itself (:613) — correct factor for every
  round, since the quartic runs over B-sized tensors and every bit is a "row"
  bit; the eq-tensor pairing (bit i ↔ u_b2[i], LSB-first in
  `build_eq_tensor`) matches the MSB-first fold order. A_f is anchored to
  com_R by the opening expecting **A_f + 1** (MLE of the all-ones vector is
  the constant 1 — valid because B == B_pad is enforced, so there are no pad
  rows); **A_f == B_f is required** (:615) — this check is load-bearing:
  without it b̃ is unconstrained and the quartic is freely forgeable; C_f is
  anchored to com_M. With Σ_s eq(u,s) ≡ 1, the chain forces
  P1(s) ≡ 2⁶⁴C − (R(s)−1)²·M(s) (mod p) for all hypercube rows s, whp over
  u_b2 (squeezed after com_P1/com_R/com_M). Symmetric for q2 with +1/−1
  swapped and claim_q2 = ev_P2 + 2⁶⁴C (:620–635).
- **2⁶⁴C constant**: `{0,0,C,0,...}` (:599) = C·2⁶⁴ in LE 32-bit limbs ✓,
  with C < 2¹⁶ enforced in verify (:427), so 2⁶⁴C < 2⁸⁰ — consistent with the
  limb capacity.
- **Limb links** (:471–483): weights {1, 2¹⁶, 2³², 2⁴⁸, 2⁶⁴} verified against
  the W16 limb constants (:70–73). Pedersen binding ⇒
  P1(s) = Σ_i 2^{16i}·L[i,s] with each L[i,s] ∈ [0,2¹⁶) by the lookup ⇒
  P1(s) ∈ [0, 2⁸⁰) **as an integer** (max Σ = 2⁸⁰−1). Same for P2 via rows
  5–9. Both links checked, both required.
- **Why R off by ≥2 fails**: for integer R' ≥ R_true+2,
  V1 = 2⁶⁴C − (R'−1)²M < 0 strictly, with |V1| < 2¹²⁴. The prover must commit
  some P1: (a) P1 = V1 mod p = p − |V1| > 2⁸⁰ — no in-range limb decomposition
  exists, the affine link fails (this is evil=1, confirmed rejected by exactly
  that check); (b) P1 = any value in [0,2⁸⁰) (e.g. the truncation) — then
  P1 ≢ 2⁶⁴C − (R'−1)²M and the q1 chain breaks (evil=2, confirmed); there is
  no third option because the affine link and the quartic anchor the SAME
  com_P1. R' ≤ R_true−2 is the mirror image through P2/q2 (code is symmetric;
  see MINOR-3 on selftest coverage). Lying about M instead is blocked by the
  SS sumcheck (evil=3). Field-element R' outside integer range: see MINOR-5.

### 5. SS sumcheck — clean

- eq factor applied **only** for rounds k < logB (:582–583), with index
  u_b[logB−1−k]: correct, because the broadcast eq tensor depends only on the
  row bits, which are the high bits of the flat index s·C_pad+j and are bound
  in rounds 0..logB−1 by the MSB-first fold; after they are bound, E is the
  constant eq̃(u_b, w_rows) and the column rounds contribute factor 1.
- S_f2 == U_f2 enforced (:585) — load-bearing (otherwise Ũ is free and any
  ev_M could be proven); terminal identity curs == eq_acc·S_f2·U_f2 (:586).
- M̃(u_b) opened against com_M (:592) with u_b a fresh post-com challenge;
  X̃ opened at the terminal point against com_X (:590). Whp this forces
  M(s) = Σ_j X(s,j)² + C_eps for every row of the committed X — a prover
  cannot use an M inconsistent with X. Column zero-padding of X_pad makes the
  C_pad-grid sum equal the C-sum exactly; row padding doesn't exist (B==B_pad).

### 6. Range lookup — clean

- com_m_L absorbed **before** β (header batch, :497 ≺ :501); com_A_L absorbed
  after β and before α (:502–503) — the logUp commit-then-challenge order is
  right.
- Anchor recomputed as α+α² (:508), never read from disk. B_f/T_f recomputed
  from the public table folded with the verifier's own phase-2 challenges
  (:524–548). Terminal identity (:549–555) matches the pinned §9 formula
  term for term (alpha_acc over ALL rounds, alphasq_acc over phase-2 rounds
  only, k ≥ n1, :521).
- The affine link and the lookup constrain the **same** com_L: the lookup's
  folded witness S_f is opened against com_L itself (:562), so the limb matrix
  whose rows build com_P1/com_P2 is exactly the tensor proven 16-bit.
- Table = tLookupRange(0, 65536): verified upstream
  `tlookuprange_init_kernel` fills table[t] = t for t < 65536 (len is an
  exact power of two, no pad duplicates) — 0 is in the table, so the
  all-zero pad rows of L pass the lookup, and `inv_ratio`, n1 = 0 at both
  selftest scales (D_L = N = 65536).

### 7. Hadamard / outer / internal rescale — clean

- Outer product: val_W == val_R·val_g checked (:664) at a fresh point
  (u_b3, u_c3 squeezed after all commitments), with all three values anchored:
  val_R vs com_R, val_g vs the **registered** com_g (the verifier loads com_g
  from the CLI registry path, :438 — also discharging the norm-weight
  commitment_opening id), val_W vs com_W. u_pt3 = u_c3‖u_b3 with column bits
  low — matches the flat layout. Single-point MLE factorization identity is
  sound by Schwartz-Zippel. B == B_pad enforced in **verify** (:427), not
  just prove, so the W_pad = R⊗g_pad grid identity has no pad-row hole.
- Internal rescale: W_ is indeed only computed prover-side — **by design**;
  the binding obligation is the separate zkob_rescale run. What this driver
  must supply for that, it does: com_W and com_W_ are both committed, both
  saved (com_W.bin / com_Wr.bin), both absorbed (:243–244 / :498–499), and
  the unpadded W.i64 chain file is written from the exact host vector
  (:165–168) for the rescale prover. The orchestrator's byte-equality check
  (com_W == rescale com_X, com_W_ == rescale com_Xr) then closes the loop.
- Hadamard: claim_Y = Ỹ(u_h) at a fresh u_h, anchored vs com_Y; terminals
  anchored vs com_Wr and com_X at the terminal point; eq factor over ALL logD
  rounds (:688) — correct, the eq tensor here depends on every bit. No
  S_f2==U_f2 here and none needed (different tensors, each separately opened).

### 8. Padding — clean

B == B_pad required in both prove (throw, :99) and verify (RJ, :427).
g/X/W/W_/Y column padding is zero-extension via `FrTensor::pad`, and every
identity that sums over the grid (SS, hadamard, outer) is exact under zero
columns. Limb pad rows are zero and 0 is in the table. The Y chain file
strips column padding (:186–196); W chain file is built unpadded from host
ints. dims.bin is redundant with the CLI args but checked, harmless.

### 10. Numeric / representation — clean

- All host bracket math in __int128 with the bounds actually enforced:
  M < 2⁶² (throw, :121; the u128 accumulation before the check peaks at
  ~C·2⁶² + C_eps < 2⁷⁸, no wrap), so (R±1)²·M < 2⁶²·2⁶² = 2¹²⁴ < 2¹²⁷ —
  no i128 overflow (R is int32, (R+1)² ≤ 2⁶²).
- `fr_from_u128` only ever receives values < 2⁸⁰ (M, t1, t2, V1≥0 case) ≪ p,
  so the raw-limb plain form is correct; the `FrTensor(uint, const Fr_t*)`
  ctor is a raw memcpy (verified in fr-tensor.cu:163) — tensors hold PLAIN
  values like all drivers.
- Negative V1 (evil=1 only) is negated in the field via h_scalar sub (:141) —
  correct mod-p representation, no signed wraparound.
- C_eps (u64) split into two u32 absorbs and reassembled identically on both
  sides; C_eps_fr = {lo, hi, 0...} ✓.
- Montgomery: the only degree-4 kernel (`k_hp4_step`, header) follows the
  "mont-ify all factors except one" rule — checked by hand:
  mul(mont e, mont a)=eaR; mul(mont b, c)=bc; mul(eaR, bc)=eabc plain ✓
  (and the HANDOFF notes its Montgomery bug was already found and fixed, with
  glu/rescale selftests re-run). `my_eq` matches upstream
  `Polynomial::eq`/eqEvalKernel.

### 11. New CUDA kernels — none in the driver

The driver defines no kernels. It uses (all pre-existing, all Fr-only or
upstream-validated): `rescaling_kernel`, `tlookup_inv_kernel`,
`tLookup_phase1/2_*` (upstream), and the header's `k_bcast_rows`, `k_bump`,
`k_eq_expand`, `k_fr_fold`, `k_hp3_step`, `k_hp4_step` (Fr-only — outside the
-dlto G1 miscompile family, and the p(0)+p(1) round chains are runtime probes
of them). The affine link deliberately uses the 1-thread G1 helpers, citing
the documented batched-kernel miscompile. Rule followed.

---

## MINOR findings / notes

**MINOR-1. Selftest never semantically exercises the P2-side bracket
(R bumped DOWN), q2, or the affine P2 link.**
zkob_rmsnorm.cu:785–791 — evil=1/2 both use R+2 (V1 < 0 path), so only the
P1 affine link and the q1 round chain are forgery-tested; the P2 link
(:481–482) and q2 chain (:619–635) are exercised only by honest verification
and generic byte tampers. Failure it could hide: an asymmetric typo in the P2
path (wrong row offset, wrong sign) that still accepts honest proofs. I
checked the code by eye — it is exactly symmetric (rows 5+i, A_f−1, ev_P2 +
2⁶⁴C) — so this is a coverage note, not a bug.
Fix: add evil=6: R[idx] −= 2, P2 stored mod p → expect "affine link P2";
and evil=7: R[idx] −= 2, P2 = limb reconstruction → expect "q2 round 0".

**MINOR-2. No semantic out-of-range-limb forgery in THIS driver's selftest.**
The limb lookup is the only thing forcing L ∈ [0,2¹⁶), and no evil mode
plants e.g. L = 65536 with a compensating P1 (which keeps the affine link and
quartics valid and must be caught by the lookup at round 0). The identical
logUp machinery is semantically forgery-tested in zkob_rescale's selftest
(the rem covert-channel forgery, PHASE0_NOTES §11), and it is shared header
code, so the risk is low. Fix: add an evil mode L[0,s] += 2¹⁶,
P1 recomputed from limbs → expect a lookup rejection.

**MINOR-3. The A_f==B_f and S_f2==U_f2 checks are never forgery-tested.**
These two one-line checks (:615/:633 and :585) are load-bearing (each one's
absence makes its sumcheck freely forgeable, since B / U are otherwise
unopened), and no evil mode or byte tamper specifically targets them (a
tampered B_f in qp1.bin is caught, but by the terminal identity / transcript
divergence, not necessarily by the equality check). They ARE present and
correct. Fix (cheap): a selftest variant that writes qp1.B_f = A_f + 1 and
re-signs nothing else would currently be rejected by transcript divergence at
the absorb — so a meaningful test needs a prover-side evil that runs the q1
sumcheck with B = A + δ tensor; worth adding when the file is next touched.

**MINOR-4. Phase-1 lookup rounds (n1 > 0) are dead at both selftest scales.**
B_pad = 8 and 1024 both give D_L = N = 65536 → n1 = 0, pure phase 2; the
phase-1 path (and the `u_mL` truncation, alphasq_acc gating at :521) only
activates for B_pad > 4096, which never occurs for llama-68m (seq 1024). The
phase-1 code is shared header code validated with n1 ≥ 1 in zkob_rescale's
selftest. No action needed for this model; revisit if B_pad ever exceeds 4096.

**MINOR-5. Scoped assumption: the ±1 bound on R needs M to be an honest
bounded integer, which this driver inherits from the chain, not enforces.**
The verifier proves M(s) ≡ Σ_j X(s,j)² + C_eps (mod p) — it cannot and does
not enforce the prover-side M < 2⁶² bound. With X honest (chained), M ~ 2⁴²
and the windows argument leaves only the designed R ∈ {R−1, R, R+1}
(heuristic spurious-solution count ≈ 2⁻⁹¹ for field-element R'). But if this
driver were ever run with an UNCHAINED com_X (prover-chosen X), the prover
could craft M tiny (e.g. M = 1 mod p), and the two 2⁸⁰-wide windows then
admit ~2³⁹ integer values of R — a wide-open covert channel. This is the
design (HANDOFF binds R to X; X's integrity is the orchestrator's
chain/byte-equality job), and the same caveat already applies to every driver,
but it is worth stating: **the orchestrator MUST chain com_X here to the
upstream range-bound activation commitment; standalone ACCEPT of this
obligation proves much less.** Suggested action: one sentence in the
orchestrator notes; no driver change.

**MINOR-6. Limb-matrix pad rows (indices ≥ 10) are range-bound but not
proven zero.** (Author self-reported.) Confirmed harmless: they enter only
the lookup (any 16-bit value passes) and the S_f/A_f openings; they feed no
affine link, no quartic, no chain output. No action.

**MINOR-7. File naming: the spec's `com_W_` is saved as `com_Wr.bin`.**
zkob_rmsnorm.cu:224. The transcript label is `"com_W_"` (matching the spec)
but the orchestrator's byte-equality check against zkob_rescale must look for
`com_Wr.bin`. Documented in RMSNORM_REPORT.md; just don't trip on it when
wiring the manifest. No code change needed.

**MINOR-8. Byte-tamper coverage is complete.** Cross-checked PROOF_FILES
(:741–748) against every `fopen`/ctor in verify(): dims.bin, 6 proof files,
12 commitment files (including com_g via the selftest's registry path =
obdir/com_g.bin), 17 IPA files — 36/36; nothing the verifier reads escapes
tampering. The generators and Q are CLI/registry inputs, correctly out of
scope. Also verified the evil-mode plumbing is honest: the prover's own
sanity checks are precisely disabled for the mode under test
(strict=false only for the targeted sumcheck, bracket check skipped only for
evil 1/2, val_W check skipped only for evil 5, the evil==0-only convention
asserts), and the selftest requires the rejection REASON string to match the
named check — so a prover-side throw or a wrong-check rejection would FAIL
the selftest, not mask it.

**MINOR-9. exact_R's double-precision sqrt seed** (zkob_rmsnorm.cu:721) is
followed by exact __int128 fix-up loops in both directions, so the selftest's
R is exact regardless of FP rounding; selftest-only code anyway. No action.

---

## What I would do before trusting it in the gate (summary)

Accept the file. Optionally (cheap, next touch): add the three missing
semantic evil modes (R−2 affine-P2, R−2 q2, out-of-range limb) per MINOR-1/2,
and the orchestrator-notes sentence per MINOR-5/7.
