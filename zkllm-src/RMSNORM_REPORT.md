# RMSNORM_REPORT — zkob_rmsnorm.cu

Status: **ALL PASS** (full selftest, 45/45 checks, exit code 0). Date: 2026-06-10.

## What was implemented

`/root/zkllm/zkob_rmsnorm.cu` — the rmsnorm obligation driver from the FINAL design in
`HANDOFF.md` ("IMMEDIATE NEXT STEP: write zkob_rmsnorm.cu"), implemented exactly as
specified. It binds the inverse-RMS advice R to X within ±1 integer tolerance, closing
the unbound `rms_inv_temp` covert channel (≤ log2(3) ≈ 1.6 bits/row residual freedom,
the documented floor). One Fiat-Shamir transcript, 17 IPA openings, seven obligations:

1. **Limb range lookup** — P1 = 2⁶⁴C − (R−1)²M and P2 = (R+1)²M − 2⁶⁴C decomposed into
   5×16-bit limbs each, packed into the limb matrix L (max(16, 65536/B_pad) rows ×
   B_pad cols, committed with gen_B), logUp vs tLookupRange(0, 65536) via the header's
   `fs_phase1/fs_phase2`. B_f/T_f recomputed by the verifier from the public table.
2. **Homomorphic affine limb links** — com_P1 == Σ_{i<5} 2^{16i}·com_L_row[i] (and P2 vs
   rows 5–9), checked with the proven 1-thread host helpers h_mul/h_add/g1_eq. No new
   G1 kernels anywhere in the driver.
3. **SS sumcheck** binding M to X: claim = M̃(u_b) − C_eps = Σ eq·X·X over logD vars,
   `fs_hadamard` with E = k_bcast_rows(build_eq_tensor(u_b)), S = U = X_pad. Verifier
   applies the eq factor only for rounds k < logB and requires S_f2 == U_f2. Openings:
   X at reverse(ws), M̃(u_b) vs com_M.
4. **Bracket quartics** (`fs_quartic`, tags "q1"/"q2", Lagrange-5): claim_q1 =
   2⁶⁴C − P̃1(u_b2) over (E2, T1, T1, M) with T1 = R−1; claim_q2 = P̃2(u_b2) + 2⁶⁴C over
   (E2, T2, T2, M). ev_P1/ev_P2 ride in the QuarticProof.claim0 slots and are bound by
   IPA openings of P1/P2 at u_b2. No commitment for T1/T2: the verifier opens R at pt1
   expecting q1.A_f + 1 and at pt2 expecting q2.A_f − 1, and requires A_f == B_f.
5. **Outer product** W = R×g via MLE factorization, no sumcheck: absorb val_R, val_g,
   val_W; check val_W == val_R·val_g; three openings — g opens against the **registered**
   com_g path (discharging the norm-weight commitment_opening id). u_pt3 = u_c3 ‖ u_b3.
   B == B_pad is REQUIRED (throw otherwise) so W_pad = R⊗g_pad exactly.
6. **Internal rescale** W_ = rescale(W, 2¹⁶) computed with `rescaling_kernel` on the
   UNPADDED B·C tensor then padded; driver commits com_W_ (proof of the rescale itself
   is the separate zkob_rescale run, chained by commitment-file byte-equality).
7. **Hadamard** Y = W_ ⊙ X (glu part 2). Openings Y@u_h, W_@pt, X@pt.

FS schedule exactly as pinned in HANDOFF: absorb B, C, C_eps(lo,hi), com_X, com_g,
com_R, com_M, com_P1, com_P2, com_L, com_m_L, com_W, com_W_, com_Y → β → com_A_L → α →
u_L → lookup rounds + terminals + 3 openings → u_b → ev_M → SS rounds + terminals +
2 openings → u_b2 → ev_P1, ev_P2 → q1 rounds, q2 rounds, terminals + 6 openings →
u_b3, u_c3 → val_R/val_g/val_W + 3 openings → u_h → claim_Y → hadamard rounds +
terminals + 3 openings. Every challenge is derived only after absorbing the message it
binds (per-round inside all sumchecks and IPAs).

Host bracket math in __int128 with hard requirements enforced (throw): M < 2⁶², C < 2¹⁶,
B == B_pad, bracket residuals in [0, 2⁸⁰), and the prover sanity-checks the ±1 bracket
on the input R ("advice R out of tolerance"). P1/P2/M FrTensors are built from host
Fr_t limb arrays (`fr_from_u128`, raw-memcpy ctor) in PLAIN form like all drivers.
File I/O uses the header's loaders / a driver-local raw int32 reader (upstream
`from_long_bin` avoided). Chain outputs: W.i64 (host vector, unpadded) and Y.i64
(save_long + column-pad strip), both UNPADDED B×C int64.

**No existing file was modified.** New files: `zkob_rmsnorm.cu`, this report.

Build (clean, no warnings):
```
nvcc -arch=sm_89 -std=c++17 -I/usr/local/cuda/include -dc -dlto zkob_rmsnorm.cu -o zkob_rmsnorm.o
nvcc -arch=sm_89 -std=c++17 -dlto zkob_rmsnorm.o bls12-381.o ioutils.o commitment.o fr-tensor.o g1-tensor.o proof.o zkrelu.o zkfc.o tlookup.o polynomial.o zksoftmax.o rescaling.o timer.o -o zkob_rmsnorm
```

## Selftest results (`./zkob_rmsnorm selftest`, exit 0)

Small case (B=8, C=5, C_eps=21475, exact R via __int128 bracket search):

- honest: **ACCEPT** — PASS
- All 5 semantic evil modes rejected by **exactly** the check the spec names:
  - evil=1 (R+2, P1 stored mod p, limbs = low 80 bits) → `affine link P1` — PASS
  - evil=2 (R+2, limbs honest-truncated, P1 = limb reconstruction) → `q1 round 0` — PASS
  - evil=3 (M[i]+1, brackets recomputed from new M) → `SS round 0` — PASS
  - evil=4 (Y[i]+1) → `hadamard round 0` — PASS
  - evil=5 (W[i]+1) → `outer product check val_W != val_R*val_g` — PASS
- Byte tampers on **every** proof/commitment/ipa file (36 files: dims.bin, 6 proof
  files, 12 commitment files, 17 IPA files) — all rejected, then restored verify
  ACCEPT — all PASS.

Real-scale case (B=1024, C=768, gens /tmp/gen1024.bin via ppgen, C_eps=3298535):

- honest **ACCEPT**: **prove 9.43 s, verify 7.10 s, proof+commitments 678,876 bytes
  (~663 KB)**
- byte tamper at scale rejected — PASS

Final line: `ZKOB-RMSNORM SELFTEST: ALL PASS` (45 PASS lines, 0 FAIL).
Full log: /tmp/rmsnorm_selftest.log.

## Deviations from the spec

**None in the protocol.** One parameter note: the task text suggested C_eps for
eps=1e-5, but HANDOFF explicitly says to CHECK the eps actually used in
m68-pipeline.py before wiring. The pipeline reads `variance_epsilon` from the model
config, and JackFram/llama-68m's `config.json` has `rms_norm_eps: 1e-06`. The
real-scale selftest therefore uses **C_eps = round(1e-6 · 768 · 2³²) = 3298535**.
C_eps is a u64 CLI argument either way, so the driver is agnostic; the orchestrator
must pass the value matching the pipeline (1e-6 for llama-68m).

Minor free choices within the spec (documented for the orchestrator):
- File names in <obdir>: dims.bin; lookup.bin, hpss.bin, qp1.bin, qp2.bin, outer.bin,
  hp.bin; com_X/com_g/com_R/com_M/com_P1/com_P2/com_L/com_m_L/com_A_L/com_W/com_Wr
  (= com_W_)/com_Y .bin; ipa_AL/ipa_L/ipa_mL/ipa_X_ss/ipa_M/ipa_P1/ipa_P2/ipa_R_q1/
  ipa_M_q1/ipa_R_q2/ipa_M_q2/ipa_R_o/ipa_g/ipa_W/ipa_Y/ipa_Wr/ipa_X_h .bin.
- ev_P1/ev_P2 are serialized in the claim0 field of qp1.bin/qp2.bin (verifier
  recomputes the actual sumcheck claims 2⁶⁴C∓ev from them; both are IPA-bound).

## Concerns / notes (honest)

- L's zero-pad rows (indices ≥ 10) are only range-bound by the lookup, not forced to
  zero — they do not enter the affine links or any other identity, so this is harmless
  by construction, but they are not "proven zero".
- com_W_ is bound to com_W only through the separate zkob_rescale obligation
  (orchestrator must check com_W here == com_X there and com_W_ == com_Xr,
  byte-identical) — exactly as the design states; this driver alone does not prove the
  rescale.
- The residual covert channel is the designed ±1 tolerance on R: ≤ log2(3) bits/row,
  ~6.5 Kbit/forward over 4 norms — the measured floor, per HANDOFF.
- Per HANDOFF convention, drivers do NOT mkdir the obdir (selftest creates its own
  /tmp dirs); the orchestrator must create obligation directories.
- The shared header `zkob_lookup.cuh` was NOT modified, so the other drivers'
  selftests are unaffected (no rebuild needed).

## Hardening round (2026-06-10, post-audit)

The independent audit (RMSNORM_REVIEW.md, VERDICT: SOUND) asked for three additional
semantic evil modes (MINOR-1, MINOR-2). Added to the selftest — verify() and the honest
prove() path are byte-for-byte unchanged; only the prover-evil plumbing and the selftest
table were touched:

- evil=6 (R[idx]−2, P2 recomputed mod p — negative wraps, limbs = low 80 bits) →
  rejected by exactly `affine link P2 != sum 2^16i * com_L_row[5+i]` — PASS.
  Mirror of evil=1; the P2-side affine link is now semantically forgery-tested.
- evil=7 (R[idx]−2, limbs honest-truncated, P2 = limb reconstruction; strict=false
  only for the q2 sumcheck) → rejected by exactly `q2 round 0 p(0)+p(1) != claim` —
  PASS. Mirror of evil=2; the q2 round chain is now semantically forgery-tested.
- evil=8 (L[0,idx] += 2¹⁶ with a compensating borrow L[1,idx] −= 1, so P1's value —
  and hence the affine links, both quartics, and every commitment except com_L — is
  unchanged; m_L committed from the honest limbs, which is also the forging prover's
  best move and avoids prep()'s unchecked atomicAdd on the out-of-range index;
  strict=false only for fs_phase1) → rejected by exactly
  `limb lookup round 0 p(0)+p(1) != claim` — PASS. The range lookup is the ONLY check
  standing between this forgery and ACCEPT, and it fires, as the audit predicted
  (MINOR-2: "must be caught by the lookup at round 0").

All three were rejected by the exact named check on the first run — no expectation was
loosened. Selftest totals after rebuild (same pinned nvcc commands): **48 PASS, 0 FAIL,
exit 0** (was 45) — honest small ACCEPT, 8 semantic evil modes, 36 byte tampers +
restored ACCEPT, real-scale honest ACCEPT (prove 9.14 s, verify 7.03 s, 678,876 bytes —
matches the pre-hardening numbers) + real-scale tamper.
Full log: /tmp/rmsnorm_selftest_hardened.log.
