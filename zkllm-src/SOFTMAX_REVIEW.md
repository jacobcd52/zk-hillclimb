# SOFTMAX_REVIEW — independent soundness audit of zkob_softmax.cu

Reviewer: second engineer (independent review). Date: 2026-06-10.
Scope: `/root/zkllm/zkob_softmax.cu` (1006 lines) against the normative
`SOFTMAX_DESIGN.md` (DESIGN FINAL), the pinned conventions in
`PHASE0_NOTES.md` (§7–§14), the trusted shared machinery
(`vrf_common.cuh`, `zkob_lookup.cuh`, `tlookup.cu`), and the bar set by
`RMSNORM_REVIEW.md`.
Selftest independently re-run: `ZKOB-SOFTMAX SELFTEST: ALL PASS`, exit 0,
0 FAIL (real-scale prove 10.29 s / verify 11.81 s / 2,124,988 bytes — matches
the report). **The selftest was NOT taken on trust**: every `verify()` check
and every FS absorb was walked in source, and the relevant upstream table
construction (`tLookupRange`/`tLookupRangeMapping`) was read to confirm the
combined-table algebra.

## VERDICT: SOUND

No critical or major soundness gap. The verifier enforces exactly the relation
the spec claims — `(R1)` exp mapping + domain, `(R2)` causal row-sum,
`(R3)` `P = round_half_up(2^16·MK·E/S)` exactly with `P[masked]=0` — every
prover-supplied value `verify()` consumes is anchored to a commitment / public
recomputation / the recomputed logUp anchor, and the Fiat–Shamir schedules of
`prove()` and `verify()` are absorb-for-absorb identical. The bracket closes on
**both** sides (r1 ≥ 0 and r1 < 2S are each independently forced and each
independently forgery-tested), which is stronger coverage than rmsnorm had.
Minor notes below are coverage/robustness observations; none lets a cheating
prover pass this verifier.

---

## CRITICAL findings

**None.** Both automatic-CRITICAL categories are clean:

- **New G1 CUDA kernels: none. New Fr kernels: none.** `grep -E 'KERNEL|__global__'
  zkob_softmax.cu` is empty. All G1 work goes through the proven 1-thread
  helpers (`h_mul`/`h_add`/`g1_eq`) and the pinned `fold_chain`/`dev_msm`/IPA
  paths; the homomorphic `com_comb = com_z + r·com_E` is built row-by-row with
  `h_add`/`h_mul` (lines 539–546), exactly per the −dlto rule. Every recursion
  uses header kernels already validated (`k_fr_fold`, `k_fr_emul`,
  `k_bcast_rows`, `k_eq_expand`, `k_hp3_step`, `tlookup_inv_kernel`,
  `tLookup_phase{1,2}_*`). No Montgomery-convention question arises.
- **Verifier accepting an unanchored prover value: none found** (full
  enumeration below).

## MAJOR findings

**None.** Each checklist category was walked; why each is clean follows.

### 1. Fiat–Shamir ordering — clean (absorb-for-absorb identical)

Compared `prove()` (94–432) and `verify()` (458–774) event by event. They are
identical, and match §5 label-for-label:

```
absorb B,NCOL,LOW_E,LEN_E,LEN_R,LOG_OUT
absorb com_z,com_E,com_P,com_S,com_L,com_m_E,com_m_L          → r → beta_E
absorb com_A_E                                → alpha_E → u_E(logD)
[exp lookup logD rounds: absorb p0..p3 → w] → absorb A_f,S_f,m_f
   → IPA(A_E)@u_ptE, IPA(comb)@u_ptE, IPA(m_E)@u_mE          → beta_L
absorb com_A_L                               → alpha_L → u_L(logDL)
[limb lookup logDL rounds] → absorb A_f,S_f,m_f
   → IPA(A_L)@u_ptL, IPA(L)@u_ptL, IPA(m_L)@u_mL
→ u_b(logB) → absorb ev_S → [rs logD rounds] → absorb S_f2,U_f2
   → IPA(E)@pt_rs, IPA(S)@u_b
→ u_r(logD) → absorb c1 → [V1 logD rounds] → absorb S_f2,U_f2 → IPA(E)@pt1
→ absorb c2 → [V2 logD rounds] → absorb S_f2,U_f2 → IPA(P)@pt2, IPA(S)@pt2_rows
→ absorb v00,v10,v01,v11 → IPA(L00,L10,L01,L11) → absorb S_id → IPA(S)@ur_row
[verifier-only I1,I2 — no absorbs]
```

Every challenge is squeezed only after the message it binds: `r`/`beta_E` after
all seven base commitments; `alpha_E` after `com_A_E`; each sumcheck round
challenge after that round's evals; `u_r` (the load-bearing bracket point) after
the row-sum is fully closed; each IPA round challenge after its own L,R. Each
opened value is absorbed before its `open_verify` runs. Labels match exactly on
both sides (including the reused `"hp0".."hp3"` across the three hadamards and
`"A_f"/"S_f"/"m_f"` across the two lookups — positionally disambiguated by the
interleaved `ev_S`/`c1`/`c2`/`v*` absorbs, exactly as §5 note 1 anticipates).
The 16-opening order is identical in `prove` and `verify`, so each IPA consumes
the same `logC` transcript challenges on both sides. No absorb present on one
side and missing on the other; no order difference.

### 2. Verifier independence — every disk value anchored

Everything `verify()` reads, and its anchor (RMSNORM table format):

| value (file) | anchor |
|---|---|
| dims.bin (B,NCOL,LOW_E,LEN_E,LEN_R,LOG_OUT) | cross-checked == CLI/const args (479–481); the same scalars absorbed into the transcript |
| com_z | absorbed (524); used only homomorphically to form `com_comb`; pinned into the chain by the orchestrator's `com_z == rescale10 com_Xr` byte-equality (§4.7, external) |
| com_E | absorbed; opened at pt_rs (row-sum 675), pt1 (V1 707); participates in `com_comb` |
| com_P | absorbed; opened at pt2 (V2 730) |
| com_S | absorbed; opened at u_b (ev_S 677), pt2_rows (V2 U_f2 734), ur_row (S_id 753) |
| com_L | absorbed; opened at u_ptL (limb S_f 641) **and** the 4 plane points (748) — same object feeds the lookup and the I1/I2 reconstruction |
| com_m_E / com_m_L | absorbed before β; opened at u_mE / u_mL (m_f 595/643) |
| com_A_E / com_A_L | absorbed after β before α; opened at u_ptE / u_ptL (A_f 591/639) |
| lookup_E/L: ev[] | round-by-round p(0)+p(1)==claim from the **recomputed** anchor α+α² (551/603) — prover claim0 never trusted |
| lookup_E/L: A_f,S_f,m_f | IPA openings; B_f,T_f recomputed from the public table folded with the verifier's own phase-2 challenges (567–579 / 618–627) |
| hp_rs: claim_H(ev_S) | IPA vs com_S @u_b (677); start of the round chain whose terminal is anchored |
| hp_rs/hp_v1: S_f2,U_f2 | S_f2 opened vs com_E; **U_f2 forced ==1** (660/695) — load-bearing |
| hp_v1: claim_H(c1) | bound by the V1 sumcheck to W̃1(pt1)·Ẽ(pt1)·1 with Ẽ anchored to com_E and W1 recomputed (697–703) |
| hp_v2: claim_H(c2) | bound by the V2 sumcheck to eq̃(u_r,pt2)·P̃(pt2)·S̃b(pt2); P̃ vs com_P, S̃b vs com_S (725/730/734) |
| hp_v2: S_f2,U_f2 | S_f2 opened vs com_P @pt2; U_f2 opened vs com_S @pt2_rows |
| lvals.bin: v00,v10,v01,v11,S_id | each an IPA opening vs com_L (planes) / com_S (S_id) (748/753) |
| ipa_*.bin (16) | each consumed by exactly one `open_verify`, result required (`if(!…) RJ`) |

Round-count guards pin ev lengths to `4·logD` / `4·logDL` / `4·logD`×3
(507–511); commitment row counts pinned (492–496); layout params re-checked
(468–472). No early-accept path; `ACCEPT` is the last line after all 16
openings and both identities.

### 3. The 16 IPA openings — all present, required, right points/commits/gens

Traced (3 exp + 3 limb + 2 row-sum + 1 V1 + 2 V2 + 4 L-plane + 1 S_id = 16,
matching the 16 `ipa_*` files, all byte-tampered). Every call is
`if(!open_verify(...)) RJ(...)` — none ignored. All use `gen` (size NCOL); the
single-row `com_S` openings (u_b, pt2_rows, ur_row) carry `|u_pt| == logB ==
logC`, so `open_verify` requires `com.size == 2^0 == 1` ✓. The constant-0/1
plane bits enter `u_pt` as ordinary field elements consumed by `fold_chain`
(orientation 0); for boolean values `k_com_me` selects the correct
commitment-row half, so the L-plane MLE is evaluated at `(u_r ‖ planebit0 ‖
planebit1)` with the plane occupying flat-index bits 20,21 — consistent with
`flat = plane·D + i·NCOL + j` (prove 408–424 / verify 738–751, plane `p` →
bits `(p&1, p>>1)` ✓). `com_comb` is the homomorphic `com_z + r·com_E`, formed
on host from the two on-disk commitments; `S_f` (the `comb` fold) opens against
it, jointly binding `z_` and `E` through the chained `com_z`/`com_E` (no
separate z_ link sumcheck, as designed).

### 4. The rounding bracket (R3) — enforced exactly, both sides forced

- **I1** (765): `2^17·c1 + S_id − 2·c2 == v00 + LEN_R·v10`. The constants are
  right: `F_2P17 = 1<<(LOG_OUT+1) = 2^17` (758), radix `F_LENR = LEN_R = 2^20`
  at real scale (757). `c1 = MLE(MK⊙E)(u_r)`, `c2 = MLE(P⊙S_bcast)(u_r)`,
  `S_id = S̃(u_r rows) = MLE(S_bcast)(u_r)` — so the LHS is the MLE at `u_r` of
  `r1 = 2^17·MK·E + S_bcast − 2·P·S_bcast`. The RHS is the MLE of the
  limb-reconstructed `r1` (`v00 = L̃` plane 0 = lo, `v10` = plane 1 = hi). By
  Schwartz–Zippel at the post-commitment random `u_r`, this is the tensor
  identity `r1 = 2^17·MK·E + S − 2PS`. ✓
- **I2** (770): `r̃1 + r̃2 + 1 == 2·S_id`, i.e. `r1 + r2 + 1 = 2S` as tensors.
- **Limb lookup** binds every plane of the **same** `com_L` to `[0,LEN_R)`
  (S-slot of the lookup is `L_t`, opened vs `com_L` @u_ptL — same object the
  plane openings use), so `r1,r2 ∈ [0,LEN_R²)` **as integers**. With
  `2·LEN_R² = 2^41 < p` and `2S < 2^40`, the I2 congruence forces integer
  equality ⇒ `r1 = 2S−1−r2 ≤ 2S−1` and `r1 ≥ 0`. Hence `r1 ∈ [0,2S)`, which is
  the unique-integer bracket ⇒ `P = round_half_up(2^16·MK·E/S)`. A committed `P`
  off the honest integer (or any non-integer field element) forces `r1` outside
  `[0,2S)`, hence its 2-limb reconstruction outside `[0,LEN_R²)` — impossible.
- **Masked ⇒ P=0** falls out of the same bracket: at `MK=0`, `r1 = S(1−2P) ∈
  [0,2S)` ⟺ `P=0`. No separate mask-on-P constraint is needed, and the mask is
  recomputed by the verifier from `B` (514–517), so it cannot be smuggled.
- The `v00 + 2^20·v10` reconstruction is consistent with the committed L layout
  (plane 0/1 = r1 lo/hi, 2/3 = r2 lo/hi; prove 162–165), and the limb lookup
  constrains that exact `com_L`.

Both directions are semantically forgery-tested: evil=2 (P+1 at a masked entry,
true r1 = −S, mod-truncated limbs) → I1; evil=3 (P−1 at a diagonal,
`r1'=r1+2S` honest limbs, r2 mod-truncated) → I2. I independently confirmed the
check ordering makes the named identity fire first (I1 at 765 precedes I2 at
770; both precede nothing else after the openings).

### 5. Row-sum (R2) — clean

`ev_S = S̃(u_b)` opened vs `com_S` @u_b (677) **and** bound by the sumcheck to
`Σ_b W_rs(b)·E(b)·1 = Σ_i eq(u_b,i)·Σ_{j≤i}E[i,j] = S̃_true(u_b)`. The verifier
rebuilds `W_rs = bcast_rows(eq(u_b)) ⊙ MK` itself (663–668, same construction as
prove) and folds it with `fold_public` over the identical round challenges, so
`W_f = W̃_rs(pt_rs)` is its own computation; `S_f2` (E) is anchored to com_E;
**`U_f2 == 1` is required** (660) and is load-bearing (without it the terminal
`cur == W_f·S_f2·U_f2` is satisfiable with a free U). Therefore `com_S` is
forced to be the causal row-sums of the committed `E`. Because the bracket's
`S_id` and the V2 `U_f2` both open the **same** `com_S`, a prover cannot use an
S in the bracket inconsistent with the row-sum-bound S. evil=4 (S+1, P/limbs
recomputed) → row-sum round 0, confirming the row-sum is the sole tie between S
and E and that nothing else fires.

### 6. V2 broadcast binding — clean (non-broadcast U cannot be smuggled)

In V2, `U = Sb` is materialised by the prover and **never committed**. The
sumcheck terminal `U_f2 = S̃b(pt2)` is opened against `com_S` at `pt2_rows =
pt2[logC..]` (734). Since the layout is LSB-first columns (flat = i·NCOL+j, j
low), the row bits of `pt2` are exactly `pt2[logC..]`, and a column-broadcast
tensor's MLE depends only on the row bits: `MLE(S_bcast)(pt2) = S̃(pt2_rows)`.
The opening forces `S̃b(pt2) = S̃(pt2_rows)`. As `pt2` is squeezed only **after**
the V2 round polys (which already fix `Sb`), any `Δ = Sb − S_bcast ≠ 0` gives
`MLE(Δ)(pt2) = 0` with probability `≤ logD/|F|`; so whp `Sb = S_bcast`, hence
`c2 = MLE(P⊙S_bcast)(u_r)` genuinely. The prover-side evil==0 block (397–399)
cross-checks `U_f2 == Sb.multi_dim_me == S̃(pt2_rows)`, pinning the convention,
and evil=5 (Sb[idx]+1, all commitments honest) → "IPA opening of V2 U_f2 vs
com_S", reached before I1. ✓

### 7. Exp lookup (R1) — clean

`com_comb = com_z + r·com_E` is homomorphic (host `h_mul`/`h_add`); `S_f`
(comb) opens against it. `B_f`/`T_f` are recomputed from the public combined
table `T_comb = table + r·mapped` (I confirmed upstream `tlLookupRange` fills
`table[t]=low+t` and `prep` maps `v→v−low`, so the lookup binds `z_ ∈
[LOW_E,LOW_E+LEN_E)` and `E = X_E[z_−LOW_E]`, masked positions included). The
table is loaded from the CLI `expmap` path; negative `z_` are carried as field
elements identically in the witness (`z_t`), the table (`low+k`), and the
combiner (`+r·mapped`), so the combine is consistent. The anchor `α+α²` is
recomputed (551), never read. glu's `mapped(0)==0` check is correctly **absent**
(no padding ⇒ no fabricated `(0,mapped(0))` row; load_expmap only checks
LEN_E pow2, 56–64). evil=1 (E+1 unmasked) → exp lookup round 0. ✓

### 8. Layout / padding — clean

`B == NCOL`, both pow2, and `gen.size == NCOL` are enforced in **both** prove
(`layout_guards`, 66–77) and verify (468–472). The two lookup layout
constraint sets (`NCOL ≤ N ≤ D`, `N | D`; n1_E, n1_L) are checked on both
sides; at real scale n1_E=0 (pure phase2), n1_L=2, matching §6. No padding path
exists — every dimension (D=2^20, D_L=2^22, LEN_E=LEN_R=2^20, com_S 1 row) is
an exact power of two, and the commitment row-count guards (492–496) reject any
shape mismatch. The L flat layout `plane·D + i·NCOL + j` (4B rows) is enforced
by the row-count check `com_L.size == 4B` and the plane-bit opening points.

### 9. Selftest honesty — clean

- The five evil modes are precisely scoped: `strict=false` is passed only to the
  recursion that is *meant* to be caught downstream (exp lookup for evil=1,
  row-sum for evil=4; lines 260/332), and the bracket-residual range throw is
  bypassed only for evil 2/3 (149–160). Every other recursion runs `strict=true`
  on a witness that is self-consistent with the planted corruption, so no prover
  throw masks the verifier check. I confirmed each mode rejects at exactly the
  named check **and** that earlier checks pass (e.g. evil=5 reaches the V2
  opening before I1; evil=2 passes the limb lookup + V1 + V2 before I1).
- The selftest requires the reject **reason string** to contain the expected
  check (861), so a wrong-check rejection or a prover-side throw would FAIL, not
  mask.
- Byte-tamper coverage is complete: `PROOF_FILES` (791–802, 32 entries) ==
  every file `verify()` opens (dims + 9 commitments + lookup_E/L + hp_rs/v1/v2 +
  lvals + 16 ipa). Cross-checked against the `open_or_die`/`G1TensorJacobian`
  ctor/`read_*` calls (473–505) — 32/32, nothing the verifier reads escapes.
- The evil==0 fold-vs-`multi_dim_me` convention checks are present for the exp
  lookup, limb lookup, row-sum, V1 and V2 terminals (266–272, 307–313, 337–343,
  369–375, 395–401), each run inside every honest prove.

### 10. Numeric / representation — clean

Host bracket math is `long long` with the §6 bounds: `E<<17 < 2^45`,
`2PS < 2^55`, `r1,r2 < 2S < 2^39 < LEN_R² = 2^40`, `LENR2 = (long long)LEN_R²`
fits int64 — all `< 2^63`. The honest-prover throws (`z_` out of domain,
`r1∉[0,2S)`, residual `≥ LEN_R²`, `S<1`) are completeness guards the verifier
does not rely on. Committed values are PLAIN Fr throughout (`FrTensor(uint,const
int*/long*)` ctors). X_E ≤ 2^28 fits the int32 table format. No `__int128`
needed and none used; no new kernel ⇒ no Montgomery question.

---

## MINOR findings / notes (none block the gate)

**MINOR-1. No semantic out-of-range-limb forgery in this selftest.** The limb
range lookup is the sole enforcer of `r1,r2 ≥ 0 ∧ < LEN_R²`, and is load-bearing
for the bracket (a broken lookup would let `r̃1` reconstruct to an arbitrary
value and defeat the `[0,2S)` bound). No evil mode plants e.g. `L[0,idx] +=
LEN_R` with a compensating reconstruction (the analogue of rmsnorm's MINOR-2 /
rescale's rem forgery, which keep I1/I2 valid and must be caught by the lookup
at round 0). The machinery is the shared, validated logUp path and IS
byte-tampered, so risk is low. Fix (cheap, next touch): add evil=6 that bumps a
committed limb out of range and recomputes the reconstructed residual to keep
I1/I2 satisfied → expect a "limb lookup round 0" rejection.

**MINOR-2. The four L-plane openings have no dedicated evil==0
convention block.** Obligations 1–5 each cross-check their fold terminals
against `multi_dim_me`; obligation 6 (plane openings + S_id) does not. The
convention is the *same* `open_prove`/`open_verify` + `multi_dim_me({…,
NCOL},{4B,NCOL})` path already cross-checked by the limb-lookup evil==0 block
(307–313), so it is transitively validated and honest ACCEPT exercises it. Not a
gap; a coverage note.

**MINOR-3. Verifier-side public-weight folds (`W_rs`, `W1`) are not
cross-checked against `multi_dim_me` in evil==0** (design §9.5 risk 5 suggested
this). They use `fold_public` = the identical `k_fr_fold` kernel and challenge
sequence as the in-sumcheck weight fold, so `W_f` is necessarily the genuine
`W̃(pt)` for honest ACCEPT, and the E/P `S_f2` evil==0 checks already exercise
`multi_dim_me`-vs-fold for the shared convention. Defense-in-depth only.

**MINOR-4. Missing-proof-file handling is a throw, not a graceful REJECT.**
`open_or_die`/the `G1TensorJacobian(path)` ctor throw on a missing/short file,
which propagates out of `verify()` (uncaught) rather than returning `false`.
This still cannot cause a false ACCEPT (the process aborts with nonzero exit),
so it is not a soundness gap; only a robustness nicety. The byte-tamper tests
mutate-in-place and so never exercise it.

**MINOR-5. Scoped assumption (inherited, not enforced): the chain byte-equalities
of §4.7 must hold.** `com_z` is only used homomorphically and is **not** opened
standalone; `com_P`'s integer interpretation flows downstream via the committed
tensor, not the `P-int32-out.bin` data file (which the verifier never reads).
Soundness of the obligation in the pipeline therefore relies on the orchestrator
enforcing `com_z == scores_rescale10/com_Xr.bin` and `com_P ==
values-matmul/com_X.bin` (byte-identical). A standalone ACCEPT binds `(z_,E,S,P)`
internally consistently but does not pin `z_` to the upstream score tensor.
This is by design (§4.7); worth one sentence in the orchestrator notes, no
driver change.

**MINOR-6. Reused FS labels across the three hadamards (`"hp0".."hp3"`) and two
lookups (`"A_f"/"S_f"/"m_f"`).** Positionally distinct and separated by the
`ev_S`/`c1`/`c2`/`v*` absorbs, exactly as §5 note 1 / rmsnorm precedent. Verified
harmless (the transcript state is positional); noted so nobody "fixes" it.

---

## What I would do before trusting it in the gate (summary)

**Accept the file.** The verifier is independent, the FS schedule is identical
to the prover's and honors §5 label-by-label, all 16 openings are required and
correctly pointed, the rounding bracket is forced exactly on both the `r1 ≥ 0`
and `r1 < 2S` sides, the never-committed broadcast is genuinely pinned to
`com_S`, and there are no new CUDA kernels. Optional, cheap, on next touch: add
the out-of-range-limb semantic evil mode (MINOR-1) and add the
orchestrator-notes sentence per MINOR-5.
