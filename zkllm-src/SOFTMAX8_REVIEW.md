# SOFTMAX8_REVIEW — independent soundness audit of zkob_softmax8.cu + the zkob_headmerge perm-flag diff

Reviewer: second engineer (independent review). Date: 2026-06-11.
Scope:
- `/root/zkllm/zkob_softmax8.cu` (1434 lines, NEW driver) against the normative
  `STAGE3_FAITHFUL_DESIGN.md` §4.3 (DESIGN FINAL 2026-06-11), the temp-128 ancestor
  `SOFTMAX_DESIGN.md` (whose §4 obligations softmax8 inherits with §4.3's redone
  arithmetic), `PHASE0_NOTES.md` (§7–§15 pinned machinery), the trusted shared headers
  (`vrf_common.cuh`, `zkob_lookup.cuh` — audited for USAGE, not edited), and the bar set
  by `SOFTMAX_REVIEW.md` / `ROPE_REVIEW.md`.
- `/root/zkllm/zkob_headmerge.cu` — **the DIFF only** vs the previously-audited
  pre-flag version at commit `f792978` (`git -C zkllm-src show
  f792978:zkllm-src/zkob_headmerge.cu`), against `STAGE3_FAITHFUL_DESIGN.md` §4.2 + §1.3.

Selftests independently re-run (logs under `/tmp/s8-audit/`), both exit 0:

| driver | result | checks | real-scale prove | verify | proof+coms |
|---|---|--:|--:|--:|--:|
| zkob_softmax8 | ALL PASS | 167 | 12.15 s | 12.94 s | 2,141,552 B (2.04 MB) |
| zkob_headmerge | ALL PASS | 166 | 3.50 s (each mode) | 2.19 s | 1,966,888 B (1.97 MB) |

**The selftests were NOT taken on trust**: every `verify()` check and every FS absorb in
both drivers was walked in source; the softmax8 §4.3 arithmetic (E8 table domain/sentinel,
the rounding bracket bounds, the Dm-binding identity, the mx-broadcast pinning) was
re-derived independently; the headmerge diff was byte-compared formula-path against the
audited original; and every evil mode / splice case was traced to confirm it rejects at
exactly the named check with earlier checks passing. The persisted `zkllm-src/` copies of
both files are byte-identical to the audited `/root/zkllm` files (verified via `cmp`).

## VERDICT

- **zkob_softmax8: SOUND**
- **zkob_headmerge (perm-flag diff): SOUND**
- **Overall: SOUND** — no critical or major soundness gap in either file.

softmax8's verifier enforces exactly the §4.3 relation — (R0) the masked-diff-with-sentinel
`Dm` via the cD1/cD2/vDm "Dm identity"; (R1) `E = X_E8[Dm−LOW8]` via the homomorphic
`com_Dm + r·com_E` mapping lookup with the sentinel forcing masked `E=0` by the table;
(R2) the pure-eq causal row-sum; (R3) `P = round_half_up(2^16·E/S)` via the
2×14-bit-limb bracket forced exactly on both sides. Every prover-supplied value `verify()`
consumes is anchored to a commitment / public recomputation / the recomputed logUp anchor;
the FS schedules of `prove()` and `verify()` are absorb-for-absorb identical and match
§4.3 label-for-label; there are **zero new CUDA kernels**. The headmerge diff is minimal
and exactly the §4.2 scope (gather formula + PERM absorb + dims field + CLI + selftest);
the PERM absorb and dims cross-check make cross-mode splices diverge in both directions;
the concat gather/assembly is the §1.3 plain head-concat M; and the pi157 formula paths
are byte-identical to the audited original. The minor notes below are inherited-by-design
chain reliances and coverage observations; none lets a cheating prover pass either verifier.

---

## PART 1 — zkob_softmax8.cu

## CRITICAL findings — none

Both automatic-CRITICAL categories are clean:

- **New CUDA kernels: ZERO** (G1 *and* Fr). `grep -cE 'KERNEL|__global__' zkob_softmax8.cu`
  = 0. All G1 work goes through the proven 1-thread helpers (`h_mul`/`h_add` build
  `com_comb = com_Dm + r·com_E` row-by-row on host, verify 737–744; `h_scalar` for the
  plain-field identities). All Fr work uses pre-existing, selftest-probed header kernels
  (`k_bcast_rows`, `k_fr_emul`, `k_fr_fold` via `fold_public`, `k_bump`,
  `tlookup_inv_kernel`, `build_eq_tensor`/`k_eq_expand`, `fs_hadamard`/`k_hp3_step`,
  `fs_phase1`). STAGE3 §2.8's one new Fr kernel `k_pp_expand` belongs to `zkob_rowmax`,
  **not** softmax8 — softmax8 has none, as §4.3 requires. No Montgomery-convention
  question arises; the −dlto rule is honored in its strongest form.
- **Verifier accepting an unanchored prover value: none found** (full enumeration in §2).

## MAJOR findings — none

Each checklist item was walked; why each is clean follows.

### 1. Fiat–Shamir ordering — clean (absorb-for-absorb identical; matches §4.3)

Compared `prove()` (313–597) and `verify()` (716–1006) event by event. Identical, and
matches the §4.3 schedule block label-for-label:

```
absorb B,NCOL,LOW8,LEN8,LEN_R8,LOG_OUT,SENT
absorb com_z,com_mx,com_Dm,com_E,com_P,com_S,com_L,com_m_E8,com_m_L   → r → beta_E
absorb com_A_E8 → alpha_E → u_E(logD)
[exp lookup logD rounds p0..p3 → w] → absorb A_f,S_f,m_f
   → IPA(A_E8)@u_ptE, IPA(comb)@u_ptE [vs com_Dm+r·com_E], IPA(m_E8)@u_mE
→ u_d(logD)
absorb cD1 → [cD1 logD rounds] → absorb S_f2,U_f2 → IPA(z_)@pt_cd1
absorb cD2 → [cD2 logD rounds] → absorb S_f2,U_f2 → IPA(mx)@pt_cd2_rows
absorb vDm → IPA(Dm)@u_d
→ beta_L → absorb com_A_L → alpha_L → u_L(logDL)
[limb lookup logDL rounds] → absorb A_f,S_f,m_f → IPA(A_L)@u_ptL, IPA(L)@u_ptL, IPA(m_L)@u_mL
→ u_b(logB) → absorb ev_S → [rs logD rounds] → absorb S_f2,U_f2 → IPA(E)@pt_rs, IPA(S)@u_b
→ u_r(logD) → absorb c1 → [V1 logD rounds] → absorb S_f2,U_f2 → IPA(E)@pt1
→ absorb c2 → [V2 logD rounds] → absorb S_f2,U_f2 → IPA(P)@pt2, IPA(S)@pt2_rows
→ absorb v00,v10,v01,v11 → IPA(L00,L10,L01,L11) → absorb S_id → IPA(S)@ur_row
[verifier-only I1,I2,Dm identity — no absorbs]
```

Every challenge is squeezed only after the message it binds: `r`/`beta_E` after all nine
base commitments; `alpha_E` after `com_A_E8`; `u_d` after the exp lookup closes; each
sumcheck round challenge after that round's four evals; `u_r` after the row-sum closes;
each IPA round challenge after its own L,R (inside `open_prove`/`open_verify`). Each opened
value (cD1, cD2, vDm, ev_S, c1, c2, S_f2/U_f2, v00..v11, S_id, the three lookup terminals)
is absorbed before its opening runs. **`com_mx` is absorbed 2nd among the base commitments
(prove 318 / verify 721)** — before any challenge, so the prover cannot pick mx after
seeing the evaluation points; it is byte-pinned to the chained rowmax `com_mx` by edge RM2
(confirmed in selftest, 1366). Reused labels (`"p0".."p3"` across both lookups;
`"hp0".."hp3"` and `"S_f2"/"U_f2"` across the five hadamards; `"A_f"/"S_f"/"m_f"` across
the two lookups) are positionally disambiguated by the interleaved
`cD1/cD2/vDm/ev_S/c1/c2/v00..` absorbs — softmax/rmsnorm precedent, transcript state is
positional. No absorb present on one side and missing on the other; no order difference.
The verifier-only checks (U_f2==1, terminals, I1, I2, Dm identity) absorb nothing, so check
placement cannot diverge the transcript.

### 2. Verifier independence — every disk value anchored

| value (file) | anchor |
|---|---|
| dims.bin (B,NCOL,LOW8,LEN8,LEN_R8,LOG_OUT,SENT) | cross-checked == CLI/derived args (664–667); same scalars absorbed |
| com_z | absorbed (720); forms `com_comb` (homomorphic); opened at pt_cd1 (823, cD1 S_f2); chained upstream by edge SX8a (orchestrator) |
| com_mx | absorbed (721); opened at pt_cd2_rows (853, cD2 S_f2); chained by edge RM2 |
| com_Dm | absorbed (722); forms `com_comb`; opened at u_d (858, vDm) |
| com_E | absorbed (723); forms `com_comb`; opened at pt_rs (932) and pt1 (959) |
| com_P | absorbed (724); opened at pt2 (982) |
| com_S | absorbed (725); opened at u_b (934), pt2_rows (986, V2 U_f2), ur_row (1005, S_id) |
| com_L | absorbed (726); opened at u_ptL (904) **and** the 4 plane points (1000) |
| com_m_E8 / com_m_L | absorbed before β; opened at u_mE / u_mL (793 / 906) |
| com_A_E8 / com_A_L | absorbed after their β; opened at u_ptE / u_ptL (789 / 902) |
| lookup_E8/L: ev[] | round-by-round `p(0)+p(1)==claim` from the **recomputed** anchor α+α² (749 / 866) — prover claim0 never trusted |
| lookup_E8/L: A_f,S_f,m_f | IPA openings; B_f,T_f recomputed from the public table folded with the verifier's own phase-2 challenges (765–777 / 881–890) |
| hp_cD1/cD2: claim_H (cD1,cD2) | absorbed; bound by their round chain to the verifier-rebuilt `eq(u_d)⊙MK` fold × opened S_f2 × **forced U_f2==1** (811/818, 838/845) |
| hp_cD1/cD2: S_f2 | opened vs com_z (823) / com_mx @row-bit suffix (853) — load-bearing |
| vdm.bin (vDm) | absorbed (857); anchored by IPA vs com_Dm (858) **and** consumed by the Dm identity (1035) |
| hp_rs: claim_H (ev_S) | absorbed; bound by the row-sum chain (pure-eq `eq_acc` shortcut, row rounds only) × S_f2 × **forced U_f2==1** (927) |
| hp_v1: claim_H (c1) | bound by V1 to `eq̃(u_r)·Ẽ(u_r)·1`, S_f2 vs com_E, **U_f2==1** (954) |
| hp_v2: claim_H (c2) | bound by V2 to `eq̃(u_r)·P̃(pt2)·S̃b(pt2)`; S_f2 vs com_P, U_f2 vs com_S @pt2_rows |
| lvals.bin: v00,v10,v01,v11,S_id | each an IPA opening vs com_L (planes) / com_S (S_id) (1000 / 1005) |
| ipa_*.bin (19) | each consumed by exactly one `open_verify`, result required (`if(!…) RJ`) |
| expmap8 table (maph) | NOT from obdir — CLI/registered path; B_f/T_f rebuilt verifier-side; sentinel `maph[LEN8−1]==0` enforced (654) |

Round-count guards pin ev lengths to `4·logD` (exp/cD1/cD2/rs/v1/v2) and `4·logDL` (limb)
(701–706); commitment row counts pinned (680–684); layout params re-checked (649–653). No
early-accept path; `ACCEPT` is the last line after all 19 openings and the three identities
(1037).

### 3. The §4.3 arithmetic — re-derived independently, enforced exactly

**E8 table domain / sentinel.** Real domain `v ∈ [LOW8, LOW8+LEN8) = [−1048574, +1]`,
`SENT = LOW8+LEN8−1 = +1`, table `X_E8[k]=rint(2^16·e^{(LOW8+k)/4096})` with `tab[v>0]=0`
so the single `v=+1` slot (the last entry) is 0. The verifier **enforces the sentinel**
`maph[LEN8−1]==0` (654; prove 145–146) — this is what replaces glu's `mapped(0)==0` check
and makes the masked-position row real. **Can a prover exploit the sentinel row?** No: an
allowed position mapping to SENT is exactly evil=2 and is caught by the Dm identity (cD1/cD2
are MLEs of the TRUE masked diffs, so `vDm` for the forged Dm differs whp). **Is the
sentinel value excluded from the honest allowed range?** Yes — honest allowed
`Dm = z−mx ∈ [LOW8, 0]` (the diagonal gives 0, dominance gives ≤ 0), and `SENT = +1 > 0`,
so no allowed position can honestly reach it; the only in-domain integer > 0 is +1 = SENT,
and a committed `Dm=+1` at an allowed position is the Dm-identity catcher. The index
`Dm−LOW8 ∈ [0, LEN8)` is **forced in-domain by the mapping lookup itself** (a committed Dm
outside `[LOW8, LOW8+LEN8)` has no combined-table row ⟹ the recomputed multiplicity anchor
fails at round 0).

**The rounding bracket, bounds re-derived (S ≤ 2^26, 2×14-bit limbs).** `E ∈ [0, 2^16]`
(max at `v=0` ⟹ `2^16·e^0 = 65536`). `S = Σ_j E[i,j] ∈ [2^16, 2^26]`: the lower bound is
structural (the allowed argmax has `Dm=0 ⟹ E=2^16`, a single summand ≥ 2^16, division never
degenerate — the §4.3 "S ≥ 2^16 always" invariant); the upper bound is `≤ 1024·2^16 = 2^26`.
`num = (E<<17)+S = 2^17·E + S`; `P = ⌊num/2S⌋ ∈ [0, 2^16]` (E ≤ S componentwise since E is a
non-negative summand of S). `r1 = num − 2PS ∈ [0, 2S) ⊂ [0, 2^27)`; `r2 = 2S−1−r1 ∈
[0, 2^27)`. Both `< LEN_R8² = 2^28` ⟹ exactly two 14-bit limbs (`hi = r1/2^14 < 2^13 <
2^14`). I confirmed the host packing (219–222: `r1c%LEN_R8`, `r1c/LEN_R8`, likewise r2) and
the verifier's `F_LENR = 2^14` reconstruction weight (1009, 1011–1012). The two identities
close the bracket on **both** sides:
- **I1** (1014–1017): `2^17·c1 + S_id − 2·c2 == v00 + 2^14·v10`. `F_2P17 = 1<<(LOG_OUT+1)
  = 2^17` (1010). LHS is the MLE at `u_r` of `r1 = 2^17·E + S_bcast − 2·P·S_bcast` (c1 =
  Ẽ(u_r), c2 = MLE(P⊙S_bcast)(u_r), S_id = S̃_bcast(u_r)); RHS is the MLE of the committed
  2-limb reconstruction of r1. (The MK factor that softmax carried on the `2^17·E` term is
  unnecessary here because masked `E=0` by the sentinel — see §4.) ⟹ `r1` = the limb
  reconstruction as tensors whp.
- **I2** (1020–1022): `r̃1 + r̃2 + 1 == 2·S_id`, i.e. `r1 + r2 + 1 = 2S` as tensors.
- The **limb lookup** binds every plane of `com_L` to `[0, 2^14)`, so `r1, r2 ∈ [0, 2^28)`
  as integers. With `2·LEN_R8² = 2^29 < p` and `2S ≤ 2^27 < 2^28`, the I2 congruence forces
  integer equality ⟹ `r1 ∈ [0, 2S)` ⟹ `P = round_half_up(2^16·E/S)` uniquely; **masked ⟹
  P=0** falls out (at masked `E=0`, `r1 = S(1−2P) ∈ [0,2S) ⟺ P=0`). Headroom (2^27 vs the
  2^28 limb ceiling) is harmless — the load-bearing bound is `r1 ≥ 0` (limb non-negativity)
  + I2 (`r2 ≥ 0`). Both directions are semantically forgery-tested: evil=3 (P+1 at masked,
  true r1 = −S, mod-truncated limbs) → I1; evil=4 (P−1 at a diagonal, `r1'=r1+2S` honest
  limbs, r2 mod-truncated) → I2. Intermediates `2^17·E ≤ 2^33`, `2PS ≤ 2^43`, `LENR2 = 2^28`
  all `< 2^63` (host `long long`).

**Dm = z − mx_bcast ≤ 0 binding (indexing + in-domain).** The committed `Dm` is bound to
the exact masked-diff-with-sentinel tensor by the **Dm identity** (verify 1024–1036):
`D̃m(u_d) == cD1 − cD2 + SENT·(1 − k_MK)`, with `k_MK = Σ_b eq(u_d,b)·MK(b)` recomputed by
the verifier itself (1026–1030), `cD1 = MLE(MK⊙z_)(u_d)` opened vs com_z, `cD2 =
MLE(MK⊙mx_bcast)(u_d)` opened vs com_mx, `vDm = D̃m(u_d)` opened vs com_Dm. The RHS is the
MLE at `u_d` of `MK·(z−mx) + SENT·(1−MK)` = the correct Dm tensor; equality at the
post-commitment random `u_d` forces (Schwartz–Zippel) `Dm` = that tensor everywhere whp, so
masked `Dm=SENT` and allowed `Dm=z−mx`. The index is forced in-domain by the mapping lookup
(above). **Note the division of labor:** softmax8 alone does NOT prove allowed `z−mx ≤ 0`
(an allowed `Dm` could be `+1=SENT`, i.e. mx below the true row max, silently dropping that
probability — caught only by the chained rowmax, edges RM1/RM2, which prove `mx = allowed
row max`). This is exactly §4.3's composition-soundness note; see MINOR-1.

**The mx broadcast binding.** `mx_bcast` (cD2's `U`-factor analog: `Sc=mxb`) is materialized
by `k_bcast_rows` and **never committed**. cD2's terminal `S_f2 = m̃xb(pt_cd2)` is opened
against **com_mx at the row-bit suffix `pt_cd2_rows = pt_cd2[logC..]`** (853): since
`pt_cd2` squeezes column bits into `[0,logC)` and row bits into `[logC,logD)` (the
fold collapses the MSB/row bit first; cross-checked by the evil==0 block 423–428, which
asserts `S_f2 == mxb.multi_dim_me(...) == mx_t.multi_dim_me({pt_cd2_rows})`), the
column-broadcast MLE depends only on the row bits, so the opening forces `mxb = mx_bcast`
whp. evil=7 (masked-index `mxb` bump — cD2's *value* is unchanged because MK zeroes that
term, the sumcheck terminal is self-consistent with the corrupted tensor, so the terminal
check passes and the IPA is the SOLE catcher) → "IPA opening of cD2 terminal vs com_mx".
Confirmed live. This is the softmax V2/evil-5 mechanism applied to the new mx shift.

### 4. Inheritance check vs zkob_softmax — every load-bearing requirement present; deltas are per-design

| softmax (temp-128) load-bearing requirement | softmax8 | per-design? |
|---|---|---|
| U_f2==1 in row-sum / V1 | present (927 / 954) | yes |
| V2 U_f2 opened vs com_S @row suffix (not forced 1) | present (986) | yes |
| broadcast row-bit opening of never-committed tensor | V2 (986) **and** new cD2 mx-broadcast (853) | yes |
| homomorphic combined-lookup commitment | `com_Dm + r·com_E` (was `com_z + r·com_E`) (737–744) | **delta** §4.3 ob.1 — table now maps Dm, not z_ |
| exp lookup range-binds the committed input | binds **Dm** ∈ [LOW8,+1], **not z_** | **delta** §4.3 / open-Q5 — z_ no longer range-proven (MINOR-1) |
| row-sum / V1 / bracket carry the MK weight | MK **dropped** (pure-eq `eq_acc` shortcut; r1 = 2^17·E…) | **delta** §4.3 ob.4/5 — sound because masked `E=0` by sentinel |
| limb range lookup is the sole `r≥0` enforcer | present, `LEN_R8=2^14` (was 2^20), 2 planes/residual | **delta** §4.3 ob.3 — smaller S ⟹ smaller limbs |
| round/row-count + commitment-row guards | present (680–706) | yes |
| recomputed logUp anchor α+α² | present (749 / 866) | yes |
| evil==0 fold-vs-`multi_dim_me` convention checks | present at all 9 terminals (363, 397, 425, 502, 528, 553, 577, 592, 609) | yes |

The single conceptual load-bearing addition vs softmax is that **dropping MK from the
row-sum/V1/bracket weights is sound only because masked `E` is exactly 0**, which is itself
enforced by the *interlock* of three checks — the Dm identity (Dm=SENT at masked) + the
mapping lookup (E=X_E8[Dm−LOW8]) + the sentinel check (table[SENT]=0) — not by any single
line in the row-sum block. The interlock holds (each of the three is verifier-enforced), so
the simplification is correct; flagged as MINOR-3 so a future edit to the Dm block or the
table cannot silently loosen the row-sum semantics.

### 5. Selftest honesty — clean, incl. the guard-test fix

- The 8 semantic evil modes are precisely scoped: `strict=false` is passed only to the
  recursion meant to be caught downstream (exp lookup for evil=1, line 354; limb lookup for
  evil=8, 465; row-sum for evil=5, 494), and the bracket-range throws are bypassed only for
  evil 3/4 (206). Every other recursion runs `strict=true` on a witness self-consistent with
  the planted corruption. My re-run confirms each mode rejects at exactly the named check
  (evil=1→exp lookup r0; 2→Dm identity; 3→I1; 4→I2; 5→row-sum r0; 6→V2 U_f2 vs com_S;
  7→cD2 terminal vs com_mx; 8→limb lookup r0), and the selftest requires the reject **reason
  string** to contain the expected check (1166), so a wrong-check rejection or prover-side
  throw FAILS, not masks. evil=2 (the Dm certifier — drops a probability with a valid
  `(SENT,0)` table row so the lookup passes) and evil=7 (the masked-broadcast certifier) are
  genuinely new coverage for the new blocks.
- **The guard-test fix is real and the test exercises the intended guard.** The
  z-envelope guard (1239–1241) sets `z2[9] = 2^19` but passes the **original** mxh ("mx kept
  honest-original"). In `prove`, the per-row mx-envelope check (153) runs before the per-j
  z-envelope check (158); idx 9 is (i=1,j=1), an allowed position. With mxh honest, row 1's
  mx check passes and the loop reaches `j=1` where `z2[9] ≥ ENV` throws **"z_ outside the
  +-2^19 envelope"** (159) — the intended z guard. Had the test recomputed mx from the
  oversized z2 (the original bug), `mx[1]=2^19` would trip the mx guard (153) first
  ("mx outside"), so the z guard would never fire. The fix is correct, and the separate
  mx-envelope guard is still covered by its own case (1242–1244). All 11 guards PASS in my
  re-run.
- Byte-tamper coverage is complete: `PROOF_FILES` (1068–1081, 40 entries) == every file
  `verify()` opens (dims + 11 commitments + 2 lookups + 5 hp + vdm + lvals + 19 ipa).
  Cross-checked against the `open_or_die`/`G1TensorJacobian` ctor/`read_*` calls — 40/40,
  nothing the verifier reads escapes. The tamper loop catches parse-throws fail-closed
  (1178–1183).
- The real-scale case actually **invokes `zkob_rowmax` causal** to produce the chained mx
  (1320–1336) and checks edge RM2 (`com_mx` byte-identity, 1366) and edge SX8a/RM1
  (`com_z` byte-identity, 1371) — stronger than softmax's selftest.

### 6. New kernels — ZERO (confirmed, see CRITICAL).

---

## PART 2 — zkob_headmerge.cu (perm-flag DIFF vs f792978)

The diff is 427 lines (`/tmp/s8-audit/headmerge.diff`); every hunk falls inside §4.2's
pinned scope. No core verify algebra (the per-head hadamard chain, `Σ c_h == ev`, the IPA
openings, the eq-tensor construction, `fold_public`, FS-challenge derivation, commitment
structure) is touched except threading the `perm` argument into `gather_Wm`.

### 7. The diff is minimal per §4.2 — confirmed, nothing out of scope

| change | §4.2 scope item |
|---|---|
| `gather_Wm` gains `uint perm`; `perm==0` keeps the §1.3 π⁻¹ verbatim, `perm==1` uses `i=t; j=e` (94–104) | gather formula |
| `parse_perm` helper (107–110) | CLI |
| O2 assembly branches on `asm_perm` (163–172); `perm==1` is the plain-concat layout; `evil==5` swaps mode; old `evil==1` guarded pi157-only (173–177) | gather/assembly + selftest |
| dims.bin gains a 4th u32 `perm` (206); verify reads 4 and cross-checks `d[3]!=perm` (283–286) | dims field |
| `absorb_u32(tr,"PERM",perm)` immediately after `"HD"`, both prove (211) and verify (306) | PERM absorb |
| prove/verify signatures + the two `gather_Wm` call sites (233, 336) take `perm` | threading |
| selftest: `selftest_case`/`selftest_real` parametrized by perm, both modes looped; new `selftest_splice`; evil=5 added (rest 383–650) | selftest |
| header/usage comments | documentation |

I confirmed **nothing else changed** — in particular the per-head sumcheck, the `U_f2==1`
check, the `Σ c_h == ev` check, all 13 IPA openings, and the eq/commit machinery are
byte-unchanged. **No MAJOR finding.**

### 8. PERM absorb + dims cross-check make cross-mode splices diverge — both directions

Two independent defenses, exactly as §4.2 intends:
1. **dims cross-check** (286): the verifier is invoked with `perm` from its own CLI/public.json
   (`headmerge_perm`); dims.bin carries the prover's perm; a mode mismatch → `RJ("dims.bin
   mismatch")`.
2. **PERM absorb** (211/306): even if dims.bin's perm field is forged to match the verifier,
   the transcript was built with the prover's actual perm absorbed **before `com_O2` and
   before the grid challenge `u`**, so the squeezed challenges diverge and a downstream
   sumcheck check fails.

`selftest_splice` (540–648) exercises **both directions** (pv=0→1 and pv=1→0) and **both
layers**: the dims-mismatch rejection, then forge dims' PERM field (offset 12) and confirm
rejection via transcript divergence (`reason` must NOT contain "dims.bin"), then
restore-and-reverify ACCEPT. My re-run shows both directions reject at `dims.bin mismatch`
(layer 1) and at `merge hadamard 00 round 1 p(0)+p(1) != claim` (layer 2, transcript
divergence). **Can a prover omit/forge the absorb?** No: the verifier always absorbs its own
`perm`, so an omitted/wrong PERM in the prover transcript diverges the challenges and fails;
`perm` is a public input the prover cannot choose to match both a concat statement and a
pi157 verifier (the verifier rebuilds `Wm` with its own `perm`, so the head terminals fail
too). Because the absorb sits before any challenge, a challenge cannot be chosen before the
mode is bound.

### 9. concat gather is the §1.3 plain head-concat M; pi157 paths byte-identical to the original

**concat correctness.** §4.2 pins `Wm_h[t·HD+d] = E_u[t·C_pad + (HD·h+d)]` and `O2[t,
HD·h+d] = out_h[t,d]`. The code's `perm==1` gather (102) sets `i=t, j=e=HD·h+d` ⟹
`Wm[t·HD+d] = Eh[t·C_pad + (HD·h+d)]` ✓; the `asm_perm==1` assembly (170–172) sets
`O2g[i·C_pad+j] = outs[j/HD][i·HD + (j%HD)]` ⟹ with `j=HD·h+d`, `O2[i, HD·h+d] = out_h[i,d]`
✓ — exactly the §1.3 M (plain head-concat `O[t,64h+d]=out_h[t,d]`, pad columns `j≥C` left 0).
I checked gather and assembly are mutually consistent: `Σ_h c_h = Σ_{t,h,d}
eq(u;(t,HD·h+d))·out_h[t,d] = Õ2(u)` iff `O2 = M` with zero padding, which is what the
honest `Σ c_h == ev` identity binds; padding is forced to exact 0 by the same identity
(the old evil=2 "junk in a padding column" test still rejects in both modes). evil=5 (assemble
O2 with the OTHER mode's layout) is the concat-mode certifier replacing the now-vacuous
evil=1 (in concat, `O2:=M` IS honest) — confirmed rejecting at "sum of head claims != ev".

**pi157 unchanged.** Byte-comparing the `perm==0` formula paths against f792978: the
`gather_Wm` `perm==0` branch (`m=e*B+t; i=m/C; j=m%C`, line 102 region) and the
`asm_perm==0` O2 assembly (164–169) are character-for-character the original π⁻¹ gather and
π(concat) assembly. The only behavioral change in pi157 mode is the added `absorb_u32
"PERM" 0` — a **strengthening** (domain separation), not a weakening; it means a pi157 proof
from the new binary is not transcript-compatible with one from the pre-flag binary, which is
expected and correct (the baseline-native submission retains its own f792978 binary;
driver coexistence per §4.4). No soundness change to the pi157 path.

---

## MINOR findings / notes (none block the gate)

**MINOR-1 (softmax8; inherited-by-design, but BROADER than softmax's). z_ and mx are not
range-proven inside softmax8, and the dominance/no-wrap argument is load-bearing on the
chain edges.** Unlike softmax (whose R1 lookup range-bound `z_` directly), softmax8's exp
table indexes `Dm`, not `z_`/`mx` (verify forms `com_comb = com_Dm + r·com_E`, 737–744). So
`verify()` never bounds `|z_|` or `|mx|`; the `prove` throws (`z_/mx outside ±2^19`, 153–159;
`allowed diff > 0`, 163; `diff < LOW8`, 165) are **completeness guards only**. A standalone
softmax8 ACCEPT therefore binds `(Dm,E,S,P)` internally consistent with the committed
`z_,mx`, but (a) does not pin `z_,mx` to upstream, and (b) does not by itself prove allowed
`z−mx ≤ 0` or exclude a field-wrap of `z−mx`. The defenses are the orchestrator byte-edges
**RM1** (`rescale10/com_Xr ≡ rowmax/com_z`), **RM2** (`rowmax/com_mx ≡ softmax8/com_mx`),
**SX8a** (`rescale10/com_Xr ≡ softmax8/com_z`), **SX8b** (`softmax8/com_P ≡ values-fc/com_X`),
plus the rowmax instance itself (proves `mx = allowed row max` ⟹ allowed diffs ≤ 0 ⟹
allowed Dm ≠ SENT) — all on the **same** com_z/com_mx. This is STAGE3 §4.3's composition
note and open-Q5, the softmax-MINOR-5 posture made explicitly broader. **Orchestrator MUST
enforce RM1/RM2/SX8a/SX8b byte-identically and run the chained rowmax**; a standalone
softmax8 proof proves materially less. The optional belt (one `tLookupRange(−2^19, 2^20)`
on z_ per head, design Q5) is NOT included in v1 — acceptable given the chain reliance is
the standing accepted posture.

**MINOR-2 (softmax8; inherited). Missing/short proof file ⇒ throw, not graceful REJECT.**
`open_or_die`/`read_lookup`/`read_hp`/the `G1TensorJacobian(path)` ctor throw out of
`verify()` (uncaught), so the process exits nonzero rather than printing REJECT. Fail-closed
(never a false ACCEPT); the byte-tamper tests mutate in place and so never exercise it. Same
as softmax MINOR-4 / rope MINOR-7: **the orchestrator must treat any nonzero exit as
reject**, not parse for the REJECT line.

**MINOR-3 (softmax8). The MK-free row-sum/V1/bracket weights are sound only via the
masked-E=0 interlock.** §4.3 drops the MK factor that softmax carried (row-sum weight is
pure broadcast eq using the `eq_acc` shortcut, 916–928; V1 pure eq; r1 = 2^17·E…). This is
correct **only because** masked `E` is exactly 0, which is enforced by the conjunction of
the Dm identity (Dm=SENT at masked) + the mapping lookup (E=X_E8[Dm−LOW8]) + the sentinel
check (table[SENT]=0) — three separate verifier checks, no single line in the row-sum block.
It holds today; noted so a future change to the Dm block or the registered table cannot
silently loosen the causal row-sum / bracket without an obvious failure. (Defense-in-depth
only; not a gap.)

**MINOR-4 (softmax8). No standalone forgery test for the cD1 z_-side terminal.** The Dm
block's two certifiers target the identity (evil=2) and the mx broadcast (evil=7); cD1's
`S_f2` opening vs `com_z` is exercised by the byte tamper of `ipa_z_cD1.bin` and by every
honest pass, but no *semantic* evil bumps a never-committed cD1 intermediate (there is none
— cD1 folds the committed `z_` directly, unlike cD2's broadcast). So there is nothing
analogous to forge; coverage is adequate by construction. Coverage note only.

**MINOR-5 (softmax8). Real-scale tamper is single-file.** Like softmax/rope, the real-scale
case tampers one file (`lookup_E8.bin @ 4+32`, 1375); the three toy shapes tamper all 40
files each. Acceptable — the toy cases exercise every verifier-read file with
restore-and-reverify.

**MINOR-6 (both; cosmetic). Reused FS labels and duplicated static helpers.** softmax8
reuses `"hp0".."hp3"`/`"S_f2"/"U_f2"` across the five hadamards and `"p0".."p3"`/`"A_f"/
"S_f"/"m_f"` across the two lookups; `fold_public`/`tamper_byte`/`file_size` are
file-scoped copies shared verbatim with the other drivers. Positionally disambiguated /
identical to the audited drivers — verified harmless (transcript is positional); noted so
nobody "fixes" it. If `fold_public` et al. are ever hoisted into a shared header, the
"edit requires rerunning EVERY selftest" rule applies.

**MINOR-7 (headmerge; reproducibility, by design). pi157 transcripts are not
binary-compatible across the flag.** The added `absorb "PERM"` means a pi157 proof from the
new binary differs from one made by f792978. This is intended (§4.2: the flag, not a
constant, keeps one binary serving both registered statements; the baseline-native
submission keeps its own f792978 binary and remains re-verifiable). The orchestrator must
pass `headmerge_perm` from public.json to BOTH prove and verify for each submission; a
mode/argv mismatch is caught (item 8). No action.

---

## What I would do before trusting them in the gate (summary)

**Accept both files.** zkob_softmax8's verifier is independent (every disk value anchored,
weights/`k_MK`/`B_f`/`T_f` rebuilt verifier-side, U_f2==1 forced at cD1/cD2/row-sum/V1, the
mx broadcast pinned to com_mx, the bracket forced exactly on both sides), the FS schedule is
absorb-for-absorb identical to the prover and matches §4.3 label-for-label, the E8 sentinel
correctly reserves the masked row (an allowed-position SENT is the Dm-identity catcher), the
2×14-bit bracket bounds re-derive cleanly (S ≤ 2^26 ⟹ r1,r2 < 2^28 = LEN_R8²), and there
are zero new CUDA kernels. The headmerge diff is exactly the §4.2 scope, the cross-mode
splice diverges in both directions via the PERM absorb and dims cross-check, the concat
gather/assembly is the §1.3 plain head-concat M, and the pi157 formula paths are
byte-identical to the audited original. Both selftests reproduce ALL PASS (167 / 166).

The only binding obligation that lands on the **orchestrator, not the driver** is MINOR-1:
enforce edges RM1/RM2/SX8a/SX8b byte-identically and run the chained rowmax — without them a
standalone softmax8 ACCEPT does not pin `z_`/`mx` to the chain nor guarantee allowed diffs ≤ 0
(the softmax com_z posture, made broader here because z_ is no longer lookup-range-bound).
Optional, cheap, next touch: the design-Q5 `z_` range belt (one lookup/head) if instance-local
soundness is ever demanded; and a semantic cD1 z_-side note (MINOR-4) is not needed.
