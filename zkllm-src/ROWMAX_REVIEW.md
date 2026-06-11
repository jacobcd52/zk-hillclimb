# ROWMAX_REVIEW — independent soundness audit of zkob_rowmax.cu

Reviewer: second engineer (independent review). Date: 2026-06-11.
Scope: `/root/zkllm/zkob_rowmax.cu` (1767 lines) against the normative
`STAGE3_FAITHFUL_DESIGN.md` §2 (DESIGN FINAL 2026-06-11), the pinned
conventions in `PHASE0_NOTES.md`, the trusted shared machinery
(`vrf_common.cuh`, `zkob_lookup.cuh`, `fs_transcript.hpp`, and the upstream
`commitment.cu`/`g1-tensor.cu`/`tlookup.cu` paths it invokes — audited for
USAGE, not edited), and the bar set by `RMSNORM_REVIEW.md` /
`SOFTMAX_REVIEW.md` / `ROPE_REVIEW.md`. Claims checked against
`ROWMAX_REPORT.md`.

Independently re-run (artifacts under `/tmp/rowmax-audit/`):

| what | result |
|---|---|
| full selftest (`/tmp/rowmax-audit/selftest.log`) | **ALL PASS, 160 PASS / 0 FAIL, exit 0** |
| real causal 1024×1024 | 5.31 s prove / 1.93 s verify / 791,140 B / 0.69 GiB peak |
| real vpad 1024×32768 +t* | 44.28 s prove / 4.02 s verify / 974,264 B / **10.48 GiB peak, WITHIN the ~18 GiB gate** |
| memory-fix byte-identity | **independently REPRODUCED** (see §8 below) |
| `zkllm-src/zkob_rowmax.cu` persistence | byte-identical to the audited file (`cmp` clean) |

All numbers match ROWMAX_REPORT.md §3 (the 10.48 vs 10.99 GiB peak is the
report's own documented fragmentation delta; both within gate).

**The selftest was NOT taken on trust**: every `verify()` check, every FS
absorb on both sides, the X1–X5 algebra, the four constant-claim sites, the
Montgomery convention of the one new kernel, the fast-vs-slow helper algebra,
both modes' layout math, the tie-channel determinism, and the memory-fix
byte-identity basis (`lean_hadamard` vs the header `fs_hadamard`,
`fast_me_weights_dev` vs `build_eq_tensor`, `commit_chunked` vs
`Commitment::commit`/`rowwise_sum`) were walked in source. The LIMB block was
diffed against the validated `zkob_softmax.cu` (claim/Cc construction, the
α/α² round accumulators with the `k >= n1` phase split, and the terminal
identity are verbatim the validated pattern).

## VERDICT

**SOUND** — no CRITICAL or MAJOR finding. The verifier enforces exactly the
§2.1 relation (X1 binarity with the load-bearing `U_f2 == S_f2 − 1`, X2
one-hot-over-allowed with verifier-rebuilt masked weights, X3 attainment with
all three openings, X4 dominance via the bracket identity + limb range
lookup with the never-committed broadcast pinned to com_mx at the row-bit
suffix, X5 served-token binding gathered from the verifier's own t*), every
prover-supplied byte `verify()` consumes is anchored, the FS schedules of
`prove()` and `verify()` are absorb-for-absorb identical and match §2.6
label-for-label including the TSTAR preamble, the constant-claim discipline
is complete at ALL FOUR instances, and there is exactly one new kernel,
Fr-only. The findings below are documentation/coverage notes; none lets a
cheating prover pass.

---

## CRITICAL findings

**None.** Both automatic-CRITICAL categories are clean:

- **New CUDA kernels: exactly ONE, Fr-only** — `k_pp_expand`
  (zkob_rowmax.cu:71–78). `grep -cE 'KERNEL|__global__'` = 1; zero G1
  kernels. Montgomery convention walked: it mont-ifies the pair factors and
  multiplies into the plain-form running buffer (`mul(mont(a), in_plain)` =
  plain product), the exact `k_eq_expand` rule; for the eq specialization
  `(1−u, u)` the values are identical to `build_eq_tensor`'s because mont is
  additive (`mont(1)−mont(c) = mont(1−c)`). All G1 work goes through the
  proven 1-thread helpers and the pinned `fold_chain`/`dev_msm`/FS-IPA paths.
- **Verifier accepting an unanchored prover value: none found** (full
  enumeration in §2 below). No early-accept path; `ACCEPT` is the last line
  (:1344) after every opening and the DOM bracket identity.

## MAJOR findings

**None.** Each task category was walked; why each is clean follows.

### 1. Fiat–Shamir ordering — clean (absorb-for-absorb identical, matches §2.6 label-for-label)

Compared prove (:583–945) and verify (:1048–1334) event by event against the
§2.6 schedule:

```
absorb_u32 "B","NCOL","MODE","V","LEN_R","NPL"
[vpad+t*]  absorb "TSTAR" (B raw int32)            ← prove :587–588 ≡ verify :1052
absorb com_z, com_S, com_mx, com_L, com_m_L
→ beta_L ; absorb com_A_L ; → alpha_L ; → u_L(logDL)
LIMB rounds ("p0".."p3" → w, inside fs_phase1 / verifier loop)
absorb "A_f","S_f","m_f" ; IPA(A_L)@rev(ws_L) ; IPA(L)@rev(ws_L) ; IPA(m_L)@rev(ws_L[n1..])
→ u_bin(logD) ; BIN rounds ("hp0".."hp3" → w) ; absorb "S_f2","U_f2" ; IPA(S)@pt_bin
→ u_s(logB)  ; SUM rounds  ; absorb S_f2,U_f2 ; IPA(S)@pt_s
→ u_m(logB)  ; MASK rounds ; absorb S_f2,U_f2 ; IPA(S)@pt_m
→ u_a(logB)  ; absorb "ev_mx" ; ATT rounds ; absorb S_f2,U_f2 ;
             IPA(S)@pt_a ; IPA(z)@pt_a ; IPA(mx)@u_a [gen_mx]
→ u_r(logD)  ; absorb "c1" ; c1 rounds ; absorb S_f2,U_f2 ; IPA(z)@pt_c1
             absorb "c2" ; c2 rounds ; absorb S_f2,U_f2 ; IPA(mx)@pt_c2[logC..logD)
             absorb "v0"[,"v1"] ; IPA(L)@u_r / @(u_r‖0),(u_r‖1)
[vpad+t*]  → u_t(logB) ; T-BIND rounds ; absorb S_f2,U_f2 ; IPA(S)@pt_t
[no absorbs: U_f2 checks ; c2 − c1 == v0 [+ LEN_R·v1]]
```

Every challenge is squeezed only after the messages it binds: com_L and
com_m_L are absorbed **before** beta_L and com_A_L **after** it (the logUp
ordering that makes β unpredictable for the committed L/m — prove :592–596 vs
:604–606, verify :1056–1061); alpha_L after com_A_L; each grid challenge
(u_bin/u_s/u_m/u_a/u_r/u_t) only after every commitment and the preceding
block's IPAs; every sumcheck round challenge after that round's four absorbed
evals (`lean_hadamard` :252–254 ≡ header `fs_hadamard` :305–307 ≡ verifier
loops); every IPA round challenge inside `ipa_prove`/`fast_ipa_verify` after
that round's "L","R". The reused "hp0".."hp3"/"S_f2"/"U_f2" labels are
positionally disambiguated by the interleaved ev_mx/c1/c2/v-absorbs and IPA
absorbs (rmsnorm/softmax precedent). The verifier-only checks (constant
claims, U_f2 forms, terminal identities, the DOM bracket) absorb nothing, so
their inline placement cannot diverge the transcript. `lean_hadamard` is an
allocation-disciplined clone of the header recursion with the SAME kernels,
labels, challenge schedule, lagrange4(inv 6) chaining and terminal reads —
walked side-by-side; the only difference is buffer lifetime. No absorb
present on one side and missing on the other; no order difference.

### 2. Verifier independence — every disk value anchored

**verify() reads 27 (causal) / 28 (NPL=2) / 30 (vpad+t*) files** (table per
RMSNORM precedent):

| value (file) | anchor |
|---|---|
| dims.bin | checked == CLI args (:997–1003); the same six scalars are absorbed from argv, not the file (:1049–1051) |
| com_z | absorbed (:1053); opened TWICE — ATT at pt_a (:1219) and c1 at pt_c1 (:1251); chained upstream by edges RM1/L1 (orchestrator, §2.7) |
| com_S | absorbed; opened 4× (5× with T-BIND) at pt_bin/pt_s/pt_m/pt_a/pt_t (:1130/:1160/:1193/:1217/:1331) — every S terminal pinned |
| com_mx | absorbed; opened TWICE — ev_mx at u_a (:1221) and the c2 broadcast terminal at the row-bit suffix pt_c2[logC..) (:1281); chained by RM2 |
| com_L | absorbed; opened in LIMB at rev(ws_L) (:1103) AND at the NPL plane points (u_r‖plane-bit) (:1287/:1295) |
| com_m_L | absorbed; opened at the phase-2 suffix rev(ws_L[n1..]) (:1105); row count == LEN_R/NCOL forced (:1013) |
| com_A_L | absorbed AFTER beta_L (:1061); opened (:1101); row count == NPL·B forced (:1012) |
| lookup_L.bin: ev[] | round chain with the round-0 claim **imposed** = α+α² (:1064, never from disk); length == 4·logDL (:1030) |
| lookup_L.bin: A_f/S_f/m_f | absorbed (:1098); consumed by the verbatim-softmax terminal identity (:1090–1096) with B_f/T_f recomputed from the verifier's own range table; each anchored by a required IPA |
| hp_{bin,sum,mask,tbind}.bin: claim_H | **required == the protocol constant** (:1038–1041) AND the round-0 claim is imposed from the constant, not the file (:1110/:1135/:1165/:1304) — see §3 |
| hp_att.bin: claim_H (ev_mx) | absorbed (:1198); imposed as round-0 claim; anchored by IPA vs com_mx at u_a (:1221) |
| hp_c1/c2.bin: claim_H | absorbed (:1227/:1254); bound by the round chain to the verifier's OWN rebuilt-weight fold terminal (:1240–1247/:1267–1274); S_f2 anchored vs com_z / com_mx-suffix; both consumed by the DOM bracket identity (:1341) |
| hp_*.bin: ev[] | round-by-round chain; lengths == 4·logD forced (:1031–1035) |
| hp_*.bin: S_f2 | each consumed by exactly one required IPA |
| hp_*.bin: U_f2 | BIN: forced == S_f2 − 1 (:1124–1125, load-bearing); SUM/MASK/c1/c2/T-BIND: forced == 1 (:1146/:1176/:1239/:1266/:1315) |
| lvals.bin (v0[,v1]) | absorbed (:1286/:1290–1291) BEFORE the L-plane IPAs that anchor them; consumed by the bracket identity (:1336–1342) |
| ipa_*.bin (12/13/14) | L/R absorbed per round; round counts forced (n == 2^rounds == 2^|u_col|, fast_ipa_verify :151–153); a_final consumed by the final group equation (:170); each file consumed by exactly one required open |
| t* | **NOT from the obdir** — the verifier loads its own CLI registered-copy path (:1759), absorbs it in the preamble (:1052), range-checks every entry into [0,V) (:991–995), and gathers W_t from its own copy (:1317–1324) |
| gens, Q | CLI registered paths; gen sizes forced (gen.size == NCOL, genmx.size == B, :987) |

Layout params are re-checked inside verify, not only prove (:983–989: pow2
B/NCOL/LEN_R, NPL ∈ {1,2}, mode constraints, NCOL ≤ LEN_R ≤ DL with
LEN_R | DL); commitment row counts (:1011–1014) are double-pinned by
`fast_open_verify`'s `com.size == 2^(|u_pt|−logG)` cross-check. AL is built
by the verifier from (B, NCOL, MODE, V) only (:1043) — never committed,
never read. Missing/short files throw out of verify() (fail-closed, nonzero
exit — MINOR-8).

### 3. Constant-claim discipline — the evil=3 fix is complete at ALL FOUR instances

This was the found-and-fixed gap; verified both halves at every instance:

| instance | serialized claim_H required equal | round-0 claim imposed (not read) |
|---|---|---|
| BIN (0) | :1038 | `cur = F_ZERO` :1110 |
| SUM (1) | :1039 | `cur = F_ONE` :1135 |
| MASK (0) | :1040 | `cur = F_ZERO` :1165 |
| T-BIND (1) | :1041 (under `tb`) | `cur = F_ONE` :1304 |

The prover serializes the constants unconditionally (:655/:695/:726/:921 —
never witness-derived), so a malicious transcript cannot smuggle a nonzero
BIN claim: the imposition makes round 0 fail (evil=3 certifies, re-run
confirmed "BIN round 0 p(0)+p(1) != claim" at all four toy shapes), and the
equality check closes the residual hole where a file-honest-but-evil claim_H
would shift the rejection elsewhere. Data-dependent claims (ev_mx, c1, c2)
are absorbed per §2.6 and each is independently anchored (table above). One
coverage nit: the *equality-check* half is never negatively exercised
(MINOR-3).

### 4. The X1–X5 algebra — walked end to end

- **X1 (BIN).** Claim 0 = Σ eq(u_bin)·S·(S−𝟙) over logD vars; the verifier's
  my_eq accumulator runs over ALL logD rounds with `u_bin[logD−1−k]`
  (:1121) — correct for the front/back `k_fr_fold` orientation (binds the
  current MSB; eq factorizes per bit). **`U_f2 == S_f2 − 1` is required
  (:1124–1125) and is what binds the never-committed U to com_S** — the MLE
  of S−𝟙 at any point is S̃−1, so with S_f2 opened vs com_S (:1130) the
  terminal product is forced. Without it the prover could supply (S_f2,
  U_f2) with a fixed product and an arbitrary S; with it, the fractional
  selector of evil=3 (which passes SUM/MASK/ATT/DOM *by construction* — I
  re-derived: c+(1−c)=1, c·z1+(1−c)·z2=mx, Df from honest mx) dies at BIN
  round 0. Certified live.
- **X2 (SUM/MASK).** Weights `bcast(eq(u_s))⊙AL` and `bcast(eq(u_m))⊙(𝟙−AL)`
  are rebuilt by the verifier from its own AL and folded with its own
  k_fr_fold chain (:1147–1156/:1177–1189); `U_f2 == 1` forced (the U factor
  is the all-ones tensor, whose fold is identically 1). The claims equal the
  MLE of the row-sum vectors ⟨AL-row,S-row⟩ / ⟨(1−AL)-row,S-row⟩ at the fresh
  u_s/u_m, so constants 1/0 force those vectors ≡ 1/≡ 0 as tensors whp. With
  X1, row sums of {0,1} entries cannot wrap (≤ NCOL ≪ p), so this is genuine
  one-hot-on-allowed; in vpad, MASK's support is exactly the pad columns, so
  S's pads are forced to 0 (the §2.2 decision implemented as pinned; evil=5
  certifies). The V == NCOL degenerate case (weight ≡ 0, claim 0 vacuous) is
  by design — no pads exist to protect — and is toy case d.
- **X3 (ATT).** ev_mx is absorbed at u_a (:1198) and anchored THREE ways: the
  sumcheck round chain from it, the eq_acc terminal (:1213 — row-bits-only
  accumulator, `k < logB` :1210, valid because the weight is a pure
  column-broadcast: rounds 0..logB−1 fold the row MSBs contributing
  eq(u_a[logB−1−k], w), the column rounds fold a per-row constant
  contributing factor 1 — the rmsnorm-validated shortcut), and **all three
  openings**: S at pt_a vs com_S (:1217), z at pt_a vs com_z (:1219, the
  U_f2 slot carries z̃(pt_a)), ev_mx vs com_mx at u_a with gen_mx, u_row
  empty (:1221). Forces ⟨S-row, z-row⟩ = mx rowwise whp; with X1+X2, mx[i]
  is an attained allowed value.
- **X4 (DOM + LIMB).** The bracket: c1 and c2 are absorbed after u_r but are
  not free — each is forced by its round chain to the verifier's own
  eq(u_r)⊙AL weight fold times an anchored terminal (z̃(pt_c1) vs com_z
  :1251; the broadcast terminal vs com_mx :1281). **The never-committed
  mx_bcast is pinned to com_mx at the row-bit suffix** pt_c2[logC..logD)
  (:1277–1283): a row-broadcast tensor's MLE at (u_col, u_row) is
  m̃x(u_row)·Σ_j eq(u_col, j) = m̃x(u_row), so the suffix opening is exactly
  the broadcast binding (softmax V2 precedent; evil=7 certifies the IPA is
  the catcher). The plane openings evaluate com_L at (u_r‖0) and (u_r‖1) —
  Boolean top-bit evaluation = plane-slice MLE (plane bit = flat bit logD,
  matching L's layout plane·D + i·NCOL + j); v0/v1 are absorbed before the
  IPAs that anchor them. The verifier identity `c2 − c1 == v0 [+ LEN_R·v1]`
  (:1336–1342) equates the MLE of AL⊙(mx_bcast − z) (linearity of the two
  same-weight sums) with the MLE of the limb reconstruction at the same
  random u_r ⟹ tensor equality whp; LIMB (the verbatim-softmax logUp, with
  m forced by the post-commitment β and the table rebuilt verifier-side)
  then gives componentwise Df ∈ [0, LEN_R^NPL). Note the code uses LEN_R
  where §2.3 wrote 2^20 — identical at the pinned vpad shape (LEN_R = 2^20)
  and the correct base-LEN_R generalization the toy shapes need. Masked/pad
  entries: lo + LEN_R·hi = 0 with both limbs in [0, LEN_R) forces both
  exactly 0 (no wrap possible below LEN_R² ≪ p).
- **Field-wrap reliance (§2.1) — nothing beyond the documented assumption.**
  X4 reads Df's field values as small integers; a wrap needs |z| or |mx|
  within ~LEN_R^NPL of p. mx is an attained z value (X1–X3), so it inherits
  z's range; z's range is the documented chain-composition reliance (com_z
  byte-chained to deterministic upstream outputs, edges RM1/L1). I looked
  for *additional* silent reliances and found none: S is binary (X1), row
  sums can't wrap (≤ NCOL terms), m_L/A_L are forced by logUp at
  post-commitment β, the limb reconstruction can't wrap below LEN_R², and
  every other quantity is an MLE evaluation of a committed tensor. The vpad
  |z| < 2^25 throw (:413–414) and the causal Df-overflow throw (:484) are
  completeness guards, correctly NOT relied on for soundness.
- **X5 (T-BIND).** W_t is gathered from the verifier's OWN t* (:1317–1324)
  with every t*[i] range-forced into [0, V) (:991–995) — the gather can
  never place weight on a pad column. Claim 1 = MLE at fresh u_t of
  g[i] = S[i, t*[i]] ⟹ g ≡ 1 whp ⟹ with X1–X4, every t*[i] is a row
  maximizer. evil=8 certifies; the t*-mode asymmetries (prove-with/verify-
  without and vice versa) diverge at the TSTAR preamble absorb and reject.

### 5. Selector tie channel — matches §2.4 exactly

The honest prover's argmax scan (:426–436) uses strictly-greater on an
ascending j loop over the allowed set — the canonical LOWEST-index witness
(np.argmax convention); `host_tstar` (:1405–1417) is the same scan. All other
witness tensors are deterministic functions of (z, mx, S, AL): the limbs are
the unique base-LEN_R digits of Df, m_L is the multiplicity vector forced by
logUp at random β, A_L = 1/(L+β) elementwise. So the only protocol freedom is
which tied argmax position carries the 1 (and a tied t*), visible only in
com_S/its openings — exactly the §2.4 accounting. The orchestrator-side tie
measurement hook (§2.4 ii–iii) is correctly out of this driver's scope
(report concern iii).

### 6. k_pp_expand, fast_me_weights, fast_s_vector — convention and divergence analysis

- Montgomery: walked in §CRITICAL above; correct.
- `fast_s_vector_dev` (:118–126): pairs (xis[R−1−t], xs[R−1−t]) at bit t ⟹
  s_i = Π_r (bit_{R−1−r}(i) ? xs[r] : xis[r]) — exactly the header
  `ipa_verify` MSB-first product (vrf_common.cuh:360–364).
- `fast_ipa_verify`'s b_f = ⟨b, s⟩ replaces the header's incremental b-fold:
  re-derived — each round folds b′ = xi·lo + x·hi, so the fully folded b[0]
  is Σ_i b_i·s_i with the SAME s as the g-fold; all ops are exact canonical
  mod-p, so summation order is immaterial. Same round-count guards as the
  header plus the u_b-size check.
- **Cross-checks: present, both scales, every honest prove.**
  `crosscheck_fast_helpers` (:186–216) runs in EVERY evil==0 prove at the
  ATT block (:757–759) — i.e. in all four toy cases AND both real-scale
  selftests (u_a is logB = 10 bits ⟹ gen-1024 scale, as §2.8/§2.9 pin) —
  element-exact against the slow header `me_weights` and the slow MSB-first
  s-product, throwing STOP on mismatch.
- **Could fast/slow divergence slip through prove-only paths (gen-32768)?**
  No. (a) The prover's IPA b-vector *defines* the proven evaluation point;
  if `fast_me_weights` diverged at 15 bits, the honest proof would claim
  t̃(wrong point) while the verifier requires eval == the sumcheck terminal —
  which the verifier independently pins through the round chain + its own
  weight folds, and which the prover's evil==0 convention blocks
  (`multi_dim_me` with logC = 15 column vars, e.g. :627–631, :683–688) tie
  to the true point in every real-vpad prove. Honest real-vpad ACCEPT (re-run
  here) therefore functionally certifies fast_me_weights at 15 bits. (b)
  `fast_s_vector` is verify-side only and the prover's g-fold is the
  incremental true path (`k_g1_fold2`), so any divergence REJECTS honest
  proofs — fail-closed, never accept-forged. Residual risk ≈ a divergence
  that preserves the specific inner products — negligible.

### 7. Layout math — both modes, per §2.5

- n1 = logDL − log LEN_R computed identically in prove (:394) and verify
  (:981): **causal real n1 = 0** (pure phase-2, exercised at real scale),
  **vpad real n1 = 6**; toys 1/3/2/1 (the DESIGN's §2.9 table annotations
  "n1=0"/"n1=1" for cases a/b contradict its own formula — MINOR-2; the
  implementation and report are right).
- Commitment row counts forced (:1011–1014): com_z/com_S = B, com_L/com_A_L
  = NPL·B, com_m_L = LEN_R/NCOL, com_mx = 1 — i.e. 1024/1024/1024/1024/1024/1
  causal and 1024/1024/2048/2048/32/1 vpad, matching §2.5; double-pinned by
  every `fast_open_verify`'s shape cross-check. LEN_R/NCOL is exact (pow2,
  NCOL ≤ LEN_R guarded both sides).
- NPL handling: NPL=1 opens L once at u_r with v0 only and the identity drops
  the LEN_R·v1 term; NPL=2 appends the Boolean plane bit as flat bit logD on
  both the prover's `partial_me` row vars and the verifier's `fold_chain`
  point — consistent with L's plane·D + i·NCOL + j layout (prover's evil==0
  plane-slice check :895–899 pins it at runtime).
- vpad zero-pads z columns V→NCOL (:408–416) before commitment, with the
  |z| < 2^25 envelope throw; the chain edge L1 byte-identity premise (the
  lm_head rescale commits the identically zero-padded grid) is an
  orchestrator-time check, correctly out of scope here.
- mx chain file is the unpadded B int32 vector (:503–509), "-" to skip; all
  honest-prover throws of §2.1 are present (layout_guards :307–324, t*
  checks :396–402, input dims :409/:418, Df guards :483–484, short reads in
  load_i32).

### 8. Selftest honesty + the byte-identity assertion

- **Evil modes hit exactly the named check** — the selftest requires the
  FIRST reject reason to contain the expected string, so a wrong-check
  rejection or prover crash FAILS the test. The verify order makes most
  "everything else passes" claims *demonstrated*, not assumed: every check
  ordered before the catcher runs and passes on the evil witness; notably
  **evil=2 fires at the very LAST check** (DOM bracket identity), so it
  demonstrates LIMB, BIN, SUM, MASK, ATT, every IPA and T-BIND all pass on a
  too-low-mx witness. For checks ordered after a catcher I verified the
  constructions analytically (evil=3 §4 above; evil=4's added position has
  z = 0 so ATT is unchanged; evil=5's pad has AL = 0 and z = 0 so
  SUM/ATT/DOM are unchanged; evil=6 preserves lo + LEN_R·hi so the bracket
  passes and m_L-from-honest-limbs isolates the lookup). Evil coverage spans
  all four toy shapes (24 evil runs total); strict=false is scoped to
  exactly the targeted recursion per mode; the evil=3 + t* interaction
  (T-BIND disabled, certified separately by evil=8) is handled as the
  report describes (:1494–1496).
- **Byte tampers cover every file verify() reads**: the `proof_files`
  enumeration (27/28/30) equals the verify read set exactly (cross-checked
  §2 table); offsets land in absorbed or checked bytes (hp@36 = ev[0],
  lookup@36 = round-0 eval, ipa@−32 = a_final, com@24 = first point,
  lvals@4 = v0, dims@0); each case restores and re-verifies ACCEPT.
- **The memory-fix byte-identity assertion: basis verified AND reproduced.**
  Basis walked in source: (i) `lean_hadamard` (:230–275) emits the identical
  field elements as the header `fs_hadamard` — same `k_hp3_step` on the same
  data, same `FrTensor::sum`, same labels/challenge schedule, same
  `lagrange4(·, inv 6)` chaining, same terminal reads; the only delta is
  allocation lifetime, which cannot change values; (ii)
  `fast_me_weights_dev` ≡ `build_eq_tensor` (same doubling recurrence, mont
  linearity); (iii) `commit_chunked` is bit-identical per row because
  `Commitment::commit` = elementwise mul + `rowwise_sum`, whose per-row
  reduction tree depends only on ncol = G (walked g1-tensor.cu) — row
  outputs are functions of that row alone; (iv) the S/L re-upload paths
  restore exact committed bytes (S device bytes stashed post-edit). Then
  reproduced empirically: re-ran `diff -r` on the retained
  `/tmp/biden_old_*` vs `/tmp/biden_new_*` (byte-identical), AND regenerated
  both toy proof sets with the CURRENT binary from the preserved
  inputs/seeds (`/tmp/rowmax-audit/{causal,vpad}`) — **byte-identical to the
  PRE-FIX directories** (all 27 + 30 files incl. hp_tbind/ipa_S_tbind/
  ipa_L_p1, plus both mx chain files), and both regenerated dirs verify
  ACCEPT. One caveat noted as MINOR-6: the toy shapes take `commit_chunked`'s
  fall-through branch, so the *chunked* branch's byte-identity rests on the
  source walk + the real-vpad ACCEPT (which pins every row commitment
  through the IPAs at group-equation strength) until the first chain-edge
  byte-compare.

### 9. New kernels — exactly one, Fr-only

Covered under CRITICAL. `k_pp_expand` is driver-local; the shared headers are
unedited (`zkllm-src` persistence confirms the audited file is the only
addition); all other kernels invoked (`k_hp3_step`, `k_fr_fold`,
`k_bcast_rows`, `k_fr_emul`, `k_bump`, `tlookup_inv_kernel`) are
pre-existing, selftest-probed header/upstream kernels.

---

## MINOR findings

**MINOR-1. File-count headline 26/29 is wrong in both design and report §1.**
§2.7's own file LIST enumerates 27 (causal) / 30 (vpad+t*), which is what the
implementation produces and what the report's own §4 diff says ("all 27
files" / "all 30 files"). Cosmetic counting slip propagated from the design;
fix the numbers on next touch. No soundness impact.

**MINOR-2. Design §2.9 toy n1 annotations are internally inconsistent**
(case a says "n1=0", case b "n1=1"; the §2.5 formula gives 1 and 3). The
implementation computes n1 correctly and the report documents the actual
values. Consequence: no toy-scale pure-phase-2 (n1=0) case exists — but real
causal IS n1=0 and is selftested, so the path is covered. Design-doc errata
only.

**MINOR-3. The serialized-claim_H equality check is never negatively
exercised.** The hp byte tampers hit ev[0] (offset 36), not claim_H (offsets
0–31), and no evil mode writes a non-constant claim_H (the prover hardcodes
the constants). The load-bearing half — imposing the constant at round 0 —
is certified by evil=3; the equality half (:1038–1041) is belt-and-suspenders
and verified by inspection. Cheap fix: add an hp_bin.bin@0 tamper expecting
"BIN claim_H != protocol constant 0".

**MINOR-4. lvals.bin tamper covers v0 only** (offset 4); v1 (NPL=2) is never
tampered. v1 is absorbed, anchored by ipa_L_p1, and consumed by the bracket
identity — same three anchors as v0 — so coverage-only.

**MINOR-5. Guard selftest omits three §2.1 throws** (B not pow2, NCOL not
pow2, NPL ∉ {1,2}); the guards exist in `layout_guards` (:309–312) and the
verifier re-checks the same predicates in RJ form (:983–989). Coverage nit.

**MINOR-6. `commit_chunked`'s chunked branch has no byte-level test against
the unchunked upstream commit.** Toy shapes (rows ≤ CHUNK_ROWS) take the
fall-through `gen.commit(t)` path, so the §8 byte-identity diff doesn't
exercise chunking; real vpad exercises it functionally (ACCEPT ⟹ every row
commitment satisfies its IPA group equation ⟹ the rows are genuine
commitments of the committed tensor). The per-row bit-identity argument is
sound (row-independent reduction, walked), and a hypothetical
representation-level difference would be a loud completeness failure at the
first RM1/RM2/L1 byte-edge, never a soundness gap. Note for the wiring
session: the first edge comparison is the live test.

**MINOR-7. Scoped assumption (inherited, by design): standalone ACCEPT binds
only internally.** com_z/com_mx are pinned into the chain by the orchestrator
byte-edges (RM1/RM2 causal, L1 vpad) and t* by the public.json hash over
tstar.i32.bin (§3.4) — these MUST be enforced at wiring (softmax MINOR-5 /
rope MINOR-6 precedent). The §2.1 field-wrap note is part of the same
reliance and is correctly documented in the file header.

**MINOR-8. Missing/short proof files throw rather than printing REJECT**
(`open_or_die`/`read_pod_vec`/G1 ctor). Fail-closed — nonzero exit, never a
false ACCEPT (the selftest's tamper harness counts throws as rejects, the
documented MINOR-4 softmax posture). Orchestrator must treat any nonzero
exit as reject.

**MINOR-9. Report §2 byte-tamper "(n1=1)" label for the causal 8×8 toy** and
the §3 vpad GPU peak (10.99 vs my 10.48 GiB) are both explained in the report
itself (correct formula values; fragmentation delta) — re-runs match. No
action.

---

## What I would do before trusting it in the gate (summary)

**Accept the file.** The verifier is independent (every disk value anchored —
§2 table; weights, AL, the range table and W_t all rebuilt verifier-side;
U_f2 forced at all six sites with BIN's S_f2−1 form), the FS schedules are
absorb-for-absorb identical and match §2.6 label-for-label including the
TSTAR preamble, the constant-claim discipline is complete at all four
instances (both halves, with evil=3 certifying the load-bearing half), the
X1–X5 algebra is exactly the §2.1 statement with the broadcast-suffix and
plane-bit bindings in place, the tie channel is the canonical lowest-index
witness with zero other freedom, the single new kernel is Fr-only with a
walked Montgomery convention and runtime-crosschecked consumers, both modes'
layouts match §2.5 with row counts forced, the selftest is honest (re-run:
160/0) and its byte-tamper set covers the entire verify read surface, and
the memory-fix byte-identity claim was independently reproduced from the
preserved pre-fix artifacts with the current binary. Optional, cheap, next
touch: fix the 26/29 counts and the §2.9 n1 annotations (MINOR-1/2), add the
claim_H@0 tamper (MINOR-3), and carry the MINOR-6/7 sentences into the
wiring notes.
