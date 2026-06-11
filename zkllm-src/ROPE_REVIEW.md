# ROPE_REVIEW — independent soundness audit of zkob_rope.cu, zkob_headslice.cu, zkob_headmerge.cu

Reviewer: second engineer (independent review). Date: 2026-06-11.
Scope: `/root/zkllm/zkob_rope.cu` (606 lines), `/root/zkllm/zkob_headslice.cu`
(545 lines), `/root/zkllm/zkob_headmerge.cu` (539 lines) against the normative
`ROPE_ATTENTION_DESIGN.md` (DESIGN FINAL 2026-06-10), the pinned conventions in
`PHASE0_NOTES.md` (§7–§15), the trusted shared machinery (`vrf_common.cuh`,
`zkob_lookup.cuh`, `fs_transcript.hpp` — audited for USAGE, not edited), and the
bar set by `RMSNORM_REVIEW.md` / `SOFTMAX_REVIEW.md`.

Selftests independently re-run (logs under `/tmp/rope-audit/`), all exit 0:

| driver | result | checks | real-scale prove | verify | proof+coms |
|---|---|--:|--:|--:|--:|
| zkob_rope      | ALL PASS | 50  | 1.99 s  | 1.34 s  | 309,040 B (309 KB) |
| zkob_headslice | ALL PASS | 105 | 29.00 s | 24.37 s | 4,275,660 B (4.28 MB) |
| zkob_headmerge | ALL PASS | 58  | 3.48 s  | 2.19 s  | 1,966,884 B (1.97 MB) |

**The selftests were NOT taken on trust**: every `verify()` check, every FS
absorb, the π/π⁻¹ and rotate_half index algebra, and the relevant upstream
conversions (`int_to_scalar`/`long_to_scalar` negative handling,
`Fr_elementwise_mul` vs `k_fr_emul` plain-product equivalence,
`Commitment::commit` row partitioning, `FrTensor::pad` zero-extension) were
walked in source. The registered rope tables were independently regenerated
from the §2.1 formula and match `rope-cos-table.bin`/`rope-sin-table.bin`
**bit-exactly** (cos[0,·] = 65536, max |entry| = 65536, int32 required ✓).
Sources persisted to `zkllm-src/` are byte-identical to the audited files.

## VERDICT

- **zkob_rope: SOUND**
- **zkob_headslice: SOUND**
- **zkob_headmerge: SOUND**
- **Overall: SOUND** — no critical or major soundness gap in any driver.

Each verifier enforces exactly the relation the design claims (R-ROPE bound to
com_T and the registered tables with the rotate_half permutation load-bearing;
every per-head slice pinned to the full-tensor commitment by paired openings at
one challenge; O2 forced to be the π-gathered head concat with exactly-zero
padding), every prover-supplied value each `verify()` consumes is anchored, and
the Fiat–Shamir schedules of `prove()` and `verify()` are absorb-for-absorb
identical and match §5 label-for-label. The minor notes below are
coverage/reporting observations; none lets a cheating prover pass.

**The ROPE_IMPL_REPORT open question is RESOLVED** (see MINOR-1): the report's
rope and headmerge rows have their measurement figures swapped. **rope = 309 KB**
(9 files; 2 × 144 KB gen-1024 commitments dominate — matches §7.2's ≈295 KB)
and **headmerge = 1.97 MB** (40 files; 13 × 144 KB commitments = 1.87 MB —
matches §7.2's ≈1.9 MB). Confirmed both from the retained acceptance logs
(`rope_selftest_accept.log:119` = 309,040 B; `headmerge_selftest_accept.log:134`
= 1,966,884 B) and from my independent re-runs, and re-derived from §7.2.

---

## CRITICAL findings

**None.** Both automatic-CRITICAL categories are clean in all three drivers:

- **New CUDA kernels: ZERO** (G1 *and* Fr). `grep -cE 'KERNEL|__global__'` = 0
  for all three files. All G1 work goes through the proven 1-thread helpers
  (`h_mul`/`h_add`/`g1_eq`) and the pinned `fold_chain`/`dev_msm`/FS-IPA paths;
  all Fr work uses pre-existing, selftest-probed header kernels (`k_fr_emul`,
  `k_fr_fold` via the drivers' host-loop `fold_public`, `k_eq_expand` inside
  `build_eq_tensor`, `k_hp3_step` inside `fs_hadamard`). No Montgomery-convention
  question arises. The −dlto rule is followed in the strongest form, exactly as
  §0 promised.
- **Verifier accepting an unanchored prover value: none found** (full
  enumeration per driver below).

## MAJOR findings

**None.** Each checklist category was walked; why each is clean follows.

### 1. Fiat–Shamir ordering — clean, all three (absorb-for-absorb identical)

Compared prove/verify event by event; each matches §5.1/5.2/5.3 label-for-label:

```
rope:   absorb B,C,HD,SCALE_R; com_T; com_Y64 → u(logD)
        absorb ev; absorb c1; [h1: logD × (hp0..hp3 → w)]; absorb S_f2,U_f2;
        IPA(T)@pt1; absorb c2; [h2 rounds]; absorb S_f2,U_f2;
        IPA(T)@pt2'; IPA(Y64)@u
slice:  absorb B,C,HD; com_Q,com_K,com_V; per hh: com_Qh,com_KhT,com_Vh
        → v(logHD+logB); per hh: absorb eQ{hh}; IPA(Qh); IPA(Q);
        absorb eK{hh}; IPA(KhT); IPA(K); absorb eV{hh}; IPA(Vh); IPA(V)
merge:  absorb B,C,HD; com_O2; per hh: com_O{hh} → u(logB+logCp); absorb ev;
        per hh: absorb c{hh}; [logB+logHD rounds]; absorb S_f2,U_f2;
        IPA(out_h)@reverse(ws); IPA(O2)@u
```

Every challenge is squeezed only after the message it binds: the grid challenge
u/v only after ALL commitments (so a prover cannot choose a slice/permutation/
output after seeing the evaluation point); every sumcheck round challenge after
that round's four absorbed evals; every IPA round challenge inside `ipa_prove`/
`ipa_verify` after that round's L,R; every opened eval (ev, c1, c2, eQ/eK/eV,
c{hh}, S_f2, U_f2) absorbed before the `open_verify` that discharges it. The
verifier-only checks (U_f2 == 1, weight terminals, c1+c2 == ev, Σc == ev,
paired-eval equality) absorb nothing, so check placement cannot diverge the
transcript. The reused "hp0".."hp3" labels are positionally disambiguated by
the interleaved c{hh}/S_f2/U_f2 absorbs (rmsnorm/softmax precedent). No absorb
present on one side and missing on the other; no order difference.

### 2. Verifier independence — every disk value anchored

**zkob_rope verify() reads 9 files** (RMSNORM table format):

| value (file) | anchor |
|---|---|
| dims.bin (B,C,HD,SCALE_R) | checked == CLI args (:279); same scalars absorbed |
| com_T | absorbed; opened TWICE — pt1 (:335, eval h1 S_f2) and the verifier-computed flipped point pt2′ (:366, eval h2 S_f2); chained upstream by edge A3 (orchestrator) |
| com_Y64 | absorbed; opened at u (:370, eval ev); chains to the rope rescale (edge A4) |
| ev.bin | absorbed (:305); anchored by IPA vs com_Y64 AND consumed by c1+c2==ev (:372) |
| hp1/hp2.bin: claim_H (c1/c2) | absorbed; bound by the round chain (p(0)+p(1)==claim per round, :316/:344) whose terminal is the verifier's own `fold_public(eq(u)⊙W_i)` × opened S_f2 × forced U_f2 (:330/:358) |
| hp1/hp2.bin: ev[] | round-by-round chain; length pinned == 4·logD (:287) |
| hp1/hp2.bin: S_f2 | IPA openings vs com_T (:335/:366) |
| hp1/hp2.bin: U_f2 | forced == 1 (:323/:351) — load-bearing |
| ipa_T1/T2/Y.bin | each consumed by exactly one required `open_verify` |
| cos/sin tables | NOT from the obdir — CLI registered-copy paths; W1/W2 rebuilt by the verifier itself (:296) with the §2.2 formula incl. the [e<C] rule and σ(e) |

**zkob_headslice verify() reads 5 + 9·NH files** (113 at NH=12):

| value (file) | anchor |
|---|---|
| dims.bin (B,C,HD) | checked == CLI args (:264); absorbed |
| com_Q/K/V | absorbed; each opened NH times at the head-selector points (v_d ‖ bits(h) ‖ v_t); chained upstream by edges A3v/A5 (orchestrator) |
| com_Qh/KhT/Vh{hh} | absorbed; each opened once; row counts pinned (B/HD/B, :276); these ARE the downstream fc operand commitments (edges A6/A7/A12) |
| evals.bin (3·NH evals) | exact-size-checked (:282); each eval absorbed, then anchored by TWO required IPAs — slice-side AND full-side — **against the same value** (:311–330), which is the entire S-Z binding |
| ipa_{Q,K,V}{h,f}{hh}.bin (6·NH) | each consumed by exactly one required `open_verify` |

**zkob_headmerge verify() reads 4 + 3·NH files** (40 at NH=12):

| value (file) | anchor |
|---|---|
| dims.bin (B,C,HD) | checked == CLI args (:257); absorbed |
| com_O2 | absorbed; opened at u (:318, eval ev); ≡ com_attn_out (edge A15, orchestrator) |
| com_O{hh} | absorbed; opened at the head terminal (:314, eval S_f2); row counts pinned == B (:265); ≡ values-rescale com_Xr (edge A14) |
| ev.bin | absorbed; anchored by IPA vs com_O2 AND consumed by Σc==ev (:320) |
| hp{hh}.bin: claim_H (c_h) | absorbed; bound by the round chain (lengths pinned 4·(logB+logHD), :267) whose terminal is the verifier's OWN π⁻¹-gathered weight fold (:306–310) × opened S_f2 × forced U_f2==1 (:304) |
| ipa_O{hh}/ipa_O2.bin | each consumed by exactly one required `open_verify` |

Layout params are re-checked inside every verify (not only prove): pow2 B/HD,
HD | C, HD | C_pad, NH ≥ 2, gen sizes (rope :272–274, slice :254–258, merge
:249–252). No early-accept path in any driver; ACCEPT is the last line after
all openings and identities. Missing/short proof files throw out of verify()
(fail-closed, nonzero exit — softmax MINOR-4 precedent; orchestrator must
treat any nonzero exit as reject).

### 3. Openings — all present, required, right points/commitments/gens

- **rope: 3 IPAs.** The flagged one is correct: **pt2′ is computed by the
  VERIFIER** (zkob_rope.cu:363–365: `pt2p[fb] = h_scalar(F_ONE, pt2[fb], 1)`,
  fb = log2(HD)−1 = 5 at real scale = the §1.4 coordinate, bit 0 at toy HD=2 —
  tested), and `open_verify(com_T, …, pt2p, hp2.S_f2, ipa_T2)` opens the SAME
  com_T expecting the h2 terminal S_f2. That is exactly the rotate_half MLE
  identity T̃x(pt2) = T̃(pt2[0..fb), 1−pt2[fb], pt2[fb+1..)) — Tx is never
  committed. If the prover runs h2 unpermuted (evil=2), S_f2 = T̃(pt2) ≠
  T̃(pt2′) whp and the opening rejects — confirmed; with it passing, un-rotated
  RoPE would ship, so this opening is THE load-bearing line and it is in place.
- **slice: 6 per head × NH = 72.** Both members of each pair verify against
  the **same absorbed eval** (the single `eQ`/`eK`/`eV` local is passed to both
  `open_verify` calls). bits(h) are F_ZERO/F_ONE field constants at point
  positions logHD..logCp−1 (6–9 at real scale) — they live entirely in the
  IPA's `u_col`/`me_weights` half (logCp = 10 > 9), the softmax-L-plane-bit
  precedent. The KhT swap is right: KhT flat = d·B + t ⟹ u_pt = (v_t ‖ v_d)
  with gen = the B-sized set (`gen_for_B`, unambiguous: gen_big.size = C_pad =
  HD·NH > HD = gen_small.size, both can't equal B); `open_verify`'s
  com.size == 2^(|u_pt|−logG) cross-check plus the explicit row-count checks
  pin all three shapes per head.
- **merge: 13 IPAs** (12 × com_O{hh} at each head's terminal + com_O2 at u),
  matching the 13 ipa files, all byte-tampered. Every call is
  `if (!open_verify(…)) RJ(…)` in all three drivers — none ignored.
- Commitment/gen interop with downstream (walked against `Commitment::commit`'s
  row partitioning, com = t.size/gen.size rows): com_Qh/com_Vh = 1024 rows
  gen64 ≡ scores-fc com_X / values-fc com_W; com_KhT = 64 rows gen1024 ≡
  scores-fc com_W (= `gen_out.commit(W.pad({64,1024}))`); com_O{hh} = 1024 rows
  gen64 ≡ values-rescale com_Xr; com_T/com_Y64/com_O2 = 1024 rows gen1024 ≡
  their §7.4 partners. Deterministic commit ⟹ byte-identity for equal data.

### 4. The algebra — walked end to end

- **rope (c1 + c2 == ev with rebuilt W1/W2, signs).** The verifier rebuilds
  W1/W2 from its own table copies via the same `build_weights` formula as §2.2:
  `W1 = [e<C]·cos[t, e mod HD]`, `W2 = [e<C]·((e & HD/2) ? +1 : −1)·sin[…]` —
  checked against rotate_half semantics: d < 32 ⟹ q′ = q·cos − q[d+32]·sin
  (e&32 = 0 ⟹ σ = −1 ✓), d ≥ 32 ⟹ +q[d−32]·sin ✓. **Mod-p negatives are
  consistent end to end**: negative W2 ints map through `int_to_scalar` to
  p−|v| (fr-tensor.cu:707–710) on BOTH sides; the witness Y64 is exact signed
  host int64 mapped via `long_to_scalar` (same negation rule, :696–700); the
  integer relation therefore transports homomorphically to F_p, so
  Schwartz–Zippel at u (squeezed post-commitment, 20 vars) forces
  Y64 = T⊙W1 + Tx⊙W2 as committed tensors. Prover-side `FrTensor::operator*`
  (`Fr_elementwise_mul` = mont(mul(a,b)) = plain a·b) and verifier-side
  `k_fr_emul` (mul(mont(a),b) = plain a·b) compute the identical plain product,
  so the weight folds agree — also pinned at runtime by the evil==0
  fold-vs-`multi_dim_me` blocks and the honest weight-terminal passes. Each
  hadamard's chain is anchored: c_i absorbed → logD round checks → terminal
  cur == W_f·S_f2·U_f2 with W_f the verifier's own `fold_public(eq(u)⊙W_i, ws_i)`
  (same k_fr_fold orientation as the prover's in-sumcheck E-fold), S_f2 opened
  vs com_T, **U_f2 == 1 required** (U never committed — without this check the
  sumcheck is freely forgeable; evil=4 confirms it fires). Finally ev (opened
  vs com_Y64 at u) must equal c1+c2 — evil=1 confirms this is the sole and
  sufficient defense for the output tensor. A useful extra the design only
  implies: since W1 = W2 = 0 on padding columns and ev sums the whole padded
  grid, **the identity also forces Y64's padding to exact zero**.
- **headslice (S-Z binding).** Would a slice differing from the head block
  actually fail? Yes: the IPA binds eQ to the committed slice's MLE at
  (v_d‖v_t) and (separately, same absorbed value) to the full tensor's MLE at
  (v_d‖bits(h)‖v_t) = the head-block MLE at (v_d‖v_t) (Boolean eq-factors
  select columns with head bits = h exactly — checked against `me_weights`'
  LSB-first pairing, and cross-checked prover-side against `multi_dim_me` in
  the evil==0 block :219–225, which runs in every honest selftest case). If
  slice ≠ block as integer tensors (< 2^31 ≪ p ⟹ as field tensors), the two
  multilinears differ, and they collide at the post-commitment random v w.p.
  ≤ (logHD+logB)/|F| ≈ 2^−251. Setting eQ to either side's true value fails
  the other side's IPA; there is no third option. The three evil modes confirm
  the full-tensor side catches wrong-head, off-by-one-column, and untransposed
  slices, and the honest pass + evil=3 jointly pin the (v_t‖v_d) swap (a
  verifier-side swap bug would fail honest ACCEPT).
- **headmerge (Σ c_h == ev with the π⁻¹ gather).** `gather_Wm`
  (zkob_headmerge.cu:85–96) implements **exactly** the §1.3/§4.6 formula:
  e = HD·h+d; **m = e·B + t; i = m div C; j = m mod C**; weight =
  E_u[i·C_pad + j]. I re-derived π from the pipeline source quoted in §1.3
  (O2 = reshape(flatten(Mᵀ), (B,C)) ⟹ O2[i,j] = M[m mod B, m div B], m = i·C+j)
  and inverted it: m = e·B+t, i = m div C, j = m mod C — the code matches the
  derivation and the design symbol-for-symbol; no index swap. The forward
  gather in prove (:144–148) is the independent inverse-direction loop, and
  the selftest's chain-file check (:382–397) is a third independently-written
  instance — a one-sided swap anywhere makes the honest prover throw at the
  evil==0 `csum == ev` guard (:230) or fail the chain-file check. **Which π
  errors would the toy cases miss?** I enumerated the plausible swaps: i↔j,
  div↔mod, m = t·C+e, C vs C_pad in div/mod — every one differs from π at the
  B≠C, C≠C_pad toy shapes (8,6,2) and (16,12,4) and is caught (the B==C case b
  alone would indeed miss several of these — but b is not the only case). The
  single survivor class is a **B↔C_pad symbol confusion**, invisible because
  every headmerge case INCLUDING real scale has B == C_pad — see MINOR-2; the
  code uses the correct symbols, and at the only deployed shape (1024 = 1024)
  the confusion would be semantically vacuous anyway. The binding itself: each
  c_h is forced by its sumcheck (verifier-rebuilt Wm_h fold, U_f2 == 1
  required, S_f2 opened vs com_O{hh}) to Σ Wm_h⊙out_h over the committed head
  tensor; (h,t,d) ↦ (i,j) is a bijection of the real index sets hitting no
  padded E_u entry, so Σ_h c_h = MLE-at-u of "π(concat), 0 on padding"; ev =
  Õ2(u) over the full padded grid; equality at the post-commitment u forces
  O2 = π(concat) AND O2's padding ≡ 0, in one check (evil=1 and evil=2 each
  confirm, hitting the same named check for the two distinct properties).

### 5. Public-weight folds and U_f2 == 1 — present and load-bearing everywhere the design says

14 weight-terminal sites total: rope h1/h2 (W1/W2 rebuilt from the verifier's
own registered-copy CLI paths — never from the obdir) and merge × 12 (Wm_h
gathered by the verifier from its own `build_eq_tensor(u)`; the prover's
gather is never read). All 14 require `U_f2 == 1` before the terminal product
check. Evil modes confirm both check types fire (rope evil=3/4, merge
evil=3/4 — gather-with-h+1 is caught by the verifier's own-gather terminal,
exactly the §8.2 intent). headslice has no weights — by design (pure paired
openings); nothing to rebuild, and nothing prover-supplied stands in for it.

### 6. Padding — clean

- rope: the `[e < C]` weight rule is in the single shared `build_weights` used
  by both sides (the verifier calls it on its own table copies); flip stays
  inside head blocks and inside padding (HD | C and HD | C_pad guarded in BOTH
  prove and verify), so Tx's padding is zero and W's padding is zero — belt
  and suspenders as §2.2 specifies. Y64 padding additionally forced zero by
  the identity (above). The Y64 chain file is pad-stripped (:159–165).
- headslice: slice tensors are exact-pow2 shapes (B×HD, HD×B) — no padding
  exists to go wrong; full tensors are zero-padded via `FrTensor::pad`
  (default pad_val 0, verified) and the padded head columns h ∈ [NH, 2^nhb)
  are never sliced. Junk planted in com_Q/K/V padding is not detectable here
  but breaks the A3v/A5 byte-equality — the designed chain defense.
- headmerge: O2 padding forced to exact zero by the sum identity (evil=2);
  B = B_pad always (pow2 guard), so no pad rows exist; the O2 chain file is
  pad-stripped (:158–164).

### 7. Selftest honesty — clean

- All `fs_hadamard` calls run **strict=true** in all modes; every evil witness
  is internally consistent with its planted corruption, so no prover throw can
  mask a verifier check (verified by the re-runs: every evil mode is rejected
  by exactly the named check, and the selftest requires the reason **string**
  to match, so a wrong-check rejection or a prover crash FAILS the test).
  Completeness guards are disabled exactly for the mode under test and no
  wider (rope's T̃(pt2′) guard skipped only for evil=2; the evil==0-only
  convention and csum checks).
- evil==0 convention checks present and run in every honest prove: rope h1
  terminal + U (:217–223), rope flip identity (:243–250 — run for ALL evil≠2,
  stronger than required), slice both-layout MLE equality for every head
  (:219–225), merge per-head terminal + U (:221–227) plus the §4.6 sum
  identity itself (:230). The rope Y64 / merge O2 chain files are re-checked
  against independently-written spec recomputations (rope :429–448, merge
  :382–397 via π⁻¹).
- Byte-tamper coverage is **complete** in all three: I cross-checked each
  selftest's file list against every `fopen`/ctor in the corresponding
  verify() — rope 9/9, slice 5+9·NH (113/113 at real NH), merge 4+3·NH
  (40/40); nothing the verifier reads escapes tampering. Offsets hit real
  content (com @24 = first point x-limbs within the 144-B point; hp @36 =
  round-0 p(0) after the 32-B claim_H + 4-B count; ipa @−32 = a_final;
  ev/evals @4 = inside the first raw Fr_t). Restore-and-reverify ACCEPT closes
  each case. Gens and q.bin are CLI/registry inputs, correctly out of scope.
- The named evil for the headslice transpose ("swapped-head / transposed-wrong")
  and rope's evil=2 (wrong permutation) both verified live in my re-runs.

### 8. Numeric / representation — clean

- rope is the only driver with arithmetic: host math in int64 with
  |T| < 2^31 (int32 file format), |W| ≤ 2^16 (real tables; 2^7 toy) ⟹ each
  product < 2^47, |Y64| < 2^48 ≪ 2^63 — no overflow before the guard; the
  **|Y64| ≥ 2^47 throw is present** (zkob_rope.cu:155–156), placed before the
  evil-1 bump, guarding the downstream int32 chain format exactly per §6/§9.3
  (completeness, not soundness).
- slice and merge are pure int32 gathers — field fit trivial; all committed
  values PLAIN Fr via the int/long ctors (negative handling verified, see §4).
- No new kernels ⟹ no Montgomery rule question; the one mont-sensitive
  comparison (prover `operator*` vs verifier `k_fr_emul`) was verified
  equivalent in source and is runtime-probed by every honest weight-terminal
  pass.

### 9. ROPE_IMPL_REPORT proof-size question — RESOLVED

The **1.97 MB belongs to zkob_headmerge** and the **309 KB to zkob_rope**; the
report's rope and headmerge rows have prove/verify/bytes swapped, and the
checks column has headslice and headmerge swapped. Correct values are the
table at the top of this review (log lines: rope_selftest_accept.log:119,
headslice…:226, headmerge…:134; re-derivation: rope = 2 × 144 KB coms + 2 hp
(≈2.7 KB) + 3 IPAs ≈ 309 KB ✓ §7.2's ≈295 KB; merge = 13 × 144 KB coms +
12 hp + 13 IPAs ≈ 1.97 MB ✓ §7.2's ≈1.9 MB). The drivers themselves are
correct; only the report table needs fixing (MINOR-1).

### 10. me_weights memoization (report's open note) — NOT applied

zkob_headslice contains no memoization; each of the 72 `open_verify`/
`ipa_verify` calls rebuilds `me_weights` and refolds independently (and so
trivially "does not change results" — there is nothing to compare). The §9.1
gate passes without the sanctioned fallback: 29.00–29.25 s prove /
24.37–24.98 s verify < 30 s. See MINOR-5 on the thin margin.

---

## MINOR findings / notes (none block the gate)

**MINOR-1. ROPE_IMPL_REPORT.md table is wrong (rows swapped).** As resolved in
§9 above: rope row should read 50 checks / 1.99 s / 1.34–1.40 s / 309 KB;
headmerge row 58 checks / 3.48–3.49 s / 2.19–2.20 s / 1.97 MB; headslice row
105 checks (its timings/bytes are already correct). Purely a reporting error —
fix the report before the orchestrator wiring quotes it.

**MINOR-2. headmerge's selftest never exercises B ≠ C_pad.** All four cases —
(8,6,2)→C_pad 8, (4,4,2)→4, (16,12,4)→16, and real (1024,768,64)→1024 — have
B == C_pad. A B↔C_pad symbol confusion in `gather_Wm` (`m = e·B + t` vs
`Eh[i·C_pad + j]`), the forward π loop, or the chain-file check would
therefore be invisible to every test. I verified by inspection that the code
uses the correct symbols in all three places, and at the only deployed shape
the confusion is semantically vacuous (B = C_pad = 1024) — so this is a latent
portability trap, not a bug. Fix (cheap, next touch): add toy case B=4, C=6,
HD=2 (C_pad = 8 ≠ B). All OTHER π index swaps (i↔j, div↔mod, t·C+e, C vs
C_pad) are already caught by the B≠C cases via the honest csum throw and the
independently-written chain-file π⁻¹ check.

**MINOR-3. rope's sin/W2 weight terminal is never semantically forgery-tested.**
evil=3 bumps the COS table (h1 path) only; no evil targets the h2 weight
rebuild — the path carrying the σ sign. The code is symmetric by inspection
(same `build_weights`, same terminal check), the sign formula is exercised by
every honest case (random-sign toy tables), and a same-binary divergence is
impossible since both sides share `build_weights`; so this is a coverage note
in the spirit of RMSNORM MINOR-1. Fix (cheap): evil=5 — bumped sin table used
consistently → expect "h2 weight terminal".

**MINOR-4. headslice tests each error family on one tensor only** (wrong-head
on Q, column offset on V, untransposed on K). The three per-tensor paths are
structurally identical (same loop body, same `full_point`), so cross-coverage
is acceptable; noted so nobody assumes e.g. a K wrong-head evil exists.

**MINOR-5. headslice's §9.1 gate margin is < 1 s on prove** (29.0–29.25 s vs
~30 s), with no memoization applied (§10 above). Any regression (slower GPU,
driver change) trips the gate. The sanctioned host-only `me_weights`
memoization remains available — the dominant cost is 72 IPA verifies each
rebuilding `me_weights`; the 36 full-side openings share u_row = v_t and could
share one `fold_chain` per com too. Performance only; no action required now.

**MINOR-6. Scoped assumption (inherited, by design §4.7): standalone ACCEPTs
bind only internally.** rope's com_T, slice's com_Q/K/V, merge's com_O{hh} are
self-committed from input files; the §7.4 edges (A3/A3v/A4/A5/A6/A7.hh/
A12.hh/A14.hh/A15) and the registered-table sha256 checks are the defense that
pins them into the chain — the orchestrator MUST enforce them (softmax MINOR-5
precedent). In particular A15 (com_O2 ≡ com_attn_out) is also the end-to-end
backstop that would catch any hypothetical consistent-wrong-π implementation.

**MINOR-7. Missing-proof-file behavior is a throw, not a graceful REJECT**
(all three drivers; `open_or_die`/G1 ctor). Fail-closed — nonzero exit, never
a false ACCEPT. Same as softmax MINOR-4; orchestrator must treat any nonzero
exit as reject.

**MINOR-8. Cosmetic: `fold_public`, `tamper_byte`, `file_size`, `hh2` are
duplicated** across the drivers as static copies (rope/merge share
`fold_public` verbatim). Consistent with the file-scoped style of the earlier
drivers; if ever hoisted into the shared header, the "edit requires rerunning
EVERY selftest" rule applies. No action.

---

## What I would do before trusting them in the gate (summary)

**Accept all three files.** The verifiers are independent (every disk value
anchored, weights and π⁻¹ gathers rebuilt verifier-side, U_f2 == 1 forced at
all 14 sites), the FS schedules are absorb-for-absorb identical to the provers
and match §5 label-for-label, the rotate_half binding is carried by a
verifier-computed flipped-point opening of com_T and is forgery-tested, the
paired slice openings share one absorbed eval per claim, the π⁻¹ formula
matches §1.3 exactly with three independently-written instances
cross-checking it, padding is forced to zero where it matters, and there are
zero new CUDA kernels. Optional, cheap, next touch: fix the IMPL_REPORT table
(MINOR-1), add the B ≠ C_pad merge toy case (MINOR-2) and the sin-table evil
(MINOR-3), and carry the MINOR-6 orchestrator sentence into the wiring notes.
