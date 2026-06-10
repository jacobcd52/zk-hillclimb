# SOFTMAX_REPORT.md — zkob_softmax implementation report

Author: implementation agent. Date: 2026-06-10.
Spec: `/workspace/projects/zk-hillclimb/SOFTMAX_DESIGN.md` (DESIGN FINAL 2026-06-10),
followed exactly — relations, scales, FS schedule (absorb-by-absorb, labels
included), file names, CLI, selftest plan. **No design deviations.**

## Verdict

**`zkob_softmax selftest`: ALL PASS — 119 PASS / 0 FAIL, exit 0.**
Final line `ZKOB-SOFTMAX SELFTEST: ALL PASS` printed only because every check
passed. Full log: `/tmp/softmax_selftest.log`.

Real-scale (B = NCOL = 1024, LOW_E = −2^19, LEN_E = LEN_R = 2^20):
**prove 10.18 s, verify 11.60 s, proof+commitments 2,124,988 bytes (~2.03 MB)**
— inside the design's 8–12 s / 10–14 s prediction window.

## What was implemented

One new file, `/root/zkllm/zkob_softmax.cu` (compiles clean with the pinned
sm_89 `-dc -dlto` command and links against the standard object list), plus
the pinned table generator `/root/zkllm/gen_softmax_exp_table.py` (§7.4
script verbatim; NOT run as authority — the selftest used the in-driver
host-double fallback, printed as NON-AUTHORITATIVE). No existing file was
edited; `zkob_lookup.cuh` / `vrf_common.cuh` used as-is. **No new CUDA
kernels** — k_fr_emul, k_fr_fold, k_bcast_rows, k_bump, build_eq_tensor and
the upstream lookup/commit kernels cover everything, as the design promised.

The six in-driver obligations, one FS transcript, 16 IPA openings:

1. **Exp mapping lookup (R1)** — glu pattern verbatim: comb = z_ + r·E vs
   T_comb = table + r·mapped; verifier forms com_comb = com_z + r·com_E
   homomorphically (1-thread h_mul/h_add); B_f/T_f recomputed from the public
   table. glu's `mapped(0)==0` check deliberately NOT ported (no padding ⟹ no
   fabricated (0, mapped(0)) row; per design §4.1/§7.1).
2. **Limb range lookup** — L = 4 planes (r1 lo/hi, r2 lo/hi, 20-bit limbs at
   real scale) vs tLookupRange(0, LEN_R); D_L = 4·D, n1_L = 2 at real scale.
3. **Row-sum sumcheck (R2)** — ev_S = S̃(u_b) = Σ W_rs·E·𝟙 with
   W_rs = bcast_rows(eq(u_b)) ⊙ MK; the verifier **recomputes the public
   weight fold itself** (builds W_rs from u_b + the public mask, k_fr_fold
   chain) and **requires U_f2 == 1** (all-ones MLE, no opening).
4. **Bracket V1** — c1 = MLE of MK⊙E at u_r, weight eq(u_r)⊙MK recomputed and
   folded by the verifier, U_f2 == 1 required.
5. **Bracket V2** — c2 = MLE of P⊙S_bcast at u_r (pure-eq weight, my_eq
   accumulator); S_bcast is never committed: **U_f2 opens against com_S at
   the row-bit suffix of pt2** (broadcast-MLE = row-vector MLE at row bits).
6. **Residual reconstruction** — four Boolean-plane openings of com_L at u_r
   (v00/v10/v01/v11) + S_id = S̃(u_r rows) vs com_S, then the two plain-field
   identities checked with h_scalar:
   I1: 2^17·c1 + S_id − 2·c2 == v00 + LEN_R·v10  ("bracket r1 identity"),
   I2: r̃1 + r̃2 + 1 == 2·S_id                    ("bracket sum identity").

Verifier guards all present: dims.bin cross-checked against argv
(B, NCOL, LOW_E, LEN_E, LEN_R, LOG_OUT = 16); commitment row counts
(z/E/P/A_E = 1024, S = 1, L/A_L = 4096, m_E = LEN_E/1024, m_L = LEN_R/1024);
round counts (exp 4·logD, limb 4·(logD+2), each hadamard 4·logD); layout
constraint sets; every FS challenge re-derived after absorbing exactly what
it binds (§5 schedule absorb-for-absorb, labels included).

Honest-prover completeness throws (host math all `long long`, int64-safe per
§6): z_ out of [LOW_E, LOW_E+LEN_E); r1 ∉ [0, 2S) (defensive); residual
≥ LEN_R²; B ≠ NCOL or not pow2; gen.size ≠ NCOL; plus a defensive S[i] ≥ 1.

CLI, files (32 in obdir per the §7.2 list), chain interface as pinned:
`com_z` re-committed from the input file (byte-identity with rescale-10's
com_Xr is the orchestrator's check), `[P-int32-out.bin]` written unpadded
int32 at scale 2^16; the driver does NOT mkdir the obdir.

## Selftest summary (the full §8 plan, all green)

- **§8.1 small honest cases** (toy table mapped[k] = k+1, LOG_OUT = 16):
  (a) B=8, LOW_E=−8, LEN_E=16, LEN_R=32 (n1_E=2, n1_L=3);
  (b) B=4, same tables (n1_E=0, pure phase2 — the real-scale shape, n1_L=1);
  (c) B=16, LOW_E=−32, LEN_E=64, LEN_R=64 (n1_E=2, n1_L=4).
  Honest ACCEPT in all three. The §9.5 evil==0 convention sanity checks
  (every fold terminal == multi_dim_me: exp/limb lookup A_f & S_f; row-sum,
  V1, V2 S_f2; U_f2 == 1 for row-sum/V1; V2 U_f2 == broadcast MLE == S̃ at
  the row-bit suffix) run inside every honest prove and never threw —
  i.e. the fold/ME conventions all line up.
- **§8.2 semantic evil modes** — run in ALL three small cases, each rejected
  by EXACTLY the named check (reject string matched):
  - evil=1 (E[2,1]+=1 unmasked, S/P/limbs recomputed) → `exp lookup round 0`
  - evil=2 (P[0,1]+=1 MASKED, limbs mod-LEN_R² truncated) → `bracket r1 identity`
  - evil=3 (P[1,1]−=1 diagonal, honest P≥1 asserted, r1'=r1+2S limbs honest)
    → `bracket sum identity`
  - evil=4 (S[1]+=1, P/limbs recomputed) → `row-sum round 0`
  - evil=5 (V2 broadcast buffer Sb+=1, all commitments honest)
    → `IPA opening of V2 U_f2 vs com_S`
- **§8.3 byte tampers** — all 32 §7.2 files per case at the pinned offsets
  (lookup/hp @ 4+32, lvals @ 4, ipa_* @ −32, com_* @ 24, dims @ 0): all
  rejected; restore-and-reverify ACCEPTs at the end of every case.
- **§8.4 real scale** — B=NCOL=1024, gen1024 from ppgen (already present),
  exp table generated in-driver (flagged non-authoritative), z_ ~
  round(N(0, 2^13)) clipped. Honest ACCEPT; lookup_E.bin@36 tamper REJECTED
  (at exp lookup round 0); timings printed.

Independent sanity of the integer softmax spec (numpy over the emitted
P chain file): P ∈ [0, 65536], masked entries exactly 0, row 0 = {65536, 0…},
row sums ∈ [65515, 65557] — well inside the design's ±512 envelope.

Additionally (beyond §8, since the selftest drives prove()/verify()
in-process): the §7.1 CLI was smoke-tested at toy scale — `prove` then
`verify` ACCEPT (exit 0) through argv; `verify` with a mismatched LEN_R
rejects with `dims.bin mismatch` (exit 1); `verify` with a wrong seed
rejects at `exp lookup round 0` (exit 1, the splice/replay defense).

## Real-scale measurements

| metric | value | design estimate |
|---|---|---|
| prove (per head) | 10.18 s | 8–12 s |
| verify (per head) | 11.60 s | 10–14 s |
| proof+commitments | 2,124,988 B | "≈ 1.42 MB" (see note) |
| P chain file | 4,194,304 B (1024×1024 int32) | — |

24 heads ⟹ ≈ 4.1 min prove / 4.6 min verify for the softmax obligations
proper (score rescales and matmuls separate, per the design's accounting).

## Deviations

None from the normative sections (§2, §4, §5, §6, §7, §8). Two documentation
notes (not deviations):

1. **§7.2 file count.** The design prose says "31 files" but its own list
   enumerates 32 (1 dims + 9 com + 2 lookup + 3 hp + 1 lvals + 16 ipa). The
   list is implemented as written; the on-disk obdir has 32 files.
2. **§7.2 size estimate.** "≈ 1.42 MB/head" assumed 96 B per G1 point (the
   PHASE0_NOTES §3 figure); the actual serialized G1Jacobian_t is 144 B
   (3 × 48-byte Fp coordinates), so the 14,337 committed rows weigh ~2.06 MB
   and the dir totals 2.12 MB. Commitments dominate exactly as predicted
   (com_L = com_A_L = 589,824 B each); com_z is the dedup candidate the
   design already flags.

One spec-internal observation, no behavioral impact: §8.2's evil=2 narration
says I2 is "constructed to pass". With in-range limbs, r2 = mod-LEN_R²(2S−1−r1c)
makes I2 fail *as a tensor identity too* (the truncation offset +LEN_R²
survives into r̃1+r̃2+1−2S); since the verifier checks I1 before I2, the
rejection still lands on exactly the named check ("bracket r1 identity"),
which is what the selftest asserts and what passes.

## Concerns

- The selftest's real-scale exp table is the in-driver host-double fallback
  (no `softmax-exp-table.bin` was present). For the orchestrator's
  registration step, run `gen_softmax_exp_table.py` once and register the
  output by sha256 — the C++ driver must never be the table authority.
- The chain byte-equalities (§4.7: com_z == rescale-10 com_Xr, com_P ==
  values-matmul com_X) are orchestrator-level and untested here by design;
  z_-tampering inside the obligation is deliberately not an evil mode for
  the documented reason (com_z is self-committed; the chain check is the
  defense).
- Per §8.5, the other drivers' selftests were re-run after this validation
  (no shared file was touched): zkob_rescale, zkob_glu, zkob_rmsnorm — see
  the addendum line below for results. The remaining §8.5 steps (persist the
  driver to zkllm-src/, document as the next § of PHASE0_NOTES.md) edit
  existing files and are therefore left to the coordinator, per this task's
  "only create zkob_softmax.cu and the report" rule.

## Cross-driver selftest re-run (§8.5)

- zkob_rescale: ALL PASS (exit 0)
- zkob_glu: ALL PASS (exit 0)
- zkob_rmsnorm: ALL PASS (exit 0)

## Hardening round (2026-06-10, post-audit)

The independent audit (SOFTMAX_REVIEW.md, VERDICT: SOUND) recommended one new
semantic evil mode (MINOR-1) and noted a missing evil==0 convention block
(MINOR-2). Both applied. verify() is byte-for-byte unchanged; prove()'s honest
path is behaviorally identical (the only textual touches are the evil-6
plumbing — an m_L ternary and the limb-lookup strict flag, both no-ops at
evil==0 — and the new evil==0 sanity checks below, which are throws-on-
convention-bug that run inside every honest prove, like the existing five).

- **MINOR-1 → evil=6**: `L[0,idx] += LEN_R` with a compensating borrow
  `L[1,idx] -= 1` at an entry with r1 ≥ LEN_R (planted index, with a scan
  fallback). The reconstructed r1 = L0 + LEN_R·L1 — and hence I1, I2, all
  plane openings, and every commitment except com_L/com_A_L — stays
  consistent; m_L is committed from the HONEST limbs (the forged value is
  outside the range table, where prep's unchecked atomicAdd would write out
  of bounds; honest multiplicities are also the forging prover's best move);
  strict=false only for the limb-lookup fs_phase1. Rejected in all three toy
  cases by exactly `limb lookup round 0 p(0)+p(1) != claim` — PASS. The limb
  range lookup is the ONLY check standing between this forgery and ACCEPT,
  and it fires at round 0, as the audit predicted.
- **MINOR-2 → two new evil==0 convention blocks** (obligation 6, honest runs
  only): (a) each plane opening v00/v10/v01/v11 — the multi_dim_me of the
  {4B, NCOL} L at (u_r, plane bits) — is cross-checked against the
  multi_dim_me of that plane's D-slice at u_r itself, pinning the
  plane-bit/flat-layout convention I1/I2 rely on; (b) S_id is cross-checked
  against the multi_dim_me of the (never-committed) broadcast grid at u_r.
  Both ran inside every honest prove of the selftest and never threw.
- **MINOR-4 — no code change, by the audit's text.** The audit notes the
  missing-proof-file throw as "not a soundness gap; only a robustness
  nicety" and its summary recommends only MINOR-1 plus the MINOR-5
  orchestrator-notes sentence, so verify() was left untouched. Documented
  behavior: a missing/short proof file makes open_or_die / the
  G1TensorJacobian ctor throw out of verify() uncaught — the process aborts
  with a nonzero exit, never a false ACCEPT (fail-closed either way).
- **MINOR-5** → documented as a pinned orchestrator obligation in
  PHASE0_NOTES.md §15 (chain byte-equalities com_z == rescale-stage-2 com_Xr,
  com_P == values-matmul com_X).

Rejections landed on the exact named check on the first run — no expectation
was loosened. Selftest totals after rebuild (same pinned nvcc commands):
**122 PASS, 0 FAIL, exit 0** (was 119) — per toy case: honest ACCEPT +
6 semantic evil modes + 32 byte tampers + restored ACCEPT (×3 cases), plus
real-scale honest ACCEPT + tamper. Proof+commitments unchanged at
2,124,988 bytes. Real-scale wall-clock in this run (prove 42.7 s / verify
67.3 s) is inflated by a concurrent job on the shared GPU; the uncontended
figures remain the table above (10.18 s / 11.60 s, independently reproduced
by the audit at 10.29 s / 11.81 s). Full log:
/tmp/softmax_selftest_hardened.log.
