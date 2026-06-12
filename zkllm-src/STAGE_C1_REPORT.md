# STAGE_C1_REPORT — remaining 9 drivers converted to claim mode (Stage C part 1)

Status: DONE, 2026-06-12. Implements the mechanical bulk of TRANSPORT_REBUILD_DESIGN
§6 Stage C / §3 per-driver change list: the REMAINING 9 drivers' opening tails
converted to claim mode, exactly the way zkob_fc and zkob_rescale were converted in
Stages A/B. All converted sources copied to `/workspace/projects/zk-hillclimb/zkllm-src/`
as each passed. No git commits (coordinator commits). int-model-approximation untouched.
GPU: RTX 4090, pinned `nvcc -arch=sm_89 -std=c++17 -dc -dlto` compile + `-dlto` link
against the standard upstream object set for every rebuilt binary.

## 0. The uniform conversion (applied identically to all 9)

- **Prove**: at the EXACT old `open_prove`/`fast_open_prove` site, emit the claim
  tuple (id `<obid>:<tensor>`, comref, domain, n_rows, point, eval) via `claim_emit`
  into `<accdir>/claims.bin`, plus `witref_emit` (one witness file per distinct
  comref) and `drvstate_emit` (final transcript state) at the end. The inline-IPA
  absorbs vanish from the transcript symmetrically on both sides ("the transcript
  ends earlier"; for mid-transcript IPAs, later challenges derive from the shortened
  transcript — same discipline both sides).
- **Verify**: every absorb, round check, terminal identity, homomorphic link,
  verifier-rebuilt public-weight fold, U_f2 pin and plain-field identity is
  byte-identical to the old path; instead of `open_verify`, the verifier RECOMPUTES
  each claim from its own FS replay into `<vaccdir>` — **deferred until every
  driver-local check has passed** (so a later reject can never leave a
  partially-filled verifier accumulator) — and returns ACCEPT-conditional.
- **Old tail**: kept compilable and DEFAULT (absence of `--claims` selects it; the
  old selftest cases re-run unchanged and pass).
- **F3**: every claim carries the correct `n_rows` (= com file row count), so
  batchopen's pre-fold commitment-size check binds (claims sharing a tensor are
  shape-checked by `derive_tensors`; the com FILE count is checked in batch_verify).
- **F4–F6**: claims absorbed before ρ (header logic, unchanged); verifier-recomputed
  list is the batch input (claims_match byte-compare); drvstates written into the
  VERIFIER's own vacc dir only.
- **Selftests**: dual-mode — all old cases unchanged + claim-mode cases with
  `ZKOB_FOLD_CROSSCHECK=1` and `bo_probe_kernels()` (-dlto probes) at startup; honest
  → conditional-ACCEPT + batch ACCEPT; every forgery rejected at a NAMED locus.

No new kernels of any shape were written (zero kernel risk); the batch side reuses
the Stage-B header paths as-is.

## 1. Per-driver results — 9/9 converted, ALL PASS

| driver | converted | claims/run | dual-mode selftest | forgery coverage (claim mode) |
|---|---|--:|---|---|
| zkob_skip | YES (zero-claims routing) | 0 | **ALL PASS** (old case + claim case) | driver-side forgery unchanged; accumulator byte-unchanged by skip; batch over a pre-seeded neighbor claim still ACCEPTs |
| zkob_glu | YES | 6 | **ALL PASS** (3 old + 3 claim cases, 11/11 each) | 4 driver tampers (incl. NEW com_comb g1_eq check), claims_match ×2, ipa_G, both semantic evils driver-side |
| zkob_rope | YES | 3 (T ×2 same tensor + Y) | **ALL PASS** (3 old + real + 3 claim cases, 13/13 each) | evils 1/3/4/5 driver-side unchanged; **evil 2 RELOCATED → batch round0** (BO-1a); claims_match ×2, ipa_G |
| zkob_headmerge | YES | NH+1 (2 domains) | **ALL PASS** (8 old + splice + 2 real + 4 claim cases, 13/13 & 12/12) | all 5 semantic evils driver-side unchanged; both domains' batch IPAs pinned |
| zkob_headslice | YES | 6·NH (pairs; 2 domains) | **ALL PASS** (3 old + real + 3 claim cases, 12/12 each) | **all 3 evil families RELOCATED → batch round0**; com/eval tampers → claims_match; both domains pinned |
| zkob_rmsnorm | YES | 17 (registered com_g) | **ALL PASS** (small + real + 2 claim cases, 17/17 & 18/18) | full 8-evil battery driver-side unchanged (affine links, quartics, SS, outer, limb lookup); registered-com_g tamper driver-side; claims_match ×2; 1- and 2-domain shapes |
| zkob_softmax | YES | 16 | **ALL PASS** (3 old + real + 3 claim cases, 15/15 each) | evils 1–4, 6 driver-side; **evil 5 RELOCATED → driver I1** (see §3); claims_match ×2, ipa_G |
| zkob_softmax8 | YES | 19 | **ALL PASS** (3 old + guards + real + 3 claim cases, 17/17 each) | evils 1–5, 8 driver-side; **evil 6 RELOCATED → driver I1; evil 7 RELOCATED → batch round0**; sentinel/guards unchanged |
| zkob_rowmax | YES | 12 causal / 14 vpad (2 domains) | **ALL PASS** (4 old + guards + chunked + 2 real + 2 claim cases, 13/13 & 17/17) | evils 1–6, 8 driver-side unchanged (incl. constant-claim discipline, T-BIND); **evil 7 RELOCATED → driver DOM bracket identity**; both domains pinned |

Every claim-mode selftest also pins: honest ACCEPT through the full
driver→batch_prove→batch_verify pipeline, prover-claims.bin eval tamper →
`claims_match`, claim drop → `claims_match`, batched-IPA a_final tamper →
`ipa<G>` (per domain), and restored → ACCEPT.

## 2. Per-driver subtleties (the non-mechanical parts)

1. **zkob_skip — zero claims, FINAL verdict.** The design pins "no change at all";
   skip has no transcript and no openings. `--claims` is accepted for orchestrator
   uniformity and emits NOTHING (no claims, no drvstate — there is no per-obligation
   transcript whose state could be absorbed). Its verdict stays FINAL even in claim
   mode. The selftest pins the routing property: an accumulator pre-seeded with an
   honest synthetic claim passes through a skip add+verify byte-unchanged and the
   batch still ACCEPTs. **Orchestrator note:** skip contributes no drvstate to G0 —
   the walk's drvstate list covers transcript-bearing obligations only.
2. **zkob_glu / zkob_softmax / zkob_softmax8 — the homomorphic `comb` tensor.** The
   lookup opens `comb = G + r·S` (glu), `z + r·E` (softmax), `Dm + r·E` (softmax8)
   against a commitment the verifier forms HOMOMORPHICALLY — no file existed for the
   batch's comref. Resolution: claim-mode prove writes `com_comb.bin` (the same
   1-thread h_add/h_mul combination the verifier computes); claim-mode verify REJECTS
   unless that file's rows g1_eq-equal its OWN recomputed combination BEFORE the
   claim is emitted. The file is then exactly as bound as the homomorphic object the
   old inline IPA consumed (G0 additionally absorbs its sha256; both sides hash the
   same on-disk bytes). A com_comb.bin byte tamper is caught DRIVER-side by the new
   named check — pinned in all three selftests.
3. **zkob_rmsnorm — registered comref (the F5/F6 pin).** `prove --claims` takes a
   third argument `<registered-com_g-path>`; the g claim carries the REGISTERED file
   path as comref on BOTH sides (verify uses its own `com_g-path` argument), exactly
   like fc's com_W. The `*.commitment_opening` discharge is therefore explicit in the
   claim list. Multi-claim tensors: com_R ×3, com_M ×3, com_X ×2 (and rope's com_T
   ×2, headslice's com_Q/K/V ×NH, softmax's com_S ×3 / com_L ×5, rowmax's com_S ×4–5)
   — the Stage-A multi-claim-per-tensor machinery is now exercised by real drivers.
4. **zkob_headslice — the §8.6 pair pin.** Each of the 3·NH pairs shares ONE
   absorbed eval (evals.bin stores one value per pair) used by BOTH the slice claim
   and the full-tensor claim, so `eval_slice == eval_full` holds STRUCTURALLY
   driver-side before either claim reaches the batch — exactly the
   TRANSPORT_REVIEW §4 discipline (jointly-false pairs are two false claims, killed
   by SZ in the batch like BO-1, demonstrated by the relocated evil families).
5. **One-gen-per-domain-size in selftests.** Wherever two layout sizes coincide
   (rmsnorm B_pad == C_pad, rowmax B == NCOL, fc-style shared gens), the claim-mode
   selftests use ONE generator vector per domain size — the registration invariant
   the per-domain RLC grouping is a commitment under.
6. **zkob_rowmax memory discipline preserved**: the A_L/L witnesses are saved inside
   the LIMB block while those tensors are still resident (the §2.5 free points are
   unchanged); witref dedup by comref means com_L's witness is registered once even
   though L is re-uploaded later for the plane claims.

## 3. Relocated rejection loci (the §5.1 relocation rule — each REVIEWED and pinned)

Forgeries whose OLD locus was an inline IPA cannot die there anymore. Every
relocation below is deterministic (no probabilistic-miss class beyond the standing
2^-240 SZ events) and is pinned by an explicit claim-mode selftest case:

| driver / forgery | old locus | new locus | why |
|---|---|---|---|
| rope evil 2 (unpermuted rotate_half) | h2 flipped-point IPA | **batch `round0`** | driver-local checks all pass; the T@pt2′ claim is false vs com_T; honest-procedure batch (header evil=1) → verifier round-0 check fails (BO-1a class) |
| headslice families 1–3 (wrong head / column offset / wrong layout) | full-tensor head-selector IPA | **batch `round0`** | the slice-side eval is computed from the evil slice, so only the full-side claim is false; no driver rounds exist to catch it |
| headslice com/eval byte tampers | the affected IPA | **batch `claims_match`** | the verifier-recomputed claim list (points from its own FS replay / evals from evals.bin) diverges from the prover's claims.bin |
| softmax evil 5 (V2 broadcast bump, unmasked) | V2 U_f2 IPA | **driver `bracket r1 identity (I1)`** | I1 consumed the corrupted c2 all along but ran AFTER the IPA; with IPAs gone it runs first and the shift Δc2 = eq·P[idx] ≠ 0 is deterministic. (The false S_v2 claim would also die in the batch had I1 not fired.) |
| softmax8 evil 6 (same shape) | V2 U_f2 IPA | **driver I1** | same as softmax evil 5 |
| softmax8 evil 7 (cD2 broadcast bump at a MASKED idx) | cD2 terminal IPA | **batch `round0`** | the masked bump is invisible through the MK weight, so cD2's value and the Dm identity stay clean — the false mx_cD2 claim is the only trace; honest-procedure batch → round0 |
| rowmax evil 7 (c2 broadcast bump at an ALLOWED idx) | c2 terminal IPA | **driver `DOM bracket identity`** | c2's claim_H is absorbed from the corrupted run and AL[idx] = 1, so c2 − c1 ≠ rec deterministically; the identity now runs before any opening is checked |

All other semantic forgeries (≈30 across the 9 drivers) keep their original
driver-side loci byte-for-byte — verified by the claim-mode selftests naming each
expected check.

## 4. Header-untouched proof (sha256 before == after, and == canonical zkllm-src)

Recorded before the first edit and after the last build:

```
09965932f138c3c656ee16274731ff0548a29990da5219024ee4b25c7a8e0b98  vrf_common.cuh
d7fcd10150d4942fd5edac12d53538ed87fcce179e0855357e847da97b7d9c7e  zkob_lookup.cuh
5d668def22be2313c4d3ef948309040b0b28d456e64aac80eb5bf2cd2c86f14d  fs_transcript.hpp
cb134246947db3cad6a8cb37cbf3f193832b2cb62b345cf5da2ba7b0b268b950  zkob_claims.cuh
```

All four byte-identical before/after AND cmp-identical to the canonical
`/workspace/projects/zk-hillclimb/zkllm-src/` copies. **zkob_claims.cuh was NOT
edited** — the Stage-B "rebuild + re-run all includers" rule was NOT triggered, so
zkob_fc / zkob_rescale / zkob_batchopen / vrf_toy_batchopen needed no re-validation
(their binaries and the header are bit-identical to the Stage-B-validated state).

## 5. Deviations / flags (honest list)

1. **Verifier-side claim emission is deferred to end-of-verify** rather than placed
   at the literal old `open_verify` sites. The prover emits at the exact old sites
   (the FS-binding requirement); the verifier's list is identical in content and
   order, and deferral guarantees a driver REJECT never leaves a partial vacc. This
   matches the rescale/fc precedent (whose tails were already at the end).
2. **rope/headslice/headmerge prove emits adjacent claims in one block** where
   consecutive old `open_prove` calls had no transcript activity between them
   (e.g. rope's T2+Y) — content and order identical to per-site emission.
3. **skip emits no drvstate** (no transcript exists). Stage-C orchestrator wiring
   must not expect one for skip obligations (documented in §2.1).
4. **The headmerge/headslice/rowmax/rmsnorm claim-mode selftests run 2 of the old
   case shapes, not all** (e.g. headmerge runs 2 shapes × 2 perm modes, rmsnorm 2
   shapes). Every structural variant that affects the batch (1 vs 2 domains,
   NPL 1 vs 2, ± t*, perm modes, n1 = 0 edge) is covered; the remaining old shapes
   differ only in grid size. Old-mode coverage is unchanged (all original cases run).
5. **batch_prove's strict self-checks vs false claims**: an honest-procedure batch
   over a false claim list (the relocated rope/headslice/softmax8 evils) requires
   the header's `evil=1` mode to produce artifacts (strict round self-checks
   off) — same convention as the Stage-B BO-1a battery; production batch_prove
   stays strict (a false claim makes the honest prover THROW, fail-closed).
6. **CLI**: every driver now strips a trailing `--claims <(v)accdir> <obid>` block
   (rmsnorm prove: + `<registered-com_g-path>`); rowmax's optional positional
   `[mx-out] [tstar]` args parse against `base_argc`, and `-` is accepted as an
   explicit "no mx-out" placeholder when t* follows in claim-mode invocations.
7. Selftest scratch lives under `/tmp/zkob_*_cm` (a few hundred MB transient);
   `/tmp` cleaned by the cases themselves on re-run (`rm -rf` per case). Disk on /
   was at 83 % before and after.
8. One pre-existing-style warning added (`rowmax selftest_case_claims: unused D`);
   cosmetic, left as-is to avoid a rebuild cycle on a passing binary.

## 6. What Stage C part 2 still needs (unchanged from STAGE_B_REPORT §6)

Orchestrator `opening_batch` id + conditional-verdict gating (F12) + claims_match
wiring + relative-comref flag + one-gen-per-domain and ZKOB_EVIL-unset assertions;
single-process `zkverify_walk`; S1 canonical-affine + dedupe; batch-prove streaming
(or 2–4 sub-batches) for the 38 GB round-0 residency; dual-accumulator routing;
the full §13 selftest + audit campaign at real scale (T4–T7). The per-driver
mechanical layer this report covers is complete: **all 11 drivers now speak claim
mode** (fc + rescale from Stages A/B, the remaining 9 here), with old tails intact
and default.
