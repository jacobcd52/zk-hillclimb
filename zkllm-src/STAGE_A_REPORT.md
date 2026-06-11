# STAGE_A_REPORT — batched-opening primitive + zkob_fc claim mode, validated

Status: DONE, 2026-06-11. Implements TRANSPORT_REBUILD_DESIGN §6 Stage A with the
TRANSPORT_REVIEW required pins (F3/F4/F5/F6), the F8 battery-locus correction, F9, and
F10. Gates T1 and T2 below. All sources copied to
`/workspace/projects/zk-hillclimb/zkllm-src/`. No git commits (coordinator commits).
**vrf_common.cuh / zkob_lookup.cuh / fs_transcript.hpp untouched** (diff-verified against
the canonical zkllm-src copies), so the header-edit all-driver re-validation rule was NOT
triggered; zkob_rescale selftest re-run anyway as insurance: ALL PASS.

## 1. What was built

| file | status | what |
|---|---|---|
| `zkob_claims.cuh` | **NEW header** | Claim struct + claims.bin/drvstates.bin/witrefs serialization (EvalVar tag from day one), the F4 absorb encoding, the four new Fr kernels (`k_bo_eq_expand/k_bo_fold/k_bo_hp2/k_bo_axpy`) + runtime -dlto probe `bo_probe_kernels()`, `batch_prove` / `batch_verify` (the §2.2 G0–G5 protocol), per-phase `ZKOB_PROF=1` timers |
| `vrf_toy_batchopen.cu` | NEW | toy ground-truth harness, pins P1–P9 (below) against brute force — **TOY-BATCHOPEN: ALL PASS** |
| `zkob_batchopen.cu` | NEW | opening_batch driver CLI (`prove`/`verify`/`genq`/`selftest`) + the BO forgery battery — **33/33 ALL PASS** |
| `zkob_fc.cu` | EDITED (tail only) | claim mode behind `--claims` (runtime flag; old inline-IPA tail intact and default), claim-recompute in verify, prof timers, dual-mode selftest — **ZKOB-FC SELFTEST: ALL PASS** (old 3 cases byte-identical behavior + 3 claim-mode cases 9/9 each) |

New binaries link with the pinned build commands
(`nvcc -arch=sm_89 -std=c++17 -dc -dlto`, link `-dlto` against the standard upstream
object set). A new header (not an edit to the shared ones) was the deliberate choice so
the §13/header-edit rule stays untriggered in Stage A; only the three TUs that include
`zkob_claims.cuh` needed validation, and zkob_fc's selftest re-ran as required.

Protocol exactly as designed (§2.1/§2.2): G0 absorbs n_claims, every claim blob, every
distinct comref + sha256(file bytes), every drvstate → ρ (powers ρ^i, i = 0-based claim
index) → G2 batch-evaluation sumcheck, m_max rounds, degree-2 (3 evals/round, labels
bp0/bp1/bp2), round t binds VARIABLE m_max−1−t (k_fr_fold MSB orientation; r[] indexed
by variable per review §9) → G3 per-tensor v'_j absorbs ("vfin", canonical tensor order
= first appearance in claim order; tensor key = comref string) → verifier-computed
terminal `cur == Σ_j M_j(r)·κ_j·v'_j` → ρ' (powers by GLOBAL tensor index) → G5 one
fold+RLC+IPA per generator domain, ascending, `coef_j = ρ'_j·κ_j` on both C* and v*.
Batch transcript seed = `<run_seed>:opening_batch`. Artifacts: `batch_sumcheck.bin`
(n_claims/m_max cross-check fields + 3·m_max evals), `batch_vfin.bin`, `ipa_batch_<G>.bin`.

## 2. T1 — toy pins + selftest battery: **PASS**

`vrf_toy_batchopen` (3 cases: mixed-domains {4,8} with a 2-claim tensor + single-row +
max-vars; single-domain multi-tensor; minimal 1×4 single claim) — every pin PASS:

- **P1** flat layout: brute-force MLE over the flat table at point u_col‖u_row ==
  upstream `multi_dim_me({u_row, u_col})` bit-exact, every claim.
- **P2** old-primitive cross-pin: the audited §10 `open_prove`/`open_verify` ACCEPTs
  every emitted (point, eval) tuple as-is — the batch opens exactly the tuples the
  inline IPAs used to.
- **P3** honest batch ACCEPT (with the prover-side `<gen,a_g>` == RLC-of-folded-coms
  self-check enabled).
- **P4** completeness: Σ ρ^i·v_i == round-0 p(0)+p(1) == full embedded-hypercube brute
  force Σ_j Σ_x M_j^bf(x)·T_j[x] (M_j^bf computed with explicit zero padding over ALL
  m_max variables — independent of both prover tables and verifier split formula). This
  is also the "batched == sum of the individual opens" check, each individual open
  pinned by P2.
- **P5** fold orientation: every v'_j == brute-force P_j(r[0..vars_j)).
- **P6** M_j(r)/κ_j split formulas == zero-padded full-variable brute force.
- **P7** G5 split: <a_g, me_weights(r_col)> == v*_g with a_g rebuilt by host
  brute-force row-folds.
- **P8 (F9)** ρ-sensitivity: 1-bit change in the LAST claim's eval, or the LAST
  drvstate, changes ρ — the off-by-one-absorb guard the review asked for.
- **P9** forgery smoke with locus checks (round0 / terminal / ipa<G>).
- All four new kernel shapes probed at runtime against the 1-thread h_scalar helpers
  (`bo_probe_kernels`, run at the top of the toy and BOTH selftests): clean under
  -dlto. The per-round strict p(0)+p(1)==cur prover self-checks re-probe every prove.

`zkob_batchopen selftest` — **33/33, every forgery dying at its NAMED locus**:

| case | locus (expected == got) |
|---|---|
| honest / all restores | accept |
| **BO-1a** false eval, honest-procedure batch (F8 split) | round0 |
| **BO-1b** fully adaptive prover (per-round p1 := cur−p0, v'_last forged to pass G3) | ipa8 (the poisoned tensor's group IPA) |
| RLC-cancellation attempt (δ' = −δ·ρ^{-2} from the doctored list's own ρ) | round0 (ρ re-randomizes through G0; the F9 property live) |
| BO-2 omitted / BO-10 duplicated / reordered / BO-3 comref-swapped claim | claims_match |
| BO-4 ρ derived from doctored list, honest claims.bin shipped | round0 |
| BO-5 round-eval tampers (round 0 AND round 1) | round0 / round1 |
| BO-6 vfin tamper | terminal |
| BO-8 claims.bin field tampers (id, eval) | claims_match (file never parsed — F5) |
| BO-8 batch_sumcheck n_claims field | xcheck (F6 cross-check) |
| BO-8 ipa tampers (a_final, round-0 L) | ipa4 / ipa8 |
| BO-8 verifier-side drvstate divergence | round0 |
| BO-7 vars lie (junk-padded point, BOTH lists); consistent n_rows lie | shape |
| **F10 com file with extra trailing rows** (the F3 covert-channel case); truncated com | shape — before any fold_chain, no segfault |
| BO-9a full cross-run replay / BO-9b batch-artifacts-only replay | claims_match / round0 |
| F11 single foreign ipa_batch file | ipa8 |
| BO-11 substituted com file post-prove (honest hash of substituted bytes) | round0 (G0 comref-hash divergence) |
| BO-12 missing domain group IPA | group_missing |
| Committed EvalVar claim | evalvar (path closed until Stage D) |
| n_claims == 0 | empty |

`zkob_fc selftest` — old-mode cases (4,6,3)/(8,8,8)/(16,12,5) PASS unchanged; claim-mode
cases 9/9 each: honest = conditional-ACCEPT + batch ACCEPT; sumcheck round/terminal and
com_Y/registered-com_W tampers still die DRIVER-side (unchanged loci — this is the BO-11
driver-level catch); claims.bin eval-tamper and dropped-claim die at claims_match; the
batched-IPA tamper dies at ipa<G>. The (8,8,8) case exercises IN_pad == OUT_pad with ONE
shared generator file (see §5 invariant note); (16,12,5) exercises a two-domain batch.

## 3. T2 — real-scale old-tail vs claim-mode+batch: measured

RTX 4090, three runs each, stable to ±2%. `ZKOB_PROF=1` splits. Honest ACCEPT in all
configurations; at-scale negatives re-confirmed (trailing-row on com_W → `shape`;
ipa_batch_32768 a_final tamper → `ipa32768`; restored → ACCEPT).

**gate_proj shape, 1024×768×1024 (domains: all gen1024):**

| | OLD (inline IPAs) | NEW (claim mode + batch) |
|---|--:|--:|
| fc verify protocol time | **1.334 s** = absorb 0.008 + rounds 0.002 + ipa_tail **1.324 (99.3 %)** | **0.011 s** (absorb 0.008 + rounds 0.002 + claim_emit 0.001) |
| batch verify | — | **0.585 s** = claims_match 0.000 + shape 0.001 + absorb 0.008 + rounds 0.003 + terminal 0.004 + fold_1024 0.154 + ipa_1024 0.416 |
| **total verify (protocol)** | **1.334 s** | **0.596 s** → **2.24×** |
| prove wall | 2.19 s | 1.03 s (fc) + 1.11 s (batch) = 2.14 s |

**lm_head, 1024×768×32000 (domains gen1024 + gen32768 — the stress case):**

| | OLD | NEW |
|---|--:|--:|
| fc verify protocol | **27.14 s** (ipa_tail 27.13 = **99.96 %**: 2 gen32768 IPAs + 1 gen1024) | **0.011 s** |
| batch verify | — | **15.77 s** = fold_1024 0.056 + ipa_1024 0.422 + fold_32768 0.101 + ipa_32768 **15.16** + 0.03 checks |
| **total verify (protocol)** | **27.14 s** | **15.78 s** → **1.72×** |
| prove wall | 21.4 s | 14.3 s (fc; the two inline gen32768 ipa_proves gone) + 11.4 s (batch: setup 2.2, 25 sumcheck rounds 0.26, ipa 8.5) = 25.7 s (+20 %, at the edge of the design's ±20 % flag; per-run the batch cost amortizes over ALL drivers, not one) |

Batch artifacts: claims.bin 2.2–2.5 KB, batch_sumcheck 1.9–2.4 KB, batch_vfin 104 B,
ipa_batch_* 2.9–4.4 KB — the ~50–60 KB/run class the design predicted.

### Does T2 support the §2.7 verify projection?

**The load-bearing claims: confirmed.**
- Inline IPAs are 99.3–99.96 % of per-driver fc verify protocol time — *stronger* than
  the design's ~85–90 % whole-walk estimate (the remainder of the walk gap is the
  subprocess overhead §1.3 already accounts separately).
- The per-driver residual after claim conversion is ~11 ms (absorbs + round checks).
  234 drivers × this class ≈ seconds, matching §2.7's "round checks unchanged" row.
- The batch replaces N inline IPAs with exactly ONE per domain: measured 0.42 s
  (gen1024) and 15.2 s (gen32768) fixed cost per domain per run, independent of claim
  count. The 2.24×/1.72× single-driver end-to-end ratios are the *floor* of the win —
  a single fc instance has only 3 IPAs to batch; the full walk has 1,535 collapsing
  into 4 domain IPAs.

**Two measured caveats the projection depends on (both already flagged in the design,
now with numbers):**
1. **Per-tensor fold_chain is launch-latency bound: ~34–51 ms per 1024-row tensor**
   (fold_1024 = 0.154 s for 3 tensors here). At faithful-arch-v1 scale (1,242 distinct
   commitment files) the G5 fold pass as currently implemented costs ~45–65 s — it
   would *become* the post-rebuild bottleneck and eat the §2.7 "RLC + folds 1–3 s" row.
   That row assumed flat batched GPU kernels; the per-tensor sequential fold must be
   flattened (one kernel over all tensors per fold level, or stream-batched) in
   Stage B/C. This is an implementation item, not a protocol change — flagged as the
   first Stage-B work item.
2. **The gen32768 IPA at 15.2 s is host-loop dominated** (me_weights = G·logG ≈ 491k
   1-thread h_scalar round-trips, plus the s-vector build). This is the known §6.2
   fast-helpers item; until it lands, the 4-domain batch costs ≈ 16–17 s in IPAs alone
   (0.4 + ~1 + ~2 + 15.2), which still fits the §2.7 "2–6 s with GPU-lift" row only
   AFTER the lift. Verify-total projection ≈ 40–85 s remains plausible: ~16 s IPAs
   (pre-lift) + folds (post-flattening 1–3 s) + ~30–60 s round checks + single-process
   overhead.

Net: T2 supports the projection's structure (kill 1,535 IPAs → 4; per-driver tails
collapse to ms), with the fold-flattening and me_weights GPU-lift now *measured* as the
two items standing between the as-built batch and the §2.7 mid-estimate.

## 4. The audit pins, as implemented (file:line)

- **F3 (covert-channel pin):** `zkob_claims.cuh:818` — per distinct tensor,
  `com_file_point_count == n_rows` checked when loading each com, with
  `n_rows == 2^{vars−logG}` enforced structurally at `zkob_claims.cuh:263`
  (`derive_tensors`), both BEFORE any fold_chain (G5 is ~200 lines later; coms are
  loaded once at the shape check and reused). Battery: F10 trailing-row/truncation
  cases + the consistent-n_rows lie, plus the at-scale trailing-row re-check (REJECT
  `opening_batch.shape`, row count 1025 != 1024). Driver-side complements unchanged
  (fc `zkob_fc.cu:290` "commitment row counts"). The Stage-C audit item — every driver
  checked for commitments whose ONLY size check was open_verify — remains scheduled
  for Stage C per the review.
- **F4 (absorb encoding):** `zkob_claims.cuh:88-99` — `claim_blob` length-prefixes id
  and comref, length-prefixes the point vector, and appends the EvalVar tag byte
  explicitly before the (tag-dependent) eval bytes; the SAME canonical bytes are used
  for claims.bin and for the per-claim `tr.absorb("claim", …)` at
  `zkob_claims.cuh:299`. Comref absorbs are `len(key) ‖ key ‖ sha256(file bytes)`;
  drvstate absorbs are `len(id) ‖ id ‖ state32`. Tensor-identity key pinned = comref
  string, canonical tensor order = first appearance (`derive_tensors`,
  `zkob_claims.cuh:253`).
- **F5 (claims_match load-bearing & inseparable):** `zkob_claims.cuh:792-801` — batch
  verify takes the VERIFIER-recomputed list (vaccdir) as its input; the prover's
  claims.bin is read as raw bytes and memcmp'd against the re-serialized verifier
  list, never parsed, never consumed. claims_match is a reject branch INSIDE
  `batch_verify` (a component of the opening_batch verdict, locus
  `opening_batch.claims_match`), not a separable orchestrator check. The W claim
  carries the REGISTERED com path as comref on both sides (`zkob_fc.cu:241` prove,
  `zkob_fc.cu:346` verify-recompute), so the `*.commitment_opening` discharge is
  explicit in the compared lists.
- **F6 (drvstate provenance + structural edges):** `zkob_claims.cuh:787` — the
  drvstates absorbed by batch verify come from the verifier's own accumulator dir
  (written by the verifier's per-driver runs via `drvstate_emit(vaccdir, …)`,
  `zkob_fc.cu:349`); the prover's drvstates.bin is never read by the verifier (it is
  prover-internal transcript input only). batch_sumcheck.bin's redundant
  n_claims/m_max fields are cross-checked (`zkob_claims.cuh:852-854`, locus `xcheck`);
  n_claims == 0 REJECTS explicitly (`zkob_claims.cuh:788`, locus `empty`); Committed
  EvalVar REJECTS (`:790`, locus `evalvar`); single-claim and single-row (u_row empty)
  edges covered by the toy "minimal" case end-to-end.
- **F8 (BO-1 locus split):** implemented as battery cases BO-1a (honest-procedure
  prover, evil=1 → `round0`) and BO-1b (fully adaptive: per-round p1 := cur−p0 and
  v'_last forged via verifier-formula inversion to pass G3, evil=2 → the poisoned
  tensor's group IPA). The intermediate adaptive variant (honest v') is the same locus
  class as BO-6 (`terminal`), covered there.
- **F9:** toy P8 + the live demonstration in the RLC-cancellation battery case.
- **F10:** battery cases above.
- **F12** (conditional-verdict gating) is orchestrator wiring — Stage C, as scheduled
  in the review's table ("A/B" items done here; the orchestrator does not yet exist
  for opening_batch).

## 5. Deviations / flags (honest list)

1. **H-slot delivered via `zkob_batchopen genq`, not an edit to ppgen.cu.** The design
   (§4.4.3) says "ppgen/q.bin format gains the H slot". ppgen.cu is an upstream-built
   target and the standing rule is upstream stays pristine, so the 2-slot q file
   ([Q, H], 288 bytes) is produced by the new `genq` mode instead. Format-compatible:
   `G1TensorJacobian(filename)` sizes from the file, every existing consumer reads
   index 0 only — verified live (the toy, the batch selftest, and BOTH T2 runs all ran
   on 2-slot q files, including old-tail fc verify). Existing 1-point q.bin files also
   still load (H simply absent until Stage D registration re-issues q.bin via genq —
   a register.py one-liner in Stage C, no driver change). NOT a header edit; no
   re-validation triggered.
2. **One-generator-file-per-domain-size is load-bearing and now explicit.** The G5
   grouping keys on domain SIZE; an RLC across different generator vectors of the same
   size would be unsound, and commitments do not carry generator identity. The real
   registration already satisfies this (one gen<G>.bin per size); the batch CLI binds
   each domain to exactly one gen file and checks sizes. The fc claim-mode selftest
   was built to share one generator per size accordingly (old-mode selftest keeps its
   independent random gens — unchanged, since inline IPAs are per-tensor). Stage C
   should add an orchestrator assertion that claim domains map 1:1 onto registration
   gen files. (This also matches §4.4.4's (domain, comref-class) grouping plan.)
3. **claims_match compares absolute comref paths.** Prover and verifier must name com
   files identically; fine for selftests and the single-box walk, but Stage B/C should
   canonicalize comrefs to run-dir-relative paths before the orchestrator wires
   multi-process walks. Flagged, not fixed here (no orchestrator exists to define the
   root yet).
4. **Prover-side drvstates.bin and witrefs.txt are unverified prover-internal inputs**
   (the verifier never reads them) — byte-tampering them post-prove is inert on its
   own and surfaces only as transcript divergence if the artifacts were rebuilt; the
   battery covers the verifier-side divergence path (BO-8 drvstate case).
5. **Stage-A scope of "ACCEPT-conditional":** fc claim-mode verify prints
   `ACCEPT-conditional` and exits 0; the gating of overall verdicts on opening_batch
   (F12) is orchestrator logic scheduled for Stage C. In the selftests the pipeline
   helper enforces the gate manually (driver REJECT or batch REJECT ⇒ case REJECT).
6. **evil modes in batch_prove** (selftest forgery construction, modes 1–3) live in
   the shared header; they are selftest-only by convention (mode 0 is the only path
   the CLI exposes). Kept because adaptive-forgery construction (BO-1b) needs prover
   internals; mirrors the `strict=false` convention in zkob_lookup.cuh.
7. **Performance flags for Stage B** (measured, §3): per-tensor fold_chain ≈ 34–51 ms
   (→ flatten before full-walk batch), me_weights host loop ≈ 12 s at G=32768
   (→ §6.2 GPU lift). Neither blocks Stage B correctness work (fc→rescale chain).

## 6. Gate verdicts

- **T1: PASS.** TOY-BATCHOPEN ALL PASS (3 cases, all pins, kernel probes clean);
  ZKOB-BATCHOPEN SELFTEST 33/33 ALL PASS with named loci incl. F3/F10 trailing-rows,
  dropped/reordered/misattributed claims, RLC-cancellation, BO-1a/1b per F8;
  ZKOB-FC SELFTEST ALL PASS both modes.
- **T2: MEASURED.** 1024×768×1024: protocol verify 1.334 s → 0.596 s (2.24×), inline
  IPAs confirmed 99.3 % of old verify. lm_head 768×32000: 27.14 s → 15.78 s (1.72×),
  IPAs 99.96 % of old verify. Single-driver ratios are the floor (3 IPAs → 1-per-domain);
  the design's "~90 % of verify is inline IPAs" is CONFIRMED on this driver, and the
  §2.7 projection holds structurally conditional on the two flagged Stage-B
  implementation items (fold flattening, me_weights GPU lift).

Next (Stage B per §6): convert zkob_rescale, two-driver accumulator through the
orchestrator on the fc→rescale pair incl. lm_head gen32768 instances, BO battery at
pair scale, prover batch-sumcheck overhead measurement — plus the two performance items
above and comref canonicalization.
