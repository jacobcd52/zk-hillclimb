# STAGE_C2_REPORT — orchestrator batch integration + batched verifier + the full-walk before/after

Status: DONE, 2026-06-12. Closes TRANSPORT_REBUILD_DESIGN §6 Stage C gates T4/T5:
the full faithful-arch-v1 walk runs end-to-end in batched transport (every driver
in claim mode, ONE zkob_batchopen discharge per sub-batch), honest ACCEPT with
checked = 65, every forgery rejected at its correct NAMED locus, and the verify
wall measured at **27.1 s** (inline transport: 999.5 s — 36.9×; first batched cut:
383 s). Selftest: `selftest.sh c2sp --batched-only` → **15 PASS / 0 FAIL, ALL
PASS** (`/root/zkorch/selftest_c2sp_batched.log`; run dir `/root/zkorch/c2sp-fab`).
All 12 driver selftests re-run after the rebuild: ALL PASS (incl. batchopen 33/33).
Protected headers diff-verified untouched (sha256 before == after for
vrf_common.cuh, zkob_lookup.cuh; zkob_claims.cuh also untouched this stage).
No git commits. int-model-approximation untouched.

## 1. T5 — the before/after table (full faithful-arch-v1 walk, RTX 4090)

| | inline (pre-rebuild) | batched, first cut | batched, Stage C2 FINAL | vs inline |
|---|--:|--:|--:|--:|
| prove wall | 1062 s | 636 s | **522.0 s** | 2.03× |
| verify wall | 999.5 s | 383 s | **27.1 s** | **36.9×** |
| proof size | 175.6 MB | 176.3 MB | 176.3 MB | 1.0× (S1 re-layout is a separate, designed item — §2.6) |
| openings | 1,535 inline IPAs | 7 batched | 7 batched (1,535 claims, 1285M elements) | — |

Where the final verify wall goes (transcript.json timing, honest run):

| component | time | n |
|---|--:|--:|
| zkob_batchopen verify (7 sub-batches: RLC+folds, sumcheck, per-domain IPAs, claims_match) | 7.6 s | 7 |
| softmax8 claim recompute | 5.4 s | 24 |
| rowmax claim recompute | 3.9 s | 25 |
| rescale-family claim recompute (99 instances) | 5.9 s | 99 |
| fc claim recompute | 1.3 s | 63 |
| everything else (rmsnorm/glu/rope/slice/merge/edges) + registration hashes | ~3.0 s | — |
| serve-worker spawn (12 CUDA inits) | 0.04 s* | 12 |

*workers spawn lazily while the registration step runs; marginal cost ≈ 0.3 s each,
12 total, overlapped.

## 2. Where the 383 s actually went (the quantification the task asked for)

Profiled with the new env-guarded `ZKOB_PROF=1` laps (closing §1.3's per-phase gap):

- **~285 s: per-row 1-thread G1 host loops** — NOT subprocess overhead:
  - zkob_rescale affine link `g1_eq(X[j], sf·Xr[j] + rem[j])`: **1.70 s of its
    1.99 s/call** (1024 rows × one-thread `k_g1_mul` round-trips) × 99
    rescale-family instances ≈ 168 s;
  - glu/softmax8 `comb[j] = G[j] + r·S[j]` loops: **4.46 s of softmax8's
    4.62 s/call** (full 256-bit challenge ⇒ ~4.3 ms per 1-thread G1 mul) × 26
    instances ≈ 117 s.
- **~70 s: subprocess floor** — ~0.29 s/process (CUDA init + binary load; fc claim
  verify measured 0.31 s as a process vs 0.018 s in-pool) × 241 invocations.
- **~28 s: irreducible recompute** (lookup/sumcheck round checks, table folds,
  absorbs, batchopen) — matching §1.3's "~30–60 s unchanged" estimate.

So the design's single-process verifier alone would have landed at ~310 s; the §6
contingency ("Stage C verify > 100 s → fast-helpers on round-check host loops
before declaring") was triggered and BOTH fixes were built.

## 3. Single-process verifier — `serve` transport (design §1.3/§2.7/§6)

Constraint discovered: the naive "link all driver verify routines into one binary"
is blocked by the shared headers' `KERNEL = extern "C" __global__` kernels — every
driver TU defines the same C-linkage device symbols, so a multi-TU link collides;
and a dlopen/.so route would deviate from the pinned non-PIC upstream object set.
Chosen shape, preserving the pinned build line AND the validated artifacts:

- **`zkob_serve.cuh` (NEW header)** + a 3-line `main` wrapper in every driver: the
  old `main` body became `static int zkw_run1(int, char**)` verbatim; `<driver>
  serve` reads one request per line from stdin (the argv tail of a normal one-shot
  invocation), runs it through the SAME `zkw_run1` entry — byte-identical FS
  schedules, checks, verdicts — and prints a `ZKW-RC <rc>` sentinel after flushing.
  Exceptions → rc=3 (one-shot would have died nonzero; both are crash, fail-closed).
- **`common.DriverPool`**: verify_walk keeps ONE serve process per driver binary
  alive for the whole walk (12 CUDA inits instead of ~241), holding the GPU lock
  for its lifetime, issuing requests strictly in plan order — so the vacc
  claims.bin append order stays canonical (claims_match is order-sensitive by
  design). Fail-closed: crash/timeout/unexpected-rc raises (caller records REJECT);
  NO automatic retry (a claim-mode re-run would double-append into the verifier
  accumulator); a dead worker is respawned only for the next request.
  `verify_walk.py --no-pool` keeps the old one-process-per-invocation transport.
- The per-driver binaries remain THE artifacts: selftests exercise the same
  binary+entry that production serves (no separately-compiled mega-binary whose
  device code would need its own -dlto revalidation).

## 4. Fast-helpers — `zkob_fastg1.cuh` (NEW header; the §6.2 contingency)

The hot loops were per-row **G1** ops, where the documented -dlto rule forbids new
batched G1 kernel shapes. The helper therefore launches the SAME long-validated
1-thread kernels (`k_g1_mul`, `k_g1_addsub` from vrf_common.cuh — untouched)
CONCURRENTLY across a 128-stream pool: same compiled device code, same per-element
inputs and h_add/h_mul/g1_eq argument order, bit-identical per-element results —
pure scheduling, zero new kernel shapes, no protocol or algebra change.

- `hb_addmul(a, r, b, mul_first, out)` — replaces the rescale affine-link combo
  and the glu/softmax/softmax8 comb loops (prove AND verify sides — com_comb.bin
  bytes stay identical across sides, checked below);
- `hb_neq_count(x, y)` — replaces the per-row `g1_eq` count (exact z==0 semantics).
- Validation, house style: every call cross-checks two sample rows byte-exact
  against the sequential helpers (throws on mismatch, fail-closed);
  `ZKOB_SLOW_G1LOOP=1` forces the original sequential loop outright; slow-vs-fast
  claim files compared **byte-identical** on a real obligation; all driver
  selftests + the at-scale ZKOB_FOLD_CROSSCHECK=1 re-verify of all 7 sub-batches
  pass.
- Effect: rescale affine link 1.70 s → 0.026 s (65×); softmax8 verify 4.62 s →
  0.23 s; prove side gains too (636 → 522 s, the comb loops in softmax8/glu prove).
- Also new: fine-grained `ZKOB_PROF=1` laps in rescale/softmax8/glu verify
  (load_coms / affine_link / absorbs / rounds_loop / table_fold / comb_loop / …).

## 5. T4 — tamper localization (the 3 failing phases, root-caused and fixed)

Root cause was ORCHESTRATOR GATING, not the protocol: verify_walk gated every
`*.commitment_opening` discharge on the blanket `batch_ok`, so ANY sub-batch
failure dragged all 16 opening ids (+ statement.prompt_binding via drivers_ok)
out of `checked`, regardless of which claim failed.

Fix (verify_walk.py + common.localize_batch_failure; the soundness verdict is
UNCHANGED — overall ACCEPT still requires every sub-batch, F12):

1. **On sub-batch failure, localize**: multiset-diff the prover's claims.bin
   against the VERIFIER-recomputed list on the canonical per-claim record bytes
   (F4 encoding). A diverging record names exactly the offending obligation:
   a tampered absorb diverges that driver's recomputed points; a driver that
   rejected driver-side never emitted (its claims appear prover-side only); a
   forged prover record appears prover-side only. Diverging claim ids map
   obid→mid through the claim plan and ONLY those ids are failed, with the named
   reason `opening_batch b<k>: recomputed claim(s) diverge …`.
2. **Byte-identical claim lists** (e.g. batch_vfin / batched-IPA / sumcheck-bytes
   tampers, or an SZ-caught false-claim batch): the failure is
   information-theoretically not attributable to a single id — the named locus
   stays the batch check itself (`opening_batch.terminal` / `.ipa<G>` /
   `.round0`), exactly the C1 §3 relocated-locus discipline, and NO driver id is
   blamed. This is genuinely as tight as attribution can be: with witnesses
   freed after batch prove (by design), a false-but-well-formed claim kills only
   the batch identity. (The selftest's vfin expectation already codified this.)
3. **commitment_opening discharge** now follows its matmul's detail (which
   includes any batch-localized blame on its W claim), not blanket batch_ok.

Selftest results at the previously-failing loci (all in the 15/15 run):

- slice tamper (com_KhT05): REJECT, `rejected` layer ids == exactly
  `[layer0.attn.scores_matmul]` — driver-side transcript divergence
  (fc.h05 sumcheck round 1) AND b0 claims_match localization both name it.
- com_mx tamper: REJECT localized to `layer0.attn.softmax` with BOTH routes
  (rowmax.h05 driver divergence + edge RM2.h05 byte-equality) — and no
  bystander ids.
- batch_vfin tamper: REJECT at `opening_batch.terminal` (b0) while every driver
  verdict stays conditional-ACCEPT and NO layer/final/lm_head/logit id is
  rejected — the F12 gating case, now with clean attribution.

No selftest expectation was loosened; the previously-failing assertions pass
as originally written.

## 6. Files changed (all copied to /workspace/projects/zk-hillclimb/)

- **NEW** `zkob_fastg1.cuh` (stream-pool fast-helpers), `zkob_serve.cuh` (serve
  transport). Neither is a protected header; zkob_claims.cuh NOT touched.
- All 12 driver .cu: `main` → `zkw_run1` + serve wrapper (mechanical, identical);
  zkob_rescale/zkob_glu/zkob_softmax/zkob_softmax8 additionally: fastg1 loop
  replacements + ZKOB_PROF laps. Rebuilt with the pinned line (sm_89 -dc -dlto,
  -dlto link, standard upstream objects); **all 12 selftests ALL PASS**.
- orchestrator/common.py: DriverPool; parse_claims now returns raw record bytes;
  localize_batch_failure.
- orchestrator/verify_walk.py: --no-pool flag; pooled drive() for driver
  verifies + batchopen + skip edges; per-sub-batch failure localization;
  commitment_opening gating fix.
- selftest.sh: UNCHANGED. Protected headers: UNCHANGED (sha256-verified).

## 7. Honest notes / residuals

- The remaining 27 s ≈ 7.6 s batchopen + ~17 s per-driver round-check recompute
  + ~2 s registration/edges/python. Further cuts (e.g. the rope 0.06 s→? or
  batchopen's per-batch 1.1 s) are possible but now inside the design's 40–85 s
  window with margin; not pursued.
- DriverPool deliberately does not retry a crashed request (double-append risk
  into the accumulator); the old transport retried rc≥2 once. Crash ⇒ that check
  records REJECT (fail-closed) and the worker respawns for the next request.
- Localization consumes the prover's claims.bin only for the byte-diff against
  the verifier-recomputed list (never enters any computation), so a malicious
  claims.bin can at worst mis-describe WHICH id is blamed in an
  already-REJECTed transcript — it cannot flip a verdict.
- Prove-side wall (522 s) still includes ~70 s of one-process-per-invocation
  overhead; prove_walk was intentionally left on the old transport this stage
  (risk containment). Wiring it through DriverPool is a known cheap follow-up.
- The §13 campaign items beyond T4/T5 (audit walkthroughs T6, S1 proof-size
  re-layout, harness FORGERIES B/C) remain for the coordinator's schedule.
