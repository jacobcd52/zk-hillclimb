# STAGE_B_REPORT — fold flattening, me_weights GPU lift, rescale claim mode, two-driver chain

Status: DONE, 2026-06-12. Implements TRANSPORT_REBUILD_DESIGN §6 Stage B (gate T3)
plus the two Stage-A-flagged performance items and the Stage-A comref-canonicalization
flag. All sources copied to `/workspace/projects/zk-hillclimb/zkllm-src/`. No git
commits (coordinator commits). **vrf_common.cuh / zkob_lookup.cuh / fs_transcript.hpp
untouched** (diff-verified against the canonical zkllm-src copies), so the §13
all-driver re-validation rule was NOT triggered. Pinned sm_89 `-dc -dlto` compile +
standard link list for every rebuilt binary.

## 1. What was built

| file | status | what |
|---|---|---|
| `zkob_claims.cuh` | **EDITED** (Stage-A-new header — see §5.1 flag) | (a) `k_bo_rowweights` + `bo_batched_group_fold`: the G5 fold FLATTENED to one Fr-weight launch + one `dev_msm` per domain group; (b) `k_bo_pp_expand` + `bo_fast_me_weights` / `bo_fast_s_vector` / `bo_fast_ipa_verify` (the rowmax §2.8 pattern, `bo_`-renamed): batch IPAs freed of the G·logG host loops, prove and verify side; (c) both new shapes added to the runtime `bo_probe_kernels()` -dlto probe (element-exact vs h_scalar brute force); (d) env switches `ZKOB_SLOW_FOLD` / `ZKOB_SLOW_IPA` (pre-Stage-B paths kept selectable) and `ZKOB_FOLD_CROSSCHECK` (per-tensor batched-fold == fold_chain element-exact check, THROWS on divergence; ON in every selftest); (e) `ZKOB_REQUIRE_RELATIVE_COMREF` policy check in batch_verify's shape phase (comref canonicalization, §5.3) |
| `zkob_batchopen.cu` | EDITED | selftest sets `ZKOB_FOLD_CROSSCHECK`; `ZKOB_EVIL` env exposed on the prove CLI (selftest/battery-only forgery construction at pair scale; mode 0 in production) — **33/33 ALL PASS** |
| `zkob_fc.cu` | EDITED (selftest setup only) | selftest sets `ZKOB_FOLD_CROSSCHECK` — **ALL PASS both modes** (3 old + 3 claim cases, 9/9 each) |
| `zkob_rescale.cu` | **EDITED (tail only)** | claim mode behind `--claims` (runtime flag; old inline-IPA tail intact and DEFAULT): prove emits the A/rem/m lookup-terminal claims at the exact old `open_prove` sites (order A, rem, m) + witrefs + drvstate; verify keeps the affine link, all rounds, the recomputed B_f/T_f and the terminal identity byte-identical, then recomputes the 3 claims from its own FS replay into the verifier accumulator → ACCEPT-conditional. ZKOB_PROF lap timers added. Dual-mode selftest — **ALL PASS** (3 old cases unchanged + 3 claim cases 10/10 each, incl. the semantic out-of-range-rem covert-channel case still dying DRIVER-side) |
| `vrf_toy_batchopen.cu` | EDITED (setup only) | sets `ZKOB_FOLD_CROSSCHECK` — **TOY-BATCHOPEN: ALL PASS** (all pins P1–P9 re-run after the header edit) |
| `pair_walk.py` | NEW | the two-driver chain through the orchestrator conventions (§3) |
| `pair_battery.py` | NEW | BO-1..BO-12 + F3/F10 + F8 split + cross-driver drop + chain-edge tamper at pair scale (§4) |

## 2. The two kernel fixes — verify split before/after (the §6 measurement gate)

RTX 4090. "Before" = the SAME binary forced onto the Stage-A code paths
(`ZKOB_SLOW_FOLD=1 ZKOB_SLOW_IPA=1`) over the SAME batch artifacts and claims —
apples-to-apples; three runs of the new path stable to ±3 %.

Pair batch = 12 claims, 12 tensors, 3 domains (gen1024 ×2 tensors, gen4096 ×5,
gen32768 ×5; m_max = 25; four of the tensors are full 2^25 lm_head-class).

| `batch_verify` phase | Stage-A path | Stage-B path | win |
|---|--:|--:|--:|
| fold_1024 (2 tensors) | 0.112 s | 0.009 s | 13× |
| fold_4096 (5 tensors) | 0.258 s | 0.010 s | 26× |
| fold_32768 (5 tensors) | 0.222 s | 0.009 s | 25× |
| ipa_1024 | 0.411 s | 0.116 s | 3.5× |
| ipa_4096 | 1.590 s | 0.133 s | 12× |
| **ipa_32768** | **14.92 s** | **0.162 s** | **92×** |
| checks (claims_match+shape+absorb+rounds+terminal) | 0.06 s | 0.06 s | — |
| **total** | **17.6 s** | **0.78 s** | **22.5×** |

- **Fold flattening:** the per-tensor fold_chain cost (34–56 ms/tensor, launch-latency
  bound — re-measured here at 44–56 ms) is GONE; the whole fold pass is now ~28 ms for
  12 tensors and scales as one Fr launch + one MSM per domain. At faithful-arch-v1 scale
  (1,242 tensors, ≈1.18 M total commitment rows ≈ a 2^21-point MSM split across 4
  domains) the projection is **~1–2.5 s** for the entire G5 fold pass — vs the ~45–65 s
  the Stage-A path would have cost. The §2.7 "RLC + folds 1–3 s" row is back on its feet.
- **me_weights lift:** the gen32768 IPA fell 15.16 s (Stage A T2) / 14.92 s (re-measured
  here) → **0.162 s**. All four-domain IPAs together now cost ≈ 0.5 s/run. The prove side
  got the same lift free: batch-prove `ipa` lap 8.5 s (Stage A, lm_head only) → **1.39 s**
  (this run, with MORE gen32768 tensors).
- **Conventions held:** `ZKOB_FOLD_CROSSCHECK=1` at pair scale — every tensor's batched
  fold == `h_mul(fold_chain(com_j, r_rows), coef_j)` element-exact (G1 point equality),
  group RLC equal, AND the full slow path (`fold_chain` + header `ipa_verify`) reaches the
  same ACCEPT on the same artifacts. `bo_fast_me_weights`/`bo_fast_s_vector` probed
  element-exact vs the slow host loops at startup of every toy/selftest run (the
  k_bo_rowweights probe covers level counts {0,2,5} incl. the single-row tensor edge).
  No new G1 kernel shape exists: the G1 side of the batched fold is the long-probed
  `dev_msm` (`k_g1_scale`/`k_g1_add_pairs`); zero-scalar padding verified safe
  (`G1Jacobian_mul(·, 0)` = identity by construction, g1-tensor.cu:445).

## 3. T3 — the two-driver chain through the orchestrator conventions

`pair_walk.py` runs the REAL validated pair shapes from the faithful-arch-v1 walk,
against the official registration (`/root/zkorch/stage3v2-fa/registration`, symlinked)
and the official chained activations as inputs:

- `layer0.mlp.gate_proj.matmul` → `.rescaling` (B=1024, 768×3072, gen4096, sf 2^20;
  X = the official post_attn_norm output)
- `lm_head.matmul` → `.rescaling` (B=1024, 768×32000, **gen32768**, sf 2^16;
  X = the official final_norm output) — the stress instances the work item names

Orchestrator conventions mirrored exactly: `run_seed = sha256(public.json bytes)`
(a pair statement pinning registration + input hashes), per-obligation transcript seed
`f"{run_seed}:{obligation_id}"` (verify_walk.py:210 convention), registration gens with
**one gen file per domain size**, chain edges = the byte-equality edges from common.py
("gate_fc com_Y == gate_rescale com_X"). All four driver runs emit into ONE prover
accumulator; ONE `zkob_batchopen` discharges all 12 claims; the verify side re-runs all
four drivers (fresh verifier accumulator), checks both chain edges, asserts the
gen-registration invariant + 12-claim count + relative comrefs, then batch-verifies.
Verdict gating (F12-style, harness-level as in the selftests): ACCEPT only if 4×
ACCEPT-conditional ∧ 2 edges ∧ opening_batch.

**Honest pair: ACCEPT.** Wall times:

| step | prove | verify (claim mode) | verify (OLD tail, same data) |
|---|--:|--:|--:|
| gate fc | 2.1 s | 0.36 s | 3.69 s (ipa_tail 3.39) |
| gate rescale | 3.7 s | 2.04 s | 6.37 s (ipa_tail 4.38, rounds 1.70) |
| lm_head fc | 14.5 s | 0.30 s | 28.49 s (ipa_tail 28.19) |
| lm_head rescale | 29.8 s | 2.06 s | 43.03 s (ipa_tail 41.03) |
| opening_batch | 7.4 s | 0.78 s | — |
| **subset total** | 57.5 s | **5.5 s** | **81.6 s** → **14.8×** |

The per-driver verify residual is round checks + process overhead: lm_head rescale's
2.06 s is 1.71 s of (unchanged-by-design) lookup round checks + B_f/T_f recompute +
affine link; fc's 0.30 s is almost entirely subprocess/CUDA-init floor. The inline IPA
tails (77.0 s of the old subset's 81.6 s = 94 %) collapsed into the 0.78 s batch.

Batch artifacts: claims.bin 9.9 KB, batch_sumcheck 2.4 KB, batch_vfin 392 B, 3 ipa files
2.9–4.4 KB — the predicted ~50–60 KB/run class holds with 9 KB of claims for 12 claims
(scales to ~0.6 MB at 1,535 claims; fine).

### Prover batch-sumcheck overhead (the §2.2 unmeasured flag — now measured)

`batch_prove` at pair scale (m_max = 25; four 2^25 P̂/M table pairs + eight smaller;
~16 GB peak VRAM, fits): **7.4 s total** = setup 5.01 (witness loads + per-claim eq
builds + M accumulation) + **rounds 0.61** + terminal 0.02 + ipa 1.39. The G2 sumcheck
rounds themselves are nearly free; setup (eq expansion ∝ Σ_i 2^vars_i) dominates.
Linear extrapolation to the full walk (Σ_i 2^vars ≈ 1.6 G vs ≈ 0.15 G here): setup
≈ 50–55 s, rounds ≈ 4–6 s, IPAs ≈ 2–3 s ⇒ **batch prove ≈ 60–70 s ≈ +6 % of the
1,062 s prove wall**, well inside the design's ±20 % flag — BUT full-scale round-0
residency (Σ ≈ 38 GB as Fr) still exceeds the 24 GB VRAM, so the §2.2 streaming plan
(or its 2–4-sub-batch contingency, sound per Lemma-3-per-batch) remains a Stage-C item.
Per-driver prove in claim mode is cheaper than old-tail (lm fc 14.5 s vs Stage A's
21.4 s old / 14.3 s claim — reproduced).

## 4. BO forgery battery at pair scale — 38/38 ALL PASS

`pair_battery.py` re-runs the battery against the real-scale pair run; every case
rejected by EXACTLY the named check (locus parsed from the REJECT line):

| case | locus (expected == got) |
|---|---|
| honest full pair verify / every restore (6×) | accept |
| **BO-1a** false eval (gate fc X), honest-procedure batch (`ZKOB_EVIL=1`) | round0 (F8) |
| **BO-1b** fully adaptive prover (per-round p1:=cur−p0, forged v'; `ZKOB_EVIL=2`) | ipa32768 (the last tensor's group — F8) |
| **cross-driver claim-drop** (rescale's 3 claims omitted, fc's still present) | claims_match |
| BO-2 single claim omitted / BO-3 comref swap (gate A↔rem, list only) / BO-10 duplicate / reorder | claims_match |
| BO-4 ρ from doctored list, honest list shipped (`ZKOB_EVIL=3`) | round0 |
| BO-5 round-0 p(1) / round-1 p(0) tampers | round0 / round1 |
| BO-6 vfin tamper | terminal |
| BO-7 vars lie (junk-padded point, BOTH lists) | shape |
| BO-8 batch_sumcheck n_claims field | xcheck |
| BO-8 a_final tampers on ALL THREE group IPAs + round-0 L | ipa1024 / ipa4096 / ipa32768 |
| BO-8 claims.bin id byte (file never parsed — F5) | claims_match |
| BO-8 verifier-side drvstate divergence | round0 |
| **F10/F3** com_A extra trailing row / truncated com_A | shape — before any fold, no segfault, at REAL row counts |
| BO-9b batch artifacts proven under a different run_seed | round0 (full-replay variant is toy/selftest-covered) |
| BO-11 substituted com file post-prove (self-consistent bytes) | round0 (G0 comref-hash divergence) |
| BO-12 missing domain-4096 group IPA | group_missing |
| Committed EvalVar claim / n_claims == 0 | evalvar / empty |
| absolute comref under the relative-comref policy | shape |
| **chain-edge tamper**: gate rescale RE-PROVEN honestly over a doctored Y (+sf on one element, stays in range) — all four drivers ACCEPT-conditional, batch would pass its own claims | **the chain-edge byte-compare is the named catcher** ("gate_fc com_Y != gate_rescale com_X") |
| at-scale convention checks: `ZKOB_FOLD_CROSSCHECK` accept; slow-path same verdict | accept |

The chain-edge case is the one the batch CANNOT catch by construction (each driver's
claims open its own files honestly); it pins that the orchestrator edge check stays
load-bearing post-rebuild, exactly as §2.4 claims.

## 5. Flags / deviations (honest list)

1. **`zkob_claims.cuh` was edited — FLAGGED LOUDLY as required.** This is the
   Stage-A-NEW header, NOT one of the three protected shared headers (those are
   diff-verified untouched, §13 all-driver rule NOT triggered). Its include set is
   exactly {vrf_toy_batchopen, zkob_batchopen, zkob_fc, zkob_rescale}; per the
   header-edit discipline **all four TUs were rebuilt and their full selftests re-run:
   ALL PASS** (toy pins P1–P9; batchopen 33/33; fc 3+3 cases; rescale 3+3 cases). The
   work item's "prefer driver-local to batchopen" was not literally satisfiable: the
   IPA call sites being lifted live in `batch_prove`/`batch_verify`, which are in this
   header by Stage-A design. New kernels are Fr-only; no new G1 kernel shape (work-item
   1's "confirm no G1 kernel": confirmed — G1 work routes through the already-probed
   dev_msm shapes).
2. **Comref canonicalization is convention + a policy check, not a format change.**
   Drivers and batch are invoked with cwd = run dir and RELATIVE paths (the harness
   does this), so claims.bin carries run-dir-relative comrefs on both sides;
   `ZKOB_REQUIRE_RELATIVE_COMREF=1` (set by the harness) makes batch_verify REJECT
   (`shape`) any absolute comref. Selftests keep absolute /tmp paths (flag unset).
   Stage C should set the flag in verify_walk and keep invoking drivers run-relative.
3. **One-gen-file-per-domain-size** is asserted harness-side (genspecs built ONLY from
   the registration's `gen<G>.bin`, claim domains parsed from claims.bin and checked
   ⊆ registration sizes, exactly one file per size) on top of the existing batch-CLI
   size binding. The Stage-C orchestrator should carry the same assertion (one line).
4. **The dual-accumulator routing (§4.4 item 2, "weight-tagged claims routed by comref
   from Stage B") was NOT built.** It is not in this stage's work-item list and the
   pair harness has a single accumulator; the format already carries the EvalVar tag
   and the comref-based routing is a small orchestrator-side change with no driver
   impact. Deferred to Stage C wiring — flagged so it isn't silently lost.
5. **gen64 not exercised at pair scale** (no gen64 tensors in these pairs; it is the
   smallest domain and is covered by toy/selftest multi-domain cases). BO-9a full
   cross-run replay and the RLC-cancellation case were run at toy/selftest scale only
   (deterministic transcript-divergence paths identical at any scale); BO-9b
   foreign-seed batch artifacts WAS run at pair scale.
6. **G3 terminal host loops are the next latency item** at full scale: `bo_Mj_at_r` is
   ~3·vars h_scalar round-trips per claim (0.019 s for 12 claims here → ~3–5 s at 1,535
   claims), plus G0's per-claim absorbs. Inside the §2.7 "2–6 s" row but worth a flat
   kernel in Stage C if the budget tightens. Same class: rescale's 1.7 s round checks
   per instance (the §2.7 "round checks unchanged" row, 15 instances ≈ 26 s of the
   full-walk budget — the §6.2 fast-helpers-on-round-checks stretch item).
7. **`ZKOB_EVIL` on the batchopen prove CLI** mirrors the header's selftest-only evil
   modes (Stage A deviation 6); production callers don't set it. The orchestrator
   should never propagate it (Stage C: assert unset in prove_walk).
8. Battery scratch (`battery_bak/`, `vacc*`) left in the run dir for audit; witness
   .fr files (≈4.3 GB) must persist for batch re-proves — delete `/root/zkorch/pairB`
   wholesale when Stage C supersedes it. Disk on / is at 93 %.

## 6. Gate verdicts

- **T3: PASS.** Two-driver fc→rescale chain (incl. the lm_head gen32768 instances)
  through one accumulator + one batch ACCEPTs honestly with both chain edges; the BO
  battery at pair scale is 38/38 with every forgery dying at its named check, incl. the
  F3 trailing-rows case and the F8 BO-1a/BO-1b locus split; batched-fold and
  fast-me_weights/s-vector cross-checks element-exact (probes + at-scale fold
  cross-check + slow-path verdict agreement); fc AND rescale selftests pass both modes;
  toy + batchopen selftests re-validated after the header edit.
- **PCS go/no-go (§2.5 contingency): NO-GO — stay on batched-IPA (a).** The decision
  threshold was "four batched IPAs + RLC folds > 30 s despite GPU-side helpers". With
  (1)+(2) landed, measured: all group IPAs ≈ 0.41 s + folds ≈ 0.03 s at pair scale;
  full-walk projection ≈ 0.6 s IPAs + 1–2.5 s folds ≈ **2–3 s total — an order of
  magnitude under the threshold**. The single batched IPA per domain is decisively NOT
  the post-rebuild bottleneck; what remains of the verify budget is per-driver round
  checks (~30–60 s) and the 234-subprocess overhead (~80–150 s), i.e. the
  single-process `zkverify_walk` (§2.7) is the operative Stage-C item, exactly as the
  design ordered. HyperKZG would buy nothing measurable; rejected per §2.5's standing
  reasoning (trusted setup, second curve, FFI into the -dlto environment).
- **§2.7 budget reachability:** on this subset old 81.6 s → new 5.5 s (14.8×), with the
  batch at 0.78 s and per-driver tails at 0.3 s (fc) / 2.0 s (rescale, round-check
  bound). Projected full walk: ~3–8 s batch (incl. G3/absorb growth) + ~30–60 s round
  checks + subprocess overhead → **with the single-process verifier the 10–60 s window
  now looks reachable from the MIDDLE, not the top** (the two items that were eating
  the budget are measured dead: 1,535 inline IPAs and the fold/host-loop costs).

Next (Stage C per §6): remaining 9 drivers' tails; orchestrator `opening_batch` id +
conditional-verdict gating (F12) + claims_match wiring + relative-comref flag + the
one-gen-per-domain and ZKOB_EVIL-unset assertions; single-process zkverify_walk; S1
canonical-affine + dedupe; batch-prove streaming (or 2–4 sub-batches) for the 38 GB
round-0 residency; dual-accumulator routing (§5.4); full §13 selftest + audit campaign.
