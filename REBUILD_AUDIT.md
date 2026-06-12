# REBUILD_AUDIT — external-style soundness audit of the AS-BUILT batched-opening + weight-privacy code

Status: AUDIT, 2026-06-12. Independent adversarial review of the **as-built implementation**
of the transport rebuild (Stages A–D), against the audited design (TRANSPORT_REBUILD_DESIGN.md
§2/§4) and the prior design audit (TRANSPORT_REVIEW.md, required pins F3–F6 + BO‑1..BO‑12 +
F7). The design was already found SOUND-AS-DESIGNED; this pass asks the narrower, sharper
question: **does the code match the audited design, are the required pins actually present in
the code path that runs, and does the ZK/weight-privacy property hold against a cheating
prover the verifier accepts.**

Method: full read of zkob_claims.cuh, zkob_wpriv.cuh, zkob_batchopen.cu, the claim-mode and
wpriv tails of zkob_fc.cu / zkob_rmsnorm.cu, the orchestrator (verify_walk.py, common.py,
prove_walk.py, wpriv_leak_scan.py); confirmation that the three protected headers are
unmodified; running the as-built selftests on this box (RTX 4090); and constructing my own
adversarial cases under `/tmp/audit2/`. Code citations use `file:line`. "As-built" = the
deployed tree `/root/zkllm` (see F-1 on a repo/build drift that does not affect the protocol).

---

## VERDICTS

| category | verdict |
|---|---|
| **(a) batching** (claim accumulation + one batched opening) | **SOUND** |
| **(b) weight privacy** (hiding registration + hidden weight claims + ZK weight sub-batch) | **SOUND** (two leaks remain; both are documented *statement-layer* residuals, not proof-layer holes, and are correctly confined) |
| **(c) localization** (the C2 per-id blame fix) | **SOUND** |

No cheating prover was found that the as-built verifier accepts. Every required pin (F3–F6),
the RLC/sumcheck/IPA binding, the weight-privacy hiding+binding, and the per-id localization
are present **in the code path that actually runs**, and reject every forgery I constructed at
the named locus. Two process-level **observations** (F-1 repo/build drift; F-2 the
fast-kernel cross-check is selftest-only) are raised below: neither is a soundness hole, both
are worth closing before relying on the system.

The three protected headers are byte-identical to their Phase-0 trusted baseline (commit
`7d816a1`) and to the deployed copies — confirmed by `git diff` and `cmp`:
`vrf_common.cuh`, `zkob_lookup.cuh`, `fs_transcript.hpp` — **unmodified**. The
int-model-approximation was not touched.

---

## What I ran (as-built evidence)

All on the deployed `/root/zkllm` binaries (built 2026-06-12):

- `zkob_batchopen selftest` → **33/33 ALL PASS** (BO‑1a/1b, BO‑2..BO‑12, F10 structural,
  EvalVar/empty edges).
- `zkob_batchopen wselftest` → **26/26 ALL PASS** (D1/D2/D3 honest, D4 mini leak scan +
  positive control, WBO‑1a/1b, blind-bookkeeping lie, byte tampers, BO‑13a/b routing).
- `zkob_fc selftest` → **ALL PASS** (3 legacy + 3 claim-mode 9/9 + 2 wpriv 15/15; D4
  claim_W scan CLEAN + plain-mode positive control).
- `zkob_rmsnorm selftest` → **ALL PASS** (incl. 2 wpriv 11/11; D4 val_g scan CLEAN; the
  documented `val_g = val_W/val_R` residual asserted explicitly).
- My own `/tmp/audit2` cases, run through the **production** verify path (envs
  `ZKOB_FOLD_CROSSCHECK`/`ZKOB_BATCH_SELFCHECK`/`ZKOB_EVIL` explicitly unset):
  - tamper `batch_vfin.bin` → `REJECT[opening_batch.terminal]`;
  - tamper `ipa_batch_8.bin` a_final → `REJECT[opening_batch.ipa8]`;
  - tamper the real comref-target commitment (`com_A.bin`, two byte positions) →
    `REJECT[opening_batch.round0]` (the G0 comref-SHA‑256 absorb binds the file bytes);
  - `localize_batch_failure` on a one-byte claim tamper → implicates exactly `drv.C:c0`;
    on a dropped middle claim → implicates exactly `drv.B:c0`.

---

## (a) BATCHING — SOUND

### F3 (covert-channel pin) — PRESENT, runs before any fold, for every tensor

`zkob_claims.cuh:1091-1100` (`batch_verify`): the per-distinct-tensor loop loads each
commitment file and checks `coms.back().size != t.n_rows` **before** the G0 absorb and before
any `fold_chain`/`bo_batched_group_fold`. `t.n_rows` is itself forced to equal `2^{vars-logG}`
by the structural check in `derive_tensors` (`zkob_claims.cuh:279-280`,
`c.n_rows != (1u << (vars - logG))`), and the fold consumes exactly `vars-logG` row
challenges (`u_row = r[logG..vars)`), so there are **no rows the fold ignores** — the original
F3 attack (trailing prover-chosen rows silently dropped by `fold_chain`) is closed at the new
location. The weight path repeats the identical check (`zkob_wpriv.cuh:739-748`).

Runtime confirmation: the public selftest's F10 cases (`zkob_batchopen.cu:307-321`) — com file
with an extra trailing row, and a truncated com file — both `REJECT[opening_batch.shape]`
before any fold (observed). BO‑7 (a `vars` lie consistent in both lists) also dies at `shape`.

A cheating prover that pads a registered/activation commitment with extra prover-chosen bytes
gets a row-count mismatch (deterministic REJECT), so the zero-advice/capacity argument is not
reopened. **No hole.**

### F4 (absorb encoding) — PRESENT

`claim_blob` (`zkob_claims.cuh:104-116`) length-prefixes `id`, `comref`, and `point`, and
writes the `EvalVar` tag as an explicit byte before the eval/ceval — two different claims
cannot serialize to the same bytes. The same blob feeds **both** `claims.bin` and the G0
absorb (`batch_absorb_g0`, `zkob_claims.cuh:309-332`), so the byte-compare and the transcript
bind the identical encoding.

### Claims absorbed before the RLC challenge; verifier recomputes the list — PRESENT

`batch_absorb_g0` absorbs `n_claims`, every claim blob, every distinct comref + its
SHA‑256 file hash, and every drvstate, and only then is `rho` squeezed
(`zkob_claims.cuh:1118-1120`). Order: G0 ≺ ρ ≺ rounds ≺ vfin ≺ ρ' ≺ per-domain IPAs — each
challenge after the message it binds. `M_j(r)` and `kappa_j` in the G3 terminal are
**verifier-computed** from `(rho, points, r)` (`bo_Mj_at_r`/`bo_kappa`,
`zkob_claims.cuh:747-767`, used at `1161-1167`); the only prover input to G3 is `vfin`, bound
before ρ′. The per-domain IPA binds the RLC of the row-folded commitments to `v*_g`
(`zkob_claims.cuh:1196-1243`). This is exactly the audited §2.2 reduction; the algebra was
re-checked against the design's Lemmas 1–5 and matches.

The cancellation attack (choose evals so `Σ δ_i ρ^i = 0`) is closed because every eval feeds ρ
through two SHA‑256 paths (the claim blob in G0 and the driver transcript → drvstate in G0),
so ρ cannot be known before the evals are fixed. Runtime: BO‑4 (ρ derived from a doctored
list, honest list shipped) → `round0`; the explicit RLC-cancellation selftest → `round0`.

### F5 (claims_match is load-bearing and inseparable) — PRESENT

`batch_verify` takes the **verifier-recomputed** list (`vaccdir/claims.bin`) as its
computation input (`zkob_claims.cuh:1055`); the prover's `claims.bin` is read **only** for the
byte-compare (`1062-1070`) and is never parsed into the protocol. `claims_match` is a REJECT
locus of `opening_batch` itself, not a separable orchestrator check. The verifier builds
`vacc` by re-running each driver in canonical plan order (`verify_walk.py:324-337`,
`common.py:claim_plan`), and `prove_walk.py:168-172` asserts the prover walks the identical
plan order — so a claim cannot be silently dropped, reordered, or smuggled across sub-batches.
Runtime: BO‑2/3/10/reorder all → `claims_match`.

### F6 (drvstate provenance; redundant fields; empty edge) — PRESENT

drvstates are loaded from `vaccdir` (`zkob_claims.cuh:1057`) — the verifier's own per-driver
runs, never a prover artifact; under the single-process pool this is structural (the same
process writes and reads vacc). `n_claims == 0` REJECTs explicitly (`1058`). The redundant
`n_claims`/`m_max` header fields in `batch_sumcheck.bin` are cross-checked against the derived
values (`1130-1135`, locus `xcheck`). Runtime: BO‑8 n_claims field tamper → `xcheck`;
verifier drvstate divergence → `round0`.

### Registered-weight comref discharge — PRESENT

The W-claim that discharges each `*.commitment_opening` id carries the **registered** file
path as comref (`zkob_fc.cu:376` non-wpriv, `:321` wpriv; `zkob_rmsnorm.cu:530/542`). The
orchestrator asserts every registered weight comref appears in the verifier-recomputed claim
lists (`verify_walk.py:506-544`, `common.py:registered_weight_comrefs`) — making the discharge
explicit rather than emergent, and computed from vacc (not trusting the prover).

### Fast-vs-slow fold/IPA — algebra equivalent; cross-check is selftest-only (see F-2)

`bo_batched_group_fold` + `bo_fast_ipa_verify` (the Stage-B GPU helpers) are cross-checked
element-exact against the per-tensor `fold_chain` and the header `ipa_verify` **when
`ZKOB_FOLD_CROSSCHECK` is set** (`zkob_claims.cuh:1204-1231`), and the new kernel shapes are
`-dlto`-probed in `bo_probe_kernels`. Both run in every selftest. They do **not** run in the
production verify (F-2). Soundness impact is low — see F-2 — and my `/tmp/audit2` tampers
confirm the production fast path still rejects forged vfin/IPA/commitment bytes at the correct
locus.

**Conclusion (a): SOUND.** Every pin is in the executed path; every constructed forgery is
deterministically rejected at its named locus.

---

## (b) WEIGHT PRIVACY — SOUND (with two documented, confined statement-layer residuals)

### Hiding commitment — real (blind included, independent H, hash-pinned)

`wp_hide_rows` adds `s_r·H` per row with fresh urandom blinds (`zkob_wpriv.cuh:109-129`,
`wp_rand` 61-77); `H` is q.bin slot 1 (`qg(1)`), an independent generator registered since the
Stage-A §4.4 touch. The orchestrator enforces that under `weight_privacy="hiding"` the
hash-pinned `q.bin` is the 2-slot `[Q, H]` file (`verify_walk.py:196-205`, `WPRIV_Q_SLOTS=2`)
— otherwise every hiding commitment would be vacuous; this is fail-closed before any driver.
The prover **recomputes** the hiding com from `(W, blinds)` rather than loading the registered
file (`zkob_fc.cu:197-205`, `zkob_rmsnorm.cu:276-282`), so a tampered/substituted registration
file diverges the transcript (driver REJECT), and the G0w comref-SHA‑256 absorb catches a
post-driver swap (wselftest: substituted re-blinded com → `wround0`; single-claim fc case →
`wround1`).

### Binding preserved under blinding

Pedersen with an extra independent generator H remains binding under DLOG: a second opening of
any row yields a nontrivial relation among `(g[], H)`. Weight rows appear in **no** affine
link (links are activation-side — rescale `com_X/X̂/rem`, rmsnorm `com_L`/`com_P1`/`com_P2`),
re-confirmed against the driver code, so blinds break no homomorphic identity; the batch fold
simply carries the folded blind on H. Hiding (each row uniform) and binding therefore both
hold.

### Masked rounds + ZK IPA prevent weight-eval leakage

The F7 mechanism is built as designed (committed round messages, **not** Libra masks):
`C_p0/C_p1/C_p2 = p_t·Q + τ_t·H` with `τ1 = τ_cur − τ0` and the homomorphic round check
`C_p0 + C_p1 == C_cur` (`zkob_wpriv.cuh:536-541, 788-795`; the fc driver sumcheck likewise,
`zkob_fc.cu:258-270, 462-469`). Committed terminals `C_vfin_j` with the homomorphic G3
`Σ_j M_j(r)κ_j·C_vfin_j == C_cur` (`zkob_wpriv.cuh:804-813`). Driver terminals are Schnorr PoKs
of an H‑component (fc `C_cur − claim_X·C_W`, `zkob_fc.cu:484-493`; rmsnorm
`val_R·C_val_g − val_W·Q`, `zkob_rmsnorm.cu:871-879`). The final IPA is the ZK variant: blinded
L/R, `β' = β + x²β_L + x⁻²β_R`, and a 2‑base Schnorr replacing the a_final reveal
(`zk_ipa_prove/verify`, `zkob_wpriv.cuh:275-352`). No plaintext round eval, no `claim_W`, no
`a_final` appears in any weight-path artifact.

### D4 leak scan — re-inspected, meaningful, CLEAN

`wp_leak_scan`/`wp_file_contains` byte-scan each verifier-consumed artifact for the 32‑byte
LE image of the hidden eval (`zkob_wpriv.cuh:885-903`); the walk-scale scanner
(`wpriv_leak_scan.py`) reads the secrets from the **prover-private** `data/wpriv/cblinds.bin`
(relocated out of `proofs/` by `prove_walk.py:131-136`) and scans the full visible surface
(run-root JSON, hash-pinned registration, all of `proofs/` and `vacc/`). I verified the scan
is **not vacuous**: every selftest carries a positive control that plants a known plaintext and
confirms the matcher finds it (`zkob_batchopen.cu:545-552`, `zkob_fc.cu:863-869`). Observed:
fc D4 CLEAN + positive control PASS; rmsnorm D4 CLEAN; wselftest mini-scan CLEAN + control.
A weight-MLE eval is therefore not recoverable from transcript, claims, ipa, or blinds.

### The documented residual is the ONLY proof-layer-adjacent leak, and it is confined

`val_g = val_W / val_R` is computable from **public** claims because `W = R⊗g` is a public
chain tensor: in wpriv mode rmsnorm still emits `val_W` (claim on its own `com_W.bin`) and
`val_R` (claim on `com_R.bin`) as **plain** public claims (`zkob_rmsnorm.cu:941, 956`), while
only `g` is Committed to the weight batch (`942-952`). This is a **statement-layer** leak
(the activation statement `Y = W∘X` already exposes g multiplicatively), not a symptom of a
broader hole, and it is **confined to the rmsnorm gain g**: for the matmul weights there is no
analogous public product tensor — only the layer‑0 public‑X claim functionals already
accounted in §4.1. The rmsnorm selftest asserts this residual explicitly and shows `val_g`
itself appears in no artifact. The candidate-confirmation residual (D5(i), guess-and-recompute
against deterministic activation commitments) is the third documented item; both are
threat-model limits requiring hidden activations, correctly out of scope.

### Routing forgeries closed

A plain claim in the weight batch → `wevalvar` (`zkob_wpriv.cuh:718-720`); a Committed claim in
the public batch → `evalvar` (`zkob_claims.cuh:1059-1061`); the orchestrator additionally
asserts the public sub-batches carry only `tag==0` and the weight batch only `tag==1`
(`verify_walk.py:376, 449`), and that no registered weight comref is opened PLAIN in a public
batch under weight privacy (`verify_walk.py:517-524`). Runtime BO‑13a/b both rejected as
specified.

A cheating prover trying to pass a false weight eval is caught at `wround0` (honest-procedure
batch over a false committed claim) or, if it lies coherently through every round and poisons
the last `C_vfin`, at that tensor's `wipa<G>` — observed WBO‑1a→`wround0`, WBO‑1b→`wipa8`.

**Conclusion (b): SOUND.** Hiding and binding both hold; no weight-MLE eval is recoverable
from any artifact; the two surviving leaks are statement-layer, documented, and confined.

---

## (c) LOCALIZATION (the C2 fix) — SOUND

`localize_batch_failure` (`common.py:1137-1167`) multiset-diffs the prover `claims.bin` against
the **verifier-recomputed** list on canonical per-claim record bytes; diverging records map
back to their claim ids via `raw2id`, and `verify_walk.py:404-425` blames exactly those ids
(`opening_batch b{k}: ... recomputed claim(s) diverge`), the weight batch likewise
(`477-499`).

The attribution cannot be gamed:

- The verifier's vacc is ground truth and always contains the **true** record for every
  obligation. Any divergence on obligation X's claim puts X's true record in `vc − pc`, so X is
  named. A prover cannot remove X's true record from the verifier side (the verifier recomputes
  it). The only way X's claim does *not* diverge is if the prover ships it byte-identical to the
  recomputation — i.e. X is honest. **No corrupt obligation can hide behind another id.**
- A forged record whose id matches no plan obligation does not get silently dropped: it stays
  on the batch locus (`verify_walk.py:414-418`, "forged id — stays on opening_batch").
- A failure with byte-identical claim lists (the corruption is in the sumcheck/vfin/IPA bytes,
  or an SZ-caught false claim) returns empty ids and keeps the batch as the named locus
  (`common.py:1163-1166`) — information-theoretically correct.

Runtime confirmation (my `/tmp/audit2` cases): a one-byte tamper of the last claim implicates
exactly `drv.C:c0`; dropping the middle claim implicates exactly `drv.B:c0`. The fc/rmsnorm
claim-mode selftests confirm driver-caught forgeries stay at the driver locus and
batch-relocated forgeries land at `claims_match`/`ipa<G>`.

**Conclusion (c): SOUND.**

---

## (6) SINGLE-PROCESS VERIFIER — fail-closed, no state leakage between obligations

`zkob_serve.cuh` runs each request through the **same** `zkw_run1` entry as the one-shot CLI
(byte-identical FS schedule); a thrown exception is caught and reported as `rc=3`
(`zkob_serve.cuh:39-47`). The pool treats any rc ∉ {0,1} as a crash → `RuntimeError` → REJECT,
**with no automatic retry** (a retry would double-append into the verifier accumulator —
explicitly avoided, `common.py:198-201, 275`); a dead worker is respawned only for the next
request (`common.py:211-229`). Timeouts kill the worker and raise (`common.py:248-260`).
Whitespace in argv is asserted away so the line protocol cannot be desynchronized
(`common.py:234-235`). Per-request state is fresh (`fs::Transcript` constructed per call,
generators/commitments loaded per call); the only process-global is `wp_rand`'s urandom handle,
which is prover-side only and never touched in verify. The accumulator dirs are append-only by
design and verifier-internal (`vacc`, never under `proofs/`). No check present in the old
per-subprocess path is missing in the consolidated one — the discharge, edge, statement, and
batch gates all run in `verify_walk.py` regardless of pool/no-pool. **Fail-closed; no leakage.**

---

## OBSERVATIONS (not soundness holes; close before relying on the system)

### F-1 — repo/build drift: the committed tree does not compile; the audited binary is uncommitted

`zkob_wpriv.cuh` (committed at Stage D) calls `bo_malloc` 9× (`zkob_wpriv.cuh:469-655`), but
the committed `zkllm-src/zkob_claims.cuh` (last touched at the **Stage B** commit `7173082`)
does **not** define `bo_malloc`. So a clean checkout of the repo will not build the weight-
privacy TUs. The **deployed** `/root/zkllm/zkob_claims.cuh` carries an *uncommitted* Stage‑C2
streaming revision that adds `bo_malloc` and transient witness streaming; that is the tree the
selftests pass on and the only tree this audit could exercise.

The diff between the deployed and committed `zkob_claims.cuh` is confined to (i) the
`bo_malloc` checked-allocation wrapper and (ii) loading witnesses transiently from `witpath`
instead of holding a resident `wit[]` vector — the transcript, challenges, round evals, vfin,
and IPA bytes are unchanged by construction (the streaming comments assert "byte-identical …
only WHEN tensors occupy device memory changed", and the algebra confirms it: same
`bo_build_eq`/`k_bo_axpy`/`k_bo_hp2`/`k_bo_fold` calls, same `partial_me` row-fold in G5).

Impact: **reproducibility/trust-gate**, not protocol soundness. The thing I validated is not
the thing in git. Recommend committing the streaming `zkob_claims.cuh` so the repo is
buildable and the audited artifact is the committed one. (Severity: medium — process.)

### F-2 — production verify runs neither the fold cross-check nor the `-dlto` kernel probe

`bo_probe_kernels()` and `ZKOB_FOLD_CROSSCHECK` are enabled **only** in selftest entry points
(`zkob_batchopen.cu:168/173`, `zkob_fc.cu:959/963`, every driver's `selftest` branch). The
production `verify_walk.py` sets `ZKOB_REQUIRE_RELATIVE_COMREF=1` and asserts the
`ZKOB_SLOW_*`/`ZKOB_EVIL` envs are **un**set (`verify_walk.py:145-148`), but never enables the
fast-vs-slow fold cross-check and never runs the kernel probe — so the GPU fast fold
(`bo_batched_group_fold`) and fast IPA (`bo_fast_ipa_verify`) run **unchecked at runtime**.

Soundness impact is **low**: (i) a miscompiled fast fold produces a wrong `C*_g`, which makes
the honest prover's IPA **fail** (false REJECT / liveness), not pass — the prover does not
control the verifier's kernels and cannot engineer a coincidence with a miscompile; (ii) a
commitment tamper is caught by the G0 comref-SHA‑256 absorb at `round0`, independent of the
fold (confirmed in `/tmp/audit2`); (iii) any miscompile is caught by the selftest on the
**same** binary. So this is a build-discipline dependency: the deployed binary must be the one
the selftest validated. Recommend running the (~ms) `bo_probe_kernels()` once at `serve`
startup to convert build-discipline into a runtime guarantee; optionally gate one batched
verify per walk on `ZKOB_FOLD_CROSSCHECK` as a regression. (Severity: low — hardening.)

---

## Cross-reference to the design's required pins

| pin (TRANSPORT_REVIEW) | as-built location | status |
|---|---|---|
| F3 row-count before fold, every tensor | `zkob_claims.cuh:1091-1100`, `:279-280`; `zkob_wpriv.cuh:739-748` | PRESENT, runs; runtime-confirmed (F10) |
| F4 length-prefixed claim absorb + explicit tag | `zkob_claims.cuh:104-116, 309-332` | PRESENT |
| F5 verify consumes recomputed list; claims_match inseparable | `zkob_claims.cuh:1055, 1062-1070`; `verify_walk.py:324-337` | PRESENT |
| F6 drvstate verifier-internal; redundant fields; n_claims==0 | `zkob_claims.cuh:1057-1058, 1130-1135` | PRESENT |
| F7 committed-round-message weight sumcheck | `zkob_wpriv.cuh:536-541, 788-795`; `zkob_fc.cu:258-270` | PRESENT (built as the F7 amendment) |
| BO‑1a/1b split loci | `zkob_batchopen.cu:179-221` → `round0`/`ipa8` | PRESENT, runtime-confirmed |
| registered comref discharge (F5/F6 pin) | `verify_walk.py:506-544` | PRESENT |
| protected headers unmodified | `git diff 7d816a1..HEAD` empty; `cmp` deployed==repo | CONFIRMED |

---

## Bottom line

The as-built batched-opening protocol, the weight-privacy endgame, and the C2 localization
faithfully implement the audited design, with every required pin present in the executed code
path and every forgery I constructed rejected at its named locus. I could not construct a
cheating prover the as-built verifier accepts. **(a) batching SOUND, (b) weight privacy SOUND,
(c) localization SOUND.** Close F-1 (commit the streaming `zkob_claims.cuh` so the repo
matches the audited binary) and F-2 (probe the fast kernels at serve startup) before treating
this as the production trust gate; neither blocks the soundness conclusion.
