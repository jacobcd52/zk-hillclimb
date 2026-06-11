# ORCHESTRATOR_REPORT — stage 1 (MLP subgraph) end-to-end

Status date: 2026-06-10. Code: `orchestrator/{common,register,prove_walk,verify_walk,make_stage1_manifest}.py`,
`selftest.sh`. Design: ORCHESTRATOR_DESIGN.md. All numbers below MEASURED on the
shared RTX 4090 (driver invocations serialized via /tmp/zkorch.gpu.lock; the
zkob_softmax build job was running concurrently — timings are sequential-honest
but not exclusive-GPU official numbers; HARNESS.md timing protocol still applies
for scoring).

## What ran

Two full runs: `smoke1` (development) and `official1` (selftest.sh end-to-end:
register → prove_walk → verify_walk honest ACCEPT + two tamper REJECTs +
restored re-ACCEPT). Real llama-68m weights (byte-identical to the pipeline's
own integer dumps — provenance-checked at registration); run input = the
pipeline's random-normal-activation convention, seeded for reproducibility,
registered by digest.

Covered: 30 manifest ids (28 layer obligations + `statement.registered_weight_hash`
+ `statement.prompt_binding`) — full MLP path of both layers (gate/up/down
commitment_opening + matmul + rescaling, swiglu), all 4 rmsnorm sites (each =
zkob_rmsnorm + W-rescale + Y-rescale), both skip connections per layer.
SKIPPED (explicit, with reasons in the transcript): 24 attention-side ids,
`lm_head.commitment_opening`, `statement.logit_binding` — 26 total.

`check_transcript.py` results (both run and recorded, per design §6):
- vs `manifest_stage1.json` (frozen manifest with stage-1 skips marked waived,
  harness file untouched): `required: 30 checked: 30 missing: 0 unknown: 0` → PASS.
- vs the FULL frozen manifest: `required: 56 checked: 30 missing: 26 unknown: 0`
  → the expected stage-1 gap, printed id-by-id, not hidden. The verifier covers
  every obligation it claims and claims nothing it doesn't.

## Timings (smoke1; official1 within noise) — prove and verify separately

| obligation (manifest id) | prove s | verify s |
|---|---|---|
| layer{0,1}.input_norm.rmsnorm | 14.2 / 14.0 | 13.9 / 13.9 |
| layer{0,1}.post_attn_norm.rmsnorm | 14.0 / 14.0 | 13.9 / 13.9 |
| layer{0,1}.attn_skip.add | 0.4 | 0.3 |
| layer{0,1}.mlp.gate_proj.matmul (+opening) | 3.8 | 3.7 |
| layer{0,1}.mlp.gate_proj.rescaling (sf 2^20) | 6.8 | 6.4 |
| layer{0,1}.mlp.up_proj.matmul (+opening) | 3.8 | 3.7 |
| layer{0,1}.mlp.up_proj.rescaling (sf 2^16) | 6.8 | 6.4 |
| layer{0,1}.mlp.swiglu (glu + hrescale) | 18.1 | 20.2 |
| layer{0,1}.mlp.down_proj.matmul (+opening) | 3.7 | 2.7 |
| layer{0,1}.mlp.down_proj.rescaling | 2.5 | 3.3 |
| layer{0,1}.mlp_skip.add | 0.2 | 0.3 |
| **TOTAL (driver time, both layers)** | **148.4** | **149.9** |
| wall clock (incl. python witness gen / edge checks) | 152.9 | 149.9 |

One-time registration (not in the prove path): ~25 s total — ppgen 1024/4096/1,
16 weight exports + commitments, input commitment, swiglu table. Unproven
attention data segment (python, both layers): < 7 s inside prove wall time.
rmsnorm advice R (integer-exact bracket, python): ~1.4 s per site.

## Proof size (covered subgraph, both layers)

16,622,704 bytes (15.9 MiB) on disk across 22 obligation dirs; 11.6 MiB after
content dedup (chained commitment files are intentionally present in BOTH
neighboring obdirs — that duplication is what the byte-equality edges check).
Biggest items: each rmsnorm site 1.86 MiB (3 sub-proofs), each swiglu 1.45 MiB,
each 2^20-rescale ~0.6 MiB.

## Soundness checks performed (selftest.sh `official1`)

- Honest end-to-end: verdict ACCEPT, 30/30 covered ids in `checked`, all 39
  chain edges OK (35 byte-equality + 4 zkob_skip Pedersen point checks).
- Tamper (b): one byte flipped in `proofs/layer0.mlp.swiglu/glu/com_H.bin`
  (a CHAINED commitment) → verify_walk REJECT with `layer0.mlp.swiglu` in
  `rejected` (driver transcript divergence + chain edge M7 failure). Restored.
- Tamper (c): one registered-weight sha256 character flipped in public.json →
  REJECT at the registration check (`statement.registered_weight_hash`
  ok=false), with NO driver verifies run (untrusted registration aborts the
  walk). Restored.
- Restored run re-ACCEPTs.
- Result: ALL PASS (see /root/zkorch/selftest_official1.log).

## Problems hit / notes for the next stage

1. **Upstream ffn.cu writes the UNRESCALED down_proj output** through an
   int32-truncating path as its layer output (values at scale 2^32 — garbage
   after truncation). The orchestrator chains the rescaled output `down_out_`
   instead (the manifest's `down_proj.rescaling` makes this the only coherent
   choice). Documented deviation from the upstream binary's file output; no
   effect on proof semantics.
2. **Pipeline attention quirks replicated verbatim** in the unproven data
   segment (m68-pipeline.py lines 149, 156-157): the `max(A*~mask)` zero-floor
   and the double transpose/reshape that scrambles the head layout. We bind
   what the pipeline computes; flagged for whoever lands the attention chain —
   the values matmul will have to consume this exact layout (or the pipeline
   owner decides line 157 is a bug — that's a pipeline-authority decision, not
   ours).
3. **Jacobian non-uniqueness**: homomorphic skip sums are point-equal but NOT
   byte-equal to fresh commitments, so skip edges are zkob_skip point checks
   while all other edges are byte checks. Anyone adding edges must keep that
   distinction.
4. **com_Wr.bin naming** (PHASE0 §14 MINOR-7) honored in the edge map; the
   rmsnorm com_X chaining obligation (MINOR-5) is closed by edges
   R0/S1/S2-into-com_X for all four sites.
5. **GPU sharing** worked: flock-serialized subprocess calls, retry-once;
   zero retries were actually needed in either run.
6. `zkob_fc commit` doubles as the generic activation/gain registration
   primitive ((1,768) for rmsnorm gains, (1024,768) for the input activation) —
   byte-compatible with the drivers' internal commits (confirmed by the 35
   passing byte edges); no new CUDA was needed.

## Honest status

- The verifier is genuinely independent (reads public.json + registration/ +
  proofs/ only; never data/; re-derives run_seed; subprocesses are verify modes
  only) — but its trust anchors are the five validated drivers; this
  orchestrator adds no new cryptography.
- The attention boundary is OPEN: `com_attn_out` in the attn_skip check is a
  fresh prover commitment of UNPROVEN data. A dishonest prover could put
  anything there and stage 1 would accept (it would still have to be
  rmsnorm/MLP-consistent downstream). This is exactly the declared gap — the
  26 SKIPPED ids — and closes when the attention+softmax chain lands
  (SOFTMAX_DESIGN §1.1/§7.3 wiring, incl. the int32→int64 widening shim already
  implemented and exercised in `prove_walk.widen_i32_to_i64`).
- `statement.prompt_binding` is checked in the random-input sense (input
  digest bound via run_seed = H(public.json) into every FS transcript); real
  prompt-token binding starts existing when embeddings do.
- `statement.logit_binding` / final-argmax: reserved slot only
  (public.json `future_slots`, manifest id kept SKIPPED) — no logits are proven
  until lm_head.
- Composed sub-obligations (wrescale/yrescale/hrescale) are recorded
  per-sub-run in both prove_manifest.json and transcript details; nothing is
  silently merged.

## Hardening round (2026-06-10, post-audit)

All six VERIFIER_REVIEW MINOR findings applied (verify_walk.py +127/-28, common.py +27/-1;
diff reviewed line-by-line by the coordinator):
- MINOR-1: run_dir absolutized (hash check, edges, drivers resolve identical paths).
- MINOR-2: public.json read ONCE; constants and run_seed derived from the same bytes.
- MINOR-3: 900s timeout on every driver subprocess; timeout = fail-closed RuntimeError.
- MINOR-4: commitment_opening discharge moved AFTER the chain-edge phase, so a failed
  edge drags the opening id out of `checked`.
- MINOR-5: structural assertion that every registration/ path consumed by the walk is
  hash-pinned; violation = REJECT-and-STOP before any driver runs.
- MINOR-6: proofs/ hygiene walk — symlink escapes and non-regular files = REJECT-and-STOP.
Re-validated: selftest.sh 9/9 ALL PASS (run selftest-20260610-230433, coordinator-run).

---

# Stage 2 (2026-06-11): full attention chain — every non-lm_head obligation verified

Run: `stage2-official1` (`selftest.sh` end-to-end, 10/10 ALL PASS,
`/root/zkorch/selftest_stage2.log`). Coverage: **54 manifest ids checked = 56
non-waived in the FROZEN manifest − 2 stage-2 waived** (`lm_head.commitment_opening`,
`statement.logit_binding`) — the arithmetic is recounted from the manifest file by the
selftest itself and asserted, not assumed. `embedding.lookup` needs no stage waiver:
it is waived in the frozen manifest. `skipped` contains exactly the 2 waived ids.
`check_transcript` vs the stage-2 scope manifest: `required: 54 checked: 54 missing: 0
unknown: 0` → PASS; vs the FULL frozen manifest: `required: 56 checked: 54 missing: 2`
with exactly those two ids printed — the only remaining non-waived gap.

## What changed (extend, not rewrite — all stage-1/hardening invariants kept)

- **common.py**: attention constants (HEAD_DIM 64, 12 heads, rescale logs 16/16/13/10/16,
  softmax LOW_E/LEN_E/LEN_R), gen64 + rope-cos/sin + softmax-exp registration paths,
  `attn_proj_block()` (q/k/v fc vs REGISTERED com_W + rescale) and `attention_spec()` —
  the §4.0-composed sub specs (scores_matmul = rope.q/k + rescales + slice + 12 scores fc;
  softmax = 12×(rescale13+rescale10+softmax); values_matmul = 12×(fc+rescale) + merge,
  seeds per §7.3) plus EVERY §7.4 edge A1..A15. New edge kind `path` (A7/A12): the
  per-head fc's com_W argv must BE the slice commitment file (structural check; the
  security is the driver absorbing that file — a divergent operand rejects the transcript).
- **register.py**: `ppgen 64` → gen64.bin; runs the PINNED generators
  (`/root/zkllm/gen_rope_tables.py`, `gen_softmax_exp_table.py`, cwd = registration/ so
  /root/zkllm is never written; outputs byte-identical to the driver-validation copies);
  all four new artifacts sha256-pinned in public.json (registration now pins 26 hashes,
  was 22). Anchor asserts: cos[0,0] = 2^16, exp(0) = 2^16.
- **prove_walk.py**: the python-float attention segment (and the torch/transformers model
  load) is GONE. Per layer, the §7.3 integer chain runs instead; **attn_out := the
  headmerge O2 output** (witness-authority rule), feeding attn_skip.add and everything
  downstream. The orchestrator computes nothing in the segment except the int32→int64
  widening shim between the two score rescales (lossless; stage-1 unit-checked) and a
  completeness guard that scores stay inside the exp-table domain.
- **verify_walk.py**: registration check extended to all gens incl. gen64 + ALL tables
  (iterates public.json's pinned dicts — the MINOR-5 structural assertion sees and
  requires every new artifact: argv-consumed unpinned paths still REJECT-and-STOP);
  168 new driver verify invocations; 283 chain edges total (231 byte + 48 path + 4 skip),
  all OK in the honest run. Per-id ok-rule per §9.5: id ∈ `checked` only if ALL composed
  sub-runs AND all its edges pass (spot-checked: scores_matmul lists 17 sub-runs + 43
  edges; softmax 36 + 36; values_matmul 25 + 37 in its details string).
- **Edge S1's open boundary is CLOSED** (A15: merge com_O2 ≡ attn_skip com_attn_out —
  byte-identical commitments). The residual stream is now chained registered-input →
  terminal output commitment with no prover-chosen gap.

## Timings (stage2-official1, honest run) — prove and verify per obligation

| obligation (manifest id, both layers) | prove runs | prove s | verify runs | verify s |
|---|--:|--:|--:|--:|
| layer{0,1}.input_norm.rmsnorm | 8 | 28.7 | 6 | 27.7 |
| layer{0,1}.attn.{q,k,v}_proj.matmul (+opening) | 6 | 13.2 | 6 | 9.9 |
| layer{0,1}.attn.{q,k,v}_proj.rescaling | 6 | 15.6 | 6 | 19.9 |
| layer{0,1}.attn.scores_matmul (rope×2+rescales+slice+12 fc) | 34 | 116.8 | 34 | 101.7 |
| layer{0,1}.attn.softmax (12×rescale13+rescale10+softmax) | 72 | 382.6 | 72 | 837.6* |
| layer{0,1}.attn.values_matmul (12×fc+rescale, merge) | 50 | 63.6 | 50 | 150.9* |
| layer{0,1}.attn_skip.add | 2 | 1.0 | 2 | 0.7 |
| layer{0,1}.post_attn_norm.rmsnorm | 8 | 28.4 | 6 | 113.2* |
| layer{0,1}.mlp.gate_proj.matmul (+opening) | 2 | 7.8 | 2 | 7.5 |
| layer{0,1}.mlp.gate_proj.rescaling (sf 2^20) | 2 | 13.8 | 2 | 13.0 |
| layer{0,1}.mlp.up_proj.matmul (+opening) | 2 | 7.8 | 2 | 7.6 |
| layer{0,1}.mlp.up_proj.rescaling | 2 | 13.7 | 2 | 13.1 |
| layer{0,1}.mlp.swiglu (glu + hrescale) | 4 | 36.6 | 4 | 40.8 |
| layer{0,1}.mlp.down_proj.matmul (+opening) | 2 | 7.6 | 2 | 5.4 |
| layer{0,1}.mlp.down_proj.rescaling | 2 | 5.2 | 2 | 6.6 |
| layer{0,1}.mlp_skip.add | 1 | 0.3 | 2 | 0.7 |
| **TOTAL (driver time)** | **203** | **742.7** | **200** | **1356.3** |

**Headline numbers (full forward pass, both layers):**
- **prove: 743.0 s wall ≈ 12.4 min** (driver time 742.7 s — python overhead is now
  negligible; rmsnorm R advice ≈ 1.4 s/site is inside the wall number; 0 driver retries).
- **verify: 1356.6 s wall ≈ 22.6 min** (*see contention caveat below — sequential-honest,
  NOT exclusive-GPU*).
- **proof bytes: 150,577,648 (≈ 143.6 MiB)** across all obligation dirs, before content
  dedup (chained commitment files intentionally exist in both neighboring obdirs — that
  duplication IS what the byte edges check). Design §8.5 predicted ≈ 120 MB + the stage-1
  16.6 MB ≈ 137 MB; measured 5% over. Witness data/ (prover-only): 1.1 GiB/run.
- Per-instance prove vs design §8.5 predictions: rope 2.29 s (~3 predicted), rope rescale
  2.64 s (2.6 measured ref), headslice 29.6 s (the §9.1 gate's 29.2 s — no fallback
  needed), scores fc 1.58 s (~1.0), softmax 10.59 s (10.2), values fc 1.33 s (~0.8),
  values rescale 1.00 s (~0.5), headmerge 3.79 s (~2.5). Attention-segment driver time:
  592.8 s prove ≈ 9.9 min (predicted ≈ 8.3 min, +19% — within the ±30% error bars).

\* **GPU-contention caveat (timing honesty):** a concurrent driver-hardening job ran the
raw zkob_* selftests during the verify phases WITHOUT the orchestrator's GPU lock
(raw selftests don't take it). Prove was nearly clean; verify shows clear contention
spikes: softmax verify per head median 12.2 s (matches the 11.6 s exclusive-GPU
measurement) but max 123.1 s; post_attn_norm verify 113.2 s vs 27.7 s for the identical
input_norm trio. Median-based, an exclusive-GPU verify would be ≈ 17–18 min; the §8.6
prediction (stage-1 ~2.5 + attention ~9.5 ≈ 12 min) was for exclusive GPU and the
per-instance medians are consistent with it except headslice (24.6 s vs predicted ~12,
known from ROPE_IMPL_REPORT) and the values-fc/rescale rows (~3 s vs ~0.8 predicted,
small-IPA verifies are subprocess/CUDA-init dominated). HARNESS.md timing protocol still
applies for official scoring. The 900 s per-driver timeout was never approached (max
single invocation 123 s, under contention) — NOT raised, per instructions.

## Selftest (10/10 ALL PASS, `/root/zkorch/selftest_stage2.log`)

- (a) register → prove_walk → verify_walk: ACCEPT, checked = 54 (= 56 − 2, recounted
  from the manifest), skipped = exactly {lm_head.commitment_opening,
  statement.logit_binding}; check_transcript PASS vs stage-2 scope manifest; vs FULL
  manifest exactly the 2 waived ids missing (printed).
- (b) one byte flipped in `proofs/layer0.attn.scores_matmul/slice/com_KhT05.bin` →
  REJECT with the layer-obligation rejection localized to exactly
  `layer0.attn.scores_matmul`: the headslice transcript diverges (`REJECT: IPA opening
  of eQ00 vs com_Qh00` — the tampered com is absorbed before the challenges) AND the
  scores fc.h05 verify diverges (its com_W argv IS that file): two independent
  detections, as designed. Restored.
- (c) the registered rope-cos sha256 flipped in public.json → REJECT at the
  registration check (`hash mismatch: tables.rope-cos-table.bin`), checked = [],
  NO drivers run (fail-closed stop). Restored.
- Restored run re-ACCEPTs (full verify re-run, not cached).

## Honest caveats / notes

1. **Witness provenance switch, difr unmeasured here.** attn_out now comes from the
   integer chain (rope tables + driver rescales) instead of the float-replicated
   pipeline attention — drift ≤ 1 ulp per entry per rounding site vs the float path
   (ROPE_ATTENTION_DESIGN §2.2/§1.3). The design's §9.6 note stands: an end-to-end
   difr measurement of the new witness against the pipeline output has NOT been run in
   this stage (orchestrator scope); coordinator/pipeline-side item before the writeup
   claims approximation numbers.
2. **Rope tables are float64-generated** (pinned script); the pipeline's cos/sin are
   float32 — a few ±1-ulp@2^16 entries may differ from a float32-faithful table.
   Soundness unaffected (both sides load the same sha256-pinned bytes); approximation
   note only (design §9.2).
3. **Verify-side trust anchors unchanged**: the verifier still reads only public.json +
   registration/ (26 hashes re-checked first, fail-closed) + proofs/; never data/;
   re-derives run_seed; verify modes only. The three new drivers were independently
   audited (ROPE_REVIEW: SOUND per driver) before this stage consumed them; this
   orchestrator still adds no cryptography.
4. **A7/A12 "path" edges are structural**, not cryptographic: they assert the wiring
   (the fc verify argv references exactly the slice commitment file). The binding force
   is the driver's transcript absorb of that file — demonstrated by tamper (b) rejecting
   through BOTH routes.
5. **Concurrent driver rebuilds**: the hardening job may have relinked the three new
   driver binaries mid-run (selftest-only changes; verify behavior pinned unchanged by
   its constraints). Zero retries and all 400+ invocations consistent — no observed
   effect; flagged for completeness.
6. The stage-1 attention quirk note (report item 2) is now obsolete: the line-156/157
   permutation is BOUND by zkob_headmerge's π (the values matmul consumes the §4.3
   slice layout; nothing downstream sees a scrambled head layout unproven).
7. Disk: each full run ≈ 1.4 GiB (proofs 144 MiB + data 1.1 GiB + registration 95 MiB).
   Old stage-1 runs left in /root/zkorch untouched.

## Stage 3 (2026-06-11) — manifest closed (coordinator-completed report)

The stage-3 agent's session ended mid-validation; the coordinator fixed one wiring bug
(headmerge's new <perm> arg missing at both call sites — added "pi157") and re-ran
selftest.sh stage3-official1 end-to-end: **ALL PASS (12/12), checked=59 rejected=0
skipped=0, VERDICT ACCEPT** (incl. tamper phases and restored re-ACCEPT).

Coverage: 56/56 non-waived manifest ids + 3 covered-waived (final_norm.rmsnorm,
lm_head.matmul, lm_head.rescaling); only embedding.lookup remains waived-uncovered.
statement.logit_binding is live: t* = argmax(logits) at all 1024 positions, t*
sha256-pinned in public.json, bound by a zkob_rowmax vpad instance (verify 4.38 s —
the fast-IPA path).

Headline timings (stage3-official1, sequential-honest): full-forward PROVE wall
875.5 s ≈ 14.6 min (vs 743 s stage-2; Part B added ~132 s — well under the §5.1
420-580 s prediction, the fast-IPA kernel being the difference), proof+commitments
154.4 MB. Notable verify rows: lm_head.matmul 27.6 s, lm_head.rescaling 48.3 s
(gen-32768 me_weights host loop — the §6.2 header-lift remains the known remedy),
rowmax logit binding 4.4 s, final_norm trio 14.1 s.
Selector-tie count: see prove_manifest.json (reported per §2.4 duty).
