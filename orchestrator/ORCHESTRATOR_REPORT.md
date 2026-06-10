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
