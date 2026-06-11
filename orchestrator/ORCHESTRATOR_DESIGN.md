# ORCHESTRATOR_DESIGN — registration → prover walk → separate verifier → transcript.json

Status: 2026-06-10 (stage 1); UPDATED 2026-06-11 (stage 2). Stage 1 covered subgraph =
full MLP path of both layers + all 4 rmsnorm sites + both skip connections per layer,
attention SKIPPED. **Stage 2 added the complete attention chain** (q/k/v proj, rope,
headslice, per-head scores/softmax/values, headmerge — composition + edges per
ROPE_ATTENTION_DESIGN §4.0/§7.3/§7.4; edge S1's open boundary closed by A15): 54 of the
56 non-waived manifest ids checked; only lm_head.commitment_opening +
statement.logit_binding remain SKIPPED (recorded, never silently dropped). Sections
below describe stage 1 and remain accurate for the mechanisms they pin; stage-2 specifics
(new registration artifacts gen64 + rope/exp tables, the attention walk + edge map) live
in common.py's attention_spec and ORCHESTRATOR_REPORT.md's Stage 2 section.

Python: `/root/int-model-env/bin/python` (same env the pipeline runs under — torch
2.7.1+cu128, transformers 4.57.3, numpy 2.4.6). Drivers are the validated binaries in
`/root/zkllm` (`zkob_fc`, `zkob_rescale`, `zkob_skip`, `zkob_glu`, `zkob_rmsnorm`),
invoked as subprocesses, strictly one at a time (shared GPU), one retry on failure.
No CUDA is built or edited by the orchestrator.

## 1. Directory layout per run

All run state lives on local disk (never executed from FUSE):

```
/root/zkorch/<run_id>/
  public.json                  # THE public statement; run_seed := sha256(file bytes)
  registration/                # verifier-readable, hash-pinned by public.json
    gen1024.bin gen4096.bin q.bin          # ppgen output (one-time)
    weights/<wid>-int.bin                  # integer weights (pipeline semantics)
    weights/<wid>-com.bin                  # registered commitments (zkob_fc commit)
    com_input.bin                          # commitment of the run input activation
    input.i32.bin                          # the public input activation (statement)
    swiglu-table.bin                       # public silu table (pipeline semantics)
  data/                        # PROVER-ONLY witness files; verifier NEVER reads these
    *.i32.bin *.i64.bin R_*.bin ...        # activations, chain data files, advice
  proofs/<manifest_id>/[<sub>/]            # obdirs (orchestrator mkdirs; drivers don't)
  prove_manifest.json          # what ran, seeds, timings, proof bytes (prover record)
  transcript.json              # verifier output (harness format)
```

`<wid>` = e.g. `layer0.mlp.gate_proj`, `layer0.input_norm.g`. Weight ids registered:
gate/up/down + q/k/v per layer (q/k/v registered now for forward-compat with the
attention stage) + the 4 rmsnorm gains. `lm_head` is NOT yet registered (32768-wide
generator set not generated; slot documented, see §6).

## 2. Registration (`register.py`, one-time per run)

1. `ppgen 1024 gen1024.bin`, `ppgen 4096 gen4096.bin`, `ppgen 1 q.bin`.
2. Export integer weights with the pipeline's exact semantics
   (`round(w.float().T * 2^16) → int32` for 2-D, `round(w * 2^16)` for 1-D), loading
   `JackFram/llama-68m` from the pipeline's model cache. Where the pipeline's own dump
   (`/root/zkllm/zkllm-workdir/llama-68m/layer-*-int.bin`) exists, byte-compare and
   fail on mismatch (provenance guard). `int-model-approximation` repo is never touched.
3. Commit every registered weight with the existing driver pattern
   `zkob_fc commit <int.bin> <IN> <OUT> <gen_out> <com.bin>`:
   2-D weights as (IN, OUT) with gen of size OUT_pad; rmsnorm gains as (1, 768) with
   gen1024 (produces the 1-row commitment layout zkob_rmsnorm's verifier expects);
   the input activation as (1024, 768) with gen1024 → `com_input.bin`.
4. Generate the run input with the pipeline's convention (`round(randn(1024,768)·2^16)`,
   numpy RNG seeded from the run id — the pipeline itself uses unseeded randn; we pin a
   seed for reproducibility) and the swiglu table with the pipeline's exact code
   (`save_int(Xs·sigmoid(Xs), 2^16)`, `Xs = arange(-2^9, 2^9, 2^-12)` on the GPU,
   float32 — bit-faithful to m68-pipeline.py line 104-105). The registered file is the
   source of truth thereafter (sha256, never regenerated at verify time).
5. Write `public.json`: model card, seq_len, scales/constants (LOG_SF=16, gate rescale
   2^20, C_eps=3298535), `prompt_token_ids: null` (random-input pipeline; embedding
   waived), sha256 of: every gen file, every registered weight commitment, com_input,
   input.i32.bin, swiglu-table.bin. Plus `future_slots.statement.final_argmax`
   (reserved — THREAT_MODEL_NOTES §1 greedy-argmax binding; gets a transcript slot via
   manifest id `statement.logit_binding` when lm_head lands).

**run_seed := sha256(public.json bytes).** Every driver transcript is seeded
`"<run_seed>:<obligation_id>"`, so every proof is bound to the full public statement
(registered hashes, input digest, constants). The verifier RE-DERIVES run_seed from
public.json; it never trusts a stored seed.

## 3. Manifest walk order (prove_walk.py) and witness semantics

Walk follows the manifest's architectural order, per layer l ∈ {0,1}:

```
input_norm.rmsnorm → [attention: SKIPPED — data computed in python, unproven]
→ attn_skip.add → post_attn_norm.rmsnorm
→ mlp.gate_proj.{matmul+opening, rescaling 2^20} ∥ mlp.up_proj.{matmul+opening, rescaling 2^16}
→ mlp.swiglu (glu + hidden rescale 2^16) → mlp.down_proj.{matmul+opening, rescaling 2^16}
→ mlp_skip.add → next layer
```

Composition under manifest ids (sub-obligation seeds in parentheses):

| manifest id | driver runs (seed suffix) |
|---|---|
| layer{l}.input_norm.rmsnorm | zkob_rmsnorm (`…rmsnorm`) + zkob_rescale on W.i64 (`…rmsnorm.wrescale`) + zkob_rescale on Y.i64 (`…rmsnorm.yrescale`) |
| layer{l}.post_attn_norm.rmsnorm | same trio |
| layer{l}.attn_skip.add | zkob_fc commit of attn_out (com_B); check is verify-side |
| layer{l}.mlp.gate_proj.matmul + .commitment_opening | one zkob_fc run (seed id = `…gate_proj.matmul`; the fc proof's IPA(W) vs the REGISTERED com discharges the opening id) |
| layer{l}.mlp.gate_proj.rescaling | zkob_rescale sf 2^20 (gate must land at scale 2^12 for the silu table — upstream ffn.cu's `gate_rescale(1<<20)`) |
| layer{l}.mlp.up_proj.* | as gate, rescale sf 2^16 |
| layer{l}.mlp.swiglu | zkob_glu (`…swiglu`) + zkob_rescale on H.i64 (`…swiglu.hrescale`) |
| layer{l}.mlp.down_proj.* | zkob_fc (IN=3072) + zkob_rescale sf 2^16 |
| layer{l}.mlp_skip.add | zkob_skip add for the terminal com_Z (l=1 only); check verify-side |

Witness/data authority (the pinned rules):
- Chain data files come from the DRIVERS themselves wherever a driver emits them
  (fc → Y.i64, rescale → Xr.i32, glu → H.i64, rmsnorm → W.i64/Y.i64). The orchestrator
  only computes: the run input, skip sums (`Z = A + B`, int32), the rmsnorm advice R,
  and the unproven attention segment.
- **rmsnorm advice R is computed INTEGER-EXACTLY by the orchestrator** (python
  arbitrary-precision ints): per row, M = Σ X² + C_eps; R = largest r with
  r²·M ≤ 2^64·C, found by isqrt + ±1 fix-ups — the same bracket as zkob_rmsnorm.cu's
  `exact_R`, replacing the pipeline's float `1/sqrt` path (pinned authority change;
  same rule applies to softmax's P later).
- Unproven attention data replicates m68-pipeline.py lines 137-159 **verbatim**
  (including the `A*~mask` max quirk, float32 `shift`, and the line-156/157 double
  transpose+reshape), with q/k/v int matmuls + driver-semantics rescale
  (rem ∈ [−sf/2, sf/2), i.e. floor((x+sf/2)/sf)) standing in for the upstream
  self-attn linear binary. Deviations from the upstream binaries are documented in
  ORCHESTRATOR_REPORT.md; soundness is unaffected (the segment is declared SKIPPED).
- ffn output := the down_proj **rescaled** output (down_out_). Upstream ffn.cu saves
  the pre-rescale tensor through an int32-truncating path (upstream bug); the manifest
  requires `down_proj.rescaling`, so the rescaled tensor is the only coherent chain.
  Documented deviation.
- int32→int64 widening shim: `widen_i32_to_i64()` (lossless numpy copy) is provided
  for chains where a rescale stage feeds another rescale stage (attention scores
  2^13→2^10 per SOFTMAX_DESIGN §7.3). Stage 1 itself needs no widening (every rescale
  input is already a driver-emitted int64); the shim is wired and unit-checked so the
  attention chain can use it unchanged.

Obligation dirs are created by the orchestrator (`mkdir -p`) before each prove —
drivers do NOT mkdir. GPU invocations are strictly serialized; each is retried once.

## 4. The chain byte-equality map (EVERY edge in the covered subgraph)

`≡` = byte-identical files (commitments are deterministic MSMs of identical padded
data → identical bytes; validated in PHASE0 §12). `⊕` = Pedersen point-equality via
`zkob_skip verify` (homomorphic sums are point-equal but NOT byte-equal in Jacobian
coordinates, so skip edges are point checks by design). Paths relative to
`proofs/`; `reg/` = `registration/`. Per layer l (N = layer{l}.input_norm.rmsnorm,
P = layer{l}.post_attn_norm.rmsnorm, ids shortened):

| # | edge | binds |
|---|---|---|
| R0 | `layer0…input_norm/rmsnorm/com_X.bin ≡ reg/com_input.bin` | statement input → layer 0 residual stream |
| R1 | `N/rmsnorm/com_g.bin ≡ reg/weights/layer{l}.input_norm.g-com.bin` | (redundant with the driver's registered-com_g absorb; defense in depth) |
| W1 | `N/rmsnorm/com_W.bin ≡ N/wrescale/com_X.bin` | internal W → its rescale proof |
| W2 | `N/rmsnorm/com_Wr.bin ≡ N/wrescale/com_Xr.bin` | **file name com_Wr.bin, NOT com_W_.bin — pinned PHASE0 §14 MINOR-7** |
| Y1 | `N/rmsnorm/com_Y.bin ≡ N/yrescale/com_X.bin` | rmsnorm output → output rescale |
| S1 | ⊕ `skip(A = N/rmsnorm/com_X.bin, B = attn_skip/com_attn_out.bin, Z = P/rmsnorm/com_X.bin)` | attn skip; **com_attn_out is an OPEN BOUNDARY** (attention unproven — see §6) |
| R1′,W1′,W2′,Y1′ | same four edges for P | post-attn norm site |
| M1 | `gate_fc/com_X.bin ≡ P/yrescale/com_Xr.bin` | MLP input chained to post-attn norm output |
| M2 | `up_fc/com_X.bin ≡ P/yrescale/com_Xr.bin` | same activation, second consumer |
| M3 | `gate_fc/com_Y.bin ≡ gate_rescale/com_X.bin` | pre-rescale gate product |
| M4 | `up_fc/com_Y.bin ≡ up_rescale/com_X.bin` | pre-rescale up product |
| M5 | `swiglu/glu/com_G.bin ≡ gate_rescale/com_Xr.bin` | gate at scale 2^12 into the silu lookup |
| M6 | `swiglu/glu/com_U.bin ≡ up_rescale/com_Xr.bin` | up at scale 2^16 into the hadamard |
| M7 | `swiglu/glu/com_H.bin ≡ swiglu/hrescale/com_X.bin` | hidden product → hidden rescale |
| M8 | `down_fc/com_X.bin ≡ swiglu/hrescale/com_Xr.bin` | rescaled hidden into down_proj |
| M9 | `down_fc/com_Y.bin ≡ down_rescale/com_X.bin` | pre-rescale down product |
| S2 | ⊕ `skip(A = P/rmsnorm/com_X.bin, B = down_rescale/com_Xr.bin, Z = next)` | mlp skip; `next` = `layer{l+1}…input_norm/rmsnorm/com_X.bin` (l=0) or `mlp_skip/com_Z.bin` (l=1, terminal output commitment) |

Weight bindings need no obdir edge: `zkob_fc verify` / `zkob_rmsnorm verify` take the
REGISTERED commitment path as an argument and absorb it into the transcript — a prover
that used different weights diverges and is rejected. The registered files themselves
are hash-checked against public.json before any driver runs.

**Why the rmsnorm com_X edges are load-bearing (PHASE0 §14 MINOR-5):** a standalone
rmsnorm ACCEPT with prover-chosen X admits ~2^39 values of R (the bracket is only as
tight as M is honest). Edges R0/S1/S2 chain every rmsnorm com_X byte-identically to
the upstream activation commitment, which is what the ±1 bound actually relies on.

## 5. The verifier (verify_walk.py) — what it reads, what it checks, what it never trusts

Independence argument (one paragraph): verify_walk is a separate process that reads
ONLY `public.json`, `registration/` (gens, registered commitments, the public input
file and table — each first re-hashed and compared against public.json), `proofs/`
(the serialized proof artifacts), and the frozen harness manifest. It re-derives
run_seed = sha256(public.json) itself. It then (a) re-runs every covered obligation's
`zkob_* verify` as a subprocess with registered public inputs only, (b) checks every
edge of the §4 map (byte-equality via file compare; skip edges via `zkob_skip verify`
point arithmetic on commitment files), and (c) emits transcript.json. It NEVER reads
`data/` (witness activations, R advice, chain data files, weight int files — weight
int files live under registration/ but are never opened by the verifier), never
invokes any `prove` mode, and trusts nothing a prover could choose: not stored seeds,
not obdir dims (drivers cross-check dims.bin against CLI args derived from public
constants), not prover-supplied commitments except as objects to be proven against and
chained byte-exactly into the registered boundary. Soundness rests only on:
commitments + IPA openings (driver verify), public constants/tables (hash-pinned),
and byte-/point-equality of commitment files.

Verify order: (1) registration hash check — on ANY mismatch, record
`statement.registered_weight_hash: ok=false`, verdict REJECT, and STOP (running
drivers against an untrusted registration proves nothing); (2) per-obligation driver
verifies, walk order; (3) chain edges; (4) statement obligations. A driver REJECT or
edge failure marks that manifest id `ok=false` and flips the verdict.

## 6. transcript.json, SKIPPED ids, and check_transcript.py

```json
{
  "verdict": "ACCEPT" | "REJECT",
  "checked":  ["<manifest ids that PASSED a real check>", ...],
  "rejected": ["<covered ids that failed>", ...],
  "skipped":  {"<manifest id>": "<reason>", ...},
  "details":  {"<id>": {"ok": true, "reason": "<per-sub-run + per-edge results, '; '-joined>"}},
  "chain_edges": [{"edge": "<label>", "ok": true}, ...],
  "registration": {"ok": true, "n_hashes": ...},
  "timing": {"<id>[<sub>]": seconds, ..., "total_verify_wall_s": ...}
}
```

- `checked` contains ONLY genuinely verified ids (PHASE0 §3 semantics). Putting
  skipped ids there would be exactly harness hack #1.
- SKIPPED ids are recorded in `skipped` with explicit reasons (attention/softmax in
  flight; lm_head.commitment_opening deferred to the lm_head stage — an "opening"
  with no matmul to discharge would be vacuous; statement.logit_binding has no
  meaning until lm_head logits exist — this is the reserved final-statement slot,
  which will also carry the served-token-=-argmax binding per THREAT_MODEL_NOTES §1).
- Statement ids covered now: `statement.registered_weight_hash` (every registered
  file re-hashed vs public.json) and `statement.prompt_binding` (run_seed =
  H(public.json) which includes the input digest — the random-input analog of prompt
  ids — and seeds every FS transcript; the verifier derives it independently).
- `check_transcript.py` is run twice and both results reported: against the FROZEN
  full manifest (stage 1 honestly FAILS coverage with exactly the skipped non-waived
  ids — printed, not hidden) and against `manifest_stage1.json` (a generated copy
  with the skipped ids marked waived `"stage1: attention/softmax/lm_head pending"`,
  the harness file untouched) which must PASS — proving format compatibility and
  exact accounting of declared scope. No quiet narrowing: the report leads with the
  full-manifest gap.

## 7. Selftest (selftest.sh)

(a) register → prove_walk → verify_walk on real llama-68m weights; expect verdict
ACCEPT, all 30 covered ids in `checked`, check_transcript(stage-1 manifest) exit 0.
(b) Tamper one byte of a chained commitment file
(`proofs/layer0.mlp.swiglu/glu/com_H.bin`), re-run verify_walk → REJECT with
`layer0.mlp.swiglu` ok=false (edge M7 and/or the glu transcript divergence); restore.
(c) Tamper one registered-weight sha256 inside public.json → verify_walk REJECT at
the registration check (`statement.registered_weight_hash` ok=false) before any
driver runs; restore. PASS/FAIL line per phase, final ALL PASS.
