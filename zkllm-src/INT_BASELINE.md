# INT_BASELINE — the real end-to-end integer prover (zkob/BLS12-381), resurrected

> STATUS: COMPLETE (2026-07-15). Part 1 archaeology, Part 2 rebuild + single-op
> benches + full end-to-end walk rerun (VERDICT ACCEPT), Part 3 corrected
> comparison. Headline: at matched toy shapes the fp8 zk=1 prover costs
> **0.2–3.0x** the REAL measured BLS integer prover (fp8 is often *cheaper*),
> an estimated **~2–134x** vs the same integer pipeline ported to Goldilocks,
> and **~10²·⁷–10³·⁶x** vs zkLLM's published at-scale numbers — the audit's
> 10²–10³x claim survives only as that literature extrapolation, not as a
> local measurement. See §3.5.

Purpose: replace the circular "integerized model" estimate that
`COMPARISON_AUDIT.md` (read it first) found in `bench_saturated.py` — a
denominator that was 92–98% our own fp8 pipeline's gadget floor — with an
HONEST integer baseline measured on the repo's real integer prover: the
zkob/BLS12-381 stack (zkLLM-derived: row-wise Pedersen commitments + sumcheck +
tlookup) plus the orchestrator walk over real llama-68m, built BEFORE the fp8
work, with tailor-made integer gadgets for every op class.

All numbers RTX 4090 unless stated. Sources are cited per table; raw June-2026
walk artifacts (transcripts, prove manifests, selftest logs) are archived under
`zkllm-src/zkorch_archive/` (copied from `/root/zkorch` on 2026-07-14).

---

## Part 1 — archaeology: every previously-measured zkob/orchestrator number

### 1.1 The orchestrator walk timeline (all measured, real llama-68m weights, seq 1024)

llama-68m = JackFram/llama-68m: 2 layers, d=768, dff=3072, 12 heads (dh=64),
vocab 32000, seq 1024 prefill. "Walk" = prove/verify every manifest obligation
of the forward pass with byte-equality/homomorphic chaining between obligations.

| stage (date) | scope | prove | verify | proof size | source |
|---|---|--:|--:|--:|---|
| Stage 1 (06-10) | 2 layers, **MLP+norms+skips only; attention UNPROVEN** (30/56 manifest ids; 26 skipped) | **148.4 s** (driver; 152.9 s wall) | 149.9 s | 15.9 MiB | ORCHESTRATOR_REPORT.md |
| Stage 2 (06-11) | + full attention chain (54/56 ids; rope, headslice, per-head scores/softmax/values, headmerge) | **743.0 s** | 1356 s (GPU-contended; ≈17–18 min exclusive est.) | 143.6 MiB | ORCHESTRATOR_REPORT.md §stage-2 |
| Stage 3 (06-11) | manifest closed: + final_norm, lm_head (768×32000), logit binding; 59 ids | **875.5 s** | — | 154.4 MB | ORCHESTRATOR_REPORT.md §stage-3 |
| faithful-arch-v1, inline transport (06-11) | full walk, 65 obligations, ACCEPT (o_proj added, rowmax+softmax8 per head) | **1062.47 s** | 999.5 s single-process wall (1999 s as sum of 235 per-driver one-shot verifies) | 175.6 MB | raw: zkorch_archive/stage3v2-fa/prove_manifest.json; STAGE_C2_REPORT.md; BACKEND_DECISION.md |
| Stage C2, batched transport (06-12) | same 65-obligation walk; claim accumulators + 7 batched openings replace 1,535 inline IPAs; DriverPool single-process verify; fast-G1 helpers | **522.0 s** | **27.1 s** | 176.3 MB | STAGE_C2_REPORT.md §1 |
| Stage D wpriv walk (06-12) | + weight privacy (hiding Pedersen registration, committed-round sumchecks, ZK blinded IPA); ACCEPT 65/65, leak scan CLEAN | **526.3 s** | **30.4 s** | 176.4 MB | raw: zkorch_archive/wprivconfirm-fawp/{prove_manifest,transcript}.json |

**Reconciliation of the figures the task asked about:**

- "**~17.7 min prove / 16.7 min verify** full llama-68m seq-1024" = the
  faithful-arch-v1 **inline-transport** walk: 1062 s ≈ 17.7 min prove, 999.5 s
  ≈ 16.7 min verify. Scope: ALL 65 obligations — attention fully proven
  (rope + per-head rowmax/softmax8/scores/values + headmerge + o_proj),
  final_norm, lm_head, logit binding. Nothing waived.
- "**C2 baseline ~522 s prove / 27 s verify / 176 MB**" = the SAME scope after
  the transport rebuild (batching + single-process verifier + fast-G1
  scheduling; no protocol change). 2.03× prove, 36.9× verify vs inline.
- The **148.4 s** stage-1 number is 2 layers MLP+norms ONLY (attention
  unproven at that stage) and must not be quoted as a full-model cost.
- Weight privacy costs +0.9% prove / +3.4 s verify / +0.08 MB on top of C2
  (WRITEUP.md §2; raw manifest confirms 526.3 s / 30.4 s / 176.4 MB).

Per-obligation detail (stage-2 exclusive-ish, per instance at seq 1024, d=768):
scores fc 1.58 s/head, softmax(temp-128) 10.59 s/head, values fc 1.33 s/head,
rope 2.29 s, headslice 29.6 s, headmerge 3.79 s, rmsnorm site 14.2 s
(3 sub-proofs incl. rescales). Faithful-arch per-driver averages in the wpriv
walk (claim mode, batched): fc 1.04 s (n=63), rescale 1.48 s (n=87), rowmax
3.34 s (n=25), softmax8 3.42 s (n=24), glu 5.43 s (n=2), rmsnorm 1.22 s (n=5),
rope 0.90 s (n=4), batchopen 15.2 s (n=7) + weight batch 8.1 s (n=1).

### 1.2 Single-FC benchmarks (STAGE_D_REPORT.md §6, llama-68m gate_proj scale B=64, IN=768, OUT=3072)

| step | plain (C2) | weight-private (D) |
|---|--:|--:|
| registration (one-time) | 0.91 s | 0.91 s |
| driver prove | 1.09 s | 1.38 s |
| driver verify | 0.29 s | 0.44 s |
| public batch prove / verify | 1.39 / 0.57 s | 1.07 / 0.56 s |
| weight batch prove / verify | — | 1.76 / 0.74 s |
| TOTAL prove / verify | 2.48 / 0.85 s | 4.21 / 1.74 s |
| proof bytes (this obligation) | 11,815 | 28,506 |

Each step is one CLI invocation and carries a ~0.3 s process+CUDA-init floor
(removed at walk scale by the C2 serve-mode DriverPool).

### 1.3 Selftest expectations (STAGE_D_REPORT.md §4, 2026-06-12 battery)

13-TU battery ALL PASS: batchopen 33/33 public + 26/26 wselftest, fc 15/15
(incl. wpriv), rescale, skip, glu, rope, headmerge, headslice, rmsnorm 11/11,
softmax, softmax8, rowmax, toy-batchopen. Independent as-built soundness audit
(REBUILD_AUDIT.md): batching SOUND, weight privacy SOUND, tamper localization
SOUND.

### 1.4 What the old system is (and is not)

- **Arithmetic**: int32/int64 fixed-point (residual stream 2^16), BLS12-381
  Pedersen row commitments, sumcheck + tlookup gadgets — the zkLLM (CCS'24)
  protocol family, re-engineered with chained obligations.
- **ZK status**: weight privacy REAL (hiding commitments, committed-round
  sumchecks, ZK blinded IPA; leak scan clean, audited). Activations are the
  public statement (deterministic commitments) — NOT activation-private, by
  design. The fp8 side's zk=1 hides activations too, so at layer level fp8
  zk=1 is being compared against a *weight-private-only* integer prover; the
  measured fp8 zk premium is 2.02–2.22× (BENCH_ZK_AT_SCALE.md §C) if one wants
  to discount that asymmetry.
- **Semantics**: proves the integerized model M_int exactly (DiFR of
  integerization+proof pipeline = 1.35e-6 nats vs its float target,
  WRITEUP.md) — i.e. this is a REAL "integerized model" prover of the class
  our synthetic estimate pretended to be.

---

## Part 2 — rebuild + rerun (2026-07-14, RTX 4090, system CUDA, sm_89)

### 2.1 Rebuild + selftests

Sources copied to `/root/zkob_build`, built per `build_zkob.sh` conventions
(`nvcc -arch=sm_89 -std=c++17 -dc -dlto`, log: `/root/zkob_build.log`).
Binaries: ppgen + zkob_{fc, batchopen, rescale, rmsnorm, glu, rowmax,
softmax8, rope, headslice, headmerge, skip, softmax}. pp: registered June
generators (gen64/1024/4096 + q.bin sha-pinned in the walk registration)
reused; missing power-of-2 sizes (128–8192) generated with ppgen.

Selftest battery (2026-07-14, all logged in `/root/zkob_bench.log` shell
history): **ALL PASS** — fc 11/11 (apriv mode incl. weight privacy),
rescale 10/10, rmsnorm 11/11 (wpriv), batchopen 33/33, glu 11/11,
rowmax 17/17, softmax8 17/17, rope 13/13, headslice 12/12, headmerge 12/12,
skip. Matches the June STAGE_D battery expectations (§1.3).

### 2.2 Hard instance floors (a real property of the integer prover)

Every tlookup gadget requires `table_len <= padded instance size` (table and
witness share the sumcheck domain), and the table sizes are intrinsic to the
registered fixed-point format (residual scale 2^16, gate scale 2^12):

| gadget | table | minimum instance |
|---|---|---|
| rescale sf16 / sf20 | 2^16 / 2^20 remainders | 2^16 / 2^20 elements |
| swiglu (glu) | 2^22 silu entries | 2^22 elements (= 32768 rows at dff=128) |
| softmax8 + rowmax | 2^20 exp/limb domain | B == NCOL ≥ 1024 (per head!) |
| rope | pairwise head rotation | NH = d/64 ≥ 2 (d ≥ 128) |

So at our small sweep shapes (seq 64–512, d 64) the prover MUST row-pad each
such obligation up to the floor: e.g. an s64 layer's softmax costs the full
1024×1024 grid, and its swiglu costs a 2^22-element instance. All benchmarks
below run at the padded shape (that IS the zkob cost of the obligation; the
gadgets were built for llama-68m at seq 1024). Shrinking the tables would
change the fixed-point semantics — a different, unregistered statement.

### 2.3 Single-op measurements (2026-07-14/15, plain mode, standalone CLI)

Full set: 104 measurements in `/root/zkob_bench_results.json` (log:
`/root/zkob_bench.log`); every invocation mirrors the walk's argv conventions.
Mode: **plain** (weight commitments public, activations public). This is the
integrity-optimal mode; weight privacy adds +0.9% prove at walk level
(§1.1) and ~+0.3 s per standalone FC (§1.2), so plain numbers understate the
weight-private cost only marginally. Each standalone run carries a ~0.3 s
process+CUDA-init floor (removed at walk scale by the serve-mode pool).
Representative rows:

| op (B,dims) | prove s | verify s | proof KB | note |
|---|--:|--:|--:|---|
| fc(64,64,64) | 0.91 | 0.65 | 24 | matmul, s64/p64 qkvo shape |
| fc(64,128,128) | 1.03 | 0.71 | 25 | p128 qkvo |
| fc(64,256,256) | 1.17 | 0.85 | 26 | p256 qkvo |
| fc(64,512,512) | 1.37 | 1.09 | 27 | p512 qkvo |
| fc(64,512,2048) | 1.86 | 2.09 | 28 | p512 gate/up |
| fc(1024,64,64) | 0.92 | 0.68 | 294 | s1024 qkvo |
| fc(2048,64,64) | 0.93 | 0.69 | 582 | b16-stacked qkvo |
| fc(8192,64,64) | 1.05 | 0.75 | 2310 | b64-stacked qkvo |
| fc(16,768,3072) | 2.56 | 3.97 | 15 | STAGE_D anchor (cf. §1.2: 1.09 s driver-only + floor) |
| fc(64,768,3072) | 2.57 | 3.72 | 29 | llama gate/up, 64 tok |
| fc(1024,768,768) | 1.76 | 1.60 | 298 | llama qkvo, seq 1024 |
| rmsnorm(64,64) | 3.78 | 2.12 | 506 | s64/p64 site (3 sub-proofs incl. rescales) |
| rmsnorm(64,512) | 4.79 | 3.01 | 511 | p512 site |
| rmsnorm(1024,64) | 6.99 | 5.49 | 655 | s1024 site |
| rmsnorm(8192,64) | 26.86 | 37.39 | 4677 | b64-stacked site |
| rmsnorm(1024,768) | 8.95 | 7.42 | 663 | llama site |
| rescale(1024,64,sf16) | 0.99 | 0.75 | 727 | sf16 table floor at C=64 (§2.2) |
| rescale(64,2048,sf16) | 2.73 | 2.62 | 52 | p512 h-rescale |
| rescale(1024,768,sf16) | 2.19 | 1.65 | 596 | llama rescale |
| glu(32768,128) | 4.32 | 3.29 | 27666 | swiglu 2^22 floor at dff=128 |
| glu(8192,512) | 4.59 | 2.62 | 6933 | dff=512 |
| glu(4096,1024) | 5.28 | 3.40 | 3479 | dff=1024 |
| glu(2048,2048) | 6.58 | 5.38 | 1753 | dff=2048 |
| glu(1024,3072) | 9.00 | 9.58 | 890 | llama dff=3072 |
| rowmax(1024) | 5.25 | 2.27 | 773 | per head (1024-grid floor) |
| softmax8(1024) | 11.59 | 8.81 | 2091 | per head (1024-grid floor) |
| rope(64,128) | 1.06 | 0.72 | 28 | p128 |
| rope(64,512) | 1.46 | 1.10 | 30 | p512 |
| headslice(64,512) | 14.01 | 9.32 | 347 | p512 |
| headmerge(64,512) | 2.40 | 1.53 | 110 | p512 |
| skip(1024,64) | 0.31 | 0.33 | 144 | s1024 skip-add (homomorphic, near-free) |

Cross-check vs June: fc at the STAGE_D anchor and rmsnorm/glu at the walk
shapes are within ~10–25% of the §1.1/§1.2 figures (same binaries rebuilt;
standalone floor included here).

### 2.4 End-to-end walk RERUN (2026-07-15) — the anchor, re-measured

Full faithful-arch-v1 walk over real llama-68m (2 layers, d=768, dff=3072,
12 heads, seq 1024), 65 obligations, **weight privacy ON** (hiding
registration + committed-round sumchecks + ZK weight sub-batch), batched
transport, rebuilt binaries. Logs: `/root/zkob_walk_prove4.log`,
`/root/zkob_walk_verify.log`; transcript `/root/zkorch/rerun2-fawp/`.

| metric | 2026-07-15 rerun | June baseline (§1.1 wpriv) |
|---|--:|--:|
| prove wall | **407.71 s** (driver 406.44 s) | 526.3 s |
| verify | rc=0, **VERDICT ACCEPT**, checked=65 rejected=0; per-step sum 28.8 s | 30.4 s |
| proof size | 176,411,934 B = **176.4 MB** | 176.4 MB |

The rebuilt binaries are ~23% faster at identical proof bytes and identical
scope. Prove-time split: layer0 **148.5 s**, layer1 **160.8 s** (each incl.
its opening sub-batches b0–b4, 16.8/17.1/13.4/16.9/15.4 s), head 97.2 s
(logit-binding rowmax 39.5 s, lm_head rescale 16.0 s + fc 7.3 s, final_norm
2.5 s, b5+b6 24.2 s), weight sub-batch 7.7 s. So: **≈155 s per layer body**,
or **≈204 s/layer** with the final_norm/lm_head/logit head amortized in
(407.71/2). Attention is fully proven (rope, headslice, 12× per-head
scores/rowmax/softmax8/values, headmerge, o_proj).

**Binary-provenance lesson (cost us one failed rerun):** the first rerun
attempt failed round-0 transcript inconsistency because `common.py` pointed
`ZKLLM=/root/zkllm` at the *June* binaries while registration had used the
*rebuilt* set — Fiat–Shamir transcripts are binary-set-specific, so
registration, prove and verify must all run one identical binary set. Fixed
by installing the rebuilt binaries into `/root/zkllm` (June set preserved in
`/root/zkob_jun_backup`). Any future zkob work: check which set `common.py`
resolves BEFORE registering.

---

## Part 3 — the corrected comparison

### 3.1 "Integer layer" totals at our sweep shapes, assembled from measured zkob ops

Assembly rule (mirrors the walk's real obligation structure; A = d/64 heads,
T = batch·seq tokens): per layer **2×** rmsnorm site (each incl. w/y
rescales) + **4×** fc(T,d,d) qkvo + **5×** sf16 rescale (qkvo + down) + rope +
(A>1: headslice + headmerge) + **per head-instance (batch·A)**: scores
fc(seq,64,seq), sf13 rescale, rowmax, softmax8, values fc(seq,seq,64), sf10
rescale + **2×** fc(T,d,dff) gate/up + 2× sf20 rescale + glu + sf16 hrescale +
fc(T,dff,d) down + 2× skip. Every term is a measured row from §2.3 at the
walk's registered scales, floor-padded per §2.2 (that IS the zkob price of
the obligation); the only extrapolated rows are rope/skip at the b16/b64
stacked token counts (≤0.5% of the total). **Attention IS included** — zkob
has native attention gadgets (rope/rowmax/softmax8/headslice/headmerge) and
the attention matmuls are FC calls at per-head granularity, exactly as the
walk proves them. Assembly script: `/root/zkob_assemble.py`, output
`/root/zkob_layer_totals.json`.

| cfg | shape (seq,d,dff,batch) | ops | int prove s | verify s | proof MB | attn share |
|---|---|--:|--:|--:|--:|--:|
| s64 | 64,64,128,1 | 27 | **50.0** | 33.9 | 48.6 | 51% |
| s128 | 128,64,128,1 | 27 | **51.3** | 34.7 | 48.5 | 49% |
| s256 | 256,64,128,1 | 27 | **53.0** | 36.2 | 49.0 | 43% |
| s512 | 512,64,128,1 | 27 | **55.5** | 38.8 | 50.3 | 42% |
| s1024 | 1024,64,128,1 | 27 | **60.0** | 44.0 | 53.3 | 42% |
| b4 | 128,64,128,4 | 45 | **117.0** | 79.4 | 59.2 | 73% |
| b16 | 128,64,128,16 | 117 | **376.9** | 258.3 | 107.0 | 89% |
| b64 | 128,64,128,64 | 405 | **1420.0** | 984.6 | 304.6 | 95% |
| p64 | 64,64,128,1 | 27 | **50.0** | 33.9 | 48.6 | 51% |
| p128 | 64,128,512,1 | 35 | **78.1** | 51.5 | 19.7 | 59% |
| p256 | 64,256,1024,1 | 47 | **127.6** | 86.1 | 20.1 | 71% |
| p512 | 64,512,2048,1 | 71 | **226.4** | 157.3 | 29.9 | 81% |

Caveats, all in the direction "real zkob at walk scale would be cheaper than
these sums":

1. **Standalone floors**: each row carries ~0.3 s process+CUDA-init (~8 s at
   s64) and inline IPA openings. Measured batched-vs-standalone pairs at the
   llama shapes (walk serve-mode driver averages §1.1 vs §2.3 rows):
   softmax8 3.42 vs 11.59 s, rmsnorm site 2.59 vs 8.95 s, glu 2.86 vs 9.00 s,
   rowmax 3.34 vs 5.25 s, fc ~0.6–0.7×, rescale ~0.7× — i.e. **walk-style
   batched transport ≈ 0.4–0.5× these sums** (opening sub-batches included;
   cross-checked below against the actual walk). Batched layer estimates ≈
   half the table; the fp8 premium doubles correspondingly.
2. **Table floors dominate at toy shapes** (§2.2): at s64, softmax8+rowmax
   (1024-grid) + glu (2^22) alone are 21.2 s of 50.0 s. These floors are
   *saturated* (no waste) at the llama-68m walk shape, so toy-shape ratios
   are the floor-heavy worst case for zkob.
3. **b-configs**: assembled per-sequence-per-head (walk granularity, 64
   attention blocks at b64). Row-stacking sequences into shared 1024-grids
   could cut the b64 attention term up to ~8x; not implemented in zkob.
4. **Privacy asymmetry**: rows are plain mode; weight privacy adds +0.9%
   (walk-measured). zkob hides weights only — fp8 zk=1 hides activations too
   (fp8's own zk premium: 2.02–2.22x, BENCH_ZK_AT_SCALE.md §C).

Walk cross-check: an assembly at the llama-68m layer shape from §2.3 rows
(with stage-2 values for the unmeasured headslice/headmerge-at-1024 rows)
gives ~350–430 s/layer standalone vs **155 s/layer measured in the batched
walk** (§2.4) — consistent with caveat 1's 0.4–0.5× factor.

### 3.2 The corrected premium table (fp8 zk=1 vs REAL integer prover)

fp8 zk=1 from `/root/zkllm/bench_sat.json` (`fp8_s`, Goldilocks Hawkeye,
composed single-proof layer, activations+weights hidden). PRIMARY denominator
for overhead-vs-native on both sides: `fwd_native_ms` (plain bf16 GEMM
forward). The old synthetic `int_s` (the circular denominator
COMPARISON_AUDIT.md §F1 flagged) shown struck for reference.

| cfg | fp8 zk1 s | zkob BLS s | **premium fp8/int (measured)** | fp8/native | zkob/native | old int_s | old premium |
|---|--:|--:|--:|--:|--:|--:|--:|
| s64 | 10.0 | 50.0 | **0.20** | 3.9e6 | 1.9e7 | 1.3 | 7.7 |
| s128 | 18.0 | 51.3 | **0.35** | 3.1e6 | 8.9e6 | 2.8 | 6.4 |
| s256 | 32.2 | 53.0 | **0.61** | 1.9e6 | 3.1e6 | 4.1 | 7.8 |
| s512 | 62.0 | 55.5 | **1.12** | 1.1e6 | 9.9e5 | 7.4 | 8.4 |
| s1024 | 179.0 | 60.0 | **2.98** | 8.9e5 | 3.0e5 | 24.7 | 7.2 |
| b4 | 53.5 | 117.0 | **0.46** | 2.3e6 | 5.1e6 | 8.4 | 6.4 |
| b16 | 164.9 | 376.9 | **0.44** | 1.8e6 | 4.1e6 | 31.6 | 5.2 |
| b64 | 981.8 | 1420.0 | **0.69** | 2.7e6 | 3.9e6 | 129.6 | 7.6 |
| p64 | 10.2 | 50.0 | **0.20** | 3.9e6 | 1.9e7 | 1.3 | 7.7 |
| p128 | 29.0 | 78.1 | **0.37** | 4.8e6 | 1.3e7 | 2.5 | 11.8 |
| p256 | 65.0 | 127.6 | **0.51** | 5.1e6 | 1.0e7 | 3.7 | 17.4 |
| p512 | 239.2 | 226.4 | **1.06** | 7.7e6 | 7.2e6 | 5.8 | 41.5 |

Readings:

- **Measured premium of exact-fp8 over the real BLS integer prover: 0.20–2.98x
  at matched shapes** — at 9 of 12 configs fp8 zk=1 is *cheaper* than the
  integer prover it was being compared against. With the batched-transport
  adjustment (caveat 3.1.1, zkob ≈ 0.5×), the premium is ≈ **0.4–6.0x**.
  Either way it is nowhere near the old synthetic table's shape (flat 5–8x on
  s/b, rising to 41x on p) — the real integer prover's floors invert the
  small-shape end entirely.
- The premium *rises with both seq and d* (0.20 → 2.98 along s, 0.20 → 1.06
  along p): zkob's floor-padded costs are near-flat while fp8 grows with
  work, so at shapes past the floors (seq ≥ ~512, d ≥ ~512) fp8 becomes the
  more expensive side, as expected.
- Overhead vs native forward: **fp8 zk=1 = 0.9–7.7×10⁶x** (rising with d);
  **zkob BLS = 3.0×10⁵–1.9×10⁷x** (falling as floors saturate). Secondary
  Triton-denominator numbers (`fwd_eff_ms`, the saturated Hawkeye forward,
  itself 5.8–60x slower than native bf16 GEMM at these shapes and worse at
  scale): fp8 reads 1.3–6.7×10⁵x — flattering by exactly that kernel-gap
  factor; use the native column.
- End-to-end anchor at real-model scale: llama-68m, seq 1024, everything
  proven, weight-private: **407.7 s prove / 28.8 s verify / 176.4 MB ≈
  155 s per layer body (204 s/layer amortized)** on BLS12-381 (§2.4).
- Proof sizes: zkob standalone sums are 20–305 MB/layer, but batched
  transport dedups openings — the full 2-layer model is 176 MB total.

### 3.3 SUBSTRATE CAVEAT, quantified — the Goldilocks-ported bracket

zkob pays for a 256-bit pairing curve. Our measured June substrate factors:
**Goldilocks field mul 11.9x faster than BLS12-381 scalar mul; hash-based
commit 45x faster than Pedersen at 2^22 elements.** ESTIMATE (not a
measurement): the same integer pipeline ported to the Goldilocks substrate
would cost between `zkob/45` (if commit-bound) and `zkob/11.9` (if
field-op-bound; the sumcheck-heavy gadgets mostly are). The fp8 premium vs
this *ported integer prover* is then bracketed by 11.9×–45× the measured
column:

| cfg | premium vs BLS (measured) | vs ported, /11.9 (EST) | vs ported, /45 (EST) |
|---|--:|--:|--:|
| s64 | 0.20 | 2.4 | 9 |
| s1024 | 2.98 | 35.5 | 134 |
| b64 | 0.69 | 8.2 | 31 |
| p128 | 0.37 | 4.4 | 17 |
| p512 | 1.06 | 12.6 | 48 |
| full range | **0.20–2.98** | **2.4–35.5** | **9–134** |

(Batched-transport adjustment roughly doubles all three columns; the two
corrections partially offset within the bracket's width.) Ironically, the
old circular 8–40x headline **lands inside this ported-integer bracket** —
the synthetic estimate got the right decade for the wrong reasons: its
denominator wasn't an integer prover, but a Goldilocks-substrate gadget
floor is apparently not a terrible proxy for a Goldilocks-ported integer
prover at these shapes.

### 3.4 Reconciliation with COMPARISON_AUDIT.md's 10²–10³x claim

The audit claimed the honest fp8 premium "against a *competitive* published
integer prover is plausibly 10²–10³x", calibrated on zkLLM (CCS'24: LLaMA-2
13B, 40 layers, seq 2048, <15 min on one A100 ≈ 22 s/layer ≈ 2×10³x native).
What the resurrected real-prover data says:

1. **NOT supported by anything measurable locally.** Against the only real
   integer prover we can run — zkob, itself zkLLM-derived with tailor-made
   integer gadgets — the measured premium is 0.2–3.0x (standalone) / ~0.4–6x
   (batched), and vs a Goldilocks port an estimated ~2–134x. All at least
   one order below 10².
2. **The gap is on the integer side, as the audit itself predicted (F1):**
   zkob at the walk anchor runs ~155 s/layer at d=768/seq 1024 ≈ 10⁵·⁵–10⁶x
   native — 2–3 orders above zkLLM's published 2×10³x, despite the shared
   protocol family. Where it goes: toy/68m shapes under-fill the fixed
   tlookup floors, per-head chained-obligation granularity (26 obligations
   per attention layer vs zkLLM's aggregated proofs), byte-exact rescale
   obligations everywhere, and a consumer GPU. So our repo's integer prover
   is real but NOT competitive-at-scale; the audit's "25–170x cheaper
   competitive floor" is precisely the part no local measurement reproduces.
3. **The claim survives only as a literature extrapolation, and the data is
   consistent with it at face value:** fp8 zk=1 at 0.9–7.7×10⁶x native vs
   zkLLM's ≈2×10³x native gives **~450–3,850x ≈ 10²·⁷–10³·⁶x** — inside the
   audit's band. But that comparison crosses hardware (4090 vs A100), scale
   (d=64–512 toy layers vs d=5120), and measurement provenance (ours vs
   paper-reported). Nothing in this repo measures a 10²x-or-worse premium.

Verdict: the audit was right to kill the synthetic denominator (it was
circular) and right that the 8–40x headline is not "fp8 ≈ integer" — but its
replacement number was an extrapolation, and the real prover it asked for
lands far below it. The corrected, fully-measured statement is §3.5.

### 3.5 Bottom line (what we can now say honestly)

- Against the **real, measured, end-to-end BLS12-381 integer prover in this
  repo** (weight-private, attention included, ACCEPT-verified at model
  scale): exact-fp8 zk=1 costs **0.2–3.0x** at matched toy shapes (≈0.4–6x
  after batched-transport adjustment), premium rising with seq and d.
- Against the **same integer pipeline ported to our Goldilocks substrate**
  (ESTIMATE from measured 11.9x field / 45x commit factors): **~2–134x**.
  The retired 8–40x headline sits inside this bracket.
- Against a **competitive published integer prover at production scale**
  (zkLLM's numbers taken at face value): **~10²·⁷–10³·⁶x** — the audit's
  10²–10³x band, but this is a literature calibration, not a measurement;
  no integer prover we can run gets within 2 orders of that efficiency at
  these shapes.
- Integer-side absolutes (this repo, RTX 4090, BLS12-381): **~50–1,420 s per
  toy layer** standalone (table 3.1), **407.7 s prove / 28.8 s verify /
  176 MB** for the full weight-private llama-68m walk, ≈ **10⁵·⁵–10⁷x**
  native overhead.
