# P3 prover improvement roadmap (living document — the improvement loop works this file)

Status values: TODO / IN-PROGRESS / DONE(measured win) / REVERTED(reason) / INFEASIBLE(reason).
Every DONE item must cite its measured win and the gate run that validated it.
Agents append new proposals each iteration under "Proposed"; the coordinator
promotes them to TODO or rejects them.

## TODO (ranked by expected impact)

1. **[memory, BIG] Disk-backed Packed store** — DONE(unlocked 8192 tokens:
   s128 b64 zk=1 = 1176.7 s / 37.5 GB / verify_ok=1, was cgroup-kill at
   35.4 GB before any output; witness build 35.4 → 4.8 GB).  Root cause
   measured: glibc arena retention of per-instance transients in
   build_witness (~15 GB/64 attention instances) plus raw lidx retention
   (~16 B/product).  Implemented: trim_heap per instance/section in
   build_witness, lidx spill at compact_wit (LayerWit.limap), spill
   default-on (/workspace/p3_spill; P3_PK_SPILL=0 disables), min 1 MB.
   Gates: iteration-1 endgame run (see IMPROVEMENT_LOG).
2. **[speed, BIG] logUp GPU offload** — DONE(measured −25.6% end-to-end at
   d256 s128: 214.8 → 159.9 s, RSS 16.2 GB unchanged, proof bytes identical;
   lug/Am 51.4 → 4.3 s).  Device Am build (p3lu_amfill/amaxpy kernels) +
   mat_col_range_dev in the claims path; kill switch P3_LUG_DEVAM=0.
   Identity pairs green (42.658/41.569).  Full gates at end of iteration.
3. **[speed, MED] Batched sumcheck round messages** — DEFERRED(measured low
   headroom): after item 2, the per-round reduction targets are zcdp+zcdg =
   9.6 s of 159.9 s at d256 s128 and 24.3 s of 644 s at 4096 tok (71 chains),
   and much of zcdp is real fold work, not launch overhead.  Even a perfect
   batching saves <2-3%; the mm stage is 73% cwit (commits) instead — see the
   packed-direct commit lever below.
4. **[speed, MED] Commit/witness-gen overlap** — DONE via a stronger,
   simpler form: packed-direct commits (`commit_pk_nc`) — the cpk_dev path
   host-rematerialized the already-packed column, re-packed it and uploaded
   raw; now the pack itself uploads (2-8x less PCIe) + device unpack.
   Applied to the CDp loop.  Measured: d256 s128 159.9 → **151.3 s (−5.4%)**,
   proof bytes identical, mm 72.3 → 63.4 s.  CAVEAT: RSS 16.2 → 18.4 GB
   (+2.2 GB) — regression investigation proposed below.  Kill switch
   P3_CPK_DEV=0.  Stream overlap proper: see Proposed.
5. **[speed, MED] Batch-open eq-rebuild caching** — INFEASIBLE(no headroom):
   the strG round-loop eq rebuild sits inside bo/red = 1.1 s of 159.9 s at
   d256 s128 (19 classes).  The batch stage's actual costs are q0/enc 17.7 s,
   bo/G 10.6 s, ys+rlc 10.2 s, blinder 9.3 s — different levers (proposed
   below).
6. **[speed/memory, MED] mm-stage chain scheduling** — ASSESSED, superseded by
   Proposed P6 with the measured 8192-token profile: per-chain fixed cost is
   sc5z/chain 127.4 s across x4040 chains (~31 ms/chain, 10.8% e2e) at 8192 tok,
   52.3 s x1064 at 4096 tok, plus sc5z/blind 52.2 s at 8192.  Real headroom at
   batch configs, negligible at d256 s128 (8.9 s).  See P6 for the concrete
   lever and risk class.

## Explicitly OUT OF SCOPE for the autonomous loop

- **Naive Binius/GF(2^128) swap into the layer prover** — cross-field binding
  seams make it UNSOUND; it needs supervised protocol design (the sound bridge
  is a research task, not an implementation task). Drafting a design document
  for the sound bridge IS allowed as a proposal.
- Anything that weakens ZK hiding or soundness gates.

## Proposed (new items land here each iteration)

### Iteration 1 proposals (2026-07-13, from the i1c final profiles — logs
### zkrun_i1c_d256s128 / _d256s256 / _s256b16, zkrun_i1_s128b64_v2)

Post-iteration-1 stage picture at d256 s128 (151.3 s): mm=63.4 (of which
cwit≈35.3 s = 56%), batch=57.5 (of which the single 1 GB-column class tf-bo14
is 29.3 s), lug=24.1 (of which hcommit≈inv=11.4 s).  Global across stages:
commit_salt 16.0 s + mask_gen 10.2 s = 26.1 s (17% e2e).

P1. **[memory, UNLOCK, rank 1] Per-instance QKV witness compaction — unlock
    16384 tokens.**  Measured mechanism of the 16384 kill (exit=137 at 41 s,
    only `# rss 0.2 GB at wit:rms1` logged): the three Wq/Wk/Wv
    p3hwl::gen_witness results coexist RAW until the compact_wit loop
    (p3_transformer.cuh:288).  8192 tok measured 21.5 GB transient in this
    window; 16384 doubles P to 2^27/instance ≈ 3×14 GB ≈ 43 GB > 41 GB cap.
    The spill engages only at compact_wit — never reached.  Fix: gen →
    compact_wit → trim_heap per matmul instance (and same interleave in the
    wo/ffn sections); peak drops to ~1 raw + 2 packs ≈ 16 GB.
    Transcript-identical (packing is representation-only; identity pairs
    suffice).  Risk: LOW for the witness wall; MEDIUM for the full config —
    the 16384 PROVE phase is unmeasured and the 8192 prove already peaked at
    37.5 GB, so prove-phase spill coverage may also be needed.  Confidence:
    high (wall mechanism), medium (end-to-end unlock).

P2. **[memory, BIG, rank 2] cpk_dev/devam host-RSS regression.**  Measured:
    +2.2 GB at d256 s128 (16.2→18.4), +6.3 GB at d256 s256 (30.7→37.0, now
    4 GB from the cap), +2.9 GB at 4096 tok.  Hypothesis: the pack now stays
    host-resident through the device unpack and the classic path's early free
    is skipped (see iteration-1 packed-direct notes); devam staging buffers
    second.  Fix: free/spill the pack immediately after upload; measure each
    contributor with P3_CPK_DEV=0 / P3_LUG_DEVAM=0 bisection (both
    bytes-identical kill switches, gate pairs already green).  Estimated win:
    reclaim most of +6.3 GB at s256 (−17% RSS) keeping the −5.4% cwit time.
    Transcript-identical.  Confidence: high that most is reclaimable.

P3. **[speed, BIG, rank 3] Chunked device pass for giant batch-open classes.**
    The strGdev gate (p3_batchopen.cuh:407, (T+2)*colbytes <= 0.92*devfree)
    rejects the biggest class everywhere it matters: d256 s128 tf-bo14
    (1 GB cols, need 21.0 vs 20.8 GB budget — misses by 200 MB) runs the host
    path at 29.3 s = 51% of the batch stage; d256 s256 tf-bo13 (2 GB cols,
    need 42.0 GB) costs 81.4 s = 62% of batch.  The 512 MB class that DID take
    strGdev runs ~4x cheaper per byte (tf-bo16: 3.8 s for 512 MB vs tf-bo14:
    29.3 s for 1024 MB).  Fix: tile the strGdev round-0/G pass over T in
    column chunks sized to devfree — same G values, same transcript.
    Estimated win: ~−20 s at d256 s128 (−13% e2e), ~−50 s at d256 s256
    (−14% e2e), large at 8192 tok (bo/G+red+ys = 152 s).  Risk:
    transcript-identical if the chunked accumulation reproduces exact field
    op order; watch host staging RSS (comment at p3_batchopen.cuh:395 records
    a prior d1024 breach).  Confidence: medium-high.

P4. **[speed, MED-BIG, rank 4] Device batched inversion for logUp helper
    columns.**  lug/inv ≈ lug/hcommit in every profile (11.415 vs 11.460 s at
    d256 s128; 26.3/26.3 at 4096 tok; 40.9/40.9 at 8192) — the helper-commit
    wall time is essentially all field inversions, host-side.  Move to device
    Montgomery batched inversion (precedent: devam took lug/Am 51.4→4.3 s,
    x137 on the same data shapes).  Estimated win: −7% e2e at d256 s128,
    −5% at 4096 tok, −3.5% at 8192.  Transcript-identical (exact field ops).
    Confidence: medium-high.

P5. **[speed, MED, rank 5] Fused/batched salted commits + device mask PRNG.**
    commit_salt + mask_gen = 26.1 s (17% e2e) at d256 s128 across x3530
    commits, 75.8 s (15.5%) at 4096 tok (x21085), 106.6 s (9.1%) at 8192
    (x80771).  Per-call cost is flat (~7+3 ms) — batch the salt hashing and
    mask PRNG across the columns of one class/group into single kernel
    launches.  MUST preserve the exact salt/mask byte streams and seed order
    (identity pairs are the gate).  Estimated win: half → −6..8% e2e at d=256
    configs, −4% at 8192.  Risk: transcript-identical only if stream order is
    bit-exact — subtle.  Confidence: medium.

P6. **[speed, MED at batch configs, rank 6] Per-chain sumcheck overhead
    (subsumes old item 6).**  sc5z/chain = 127.4 s x4040 chains (10.8% e2e)
    + sc5z/blind 52.2 s at 8192 tok; 52.3 s x1064 + 4.3 s at 4096 tok;
    negligible (8.9 s) at d256 s128.  ~31-49 ms/chain fixed cost: per-chain
    device sync + tiny-launch round loops.  Levers: persistent per-chain
    device contexts, batching round messages across the many same-shape
    attention chains.  A batched implementation likely reorders transcript
    ops → transcript-CHANGING: full battery + compact teeth + hiding suites
    required.  Estimated win: −5..9% at 4096-16384 tok, ~0% at d=256 single.
    Confidence: medium-low (risk-adjusted).

P7. **[speed, MED at batch configs, rank 7] lug/cnt GPU offload.**  87.5 s
    (7.4% e2e) at 8192 tok, 26.8 s (5.5%) at 4096, only 1.5 s at d256 s128 —
    host-side occurrence counting scales with tokens.  GPU histogram
    (atomics or sort-reduce) over the lookup index streams; counts are
    committed data so bytes must match exactly (transcript-identical if the
    same counts come out — order-free).  Estimated win: −6% at 8192, −4% at
    4096.  Confidence: medium.

Near-exhausted (honest negatives):
- Batched round messages (old item 3): zcdp+zcdg = 9.6 s of 151.3 s at
  d256 s128, mostly real fold work — still <2-3% even if perfect.
- Batch-open eq-rebuild caching (old item 5): bo/red = 1.1 s at d256 s128 —
  dead at single configs; at s256 bo/red = 20.2 s but that is the tf-bo13
  host path, addressed by P3, not by caching.
- lug/Am: 4.3 s left at d256 s128 (was 51.4) — devam took the headroom; the
  66.3 s at 8192 tok is x180 calls dominated by small-group host paths below
  the NM=2^16 device threshold; lowering the threshold is a cheap experiment
  but bounded ~2-3% (fold into P4 if touched).
- Forced-cap streaming (P3_SC5ZG_CAP) is 2.1x slower on i1c at seq=64 with
  identical bytes — cosmetic (test env only), fix opportunistically.

## DONE

(2026-07-13 baseline: the five scaling levers, commit c149475 — see SCALING_STUDY.md)

(2026-07-13 iteration 1, binary /root/p3_tb_i1c: devam logUp offload −25.6%,
packed-direct commits −5.4%, disk-backed Packed store unlocking 8192 tokens.
Final gates ALL GREEN incl. identity pairs on both binaries under all 8 lever
envs; headline: d256 s128 151.3 s (−29.6% cumulative), d256 s256 352.1 s
(−25.1%), 4096 tok 487.6 s (−24.3%), 8192 tok 1176.7 s UNLOCKED — proofs
byte-identical.  Open: host-RSS regression (P2), 16384 tokens (P1).
See IMPROVEMENT_LOG.md iteration-1 sections.)
