# Scaling study: memory + speed of the P3 full-ZK prover (2026-07-12)

RTX 4090 24 GB, 41 GB container cap, git ced1fa5+3f34c99 base.  Bench =
`p3_transformer_bench.cu` (`/root/p3_tb_c24r` = base, `/root/p3_tb_ecpk` =
+section-22b lever, below).  Raw logs `/root/zkrun_*.log`, BENCH lines in
`/root/zk_scale_results5.log`, GPU traces `/root/gpu_trace_*.csv`.

STATUS: in progress — running log; sections fill in as measured.

## A. Overhead growth with model size (the d-sweep)

Fixed 128 tokens (seq=128 b=1), dff=4d, zk=1 and zk=0.  d=64: nh=2 dh=32
tables_ld6; d=128: nh=4 dh=32 p3_rmsnorm_tables_ld7.bin (NOTE: `tables_ld7.bin`
in the repo is a MISLABELED ld=6 artifact — the bench rejects it); d=256:
nh=4 dh=64 tables_ld8.

| d | zk | prove (s) | mm | lug | batch | other | proof MB | RSS GB |
|---|---|---|---|---|---|---|---|---|
| 64  | 1 | 32.1  | 10.8 | 11.2 | 7.5  | 2.6 | 48.7 | 2.7 |
| 128 | 1 | 79.3  | 28.9 | 26.0 | 19.4 | 5.0 | 70.0 | 7.5 |
| 256 | 1 | 230.3 | 87.0 | 80.1 | 57.0 | 6.2 | 77.5 | 25.1 |
| 64  | 0 | 15.2  | 5.2  | 7.9  | 1.0  | 1.1 | 25.1 | 1.6 |
| 128 | 0 | 36.4  | 13.2 | 17.4 | 3.9  | 2.0 | 38.0 | 4.7 |
| 256 | 0 | 116.7 | 39.8 | 55.0 | 18.7 | 2.4 | 43.3 | 16.2 |

**Work model.** Products per layer at seq=128: QKV+out-proj 4 instances of
128·d·d, dff 2 instances of 128·d·4d (=8·128·d²), attention 2·128·128·d
→ P_tot ≈ 1536·d² + 32768·d = 8.4M / 29.4M / 109.2M for d=64/128/256.

**Per-unit-work trends (zk=1):**

| d | P_tot | prove/P_tot (µs/product) | prove/token (s) | zk1/zk0 |
|---|---|---|---|---|
| 64  | 8.4M   | 3.82 | 0.251 | 2.11 |
| 128 | 29.4M  | 2.70 | 0.620 | 2.18 |
| 256 | 109.2M | 2.11 | 1.799 | 1.97 |

Prove time grows ×2.47 then ×2.90 per doubling of d, vs ×3.50/×3.71 growth
in products.  **Per-product cost FALLS with d** (3.8 → 2.1 µs): no stage is
superlinear in the work; per-chain fixed costs amortize.  The end-to-end
time is ~Θ(d²) because the dff matmuls are Θ(d²) at fixed tokens.  The
overhead RATIO (vs the flat ~2.2 ms launch-bound toy forward) therefore
grows ∝ d² — it is a property of the flat baseline, not prover superlinearity.
The same holds for tokens at fixed d (BENCH_ZK_AT_SCALE.md A: 63/118/299 s
for 256/512/1024 tokens).

**Stage growth (zk=1, ×d64→d256):** mm ×8.1, lug ×7.1, batch ×7.6, all ≈
the ×13 product growth minus amortization; "other" (rms/qnt/rope/smx/bfa/swg)
grows only ×2.4 and is 3% of time at d=256.  mm+lug+batch = 97% of prove
time at d=256.

**Inside mm (hwl prof, sum over chains ≥2^22, d=256 zk1):** cwit=50.1 s of
79.6 s profiled — 63% of the matmul stage is COMMITTING witness columns,
not proving.  zcdp (the actual zero-check sumcheck) is 9.5 s.  The ZPROF
cross-cut for the whole run: commit_salt 18.1 s + mask_gen 18.6 s over 3530
commits (these two are inside cwit/lug-hcommit), lug/Am 51.4 s (host
merged-witness build), bo/q0-subset 19.8 s (of which q0/enc 17.7 s = opening
re-encode), lug/inv 11.4 s (GPU pm/qm mask commits), sc5z/chain 12.1 s,
lug/hcommit 11.5 s, bo/G 10.5 s, bo/ys+rlc 10.1 s, bo/blinder 9.0 s,
lug/scA 7.6 s, lug/claims 7.7 s.

**GPU idle share (nvidia-smi 1 Hz trace during the sweep):**

| window | mean util | samples at 0% | samples ≥90% | peak mem |
|---|---|---|---|---|
| d=64 zk1  | 15.7% | 36% | 3% | 4.2 GB |
| d=256 zk1 | 26.1% | 51% | 8% | 21.4 GB |
| d=64 zk0  | 9.2%  | 58% | 0% | 4.2 GB |
| d=256 zk0 | 11.3% | 65% | 1% | 10.8 GB |

The GPU is idle half the wall time even at the largest working config: the
prover is HOST-compute and PCIe bound (lug/Am host axpy loops, mask PRNG,
pack/unpack, per-round host reductions), not GPU-FLOP bound.  This is the
single largest structural speed lever.

## B. Memory accounting of the witness wall

At the failing config d=256 seq=256 zk=1 (base binary, `zkrun_d256s256r.log`):
process baseline is 20.5 GB entering the FIRST matmul chain (packed layer
witness of all 7+2A instances + operand commitments + deferred-lookup queue +
non-mm gadget state), creeping to 23.2 GB entering the P=2^27 dff chain
(+~0.15-0.3 GB compact retention per proven instance).  The dff chain then
commits NDP=12 product-domain columns; in zk mode each is AUGMENTED to
2^28 gl_t = 2 GB, all 12 held RAW on host from commit through the Dp
zero-check = **+24 GB on a 23.2 GB baseline → cgroup kill #1**.  The columns
are provably droppable: their real region is small-int packable (the same
pack that section 22 applies AFTER the zero-check) and their mask region is
a recorded PRNG chain — nothing in the window actually needs them raw.

There are THREE stacked walls behind exit 137, uncovered one at a time:

1. **Commit→zero-check window** (above): 12 × 2 GB raw augmented columns.
   Fixed by early compaction (section C, lever 1).
2. **Streamed-prefix round-0 halves**: with wall #1 fixed the chain enters
   the streamed sumcheck prefix at 29.2 GB; `dev_fits` used `cudaMemGetInfo`
   alone, but the async mempool retains freed blocks (release threshold
   = ∞), so post-commit the driver reports the card full and the round-0
   halves (14 cols × 1 GB) went to HOST → 43 GB → kill.  Fixed by counting
   the pool's idle reservation as free (lever 2); the halves then park on
   the device (measured: GPU 17.4 GB used during the prefix, host flat).
3. **Batch-open G-parking**: the v=28 size class (nc=55 columns × 2 GB,
   T=19 distinct points) needs (T+2)·colbytes = 42 GB of per-point RLC
   columns G_t; device fits ~7, the mixed-parking remainder (12 × 2 GB =
   24 GB) is host-parked → kill.  Fixed by the two-pass G build (lever 3):
   G_t full-length now never touches host memory.

## C. The levers implemented (section 22b) — all transcript-identical

**Lever 1 — early compaction + on-demand rematerialization.**  Compact each
product-domain committed column IMMEDIATELY after its commit (pack_ints
real region + mseed-seeded mask), and make every later reader rematerialize
bit-identical values on demand:

- `p3_zkc.cuh`: `unpack_ints_range` (range inverse of pack_ints),
  `zprng_fill_at` (jump-ahead range fill of a seeded mask chain).
- `p3_logup.cuh`: `mat_col_range` (range materializer of a compacted Col).
- `p3_hawkeye.cuh`: `ColSrc` gains a compacted-column view; `sc5z_gpu`
  streams/uploads through a chunk loader (compacted chunks materialize
  through a bounce buffer); `claimc` materializes transiently for the
  post-sumcheck evaluation claims; the CDp commit loop compacts each column
  right after its root is absorbed (P3_EARLY_CPK_MIN, default: augmented
  length ≥ 2^25; gated to the GPU batch path).  The AL/SEL linked-mask
  vectors are freed after the commits (recorded chains).

**Lever 2 — pool-aware `dev_fits`** (`p3_hawkeye.cuh` sc5z_gpu): count the
async mempool's idle reservation (reserved − used) as free device memory
when deciding where the streamed-prefix halves go (matches what
p3_batchopen's parking already did).  Placement-only: the stream/resident
split is transcript-identical by design (the forced-stream gate pairs
validate exactly this).

**Lever 3 — two-pass G build in batch-open** (`p3_batchopen.cuh`): when the
mixed-parking host remainder exceeds P3_BO_G2MAX (default 6 GB), pass 1
builds each G_t transiently only for its round-0 message contribution;
after the round-0 challenge, each G_t is rebuilt with the identical axpy
order and immediately bound to half length, halves parked device-first.
Cost: the G build runs twice for that class; full-length G_t never reaches
host memory.

**Lever 4 — device-mask pre-compacted commits** (`p3_logup.cuh`
`commit_col_nc(cpk_dev)`): for the early-compacted product-domain columns,
pack the real region on host, generate the fresh mask region ON DEVICE from
its recorded seed (`p3zkc_lcgchain_kernel`, bit-identical chain), and commit
via `salted_commit_root_dev` — the 2 GB augmented host column never exists,
and the host-side mask PRNG fill + re-pack disappear.  Falls back to the
classic path for linked-mask (AL/SEL), non-power-of-two, or unpackable
columns.  P3_CPK_DEV=0 reverts.

Transcript identity: pack/unpack and the PRNG chain are exact; the streamed
chunk kernel sums see identical device bytes in the identical order; claimc
evaluates identical values; bind distributes exactly over the RLC.
Validated at seq=64 across all path combos (resident / forced-stream ×
merged / structured blinds × cpk_dev on/off): verify_ok=1 with proof_mb
byte-identical to the base binary (42.658 / 41.569).  RSS at seq=64 drops
1.47 → 0.98 GB.

**Results at scale** (P3_MEMLOG=1 P3_ZKPROF=1, zk=1, verify_ok=1 everywhere):

| config | binary | prove (s) | RSS (GB) | proof (MB) | notes |
|---|---|---|---|---|---|
| d256 s128 | base (c24r) | 230.3 | 25.1 | 77.527 | reference |
| d256 s128 | levers 1-4 | 257.1 | **16.2** | 77.527 (identical) | −35% RSS, +12% time (host-remat tax; see below) |
| d256 s256 | base | exit 137 | >41 | — | wall #1 |
| d256 s256 | levers 1+2 | exit 137 | >41 | — | died at wall #3 (batch-open G park) |
| d64 s256 | all levers | 62.2 | 4.9 | 51.996 (identical) | small-config guard: no regression (base 63.4) |
| d256 s128 | + device remat | **221.2** | **16.2** | 77.527 (identical) | **4% faster than base AND −35% RSS** |
| d256 s256 | all levers | **469.8** | **30.7** | 84.430 | **WALL BROKEN** (base: impossible); mm=197 lug=131 batch=132 |
| s256 b16 (4096 tok) | all levers + spill | **644.2** | **34.4** | 397.961 | **WALL BROKEN** (base: impossible); mm=251 lug=230 batch=116 |
| s128 b64 (8192 tok) | all levers + spill | exit 137 | >41 | — | the NEXT wall, as predicted (F): packed-witness retention itself; needs the disk-backed Packed store |
| d256 s128 | final gate rerun | **214.8** | **16.2** | 77.527 (identical) | 7% faster than base, −35% RSS |
| d64 s256 | final gate rerun | **56.9** | 4.9 | 51.996 (identical) | 10% faster than base |

**Lever 5 — Packed disk spill** (`p3_zkc.cuh` `spill_packed`, opt-in
`P3_PK_SPILL=<dir>`): packed magnitude bytes of big columns (witness packs,
compacted committed columns) move to an unlinked mmap'd file — clean file
pages the kernel reclaims under pressure instead of anonymous RSS.  For the
4096-token config whose baseline is the per-instance packed-witness
retention itself (23.7 GB at first chain, +11 GB creep over 64 attention
instances).

The +12% at d256 s128 decomposes into: resident-upload host rematerialization
(+1.0 s/chain in zcdp), gsum AL/SEL remat (+1.6 s/chain), and claimc host
mat_col_into (+~5 s/chain untimed).  All three were then moved to
device-side materialization (`unpack_ints_dev` + `mat_col_range_dev` +
claimc device eval — packed bytes are 2-8× smaller on PCIe than raw
elements), re-measured below.

## D. Speed levers

Implemented (all transcript-identical, validated by proof-byte identity):

1. **Device-mask pre-compacted commits** (lever 4 above): host mask PRNG
   fill disappears for the big product-domain columns — ZPROF mask_gen
   18.6 s → 10.0 s at d=256 seq=128; the commit upload halves (real region
   only).
2. **Device-side rematerialization** (`unpack_ints_dev`, `mat_col_range_dev`,
   claimc device eval): the early-compaction read-back tax (host unpack +
   raw-size PCIe) becomes a packed-size upload + unpack kernel — 2-8× less
   PCIe than even the BASE binary's raw uploads on those paths.
3. **Pool-aware placement** (lever 2): also adopted by batch-open G-parking
   decisions; the biggest d256 s128 class's G build dropped 17.6 s → 8.0 s
   because 17 of 19 points now park on device instead of 11.

Evaluated, NOT adopted (roadmap, section F): GPU offload of lug/Am (the
single biggest item at 51 s), batched round-message reduction, commit/witgen
stream overlap.  OpenMP thread count: p3bf::nthr already scales teams by
size (section 19.3 work); not touched.

(measured results fill in below)

## E. Gates (final source, /root/gates2_result.log, 2026-07-13)

- 26-suite battery, rebuilt from the lever source: **ALL GREEN**.
- Compaction teeth (run_compact_teeth.sh): **OK**.
- Identity pairs at seq=64 zk=1, six env combos (default / forced-stream /
  structured blinds / both / P3_CPK_DEV=0 / P3_PK_SPILL): every run
  verify_ok=1 with proof_mb byte-identical to base (42.658 / 41.569).
- Guard reruns: d64 s256 = 56.9 s / 4.9 GB (base 63.4), d256 s128 =
  214.8 s / 16.2 GB (base 230.3 / 25.1) — the levers are now a net WIN on
  both time and memory at every measured config.

## F. Roadmap items NOT done

**Binius/GF(2^128) integration (NOT attempted — cross-field binding seams
make a naive swap unsound).**  Bound on the win, from measured stage shares
at d=256 seq=128 zk=1 (prove 230.3 s):

- The standalone Binius hawkeye product gadget is ~20× faster than GL and
  commits 46× less data (p3_binius_hawkeye benchmarks).  The mm stage
  (87.0 s, 38%) would drop to ~4-9 s; the batch-open share attributable to
  the product-domain classes (the v≥26 classes are ~80% of batch's 57.0 s
  by the bo-class timing lines) would shrink with the 46× committed-data
  reduction to ~5-10 s.
- If the Binius logUp (p3_binius_logup.cuh) also replaces the GL lookup
  stack, lug (80.1 s, 35%) scales similarly.
- Composite estimate: mm+lug+batch 224 s → ~20-35 s, non-mm gadgets ~6 s
  unchanged → **~26-41 s, i.e. a 5.5-9× end-to-end speedup** at d=256, and
  proportionally at other sizes (the three stages are ≥93% of prove time
  everywhere measured).  The blocker is soundly binding GF(2^128) gadget
  claims to the Goldilocks Basefold commitments (seam argument); that is a
  protocol-design work item, not an engineering one.

**GPU offload of lug/Am** (51.4 s at d=256 seq=128): the merged-lookup
combined-witness build is an exact-field axpy accumulation — commutative,
so a device build is transcript-identical; the devleaf path already wants
the result on device (it uploads Am today).  Est. 3-5× on that item
(~-35 s at d=256), not done here.

**Witness/lookup-index spill for ≥8192-token configs**: the packed witness
+ lidx baseline scales linearly with tokens and will re-hit the 41 GB cap
around 8-16k tokens even with all levers; a disk-backed (mmap) Packed store
is the next lever (sequential access only, /workspace FUSE is acceptable
for data).
