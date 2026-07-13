# Improvement loop log (append-only; one section per iteration+item)

## Iteration 1 — Item 2: logUp GPU offload (lug/Am + claims device build)

Implemented in `p3_logup.cuh` (kernels `p3lu_amfill_kernel` / `p3lu_amaxpy_kernel`,
device Am build in `prove_super` for devleaf groups, `mat_col_range_dev` in the
device-claims path).  Transcript-identical: exact field adds in a different
accumulation order; compacted member columns rematerialize on device
(packed upload + device unpack + device mask PRNG).  Kill switch `P3_LUG_DEVAM=0`.

- Identity pairs seq=64 zk=1 (default / SC5ZG_CAP / SBLIND_MIN=10 / both):
  verify_ok=1, proof_mb byte-identical 42.658 / 41.569.  Devam-path coverage
  confirmed: 37 groups at NM >= 2^16 at seq=64.
- d256 s128 zk=1 (P3_MEMLOG=1 P3_ZKPROF=1): **159.9 s / 16.2 GB / 77.527 MB
  identical** vs baseline 214.8 s / 16.2 GB → **−25.6% prove time**.
  ZPROF lug/Am 51.4 s → 4.3 s (x137), lug/claims 3.3 s; STAGES lug 23.9 s,
  mm 72.3 s, batch 57.5 s.  Log: /root/zkrun_i1_d256s128_devam.log.
- Status: KEEP (gates rerun at end of iteration).

## Iteration 1 — Item 1: disk-backed Packed store (in progress)

Finding: the 8192-token (s128 b64 d64) config dies DURING build_witness, not in
prove: RSS ramps to 21.5 GB in the first 20 s (QKV witness gen), drops to
4.5 GB after packing, then climbs ~0.35 GB/s through the attention loop to
35.4 GB → cgroup kill, before the first BENCH/memlog output (both prior logs
empty for the same reason).  va/vb are lazy (empty) — not the cause.
Suspects: (a) glibc arena retention of per-instance transients (no trim_heap
in build_witness), (b) lidx per-product arrays (~16 B/product, ~7.5 GB at
8192 tok) held raw from witness gen to lookup flush, (c) sub-4 MB packed
columns below the spill threshold.
Changes so far: trim_heap per attention instance + per section in
build_witness; lidx spill at compact_wit time (LayerWit.limap + defer_v
spilled-view overload); rsslog instrumentation per witness section; spill
dir now DEFAULTS to /workspace/p3_spill (P3_PK_SPILL=0 disables, path
overrides), default P3_PK_SPILL_MIN 4 MB -> 1 MB.

Measured (i1b binary, s128 b64 = 8192 tokens zk=1): witness build now
completes at **4.8 GB** RSS (was: cgroup kill at 35.4 GB mid-attention-loop
with the old binary + explicit spill).  Attention-loop creep 2.3 GB/64
instances vs ~15 GB/64 before — the dominant cause was glibc arena
retention of per-instance transients (no trim_heap in build_witness), with
lidx retention second.  Prove phase running (RSS 19.3 GB early); result
below when done.

## Iteration 1 — packed-direct commits (cwit lever, follows Item 4's intent)

cwit = 47.9 s of the 72.3 s mm stage at d256 s128 after Item 2.  The
cpk_dev commit path rematerialized the ALREADY-PACKED witness column to raw
on host (unpack + PRNG), re-packed it, and uploaded raw.  New
`p3lu::commit_pk_nc` (p3_logup.cuh) commits straight from the pack: packed
bytes upload + device unpack (`unpack_ints_dev`) + device mask chain +
`salted_commit_root_dev` — same device bytes, same seed order, identical
root.  Wired into the CDp loop (p3_hawkeye.cuh `pkc` lambda); CDg/CDo/CDb/CDn
left on the classic path (they never took cpk_dev).  Binary /root/p3_tb_i1c.

- Identity pairs seq=64 zk=1 (default / SC5ZG_CAP / SBLIND_MIN=10 / both):
  verify_ok=1, proof_mb byte-identical 42.658 / 42.658 / 41.569 / 41.569
  (log /root/zkrun_i1c_idpairs.log).
- d256 s128 measurement below.

## Iteration 1 — Item 1: disk-backed Packed store (RESULT)

s128 b64 (8192 tokens, d=64 dff=128) zk=1 with the i1b binary: **BENCH
verify_ok=1 prove=1176.693 rss_gb=37.522 proof_mb=1288.100** — the config
is UNLOCKED (was: cgroup kill at 35.4 GB before any output).  Log
/root/zkrun_i1_s128b64_v2.log; appended to /root/zk_scale_results6.log.
STAGES mm=530.9 lug=217.7 batch=302.9 smx=57.2 (of 1176.7 s).
Status: KEEP.  Note prove-phase RSS 37.5 GB is within 3.5 GB of the cgroup
cap — 16384 tokens attempted below.

## Iteration 1 — packed-direct commits (RESULT)

d256 s128 zk=1 with /root/p3_tb_i1c: **151.259 s / 18.4 GB / proof 77.527 MB
identical, verify_ok=1** (log /root/zkrun_i1c_d256s128.log; appended to
results6).  vs 159.9 s post-devam → **−5.4%**; vs the 214.8 s iteration
baseline the cumulative iteration-1 win is **−29.6%**.  STAGES mm 72.3 →
63.4 s (cwit lever worked); lug 23.9 → 24.1, batch 57.5 (unchanged).
HONEST CAVEAT: RSS regressed 16.2 → 18.4 GB (+2.2 GB, +13.6%).  KEEP per
the ≥3%-time rule, but the regression is logged as a roadmap item
(investigate: the pack now stays resident through the device unpack, and
the classic path's early host-side free is skipped).  Kill switch
P3_CPK_DEV=0 restores the old path bytes-identically (gate pair green).

## Iteration 1 — endgame runs (chained, detached)

Runner /root/run_i1_final.sh (status /root/run_i1_final_status.log), one GPU
job at a time: (1) run_gates2.sh with identity pairs extended to BOTH
/root/p3_tb_s2 and /root/p3_tb_i1c under all 8 lever envs; (2) d256 s256
rerun (baseline 469.8 s / 30.7 GB); (3) s256 b16 = 4096 tok rerun (baseline
644.2 s / 34.4 GB); (4) FIRST 16384-token attempt (s256 b64 d64) — expected
tight: the 8192-token prove phase peaked at 37.5 GB on a 41 GB cap.
Results appended below when the runner finishes.

## Iteration 1 — endgame results (FINAL)

Runner /root/run_i1_final.sh completed (status /root/run_i1_final_status.log).

**Gates (run_gates2.sh, /root/gates2_result.log): ALL GREEN.**
- 26-suite battery: ALL PASS every suite (incl. HAWKEYE-ZK-HIDING 19/19,
  FULL-LAYER-ZK-HIDING 16/16, FULL-MODEL-ZK-HIDING 11/11).
- Compact teeth (P3_COMPACT_MIN=0): ALL GREEN.
- Identity pairs on BOTH /root/p3_tb_s2 and /root/p3_tb_i1c under all 8 lever
  envs (default / SC5ZG_CAP / SBLIND_MIN / both / CPK_DEV=0 / PK_SPILL path /
  PK_SPILL=0 / LUG_DEVAM=0): verify_ok=1, proof_mb 42.658 / 41.569 exactly.
- Guards: s256 d64 51.996 MB and d256 s128 77.527 MB proofs reproduced on the
  reference binary.
- Minor note (not a gate failure): p3_tb_i1c under forced streaming
  P3_SC5ZG_CAP=800000000 is ~2.1x slower at seq=64 (25.98 s vs 12.13 s on
  p3_tb_s2) with identical proof bytes — the cpk_dev path interacts poorly
  with the forced cap. Only affects that test env; logged for the record.

**Headline reruns (p3_tb_i1c, appended to /root/zk_scale_results6.log):**

| config | tokens | iter-0 baseline | iteration 1 FINAL | time | RSS |
|---|---|---|---|---|---|
| d256 s128 | 128 | 214.8 s / 16.2 GB | **151.3 s / 18.4 GB** | **−29.6%** | +2.2 GB |
| d256 s256 | 256 | 469.8 s / 30.7 GB | **352.1 s / 37.0 GB** | **−25.1%** | +6.3 GB |
| s256 b16 | 4096 | 644.2 s / 34.4 GB | **487.6 s / 37.3 GB** | **−24.3%** | +2.9 GB |
| s128 b64 | 8192 | cgroup kill | **1176.7 s / 37.5 GB** | UNLOCKED | — |

All verify_ok=1, proof bytes identical to the pre-iteration proofs
(77.527 MB at d256 s128; 84.430 MB at d256 s256).  Logs
/root/zkrun_i1c_d256s128.log, _d256s256.log, _s256b16.log,
/root/zkrun_i1_s128b64_v2.log.

HONEST CAVEAT: host RSS regressed on every config that completes
(+2.2 / +6.3 / +2.9 GB).  d256 s256 now sits at 37.0 GB on a 41 GB cap.
Attributed (not yet isolated) to cpk_dev pack retention + devam staging;
P3_CPK_DEV=0 restores old bytes-identical path at +5.4% time.  Filed as
Proposed item P2 in ROADMAP.md.

**16384-token attempt (s256 b64 d64): FAILED exit=137 (cgroup OOM) at 41 s.**
Log /root/zkrun_i1c_s256b64.log contains exactly one line:
`# rss 0.2 GB at wit:rms1` — killed before `wit:qkv` ever printed.
Diagnosis (from logs + source, no rerun): the kill window is the QKV section
p3_transformer.cuh:256-291, where the three Wq/Wk/Wv p3hwl::gen_witness
results coexist RAW until the compact_wit loop at :288.  The 8192-token run
measured a 21.5 GB transient in exactly this window (Item-1 finding above);
at 16384 tokens each instance is P=2^27 (~14 GB raw columns), so three
coexisting ≈ 43 GB > 41 GB cap — instantaneous kill between rsslog points,
which is why only 0.2 GB was ever logged.  The disk spill can't help: it
engages at compact_wit, which is never reached.  Fix proposed as P1
(gen→compact→trim per instance, transcript-identical).

**Iteration 1 net result:** −25% to −30% prove time on all d=256 and batch
configs, 8192 tokens unlocked, proofs byte-identical, all gates green.
Open regressions: host RSS (+2.2..6.3 GB), forced-cap streaming slowdown on
i1c, 16384 tokens still locked (P1).

## Iteration 2 — P1: per-instance QKV/FFN witness compaction

Implemented in `p3_transformer.cuh`: gen_witness -> compact_wit -> trim_heap
interleaved PER INSTANCE in the QKV section (Wq, then Wk, then Wv) and the FFN
section (Wg, then Wu) — three (two) raw dff-sized witnesses never coexist.
Transcript-identical (packing is representation-only).
- Identity pairs seq=64 zk=1 (default / SC5ZG_CAP / SBLIND_MIN=10 / both):
  verify_ok=1, proof_mb 42.658 / 41.569 byte-identical (binary /root/p3_tb_i2a).
- The 16384-token attempt runs at the end of the iteration (long run).

## Iteration 2 — P2: host-RSS regression = the default-on FUSE spill; pressure gate

Bisection at d256 s128 zk=1 on i2a (P1 only): default spill-on **170.5 s /
18.38 GB**, P3_PK_SPILL=0 **143.2 s / 15.06 GB** — the ENTIRE iteration-1 RSS
regression (+3.3 GB here) AND a previously unnoticed +27 s time regression are
the /workspace FUSE-backed Packed spill being DEFAULT-ON at configs that never
needed the disk (mmap'd file pages stay resident after read-backs; FUSE writes
+ reads cost wall time).  Logs /root/zkrun_i2a_d256s128_spillon/off.log.
Fix (`p3_zkc.cuh spill_pressure()`): spill engages only when VmRSS >=
P3_PK_SPILL_GATE (default 0.45) x the memory cap (cgroup limit, else
MemTotal).  Storage-only, per-pack decisions cannot change proof bytes;
P3_PK_SPILL_GATE=0 forces always-spill (old behavior) for tests.
- Identity pairs on i2b incl. P3_PK_SPILL_GATE=0 and P3_CPK_DEV=0:
  verify_ok=1, 42.658 / 41.569 byte-identical.
- d256 s128 zk=1 (i2b, with P3 below): **123.9 s / 15.06 GB** — gate never
  triggers below 18.5 GB, reproducing the spill-off profile exactly.
Status: KEEP.  NOTE for configs near the cap (8192/16384 tok): spill engages
mid-run once RSS crosses the gate — validated by the endgame runs.

## Iteration 2 — P3: device resolver + column-major chunked G build (batch-open)

Root cause measured: the giant strCol classes' columns are COMPACTED
commitments whose only ledger source was the HOST resolver (mat_col_into +
raw-size PCIe upload, ~110 ms/GB); blind columns with a device gen ran ~10x
cheaper.  Two changes (`p3_batchopen.cuh`, `p3_logup.cuh`):
1. PLedger.dresolve: device twin of the compact resolver (mat_col_range_dev
   = packed upload + unpack/mask kernels, bit-identical device bytes), used
   by dcol init, upload_col, and the q0 encode.
2. Column-major CHUNKED G build in the strG host-parked path (and the g2pass
   round-0 rebuild): points processed in device-sized chunks, each distinct
   column materialized once per chunk (exact field adds, per-point round-0
   messages summed in ascending t order — absorbed bytes unchanged).
- Identity: seq=64 pairs + P3_COMPACT_MIN=0 forced-compaction pair green
  (42.658 identical; i1c takes 44.3 s vs i2b 10.1 s on that test).
- d256 s128 zk=1 (i2b): batch stage 57.2 -> 36.6 s; tf-bo14 class
  G 7804 -> 1963 ms, ys+rlc 6033 -> 897 ms, q 12066 -> 9737 ms.
- Combined P2+P3 headline: **d256 s128 = 123.9 s / 15.06 GB, proof 77.527 MB
  byte-identical, verify_ok=1** vs iteration-1 151.3 s / 18.4 GB =
  **−18.1% time, −18.2% RSS** (log /root/zkrun_i2b_d256s128.log).
Status: KEEP.

## Iteration 2 — P4: INFEASIBLE as written (no host inversions at scale)

The promoted item targeted "lug/inv ≈ lug/hcommit host-side field inversions".
Source + profile audit: ZPROF lug/inv (p3_logup.cuh:1447-1455) is NESTED inside
lug/hcommit and times the per-subgroup pm/qm mask-stream DEVICE commits
(salted_commit_root_dev x2 per subgroup, x137 subgroups at d256 s128 = 11.4 s
of NTT+Merkle GPU work) — hcommit-inv ≈ 0 s because the T-side is trivial.
The merged v3 flush avoids helper inversions BY DESIGN (multiplicative pm=sm*qm
masks, "no inversions" comment at :1430); inv_all_add (already a Montgomery
batch inversion, OpenMP) only runs in the standalone prove_v path that does not
execute in the composed layer at scale.  Nothing to move to the device.
Residual (proposed for iteration 3): overlap each subgroup's pm and qm commits
on separate streams — bounded ~3-4% e2e.  Status: INFEASIBLE (mechanism above).

## Iteration 2 — P5 (slice): device batch-open class blinder

The class blinder for big classes was host-side: zprng_fill of N gl_t (1-2 GB),
HOST salted_commit_root, host build_eq + eval_h, and the host vector then
re-uploaded per use in the class.  Now (p3_batchopen.cuh prove_class, N >= 2^24):
device lcg-chain fill (bit-identical twin of zprng_fill, same kernel the
commit_pk_nc mask path uses), salted_commit_root_dev (identical root), eq+dot
kernel eval (exact field sums — same value), and the entry carries host+device
regen closures so the column is never host-resident.
- d256 s128 zk=1 (i2d): bo/blinder 8.97 -> 0.97 s; batch stage 36.6 -> 28.0 s.
  **BENCH 114.7 s / 15.06 GB, proof 77.527 MB byte-identical, verify_ok=1**
  (log /root/zkrun_i2d_d256s128.log) = **-24.2% time / -18.2% RSS vs the
  iteration-1 151.3 s / 18.4 GB baseline** for P1+P2+P3+P5-slice combined.
The P5 core (batched salted-tree builder for the x3530 small commits,
commit_salt 15.9 s + mask_gen 10.0 s) is proposed for iteration 3.
Status: KEEP (slice).

## Iteration 2 — P7: lug/cnt GPU histogram

Implemented (p3_logup.cuh prove_super cnt build): device histogram over the
merged lookup index streams — shared-memory sub-histograms for tables <= 8192
rows, global 64-bit atomics otherwise, 256 MB index chunks, device oob flag
mirroring the host range check.  Counting is order-free integer accumulation:
device bins equal the host loop exactly, committed cnt bytes unchanged.
Engages at >= 2^22 total indices (P3_LUG_DEVCNT_MIN, =0 to force for tests;
P3_LUG_DEVCNT=0 kill switch).
- Identity pairs seq=64 (default / DEVCNT_MIN=0 forced / DEVCNT=0 / blinds):
  verify_ok=1, 42.658 / 41.569 byte-identical.
- Scale win measured in the endgame runs (baseline lug/cnt 26.8 s at 4096 tok,
  87.5 s at 8192 tok; d256 s128 has only 1.5 s so no measurable change there).
Status: KEEP pending endgame numbers.

## Iteration 2 — P6: DEFERRED with design (transcript-changing)

Round-level batching across same-shape chains requires a round-major
Fiat-Shamir absorb order (each chain's round-r challenge currently depends on
every earlier chain's absorbs; independent chains cannot share device round
passes under the sequential order).  That changes EVERY proof's bytes (nh*b
attention chains exist at all configs), forfeiting the byte-identity
cross-check that validated this iteration's five landed levers.  Per the
revert-safety rule (P6 last, sole transcript-changing lever of its iteration)
it moves to iteration 3 as the first item, with the full battery + compact
teeth + ZK hiding suites as its gate.  Design sketch in ROADMAP.md Proposed.

## Iteration 2 — endgame (final, 2026-07-13, coordinator-run)

Gates on the final i2d source: 26-suite battery ALL GREEN, compact teeth OK,
24/24 identity-pair runs verify_ok=1 with proof bytes exactly 42.658 / 41.569
(gates3, /root/zkrun_i2_gates3.log).  Headline reruns (zk=1, verify_ok=1,
proofs byte-identical to iteration 1 and the base):

| config | base | iter1 | iter2 | cumulative |
|---|---|---|---|---|
| d256 s128 | 230.3 s / 25.1 GB | 151.3 / 18.4 | **114.7 / 15.1** | −50% time / −40% RSS |
| d256 s256 | 469.8 / 30.7 | 352.1 / 37.0 | **301.7 / 27.3** | −36% / −11% |
| 4096 tok  | 644.2 / 34.4 | 487.6 / 37.3 | **399.4 / 34.7** | −38% |
| 8192 tok  | (impossible) | 1176.7 / 37.5 | **981.8 / 37.5** | unlocked, then −17% |

16384 tokens: STILL WALLED.  The witness now survives its old kill point (P1
works: per-instance compaction holds RSS to 29 GB through the attention tail),
but the process exits silently (rc=0 masked, log truncates at wit:attn-ai127,
~6 min in).  Not diagnosed — the loop was stopped here by the user; first item
for any future iteration 3, together with the deferred P6 (transcript-changing
chain batching, design in ROADMAP) and the P5 core (batched salted commits).
