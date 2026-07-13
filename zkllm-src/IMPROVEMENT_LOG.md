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
