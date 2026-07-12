# Benchmark: ZK at scale — full-ZK composed layer/model proofs

RTX 4090 (24 GB), 41 GB container-memory cap, 2026-07-12.  Prover = the
section-22 compact-retained-storage build PLUS the four post-hoc scale fixes
(git `ced1fa5`; root causes + validation in `scale_debug_report.md`).  All
zk=1 rows below are `verify_ok=1`.  Sources: `sweep_run.log`,
`zk_scale_results*.log`, `final_sweep_driver.log`, `gates_result.log`.

**Overhead column** = prove_ms / native-forward_ms with the **batched-attention
forward** (`tf_fwd_bench.py` batched mode, `fwd_results_batched.log`) as the
denominator — the honest baseline.  The canonical per-op forward mode
(`fwd_results.log`) is launch-bound (~90% Python-loop overhead at big batch:
120 ms at b=64/seq=128 vs 2.3 ms batched) and is shown nowhere in ratios here.
Configs with no measured batched forward get `-` (nothing extrapolated).

## A. Single-layer sweep, zk=1 (one full composed transformer LAYER)

d=64 rows: nh=2 dh=32 dff=128.  d=256 row: nh=4 dh=64 dff=1024.

| config | tokens | fwd (ms) | prove (s) | verify (s) | proof (MB) | RSS (GB) | overhead |
|---|---|---|---|---|---|---|---|
| seq=64 d=64 b=1 | 64 | - | 10.85 | 0.57 | 42.7 | 1.5 | - |
| seq=256 d=64 b=1 | 256 | 2.240 | 63.44 | 0.74 | 52.0 | 4.9 | 28,322x |
| seq=512 d=64 b=1 | 512 | 2.240 | 118.32 | 0.83 | 57.5 | 10.0 | 52,816x |
| seq=1024 d=64 b=1 | 1024 | 2.472 | 298.85 | 0.94 | 64.2 | 31.3 | 120,893x |
| seq=128 d=256 b=1 | 128 | 2.178 | 259.52 | 1.00 | 77.5 | 29.0 | 119,160x |
| seq=128 d=64 b=4 | 512 | 2.108 | 86.00 | 1.48 | 111.3 | 6.1 | 40,789x |
| seq=128 d=64 b=16 | 2048 | 2.221 | 269.36 | 4.43 | 344.4 | 30.1 | 121,290x |

- Prove time is near-linear in tokens at fixed d (63/118/299 s for 256/512/1024
  tokens; the seq=1024 row also pays the flush/rematerialization tax, section E).
- The batched forward is flat (~2.2 ms) at every one of these shapes — the GPU
  is launch-bound at toy dims — so the overhead ratio grows with tokens even
  though the prover itself is linear.  At real model sizes the forward grows
  and the ratio stops inflating.
- Batch=16/seq=128 and seq=1024 (same 2048-vs-1024 token comparison at ~equal
  prove time) show token count, not shape, is the driver.

## B. Composed MULTI-LAYER models (MBENCH), zk=1

d=64 nh=2 dh=32 dff=128 vocab=256 (embedding + N layers + head, one proof).

| nlayers | seq | witness (s) | prove (s) | verify (s) | proof (MB) | RSS (GB) |
|---|---|---|---|---|---|---|
| 1 | 128 | 2.78 | 33.46 | 0.70 | 51.0 | 2.2 |
| 2 | 128 | 4.80 | 61.29 | 1.18 | 87.9 | 3.5 |
| 4 | 128 | 8.55 | 115.20 | 2.13 | 162.8 | 6.4 |
| 2 | 256 | 13.75 | 130.28 | 1.29 | 96.8 | 7.7 |

Marginal cost per extra layer at seq=128: ~27 s prove, ~37 MB proof, ~0.5 s
verify, ~1.4 GB RSS — i.e. layers compose linearly, no superlinear blow-up.

## C. No-ZK (zk=0) references and the ZK premium

| config | tokens | fwd (ms) | prove (s) | proof (MB) | RSS (GB) | overhead | zk1/zk0 |
|---|---|---|---|---|---|---|---|
| seq=512 d=64 b=1 | 512 | 2.240 | 53.21 | 33.2 | 6.7 | 23,750x | 2.22x |
| seq=1024 d=64 b=1 | 1024 | 2.472 | 148.18 | 41.0 | 21.2 | 59,944x | 2.02x |
| seq=128 d=256 b=1 | 128 | 2.178 | 118.67 | 43.3 | 16.2 | 54,488x | 2.19x |
| seq=256 d=256 b=1 | 256 | 2.426 | 228.06 | 50.7 | 33.5 | 93,990x | wall (E) |

Full zero-knowledge costs a uniform ~2-2.2x in prove time over the
integrity-only proof at these scales.

## D. Integerized reference (the fp8-exactness premium at scale)

Integerized layer estimate = sum of plain int-GEMM sumcheck proofs for the
same 7+2A matmul shapes (`p3_matmul_bench2`, measured 2026-07-12) + the SAME
composed run's measured zk=1 non-matmul stages (rms/qnt/rope/smx/bfa/swg/seam).
Same construction as BENCH_FP8_VS_INT.md section B / bench2.py.

| config | int mm (s) | non-mm (s) | int layer (s) | overhead | fp8-zk1 / int |
|---|---|---|---|---|---|
| seq=256 d=64 b=1 | 0.18 | 4.04 | 4.22 | 1,884x | 15.0x |
| seq=512 d=64 b=1 | 0.27 | 7.25 | 7.52 | 3,359x | 15.7x |
| seq=1024 d=64 b=1 | 0.56 | 23.52 | 24.08 | 9,743x | 12.4x |
| seq=128 d=256 b=1 | 0.33 | 5.87 | 6.20 | 2,846x | 41.9x |
| seq=128 d=64 b=4 | 0.32 | 8.39 | 8.71 | 4,131x | 9.9x |
| seq=128 d=64 b=16 | 0.99 | 31.13 | 32.12 | 14,461x | 8.4x |

At scale the non-matmul gadgets dominate the integerized estimate, so the
fp8-vs-int gap (~1000x on the matmuls alone, BENCH_FP8_VS_INT.md C) compresses
to ~8-42x at the whole-layer level.

## E. Memory levers: before/after (same configs, zk=1)

Before = pre-lever section-20 binary, after = current.

| config | before | after | memory | prove cost |
|---|---|---|---|---|
| seq=256 d=64 | 39.4 s / 12.0 GB | 63.4 s / 4.9 GB | 2.4x down | 1.61x up |
| seq=512 d=64 | 86.2 s / 29.1 GB | 118.3 s / 10.0 GB | 2.9x down | 1.37x up |
| seq=1024 d=64 | SIGKILL (no output) | 298.8 s / 31.3 GB | ran at all | - |

The compact storage + flush/rematerialization levers trade 1.4-1.6x prove
time for 2.4-2.9x host memory — that trade is what makes seq=1024, d=256
seq=128, 2048-token batch, and the multi-layer models exist at all under the
41 GB cap.  These numbers are the honest cost of the lever; nothing is free.

## F. Remaining walls (NOT fixed)

- **d=256 seq=256 zk=1** and **4096+-token batch configs** die at exit 137
  (cgroup SIGKILL): the P=2^27 dff chain's ~24 GB of committed host columns
  land on a ~23 GB process baseline and exceed the pod's 41 GB
  container-memory cap.  This is a HOST WITNESS SIZE wall, not a GPU wall.
- All GPU-side walls encountered on the way were fixed (section G); within
  the 41 GB cap every attempted config now proves and verifies.

## G. What changed to get here

Four post-hoc scale fixes on top of the section-22 compact storage
(mechanisms, root-cause analysis, and transcript-identity validation in
`scale_debug_report.md`; code in git `ced1fa5`):

1. **Per-column sumcheck bind + eager free** (p3_scgpu.cuh): the fused bind
   allocated all half-columns while all full columns were live (24 GB at
   N=2^27 x 16 cols) with UNCHECKED cudaMallocAsync -> illegal memory access.
2. **Streamed sumcheck prefix** (p3_hawkeye.cuh sc5z_gpu): chains larger than
   the device run early rounds from host-resident columns in 2^22-elt chunks;
   exact field sums, transcript byte-identical.  Includes the rd0
   structured-blind fix: the ScFix callbacks now get chain-global round
   numbers (without it, seq=1024 zk=1 proved but failed verification).
3. **Mixed G-parking** (p3_batchopen.cuh): per-point G_t device-parked as far
   as they fit, remainder host-parked (pure host parking was +25 GB at
   seq=1024 -> cgroup kill).
4. **Split NTT twiddle tables** (p3_ntt.cuh, logn>=26): W[k] = W[lo]*W[hi<<hb],
   4 GiB -> 384 KiB per direction at logn=30, same field elements; plus the
   size_t stage-counter fix (uint32 m<<=2 wrapped at n=2^30).

Gates: 26-suite battery ALL GREEN, compaction teeth OK, forced-stream pairs
byte-identical in both blind modes (`gates_result.log`).

## Provenance

Rows measured before the mixed-parking fix landed (seq<=512, the batch rows,
MBENCH, d=256 seq=128) have RSS values that are an UPPER BOUND for the current
binary; prove/verify/proof size are unaffected.  seq=64 and seq=1024 rows are
from the final gate run of the current binary.  Plot:
`/root/overhead_zk_at_scale.png` (`plot_zk_at_scale.py`).
