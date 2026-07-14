# Benchmark: exact-fp8 (Hawkeye) ZKP vs integerized ZKP vs native forward

RTX 4090 (24 GB), 2026-07-11.  Measured with `p3_transformer_bench` (exact-fp8
full transformer LAYER proof, Goldilocks Hawkeye), `tf_fwd_bench.py` (native fp8
forward), and `p3_matmul_bench2` (integer GEMM proof).  `bench_sweep.sh` +
`bench_sweep.log`.

## What is actually being proven (scope — read first)

- The proof covers **one full transformer layer**: rmsnorm - quant - QKV - rope -
  attention(QK^T, softmax, P.V) - Wo - residual - rmsnorm - SwiGLU - Wd -
  residual, as ONE proof.  A full model = N such layers.
- The **7 matmuls are bitwise-exact to the Hopper/H100 fp8 tensor-core
  accumulation** (Hawkeye).  The **non-matmul ops** are proven against our own
  exact-bf16 reference (`transformer_ref.py`), which is NOT matched to torch's/
  H100's bf16 RMSNorm/softmax kernels.  So: matmuls = H100-exact; whole layer =
  exact-to-a-reference.
- This prover uses the **Goldilocks Hawkeye**, not the new Binius one (§21).  So
  these numbers are the PRE-speedup state; see "the memory wall" below.
- 4090 has fp8 tensor cores (Ada), but its accumulation differs from Hopper, so
  the native fwd here uses the Triton Hawkeye kernel (= the proven semantics),
  not cuBLAS fp8 -- an H100 cuBLAS run would match the matmuls.

## A. Exact-fp8 layer proof vs native forward

zk=1 is the real zero-knowledge proof; zk=0 is the integrity-only proof.

### Model-size sweep (seq=64, batch=1, tokens=64, dff=4d)

| d | native fwd (ms) | prove zk0 (s) | prove zk1 (s) | verify zk1 (s) | proof zk1 (MB) | RSS zk1 (GB) |
|---|---|---|---|---|---|---|
| 64  | 4.4  | 8.17  | 32.95 | 0.76 | 58  | 3.6  |
| 128 | 8.0  | 21.93 | 114.3 | 1.18 | 92  | 12.1 |
| 256 | 14.9 | 75.62 | OOM   | -    | -   | >24  |
| 512 | 28.1 | OOM   | OOM   | -    | -   | >24  |

### Seq-len sweep (d=256, batch=1)

| seq | tokens | native fwd (ms) | prove zk0 (s) | prove zk1 (s) | proof zk1 (MB) |
|---|---|---|---|---|---|
| 16  | 16  | 14.5 | 20.72 | 108.1 | 142 |
| 64  | 64  | 14.9 | 75.62 | OOM   | -   |
| 256 | 256 | 14.5 | OOM   | OOM   | -   |

### Batch sweep (d=256, seq=16)

| batch | tokens | native fwd (ms) | prove zk0 (s) | prove zk1 (s) | proof zk1 (MB) | RSS zk1 (GB) |
|---|---|---|---|---|---|---|
| 1  | 16  | 14.3  | 20.77 | 108.3 | 142 | 11.0 |
| 4  | 64  | 54.9  | 81.48 | 375.0 | 471 | 36.3 (host RAM) |
| 16 | 256 | 219.9 | OOM   | OOM   | -   | >24  |

Prove-time overhead vs native fwd: ~2000x (zk0) / ~7000-14000x (zk1) at these
sizes.  NOTE the native fwd is fixed-overhead-bound at these tiny shapes
(~14 ms flat for d=256 across seq 16/64/256); on an H100 with real batch the
per-token fwd would be far faster and the ratio higher.

## B. Integerized GEMM proof (the fp8-vs-integer comparison point)

Plain integer GEMM proof (Goldilocks sumcheck matmul, no fp8 accumulation
semantics), per matmul.  proj = [T x d].[d x d], mlp = [T x d].[d x 4d].

| d | T | proj prove (ms) | proj verify (ms) | mlp prove (ms) | mlp verify (ms) | proof (KB) |
|---|---|---|---|---|---|---|
| 64  | 64  | 9.4  | 8.5  | 13.0 | 9.9  | ~700  |
| 128 | 64  | 12.5 | 9.9  | 21.4 | 13.0 | ~830  |
| 256 | 64  | 19.9 | 11.4 | 41.4 | 13.1 | ~970  |
| 512 | 64  | 36.2 | 13.2 | 99.2 | 14.9 | ~1120 |
| 256 | 256 | 29.8 | 13.0 | 51.0 | 14.8 | ~1110 |

## C. The headline: proving EXACT fp8 costs ~1000x more than integer

Per-stage prove breakdown of the exact-fp8 layer (zk=0, from STAGES lines):

| d,seq | mm (Hawkeye matmuls) | lug (fp8 quant lookups) | batch-open | rms+rope+smx+swg+bfa (non-matmul) |
|---|---|---|---|---|
| 64,64   | 2.94 s | 3.36 s | 1.08 s | 0.63 s |
| 128,64  | 28.0 s | 27.5 s | 16.4 s | 1.80 s |
| 256,16  | 7.72 s | 7.65 s | 4.00 s | 0.60 s |

- **mm + lug (the fp8-exactness machinery) is ~73% of the proof.**  The whole
  point of Hawkeye -- bit-exact fp8 accumulation + the per-row pow2 quantization
  lookups -- is where essentially all the cost lives.
- The integer version replaces those 7 matmuls (mm+lug ~ tens of seconds) with 7
  plain integer GEMMs (~7 x 10-100 ms = well under 1 s total, Table B).  So a
  full **integer LAYER proof would be ~1000x cheaper to prove** than the exact-
  fp8 layer -- the price of bit-exactness to the H100 fp8 hardware.  (The
  non-matmul ops are shared and negligible, <1 s, in both.)

## D. The memory wall + the fix

The Goldilocks Hawkeye layer prover OOMs on 24 GB at d>=256 with >16 tokens
(d=256/seq64/zk1, d=512, seq256, batch16 all OOM; RSS hit 17.8 GB at d=256/64
tokens zk0 and 36 GB host RAM at batch=4 zk1).  This is the committed-data
blow-up of the Goldilocks encoding.  The new **Binius Hawkeye** (§21) commits
**46x less data** and proves the matmul **~20x faster** per product
(standalone-measured), which is exactly what lifts this wall -- wiring the Binius
matmul + FRI opening into `p3_transformer` is the concrete next step to scale the
exact-fp8 layer proof past these sizes.

## Knobs varied / not varied

Varied: model size (d, dff=4d), seq len, batch, zk on/off, matmul shapes.  Also
worth varying (not swept here): the FRI rate R and query count Q (soundness vs
size/speed knobs); number of stacked layers (a full model = N x these per-layer
numbers, minus shared setup).

> **ERRATUM (2026-07-14, see COMPARISON_AUDIT.md):** the "integerized" comparison line here is NOT a competitive integer prover — it is 92-98% our own fp8 gadget floor, its matmul component is integrity-only (not ZK), and the "~1000x matmul-level" figure is stale (measured 84-394x on the current binary). Honest fp8 premium vs a zkLLM-class integer baseline is ~10^2-10^3x.
