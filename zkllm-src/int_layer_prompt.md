# Task: build a full-layer INTEGER transformer ZKP on the Goldilocks/hash substrate

Goal: produce the missing baseline — a composed single-proof ZK prover for ONE
integerized transformer layer's forward pass, on our FAST substrate (Goldilocks
small field + hash/Basefold commitments), at the SAME config as the fp8 layer
prover, so we can drop a real MEASURED "integer layer" line into the overhead
plot (currently only an estimate: INT_BASELINE.md §3.3 bracket 2–134x). This is
the integer counterpart of p3_transformer.cuh's fp8 Hawkeye layer.

Read FIRST: INT_BASELINE.md, COMPARISON_AUDIT.md (why the old int comparison was
circular), then the code: p3_transformer.cuh (the fp8 composed-layer template —
mirror its structure/seams/ZK landing), p3_matmul.cuh (p3mm integer FC matmul,
integrity), p3_logup.cuh (lookup arg for range checks / rescale / table ops),
p3_basefold.cuh (hash commit + Basefold open), p3_zkopen.cuh / p3_zk.cuh /
p3_zksumcheck.cuh / p3_zkmatmul.cuh (the validated ZK/hiding primitives), and
the zkob integer gadgets (../zkob_* : rescale, rmsnorm, softmax8, glu, rope) for
the integer SEMANTICS to replicate (they are BLS; you reimplement on Goldilocks
using p3_logup lookups + p3mm, NOT by porting BLS code).

Environment rules (HARD): work dir /root/zkllm, build to /root, always
`nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp -I.`; ONE GPU job at a time;
41 GB host cap; launch long builds/benches setsid + `echo 1000 > oom_score_adj`
so they survive your session AND get killed before the pod; poll logs with
tail/grep, never cat big files; NEVER git commit/push (coordinator does);
NEVER touch the fp8 sources (p3_transformer.cuh, p3_hawkeye*.cuh, p3_model.cuh)
or their benches — build NEW files (p3_int_layer.cuh, p3_int_layer_bench.cu,
gadget files p3_int_*.cuh). Do not launch other claude agents/workflows.

## What "integerized layer" means here (scope — read carefully)

A fixed-point integer transformer layer (residual stream scale ~2^16, à la
zkLLM/zkob), one llama-style block:
  RMSNorm -> Wq/Wk/Wv (int matmul + rescale) -> RoPE -> QK^T -> softmax
  -> P·V -> Wo (matmul+rescale) -> residual add -> RMSNorm -> SwiGLU
  (Wg,Wu matmul + silu·mul via lookup + rescale) -> Wd (matmul+rescale)
  -> residual add.
It does NOT need to be bitwise-faithful to a specific external model — it must
prove a SELF-CONSISTENT valid integer forward pass (prover commits X, weights,
all intermediates; verifier checks every op + the seams between them). Integer
op gadgets, all on Goldilocks via existing primitives:
- **matmul**: p3mm (already integer). Attention QK^T / P·V are matmuls too.
- **rescale** (divide accumulator by 2^sf, the after-matmul requantize):
  prove y = (x - r)/2^sf with 0<=r<2^sf via a range-check lookup (p3_logup)
  on r and on y's bit-width. This is the integer analogue of the fp8 `qnt`.
- **RMSNorm** (integer): sum of squares + integer inverse-sqrt via a
  lookup/advice bracket (zkob rmsnorm semantics; range-checked).
- **softmax** (integer/quantized): row-max subtract + exp via table lookup
  (p3_logup fixed table) + normalize by lookup-reciprocal (zkob softmax8).
- **SwiGLU**: silu(x)·y via lookup table on the gate + integer mul.
- **RoPE**: fixed cos/sin table multiply (integer), a matmul/elementwise gadget.
- **residual add**: integer add (cheap sumcheck).
Reuse the p3_logup lookup machinery for EVERY table/range op. Compose them into
ONE proof with seam-binding (shared commitment-root equality between an op's
output columns and the next op's input columns — copy p3_transformer.cuh's seam
pattern exactly). Land ZK last (hiding openings + masked sumchecks +
salted leaves via p3_zkopen/p3_zk), with the hiding-battery methodology
(chi-square uniform + negative control + witness-recovery attack) — the SAME
gates the fp8 layer passed. Do NOT claim ZK until those teeth pass.

## Iterate (checkpoint each increment to /root/zkllm/INT_LAYER_LOG.md)

1. Integer gadget set: build + selftest each gadget (honest cases accept
   bitwise vs a python/CPU integer reference you write as int_layer_ref.py;
   adversarial cases reject with per-gadget teeth). Target parity with the
   fp8 gadget test rigor (25/25-style suites).
2. Compose the full integer layer in one proof (p3_int_layer.cuh): honest
   forward accepts, matmul chain by root-equality, seam-binding rejects
   (tampered codes/padding/head-slice/concat/transpose), public I/O binding.
3. Land full ZK; run the hiding battery with negative controls.
4. Bench binary p3_int_layer_bench.cu: args `<seq> <d> <nh> <dh> <dff> <batch>
   <zk> <tables>` mirroring p3_tb_*, printing a BENCH line (prove/verify
   seconds, proof MB, RSS) + a STAGES line. Measure zk=1 at the fp8 sweep grid:
   (seq 64/128/256/512/1024, d=64) ; (d 64/128/256/512, seq=64) ;
   (batch 4/16/64, seq=128 d=64). Append to /root/zk_int_layer_results.json
   as {tag, seq, d, nh, dh, dff, batch, tokens, params, int_prove_s,
   verify_s, proof_mb, rss_gb}. OOM/skip honestly if a config exceeds 41 GB.

## Deliverables

- New sources in /root/zkllm (p3_int_layer.cuh + gadget + bench + ref; NO edits
  to fp8/model sources).
- /root/zkllm/INT_LAYER.md: what was built, the gadget/compose/ZK test results,
  the measured table, and honest caveats (faithfulness scope, any gadget
  simplification vs zkob, soundness-bits/field parity with the fp8 side —
  same Goldilocks base-field challenge width, note if GL2 needed).
- /root/zkllm/int_layer_done.flag: OK/PARTIAL <reason>; line 2 one-sentence
  summary; line 3 the measured int-layer prove seconds at d=64 seq=128 (the
  anchor to compare against fp8's 18.0 s and the est. bracket).

Honesty: never fake a number or a passing test; ZK claims require the hiding
teeth green; if a gadget is simplified vs a faithful integerized model, say so
and state what it would cost to close. If you near your session/token limit,
write the flag PARTIAL with a precise handoff at the top of INT_LAYER.md — a
later session (or the coordinator) will continue. This is expected to take
multiple sessions; incremental committed progress in the LOG is the success
condition.
