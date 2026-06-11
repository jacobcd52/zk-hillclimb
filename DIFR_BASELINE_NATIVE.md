# DIFR_BASELINE_NATIVE — end-to-end DiFR of the integer witness chain (post witness-authority switch)

Measured 2026-06-11. This is the number owed by ROPE_ATTENTION_DESIGN §9.6 /
ORCHESTRATOR_REPORT stage-2 caveat 1 ("measure difr once end-to-end before
declaring the stage done"), and the seed value for the Pareto chart's first
point ("baseline-native"). All numbers below are MEASURED (provenance: this
file's §6 commands); nothing is composed or assumed.

## 1. Headline

End-to-end DiFR of the stage-2 integer witness chain (the model the proofs
actually bind, registration `stage2-official1`) against the harness's frozen
FP8-dynamic teacher, on 8 held-out dolly-15k prompts per the
`harness/score.py --difr` protocol:

| metric (protocol definition) | value |
|---|---|
| **difr_mean** (post_gumbel_margin mean, aggregate over 8 prompts) | **8.988 nats** |
| **difr_p99** | **23.996 nats** |
| logit_l2_mean | 825.8 |
| top1 (argmax agreement) | 0.0383 |
| top5 overlap | 0.0730 |
| argmax flips | **7878 / 8192 positions (96.2 %)** |
| max \|logit Δ\| (mean of per-prompt maxima / global max) | 33.0 / 36.18 |
| \|logit Δ\| percentiles, typical prompt (p50 / p90 / p99) | ≈ 3.0 / 7.9 / 13.3 |

**Decomposition (the load-bearing finding).** A float64 replica of the exact
function the chain integerizes — same architecture including every frozen
pipeline deviation, but real arithmetic with zero rounding — scores
**DiFR 8.991** against the same teacher (per-prompt values match the chain's
to ≲ 0.01). The integer chain measured against that replica:

| chain vs. float replica (pure integerization drift — the §9.6 number) | value |
|---|---|
| difr_mean | **2.4 × 10⁻⁶ nats** |
| difr_p99 | **0.0** (exactly, all 8 prompts) |
| max \|logit Δ\| | 0.061 (worst prompt 0.104) |
| logit_l2_mean | 0.47 |
| raw argmax flips | 13 / 8192 (0.16 %) |

So the witness-authority switch (integer-exact rmsnorm R, integer softmax
spec, integer RoPE tables, driver round-half-up rescales, plus the
pipeline-path final norm + lm_head) costs essentially **nothing**: the proven
integer function tracks its real-valued target to a DiFR three orders of
magnitude below even the linears-only fixed-point student's 0.0077
(`llama_difr_results.json`). The 8.99-nat headline gap is **entirely the
frozen pipeline architecture**, not integerization:

1. **No o_proj** — m68-pipeline.py never applies the attention output
   projection (manifest waives all `attn.o_proj.*` ids: "zkLLM upstream omits
   o_proj").
2. **The line-157 permutation** — the pipeline's double transpose+reshape
   scrambles the head-concat (ROPE_ATTENTION_DESIGN §1.3); bound faithfully by
   zkob_headmerge's π.
3. **Softmax temperature 128 instead of 8** — the pipeline reads scale-2^16
   scores as scale-2^20 (SOFTMAX_DESIGN §1.4); the proofs bind what the
   pipeline computes.

These were always documented as bound-not-fixed pipeline quirks; this is the
first measurement of what they cost against the real model. **Verdict: the
integerization/proof machinery is DiFR-clean; the upstream pipeline function
is not a faithful llama-68m.** The baseline-native Pareto point as scored by
the frozen harness protocol is (prove ≈ 743 s wall, sequential-honest, NOT an
official timing — ORCHESTRATOR_REPORT stage 2; **DiFR 8.988**).

## 2. Protocol fidelity (what "per the harness" means here)

Followed exactly from `harness/score.py::score_difr` (frozen v1.0, replicated
line-for-line in `measure/difr_baseline.py`):

- Teacher: fresh `JackFram/llama-68m` in bf16 on CUDA, all 14 targeted linears
  (incl. o_proj, both layers) swapped to `FP8Linear` with
  `IMA_TEACHER_KERNEL=fp8_scaled_mm`; logits = `forced_logits` (float32).
- Prompts: dolly-15k jsonl, `random.Random(seed).sample` over line indices,
  N=8, `"\n\n".join(filter(None, [instruction, context, response]))`.
- Tiling: `add_special_tokens=True`, `reps = -(-1024 // n)`,
  `ids.repeat(1, reps)[:, :1024]`.
- Gumbel: drawn on the CUDA device, `torch.Generator(device)`,
  `manual_seed(seed + 1 + pi)` per prompt, `exponential_().log_().neg_()` —
  the exact noise score.py would generate for the same seed.
- Metrics: `int_model_approximation.metrics.post_gumbel_margin` mean/p99,
  `logit_l2`, `top1_match`; aggregate = mean over prompts.

Necessary deviations, stated precisely:

1. **`score.py` itself was not executed.** It is COORDINATOR-RUN ONLY, writes
   append-only round files, and its student contract is a `student.py`
   `replace(model)` linears swap — the integer chain produces logits directly
   and cannot be expressed in that contract. The protocol was replicated
   instead; every protocol-defining line was copied verbatim.
2. **Seed = 20260611 is a documented baseline seed, not a coordinator round
   seed.** No round has been drawn (LEDGER.md is empty; no
   `private/round*_seed.txt` exists). The seed was fixed before measurement
   and nothing was tuned against it. Official rounds still require fresh
   coordinator-drawn seeds.
3. **dolly.jsonl is cached at `measure/dolly.jsonl`** (sha256
   `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`, also
   recorded in the results JSON), not `private/` — agents must not write
   `private/`. Same upstream URL as score.py.

## 3. Which chain segment carried which authority

| segment | authority | proof status |
|---|---|---|
| embedding → int32 @2^16 (`round(embed(ids)·2^16)`, save_int convention) | pipeline convention (no integer path exists) | waived in frozen manifest |
| rmsnorm ×4 sites (advice R) | **integer-exact bracket** (`prove_walk.compute_R`) — witness-authority switch | proven (zkob_rmsnorm trio) |
| q/k/v matmul + all rescales | driver semantics (int matmul; rescale rem ∈ [−sf/2, sf/2), i.e. round-half-up) | proven |
| RoPE Q/K | **integer spec §2.2, registered int cos/sin tables** — switch | proven (zkob_rope) |
| headslice / per-head scores / rescale 2^13→2^10 | driver semantics + widening shim | proven |
| softmax | **integer spec SOFTMAX_DESIGN §2 (zero advice)** — switch | proven (zkob_softmax) |
| values matmul + rescale, headmerge incl. line-157 π | driver semantics | proven |
| skip adds | int32 adds (orchestrator) | proven (point checks) |
| final norm | **pipeline-float advice** `R = round(2^16/√(mean(X²)+eps))` (m68-pipeline lines 124–127 semantics), then the same W/Y rescale structure | NOT proven (final_norm.rmsnorm waived) |
| lm_head | **pipeline integer path**: `round(W.T·2^16)` int32, int matmul, rescale 2^16 | NOT proven (lm_head.matmul/rescaling waived; commitment_opening + logit_binding the 2 remaining non-waived gaps) |

The chain implementation (`measure/int_chain.py`) imports `compute_R` /
`skip_add` from `orchestrator/prove_walk.py` and re-implements the driver
integer semantics in numpy. **Validation:** 33/33 element-exact convention
checks against the stage2-official1 driver-emitted witness files
(`probe_semantics.py`), and the full 2-layer forward from the registered
input reproduces `data/final_output.i32.bin` **byte-exactly**
(`validate_chain.py`). The 16 registered weights were re-derived from the HF
model and byte-compared against `registration/weights/*-int.bin`
(register.py's provenance guard) before every measurement. Nothing was proven
or re-proven; no GPU driver ran for the chain (integer matmuls run as float64
BLAS with an asserted |Y| < 2^52 exactness bound).

## 4. Full numbers

Per prompt (seed 20260611; chain vs FP8 teacher):

| p | difr_mean | difr_p99 | logit_l2 | top1 | flips/1024 | max\|Δ\| | \|Δ\| p50/p90/p99 |
|--:|--:|--:|--:|--:|--:|--:|--|
| 0 | 8.245 | 23.58 | 737.6 | 0.040 | 983 | 33.32 | 2.70 / 7.01 / 11.99 |
| 1 | 8.655 | 24.34 | 851.9 | 0.059 | 964 | 32.79 | 3.24 / 8.13 / 13.40 |
| 2 | 9.483 | 25.61 | 868.5 | 0.033 | 990 | 33.68 | 3.29 / 8.34 / 14.01 |
| 3 | 8.347 | 20.34 | 775.3 | 0.010 | 1014 | 28.15 | 2.88 / 7.32 / 12.21 |
| 4 | 9.746 | 24.87 | 820.3 | 0.044 | 979 | 36.18 | 3.04 / 7.83 / 13.13 |
| 5 | 7.858 | 21.74 | 746.1 | 0.071 | 951 | 31.32 | 2.78 / 7.04 / 11.49 |
| 6 | 10.142 | 27.48 | 886.0 | 0.021 | 1003 | 33.66 | 3.37 / 8.48 / 13.97 |
| 7 | 9.428 | 24.00 | 920.7 | 0.029 | 994 | 34.72 | 3.53 / 8.86 / 14.38 |

Decomposition aggregates (same prompts, same Gumbel construction):

| comparison | difr_mean | difr_p99 | logit_l2 | top1 | flips | max\|Δ\| |
|---|--:|--:|--:|--:|--:|--:|
| chain vs teacher (headline) | 8.988 | 23.996 | 825.8 | 0.038 | 7878 | 36.18 |
| float replica vs teacher (architecture only) | 8.991 | 24.052 | 825.8 | 0.038 | 7878 | 36.18 |
| **chain vs replica (integerization only)** | **2.4e-6** | **0.0** | **0.47** | **0.9984** | **13** | **0.104** |

Completeness guards (the chain's honest-prover throws) on real-prompt inputs —
**no failures on any of the 8 prompts**:
- scores at scale 2^9: global extreme −154075 → max |score_real| ≈ **301**,
  i.e. 3.4× inside the exp-table domain (±1024). Slightly above
  SCORES_RANGE.md's 276.7 envelope (different prompts; the chain's o_proj-less
  residual feeds layer 1), margin conclusion unchanged.
- gate activations at scale 2^12: global extreme −79755 → |gate_real| ≈ 19.5,
  26× inside the silu-table domain (±512).

## 5. Comparison against pre-switch numbers

- **`results/LEDGER.md` is empty** — no round has been scored; there is no
  pre-switch end-to-end DiFR of the witness chain anywhere in harness history
  (that is exactly the gap this measurement closes). The stage-1 chain
  (float-python attention witness) was never DiFR-scored either.
- Closest existing reference:
  `int-model-approximation/results/llama_pareto/llama_difr_results.json` —
  `zkllm_native_fixed_point` difr_mean **0.0077** / p99 **0.213** / top1
  0.947 (and `codebook` 0.0075 / 0.233). **Not comparable to the headline
  number**, for three reasons: (a) that student swaps only the 14 linears
  (keeping faithful float attention with o_proj, temperature 8, correct head
  merge, float norms and lm_head); (b) its weights are the FP8-dequantized
  weights regridded to 2^-16, whereas the chain registers
  `round(w.float().T·2^16)` of the original checkpoint; (c) it was measured
  on the repo dev prompt, not held-out dolly prompts.
- The like-for-like "what did the integer/proof machinery cost" comparison is
  the **chain-vs-replica** row above (2.4e-6 mean, p99 = 0): the integer
  witness sits far *below* the 0.0077 linears-only floor relative to its own
  target function. The witness-authority switch (rmsnorm R float→exact
  bracket, softmax float→integer spec, RoPE float→integer tables, attn_out
  provenance float→integer chain) did not measurably degrade anything.

## 6. Exact commands (all scripts under `zk-hillclimb/measure/`, new dir)

```bash
cd /workspace/projects/zk-hillclimb/measure
# 1. pin every integer-op convention against stage-2 driver-emitted data (33 checks)
/root/int-model-env/bin/python probe_semantics.py /root/zkorch/stage2-official1
# 2. full-forward byte-exactness vs data/final_output.i32.bin
/root/int-model-env/bin/python validate_chain.py
# 3. the harness-protocol DiFR measurement (headline)
IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python difr_baseline.py --seed 20260611
# 4. architecture-vs-integerization decomposition
IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python decompose_difr.py --seed 20260611
```

Raw outputs: `measure/difr_baseline_native_seed20260611.json`,
`measure/difr_decomposition_seed20260611.json`. Teacher logits cached at
`/root/zkorch-difr/z_ref_20260611_*.npy`. Env:
`/root/int-model-env/bin/python` (torch 2.7.1+cu128, transformers 4.57.3,
RTX 4090). GPU steps (teacher forwards, Gumbel/metric blocks) took
`/tmp/zkorch.gpu.lock`; the integer chain ran on CPU (a 60+ min GPU selftest
was running concurrently — it shares the lock, so teacher passes serialized
cleanly; 0 contention incidents). Read-only: `m68-pipeline.py`, harness/,
int-model-approximation (verified untouched). No git commits, no pushes.

## 7. Honest caveats

1. **Not an official round.** Coordinator must re-run with a fresh round seed
   for anything entering the LEDGER; this measurement's seed is fixed and
   public (20260611), so it is tune-able-against in principle. Nothing here
   was tuned against it.
2. **Embedding + final norm + lm_head carry no proofs** (waived / pending
   ids). Their integerization follows pipeline conventions
   (`save_int`-style `torch.round` = half-even for the embedding quantization
   and the final-norm float advice, vs the drivers' round-half-up rescales —
   both conventions coexist in the pipeline itself). Their contribution is
   inside the 2.4e-6 chain-vs-replica number, i.e. negligible.
3. **The replica defines "the pipeline function" as the integer specs'
   real-valued target**, exactly as the design docs do: no max-shift
   (softmax shift-invariance, SOFTMAX_DESIGN §3.1), float64 RoPE angles (the
   registered tables are float64-generated; the pipeline's float32 cos/sin
   may differ by ±1 ulp@2^16 on a few entries — design §9.2). It is not a
   bit-faithful re-run of m68-pipeline.py's float32/float64 mix; at the
   measured drift scale (max |Δlogit| 0.06) this distinction is far below the
   architecture signal.
4. **Integer matmuls ran as float64 BLAS** with an asserted |Y| < 2^52
   exactness bound (exact integer arithmetic under that bound), not as the
   GPU drivers — but the semantics were validated byte-exactly against
   driver-emitted tensors end-to-end (registered input → final_output), so
   the chain measured IS the proven chain.
5. **Teacher nonlinears are bf16** (score.py's `fresh()` loads bf16): the
   teacher's norms/softmax/embedding run in bf16 while only linears are FP8.
   That is the frozen protocol definition, replicated as-is.
6. **Implication for the capacity budget (TOKEN_CAPACITY.md):** the DiFR
   tolerance floor IS the covert-channel capacity. If a verifier had to
   accept this baseline-native student against the FP8 teacher, the token
   check's tolerance would sit at τ ≈ 24 nats (p99) — off the measured
   δ-curve's right edge (0.25 bits/token already at τ = 1.0), i.e. the
   token-level check would be vacuous and the covert channel effectively
   unbounded. Conversely, with the teacher defined as the pipeline function
   itself, the honest floor is τ_p99 ≈ 0 (chain-vs-replica p99 = 0.0), far
   below the 0.213 operating point measured for the linears-only student.
   **The Pareto-frontier work that buys DiFR is architectural — prove o_proj
   (one more zkob_fc+rescale instance per layer), drop/bind-correctly the
   line-157 permutation (a one-line π⁻¹ change in zkob_headmerge plus
   re-registration, design §9.4), and fix the temperature-128 quirk — not
   tighter integer roundings, which are already at the floor.** Each of those
   is a pipeline-authority decision (the pipeline is frozen; changing it
   re-defines the registered statement) and belongs to the hill-climb as an
   explicit submission, with this measurement as its before/after yardstick.
