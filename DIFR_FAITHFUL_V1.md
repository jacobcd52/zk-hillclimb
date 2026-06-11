# DIFR_FAITHFUL_V1 — end-to-end DiFR of the faithful-arch-v1 chain (the after point)

Measured 2026-06-11. This is the AFTER number to DIFR_BASELINE_NATIVE.md's
before point — same protocol, same 8 held-out dolly prompts, same seed
(20260611), same FP8 teacher (the cached `z_ref_20260611_*.npy` tensors are
byte-identical inputs to the metric). The student is the faithful-arch-v1
integer witness chain (run `/root/zkorch/stage3v2-fa`, orchestrator 65/65 ids
ACCEPT): per-head exact rowmax max-shift + temperature-8 softmax8, plain head
concat (no line-157 permutation), o_proj applied, final-norm integer-exact R,
registered lm_head — STAGE3_FAITHFUL_DESIGN §4 executed. All numbers below are
MEASURED (provenance: §6 commands); nothing is composed or assumed.

## 1. Headline (before / after)

| metric (protocol definition) | baseline-native (stage 2) | **faithful-arch-v1** | ratio |
|---|--:|--:|--:|
| **difr_mean** (post_gumbel_margin mean, 8 prompts) | 8.988 nats | **0.0156 nats** | **÷ 575** |
| **difr_p99** | 23.996 | **0.422** | ÷ 57 |
| logit_l2_mean | 825.8 | 44.3 | ÷ 19 |
| top1 (argmax agreement) | 0.0383 | **0.943** | — |
| top5 overlap | 0.0730 | 0.926 | — |
| argmax flips | 7878 / 8192 (96.2 %) | **464 / 8192 (5.7 %)** | ÷ 17 |
| max \|logit Δ\| (mean of per-prompt maxima / global max) | 33.0 / 36.18 | 2.57 / 3.01 | ÷ 13 |
| \|logit Δ\| typical prompt p50 / p90 / p99 | ≈ 3.0 / 7.9 / 13.3 | ≈ 0.16 / 0.42 / 0.72 | — |

**Decomposition (the §5.2 legs, both measured).** A float64 replica of the
faithful llama-68m function (temperature 8 with allowed-max shift, plain
concat, o_proj applied, original float weights, zero rounding) scores
**0.015639** against the same teacher — per-prompt values match the chain's to
≲ 1e-4. The integer chain against that replica:

| chain vs. float replica (integerization + 2^-16 weight grid) | value |
|---|--:|
| difr_mean | **1.35 × 10⁻⁶ nats** |
| difr_p99 | **0.0** (exactly, all 8 prompts) |
| max \|logit Δ\| | 0.024 (worst prompt 0.044) |
| logit_l2_mean | 0.32 |
| raw argmax flips | 3 / 8192 (0.04 %) |

So, exactly as for the baseline, the proof/integerization machinery costs
essentially nothing (1.35e-6, same class as the baseline's 2.4e-6 — the floor
survived the architecture change, the new rowmax/softmax8/o_proj/lm_head
segments included). The remaining 0.0156-nat headline gap is **entirely the
FP8-teacher-vs-float-architecture gap** — the same residual any faithful
student carries. The three frozen-pipeline quirks that made up the baseline's
9 nats (no o_proj, line-157 scramble, temperature 128) are gone from the
proven function. **Verdict: faithful-arch-v1 proves a DiFR-clean faithful
llama-68m; the architectural revision bought ~3 orders of magnitude, as
designed.**

## 2. Prediction vs. measured (STAGE3 §5.2 scorecard)

| quantity | predicted (§5.2, composed-estimate) | measured | verdict |
|---|---|--:|---|
| difr_mean | ≈ 0.008, honest band 0.004–0.016 | **0.0156** | inside the band (upper edge) |
| difr_p99 | ≈ 0.21–0.25 | **0.422** | ~1.8× above prediction |
| top1 | ≈ 0.95 | 0.943 | ✓ |
| leg 1 (chain vs replica) | ≤ 1e-5 mean | 1.35e-6, p99 = 0.0 | ✓ |

The mean lands inside the predicted band; mean and p99 both sit ~2× above the
`zkllm_native_fixed_point` anchor (0.0077 / 0.213). That anchor was measured
on the repo dev prompt, not held-out dolly prompts, and §5.2(c) explicitly
carried "~2× input-dependence in margin-sensitive metrics" as a band term —
the decomposition confirms the excess is in the replica-vs-teacher leg (i.e.
prompt/teacher-side, identical for ANY faithful student), not in anything this
submission added. The honest reading: on these prompts, a perfect float
faithful student would score 0.0156; the chain scores 0.0156.

## 3. What was measured, and its validation

The chain implementation is `measure/int_chain.py::FaithfulChain` (numpy,
exact integer semantics; baseline `IntChain` untouched and still validates).
Semantics per STAGE3 §4: per-head `mx[i] = max_{j≤i} z_[i,j]` (zkob_rowmax
causal); softmax8 `Dm = MK·(z_ − mx) + (1−MK)·SENT` (SENT = +1),
`E = E8[Dm − LOW8]` (LOW8 = −2^20+2; table sha256-pinned, last entry 0 = the
masked sentinel), `S = Σ_j E`, `P = ⌊(2^17·E + S)/(2S)⌋` (the §4.3 rounding
bracket = round-half-up of 2^16·E/S); headmerge plain concat; o_proj fc +
round-half-up rescale 2^16 between merge and skip; final-norm advice
R = `prove_walk.compute_R` (the §3.1 witness-authority switch); lm_head =
registered `round(W.T·2^16)` int32 matmul + rescale 2^16.

**Validation (validate_chain_faithful.py):** from the registered input,
**every one of the 334 driver-emitted witness files in
`stage3v2-fa/data/` is reproduced byte-exactly** (all rmsnorm X/R/W/Y/out at
all 5 sites, q/k/v/o_proj matmul+rescale, RoPE, all 24 head slices/scores/
z13/z_/mx/P/values, both merges, swiglu, final_output, lm_head logits64 +
logits) — with a coverage assert that no data/ file went uncompared — and
`argmax(logits[:, :32000])` equals the registered `tstar.i32.bin` at all 1024
positions. The 20 registered weights (16 stage-2 + 2 o_proj + final_norm.g +
lm_head) were re-derived from the HF checkpoint and byte-compared against
`registration/weights/*-int.bin` before every measurement. Integer matmuls ran
as float64 BLAS with the asserted |Y| < 2^52 exactness bound; no GPU driver
ran for the chain. **The chain measured IS the proven chain.**

## 4. Full numbers

Per prompt (seed 20260611; chain vs FP8 teacher):

| p | difr_mean | difr_p99 | logit_l2 | top1 | flips/1024 | max\|Δ\| | \|Δ\| p50/p90/p99 |
|--:|--:|--:|--:|--:|--:|--:|--|
| 0 | 0.0162 | 0.440 | 41.9 | 0.916 | 86 | 2.60 | 0.150 / 0.391 / 0.692 |
| 1 | 0.0126 | 0.392 | 44.5 | 0.943 | 58 | 2.49 | 0.160 / 0.416 / 0.724 |
| 2 | 0.0139 | 0.355 | 44.8 | 0.939 | 62 | 2.80 | 0.162 / 0.416 / 0.703 |
| 3 | 0.0223 | 0.547 | 44.4 | 0.952 | 49 | 2.32 | 0.163 / 0.412 / 0.687 |
| 4 | 0.0144 | 0.388 | 44.3 | 0.964 | 37 | 2.74 | 0.158 / 0.413 / 0.725 |
| 5 | 0.0117 | 0.374 | 40.7 | 0.946 | 55 | 1.98 | 0.149 / 0.378 / 0.628 |
| 6 | 0.0188 | 0.513 | 49.1 | 0.943 | 58 | 2.66 | 0.176 / 0.460 / 0.803 |
| 7 | 0.0152 | 0.370 | 45.0 | 0.942 | 59 | 3.01 | 0.161 / 0.421 / 0.731 |

Decomposition aggregates (same prompts, same Gumbel construction):

| comparison | difr_mean | difr_p99 | logit_l2 | top1 | flips | max\|Δ\| |
|---|--:|--:|--:|--:|--:|--:|
| chain vs teacher (headline) | 0.015628 | 0.42223 | 44.34 | 0.9434 | 464 | 3.01 |
| float replica vs teacher (architecture only) | 0.015639 | 0.42223 | 44.35 | 0.9432 | 465 | 3.01 |
| **chain vs replica (integerization + weight grid)** | **1.35e-6** | **0.0** | **0.32** | **0.9996** | **3** | **0.044** |

Updated Pareto rows (STAGE3 §5.3, prediction column replaced by measurement):

| point | difr_mean | difr_p99 | top1 | provenance |
|---|--:|--:|--:|---|
| baseline-native (stage 2) | 8.988 | 23.996 | 0.038 | measured (DIFR_BASELINE_NATIVE) |
| **faithful-arch-v1** | **0.0156** | **0.422** | **0.943** | **measured (this file)** |

Prove-time axis: still NO official timings anywhere (LEDGER empty); the
sequential-honest stage-2 reference is 743 s and the faithful walk's analog
has not been officially timed — out of scope here, flagged as the standing
gap.

Completeness guards (the chain's honest-prover throws) on real-prompt inputs —
**no failures on any of the 8 prompts**:
- scores at scale 2^9: global extreme −165086 → |z_real| ≈ **322**, 3.2×
  inside the ±2^19 envelope (slightly above the baseline measurement's 301 and
  SCORES_RANGE's 277 — different residual stream now that o_proj feeds layer 1;
  margin conclusion unchanged).
- softmax8 shifted diffs: global extreme −195025 vs LOW8 = −1048574 — **5.4×
  inside the E8 table domain** (the §6.7 off-by-one corner is nowhere near).
- gate activations at scale 2^12: global extreme −95402 → |gate_real| ≈ 23.3,
  22× inside the silu-table domain (±512).
- logits at scale 2^16: well inside the rowmax-vpad |z| < 2^25 envelope
  (max |logit_real| ≈ 3.0–24 across prompts; guard asserts in `logits_i32`).

## 5. Protocol fidelity

Identical to DIFR_BASELINE_NATIVE §2 — `difr_faithful.py` is
`difr_baseline.py` with only the student swapped (REG = stage3v2-fa
registration, `FaithfulChain`): same `harness/score.py::score_difr` replication
(prompt draw via `random.Random(seed).sample`, `"\n\n".join` assembly, tiling
`reps = -(-1024//n)`, bf16+FP8Linear teacher with
`IMA_TEACHER_KERNEL=fp8_scaled_mm`, per-prompt Gumbel
`manual_seed(seed+1+pi)` on device, `post_gumbel_margin`/`logit_l2`/`top1`,
aggregate = mean over prompts). The teacher logits were NOT recomputed: the
cached `/root/zkorch-difr/z_ref_20260611_*.npy` files from the baseline
measurement were reused, which makes the before/after comparison exact by
construction (any teacher-side nondeterminism is eliminated). Same three
documented deviations as the baseline: score.py itself is coordinator-run-only
(replicated, not invoked); seed 20260611 is the documented baseline seed, not
a coordinator round seed; dolly.jsonl cached at `measure/dolly.jsonl` (sha256
`2df90833…f794ec`, unchanged).

## 6. Exact commands (scripts under `zk-hillclimb/measure/`)

```bash
cd /workspace/projects/zk-hillclimb/measure
# 1. full-chain byte-exactness vs the stage3v2-fa driver-emitted witness
#    (334/334 files incl. every intermediate, + tstar argmax check)
/root/int-model-env/bin/python validate_chain_faithful.py
# 2. baseline path regression (still byte-exact vs stage2-official1)
/root/int-model-env/bin/python validate_chain.py
# 3. the harness-protocol DiFR measurement (headline)
IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python difr_faithful.py --seed 20260611
# 4. architecture-vs-integerization decomposition
IMA_TEACHER_KERNEL=fp8_scaled_mm /root/int-model-env/bin/python decompose_difr_faithful.py --seed 20260611
```

Raw outputs: `measure/difr_faithful_v1_seed20260611.json`,
`measure/difr_decomposition_faithful_seed20260611.json`. New/changed code:
`measure/int_chain.py` (FaithfulChain appended; IntChain untouched),
`measure/validate_chain_faithful.py`, `measure/difr_faithful.py`,
`measure/decompose_difr_faithful.py`. Env: `/root/int-model-env/bin/python`
(torch 2.7.1+cu128, transformers 4.57.3, RTX 4090). GPU was free; teacher
logits came from cache, so only the metric blocks ran on GPU (under
`/tmp/zkorch.gpu.lock`, 0 contention); the integer chain ran on CPU
(~5 s/prompt). Read-only respected: `m68-pipeline.py`, `harness/`,
`int-model-approximation`, `/root/zkorch/stage3v2-fa`. No git commits, no
pushes.

## 7. Honest caveats

1. **Not an official round.** Same status as the baseline measurement: seed
   20260611 is documented and public, fixed before measurement, nothing tuned
   against it — but a LEDGER entry requires a fresh coordinator-drawn seed and
   coordinator-run scoring. The student-contract question (forward-replacement
   vs linears-swap, STAGE3 §6.4) also remains a coordinator ruling.
2. **difr_p99 measured 0.422 vs the 0.21–0.25 prediction.** The decomposition
   pins the excess entirely on the replica-vs-teacher leg (the float faithful
   architecture itself scores p99 0.42223 on these prompts); the anchor's
   0.213 was measured on the repo dev prompt. Prompt-set dependence of ~2× in
   tail metrics is real and now measured; the coordinator's held-out round is
   the only number that settles the official operating point.
3. **Embedding still carries no proof** (`embedding.lookup` is the one
   waived-and-uncovered id). The embedding quantization
   (`round(embed(ids)·2^16)`, save_int convention) is inside the measured
   1.35e-6 chain-vs-replica number.
4. **Integer matmuls ran as float64 BLAS** with the asserted |Y| < 2^52
   exactness bound, not as the GPU drivers — but every chain tensor was
   validated byte-exactly against driver-emitted data end-to-end (§3), so the
   measured function is the proven function.
5. **Teacher nonlinears are bf16** (frozen protocol definition, replicated
   as-is, identical to the baseline measurement).
6. **The replica defines the target as the integer specs' real-valued
   function** (float64 RoPE angles, exact allowed-max shift — shift-invariant
   in real arithmetic). At the measured drift scale (max |Δlogit| 0.044) this
   convention choice is far below the architecture signal. Unlike the baseline
   decomposition, chain-vs-replica here includes the 2^-16 weight-grid effect
   (the replica keeps original float weights); even so it sits at 1.35e-6.
7. **Capacity-budget implication (TOKEN_CAPACITY.md, the §7.6 reversal):** the
   honest student's tolerance floor is now τ_p99 ≈ 0.42 on these prompts
   (vs ≈ 24 nats baseline-native — a vacuous token check). At τ ≈ 0.42 the
   measured δ-curve gives a bounded, meaningful token channel (same operating
   regime as the linears-only student's 0.213 point, ≈ 0.05 bits/token class),
   and the ZK side stays at ≈ 0 added bits (rowmax selector-tie duty is
   measured per run by prove_walk; the system keeps its zero-advice property).
   This measurement closes the loop the baseline opened: **the proven function
   is now a faithful llama-68m, and the token-level check is no longer
   vacuous.** Remaining Pareto work is prove-time (official timings, the §6.2
   header-lift), not DiFR.
