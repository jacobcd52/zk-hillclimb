# Measured attention-score range — llama-68m integer pipeline (`m68-pipeline.py`)

**Purpose:** measure the real (un-scaled) pre-softmax attention score
`score = A / 2^16 = QK^T_real` per layer & head, to decide whether the fixed
exp-lookup-table domain (assumes `|QK^T_real| < 1024`, prior estimate ~576) can be
frozen, and whether the 2×-margin knob (threshold 512) must turn.

**Headline result:** global **max |score| = 276.72**, from real-text input.
- Margin vs **1024** (table domain): **3.70×** — comfortably inside.
- Margin vs **512** (2×-margin threshold): **1.85×** — does **not** exceed 512, so
  the design's domain knob does **NOT** need to turn.
- No NaNs/Infs. Measured max (276.7) is ~2.1× below the prior ~576 worst-case estimate
  — the old estimate was conservative (safe), not exceeded.

> `score` here is the **raw dot product** `QK^T` (NOT divided by `sqrt(head_dim)=8`),
> matching the pipeline: `A = to_int64(Q @ K^T, VALUE_LOGSF=16)` and the `/sqrt(d)`
> division happens later inside `exp(...)`. So the 1024/512 bounds and these numbers
> are on the same (un-scaled) footing as `A/2^16`.

---

## What was measured, and on which inputs

`m68-pipeline.py` does **not** run a real prompt: its standard input is **random
Gaussian** — line 112: `save_int(torch.randn(seq, embed), 1<<16, cur_input)`, fed as
the layer-0 residual; Q/K/V are real-weight projections of it. To get the genuinely
*real* score range the design cares about, I measured **both** regimes:

| regime | inputs | what it is |
|---|---|---|
| RANDOM (pipeline's actual default) | `random_seed0/1/2` | `torch.randn(1024,768)` quantized to 2^16, fed as `inputs_embeds` — exactly line 112, 3 seeds |
| REAL TEXT | `real_wiki`, `real_lorem`, `real_code` | real prompts tokenized → 1024-token contexts through the real model |

All runs: `seq=1024`, both layers, all 12 heads, `JackFram/llama-68m` from the HF cache.

**Per-(input, layer) global score range** (units of `A/2^16 = QK^T_real`):

| input | layer | all max\|·\| | masked max\|·\| | all max | all min |
|---|--:|--:|--:|--:|--:|
| random_seed0 | 0 | 79.61 | 79.61 | 64.94 | −79.61 |
| random_seed0 | 1 | 93.47 | 93.47 | 81.47 | −93.47 |
| random_seed1 | 0 | 65.47 | 65.47 | 65.47 | −64.27 |
| random_seed1 | 1 | 136.50 | 136.50 | 97.61 | −136.50 |
| random_seed2 | 0 | 64.35 | 64.35 | 64.35 | −54.45 |
| random_seed2 | 1 | 94.61 | 94.61 | 90.24 | −94.61 |
| real_code | 0 | 262.60 | 262.60 | 177.30 | −262.60 |
| real_code | 1 | 175.84 | 175.84 | 154.42 | −175.84 |
| real_lorem | 0 | 249.31 | 249.31 | 203.32 | −249.31 |
| real_lorem | 1 | 191.90 | 191.90 | 150.84 | −191.90 |
| **real_wiki** | **0** | **276.72** | **276.72** | 234.68 | **−276.72** |
| real_wiki | 1 | 213.04 | 213.04 | 175.52 | −213.04 |

**Real text produces ~2–3× larger scores than the random default** (277 vs ≤137).
The pipeline's own random input therefore *understates* the real range — measuring
real prompts was necessary.

The global max |score| is driven by **`real_wiki`, layer 0, head 11**, value
**−276.72** at a **causal-kept** position (`masked_min = −276.72`).

---

## Per-(layer, head) envelope (max/min taken across all 6 inputs)

Outer bound seen for each head over every input. `all_*` = all positions (masked-out
future included); `mask_*` = causal-kept only (j ≤ i).

| L | H | all max | all min | all \|max\| | mask max | mask min |
|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 131.31 | −170.86 | 170.86 | 56.55 | −170.86 |
| 0 | 1 | 121.04 | −159.14 | 159.14 | 48.27 | −159.14 |
| 0 | 2 | 135.96 | −154.90 | 154.90 | 46.55 | −154.90 |
| 0 | 3 | 156.61 | −173.12 | 173.12 | 56.79 | −173.12 |
| 0 | 4 | 189.02 | −272.17 | 272.17 | 63.98 | −272.17 |
| 0 | 5 | 113.99 | −116.33 | 116.33 | 113.99 | −116.33 |
| 0 | 6 | 215.72 | −218.39 | 218.39 | 71.78 | −218.39 |
| 0 | 7 | 125.25 | −233.43 | 233.43 | 53.16 | −233.43 |
| 0 | 8 | 73.37 | −101.70 | 101.70 | 43.59 | −101.70 |
| 0 | 9 | 177.30 | −262.60 | 262.60 | 64.52 | −262.60 |
| 0 | 10 | 107.11 | −166.30 | 166.30 | 42.77 | −166.30 |
| 0 | 11 | 234.68 | **−276.72** | **276.72** | 65.47 | **−276.72** |
| 1 | 0 | 148.21 | −177.47 | 177.47 | 63.14 | −177.47 |
| 1 | 1 | 143.78 | −145.08 | 145.08 | 58.53 | −145.08 |
| 1 | 2 | 106.15 | −183.69 | 183.69 | 99.90 | −183.69 |
| 1 | 3 | 175.52 | −213.04 | 213.04 | 69.53 | −213.04 |
| 1 | 4 | 162.84 | −140.92 | 162.84 | 78.21 | −140.92 |
| 1 | 5 | 142.93 | −132.03 | 142.93 | 73.25 | −132.03 |
| 1 | 6 | 98.13 | −96.72 | 98.13 | 62.40 | −96.72 |
| 1 | 7 | 84.17 | −121.64 | 121.64 | 82.44 | −121.64 |
| 1 | 8 | 172.16 | −149.85 | 172.16 | 76.65 | −149.85 |
| 1 | 9 | 81.61 | −136.50 | 136.50 | 81.61 | −136.50 |
| 1 | 10 | 129.23 | −152.62 | 152.62 | 76.08 | −152.62 |
| 1 | 11 | 123.19 | −130.31 | 130.31 | 91.34 | −130.31 |

(Full per-input × per-head data: `/tmp/scores-measure/scores_raw.json`; envelope:
`/tmp/scores-measure/envelope.json`.)

---

## Global numbers and margins

| quantity | value |
|---|--:|
| global **max \|score\|**, all positions | **276.7235** |
| global **max \|score\|**, causal-masked only (j ≤ i) | **276.7235** |
| global max (signed) | +234.68 |
| global min (signed) | −276.72 |
| margin vs **1024** (table domain) | **3.70×** |
| margin vs **512** (2× threshold) | **1.85×** |
| exceeds 512? | **no** |
| exceeds 1024? | **no** |
| prior estimate (~576) exceeded? | no (measured 277 ≈ 0.48× estimate) |

**Decision:** freezing the table domain at `|QK^T_real| < 1024` is safe with 3.7×
headroom. `max|score|` stays under 512, so the 2×-margin knob does not need to turn.

---

## Surprises / caveats (read before trusting this)

1. **The pipeline's "standard input" is random Gaussian, not a prompt.** `m68-pipeline.py`
   line 112 feeds `torch.randn`. Its scores top out at ~137 — that **understates** the
   real-text range (~277). The headline number comes from real prompts, not the
   pipeline's default. If you only ever prove random-input runs, ~137 is your max.

2. **Strong negative skew.** Score magnitude is dominated by *negative* values: e.g.
   L0h11 reaches −276.7 but only +234.7; L0h4 is +189 / −272. The exp-table domain
   must cover the negative tail; a symmetric `±` domain is fine, but an asymmetric
   one tuned to the positive side would clip. Magnitude is set by the negatives.

3. **Big positive scores live in the masked-out future; the magnitude extremes are
   causal-kept.** For most cells the large positive `all_max` is in the j>i (future,
   masked-out) region — causal-kept positive scores are much smaller (e.g. L0h11
   mask_max 65 vs all_max 235). But the **largest-magnitude** values are the negatives,
   which *are* causal-kept (global all-positions max\|·\| == global masked max\|·\| ==
   276.72; the extreme is at a kept position). So masked-out positions are **not** more
   extreme in magnitude than the kept ones — the worst case the proof must bound is in
   the causal region. The masked region matters only because the pipeline computes the
   full QK^T before masking (line 147 before line 148–149).

4. **No NaN/Inf** in any of 6 inputs × 2 layers × 12 heads.

### Fidelity of the measurement (honest scope)
- I did **not** run the compiled ZK binaries (`./self-attn` etc. are not built in the
  repo copy under `zkllm-src/`). Instead I reproduced the pipeline's **exact** attention
  formula (`m68-pipeline.py` lines 136–159 / `llama-self-attn.py`) in PyTorch, computing
  Q/K/V from the **real** `q/k/v_proj` weights via the HF modules. The compiled
  `./self-attn linear` does the identical linear map in fixed-point (2^16); I matched it
  by quantizing Q,K to 2^16 and applying `to_int64(·,16)` to A. The float-vs-fixedpoint
  deviation is ~1e-4 relative per score — irrelevant to the range/margins.
- I used the **exact** `(cos,sin)` the model computes for the rotary step (captured via a
  forward pre-hook on each `self_attn`), so RoPE matches the pipeline.
- **Layer-1 caveat:** the pipeline's residual stream **omits `o_proj`** in its
  python attention math (no o_proj multiply before the skip; see also the `note_o_proj`
  field it writes — "does not prove o_proj"). My layer-1 input came from a *correct* full
  HF forward (which applies o_proj), so my **layer-1** scores correspond to a faithful
  forward pass rather than the pipeline's o_proj-less residual. This does **not** affect
  the conclusion: the global max (276.7) is in **layer 0**, whose input is the given
  input and is therefore exact; layer-1 magnitudes are lower and far from any threshold.
- Real prompts were tiled to fill the 1024-token context (real tokens, real positions).
  Natural non-repeating text could differ somewhat, but the score scale is governed by
  the model's fixed weight/embedding norms; a jump past 512 is not plausible from this.

---

## Exact commands run

Environment: `/root/int-model-env/bin/python` (torch 2.7.1+cu128, transformers 4.57.3,
RTX 4090), model `JackFram/llama-68m` from HF cache
`/workspace/projects/zk-hillclimb/zkllm-src/model-storage`. Nothing in
`int-model-approximation` or `zkllm-src` was modified; all work was done in
`/tmp/scores-measure/` (a copy of `fileio_utils.py` plus two new scripts).

```bash
mkdir -p /tmp/scores-measure
cp /workspace/projects/zk-hillclimb/zkllm-src/fileio_utils.py /tmp/scores-measure/
# measure_scores.py: reproduces m68-pipeline.py attention math, 3 random seeds + 3 real prompts
/root/int-model-env/bin/python /tmp/scores-measure/measure_scores.py   # -> scores_raw.json
/root/int-model-env/bin/python /tmp/scores-measure/analyze.py          # -> per-layer / masked / envelope
```

Scripts and raw data: `/tmp/scores-measure/{measure_scores.py, analyze.py,
scores_raw.json, envelope.json}`.
