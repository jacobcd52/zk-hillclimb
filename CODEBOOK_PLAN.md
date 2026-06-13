# CODEBOOK_PLAN.md — Proving the codebook model end-to-end through the ZKP

**Status:** investigation + plan only. No code changed. READ-ONLY pass over
`/workspace/projects/int-model-approximation` (never modified) and
`/workspace/projects/zk-hillclimb/orchestrator`.

**Question:** our ZKP today proves *faithful-arch-v1* — an all-integer fixed-point
pipeline with weights `round(w_dequant·2^16)`, full walk verifies in 27 s,
weight-private. Can we instead prove the **codebook** integerization (the one behind
the codebook DiFR/capacity numbers) with NO new drivers/crypto?

---

## 0. Where the codebook model actually lives

- The codebook scheme used for the capacity/DiFR numbers is applied to **JackFram/llama-68m**
  (same model the ZKP proves), not Qwen — `measure/capacity_dump.py:52` imports
  `MODEL_ID = "JackFram/llama-68m"` from `measure/difr_baseline.py:53`, and the codebook
  branch loads that model in bf16 and swaps linears (`capacity_dump.py:110-128`).
  (The `RedHatAI/Qwen2.5-0.5B-FP8-dynamic` string in `__main__.py:37` is the int-model-approximation
  repo's *own* default entrypoint model; the capacity harness reuses only its `CodebookLinear` class.)
- Swap set: `replace_linears` (`results/llama_pareto/llama_difr.py:80-91`) replaces exactly the 7
  projection types `q,k,v,o,gate,up,down` in every layer; **`lm_head` stays float**
  (`llama_difr.py:81` comment, `capacity_dump.py:118`).
- The integerization itself is `CodebookLinear` (`int-model-approximation/src/int_model_approximation/__main__.py:421-447`).

---

## 1. THE MAKE-OR-BREAK FINDING

### Is the codebook matmul a clean int32×int32→int64 product? **YES (the core only).**

`CodebookLinear.forward` (`__main__.py:440-447`):

```python
x_i32, x_scale = _per_token_fp8_int32(x)              # __main__.py:443
y = _int32_matmul(x_i32, self.weight_t, x_scale, self.weight_scale)   # :444
```

- Weight integers: `codebook_i32 = _fp8_e4m3_to_int32(weight)` = `round(fp8_value · 512)`
  (`__main__.py:429`, `:157-158`). Because `FP8_E4M3_CODE_SCALE = 512 = 2^9` and e4m3's
  smallest subnormal is `2^-9`, every FP8 value times 512 is an **exact** integer in
  `[-229376, 229376]` (= `±448·512 ≈ ±2^17.8`). Clean integer operand. ✓
- Activation integers: `_per_token_fp8_int32` (`:161-163`) = `round(fp8(x)·512)`, exact integers,
  same range. ✓
- The accumulation in `_int32_matmul_kernel` (`__main__.py:274-289`) is
  `accum (int64) += a.to(int64) * b.to(int64)` — **identical** to the clean
  `_int32_raw_matmul_kernel` (`:204-219`) that the already-proven `ZkllmFixedPointLinear`
  uses (`llama_difr.py:65`). So the *inner product* is structurally the same op the
  `zkob_fc` driver already proves. ✓

### But is the codebook *linear* structurally identical to the fixed-point path end-to-end? **NO.** Two independent breakers:

**Breaker A — per-token × per-output-channel FLOAT dequant (per-group scaling).**
The matmul output is *not* an integer fed forward; it is immediately turned into float by
two non-power-of-two, per-row/per-column scales inside the kernel (`__main__.py:291-293`):

```python
x_scale = load(x_scale_ptr + offs_m)     # per-TOKEN (per output row m), float32, data-dependent
w_scale = load(w_scale_ptr + offs_n)     # per-OUTPUT-CHANNEL (per col n), float32
out = accum.to(float32) * x_scale[:,None] * w_scale[None,:]
```

- `w_scale[n] = (absmax_row_n / 448) / 512` — a **per-output-channel float** (`__main__.py:430`,
  `quantize_fp8_per_row` `llama_difr.py:35`). Static/committed, but **not a power of two**.
- `x_scale[m] = (amax_token_m / 448) / 512` — a **per-token float** computed at runtime from
  the row's own absmax (`_per_token_fp8` `__main__.py:148-154`). **Data-dependent.**

The orchestrator's only post-matmul resampler is `zkob_rescale`: a single **uniform
power-of-two** truncation `y = floor((X + sf/2)/sf)`, `sf = 2^k`, validated by a range-check
lookup table of size `N = sf` (`common.py` rescale registrations `:446-454`, `:471-476`,
`:494-497`; `zkob_rescale.cu`). It takes one shift constant per op (`LOG_SF=16`, `GATE=20`, …).
**It cannot express a per-channel or per-token, non-power-of-two, data-dependent multiplier.**
This is exactly the "per-group scaling … the current drivers can't express" case.

**Breaker B — the surrounding ops are FLOAT in the codebook model.**
Only the 7 linears are swapped; rmsnorm, rope, softmax, attention scores/weights, residual
adds, and the dequant multiply itself all run in **float bf16** (the unswapped llama forward,
`capacity_dump.py:122` `forced_logits`). The ZKP proves an **all-integer** pipeline
(`int_chain.py` / `FaithfulChain`: integer `rmsnorm`/`fc`/`rescale`/`rope`/`softmax8`/`glu`/
`headmerge`/`skip`). Proving the *literal measured codebook model* would require proving float
nonlinearities — which the integer drivers cannot do. Proving an *integer-glue pipeline with
codebook integer weights* is feasible, but it is a **different model** than the one measured
(its DiFR ≠ the published codebook DiFR, because integer vs float glue differs).

> **Bottom line:** the codebook matmul *core* is provable as-is by `zkob_fc`, but the codebook
> *integerization* is a per-channel/per-token affine-quant scheme wrapped around float glue —
> **not** the global-2^16-lattice integer pipeline the drivers prove. **Make-or-break = NO**:
> the existing drivers cannot prove the codebook model end-to-end without new obligations.

### The minimal extra obligation, if we still want "codebook"
Support a **per-output-channel weight scale + per-token activation scale** rescale (a general,
non-power-of-two, data-dependent affine dequant), and decide the glue story (§5). Note the
*degenerate* alternative: if you collapse both scales to a single global `2^16` shift, the
codebook weight `round(fp8(w)·512)·w_scale` re-gridded to `round(·2^16)` **is exactly**
`ZkllmFixedPointLinear` = **faithful-arch-v1, already proven** (`llama_difr.py:40-74`). So
"no new code" and "genuine codebook integerization" are mutually exclusive: you get one or
the other, not both.

---

## 2. EXACT CHANGE-LIST (classified)

Target = prove a codebook-weighted **integer** pipeline (Breaker B resolved by integerizing
glue, as faithful-arch already does; the deviation from the measured float-glue model is
documented as a residual, like the existing rmsnorm activation-statement residual).

| # | Change | File:line | Class | Effort |
|---|--------|-----------|-------|--------|
| 1 | **Per-channel weight scale + per-token activation rescale gadget.** Replace the uniform `zkob_rescale` after each codebook matmul with a rescale that applies a committed per-output-channel fixed-point multiplier and a witnessed per-token multiplier, with a remainder/range argument. | new kernel alongside `zkob_rescale.cu`; registered in `common.py` `:446-497`, `:563-611` | **NEW driver/protocol code (crypto-adjacent)** — the one thing we hoped to avoid | Large: 1 design cycle + 1 implement/verify cycle |
| 2 | **Prove the dynamic per-token activation scale.** `x_scale[m]=amax_row/448/512` is data-dependent → the proof must bind the per-row amax (a max/argmax obligation per token per linear). Not in today's walk. | new advice + check; wire in `prove_walk.py` per-linear | **NEW witness-gen + light protocol** | Medium: folds into #1's cycle |
| 3 | **Weight provenance: source codebook integers.** Change the registered weight derivation from `round(w·2^16)` to `round(fp8(w_dequant)·512)` **plus** the per-row `w_scale` committed as a side vector. | `register.py:66-67` (`w_int = round(w_orig*sf)`); guard `int_chain.py:77-92`, `:234` | **witness-wiring + config** (provenance guard must learn the new formula) | Small–Medium |
| 4 | **Activation quant in the chain.** Change `x0 = round(x·2^16)` and every per-op `round(·*2^16)` to the per-token FP8-codebook quant `round(fp8(x)·512)` with the per-token scale carried. | `capacity_dump.py:101` analog; `int_chain.py:48-52,121-162` (`imatmul`/`rescale` sites) | **witness-wiring** | Medium (touches every linear site in `FaithfulChain`) |
| 5 | **Re-table the nonlinearities for the new activation scale.** swiglu/exp/rowmax/scores tables are calibrated to 2^16-scaled activations (swiglu domain `2^22` centered `±2^21`, softmax envelope `[-2^19,2^19)`, softmax8 `len_r 2^14`, gate lands `@2^12`). Codebook activations land on different (per-token) scales → table domains must be recomputed or the activations renormalized into them. | `common.py:71,80-94`; domain asserts `prove_walk.py:519`, `:321-325` | **config (table widths/bounds) + possible NEW renorm step** | Medium; risk of a new renorm op if scales don't fit |
| 6 | **lm_head.** Keep integerizing at `2^16` (see §4). | `int_chain.py:99,186`; `common.py:100,412-414` | **config / no change** | None |
| 7 | **Matmul driver (`zkob_fc`), rmsnorm, rope, softmax8, headmerge, skip cores.** | — | **driver/protocol change = NONE** ✓ | 0 |
| 8 | int64/field magnitude headroom for codebook-range operands (see Risks). | `int_chain.py:48-52` (`imatmul` `|Y|<2^52` assert) | **config / verify** | Small |

**Classification summary:** the matmul + nonlinearity *cores* need **no driver/crypto change**
(team hypothesis holds *for the matmul*). But the **rescale** does (#1, #2) — a genuinely new
per-channel/per-token affine-dequant gadget, plus dynamic-scale witnessing. The rest is
witness-wiring (#3, #4), config/retabling (#5), and free (#6, #7).

---

## 3. SCALE-RECONCILIATION TABLE (codebook vs orchestrator)

| Quantity | Codebook path | Orchestrator constant | Differ? |
|----------|---------------|------------------------|---------|
| Weight integerization | `round(fp8(w)·512)`, per-row `w_scale=(absmax/448)/512`; ints `±2^17.8` (`__main__.py:429-430`) | `round(w·2^16)`, int32, global lattice (`register.py:66-67`) | **YES** — per-channel float vs global 2^16 |
| Activation quant | per-token `round(fp8(x)·512)`, scale `(amax/448)/512` (`__main__.py:161-163`) | `round(x·2^16)`, clamp `±2^30` (`int_chain.py:48-52,60-64`) | **YES** — per-token float vs global 2^16 |
| Matmul accumulate | int32×int32→int64 (`__main__.py:274-289`) | int64 (`imatmul`, `int_chain.py:48-52`) | **SAME** (structurally) |
| Post-matmul rescale | `× x_scale[m] × w_scale[n]` float (`__main__.py:291-293`) | `÷2^16` uniform shift; gate `÷2^20`, scores `÷2^13` then `÷2^10`, etc. (`common.py:66-100`) | **YES** — per-row/col float vs uniform pow-2 |
| lm_head | **float**, unquantized (`llama_difr.py:81`, `capacity_dump.py:118`) | `round(w·2^16)` + int matmul + `÷2^16` (`int_chain.py:99,186`; `LM_RESCALE_LOG=16` `common.py:100`) | **YES** — codebook float vs integer |
| rmsnorm / rope / softmax / scores / residual | **float bf16** (unswapped) | integer w/ fixed scales + lookup tables (`int_chain.py`, `common.py:67-94`) | **YES** — codebook float vs integer |
| swiglu table domain | n/a (float silu) | `SWIGLU_LEN=2^22`, gate `@2^12` (`common.py:67,71`) | re-derive for new scale (#5) |
| softmax envelope | n/a (float) | scores `[-2^19,2^19)`, exp `2^20`, softmax8 `len_r 2^14` (`common.py:80-94`) | re-derive for new scale (#5) |

Constants that **match**: the int64 accumulator semantics, and `LOG_SF=16` *if* lm_head stays
fixed-point. Everything weight/activation/rescale-related **differs**.

---

## 4. lm_head RECOMMENDATION

**Recommend (a): integerize lm_head fixed-point at `2^16`, exactly as the orchestrator does
today** (`int_chain.py:99,186`, `LM_RESCALE_LOG=16`).

- (b) "leave lm_head out of proven scope" would leave the final logits — the thing the
  capacity/DiFR margin is computed on — **unproven**, gutting the end-to-end claim and the
  logit-binding the walk closes on (`common.py:412-414`).
- The deviation from codebook's float lm_head is minor and already characterized: the proven
  reference is the served `M_int`, and the fixed-point lm_head is one extra `round(·2^16)` +
  one matmul + one `÷2^16` we already prove. It is the same minor deviation faithful-arch-v1
  accepts.

---

## 5. GO / NO-GO

**"No major code changes" = FALSE** (for the genuine codebook integerization).

- ✅ True part: the **matmul and all nonlinearity drivers are scheme-agnostic and need zero
  change** — the int32×int32→int64 codebook product is exactly what `zkob_fc` proves.
- ❌ False part: the codebook scheme's **per-output-channel + per-token, non-power-of-two,
  data-dependent dequant** has no representation in the current `zkob_rescale` (uniform pow-2
  shift). That is a **new gadget + a new dynamic-scale witness** (changes #1, #2) — precisely
  the "new driver/crypto" the brief hoped to avoid. Additionally the literal measured codebook
  model has **float glue** that an integer ZKP cannot prove (Breaker B); we can only prove a
  codebook-*weighted integer* pipeline, which is a documented approximation of the measured model.

**Realistic effort (agent-cycles):**
- *Cheapest "codebook" that needs no new code:* collapse to global 2^16 → **that is
  faithful-arch-v1, already done (0 cycles)** — but it is NOT the codebook integerization that
  produced the codebook capacity numbers. Use only if "codebook" can mean "FP8-dequant weights
  re-gridded to 2^16".
- *Genuine codebook integerization (integer glue):* **~4–6 agent-cycles** —
  1 design (per-channel/per-token affine-dequant gadget + dynamic-amax binding),
  1–2 implement (#1 gadget + #2 witness, the only crypto work),
  1 rewire (#3 provenance, #4 chain quant),
  1 retable + re-bound (#5, #8),
  1 full-walk reverify (battery + DiFR + leak regression).
- *Literal float-glue codebook model:* **NO-GO** with integer drivers.

**Recommendation:** if the goal is an end-to-end proof of *the codebook integerization*, scope
the per-channel/per-token affine-dequant gadget (#1/#2) as a small new driver and accept the
integer-glue residual; do **not** advertise it as "no new drivers." If the goal is merely "a
weight-private proof over FP8-derived weights," faithful-arch-v1 already delivers it.

---

## 6. RISKS

1. **New crypto despite the hypothesis.** The per-channel/per-token affine dequant (#1) is the
   one place the "scheme-agnostic drivers" assumption fails. It is genuinely new protocol
   surface, not a config flip.
2. **Dynamic activation scale must be proven.** `x_scale[m]` is `amax_row/448/512`
   (`__main__.py:148-154`) — a per-token max the prover currently never commits. Binding it adds
   a max/range obligation per token per linear (#2); a malicious prover choosing a wrong amax is
   otherwise unconstrained.
3. **Table-domain breakage (#5).** swiglu (`2^22`, gate `@2^12`), softmax envelope `[-2^19,2^19)`,
   softmax8 `len_r 2^14`, rowmax `2^20` are all calibrated to the 2^16 activation lattice
   (`common.py:67-94`; asserts `prove_walk.py:519`, `:321-325`). Codebook activations sit on a
   different, per-token scale; without a renorm step their values can fall outside these table
   ranges → proofs abort or tables must grow (cost/■ size).
4. **Magnitude / field & int64 headroom.** Codebook operands reach `±2^17.8` vs the fixed-point
   path's `|w|·2^16 ≈ 2^16` and activation clamp `±2^30`. Worst-case `down_proj` (K=3072):
   `≈2^35.6·3072 ≈ 2^47.3` — still under `imatmul`'s `|Y|<2^52` assert (`int_chain.py:52`) and
   the kernel's `2^62` (`__main__.py:103`), so int64/field are fine, **but the `2^52` assert and
   any range tables sized for the 2^16 envelope must be re-checked** (#8).
5. **Proven model ≠ measured model.** Integer-glue codebook will not reproduce the published
   codebook DiFR (float glue). The capacity/covert-channel conclusions are about the *float-glue*
   codebook; the proof would cover a near-cousin. This must be stated, not papered over.
6. **lm_head residual.** Fixed-point lm_head (recommended) deviates from codebook's float lm_head
   — minor, same residual faithful-arch-v1 already documents.
