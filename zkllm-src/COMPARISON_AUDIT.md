# Audit: the "fp8 costs only 8–40x more than integer at the whole-layer level" claim

Independent soundness audit of our float-vs-integer ZKP benchmark methodology,
2026-07-14, RTX 4090.  Everything below was verified in source
(`p3_matmul.cuh`, `p3_matmul_bench2.cu`, `p3_transformer_bench.cu`,
`p3_transformer.cuh`, `bench_saturated.py`, `bench_final_plots.py`,
`final_plots.py`) and, where marked *measured*, re-measured with small GPU
runs (`audit_imm_phases.cu` → `/root/audit_imm_phases`,
`/root/audit_s64_dbg.log`, `audit_ratios.py`).

**Verdict up front:** the 8–40x headline is real arithmetic on real
measurements, but it does NOT support the sentence people will read into it
("exact-fp8 is within ~an order of magnitude of an integer ZKML system").
The denominator is not an integer prover; it is ~90–97% *our own fp8
pipeline's non-matmul gadget stages copied verbatim*.  The ratio is therefore
mostly a statement about our shared gadget floor, it shrinks as our gadgets
get slower, and its upper end (40x) is set by the largest width our 24 GB GPU
could reach, on a trend that is still rising ~∝ d^0.8.  Against a competitive
published integer prover (zkLLM-class), the honest premium is plausibly
10²–10³x, consistent with our own matmuls-only measurement of ~85–400x.

---

## 1. Findings, ranked by impact on the headline

### F1 (largest, direction: headline too favorable): the int denominator is our own gadget floor, not an integer prover

`bench_saturated.py` / `bench_final_plots.py` compute
`int_s = Σ standalone int-GEMM proofs + (rms+qnt+rope+smx+bfa+swg+seam)`
where the second term is copied from the **same fp8 zk=1 run's STAGES line**.
Measured shares: the copied non-matmul term is **92–98% of int_s** at every
config (e.g. s1024: 24.17 s of 24.72 s; b64: 126.0 of 129.6; p512: 5.18 of
5.77).  So the headline ratio ≈ `fp8_total / our_nonmm_floor`.  Consequences:

- **Circularity**: any inefficiency in our rms/softmax/rope/bfadd gadgets
  inflates the denominator and compresses the ratio toward 1.  The comparison
  can never look bad for fp8, because the int side is built from fp8's own
  parts.
- **Wrong gadget semantics for an int pipeline**: the copied stages are
  *exact-bf16/fp8* gadgets — bf16 residual adds (`bfa`), exact-bf16 RMSNorm,
  exact softmax against a bf16 reference, and the fp8 pow2-quantize gadget
  (`qnt`).  An int-native prover proves *integer* versions of these (rescale +
  range-check + table lookups), which published systems make far cheaper.
- **Calibration against zkLLM (CCS 2024, peer-reviewed)**: zkLLM proves an
  ENTIRE LLaMA-2-13B inference (40 layers, d=5120, seq 2048, all softmax /
  LayerNorm lookups included, ZK) in <15 min on one A100 → ≈22 s per layer ≈
  11 ms/token/layer.  Our "integerized layer" estimate at **d=64** (6,400x
  fewer params/layer) and 1024 tokens is 24.7 s ≈ **24 ms/token/layer** —
  more expensive per token than zkLLM's 13B layer.  Scaling zkLLM's floor
  down to toy widths, a competitive integer prover is plausibly **25–170x
  cheaper** than our int estimate at comparable shapes.
- **Corrected headline**: against a competitive integer prover the premium is
  ~10²–10³x, i.e. the same order as our own matmuls-only measurement
  (85–400x, F2), not 8–40x.

### F2 (direction: rising with width — the "40x" cap is a hardware artifact)

The whole-layer ratio grows monotonically with model width: 7.7x (d=64) →
11.8x (d=128) → 17.4x (d=256) → 41.5x (d=512).  Log-log slope ≈ **d^0.79**
(mechanism: Hawkeye commits ~12–16 columns of length P=B·K·N per matmul —
data ∝ T·d² — while the shared gadget floor scales ~T·d).  Naive
extrapolation to d=4096 gives ~**185x**, and nothing in the mechanism caps
it.  "8–40x" is the range *observable on a 24 GB card*, not a property of the
scheme; reporting it as a whole-layer constant is misleading.  The 8x end
comes from token-scaling configs at fixed toy width d=64, where the floor
(∝ tokens) tracks the matmuls (∝ tokens·d²) at fixed d.

### F3 (direction: headline too UNfavorable): the "~1000x at the matmul level" foil is stale/rounded up

`BENCH_FP8_VS_INT.md` §C claims ~1000x; even its own table implies ~80–520x,
and the current i2d binary gives (mm+lug+batch)/int_mm = **84–394x**
(measured; table §2 col iii).  With the int attention GEMMs amortized (F5)
the b64 config reaches ~760x, still short of 1000x.  Both ends of the
advertised contrast "1000x → 8–40x" were stretched: the matmul-level gap was
rounded UP and the whole-layer gap is floor-compressed DOWN, together
overstating how much "composition rescues fp8".

### F4 (direction: mixed, ±~20%): ZK asymmetry and mixed accounting

Verified in code: `p3mm::prove` (`p3_matmul.cuh`) is **integrity-only** — no
blinds, no masks, no salted/hiding Merkle trees, plain-GL transcript
challenges — while the fp8 side runs zk=1 (salted commits, Libra-style
masked sumchecks, blinded batch-open; measured premium 2.02–2.22x,
`BENCH_ZK_AT_SCALE.md` §C).  Worse, the reported int_s is *internally mixed*:
non-ZK int matmuls + zk=1 gadget stage times.  `final_plots.py` even labels
the line "Integerized (ZK)", which is false for its matmul component.
Corrections (table §2): charging the int matmuls the 2.1x ZK premium plus
composition costs gives 4.8–24x (col iv); comparing both sides
integrity-only gives 5.0–37x (col v).  The ratio moves <±25% because the mm
side dominates the numerator and the copied (already-zk1) floor dominates
the denominator — but the *labeling* is wrong either way.

### F5 (≤2% on layer ratio, ~2–3x on the matmul-level claim at batch): standalone per-proof fixed costs

*Measured* with `/root/audit_imm_phases`: the b64 int estimate sums 128
standalone per-head QK proofs at 11.5 ms + 128 PV at 12.0 ms = 3.01 s, while
the same work stacked into one proof costs 297 + 246 ms — a **5.0–6.3x
inflation** (open phase is ~75–85% of a tiny proof and nearly
shape-independent: 8.4 ms at 2^12 elements vs 46.8 ms at 2^19).  Effect on
int_s is ≤2% (floor dominates), but the amortized matmuls-only ratio at b64
moves from 238x to ~760x.  Direction: inflates int_s → flatters the fp8
headline (slightly).

### F6 (≤2%): commit-time accounting asymmetry

`p3_transformer_bench.cu` times `commit_all` separately (`commit=` field) and
the benches parse only `prove=`; `p3_matmul_bench2` includes its X/W/Y
commits inside `prove_ms` (10–35% of its total for small shapes).  *Verified
mitigation*: the fp8 side's heavyweight commitments (Hawkeye dp columns,
salted trees — ZPROF `commit_salt`, hwl `cwit`= 4–20 s per big matmul) occur
lazily *inside* the timed prove; the excluded `commit=` is only 0.13–1.4 s =
**0.1–1.5% of fp8 prove**.  Net bias ≲2%, flattering fp8.

### F7 (~1–3% time; soundness parity broken in fp8's favor): FRI query counts

The composed prover runs R=2, **Q=24** (`p3_transformer_bench.cu:114`); the
int bench defaults R=2, **Q=32**.  Both use the same Goldilocks Basefold PCS
and GL transcript challenges (no GL2 anywhere in `p3_transformer.cuh` —
field parity confirmed).  *Measured*: Q=32→24 changes int prove_ms by only
1–3% (11.5→11.2 ms; 38.2→38.0 ms), so the time bias is negligible — but the
fp8 proof is the one running at LOWER query soundness (~24 vs ~32 bits),
i.e. the comparison lets the expensive side buy fewer queries.

### F8 (missing composition costs on the int side — direction: headline premium overstated, −5…−15%)

The int estimate charges neither operand chaining/seams, nor batch-open, nor
any lookup machinery for its own requantization.  Quantified: the `seam`
stage IS included (tiny, 0.02–2.3 s).  The `lug`+`batch` stages are ~97%+
attributable to Hawkeye matmul columns (per-matmul committed data is
12–16 columns × P=B·K·N ≈ 10⁸–10⁹ elements vs ~50 non-mm columns × T·d ≈
10⁵–10⁶; hwl prof shows `ludp=ludg=ludo=0`, i.e. Hawkeye lookups are flushed
in `lug`), so a fair non-mm share is ~1–3%.  An int prover's own
requantization range-checks (≈9·T·d lookups/layer ≈ 3×10⁵ at p512) and
per-class batch-open fixed costs (~0.3 s) are small.  Charging all of this
(col iv) LOWERS the ratio ~15–40% — i.e. an honest composed integer prover
built from our parts would make the fp8 premium look *smaller*, not larger.
This is the one direction in which the reported number is conservative.

### Checks that PASSED (no finding)

- **STAGES sums to prove**: rms+qnt+mm+rope+smx+bfa+swg+lug+seam+batch =
  `prove=` within 0.03–0.2% at every logged config — no untimed gaps.
- **Same matmul inventory both sides**: 7 weight GEMMs + 2·A attention GEMMs,
  identical shapes (4×(T,d,d) + 2×(T,d,dff) + (T,dff,d) + A×(seq,dh,seq) +
  A×(seq,seq,dh)).
- **Saturated-forward denominator**: `fwd_eff` applied identically to both
  overhead lines (and cancels entirely in the fp8/int ratio).
- **Witness generation and verify excluded from both sides.**

## 2. The ratio under accounting X (all measured/derived, i2d binary, zk=1 logs)

Columns: (i) as reported; (ii) qnt dropped from int side; (iii)
matmul-machinery only = (mm+lug+batch)/int_mm; (iv) int side charged 2.1x ZK
on its GEMMs + 3% of lug+batch + 0.3 s batch-open fixed; (v) both sides
integrity-only (fp8/2.1 vs int_mm + nonmm/2.1).

| cfg | fp8 zk1 (s) | int est (s) | (i) as-rep | (ii) no-qnt | (iii) mm-only | (iv) int+zk+comp | (v) both non-ZK |
|---|---|---|---|---|---|---|---|
| s64 (seq64,d64)   | 10.0  | 1.30   | 7.7  | 9.6  | 84  | 5.5  | 7.1  |
| s128              | 18.0  | 2.80   | 6.4  | 8.1  | 121 | 5.3  | 6.1  |
| s256              | 32.2  | 4.12   | 7.8  | 9.7  | 160 | 6.4  | 7.5  |
| s512              | 62.0  | 7.40   | 8.4  | 10.7 | 202 | 7.1  | 8.1  |
| s1024             | 179.0 | 24.72  | 7.2  | 9.7  | 278 | 6.5  | 7.1  |
| b4 (seq128,d64)   | 53.5  | 8.38   | 6.4  | 7.9  | 142 | 5.6  | 6.1  |
| b16               | 164.9 | 31.60  | 5.2  | 6.5  | 136 | 4.8  | 5.0  |
| b64               | 981.8 | 129.64 | 7.6  | 9.5  | 238 (≈760 amortized) | 6.7 | 7.3 |
| p64 (d=64)        | 10.2  | 1.32   | 7.7  | 9.7  | 86  | 5.6  | 7.1  |
| p128 (d=128)      | 29.0  | 2.45   | 11.8 | 14.7 | 146 | 8.7  | 10.9 |
| p256 (d=256)      | 65.0  | 3.73   | 17.4 | 22.1 | 230 | 12.7 | 16.2 |
| p512 (d=512)      | 239.2 | 5.77   | 41.5 | 52.5 | 394 | 24.2 | 37.3 |

Width trend of (i): ∝ d^0.79, extrapolating to ~185x at d=4096.
Not in the table (unmeasurable here): replacing the copied gadget floor with
a *competitive* integer gadget floor (zkLLM-style), which per F1 plausibly
divides the denominator by another 25–170x at these shapes, putting the
"true" whole-layer premium in the 10²–10³x range.

## 3. Literature calibration

"Overhead" = prover time / native inference time on comparable hardware,
computed by us where the source gives absolute times; treat as order of
magnitude.  (pr) = peer-reviewed, (v) = vendor/self-reported.

| System | Arithmetic | Workload / hardware | Prover time | Overhead vs native | ZK (hiding)? | Source |
|---|---|---|---|---|---|---|
| zkLLM, CCS'24 (pr) | integer/fixed-point, scale 2^16, BLS12-381 + sumcheck/tlookup | LLaMA-2 13B, seq 2048, 1×A100 40GB | <15 min / inference (≈22 s/layer, ~11 ms/tok/layer) | ~2×10³ (native ≈0.5 s est.) | yes (model params) | [arXiv:2404.16109](https://arxiv.org/pdf/2404.16109), [ACM](https://dl.acm.org/doi/10.1145/3658644.3670334) |
| DeepProve-1, Lagrange (v) | quantized (GKR + logup) | GPT-2 & Gemma-3, hw unstated | 174 / 86 tok/min (0.34–0.7 s/tok) | ~10²(est., hw unstated) | not claimed | [lagrange.dev](https://lagrange.dev/blog/deepprove-1), [eprint 2026/1112](https://eprint.iacr.org/2026/1112) |
| DeepProve vs EZKL (v) | quantized | CNN 264k / MLP 4M | — | 54–158x faster than EZKL | — | [lagrange.dev](https://lagrange.dev/blog/announcing-deepprove-zkml) |
| zkPyTorch / Expander (v) | quantized int | Llama-3 8B, 1 CPU core | 150 s/token | ~10²–10³ vs CPU-native | claimed | [eprint 2025/535](https://eprint.iacr.org/2025/535), [blog](https://blog.polyhedra.network/zkpytorch/) |
| EZKL (halo2) (v/pr-adjacent) | quantized, lookup nonlinearities | MNIST-class | 600–700 s | ~10⁵–10⁶ | yes | [EZKL benchmarks](https://blog.ezkl.xyz/post/benchmarks/), [NANOZK](https://arxiv.org/html/2603.18046) |
| Modulus "Cost of Intelligence" (v) | quantized, zkCNN/GKR best | small MLP (8 ms native) | 0.6 s | ~10³; later GKR ~180x | varies | [Medium ch.5](https://medium.com/@ModulusLabs/chapter-5-the-cost-of-intelligence-da26dbf93307), [ch.13](https://medium.com/@ModulusLabs/chapter-13-scaling-intelligence-637d4a374153) |
| Garg et al. (pr) | **exact IEEE-754 fp32** circuits | amortized batch of 2^15 muls | 64 constraints/fp32-mul | ~10²x per-op vs int mul | yes | [PDF](https://par.nsf.gov/servlets/purl/10408517) |
| ZIP, CCS'25 (pr) | **exact IEEE-754 fp64** inference | MNIST/UTKFace/SST-2 (small) | n/a (circuits 10³x smaller than bit-decomp baselines) | n/a | yes (CP-SNARK) | [eprint 2025/1732](https://eprint.iacr.org/2025/1732) |
| **ours, fp8 Hawkeye zk1** | **exact fp8 tensor-core semantics** | 1 toy layer (d≤512), RTX 4090 | 10–982 s/layer | ~3×10⁴–4.5×10⁵ vs saturated fwd | yes | this repo |
| **ours, "integerized" estimate** | int GEMM + copied fp8 gadget floor | same | 1.3–130 s/layer | ~2×10³–6×10⁴ | mixed (F4) | this repo |

Two calibration takeaways: (a) published *integer* transformer provers sit at
~10²–10³x native; our int-side ESTIMATE sits at ~10³–10⁴x+ (per-token, toy
widths) — i.e. our floor is 1–2 orders above a competitive integer system,
which is exactly the inflation that compresses the headline ratio.  (b) the
only published exact-float ZK line of work (Garg et al. fp32 circuits,
~64–100x per multiplication over integer; ZIP at fp64 for small models)
lands in the same ~10²x per-op premium band as our measured matmuls-only
84–394x — our matmul-level result is credible and roughly
literature-consistent; no published system proves exact fp8 tensor-core
accumulation for a transformer layer, so there is no direct external
comparator for the composed number.

## 4. Honest restatement

What our data actually supports: proving one toy transformer layer with
bit-exact fp8 tensor-core semantics (zk=1, Goldilocks Hawkeye, RTX 4090)
costs 8–40x more than *the same layer with the 7+2A matmuls swapped to plain
integer sumcheck GEMMs while keeping our own fp8 pipeline's non-matmul
gadgets, lookup flush and batch-open uncharged* — a denominator that is
92–98% our own gadget floor, mixes non-ZK matmuls with zk=1 gadget stages,
and whose ratio rises ~∝ d^0.8 with width (41x is merely the largest width
that fits in 24 GB; ~185x extrapolated at d=4096).  On the matmul machinery
alone the exactness premium is 84–394x (current binary; the previously
advertised "~1000x" is not reproducible and should be retired), consistent
with published exact-float circuit costs (~10²x per multiplication, Garg et
al.).  Published integer/quantized transformer provers (zkLLM: LLaMA-2 13B,
seq 2048, <15 min on one A100 ≈ 2×10³x native) have per-token layer costs
1–2 orders of magnitude below our int estimate at comparable token counts,
so against a *competitive* integer baseline the honest whole-layer fp8
premium is plausibly 10²–10³x, not 8–40x.  The defensible headline is:
"exact-fp8 matmul machinery costs ~10²x over integer matmul proving; in our
current implementation the whole-layer ratio compresses to 8–40x only
because our shared non-matmul gadget floor is expensive and width is capped
at d=512; the ratio is width-dependent and the comparison is not against an
optimized integer system."

## Appendix: audit artifacts

- `audit_imm_phases.cu` / `/root/audit_imm_phases` — phase-split int GEMM
  bench (commit/sumcheck/open, Q knob).  Key rows: (128,32,128) Q32 =
  11.5 ms (commit 1.6, open 9.9); (16384,32,128) = 297 ms; Q24 vs Q32 ≤3%.
- `audit_ratios.py` — the §2 table generator.
- `/root/audit_s64_dbg.log` — s64 composed run with P3_SZDBG+P3_ZKPROF
  (batch-open class inventory, ZPROF stage breakdown, STAGES-sums check).
