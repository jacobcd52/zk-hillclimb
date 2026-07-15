# INT_LAYER — the measured integer-layer baseline on the Goldilocks/hash substrate

2026-07-15, RTX 4090.  This is the composed single-proof ZK prover for ONE
integerized transformer layer's forward pass that COMPARISON_AUDIT.md said was
missing: the previous "integer layer" line was an estimate assembled from our
own fp8 gadget floor (circular, F1).  This one is a real prover: every op is
an integer gadget, the whole layer is one proof, and the ZK teeth passed.

## 1. What was built

New sources (no fp8 source touched):

| file | contents |
|---|---|
| `p3_int_gadgets.cuh` | integer gadget set (namespaces p3ig/p3irs/p3imm/p3iadd/p3irms/p3irope/p3ismx/p3iswg) |
| `p3_int_layer.cuh` | composed layer (namespace p3itf): witness chain, commit_all, prove, verify |
| `int_layer_ref.py` | NORMATIVE integer reference (numpy int64); bitwise parity harness |
| `p3_int_selftest.cu` | per-gadget battery (60 checks) |
| `p3_int_layer_test.cu` | composed battery (34 checks) + trace dump |
| `p3_int_zk_test.cu` | hiding battery (18 checks, fp8-battery methodology) |
| `p3_int_layer_bench.cu` | bench binary, args mirror p3_transformer_bench |
| `run_int_sweep.py` | zk=1 sweep driver → /root/zk_int_layer_results.json |

**Semantics** (fixed-point llama block, residual scale 2^16, zkob-style; the
python reference is normative and the C++ witness replay matches it bitwise):
rmsnorm (Σx²+eps, table inverse-sqrt on a normalized mantissa, W=R⊗g,
y=rescale(W∘x)) → Wq/Wk/Wv int matmul + rescale (round-half-up ÷2^16, range
checked to ±2^19) → int RoPE (public int cos/sin at 2^14, rotate-half by the
flipped-point MLE claim) → per-(batch,head) QK^T → scores rescale (pow2 temp
2^ceil(ldh/2) folded in, scale 2^8) → int softmax (rowmax by SEL-attainment +
dominance lookup, exp by a 2^17-row table, P = round_half_up(2^16·E/S) by the
zkob bracket) → P·V → rescale → Wo → residual → rmsnorm → Wg/Wu → SwiGLU
(2-column (x, silu) mapping lookup, 2^20 rows) → Wd → residual → out.

**Proof architecture.** One Fiat–Shamir transcript, one shared p3lu::XCtx
opening ledger, ONE merged lookup flush, ONE batched-opening pass — the
p3_transformer composition pattern.  Every op is committed columns +
eq-weighted sc5z zero-checks (Libra blinds in zk) + deferred logUp lookups +
ledger claims.  The new piece is the **integer matmul** (p3imm): instead of
Hawkeye's committed product-domain columns, one hiding claim on the committed
accumulator at (z‖zex) reduced by ONE cubic sumcheck Σ EQ·Xb·Wb over the
(k,j,i,ex) product domain, where Xb/Wb are virtual broadcasts of the operand
commitments (index maps only; terminal claims land on the operands at points
with random ex-coordinates → hiding).  In zk the accumulator's mask slice 1
is LINKED = matmul of the operands' mask slice-1s, so the claim algebra holds
slice-by-slice — the p3_zkc mechanism-1 seam rule.  Committed data per matmul
is just the accumulator column (B·N), not B·K·N.

**Chaining.** All seams are shared commitments (root equality) plus
partial-point claims: head slices, the K/V "transposes" and the attention
concat are index maps inside the matmul operand views (exact MLE identities
at partially fixed points on the producer commitments).  Compared to the fp8
layer this removes the per-head rope/quantize instances, the V^T commitment +
transpose seam, the concat seam and all k-padding seams (contractions are
pow2 here).  Public statement: dims, input (values + root), 7 weight + 2 gain
roots, the public rope tables, table ids, Q/R, output values; the committed
input and final output are bound to the public arrays by real-slice claims.

## 2. Test results

- **Gadget battery** (`/root/p3_int_selftest`): **60/60**, non-zk AND zk.
  Per gadget: honest accept + adversarial teeth (rescale: rem-out-of-table /
  limb-flip / out-of-window; matmul: forged Y, operand mismatch, tamper inside
  a shared-grid instance block, slice+transpose views; rmsnorm: inflated M
  (row-sum), R±1 (limb lookup), forged W/Y (zero-check); rope: unrotated
  operand, y-shift; softmax: mx+1 (attainment), forged exp (table), P+1
  (bracket limbs), inflated S (row-sum); swiglu: forged silu (table),
  forged M; residual: forge + range).
- **Composed battery** (`/root/p3_int_layer_test`): **34/34**, non-zk AND zk:
  honest accepts; every chain tamper (rms out, Wq acc, scores in the LAST
  attention instance, P, PV acc, swiglu out, final residual) rejects at its
  owning gadget; every per-gadget witness forgery rejects in the composed
  context; tampered public input, tampered public output and a wrong weight
  root each reject.
- **Reference parity** (`int_layer_ref.py`): **REF_OK** — all 30 dumped
  intermediates (per-row M/R/mx/S included) match the independent python
  int64 chain BITWISE (nonlinearity tables shared via `int_tables.bin`).
- **Hiding battery** (`/root/p3_int_zk_test`): **18/18**, 12 000 draws at
  fixed challenges on the real int column set, fp8-battery methodology:
  (1) all 14 column classes' opened evals / blind evals / every sumcheck
  message / FRI finals / codeword values uniform (χ² 232–309 vs threshold
  400); (2) finite-difference message coefficients uniform when
  degree-matched-blinded, negative control spikes to 3.07·10⁶; (3) the matmul
  mask linkage and the row-sum linkage: claims uniform AND both sides agree
  exactly; (4) batch blinder one-time-pad; (5) witness-recovery attack:
  control collapses to 1 distinct message value, hidden transcript
  12000/12000 distinct + posterior flat across witnesses; (6) HVZK simulator
  produces the same law and verifies witnesslessly; (7) GKR lookup mask
  siblings uniform, with masks-off teeth.  **The zk=1 numbers below are
  claimed as ZK on the strength of these teeth.**

## 3. Measured table (zk=1, R=2, Q=24, RTX 4090; the fp8 grid)

fp8 columns from COMPARISON_AUDIT.md §2 (same machine/params); "old int est"
is the audit's circular estimate the plot used until now.

All 12 grid points ran to completion, `verify_ok=1` everywhere, peak RSS
4.97 GB (cap 41 GB).  Raw rows (incl. per-stage breakdowns) in
`/root/zk_int_layer_results.json` and `/root/int_sweep.log`.

| cfg | fp8 zk1 (s) | old int est (s) | **int zk1 measured (s)** | verify (s) | proof MB | RSS GB | fp8/int **measured** | fp8/est (old plot) |
|---|---|---|---|---|---|---|---|---|
| s64  (seq64,d64)  | 10.0  | 1.30   | **2.70**  | 0.19 | 6.1  | 0.55 | **3.7**  | 7.7  |
| s128              | 18.0  | 2.80   | **4.25**  | 0.17 | 6.9  | 0.65 | **4.2**  | 6.4  |
| s256              | 32.2  | 4.12   | **4.34**  | 0.18 | 6.9  | 0.72 | **7.4**  | 7.8  |
| s512              | 62.0  | 7.40   | **6.98**  | 0.24 | 7.5  | 0.87 | **8.9**  | 8.4  |
| s1024             | 179.0 | 24.72  | **15.73** | 0.37 | 7.7  | 2.48 | **11.4** | 7.2  |
| b4  (seq128,d64)  | 53.5  | 8.38   | **5.53**  | 0.19 | 7.3  | 0.64 | **9.7**  | 6.4  |
| b16               | 164.9 | 31.60  | **18.27** | 0.27 | 9.6  | 1.52 | **9.0**  | 5.2  |
| b64               | 981.8 | 129.64 | **66.28** | 0.59 | 16.5 | 4.97 | **14.8** | 7.6  |
| p64  (d=64)       | 10.2  | 1.32   | **2.59**  | 0.20 | 6.1  | 0.56 | **3.9**  | 7.7  |
| p128 (d=128)      | 29.0  | 2.45   | **4.55**  | 0.25 | 6.7  | 0.72 | **6.4**  | 11.8 |
| p256 (d=256)      | 65.0  | 3.73   | **5.59**  | 0.20 | 6.1  | 1.14 | **11.6** | 17.4 |
| p512 (d=512)      | 239.2 | 5.77   | **16.20** | 0.19 | 7.1  | 3.70 | **14.8** | 14.8→41.5 est |

Anchor: **s128 d=64: int layer proves in 4.25 s** (fp8: 18.0 s).
ZK premium of the int layer at the anchor: 4.25 / 1.21 (zk=0) = **3.5x**
(fp8's measured premium was 2.02–2.22x; ours is higher because per-instance
fixed blind/flush costs are a larger share of a much cheaper proof).

Stage shares (zk=1): at s1024 the matmul stage is 8.3 s of 15.7 (53%), with
rescale 1.3, softmax 2.7, lookup flush 0.8, batch openings 2.0; at p512 the
matmuls are 13.8 of 16.2 (85%); at b64 46.1 of 66.3 (70%) — i.e. unlike the
audit's estimate (92–98% copied gadget floor), the measured integer prover is
matmul-dominated, as an integer prover should be.

### What the measurement says about the old headline

- **The measured whole-layer fp8 premium is 3.7–14.8x on this grid**, rising
  with width (p64→p512: 3.9→14.8 ≈ (d)^0.64) and with tokens.  This lands
  inside the estimate's bracket, but the ESTIMATE itself was wrong in both
  directions, exactly as the audit predicted: at token-heavy configs its
  copied fp8 gadget floor overstated the int cost (s1024: est 24.7 vs
  measured 15.7; b64: est 129.6 vs 66.3 — F1, circularity), while at wide
  configs its uncharged matmul/composition/ZK costs understated it (p512:
  est 5.77 vs measured 16.2 — F4/F8).  The measured ratio column is the line
  the overhead plot should carry.
- **Same-substrate statement only.** This is our fp8 prover vs our int prover
  on the SAME Goldilocks/Basefold substrate, same R/Q, both fully composed
  and both ZK (teeth passed on both).  It does NOT bound the gap to a
  competitive published integer system: at s1024 the int layer costs
  15.4 ms/token/layer at d=64, while zkLLM (CCS'24) reports ≈11 ms/token/layer
  at d=5120 — roughly 2 orders of magnitude more work per parameter here, so
  the audit's "10²–10³x vs a competitive integer prover" calibration for fp8
  stands unchanged.  What this measurement removes is the circular
  denominator, not the competitive-frontier caveat.

## 4. Honest caveats

- **Faithfulness scope.** This proves a SELF-CONSISTENT integer forward pass
  (prover commits input, weights, every intermediate; verifier checks every
  op + all seams + public IO).  It is not bitwise-faithful to any external
  model.  Specific simplifications vs zkob, all documented in the sources:
  (a) rmsnorm inverse-sqrt is a 2^16-row table on a normalized mantissa
  (M = m·4^e bracketing) instead of zkob's (R±1)²·M bracket — the 2^80-bit
  bracket integers do not fit Goldilocks; precision class is the same
  (±1–2 ulp on R), and the proof is bitwise for the reference that computes
  exactly this.  (b) softmax temperature is pow2 (2^ceil(ldh/2) ≈ √dh) folded
  into the scores rescale shift.  (c) activation/weight magnitudes are bound
  to ±2^19 (±8.0 real at scale 2^16) by range lookups; the bench generates
  in-range data and witness generation refuses out-of-range traces (honest
  saturation semantics are NOT modeled).
- **Bounded advice, no slack.** Every advice column is lookup-bounded before
  entering a product, and all magnitudes stay ≪ p ≈ 2^64, so the in-field
  identities are integer identities.  Softmax P needs no range lookup: the
  r1+r2 = 2S−1 limb identity pins r1 < 2S exactly and P is then the unique
  in-field solution of the bracket (see p3ismx header).
- **Soundness/field parity with the fp8 side.** Same Goldilocks base-field
  transcript challenges, same Basefold/hash PCS, same R=2, Q=24 as the fp8
  zk sweep (~24-bit query soundness; the repo's stated GL2
  challenge-width upgrade applies to BOTH provers equally — no GL2 was needed
  here beyond what the fp8 side uses, and using it would slow both sides
  comparably).  Schwartz–Zippel events ride the same ~2^-64-per-challenge
  budget as the fp8 layer's own seam/zero-check claims.
- **Known cost structure.** The prover floor is dominated by per-instance
  fixed costs (blind commitments + lookup flush + batched opening), exactly
  the class the audit's F5 measured; at tiny configs the matmul stage is
  mostly these fixed costs, not the O(B·K·N) sumcheck work.  The witness
  build (untimed in `prove=`, printed separately, same convention as the fp8
  bench) is a plain integer forward pass plus limb bookkeeping.
- **What it would cost to close the gaps.** Bitwise zkob-rmsnorm would need
  a 2-limb bracket product argument (~4 extra 16-bit lookups + one more
  zero-check per row — cheap, per-row only).  True saturation semantics would
  add a comparison gadget per rescale (one extra selector column + lookup per
  element).  Neither changes the cost picture materially.

## 5. Reproduce

```
cd /root/zkllm
nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp -I. p3_int_selftest.cu    -o /root/p3_int_selftest    && /root/p3_int_selftest
nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp -I. p3_int_layer_test.cu  -o /root/p3_int_layer_test  && /root/p3_int_layer_test
/root/p3_int_layer_test dump && python3 int_layer_ref.py
nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp -I. p3_int_zk_test.cu     -o /root/p3_int_zk_test     && /root/p3_int_zk_test
nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp -I. p3_int_layer_bench.cu -o /root/p3_int_layer_bench
python3 run_int_sweep.py       # → /root/zk_int_layer_results.json
```
