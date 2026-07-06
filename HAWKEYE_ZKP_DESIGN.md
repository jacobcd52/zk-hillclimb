# HAWKEYE_ZKP_DESIGN — ZKP of the exact Hawkeye fp8 forward pass (single FC layer)

Fable, 2026-07-02. Status: increment 1 BUILT AND PASSING (see §7). Substrate: p3 hash stack in
/root/zkllm (Goldilocks p = 2^64−2^32+1, Basefold/FRI PCS, SHA-256 Merkle, fs_transcript).
Everything marked [VERIFIED] was run on this machine today; [REASONED] is arithmetic/estimation.

## 0. Statement and trust chain

We prove: committed fp8 operands `X (B×K)`, `W (N×K)` (E4M3 codes) and committed per-row fp32
scales produce output `Y (B×N)` **bit-identical to `hawkeye_fp8_sum`** (products_per_group G=32,
internal_width IW=14, zero_exponent −139) — i.e. to Hopper QGMMA accumulation, which Hawkeye
replicates (DiFR=0 on H100, trusted per coordinator decision).

Ground truth chain [VERIFIED today]:
`hawkeye_ref` (numpy, /root/zkllm/hawkeye_ref.py) == Triton `hawkeye_fp8_sum` on the 4090,
**bf16-bitwise**, on 12 configs: random layers (incl. K%32∈{0,1,8,31}, masked M/N tails,
8-group acc chains), tile-size invariance, non-default G=8 and IW=12, NaN codes 0x7F/0xFF
(decoded as ordinary values — the kernel is NOT IEEE here and we replicate that), zero/negative
scales, and an exhaustive 256×256 single-product grid. 0 mismatches anywhere.
`hawkeye_ref` is therefore the golden witness generator for every gadget below.

## 1. The computation, restated as constraint-friendly semantics

Per output (m,n), per K-group of G=32 products, with accumulator gfloat (sign, exp, sig):

| step | semantics | lossy? |
|---|---|---|
| decode | code → (exp_eff∈[1,15], signed_sig∈[−15,15], nonzero) | no |
| product | prod_exp = a_exp+b_exp−14; signed_prod = a_sig·b_sig ∈[−225,225]; scaled = signed_prod≪7 (|scaled|≤28800<2^15) | no |
| group max | max_exp = max(acc_exp_eff, prod_exp over present products) | no |
| **align** | aligned = trunc_toward_zero(scaled ≫ (max_exp−prod_exp)), shift clamped [0,62] | **YES (truncating)** |
| group sum | contribution = Σ present·aligned — exact integer, |·|<2^20 | no |
| acc realign | acc_base=acc_sig≫10; ≫(max_exp−acc_exp_eff); signed | **YES (truncating)** |
| normalize | width = bitlen(|total|)≤21; out_exp=max_exp+width−14; out_sig = |total| shifted to 14 bits; subnormal path (unreachable at these params [REASONED: out_exp ≥ −25]); out_sig≪10 | **YES (truncating)** |
| output | gfloat→fp32 bits (integer bit assembly) ; ×x_scale ×w_scale (two fp32 RNE muls) ; bf16 RNE store | fp32/bf16 RNE |

All loss is truncation toward zero — **simpler than RNE: quotient/remainder only, no
guard/round/sticky bits.** The per-product alignment is the irreducible O(B·N·K) cost.

## 2. Gadget decomposition

Notation: P = B·N·K products, Gc = P/32 groups, O = B·N outputs. All witness columns
Basefold-committed; all lookups via **p3_logup.cuh** (new, §7): logUp with helper-inverse
columns, batched cubic sumchecks, verifier-evaluated public tables (no table commitment).

**P1 — fused decode·multiply·scale lookup (DM).** [BUILT]
`(a, b, eb, mag, sg, pr) ∈ DM`, 2^16 rows keyed by the two 8-bit codes:
eb = prod_exp+12 ∈[0,28], mag = |a_sig·b_sig|≪7, sg = product sign, pr = present.
One lookup **removes ALL per-product decode and multiply arithmetic** (incl. the NaN-code and
subnormal-operand decode branches) — exactly the efficiency idea mandated; confirmed working.

**P2 — truncating alignment (the irreducible per-product cost).** [BUILT]
Witness q, r, pw with
`C1: q·pw + r = mag` (field equation, no wrap: <2^30 ≪ p),
`(sh, pw) ∈ SHIFT` (64 rows, pw = 2^min(sh,15); valid because mag<2^15 ⇒ mag≫sh = mag≫min(sh,15)),
`(pw, r) ∈ REM` (65536 rows, enforces 0 ≤ r < pw),
`q ∈ RANGE15` (32768 rows) ⇒ q,r nonneg and bounded ⇒ q = ⌊mag/pw⌋ = the exact truncated shift.
`C2: al = pr·(1−2sg)·q` — the signed masked contribution Hawkeye adds into the group sum.
C1+C2 enforced by ONE eq-weighted batched zero-sumcheck (degree 4, public claim 0).
Compare RNE (CanonFC plan §4): RNE needed q/r **plus** round-bit, sticky ≠0 test and half-way
tie logic (~+3 columns, +2 lookups per instance). Truncation is strictly cheaper per instance.

**P3 — group max_exp.** [increment 2]
Per group: committed max_exp (1/group). Constraints: (i) dominance — for present products
sh_k = max_exp − prod_exp_k, i.e. `pr·(sh + eb − 12 − (max_exp − 12)) = 0` linear binding of the
ALREADY range-checked sh (SHIFT domain gives sh≥0 ⇒ max_exp ≥ prod_exp free of charge); for the
acc: one more shift-witness column (P2 instance for acc realign, below) plays the same role;
(ii) attainment — one selector bit/product (+1 for acc), Σ sel = 1 per group, sel·sh = 0
(the max is achieved by a shift-0 element). Absent products (pr=0) leave sh unconstrained-in-domain
(their al is forced 0 by C2, matching kernel semantics where masked lanes are irrelevant).

**P4 — accumulator realign.** [increment 2] Per group: acc_base = acc_sig≫10 (constant shift =
lookup-free linear relation on a witness split acc_sig = acc_base·2^10 + acc_lo, acc_lo∈[0,2^10) —
1 range lookup), then one P2-style truncating shift by (max_exp−acc_exp_eff), signed. ≈4 elts +
3 lookup rows/group.

**P5 — normalize.** [increment 2] Per group: width via pow2 sandwich `2^{width−1} ≤ |total| < 2^width`
(pow2 lookup + 2 range rows); truncate-to-14-bits = one more q/r pair; out_exp linear; subnormal
selector (statically unreachable at G=32/IW=14 but constrained anyway, 1 bit); sign bit.
≈8 elts + ~5 lookup rows/group. Group sum itself (contribution = Σ_{group} al) is a cheap linear
sumcheck relation binding P2's al columns to P4/P5's total.

**P6 — output binding.** [increment 2] Per output: gfloat→fp32 bits is integer bit assembly
(linear + 2 selector bits: leading-bit-missing, clamp); two fp32 RNE multiplies by the committed
scales (24×24→48-bit product + RNE, the ONE place RNE gadgets are needed — O instances only);
bf16 RNE downcast (1 q/r + round bit). ≈12 elts + ~6 lookup rows/output. Y binding as in p3pfc.

## 3. Per-gadget cost table

Committed field elements (the cost driver: prove time ≈ Merkle+NTT ≈ linear in committed data
[VERIFIED scaling in FABLE_REPORT §6]); lookup rows = logUp helper hA (1 elt/row committed in v1,
~0 with GKR-logup); sumcheck rounds are verifier-trivial.

| gadget | committed elts | lookup rows | sumcheck rounds |
|---|---|---|---|
| P1 DM | per product: eb,mag,sg,pr = 4 (a,b broadcast copies: 2 in the current build, **0 optimized** — a(m,n,k)=X(m,k) is constant in n, so its MLE eval collapses to an X opening at a collapsed point: "virtual broadcast") | 1/product + fixed 2·2^16 (cnt,hT) | log P + 16 |
| P2 align | sh,pw,q,r = 4; al = 1 (**0 optimized**: the group-sum sumcheck can consume pr·(1−2sg)·q directly, degree 3) | 3/product + fixed (2^16+2^15+2^6 tables ×2) | log P ×3 lookups + (16+15+6); constraint: log P (deg-4) |
| P3 max_exp | 1/group + 1 bit/product (selector) | — | log P (batched into the P2 zero-check) |
| P4 realign | ≈4/group | 3/group | log Gc |
| P5 normalize | ≈8/group | 5/group | log Gc |
| P6 output | ≈12/output | 6/output | log O |

Totals per PRODUCT (groups/outputs amortized: /32 and /K):
- **current-build accounting (v1 logUp, committed broadcasts, committed al): ≈ 15.5**
  (11 columns + 4 hA + 0.5 group/output amortized)
- **straightforward opts (virtual broadcast a,b; fused al): ≈ 12.5**
- **aggressive (GKR-logup: hA/hT/cnt commit-free; bit-pack sg,pr,eb,sh into 1 column with
  unpack lookups; pack q,r 2-per-element): ≈ 5–6**

## 4. Total overhead — tiny layer and llama-68m up_proj

Baseline integer FC proof (FABLE_REPORT §6 [VERIFIED]): up_proj (768→3072, B=16) commits
2·(B·K+K·N+B·N) ≈ 4.84 M elts; prove 303 ms, verify 34 ms, 2.47 MB.

**Tiny layer [VERIFIED today]** — golden vectors = 3776 products (three random layers + directed
edge battery), padded 4096; increment-1 scope (P1+P2, sh as committed input):
commit 140 ms, prove 283 ms, verify 81 ms, 15/15 selftest checks. At this size the FIXED
table-side work (2^16 DM + 2^16 REM + 2^15 RANGE helper columns ≈ 0.4 M elts) dominates the
0.06 M witness elts — it amortizes away at layer scale.

**llama-68m up_proj (B=16, K=768, N=3072): P = 37.7 M products, Gc = 1.18 M, O = 49 K** [REASONED]

| variant | committed elts | vs integer proof (4.84 M) | est. prove time* |
|---|---|---|---|
| current build (v1) | ≈ 585 M | **×121** | ~35–40 s |
| + virtual broadcast, fused al | ≈ 470 M | ×97 | ~30 s |
| + GKR-logup + packing | ≈ 200–230 M | **×41–48** | ~13–15 s |
| (CanonFC K0=32, for reference, from FP8_EXACT_PLAN) | 13–20 M | ×2.7–4.1 | ~1 s |

*prove scales ≈ linearly with committed data at ~16 M elts/s measured on the integer proof
(Merkle+NTT bound); same-order host-sumcheck work needs the GPU port of the logUp/constraint
sumcheck loops (mechanical — kernels exist for the degree-2 case in p3_basefold.cuh).

**The honest headline: exact-Hawkeye is a per-product-witness proof — ×40–120 the integer proof,
~13–40 s/layer at B=16 vs 0.3 s.** This is the price of bit-exactness to lossy hardware
accumulation order; it is linear in B·N·K, so B=1 single-token attestation is ~2.4 M products →
~1–3 s/layer. Also note prover MEMORY: 200–585 M elts ≈ 1.6–4.7 GB witness (+4× codewords) —
needs column-streamed commits or N-sliced proofs (both natural: columns/products are independent
until the group chain; slices compose by sharing the X commitment).

Proof size [REASONED]: dominated by Basefold openings (~30 single-point openings in the current
composition). At 2^26 rows, ~1 MB each → ~30 MB. Same-point RLC batching (7 constraint openings →
1; per-lookup W-openings → 1) cuts this to ~5–6 MB; acceptable, on the roadmap.

## 5. Efficiency levers — evaluated

1. **Fused (a,b)→(prod_exp,scaled) lookup** [BUILT, mandated]: removes all per-product decode
   and multiply arithmetic; the 2^16×6 table costs the verifier one O(2^16·6) host eval per
   proof (~ms) and the prover 2·2^16 fixed helper elts. Confirmed by the DM-forgery negative
   control (mag+1 with C1 kept consistent is caught ONLY by the lookup).
2. **Truncating q/r alignment** [BUILT]: 4 committed elts + 3 lookup rows/product; strictly
   simpler than RNE (no G/R/S). The REM-forgery control confirms the range side is what
   enforces correct truncation (q−1, r+pw satisfies C1 and every other relation; REM kills it).
3. **Within-group exact sum** [increment 2]: plain linear sumcheck, no rounding — free.
4. **max_exp via comparison witnesses** [increment 2]: sh≥0 comes FREE from the SHIFT table
   domain; only attainment selectors are new data (1 bit/product).
5. **Virtual broadcast columns**: a,b (and X,W-indexed anything) need no per-product commitment
   — MLE of a broadcast is the base matrix MLE at a collapsed point. Saves 2 elts/product.
6. **GKR-logup**: replaces committed hA/hT/cnt with a GKR fractional-sum tree — the single
   biggest lever (−4 elts/product + removes the 2^16-scale fixed columns).
7. **Bit-packing** with unpack lookups: sg,pr (1b), eb (5b), sh (6b) → one column; q,r (15b) →
   pack 2/element. ~−4 elts/product for +2 lookup rows.
8. **GL2 challenges** (p3_gl2.cuh): mechanical soundness upgrade 2^-58 → 2^-116, ~2× sumcheck
   arithmetic, no committed-data change. Do before any external claim.

## 6. Binius note (scaling path, NOT this build)

The per-product witness is dominated by small integers (15-bit mag/q/r, 6-bit sh, bits sg/pr)
each occupying a 64-bit Goldilocks slot, and by lookups whose only job is "these 64-bit slots
hold small/bounded values". Over binary towers (Binius) the same witness commits at its true
bit-width (~45–50 bits/product vs 3–4 Goldilocks elts ≈ 200+ bits after helpers), and
shift-by-2^k / bit-decomposition are LINEAR maps over GF(2) — the REM/RANGE lookups largely
disappear. Expected ~4–8× committed-data reduction on P2, putting exact-Hawkeye at ~×10–20 of
the integer proof. But: no binary-tower substrate exists in this repo, and the p3 Goldilocks
stack is validated (Basefold, sumcheck, now logUp) — **recommendation: Goldilocks + limb/range
lookups for this build; Binius as the scaling path if exact-Hawkeye must reach big-model scale.**

## 7. Increment 1 — what is built and passing [VERIFIED]

New files in /root/zkllm (binaries under /root; /workspace untouched):

- `hawkeye_ref.py` — numpy golden generator. `python3 hawkeye_ref.py --dump hawkeye_prod_vectors.bin`
  → 12/12 bitwise configs vs Triton, wrote 3776 witness rows (coverage: shift0=112,
  shift≥15=1674, negative-al=967, absent=244, nan-code=65).
- `p3_logup.cuh` + `p3_logup_test.cu` — p3-native logUp (§2 preamble). 13/13:
  honest range + tuple(c=3) lookups, out-of-table witness, tuple row-mismatch, wrong
  multiplicities, tampered S / sumcheck msgs / opened value / codeword, Q=0 params forgery,
  and a **sum-preserving forged-hA pair** (caught by the eq-weighted zero-check — the control
  that distinguishes logUp from a bare sum argument).
- `p3_hawkeye_prod.cuh` + `p3_hawkeye_prod_test.cu` — P1+P2 gadget. 15/15 incl. the four
  isolated semantic forgeries (§5.1, §5.2, SHIFT pw=2^(sh+1) with q/2 — everything consistent
  except the sh→pw binding — and C2 sign flip / absent-product-nonzero-al).

Commands: `nvcc -arch=sm_89 -std=c++17 -O2 p3_logup_test.cu -o /root/p3_logup_test` (same for
p3_hawkeye_prod_test); run from /root/zkllm (vectors file). Both ALL PASS, GPU-Merkle path on.

## 8. Increment 2 (next) and decisions for the coordinator

Build order: (a) group gadget: max_exp (P3) + realign (P4) + normalize (P5) + group-sum binding
of al; (b) accumulator chain across K/32 groups (sequential composition over committed per-group
state columns); (c) output binding P6 (fp32 RNE muls + bf16); (d) compose and validate a FULL
tiny layer bitwise against `hawkeye_ref`/Triton end-to-end; (e) GPU-port the logUp/constraint
sumcheck host loops; (f) same-point opening batching.

Decisions needed:
1. **Accept the ×40–120 overhead class** (13–40 s/layer at B=16; ~1–3 s at B=1) as the cost of
   the exact-Hawkeye mandate, or bound the demo to B=1 / one layer? (CanonFC remains the cheap
   alternative if the mandate is revisited.)
2. ZK now or after composition? (Mask-slice augmentation of every column à la p3pfc roughly
   doubles committed data — I propose non-ZK until the full layer composes, then mask.)
3. GL2 challenges before or after increment 2? (Mechanical; I propose with increment 2.)
4. Weight-commitment format: commit W as raw E4M3 codes (my default: the DM lookup consumes
   codes directly, and the published commitment is then code-level canonical) — confirm.

## 9. Speed pass (2026-07-02) — summary [reconstructed]

NOTE: the detailed §9 written during the speed-pass session did not persist to this file; this is
a faithful summary from the session record. All numbers were [VERIFIED] on 2026-07-02.

up_proj B=1 (`p3_hawkeye_bench 1 768 3072`): 61.07 → **6.22 s prove**, 0.94 → 0.16 s verify,
65.7 → **7.18 MB proof**, RSS 5.5 → 2.8 GB. Tiny (2,64,64): 0.44/0.12 s, 4.8 MB. Levers landed
(cumulative): L1 batched openings (`p3_batchopen.cuh`: claims ledger, per-size-class multipoint
reduction + ρ-RLC Basefold opening, round-0 queries authenticated per column against original
roots) 61→26 s; L2 Montgomery batch inversion 26→14.3 s; L3 GPU sumcheck port (`p3_scgpu.cuh`,
byte-identical messages to the host loops) 14.3→6.2 s. Remaining profile at 6.2 s: lu_dp 2.25 s,
batch 1.32 s, witness commits 1.27 s, zc_dp 0.51 s. Next levers [REASONED]: GKR-logup (−1.5–2 s),
bit-packing (−1.5 s), device-resident witness columns (−0.5 s), pow2-slice padding recovery
(3.6× padding at B=1 is the biggest single win), GL2 challenges, mask-slice ZK.

## 10. Zero-knowledge pass (2026-07-02) — summary [reconstructed]

NOTE: as with §9, the session's detailed section did not persist here; summary from the record.

Closes the F2 weight-recovery leak (non-ZK opening's last sumcheck round emits s0 = c0·w0, a known
linear functional of the column → ~|W| openings recover W by Gaussian elimination). Fix = three
mechanisms, all [VERIFIED] 2026-07-02 in `p3_zkopen.cuh` + `p3_hawkeye_zk_test.cu` (19/19),
`leak_attack_zk.cu`, `bench_zk_overhead.cu`:
1. mask-slice augmentation [real|rand] (hides evals + codeword values);
2. Libra-blinded sumcheck (committed random blind h, published y_h, sumcheck on f+ρh — hides EVERY
   round message; mask-slice ALONE does not close F2);
3. salted hiding Merkle. HVZK simulator produces a witnessless identically-distributed accepting
   transcript. Battery: ≥12k mask draws, fixed witness+challenges, 257-bin chi-square — all
   transcript quantities uniform; negative control (blind off) shows the F2 signature at
   chisq=3.07e6. Attack: non-ZK recovers 16/16 committed weights exactly; ZK recovers 0/16 (flat
   posterior). Overhead ≈1.8–2× (projected up_proj B=1: ~12–13 s, ~14–15 MB).
RESIDUAL (honest): primitives proven in isolation + on real columns; the blind+augmentation is NOT
yet wired through all ~44 monolithic sumchecks + the batch opener. Recipe: augment every
commit_col_nc to 2N; run each zero-check over the augmented domain with (1−ex) weight + additive
Libra blind; make p3bo::prove_class hiding. GL2 (2^-116) orthogonal-pending.

## 11. Transformer layer (2026-07-03) — canonical reference + first nonlinearity gadgets

### HANDOFF (start here next run)

What compiles and passes on this machine, 2026-07-03 (second pass), all [VERIFIED] (build:
`nvcc -arch=sm_89 -std=c++17 -O2 <x>.cu -o /root/<x>`, run from /root/zkllm):
- `python3 transformer_ref.py` — canonical FULL tiny-layer reference battery: **ALL PASS**
  (22-op layer trace deterministic; scalar ops bitwise == torch bf16 in the normal range;
  RSQ/RCP tables bitwise == float64; pow2-scale e4m3 quantization bitwise == torch fp8 cast;
  faithfulness gaps measured, see below).
- `p3_rmsnorm_test` — RMSNorm gadget: **25/25 ALL PASS** (regression re-run after this pass).
  prove ≈0.2 s / case (B=4, d=64), verify ≈0.07 s, proof ≈2.6 MB (tiny-domain fixed costs).
- `p3_swiglu_test` — SwiGLU gadget: **16/16 ALL PASS** (regression re-run).
- `p3_hawkeye_test` regression after all of the above: **35/35 ALL PASS**.
- `p3_quant_test` — QUANTIZE gadget (NEW): **26/26 ALL PASS**. 13 goldens = the ACTUAL inputs
  of all 7 layer matmul call sites (from the golden trace) + random + edge (zeros/one-hot/
  huge/tiny/-0/underflow/saturation-region rows); codes AND fp32 scales bitwise == reference;
  forged emax / forged code / off-scale / operand-binding / params rejects. prove 0.37 s for
  all 13, ≈0.63 MB each, 3 logUp instances. TOTAL (no domain restriction — the extended
  QE4M3 table makes underflow lanes ordinary rows).
- `p3_bfadd_test` — bf16 ADD gadget (NEW primitive): **22/22 ALL PASS**. Goldens include the
  layer's two residual sites validated against `transformer_layer.bin` rows (x+oproj=resid1,
  resid1+down=out), the layer softmax subtracts and rope combine products, RNE exact-tie
  pairs (both parities), cancellation/zeros/subnormal/binade-crossing/far-boundary edges; a
  flush/saturate case is REJECTED by the prover (v1 domain). Teeth: round-direction flip,
  false-cancel, near-claimed-FAR, swapped-magnitude alignment, exponent forge, output/operand
  commitment binding, params/proof tampers. prove ≈0.9 s for 7 cases, ≈2 MB each.
- `p3_rope_test` — RoPE gadget (NEW, first COMPOSED gadget): **17/17 ALL PASS**. All 4 layer
  q/k head rotations bitwise == trace; instantiates the bf16-add block TWICE + 4 MUL7 product
  blocks; Q and OUT committed once with a/b + output halves opened via the fixed-index-bit
  slice trick (§11.5 move validated); cos/sin PUBLIC (verifier evaluates their MLEs itself —
  no commitment); rotated-pair-swap forgery caught by the half-slice openings; tampered
  public cos table rejected. prove ≈1.8 s for 6 cases, ≈4.2 MB each (44 lookups — the
  shared-ledger merge R3 is the fix). Underflowing-product case rejected (v1 domain).

Files this increment: `p3_quant.cuh` + `p3_quant_test.cu`, `p3_bfadd.cuh` + `p3_bfadd_test.cu`
(block-reusable core: ba_fill / ba_constraints / ba_lu_defs, all column-base relative),
`p3_rope.cuh` + `p3_rope_test.cu`; golden dumps added to `transformer_ref.py`
(`--dump-goldens-quant/-bfadd/-rope` → `p3_quant_golden.bin`, `p3_bfadd_golden.bin`,
`p3_rope_golden.bin`). SPEC DISCOVERY (recorded in §11.2a below): canonical rne_bf16
saturates eb>=255 even when exactly representable, so "far add ⇒ out = hi" FAILS for a
binade-255 hi operand — the bfadd gadget binds EO:=EH on far rows so REXP keeps far results
in [1,254] (v1 domain), and the edge golden pins it.

- `p3_softmax_test` — SOFTMAX gadget (NEW, the last nonlinearity): **19/19 ALL PASS**. Full
  masked softmax in one gadget: monotone-key rowmax (dominance DK ∈ R16 + SEL1 attainment
  binding the max PATTERN), subtract = one bfadd block (negation bound through the block's
  own operand-2 decomposition), EXP 65536-table keyed on the dp pattern, block-float
  denominator = the rmsnorm S machinery (floor SM_EMIN=100, per-lane shift q·pw+r, row-bound
  sum, pow2 sandwich), RCP lookup (REB = 277+hb−E−wd), MUL7 output gated MSK·(1−EZ). The
  causal mask and empty-row bits are PUBLIC columns (verifier evaluates their MLEs; batch
  padding = empty rows → +0 outputs, in-domain). Goldens: BOTH layer heads bitwise vs the
  layer-trace probs + full/causal randoms + edges (tied max, all-equal row, signed zeros,
  EMPTY mask row, exp-underflow lanes, masked lane ABOVE the participating max, subnormal
  score). Teeth: misplaced rowmax attainment (DK range), forged exp row (EXPT), forged
  denominator (row-binding), forged reciprocal (RCPT), masked-lane denominator leak (lane
  gating), tampered public mask, commitment/params/proof tampers. prove ≈1.5 s for 5 cases,
  ≈3.8 MB (34 lookups). Files: `p3_softmax.cuh` + `p3_softmax_test.cu`,
  `--dump-goldens-softmax` → `p3_softmax_golden.bin`.

ALL SEVEN LAYER OP KINDS NOW HAVE PROVEN GADGETS (matmul, rmsnorm, swiglu, quantize,
bf16-add/residual, rope, softmax) — every op of the §11.1 dataflow, each bitwise vs the
canonical reference on real layer data. NOT yet done: the shared-ledger composition harness
(R3), the composed full-layer proof (R5: one transcript, chained roots, bitwise vs
`transformer_layer.bin` end to end), non-pow2 d_model (R6), ZK wiring (R7), GL2 (R8). Next
step = R3 then R5. Do NOT claim a full layer until it composes bitwise end-to-end against
`transformer_layer.bin`.

### 11.1 Statement and layer decomposition

Target: a bitwise-exact, sound ZKP of one llama-style layer's fp8 forward pass,
```
x ─ rmsnorm(g1) ─ quant ─┬─ Wq ─ rope ─┐
                         ├─ Wk ─ rope ─┤ per head: quant ─ QK^T ─ softmax ─ quant ─ ·V^T
                         └─ Wv ────────┘                                        │
x ── + ◄── Wo ◄─ concat heads ◄─────────────────────────────────────────────────┘
└─ rmsnorm(g2) ─ quant ─┬─ Wg ─ silu ⊙ ┐
      resid x2 ── + ◄── Wd ◄─ quant ◄──┴─ Wu ──┘        (7 matmuls total)
```
Every matmul is the PROVEN Hawkeye fp8 atom (§0–§8) — matmuls are never reproven differently.
Activations BETWEEN ops are bf16 bit-pattern columns, committed once and shared: each gadget's
verifier takes the neighbouring gadgets' commitment roots as public inputs, so op-to-op chaining
is root equality at the orchestrator level (§11.5). The NEW work is the nonlinear/glue gadgets.

### 11.2 The canonical bit-exact layer reference (`transformer_ref.py`)

Tiny config: d_model=64, n_heads=2, head_dim=32, d_ff=128, seq=4, rope θ=10000, eps=1e-6, causal.
The canonical spec, chosen to be deterministic, total, and to map 1:1 onto the existing proof
vocabulary:
- **Scalar bf16 semantics**: RNE on the exact value; subnormal inputs/outputs flush to signed
  zero; overflow saturates to max-finite 0x7F7F; exponent-field 255 is an ORDINARY binade (no
  inf/NaN special cases — the same policy as Hawkeye's fp8 NaN handling). bf_mul/bf_add are
  exact integer routines; [VERIFIED] bitwise == torch bf16 mul/add on ~4.8k normal-range pairs.
- **Row reductions** (sum of squares, softmax denominator): "block-float" exact sums — each
  16-bit lane truncate-aligned to the row max binade E (phantom floor binade EMIN), exact
  integer sum. Same max/shift/truncate moves the matmul proof already uses.
- **1-in/1-out nonlinearities**: pinned finite tables. RSQ (rsqrt, 64K), RCP (recip, 32K), MUL7
  (RNE bf16 mantissa product, 16K), QE4M3 (bf16→e4m3 quantize magnitude, 4K), EPSA (eps aligned
  to the row binade, 512) are built with EXACT integer arithmetic (RSQ/RCP [VERIFIED] bitwise ==
  float64 on samples). EXP and SILU (64K each) use float64 libm + exact RNE; canonical = the
  pinned artifact bytes (`p3_rmsnorm_tables.bin`), whose hashes the proofs absorb.
- **Quantization** (activations→fp8): per-row POWER-OF-TWO scale 2^(emax_unb−8) (fp32 exponent
  clamped ≥1 so scales stay in Hawkeye's supported domain); each element quantized by the QE4M3
  table keyed on (eb_max−eb, mantissa). [VERIFIED] bitwise == torch `.to(float8_e4m3fn)`.
  1/√head_dim is folded into the Wq weight scales OFFLINE (weight-prep, nothing to prove).
- **Softmax**: causal mask is STRUCTURAL (public): masked lanes excluded from max/sum, outputs
  +0. Rowmax by monotone pattern key; subtract via one bf_add; e^x via the EXP table on the
  bf16 difference; denominator via block-float sum; reciprocal via RCP; p = bf_mul(e, rcp).
- **RoPE**: pinned bf16 cos/sin tables; q'_j = bf_add(bf_mul(q_j,c), −bf_mul(q_{j+h},s)) etc.
  (one RNE per product, one per combine, fixed order).
- **Residual**: elementwise bf_add.

HONESTY (measured, not hidden): "bit-exact" for nonlinear ops = bitwise vs THIS canonical spec —
exactly as the matmul is bitwise vs the Hawkeye kernel. Faithfulness to torch's kernels is a
separate question, and it is MEASURED by the battery: canonical RMSNorm differs from torch's
fp32-accumulation RMSNorm on 180/512 elements by ≤1 ulp; canonical softmax differs by up to
9 ulp on 13/16 (inherent to exp-of-bf16-rounded-difference; torch subtracts in fp32). The
matmul had DiFR=0 against the actual GPU kernel; the nonlinear ops have no such kernel-identity
claim. If kernel-faithful nonlinearities become a requirement, the canonical spec must be
re-derived from the deployed kernels (a different, larger project — flag to the coordinator).
Note also the EXP/SILU artifact caveat: entries are float64-libm-generated then exactly rounded;
regeneration on a different libm could differ in rare ≤1-ulp cases — the pinned artifact (not
regeneration) is normative.

### 11.2a Spec discovery (2026-07-03 second pass): bf16 saturation vs the e=255 binade

The canonical scalar semantics treat exponent-field 255 as an ordinary binade for DECODING,
but `rne_bf16` saturates any RE-ENCODED result whose exponent field would be >= 255 — even
when the exact value is representable in binade 255.  Consequence: bf_add with an aligned
difference d >= 10 is "out = hi bitwise" ONLY for EH <= 254; a binade-255 hi operand
saturates to ±0x7F7F.  The bfadd gadget therefore binds EO := EH on far rows and range-checks
EO with REXP, putting binade-255 far results outside the v1 domain (prover throws; verifier
never accepts).  Found by the edge-pair golden battery, not by inspection — keep goldens
adversarial.

### 11.3 Gadget vocabulary (everything reduces to six moves)

1. bit-decomposition + range lookups (pattern = s·2^15 + eb·2^7 + mb; R128/R256/R512/R16);
2. row max by dominance + attainment selectors (Σ sel + fsel = 1, sel·sh = 0, floor at EMIN);
3. truncating shift as q·pw + r with (sh,pw) and (pw,r) lookups (SH512/RM17/RM8);
4. pow2-sandwich normalization (WD24 + limb ranges + u16·PDN + NR = S'·PUP);
5. table lookups for the nonlinear kernels (RSQ/RCP/EXP/SILU/MUL7/QE4M3/EPSA/REXP), logUp with
   virtual (+128 mantissa) key columns bound by one extra opening each;
6. λ-combined quartic zero-checks + row-binding sumchecks + the shared batched opener (p3bo).

### 11.4 Gadget list, status, cost
                             domains (cols)            lookups  status
  matmul (×7)                Hawkeye atom (§2)           41     DONE (35/35), 6.2 s @ 768×3072 B=1
  rmsnorm (×2)               De:16+X,Y  Db:21  Dw:5      22     DONE (25/25): p3_rmsnorm.cuh
  swiglu (×1)                De:14+G,U,M                  7     DONE (16/16): p3_swiglu.cuh
  quantize (×7 call sites)   De:8       Db:3              3     DONE (26/26): p3_quant.cuh — QEXT
                                                                = QE4M3 extended to dexp<256 (0
                                                                rows = underflow rule) so ONE
                                                                lookup binds dexp/mb/mag; E floor
                                                                at 9 (se = max(emax-8,1) = E-8);
                                                                scale = (E-8)<<23 linear. TOTAL
                                                                (no v1 restriction). 0.63 MB
  bf16 add (residual ×2,     De:55                       16     DONE (22/22): p3_bfadd.cuh —
   rope combine, sm subtract)                                   hi/lo mux σ, FAR split at |Δ|>=10
                                                                (out=hi, EO:=EH REXP-checked, see
                                                                §11.2a), near = exact A<2^17 +
                                                                WDA sandwich + RM17 shift + RMH
                                                                (pdh,rb,rt) RNE + carry. v1: EO
                                                                in [1,254]. BLOCK-REUSABLE core
                                                                (ba_fill/ba_constraints/lu defs)
  softmax (×heads)           De:73      Db:20            34     DONE (19/19): p3_softmax.cuh —
                                                                rowmax = move 2 on the monotone
                                                                key 32768-s+(1-2s)(128e+mb)
                                                                (linear!); subtract = ONE bfadd
                                                                block (neg bound via the block's
                                                                operand-2 decomposition); EXP
                                                                keyed on dp; denom = rmsnorm's S
                                                                machinery, floor SM_EMIN=100;
                                                                RCP (REB=277+hb-E-wd); MUL7 out
                                                                gated MSK·(1-EZ). Mask + empty-
                                                                row bits PUBLIC columns. 3.8 MB
  rope (×2 per head)         De:137 (2 add blocks       44     DONE (17/17): p3_rope.cuh — first
                              + 4 muls)                         COMPOSED gadget; Q/OUT halves via
                                                                fixed-index-bit slice openings;
                                                                cos/sin public per-(pos,j) MLEs
                                                                evaluated by the verifier. 4.2 MB
                                                                (44 lookups → R3 merge is the fix)
  slice/concat/transpose     0 — MLE point permutations   0     head slice = fix high i-bits of
                                                                the opening point; V^T = swap
                                                                variable groups; concat = shared
                                                                parent column
Estimated per-layer nonlinear overhead at llama-68m dims (d=768, ff=3072, seq S≤64, B=1)
[REASONED]: element domains are ~S·d ≈ 50K rows vs the matmuls' P-domains of 19M — nonlinear
gadgets add roughly 1–3 s total against ~25–35 s for the 7 matmuls, i.e. ≤10%. Proof-size at
tiny scale is lookup-count-dominated (fixed ~50–120 KB per instance); the shared-ledger merge
(§11.6 R3) is the lever.

### 11.5 Chaining plan (op → op binding)

- One commitment per activation tensor, shared: rmsnorm's Y root IS the quantizer's X root, the
  quantizer's (codes, scales) roots ARE the matmul's X/xs roots, etc. The orchestrator carries
  the root list; each gadget verifier pins its neighbours' roots as public inputs. No
  re-commitment, no equality sumchecks needed.
- Padding conventions are part of the statement: gadget outputs are committed over padded
  domains with the CANONICAL extension (e.g. rmsnorm pads with the all-zero-row witness;
  padded-region values are defined, not free). Downstream gadgets must consume the same
  convention — this is checked bitwise when composing against `transformer_layer.bin`.
- Transposes/head-slices/concats are index-bit permutations/restrictions of the SAME committed
  column: the consumer opens the parent at a rearranged/partially-fixed point (as the DM
  virtual a/b binding and the §2 final-state slice binding already do). No data movement.
- The public statement of the composed layer: committed layer input root, committed weight
  roots (codes + scales, incl. rope/gain constants by table hash), committed layer output root.
  Intermediate roots appear in the proof but are pinned by the chain, not by the caller.

### 11.6 Roadmap (priority order)

R1. DONE 2026-07-03 — **Quantize gadget** (p3_quant.cuh, 26/26): all teeth landed (forged emax,
    forged code row, off-by-one scale; the saturation edge |x/scale|>448 is an honest-accept
    golden — the pinned QE4M3 maps the region above 464 to magnitude 0x7F, matching torch).
R2. DONE 2026-07-03 — **bf16 ADD gadget** (p3_bfadd.cuh, 22/22): all teeth landed (swapped-
    magnitude alignment, round-direction forgery, false-cancellation, near-claimed-far,
    exponent forge). Residual IS the standalone gadget (goldens = the two layer residual sites,
    bitwise vs transformer_layer.bin).
R3. **Shared-ledger composition harness**: run all gadget provers into ONE PLedger/transcript so
    every column opens once per size class across the whole layer (the tiny-scale proof-size
    fix), and dedup identical tables (MUL7/REXP appear in most gadgets; rope alone carries 44
    lookup instances at 4.2 MB — the motivating case).
R4a. DONE 2026-07-03 — **RoPE gadget** (p3_rope.cuh, 17/17): composed from 2 bfadd blocks + 4
    MUL7 blocks; rotated-pair swap caught by the half-slice openings; public-table binding
    caught by the verifier-side MLE evaluation.
R4b. DONE 2026-07-03 — **Softmax gadget** (p3_softmax.cuh, 19/19): built exactly as designed
    (monotone-key rowmax; bfadd-block subtract with the negation bound through the block's
    own operand-2 decomposition MXP = (1-S2)*32768 + E2*128 + M2; EXP keyed on the committed
    dp column with a committed EXV output column — masked lanes excluded downstream by the
    public MSK gate rather than a zeroed EXPP; denominator = rmsnorm's block-float machinery
    with floor SM_EMIN=100 and present = MSK*(1-EZ); RCP; MUL7 out). All planned teeth landed
    incl. the masked-lane denominator leak and the tampered public mask.
R5. **Compose the full tiny layer** and validate bitwise end-to-end against
    `transformer_layer.bin` (every intermediate root + final output). Only then claim the layer.
R6. Non-pow2 d_model (768): fold the /3 into RSQ'/EPSA' table variants (√3 factor), mean = /2^8.
    Mechanical; tables are per-shape public artifacts already.
R7. ZK: land §10's mask-slice + Libra-blind over the composed column set (mask once at the end),
    incl. the batch opener; rerun the §10 chi-square battery on layer columns.
R8. GL2 challenges stack-wide (2^-116 soundness); then the §9 leftovers (GKR-logup, bit-packing,
    padding recovery) if layer-scale prove time matters.

### 11.7 Soundness notes / known restrictions (honest list)

- Gadget v1 domain restrictions (prover throws, verifier never accepts a wrong claim — sound,
  not complete): rsqrt exponent TEB ∈ [1,254]; product exponents EO ∈ [1,254]; bf16-add result
  exponent EO ∈ [1,254] on near rows AND far-row EO := EH ∈ [1,254] (§11.2a — flushed
  subtracts, saturated adds and binade-255 far results are out of v1). The canonical reference
  is total (flush/saturate); rows that clamp are out of gadget-v1 domain. Same class of
  restriction as the §2 scale-domain rule. The QUANTIZE gadget has NO restriction (total).
- The teeth batteries check each forgery is caught by the INTENDED sub-argument (binding
  sumcheck / RSQ / EPSA / attainment / RM / MUL7 / QEXT / RNE-RMH / half-slice openings /
  public-MLE bindings / commitment bindings / params / proof-object tampers). 25/25, 16/16,
  26/26, 22/22, 17/17, 19/19 [VERIFIED 2026-07-03].
- Base-field challenges (soundness ≈2^-40 per check, as §2): GL2 upgrade pending stack-wide.
- Non-ZK: §10 residual unchanged; nothing new leaks-wise was added (all new columns are the same
  class of witness columns the §10 recipe masks).

## 12. Full-layer ZERO-KNOWLEDGE (2026-07-03) — the composed layer is now ZK [VERIFIED]

R7 landed: the ENTIRE composed transformer-layer proof (p3_transformer.cuh, all 34 gadget
instances + seams + shared batch opener) is now zero-knowledge, applying the §10/§11 (p3_zkopen)
mechanisms to EVERY committed column, EVERY sumcheck message chain and EVERY opening. Gated by the
global `p3zkc::G.on` (default OFF → the sound+bitwise legacy path is byte-identical; see the
regression below). New substrate: **p3_zkc.cuh**.

### 12.1 What was masked (four mechanisms, per column / per message / per opening)
1. **Mask-slice augmentation (2^e slices).** Every committed column of real length N=2^v is committed
   as `[real | mask]` of length N·2^e, mask fresh-uniform, e = `e_of(v)` chosen so N·(2^e−1) ≥
   4·(2Q+64) (the per-proof revealed-functional budget with 4× headroom for seam-linked groups).
   Constraint zero-checks run over the augmented domain with `eq((z‖0),·)` weights — the (1−ex_j)
   factors KILL the mask slices, so masks are unconstrained there; every opened eval is taken at a
   point with a random ex-coordinate → uniform over the mask.
2. **Degree-matched Libra blinds.** A plain multilinear blind hides s(0),s(1) but leaves the
   t²…t^D finite-difference coefficients of a degree-D round message as pure witness functionals
   (the *coefficient-leak class*). Every sumcheck adds `ρ·(B1 + E·B2 + E²·B3 + E³·B4)` (4 committed
   blind columns for the quartic gadget checks, 3 for the cubic logUp chains), E = the check's own
   weight column → the four terms have round degree 1..4 and span every coefficient. H = Σ blind is
   published after the point but before ρ (sound by Schwartz–Zippel in ρ).
3. **Salted hiding commitments.** Every commitment root in zk mode is a Merkle root over salted
   leaves SHA256(value‖salt); internal nodes keep the p3fri single-compress node hash so
   `p3fri::verify_path` is unchanged. Opened positions disclose (value, salt, path) only, and the
   values are already uniform by (1).
4. **Batch blinder column.** The per-size-class batched opening RLCs in one fresh full-domain
   uniform committed column, one-time-padding the combined word U → the opening-sumcheck messages,
   the fold-round openings and the final word are all uniform.

Plus two composition-specific points: **seam mask linkage** — the five seam types (shared-root,
restriction/zero-pad, head-slice, transpose, concat) link the consumer column's mask *slice 1* to
the same seam transform of the producer's slice 1, and the seam claim opens both sides at a SHARED
random ex-coordinate (touching only the real slice + slice 1), so the augmented evals agree while
the value is hidden; and **shared-output reuse** — the matmul output column O_YB is committed ONCE
(random mask/salt) and threaded into the Hawkeye sub-prover, because in zk two independent
commitments of the same values get divergent random masks and would break the shared-root chain
(the bug that first surfaced as a softmax-side transcript divergence). The logUp rational-sum S is
blinded by matching the hA-side and hT-side mask sums to the SAME m, so S' = S+m hides the value but
PRESERVES the S_A==S_T equality that detects a witness-not-in-table forgery. The per-matmul PUBLIC
output vectors (the cleartext activations) are DROPPED in zk mode; outputs bind through the seams
and the final public-output binding. Public weights: their commitment roots are the published
statement (the prover fixes them once; the test pins them from the prover's commitments).

### 12.2 Hiding results (p3_transformer_zk_test.cu — the full-layer hiding battery) [VERIFIED]
Same rigor as p3_hawkeye_zk_test: the actual full-layer private column set (extracted from a real
`build_witness`) driven through the actual hiding mechanisms over **12 000** fresh mask/blind/salt
draws at FIXED public challenges (a quantity hides iff, at fixed challenges, it is uniform over the
mask draws). 257-bin chi-square, uniform band < 400, leak band > 5000. **13/13 PASS:**
- **(1) every layer column class uniform.** rms1.Y, quant CODES, matmul X/W codes, every matmul
  output Y, rope OUT, softmax P, swiglu M, residuals, final out — claimed eval y, blind eval y_h,
  EVERY round message s0/s1/s2 (incl. the ex round where F2 lived), FRI final constant, revealed
  codeword values ALL chi-sq ≈ 235–313.
- **(2) message finite-difference coefficients** (Δ²,Δ³,Δ⁴ — the coefficient-leak class): blinded
  chi-sq 214–276; **negative control blind-off → 3.07e6** (the coefficient leak is real and closed).
- **(3) seam claims:** uniform (chi-sq 241) AND producer==consumer at the shared ex-coordinate
  (composition binding preserved).
- **(4) batch blinder:** combined word uniform (chi-sq 244).
- **(5) witness-recovery attack** on the full-layer transcript quantities: the leaking control
  collapses to **1 distinct** ex-round s0 over 12 000 draws (a recoverable functional — the F2
  attack); the hidden transcript is **12 000/12 000 distinct** and uniform → 0 bits; posterior flat
  (two different witnesses give the same uniform law).
- **(6) HVZK simulator:** witnessless transcripts, same law (chi-sq 260/277) and verify-accept.

Composed soundness under zk (zk_layer_smoke.cu, G.on): honest layer VERIFIES; all 11 gadget/seam
tampers reject **at their own site** (restriction/zero-pad/slice/transpose/concat seams caught by
the seam check, gadget forgeries by their gadget), flipped public output and wrong pinned weight
root reject; cleartext matmul outputs confirmed dropped. **15/15 PASS.**

### 12.3 Overhead (tiny layer seq=4,d=64,nh=2,dh=32,dff=128), zk OFF → ON
| quantity | non-ZK | ZK | factor |
|---|---|---|---|
| prove  | 6.95 s | 36.9 s | 5.3× |
| verify | 1.84 s | 4.37 s | 2.4× |
| proof  | 62.0 MB | 205.3 MB | 3.3× |
The proof/verify blow-up is dominated by the mask-slice augmentation (×2^e per column, e up to 7
for the smallest domains under the fixed 4·(2Q+64) budget) and the salted-Merkle + blind columns;
much of it is the tiny-layer's padding tax (e is large only because N is small) and shrinks in
relative terms at real-model scale.

### 12.4 Regression — soundness + bitwise-exactness preserved [VERIFIED 2026-07-03]
zk OFF is byte-identical to the pre-ZK code: p3_transformer_test **30/30**; gadget selftests
hawkeye **35/35**, rmsnorm **25/25**, quant **26/26**, bfadd **22/22**, rope **17/17**, softmax
**19/19**, swiglu **16/16**, logUp **13/13**; the isolated hiding primitive p3_hawkeye_zk_test
**19/19**. Every chained intermediate AND the final output stay bitwise == transformer_layer.bin.

### 12.5 Honest final status of the whole system
**PROVEN (this build, VERIFIED):** a **sound + bitwise-exact + zero-knowledge** proof of the exact
fp8 forward pass of a full transformer layer (rmsnorm · quantize · Hawkeye matmuls · rope · softmax ·
bf16-residual · swiglu, canonical = transformer_ref.py). Sound: 30/30 composed + all gadget teeth,
every tamper/seam-break rejects. Bitwise-exact: every intermediate == the golden trace. ZK: the
full-layer transcript is hiding end-to-end — every column, every message coefficient, every seam,
every batch terminal is uniform over 12k draws; the F2 weight-recovery attack extracts 0 bits; an
HVZK simulator reproduces the law without the witness.

**Remaining gaps (honest):**
- **Single layer** — one transformer block, batch/seq tiny; not a full model or multi-layer chain.
- **Canonical-vs-vendor nonlinearity faithfulness** — the gadgets prove the *canonical* reference
  (transformer_ref.py) exactly; matching a specific vendor kernel's bit-quirks is separate.
- **GL2 soundness margin** — challenges are base-field (≈2^-40/check); the mechanical degree-2
  extension to ≈2^-116 is still pending stack-wide (unchanged by this pass).
- **Real-model scale** — proven at seq=4/d=64; the mask-slice budget e_of shrinks relatively at
  scale but memory/time at production sizes is future work (GKR-logUp, bit-packing, §9 levers).

## 13. Optimization + overhead sweep (2026-07-04) [VERIFIED unless marked]

### 13.1 Speed + proof-size pass on the composed layer

Profile-driven; the baseline profile showed 61.3 of 62.0 MB was the shared batch opening, of
which ~98% was round-0 per-column Merkle authentication over 3812 distinct committed columns
(~2300 of them the per-lookup logUp helper columns cnt/hA/hT of 781 lookup instances), and
4.6 s of the 7.0 s prove was inside `p3lu::prove_v` (781 calls x 5.9 ms fixed cost: per-instance
table-side sumcheck 1.9 s + 3 helper commits/instance out of 4566 total commits at ~0.4 ms GPU
launch/alloc latency each).

Levers, cumulative, tiny layer (seq=4, d=64, nh=2, dh=32, dff=128, batch=1), non-zk:

| lever | prove | verify | proof | note |
|---|---|---|---|---|
| baseline (S12 state) | 7.03 s | 1.83 s | 62.03 MB | 30/30 |
| L1 pruned Merkle subset-paths (round-0) | 7.38 s | 1.56 s | 27.24 MB | positions deduped; 2Q overlapping paths cost each tree node once (`p3bo::BQ0C`, `subset_prove/verify`) |
| L2 layer-level MERGED lookups (R3b) | 2.97 s | 0.48 s | 13.51 MB | 781 lookup instances -> 125 groups |
| L3 NTT plan cache + pooled encode allocs | 2.41 s | 0.47 s | 13.51 MB | twiddle tables were rebuilt per commit |
| L4 GPU T-side group sumcheck (m>=14) | **2.05 s** | **0.47 s** | **13.42 MB** | byte-identical messages |

Final tiny-layer state: **prove 2.05 s (3.4x), verify 0.47 s (3.9x), proof 13.4 MB (4.6x),
peak RSS 0.45 -> 0.27 GB.**  Zero-knowledge mode (G.on) inherits both levers:
**zk prove 36.9 -> 10.7 s (3.5x), zk verify 4.4 -> 1.1 s, zk proof 205 -> 46 MB (4.5x)** --
merged groups also merge the per-lookup Libra blinds (6 per GROUP instead of per lookup).

**L2 design (the main lever).** Gadgets no longer run `p3lu::prove_v` inline: they DEFER each
lookup obligation (witness specs, index list, table, an optional binding callback) into the
shared context queue (`p3lu::XCtx::luq` via `defer_v`; verifier mirror `VCtx`/`vdefer_v`), and
the ledger owner -- the standalone gadget or the composed layer -- flushes ONCE (`lu_flush` /
`lu_verify_flush`), grouping obligations by (table id, log2 rows) in first-appearance order.
Each group is proven as ONE logUp instance over the stacked member domain
u = i | (j<<n) | (ex<<(n+g)) (member row i, member j of k, zk mask coordinates ex of the
MEMBER policy width E = e_of(n)); pad members j in [k, 2^g) hold table row 0, whose
multiplicity is added to cnt[0].  One cnt/hA/hT (+ 2x3 zk blind columns) per GROUP.  The
verifier's A-side terminal is A~(rA) = sum_j eq(rmid, j) * (sum_c gamma^c y_{j,c}) + pad*a0.
Every member's witness claims land at the SHARED point pm = (rA[0..n) || rA[n+g..)), whose
shape equals a standalone lookup's opening point in both modes -- so the per-gadget virtual-
column binding code (matmul DM a/b -> X/W, rope/softmax/rms/swiglu MUL7 keys) moved into bind
callbacks UNCHANGED; binding claim values ride the group proof's per-member `extra` slots.
In the composed layer all 34 gadget instances share one flush: the WQ/WK/WV/WO DM lookups
merge across instances, all R15/R10/R12/... range lookups of the same shape merge, etc.
Committed columns fell 4566 -> 2598, distinct batch-opened columns 3812 -> 1969.

Regression after EVERY lever (final state re-verified end to end): composed p3_transformer_test
**30/30**, full-layer hiding battery **13/13**, zk soundness smoke **15/15**, gadget selftests
hawkeye **35/35**, rmsnorm **25/25**, swiglu **16/16**, quant **26/26**, bfadd **22/22**, rope
**17/17**, softmax **19/19**, logup **13/13**, isolated hiding **19/19**; every chained
intermediate and the final output stay bitwise == transformer_layer.bin.

Remaining tiny-layer prove profile (2.05 s): merged-lookup flush 0.70 s, batch opening pass
0.77 s, matmul zero-checks/commits 0.47 s.  Next levers [REASONED]: GKR-logUp (kills the
remaining per-group hA/hT commitments entirely), fusing the per-column round-0
encode+tree work across a class into per-level kernels, bit-packing.

### 13.2 Batch/heads generalization of the composed prover

`p3tf::Config` gains `batch` (and general pow2 `nh`); attention instantiates per (b, h),
flattened a = b*nh + h (A = batch*nh instances of rope-q/k, QK^T, softmax, prob/V^T quantize,
P.V), while the token-parallel ops (rmsnorm, quantize, projection/MLP matmuls, swiglu,
residuals) run over ONE T = batch*seq token grid (token t = b*seq + s).  The head-slice,
V^T-transpose and concat seams fix the head bits of the model index AND the batch bits of the
token index in the opening point (`slice_pt`); the zk seam-linked masks generalize the same
way.  At batch=1, nh=2 the instance order and seam points coincide with the original layer --
the 30/30 battery and the 13/13 hiding battery run on exactly that shape and pass unchanged.
Instance counts: 7 + 2A matmuls, 4 + 4A quantize, 2A rope, A softmax.

### 13.3 Tokens + seq-vs-batch sweep [VERIFIED, coordinator-run 2026-07-06, optimized non-ZK prover]

Config d=64,nh=2,dh=32,dff=256. overhead = ZK-off prove / plain fp8 forward (tf_fwd_bench).

TOKENS (batch=1, vary seq):
| tokens | prove | fwd | overhead | mm | smx |
|--:|--:|--:|--:|--:|--:|
| 8  | 2.76 s | 3.18 ms | 867x  | 0.60 | 0.015 |
| 16 | 4.10 s | 2.91 ms | 1406x | 1.12 | 0.023 |
| 32 | 6.69 s | 3.27 ms | 2043x | 1.95 | 0.038 |
| 64 | 11.22 s| 3.25 ms | 3454x | 2.97 | 0.106 |

SEQ vs BATCH at FIXED tokens=64:
| seq×batch | prove | fwd | overhead | mm | smx |
|--:|--:|--:|--:|--:|--:|
| 8×8  | 13.46 s | 15.06 ms | 894x  | 3.43 | 0.126 |
| 16×4 | 12.32 s | 8.01 ms  | 1537x | 3.49 | 0.092 |
| 32×2 | 11.30 s | 4.64 ms  | 2438x | 2.91 | 0.076 |
| 64×1 | 11.30 s | 2.85 ms  | 3967x | 2.98 | 0.106 |

CONCLUSION: seq and batch do NOT matter only through their product. At fixed tokens=64 the
overhead spans 894x–3967x (4.4x) with the split. Direction: prove-time is ~flat-to-slightly-
decreasing in seq (13.46->11.30 s) because at this scale the prover cost is dominated by
per-(batch·nh) attention-INSTANCE fixed overhead (8×8 => 16 attn instances vs 64×1 => 2), not
the O(batch·nh·seq^2)=O(tokens·seq) attention compute (which is a minor component here: mm/smx
stages ~flat). The plain forward drops sharply with seq (15->2.85 ms) as it is kernel-launch-
bound in batch. So overhead RISES with seq mostly via the shrinking forward denominator.
CAVEAT: tiny-scale; both sides are overhead-bound (forward ~3 ms launch floor; prover fixed
per-instance costs). Asymptotically the attention O(tokens·seq) term would eventually make long
sequences the expensive end for prove — not reached at these sizes. Absolute overhead ~10^3
(non-ZK) / ~10^4 (ZK, ~8x the non-ZK from 13.4) — order-of-magnitude, consistent w/ zkML.
