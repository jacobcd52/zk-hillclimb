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

## 14. llama-68m scale (2026-07-06) [VERIFIED unless marked]

Goal: the composed layer proof at REAL llama-68m per-layer dims — d=1024, nh=16, dh=64,
dff=4096 (pow2-padded) — and the ZK overhead sweep there.  Bench:
`/root/p3_transformer_bench <seq> 1024 16 64 4096 <batch> <zk> tables_ld10.bin`
(`P3_MEMLOG=1` prints per-group / per-class memory shapes and phase timings).
NOTE: seq*batch = 1 is unsupported (row pads are >= 2; the T=1 corner mismatches the
swiglu virtual-column domains).  Use >= 2 tokens.

### 14.1 RMSNorm (and softmax) large-d window fix

The canonical S' window was FIXED at 2^23 (transformer_ref.py assert + gadget tables), which
is exactly the arithmetic bound for ld = 6: S = sum of aligned mantissa-squares < 2^(ld+16),
EPSA < 2^18, so S' < 2^(17+ld).  At d = 1024 (ld = 10) honest witnesses overflow it ("rms: S'
window", no in-domain witness).  Fix: the window is now PARAMETRIC, wd <= 17+ld, in both the
reference (assert) and the gadget:
  * T.R11 -> range(2^(ld+5))  (U1H/U2H high limb; low limb stays 12 bits, constraints unchanged),
  * T.RM8 -> (pw, r<pw) rows for pw <= 2^(ld+1)  (the S' remainder-alignment table),
  * T.WD24 valid rows wd <= 17+ld (32 rows still; ld <= 13),
  * witness guard S' < 2^(17+ld).
At ld = 6 every table is BIT-IDENTICAL to the historical build (17+6 = 23), so ld=6
transcripts/tests are unchanged.  Softmax has the same window over its seq-length denominator
sum (16-bit lanes, no eps): p3smx::build_tables(a, wmax) with stored Tables.wmax, prover/
verifier guard 16+ln <= wmax; the composed builder takes p3tf::build_tables(a, smx_wmax)
(default 23 == historical; the bench passes max(23, 16+ceil_log2(seq))).
[VERIFIED] p3_rmsnorm_test now takes optional (tables, goldens) argv:
ld=6 25/25 (regression), ld=8 25/25, ld=10 25/25 — bitwise vs the widened reference AND the
full must-reject battery at each size; reference validation battery ALL PASS; softmax 19/19.

### 14.2 The two real memory walls

(a) **GPU (24 GB RTX 4090)**: the biggest committed columns at d=1024 are the matmul
per-product witness columns (NDP = 10 per instance) on P = Kpad*Bpad*Npad — 2^25 per MLP
matmul at 4 tokens, ∝ tokens.  A single commit's transient (codeword 2^27 x 8B + full device
Merkle 64B/leaf = 8 GB+) plus the retained mempool exceeded the card: cudaMallocAsync failures
were UNCHECKED, so the prover silently produced corrupt proofs (honest proof rejected with
"group sumcheck A" — the first verified object).  This is now impossible: p3bf::ckcuda /
dmalloc throw at the failure point.

(b) **Host: a 41 GB cgroup cap** (`/sys/fs/cgroup/memory.max` = 41.0e9; `free` shows the
host's 251 GB but the container OOM-killer fires at 41 GB, exit 137; 9.4 GB free disk => no
swap).  Every committed column's host values are retained for the final shared batch opening,
so TOTAL committed bytes ~ bound the proof.  This cap — not the GPU — is the binding
constraint at d=1024.

### 14.3 Levers landed (all transcript-identical; every battery re-run green)

GPU side:
  * **Streamed Merkle (p3fri::StreamTree)**: pow2-aligned 2^22-leaf chunks reduce to their
    subtree roots (bit-identical to the flat tree), host upper tree; path extraction rebuilds
    only the <= 2Q touched chunks (one-pass build+paths variant for round-0 subset opens).
    Used by commit_gpu_rootonly, salted_commit_root (offset-aware salted leaf kernel), the
    batched-opening fold rounds and per-column round-0 opens for codewords >= 2^24.
  * **Streamed prove_class**: distinct columns stay on HOST, uploaded per use (strCol, > 2 GB
    per class); the per-point combined columns G_t park on host between rounds with the eq
    columns REBUILT per round from separability (strG, > 3 GB); fold codewords park on host.
    Query values for rounds >= 1 read from the parked host codewords.
  * **LU_GCAP = 26** (protocol constant, prover+verifier grouping): merged-lookup groups are
    split so the stacked domain n+g <= 26 (helper columns and the merged witness A live on
    2^(n+g+E)); inactive at tiny dims (historical transcripts unchanged).
  * zk group A-side sumcheck moved to the GPU (FLuGpuZk functor; messages byte-identical).
  Result: GPU peak 24 GB (corrupt) -> 11.7 GB (verify_ok=1) at seq=4 non-zk.

Host side:
  * **OpenMP** (build with `-Xcompiler -fopenmp`; 128 cores): build_eq, eval_h, inv_all_add
    (block-wise Montgomery batch inversion), the merged-A build, sc5_prove / sc_prove message
    loops, bind_lsb, blind-H sums.  Exact field ops => identical values, any order.
    seq=4 d=1024 non-zk prove: 169 s -> 97 s (claims 6.9 s -> 0.7 s per big group).
  * **Drop-and-regenerate ledger columns (PLedger::Ent.gen)**: columns that exist only to be
    batch-opened at the very end no longer stay resident —
      - Libra blind columns (sc5z quartic blinds; group A-side cubic blinds): pure PRNG
        streams, regenerated from their captured seed (blind_col_seeded / blind_col_aug);
      - merged-group hA helper columns: recomputed from the (still-resident) member witness
        columns + gamma/beta (rebuild closure).
    prove_class materializes them transiently per use.
  * `p3hwl::g_free_dp` / `p3lu::g_free_idx` (bench-only, witness-mutating): the 10 per-product
    witness columns free right after their commits copy them; lookup index vectors free after
    their group is proven.

### 14.4 Additional zk levers (what made ZK fit under 41 GB) [VERIFIED]

The zk proof at d=1024 was cgroup-OOM-killed three more times after 14.3; each kill pointed
at the next resident block, all now fixed (transcripts unchanged; every battery re-run green):
  * **sc5z on the device** (`sc5z_gpu`, FF5Zk functor: quartic base + Libra blind term): the
    host copy of one P-domain zero-check's columns plus its first-bind halves (~10 GB) never
    exists; the blind terminal values are read from the fully-bound device columns (the
    v-round fold IS the multilinear restriction, exact).
  * **ColSrc borrow-through-round-0**: the matmul Dp/bind zero-checks no longer copy the 10
    committed P-domain columns (borrowed for round 0; the first bind writes the owned halves).
  * **Device-generated group blinds**: the merged-group A-side Libra blinds are generated on
    the GPU from a random-access PRNG stream (`zprng_at`, splitmix64 -- the host regenerates
    bit-identically for the opening ledger), committed from device (`salted_commit_root_dev`),
    blind-sum H reduced on device; the host never holds a byte of them.  Am and hA host
    copies are dropped right after their device upload (hA's ledger entry regenerates).
  * **strGdev**: the batched-opening per-point combined columns G_t park on the DEVICE when
    T+2 columns fit in 12 GB (the v=26 zk class has T=19 x 512 MB = 9.7 GB, which as HOST
    parking breached the cap); host parking remains the fallback.
  * malloc_trim at phase boundaries (OpenMP arenas).

### 14.5 Status at d=1024 [VERIFIED]

**A zero-knowledge llama-68m layer proof works on this box**: seq=4 batch=1 zk:
verify_ok=1, **prove 750.9 s (12.5 min), verify 5.6 s, proof 167.8 MB, host RSS 36.1 GB**
(cap 41), witness 18.6 s.  Non-ZK same point: prove 154.7 s, verify 3.0 s, proof 47.5 MB,
RSS 18.6 GB, GPU peak 11.7 GB.  (The drop-and-regenerate levers cost ~1.6x prove time vs the
resident-everything build -- 97 s measured before they landed -- but they are what makes zk
and the 8-token point fit at all.)

Full regression after ALL of the above: hawkeye 35/35, rmsnorm 25/25 at ld=6 AND ld=8 AND
ld=10, swiglu 16/16, quant 26/26, bfadd 22/22, rope 17/17, softmax 19/19, logup 13/13,
isolated hiding 19/19, zk soundness smoke 15/15, composed layer 30/30, full-layer zk hiding
13/13.  (Note: the sc5z_gpu / device-blind / strGdev paths trigger only above ~2^20-row
domains, so the tiny-dim batteries exercise the host paths; the d=1024 verify_ok=1 runs are
the coverage for the big-domain paths.)

## 15. ZK-prover optimization at d=64 (2026-07-07) [VERIFIED]

Context: the two headless ZK-opt sessions (e0654904, 9130a75f) implemented these levers but died
mid-run (ScheduleWakeup on unresumable sessions) before the full battery could be re-run, and the
working tree was subsequently scrambled by partial-subset commits and recovery attempts.  This
section documents the RECONSTRUCTED and now fully re-verified result.  Reconstruction method: the
verified end-of-scaling tree (all batteries green, section 14) was rebuilt by deterministic replay
of the session transcripts (every Edit/Write plus every file-modifying bash heredoc/sed, validated
byte-identical against the 8 files committed in 37ef86a and against two independent per-file
reconstructions), then the 84 recorded ZK-opt ops were replayed on top; the result matches the
surviving /root/zkopt_backup snapshots byte-for-byte for p3_zkc / p3_logup / p3_batchopen /
p3_basefold.

Bench config: `p3_transformer_bench 8 64 2 32 128 1 {zk} tables_ld6.bin` (seq=8 d=64 nh=2 dh=32
dff=128 batch=1), RTX 4090, nvcc 12.4, sm_89.  STAGES are prover seconds per phase.

Before (end-of-scaling tree, measured on this machine, this session):
  zk=1: prove 36.5 s  verify 1.92 s  proof 47.8 MB   STAGES mm=5.6 lug=23.3 batch=7.0
  zk=0: prove 16.5 s  verify 1.55 s  proof 15.6 MB   STAGES mm=0.40 lug=11.2 batch=4.7
After (ZK-opt tree, median of 3 runs, same machine/session):
  zk=1: prove 4.1 s   verify 0.56 s  proof 41.5 MB   STAGES mm=1.25 lug=1.15 batch=1.3
  zk=0: prove 1.46 s  verify 0.26 s  proof 13.6 MB   STAGES mm=0.40 lug=0.44 batch=0.43

=> ZK prove 36.5 -> 4.1 s (8.9x); non-zk prove 16.5 -> 1.46 s (11x); the lookup phase ("lug",
the targeted ~60-70% of prove) went 23.3 -> 1.15 s zk / 11.2 -> 0.44 s non-zk.  Final ZK/non-ZK
prove ratio at d=64: 4.1 / 1.46 = 2.8x (was ~2.2-4x depending on machine load at baseline).
Proof size zk 47.8 -> 41.5 MB, non-zk 15.6 -> 13.6 MB.

What changed (all bitwise-neutral to the accepted output; transcript format changes are
prover/verifier-symmetric):
- p3_logup.cuh v2 TABLE-level lookup merge ("supergroups"): a GroupProof now bundles ALL
  obligations of one table as per-(log-rows) A-side subgroups sharing ONE cnt / hT / T-side
  chain per table per flush.  The T-side work (Tc combine + inversion, hT + cnt commits, the 3
  T-blinds, the T sumcheck and its evals) was the dominant lug cost and is now paid once per
  TABLE instead of once per (table, log-rows) group.  Soundness: logUp multiset identity over
  the union of subgroup rows (sum_s S_A,s == S_T, cnt = summed multiplicities); zk: each
  subgroup S blinded by its own hA mask-tail sum, hT mask fixed up so the augmented T-side sum
  equals sum_s S'_s (same mechanism as v1, applied to the sum).
- p3_basefold commit_gpu_rootonly + p3_zkc salted root-only device commits: lookup helper and
  blind columns are committed root-only on the GPU (no host codeword/tree retention); blind
  columns are pure zprng_at streams, dropped after their claims and regenerated transiently at
  opening time by the batched-opening ledger.
- prove_group/prove_super member-claim evals parallelized (values computed in parallel,
  absorbed in original transcript order - exact arithmetic, transcript unchanged); host
  OpenMP-threshold tuning in p3_hawkeye/p3_logup (P selection no longer flips on tiny domains).
- p3_batchopen: per-class opening work deduplicated/streamlined for the merged classes;
  p3_merkle/p3_fri: shared subset-path gather for the pruned round-0 trees; p3_ntt plan reuse.

Verification of THIS state (all run on this machine, this session, from /root/zkllm sources):
all 9 test binaries compile; hawkeye/matmul 35/35, rmsnorm 25/25, swiglu 16/16, quant 26/26,
bfadd 22/22, rope 17/17, softmax 19/19; composed layer 30/30 with EVERY chained intermediate
bitwise == transformer_layer.bin; full-layer ZK hiding battery 13/13 (per-column uniformity
chi-sq < 400 over 12000 draws, blind teeth with 3.07e6 negative control, seam agree, witness-
recovery 0 bits, HVZK simulator accept).  bench verify_ok=1 in all runs.

Remaining levers (untried): GKR-logUp for the remaining lug (1.15 s), folding the batch phase's
per-class reductions (1.3 s), GPU-porting the remaining host mm sumcheck epilogue (mm zk-side
1.25 s vs 0.40 s non-zk is mostly Libra-blind host work), bit-packed small-value columns.

## 16. GKR-logUp, opening folding, and ZK at llama-68m scale (2026-07-07) [VERIFIED unless marked]

This pass implemented four of the five planned levers on top of the section-15 tree, and in
the process found and fixed a LATENT COMPLETENESS BUG that made every honest zk proof at
d >= 128 REJECT (see 16.4 -- the section-15 tree was verified only at d=64, where the broken
path never triggers).  All numbers measured on this machine (RTX 4090, nvcc 12.4, sm_89),
bench `p3_transformer_bench <seq> <d> <nh> <dh> <dff> 1 {zk} tables_ld{6,7,9,10}.bin`.

### 16.1 Lever 1 -- GKR-logUp (p3_gkr.cuh, p3_logup.cuh v3)

The v2 supergroup lookup argument committed, per (table, log-rows) subgroup, a helper-inverse
column hA over the stacked domain plus 3 Libra blind columns, and per table an hT + cnt + 3
T-blinds; each cost a salted commit AND a batched-opening ledger entry.  v3 replaces the
committed helpers and their blinds with COMMIT-FREE GKR fractional-sum trees:

  * leaves (zk, A side, x = i|(j<<n)|(ex<<(n+g)) over 2^(n+g+E)):
      leaf(2x)   = ( ex==0 ? 1 : 0 ,  Am(x)+beta )    real / witness-mask rows
      leaf(2x+1) = ( pm(x), qm(x) )                   committed uniform mask streams
    T side: leaf(2x) = ( cnt_aug(x), ex==0 ? Tc_j+beta : 1 ), leaf(2x+1) = (pmT,qmT).
  * the tree combines (p,q) (+) (p',q') = (pq'+p'q, qq'); the root (P,Q) is published and a
    chain of cubic layer sumchecks (one per level, p3_gkr.cuh) reduces it to leaf claims;
    the verifier checks the multiset identity on the roots: sum_s P_s/Q_s == P_T/Q_T.
  * leaf-claim binding preserves the v2 member interface EXACTLY: the even-q claim equals
    A_r + beta with A_r the same gamma-combined member/eq_bits/pad aggregation at
    pm = (rA[0..n) || rA[n+g..)), so all per-gadget bind callbacks are untouched.  The
    even-p claim is chi[ex=0]~ (public); cnt's claim is the mechanism-1-blinded augmented
    eval at rT (the T tree spans cnt's mask rows as q=1 leaves whose junk is beta-INDEPENDENT).
  * zk: every real leaf gets a fresh uniform committed sibling (multiplicative masks
    pm = sm*qm, so mask fraction sums are PLAIN SUMS of the sm stream -- no inversions);
    every height-1 node is then a uniform (p,q) pair (bijective in (sm,qm) given q_real != 0),
    so the ENTIRE chain above the leaves is a deterministic function of a uniform vector and
    is simulatable; all mask columns commit BEFORE gamma/beta (the Schwartz-Zippel argument
    needs the junk fixed pre-beta), and the T-side fixup cancels it exactly:
    SmT = sum_s SmA_s - Sm_cnt.  Mask columns are pure zprng_at streams: root-only salted
    commits, values regenerated at leaf build and by the opening ledger (never resident
    across subgroups).
  * prover: host chains for small layers, fused 2-launch-per-round device rounds above 2^13,
    device-resident tree above 2^17 leaves (p3gkr devtree; byte-identical messages -- exact
    field sums).  Standalone battery p3_gkr_selftest 88/88 (root == direct rational sum,
    leaf claims == true MLEs at rfin, tampered root/message/terminal/truncation reject,
    wrong-leaf forgery shifts the bound claim).
  * hiding battery extended with GKR teeth (section 7 of p3_transformer_zk_test): root
    (P,Q), mid-layer messages, terminal mask claims all chi-sq < 400 over 12000 mask draws;
    with masks OFF the root Q collapses to 1 distinct value across draws and distinguishes
    two witnesses; with masks on both witnesses' laws are uniform (posterior flat).  16/16.

Effect: ~350 fewer committed columns per layer proof, no helper retention/regen in the
ledger, zk soundness smoke now rejects rsqrt/silu forges via "group multiset".

### 16.2 Levers 2+3 -- opening-reduction folding, merged sc5z blinds, fused GPU binds

  * p3_batchopen prove_class (resident classes): per-point G_t columns, eq columns, round
    messages and binds all run over STACKED (T x L) device buffers -- one launch per
    operation instead of one per (point, round); a CSR of ledger entries feeds one G-build
    kernel (was one axpy launch per entry, up to ~2300/class).  bo/red 0.22 -> 0.007 s,
    bo/G 0.13 -> 0.05 s at d=64.
  * sc5z Libra blinds: the 4 per-zero-check blind columns commit as ONE merged column
    (leaf j*NA+x = B_j[x]); the four published terminal evals yB[j] are bound by ONE opening
    at (r || tau0 || tau1) with tau drawn after the yB absorbs (degree-1 in each slice
    coordinate => SZ binds all four).  536 -> 134 blind commits, 402 fewer distinct opening
    columns, ONE ledger point per zero-check (T stays small in the fat classes).  HUGE
    domains (vfull+2 > 26) fall back to four separate commits (the merged codeword would be
    2^30 leaves, beyond the NTT/Merkle stack's proven range); the layout is tagged in bl.nb
    (4=merged / 5=separate) -- both layouts bind the yB evals to pre-rho commitments, so a
    dishonest tag costs nothing.
  * p3_scgpu sc_prove_gpu: per-round binds of all nc columns fused into one
    p3sg_bindn_kernel launch (was nc launches per round).
  * NOT DONE (scoped, remaining): bit-packing the small witness fields (sg/pr/eb/sh; q,r)
    with unpack lookups -- would cut the committed bytes of the fat matmul classes ~4x and
    attack commit_salt (0.64 s), q0-subset (0.55 s) and proof size together.

### 16.3 d=64 per-lever STAGES (zk=1, `8 64 2 32 128 1 1 tables_ld6.bin`, seconds)

  state                          prove   mm    lug   batch  proof_mb  verify
  section-15 baseline             4.18  1.26   1.17   1.36    41.5     0.55
  + L1 GKR-logUp                  4.05  1.27   1.26   1.08    38.2     0.55
  + L2 fused batch reduction      3.80  1.26   1.25   0.87    38.2     0.54
  + L3 merged blinds+fused binds  3.84  1.30   1.27   0.82    33.9     0.49
  (final, median of 3; non-zk: prove 1.46 -> 1.16 s, proof 13.6 -> 11.7 MB)

  Final d=64 ZK/non-ZK prove ratio: 3.84 / 1.16 = 3.3x (was 2.8x on a faster baseline --
  the ratio worsened slightly because non-zk gained MORE from GKR than zk did).
  Residual d=64 zk profile: commit_salt 0.64 s x2245, sc5z blind-gen 0.25 + host chains
  0.29, gkr scA 0.60, q0-subset 0.55, mask commits 0.37.

### 16.4 The latent >= d=128 zk rejection bug (FIXED) and first zk proofs at scale

Every honest zk layer proof at d >= 128 was REJECTED with "Dg sumcheck": the section-15 pass
added a GPU dispatch to sc5rz, but the Dg zero-check has 1 + NDG = 30 columns and the zk
path appends 4 Libra blinds = 34 > MAXC = 32, silently overflowing p3sg_msg_kernel's fixed
cur[32]/dd[32] register arrays.  At d=64 every Dg domain is < 2^14 (host chain), so the
entire battery was blind to it; the section-14.5 d=1024 zk run predates the sc5rz dispatch.
Diagnosed with a new env-gated transcript absorb trace (fs_transcript, P3_ABLOG: the
verifier's absorbed bytes MATCHED the prover through the failure point, isolating a prover
self-inconsistency, not a desync) and a host-vs-GPU A/B of the chain messages.  Fix: Msg5
GPU instantiations bumped to MAXC=40 plus explicit column-count guards on EVERY sc5 GPU
dispatch (host fallback).  Regression cover: d=128 and d=512 zk now prove AND verify.

Scale results (all verify_ok=1, host cap 41 GB):
  d=512  seq=8 zk:  prove 287.3 s, verify 1.1 s, proof  83.4 MB, RSS 28.4 GB  [FIRST ever]
  d=1024 seq=4 zk:  prove 487.7 s, verify 1.9 s, proof 134.4 MB, RSS 36.1 GB
         (section-14.5: 750.9 s / 5.6 s / 167.8 MB / 36.1 GB  => 1.54x prove, 2.9x verify)
  d=1024 seq=4 non-zk: prove 71.4 s, proof 42.5 MB, RSS 17.0 GB
         (section-14.5: 154.7 s / 47.5 MB / 18.6 GB => 2.2x prove)
  d=1024 zk/non-zk ratio: 6.8x.  STAGES zk: mm=123 lug=102 batch=257 (the batch phase's
  strCol/strG streaming classes now dominate at scale -- the fused stacked path applies only
  to resident classes; folding the streamed classes is the next opening-side lever).
What it took at d=1024 beyond 16.1-16.2: LU_GCAP 26 -> 25 (keeps mask-stream codewords
within the proven 2^28-leaf range), the sc5z separate-blind fallback (16.2), and a dynamic
strGdev budget (actual free device memory + the async mempool's reusable slack, 0.92 margin
-- the fat v=26 class grew to T=31 points with the GKR mask claims = 16.5 GB of G_t parking
that a fixed 12 GB budget pushed onto the host, breaching the cgroup cap).

Verification of the final state (this machine, this session): all batteries green --
hawkeye/matmul 35/35, rmsnorm 25/25, swiglu 16/16, quant 26/26, bfadd 22/22, rope 17/17,
softmax 19/19, logup 13/13, gkr 88/88, composed layer 30/30 with every chained intermediate
bitwise == transformer_layer.bin, full-layer hiding 16/16 (incl. the new GKR teeth), zk
soundness smoke 15/15; bench verify_ok=1 at d=64/128/512/1024 in both modes.

Residuals / known limitations:
  * sc5rz GPU dispatch below 2^14 is guarded OFF pending the small-domain FF-vs-host check
    (the MAXC fix makes 2^12..2^14 safe in principle; re-enable after an A/B sweep).
  * bit-packing (16.2) not implemented; commit_salt and q0-subset remain the d=64 residual.
  * the GKR devtree holds the whole tree on device (~8.6 GB at 2^28 leaves); a d=2048 zk
    point would need level-windowed rebuilds.
  * monolithic wiring (section 10) unchanged.

## 17. Binius (binary-tower) scoping note [REASONED -- design only, nothing built]

What it would replace.  The dominant committed data is per-product BIT-width witness:
fp8 codes (8 bits), sign/parity/exponent-bit/shift fields (1-5 bits), quotient/remainder
limbs (<= 10 bits) -- all embedded today as 64-bit Goldilocks elements and RS-encoded at
rate 1/4 into 64-byte-leaf Merkle trees.  In a binary-tower stack (Binius, Diamond-Posen
2023/1784), a column of b-bit values commits as b F_2 "bit-slices" (or one F_{2^b} column),
and the PCS cost scales with the ACTUAL bit content: the 2^26-element fp8-code column costs
8 x 2^26 BITS (~67 MB of field data as F_2, vs 512 MB as Goldilocks) -- a 8-64x committed-
data reduction depending on the column (est. ~10-20x across the matmul witness mix).  The
constraint side keeps working: sumchecks run with witness factors in tiny subfields and
challenges in F_{2^128} (small-by-large tower multiplication is cheap), and BOTH of this
codebase's load-bearing arguments port: logUp needs only field inverses (works over towers;
Tc+beta != 0 whp), and the GKR fractional-sum tree of section 16.1 is field-agnostic --
its commit-free structure would carry over unchanged.

What it requires (the honest bill):
  1. Tower field arithmetic F_2 .. F_{2^128} (Fan-Paar towers), host + CUDA.  GPU GF(2)
     arithmetic is bitwise (XOR/AND/CLMUL-style), not int64 muls: RTX-4090 LOP3/XOR
     throughput makes this feasible but it is a NEW kernel family, not a port.
  2. A binary PCS.  The modern route is FRI-Binius (ring-switching into the novel
     Lin-Chung-Han additive-NTT basis): additive NTT, ring-switching gadget, packed
     small-field leaves; roughly the scope of p3_ntt + p3_basefold + p3_fri + p3_merkle
     rewritten (~2-3 kloc of new math kernels plus proofs-of-equivalence tests).
  3. Small-field sumcheck plumbing: mixed-field columns, eq tables over F_{2^128}, and the
     Libra-style blinds/mask machinery re-derived over characteristic 2 (the mechanisms are
     field-generic, but every "uniform in F_p" hiding argument and chi-sq battery needs
     re-validation over tower elements).
  4. Table/lookup re-encoding: dyadic table domains stay; multiplicities cnt live in the
     big field (counts), fine.
  5. The nonlinear gadgets (rmsnorm rsqrt windows, softmax exp tables) are lookup-based and
     port; the bfadd/quantize arithmetic identities need re-derivation mod-2 (carry logic
     differs: today's base-2^k limb identities use integer adds in F_p -- in F_2^k towers,
     integer addition is NOT field addition, so range/carry gadgets must become explicit
     bit-decompositions... which towers make cheap, but the constraint set changes).
Item 5 is the subtle one: Goldilocks currently gives INTEGER arithmetic for free below 2^64;
binary towers do not.  The per-product dot-product accumulation (sum of q*2^s style terms)
would need a redesign around bit-sliced adders or a hybrid (keep a small-prime field for the
integer-sum side, Binius for the bit-heavy side -- at the cost of cross-field consistency
arguments, which is where most of the risk lives).

Expected win, where it lands: commit + Merkle + opening data ~10-20x on the matmul witness
(the current d=1024 zk profile spends ~75% of prove in commit/lookup/opening machinery whose
cost is proportional to committed bytes); sumcheck arithmetic roughly neutral; verifier
smaller.  End-to-end at d=1024 zk a 3-6x prove-time win is a defensible estimate, with the
integer-carry redesign (item 5) the main risk to both the estimate and the schedule.

Effort: the field+NTT+PCS core alone is several sessions of new code with its own selftest
pyramid before the first gadget moves; the integer-arithmetic redesign is a design-doc-first
effort.  Recommended smallest prototype: standalone tower-field lib + additive-NTT selftest,
then a bit-sliced commit of a real fp8-code column A/B'd against the Goldilocks commit for
size/time.  NOT built this session (honesty over completeness: levers 1-4 plus the latent
rejection-bug fix consumed it); nothing in the working tree depends on this section.

## 18. Multi-layer FULL FORWARD PASS with full ZK (2026-07-07) [VERIFIED]

The composed proof now covers an ENTIRE multi-layer model forward pass -- public token
ids in, public logits out, EVERY intermediate activation hidden -- as ONE proof over ONE
Fiat-Shamir transcript, ONE shared opening ledger, ONE model-level merged-lookup flush
and ONE batched-opening pass.

### 18.1 Canonical reference: TinyModel (transformer_ref.py)

`TinyModel` = embedding lookup (vocab=16, the committed table `emb`) -> N=2 chained
`TinyLayer`s (d=64, nh=2, dh=32, dff=128, seq=4; independent weights per layer, drawn
from one seeded rng stream) -> final RMSNorm (gain `gF`) -> LM head Hawkeye matmul
(`Wh`: vocab x d fp8 codes + fp32 row scales) -> (seq, vocab) bf16 logits.  Canonical
prompt `MODEL_IDS = [3, 1, 4, 15]`, seed 20260707; fully deterministic.  Dumps:
`--dump-model-weights transformer_model_weights.bin` ('TFMW') and `--dump-model-trace
transformer_model.bin` ('TFMT': ids + x0 + every `L<i>.<op>` intermediate + hF + logits).
The single-layer goldens are UNCHANGED (verified: `--dump-layer` / `--dump-weights`
md5-identical before/after the edit).

### 18.2 Prover composition (p3_transformer.cuh split + p3_model.cuh)

The single-layer prover/verifier was split into transcript-order-preserving sections:
`prove_subs` (header absorb + roots + all gadget sub-proofs on a caller-owned XCtx),
`prove_seams` (all composition seam claims; the public input/output binding claims are
now toggleable), and mirrors `verify_subs` / `verify_seams` on a caller-owned VCtx.
`p3tf::prove/verify` re-compose them bit-identically (composed layer battery stayed
30/30, zk smoke 15/15 after the refactor).  `commit_all` gained an `x0ext` parameter:
an already-committed input column is SHARED as rms1.X instead of committing fresh.

`p3_model.cuh` (namespace p3mdl) then chains everything:

* INTER-LAYER SEAM = ROOT EQUALITY.  Layer i's committed final-residual column
  (`res[1].OUT`) IS layer i+1's committed input column (`rms1.X`) -- the same `Col`
  object (values + zk mask + salt seed) is handed to both layers' gadgets, and the
  verifier checks `lay[i+1].rX0 == lay[i].rOut`.  Zero extra claims; no evaluation of
  any hand-off activation is ever revealed.  The head's rmsF.X shares the LAST layer's
  OUT column the same way.
* EMBEDDING = GATHER SEAMS at PUBLIC token ids.  E (vocab x d bf16 patterns) is
  committed once (secret values / public root, part of the model commitment like the
  weight roots).  For each token slot t: one hiding seam pair
  X0~(z_d, bits(t)) == E~(z_d, bits(id_t)) at a shared fresh ex-coordinate.  In zk,
  X0's mask slice 1 is the SAME gather of E's mask slice 1 (`mk_linked`), so the claim
  algebra holds slice-by-slice while every opened value is uniform.
* LM HEAD = final rmsnorm (existing gadget, gain gF) -> quantize -> Hawkeye matmul
  against the committed head matrix (restriction seam on the quantizer codes, identical
  to the in-layer pattern) -> ONE real-slice claim (clp) binds the committed logits
  column to the verifier's own MLE of the PUBLIC logits.  Non-zk additionally ships the
  head's public Y vector (checked bitwise against the statement); zk drops it.

Public statement: dims + nlayers + vocab, token ids, embedding root, per-layer weight
roots (7 matrices + g1/g2 each), gF root, head code+scale roots, pinned rope/canonical
tables, Q/R, and the LOGITS.  Everything between ids and logits is witness.

### 18.3 Results (RTX 4090, R=2 rate 1/4, Q=24) [VERIFIED]

    nvcc -arch=sm_89 -std=c++17 -O2 p3_model_test.cu   -> /root/p3_model_test
    nvcc -arch=sm_89 -std=c++17 -O2 zk_model_smoke.cu  -> /root/zk_model_smoke
    nvcc -arch=sm_89 -std=c++17 -O2 p3_model_zk_test.cu-> /root/p3_model_zk_test

* COMPOSED-MODEL battery 26/26 [VERIFIED]: honest accept with EVERY chained
  intermediate of EVERY layer + head bitwise == transformer_model.bin, layer i+1 input
  commitment IS layer i output commitment, logits bitwise == golden.  Non-zk: witness
  0.10 s, commits 0.02 s, prove 2.13 s, verify 0.43 s, proof 18.91 MB (12 model seams,
  15 batch classes).  Teeth (all reject at their own stage): embedding gather tamper,
  broken layer hand-off ("inter-layer chain root"), final-rms flip, head restriction
  seam, head matmul teleport, flipped logits claim, in-layer tampers in BOTH layers
  (rms1 flip / restriction seam / matmul teleport / rope slice in L0; softmax flip /
  concat seam / swiglu forge / residual flip in L1), statement tampers (logits, token
  ids, embedding root, L1 weight root, head weight root), proof-object tampers (chain
  root, embedding seam claim, batch opening, in-layer sumcheck message).
* ZK-MODEL-SMOKE 16/16 [VERIFIED]: with p3zkc::G.on, honest accept; commit 0.26 s,
  prove 5.65 s, verify 0.86 s, proof 59.86 MB.  ALL cleartext activation vectors
  dropped (every layer's per-matmul publics + the head's logits vector -- logits bound
  by the real-slice claim instead); 10 witness/seam tampers + 3 statement tampers all
  reject with the right reasons.
* FULL-MODEL-ZK-HIDING battery 11/11 [VERIFIED] (p3_model_zk_test, 12000 draws at fixed
  challenges, chi-sq<400 uniform / >5000 leak thresholds): (1) every NEW model column
  class uniform in every transcript quantity (embedding table, embedded input X0, the
  inter-layer hand-off L0.out==L1.in, L1 output, pre-head hF, head weight codes, deep
  L1 swiglu witness); (2) inter-layer hand-off claims uniform over mask draws, agree at
  the shared ex-coordinate; (3) embedding gather seam uniform + agrees under the mask
  linkage; (4) witness-recovery attack on the INTER-LAYER ACTIVATION: control (masks
  off) collapses to 1 distinct functional value /12000 (recoverable), hidden transcript
  12000/12000 distinct uniform = 0 bits extracted, posterior flat across different
  activations; (5) same attack on the EMBEDDING TABLE: control 1/12000, hidden
  12000/12000 = 0 bits.

Regression [VERIFIED, same session]: all 7 gadget selftests green (35/25/16/26/22/17/19),
GKR 88/88, logup 13/13, hawkeye-zk 19/19, zk-gadget-smoke 8/8, composed layer 30/30,
zk layer smoke 15/15, layer hiding 16/16.

Committed: hillclimb 627d89e (full tree).

## 19. Batch-opening drop-regen fix: zk 1.35x at d=1024 (2026-07-07) [VERIFIED]

### 19.1 Where the d=1024 zk time ACTUALLY goes (profile, this session)

`P3_ZKPROF=1 P3_MEMLOG=1 p3_transformer_bench 4 1024 16 64 4096 1 1 tables_ld10.bin`
(RTX 4090, 128 cores, `-Xcompiler -fopenmp`), baseline reproduced at prove=486.6 s
(section 16.5 recorded 487.7):

    mm=123.2  lug=102.3  batch=256.2      (of 486.6 s prove)
    batch: ONE class dominates -- tf-bo4, v=26 (512 MB columns), nc=95 distinct
    columns, k=128 claims, T=31 points, strCol=1 strGdev=1:
        G=42.4  ys+rlc=83.9  q0-subset=100.0  fold=1.7  red=0.2   (228 of 256 s)
    other: sc5z/blind=45.2  lug/hcommit=37.4  lug/scA=31.8  commit_salt=29.4
           lug/claims=22.6  mask_gen=20.8

NOT the folds: bo/fold is 3.4 s.  The section-16.5 guess that "streamed folding"
dominates was wrong -- the fused/strGdev reduction is already cheap.  The cost was in
the DROP-REGEN path: every use of a dropped (gen-backed) column called its regenerator,
which allocated a FRESH 512 MB std::vector (mmap + page-fault + free cycle, ~0.3 s per
use at v=26) -- and the G build materialized+uploaded once per ENTRY (k=128) instead of
once per distinct column (nc=95).  ~400 column-uses in tf-bo4 alone.

### 19.2 The lever: writer-style regenerators + column-major G build [VERIFIED]

* `PLedger::Gen` changed from `vector<gl_t>()` to `void(gl_t*, size_t)`: the ledger's
  `col_host` now keeps ONE per-class scratch allocation and regenerators WRITE into it
  (p3_hawkeye mblind regens via a new `p3zkc::blind_col_aug_into` -- the augmented blind
  column is one contiguous zprng chain; p3_logup GKR mask-stream regens fill in place).
  Zero per-use allocation; values bit-identical.
* G build in the `!strG || strGdev` modes runs COLUMN-major: each distinct column is
  materialized + uploaded exactly once and axpy'd into every G_t referencing it.  Only
  the accumulation order changes (exact field adds; identical sums, transcript,
  proof bytes).  The host-parked strG fallback keeps the point-major build.

Results (same commands, all verify_ok=1, proof sizes byte-identical):

    d=1024 zk: prove 486.6 -> 359.3 s (1.35x)   batch 256.2 -> 127.8 s
               bo/G 46.0 -> 13.4   bo/ys+rlc 85.9 -> 23.1   bo/q0 113.3 -> 80.1
    d=512  zk: prove 239.4 -> 186.0 s (1.29x)   batch 124.6 -> 70.9 s
    d=64   zk: 3.80 s (unchanged -- small classes never drop columns)
    zk model (2 layers + head, d=64): prove 4.88 s (was 5.65 pre-session)

Regression: ALL batteries green after the change (7 gadget selftests, GKR 88/88, logup
13/13, hawkeye-zk 19/19, zk-gadget-smoke 8/8, composed layer 30/30 + zk smoke 15/15 +
hiding 16/16, composed MODEL 26/26 + zk smoke 16/16 + hiding 11/11).

### 19.3 Failed experiments (measured, reverted) [VERIFIED]

Two "obvious" wins were tried first and made things WORSE in situ; both reverted:
* OpenMP jump-ahead parallel fill for the sequential zprng chains (LCG skip-ahead,
  bit-identical, 3.6x faster in a microbenchmark): d=512 zk prove 239 -> 260 s.  The
  128-thread fork/join cost (~10 ms on this box) at ~3k call sites plus NUMA-remote
  first-touch outweighed the fill itself (serial fill of 2^25 values is only 47 ms).
* Pinned-staging uploads (parallel memcpy into a cudaHostAlloc buffer + async H2D):
  d=512 zk prove -> 290 s.  Pageable H2D already runs at ~19.7 GB/s on this box (the
  slow path was NEVER PCIe -- it was the allocation churn of 19.2); the extra CPU read
  pass only added work.
Microbenchmarks that guided the revert: serial fill 2^25 = 47 ms, parallel = 13 ms;
pageable H2D 512 MB = 27 ms (19.7 GB/s), cudaHostRegister'd = 25 ms (21 GB/s).

### 19.4 Residual profile + next levers [REASONED, scoped only]

After the fix at d=1024 zk (359.3 s): mm=123 (matmul zero-checks/sumchecks), lug=103
(hcommit 37 + scA 32 + claims 23), q0-subset=80, sc5z/blind=45, mask_gen=21.
* q0-subset (80 s): per-column NTT re-encode + full salted-tree SHA rebuild (M0=2^28)
  for 95 columns, to open Q=24 positions.  Structural fix: retain the tree's TOP levels
  at commit time (level >= 13: 256 KB/column) and rebuild only the query-touched
  subtree chunks at opening (~48 x 2^13 leaves vs 2^28) -- ~300x less SHA, est. 80 ->
  ~15 s.  Touches commit_col_nc/Col/subset_prove plumbing; not attempted this session.
* sc5z/blind (45 s) + mask_gen (21 s): same alloc-churn class as 19.2 (blind_col_aug
  allocates c + mask + augment per commit; commit_col_nc's augment copies again) --
  writer-style in-place construction est. -20 to -30 s combined.
* GPU-side regen: extend the ledger with a TYPED gen descriptor (chain seeds/segments)
  so gen-backed columns fill directly on device (zprng chain is jump-ahead capable in a
  kernel), eliminating host materialization + upload entirely for dropped columns.
* Bit-packing small witness fields (section 16.2) remains unbuilt; it shrinks committed
  data (mm + commit_salt side), orthogonal to the opening side.

### 19.5 Second lever landed: in-place blind/mask construction [VERIFIED]

The 19.4-scoped alloc-churn fix for the COMMIT side: `blind_col_aug` builds the
augmented blind in ONE pass (the augmented column is one contiguous zprng chain --
no separate c/mask vectors + augment copy), and `commit_col_nc`'s fresh-mask path
augments IN PLACE (`fresh_mask_into` fills the mask region of the moved vals vector
from the same next_seed() chain; the linked-zkmask path is unchanged).  Byte-identical
columns, roots and transcripts (non-pow2 vals keep the mask region at vals.size(),
exactly like augment).

    d=1024 zk: prove 359.3 -> 343.1 s   (sc5z/blind 45.2 -> 35.0, mask_gen 20.8 ->
               16.2, mm 123 -> 108; cumulative session total 486.6 -> 343.1 = 1.42x)
    ALL 17 batteries green after the change (composed layer 30/30, model 26/26,
    zk smokes 15/15 + 16/16, hiding 16/16 + 11/11 + 19/19, 7 gadget selftests,
    logup 13/13, GKR 88/88, zk-gadget-smoke 8/8).
