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
