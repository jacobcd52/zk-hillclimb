# FP8-EXACT PLAN — an efficient ZKP of the exact fp8 forward pass of one FC layer

**Author:** Fable (Claude) — design/plan only, no prover code written yet.
**Date:** 2026-07-02
**Builds on:** `/workspace/projects/zk-hillclimb/FABLE_REPORT.md` §9 (Designs 1–3), the p3 prover in
`/root/zkllm/p3_*.cuh`, and the Hawkeye/codebook work in `/workspace/projects/int-model-approximation`.
**Grounding code (ran today):** `/root/fable-eval/fp8_ref_experiments.py`, `/root/fable-eval/hawkeye_vs_ada.py`.

> **[VERIFIED]** = I ran it today on the RTX 4090 (torch 2.4.1+cu124). **[REASONED]** = analysis.

---

## 0. Executive summary

**Recommendation (framing decision 1): option (a) — a canonical, deterministic fp8 kernel WE
specify, that serving commits to running.** Call it **CanonFC**. It is parameterized by one dial,
the block size `K0`:

- fp8 E4M3 decode → **exact integers** on the 2⁻⁹ grid (18-bit; 254-entry lookup),
- **exact integer** multiply-accumulate within each K0-block of the reduction dim (this is the
  existing p3 sumcheck matmul — block partials are ≤ 39-bit integers, trivially inside Goldilocks),
- **fp32 RNE** combine across blocks in a pinned order (per block: one RNE int→fp32 convert +
  one RNE fp32 add, both native hardware ops — so the serving kernel is a trivial Triton loop),
- final scale-mul + optional bf16 downcast (RNE, small gadgets).

`K0 = 32` gives MXFP8-flavored "fp32-lossy" semantics at estimated **~3–4.5×** the integer proof;
`K0 = 128` gives **~1.6–2×**; `K0 = K` (full Kulisch, single rounding per output) gives **~1.1–1.2×**
and is *more* accurate and order-free. **My recommendation: build the gadgets once (they are the
same at every K0), validate at K0 = 32 on a tiny layer, and let you pick the production K0 —
I argue below for K0 = K or K0 = 128.**

Why not the alternatives: **[VERIFIED]** stock Ada `torch._scaled_mm` is *not* fp32-RNE
accumulation at all (0.01% bitwise match against every fp32-order I tried, mean error ~2.5e-3
relative) — its internal accumulator is lossy/limited-width like Hopper's; and the existing Hopper
Hawkeye model matches Ada only on **86.5%** of outputs bitwise. So option (b) "reverse-engineer
stock" *is* option (c) "hardware replay", it would be a new Ada-Hawkeye reverse-engineering
project, tied to SM89, batch/version-fragile, and its truncating-alignment semantics need strictly
more gadgets than CanonFC. Option (a) closes the covert channel **by construction**: the proof pins
the bitwise output of a deterministic function of (X, W), so the server has zero output freedom.
The honest caveat: this requires the deployment mandate "serving runs CanonFC" — see Open
Question 1.

The single dominant cost is the **RNE-add witness data** (§4.4); every efficiency lever we have
attacks it: K0 (linear), bit-packing (~2×), commit-free lookups via GKR-logup (~1.5× at scale),
one batched aux opening, and optionally a fused convert-add (−35%).

---

## 1. What I verified today (grounding runs)

All in `/root/fable-eval/fp8_ref_experiments.py` and `hawkeye_vs_ada.py`, RTX 4090.

1. **Decode is integer.** [VERIFIED] Every one of the 254 non-NaN E4M3 codes equals `v·2⁻⁹` for a
   signed integer `v`, `|v| ≤ 229376` (18 bits). Table matches `torch.float8_e4m3fn` bit-for-bit.
   (Two zeros collapse to 0; codes 0x7F/0xFF are NaN and excluded.)
2. **A pure-integer model of the full canonical kernel is bitwise-exact vs hardware fp32.**
   [VERIFIED] I implemented RNE int→fp32 convert and RNE fp32 add in pure integer arithmetic
   (guard/round/sticky, G=3 guard bits) — exactly the operations the ZK gadgets will constrain —
   and validated: 2,000,000 random fp32 adds, 200,000 exact-tie adds, 1,000,000 converts of
   42-bit ints, **zero mismatches**; then the end-to-end kernel (B=4, K=128, N=32) at
   K0 ∈ {32, 64, 128}: **100% bitwise equal** to the numpy/hardware-fp32 implementation.
   This *is* the circuit spec, pre-validated against silicon.
3. **Within-block sums are tiny integers.** [VERIFIED] On the llama-68m up_proj shape
   (B=16, K=768, N=3072, random codes): max block-partial width **37 bits** at K0=32, **39 bits**
   at K0=K=768. Bound: `18+18+⌈log₂K0⌉` bits — Goldilocks holds exact inner products up to
   K ≈ 2²⁷ with no intermediate rounding at all.
4. **Order matters, so pinning it is the whole game.** [VERIFIED] Permuting the block order of the
   canonical kernel changes **77%** of outputs bitwise (random data; heavy cancellation inflates
   this, but nonzero on real data too). Different K0 (32 vs 64 vs 128 vs Kulisch) agree on only
   ~28% of outputs. This is the same nondeterminism the covert channel rides on.
5. **Stock Ada kernel is not fp32-RNE and not Hopper-Hawkeye.** [VERIFIED]
   `torch._scaled_mm` vs canonical-K0=32: 0.01% bitwise; vs exact Kulisch: 0.01%; mean ulp
   distance ~10⁵ (random data). Hopper Hawkeye replay vs Ada `_scaled_mm` (bf16 out): **86.5%**
   bitwise match, mean relative diff 2.5e-3 — same family (limited-width truncating alignment),
   different parameters/order. Also: at these shapes `_scaled_mm` was run-to-run deterministic
   *and* batch-slice invariant (B=1 row == B=16 row bitwise) — the channel is not per-run noise
   here; it is the server's *freedom to choose* among kernels/orders/shapes that all pass an
   approximate check. A bitwise proof removes that freedom.
6. **No subnormals, no overflow, ever, inside CanonFC.** [REASONED, tight] Every value in the
   accumulation chain is a multiple of 2⁻¹⁸ (inductively preserved by RNE, since rounding only
   discards low bits and values below 2⁵ are already exact), so the smallest nonzero magnitude is
   2⁻¹⁸ ≫ 2⁻¹²⁶; and `|acc| ≤ K·448² < 2³⁰` for K ≤ 4096 ≪ 2¹²⁸. The fp32 gadgets therefore
   need **no subnormal path and no inf/overflow path**; exponents fit in ~6 bits
   (e ∈ [−18, 31]). This is a large simplification unavailable to a general fp32 verifier and is
   why our gadget is cheap.

---

## 2. Framing decision 1 — WHAT computation to prove: canonical kernel (a)

**Definition of "exact" adopted:** *served == proven, bit for bit*, against a pinned, published
reference function `Y = CanonFC(X, W, s_x, s_w; K0, order)`. Exactness is only meaningful relative
to a pinned reference; the three options differ in *which* reference:

| option | reference | verdict |
|---|---|---|
| (a) CanonFC, deterministic kernel we publish | our spec §3 | **RECOMMENDED** — provable at 1.1–4×, buildable this month, closes channel by construction, hardware-portable |
| (b) stock cuBLASLt / `torch._scaled_mm` order | undocumented | **rejected** — [VERIFIED] Ada's kernel is not any fp32-RNE order (0.01% match): reverse-engineering it = building Ada-Hawkeye (option c), and it can change under our feet with cuBLAS version/shape heuristics |
| (c) Hawkeye-style exact hardware replay | Hopper QGMMA model | **defer** — [VERIFIED] Hopper model ≠ Ada (86.5%); replaying the 14-bit truncating alignment needs *more* gadget work than CanonFC (per-group max-exp, truncation, renormalize); the in-repo class-counts construction is exact but ~208 s/forward on a 0.5B model. Revisit only if "unmodified stock serving" becomes a hard requirement |

**The honesty clause (as you asked):** if the datacenter serves anything other than the proven
CanonFC — including "mostly CanonFC but a different kernel under load" — the near-tie freedom and
hence the covert channel **reopens**. Option (a) is only sound together with the deployment
mandate *serving runs the canonical kernel*. Mitigations that make the mandate cheap: CanonFC is
batch-invariant by construction (each output element depends only on its own row/column, blocks
combined in fixed K-order — no split-K, no batch-shape dependence), so one kernel binary serves
all shapes; and any deviation is caught by the proof failing bitwise, not by a statistical test.
Serving performance is a real cost and is quantified in §7/OQ4.

**Why (a) also dominates on proof cost:** real fp8 tensor-core hardware is lossy *within* a K=32
group (Hawkeye: products aligned to the group max exponent by truncating right-shift into a
14-bit-fraction accumulator). Proving that faithfully costs gadgets per *product*. CanonFC makes
the within-block sum exact-integer — free, because it is precisely the sumcheck matmul we already
have — and pays gadgets only per *block boundary*: `K/K0` per output instead of `K`.

---

## 3. The precise reference computation: **CanonFC-v1**

Inputs: `X` — B×K matrix of E4M3 codes; `W` — N×K matrix of E4M3 codes (row-major per output
channel); public per-tensor scales `s_x, s_w` (fp32); parameters `K0 | K` (pad K to a multiple of
K0 with zero codes; zero blocks are exact no-ops).

```
DEC[c]  : 254-entry table, E4M3 code -> signed 18-bit integer v, value = v * 2^-9.
          Codes 0x7F, 0xFF (NaN) are NOT in the table (proof unsatisfiable if present).

for each (i, k):                                   # independent per output element
  acc = +0.0f                                      # fp32
  for b = 0 .. K/K0 - 1:                           # ORDER PINNED: ascending b, sequential
      S_b  = sum_{j in block b} DEC[X[i,j]] * DEC[W[k,j]]     # EXACT integer, |S_b| < 2^(36+log2 K0)
      r_b  = RNE_fp32(S_b)  * 2^-18                # int->fp32 convert (1 rounding; 2^-18 scale exact)
      acc  = RNE_fp32(acc + r_b)                   # fp32 add (1 rounding)
  y32[i,k] = RNE_fp32(acc * RNE_fp32(s_x * s_w))   # scale (phase 2; =identity when scales are 1)
  Y[i,k]   = y32[i,k]            (v1)              # or RNE bf16 downcast (phase 2)
```

Normative details (all [VERIFIED] to match hardware fp32 via the pure-integer model):
- **RNE** = IEEE-754 round-to-nearest, ties-to-even, on a 24-bit significand.
- Convert and add are **two separate roundings** (both native ops → the serving kernel is plain
  Triton: int32/int64 block dot + `float32` adds; no exotic instructions). A fused single-rounding
  accumulate (`acc = RNE(acc + S_b·2⁻¹⁸)` exactly) would save ~35% of gadgets but forces an
  integer-emulated serving kernel — offered as CanonFC-v1f, not recommended (OQ2).
- No subnormals / no overflow / no NaN can occur (§1.6); the circuit rejects NaN codes via the
  decode table and needs no special cases.
- `+0.0` initial accumulator; an exact-zero sum is `+0` (sign of zero is pinned).
- Accumulation order across blocks is ascending `b`; there is **no** batch/tile/split-K dependence.

**Ground truth generator:** `canonical_int()` in `/root/fable-eval/fp8_ref_experiments.py` — the
pure-integer implementation, already validated 100% bitwise against numpy/hardware fp32 end-to-end.
A Triton twin of the same spec is the (later) serving kernel; CI asserts kernel == reference
bitwise on random + adversarial (tie-mined) layers.

---

## 4. Proof decomposition, gadgets, and costs

Substrate: existing p3 stack — Goldilocks, Basefold/FRI PCS, ZK-sumcheck, mask-slice augmentation.
One genuinely new infrastructure piece: **a p3-native lookup argument (logup)** — today lookups
exist only in the curve-based `zkob_lookup.cuh`; the logup *structure* ports, the commitment
backend must become Basefold. Tables needed: `T_dec` (254 rows: decoded values), `T_16` (2¹⁶ range),
`T_pow2` (64 rows: i → 2^i).

Committed vectors (all mask-slice-augmented ×2 like the operands, for the eventual ZK version):

| id | vector | length (per layer) | width |
|---|---|---|---|
| C1 | `X_int`, `W_int` decoded operands | B·K + K·N | 18-bit signed (as field elts) |
| C2 | `S` block partials | B·N·(K/K0) | ≤ 36+log₂K0 bits |
| C3 | convert witness: `a=|S|`, `m`, `rem`, packed{s, sh, case, round, lsb, tie-sign} | 4 · B·N·(K/K0) | mixed, packed |
| C4 | acc chain `(packed s,e)` + `m` per step | 2 · B·N·(K/K0) | e ∈ [−18,31] |
| C5 | add witness: `q` (aligned+sticky), signed sum `a'`, `rem'`, packed{d, w, flags} | 4 · B·N·(K/K0 −1) | mixed, packed |
| C6 | scale/downcast witness (phase 2) | ~2 · B·N | small |

Constraint layers (each a batched sumcheck over all instances — uniform relations, no branching):

**G1 — decode membership.** Every entry of `X_int`, `W_int` ∈ `T_dec` (logup). This binds operands
to *valid E4M3 values*; raw codes never enter the proof (decode is a bijection on values, ±0
collapse is harmless). The published weight commitment is over `W_int` (OQ5).
Cost: 2 lookup columns over B·K + K·N rows; with GKR-logup ~zero committed data, with v1
helper-column logup ≈ +1 committed elt/row.

**G2 — block partials.** `S~(r_i, r_k, r_b) = Σ_{b,j0} eq(r_b, b) · X~(r_i, b, j0) · W~(b, j0, r_k)`
— a degree-3 sumcheck over the same `log K`-variable cube as today's matmul sumcheck (b-bits +
j0-bits), i.e. **the existing p3_matmul/zkmatmul machinery with an eq factor and the b-half of the
j-variables left free**. Same cost class as the current integer matmul sumcheck. This is where
"within-block integer MAC reuses the existing sumcheck" lands concretely.

**G3 — RNE convert** (`S_b → r_b`): constraints `S = (1−2s)·a`; `a = m'·2^sh + rem` with
`m' ∈ [2²³, 2²⁴)`, `rem < 2^sh` (via `rem · 2^(42−sh)` range check + `T_pow2`); rounding decision
from `t = 2·rem − 2^sh` (sign + tie→lsb, ties-to-even); `m = m' + round`, carry renormalize
(`m = 2²⁴ → m/2, e+1`); small-`|S|` case (`w ≤ 24`) handled by the same relations with `sh`
selector. ≈ 4 committed elts + ~8 lookup rows per instance. **The integer algorithm being
constrained is literally `int_to_fp32()` in the experiment file — already bitwise-validated.**

**G4 — RNE fp32 add** (`acc' = RNE(acc + r)`): big/small selection bit (compare packed (e,m));
`d = e_big − e_small` (6-bit); aligned small significand `q = (m_small·2³) >> d` **plus sticky OR**
(prove: `m_small·2³ = q·2^d + rem'`, `rem' < 2^d`, sticky = [rem' ≠ 0] via an inverse witness);
signed sum `a' = ±m_big·2³ ± q`; leading-bit width `w` (via `T_pow2` bracket
`2^(w−1) ≤ a' < 2^w`); normalize + RNE exactly as G3; `d > 27 ⇒ q = sticky` (same relations,
clamped by selector). ≈ 6 committed elts + ~10 lookup rows per instance. Constrained algorithm =
validated `fp32_add()` (G=3 guard bits; 2M random + 200k tie cases, zero mismatches).

**G5 — scale mul + bf16 downcast** (phase 2): fp32 multiply = 24×24→48-bit integer product +
the same normalize/RNE tail as G3 (≈ 3 elts + 6 lookups per output); bf16 downcast = drop 16
mantissa bits with RNE (≈ 1 elt + 2 lookups). Only B·N instances — negligible.

**G6 — output binding.** `Y` is committed (and, for the last layer, opened against public values)
as packed fp32 bit patterns; a linear relation ties `(s, e, m)` of the final acc to the IEEE bits
`s·2³¹ + (e+127)·2²³ + (m − 2²³)`. Zero extra cost class.

All aux vectors (C2–C6) are combined by random linear combination into **one** batched Basefold
opening (the p3 stack currently opens each operand separately; batching is §7-item-2 of my report
and is assumed here — without it, add ~4 openings ≈ +120 ms and +2.8 MB, which would dominate).

### 4.4 Cost table and total overhead — llama-68m up_proj (B=16, K=768, N=3072) [REASONED from VERIFIED counts]

Baseline integer proof (measured in FABLE_REPORT §6): commit 2·(B·K + K·N + B·N) ≈ **4.84 M**
field elts (augmented), prove **303 ms**, verify 34 ms, proof 2.47 MB (R=1, Q=64).

Gadget instance counts [VERIFIED by direct count in E6]: `cvt = B·N·K/K0`, `add = B·N·(K/K0 − 1)`.

| K0 | cvt / add instances | extra committed elts (opt…cons) | est. total overhead vs integer proof |
|---|---|---|---|
| 32 (MX-faithful) | 1.18 M / 1.13 M | 8.1 M … 15.0 M | **×2.7 … ×4.1** |
| 64 | 590 k / 541 k | 3.9 M … 7.4 M | ×1.8 … ×2.5 |
| 128 | 295 k / 246 k | 1.9 M … 3.5 M | **×1.4 … ×1.7** |
| 256 | 147 k / 98 k | 0.8 M … 1.5 M | ×1.2 … ×1.3 |
| 768 = K (Kulisch) | 49 k / 0 | 0.15 M … 0.35 M | **×1.03 … ×1.1** |

"opt" = packed witnesses + GKR-logup (lookups commit-free); "cons" = naive one-elt-per-field +
helper-column logup. Prove time scales ≈ with committed data (Merkle+NTT dominate; measured
§6/§7 of my report), plus one extra batched opening (~+50–80 ms, ~+0.8 MB at Q=64); sumcheck
rounds grow by the aux cube sizes (small vs hashing). So at K0=32 expect **prove ≈ 0.9–1.4 s**
vs 303 ms; at K0=128 ≈ **0.5–0.6 s**; Kulisch ≈ **0.35–0.4 s**. Verify: +10–20 ms; proof:
+0.8–1.5 MB (one aux opening + longer sumcheck transcripts).

**Dominant cost: the RNE-add witness (C4+C5 ≈ 60% of extra data at K0=32).** Attacks, in order of
leverage: (1) **K0** — linear, and semantically free since *we* define the kernel; (2) **packing**
— acc as (packed s|e, m) 2 elts, flags+small fields into one elt (already assumed in "opt");
(3) **GKR-logup** — removes ~10 committed lookup-helper elts per gadget, the difference between
"cons" and "opt"; (4) **fused convert-add** (CanonFC-v1f) — 1 gadget per block instead of 2,
−35%, at the price of an integer-emulated serving kernel; (5) **Binius** — only if we ever need
Design-3-style per-product rounding (§6).

Comparison to my §9 estimates: Design 2's "~2–4×" is confirmed at K0=32 (×2.7–4.1); Design 1's
"~1×" is confirmed as ×1.03–1.1; Design 3 (per-add, ×10–50) remains excluded — CanonFC never
needs it.

---

## 5. Efficiency strategy — why this is cheap, quantified

1. **Block structure kills the rounding count.** K→K/K0 expensive ops per output: at K0=32,
   **24** rounding pairs per output instead of **768** per-add roundings (32× fewer; 1.18 M
   instead of 37.7 M gadget instances for this layer [VERIFIED count]). This is the single
   biggest lever and it is exact, not approximate, because within-block sums of decoded E4M3
   products are integers of ≤ 39 bits [VERIFIED] — Goldilocks native.
2. **The matmul stays a sumcheck.** G2 is the existing matmul argument with an eq-factor; no GEMM
   ever enters a lookup/bit circuit.
3. **Lookups only where table-shaped:** decode (254), 16-bit ranges, powers of two. GKR-logup
   makes their marginal committed cost ~0; the fallback helper-column logup is acceptable for the
   tiny build.
4. **One batched aux opening** (RLC) instead of per-vector openings — without this the PCS tax
   would eat the gadget savings.
5. **No-subnormal/no-overflow regime** (§1.6) deletes the expensive halves of a general IEEE-754
   gadget (subnormal renormalize, overflow/inf/NaN propagation): our add is ~6 elts vs ~15+ for a
   general fp32 add.

---

## 6. Substrate decision

**Stay on Goldilocks + p3 Basefold + logup for the first build.** Everything reusable is here:
commit/opening (`p3_basefold.cuh`), ZK sumcheck (`p3_zksumcheck.cuh`), matmul (`p3_matmul.cuh`,
`p3_zkmatmul.cuh`), FS transcript, test-battery style. The bit-level work (shifts, GRS bits) is
limb-decomposition + small lookups — standard Goldilocks-STARK practice, and our fp32 regime is
unusually benign (§1.6). Port logup's *logic* from `zkob_lookup.cuh` but re-base it on Basefold
commitments (the curve code stays untouched).

**When Binius pays off** [REASONED]: if the per-instance witness must become bit-granular — i.e.
(i) Design-3 per-product RNE (37.7 M instances — 32× our count) or (ii) exact replay of
truncating-alignment hardware (option c), where per-product shifts dominate. Binary towers make
mantissa/exponent bit-ops native instead of lookup-emulated, plausibly 3–10× cheaper *for those
circuits*. At K/K0 instance counts the gadget data is only ~1–3× the operands, so the substrate
switch (new PCS, new field, no reuse) cannot pay for itself. Revisit at Design-3 scale only.

---

## 7. First build target + validation plan

**Target: `tiny1` — a single FC layer, B=4, K=64, N=64, K0=32, scales=1, output=fp32 bits.**
(All dims powers of two — p3 requires it; K/K0 = 2 blocks → per output exactly 2 converts + 1 add,
the minimal complete exercise of every gadget. 256 outputs → 512 convert + 256 add instances;
committed extra ≈ 5 k vs base 9.2 k — seconds to prove even on the host path.)
Then immediately the measurement target: **llama-68m up_proj shape** (768→3072 padded to
1024→... as the current llama driver already does, B=16) at K0 ∈ {32, 128, K} to fill in the real
column of the §4.4 table against the measured 303 ms baseline.

**Ground truth:** `canonical_int()` (pure integer, [VERIFIED] == hardware fp32 bitwise) generates
`(X codes, W codes, Y fp32 bit patterns)` golden files; a second independent path (numpy fp32
`canonical_np()`) must agree bitwise before a vector is accepted — dual-implementation gate, both
already written and agreeing 100% today.

**Bitwise check:** verifier accepts iff the committed `Y` opens to *exactly* the golden bit
patterns (for tiny1, Y is public: bind the commitment to the published bits, per report §5's F4
binding — this doubles as the first F4 fix).

**Test battery (teeth, in the style of the existing p3 selftests):**
1. honest accept (100 random layers, plus tie-mined layers: inputs constructed so some step hits
   an exact half-ulp tie — assert the even-mantissa result is the only accepted one);
2. **±1-ulp tamper**: flip the last mantissa bit of one output → must reject (THE
   channel-closure test — this is exactly the freedom the integer-model proof tolerated);
3. wrong rounding direction at a tie (round-to-odd) → reject;
4. block-order swap (recompute honestly with blocks 0,1 swapped) → reject;
5. NaN code in X or W → unsatisfiable (decode lookup fails);
6. tampered `S_b` / convert witness / add witness → reject;
7. determinism: two proofs of the same instance yield identical `Y` binding;
8. (once ZK'd) mask-active checks per the existing trichotomy harnesses.

**Serving-kernel cross-check (phase 2):** Triton CanonFC == reference bitwise on 100 random +
adversarial layers at production shapes; this kernel is what the datacenter mandate points to.

---

## 8. Implementation roadmap (prioritized)

1. **p3_logup.cuh** — logup over Goldilocks/Basefold: fixed tables (T_dec/T_16/T_pow2),
   multiplicity commit, v1 helper-column form + selftest with negative controls. *(new infra,
   everything else depends on it)*
2. **p3_fp32_gadgets.cuh** — G3 convert + G4 add as batched sumcheck relations over committed
   columns; selftest driven by the validated Python bit-model's vectors (random + ties + d>27 +
   cancellation-to-zero cases).
3. **G2 block-partial sumcheck** — extend `p3_matmul.cuh` with the eq-weighted degree-3 variant;
   selftest vs direct evaluation.
4. **p3_fp8fc.cuh (v0, non-ZK)** — compose G1–G4+G6 for tiny1; golden-vector battery of §7;
   this is the *served == proven* demonstrator.
5. **Measure** on llama-68m up_proj at K0 ∈ {32, 128, K}; fill the real overhead table; pick
   production K0 with you.
6. **Batched aux opening (RLC)** — fold C2–C6 into one opening (shared with report §7-item-2 work).
7. **ZK pass** — mask-slice the aux vectors; note this *depends on the F2 opening fix* from my
   report §3 (without it the fp8 witness leaks exactly like the integer one; same fix, same
   machinery — do not double-build).
8. **Phase 2:** scales + bf16 downcast (G5); Triton serving kernel + bitwise CI; GKR-logup
   upgrade; fold into the multi-layer batching story.

Rough effort [REASONED]: items 1–4 are each ~1–3 focused days against the existing test-harness
patterns; item 5 is hours once 4 runs; 6–7 are shared with already-planned report fixes.

---

## 9. Open questions / decisions for you (coordinator) before coding

1. **Confirm option (a) + the deployment mandate** ("serving runs published CanonFC-v1"). This is
   the load-bearing assumption; if unmodified-stock-kernel serving is a hard requirement instead,
   the plan changes to an Ada-Hawkeye reverse-engineering project first (weeks, hardware-tied) —
   say so now.
2. **Pick K0 semantics** (the cost dial): K0=32 "fp32-lossy faithful" at ×2.7–4.1, K0=128 at
   ×1.4–1.7, or full Kulisch at ×1.05–1.1 (most accurate, order-free, and my recommendation
   unless "looks like standard fp8 serving numerics" is itself a requirement). Also: fused
   convert-add variant (v1f, −35% gadgets, integer-emulated serving kernel) — I recommend **no**.
3. **First build non-ZK?** I plan soundness/exactness first (v0 without mask slices), ZK as
   roadmap item 7 gated on the F2 opening fix. Confirm that ordering, and that the F2 fix is
   planned as shared work rather than duplicated here.
4. **Serving-kernel workstream ownership + perf budget.** For production GEMM speed the
   within-block integer dot can ride int8 tensor cores via 3-limb decomposition (9 int8 GEMMs ≈
   comparable-throughput-class to fp8, [REASONED, unmeasured]) or plain Triton CUDA-core int
   (slower). Who builds/benchmarks this, and what slowdown vs stock fp8 is acceptable?
5. **Weight commitment format:** publish the commitment over decoded `W_int` (my preference;
   decode-membership then constrains only X) or over raw fp8 bytes (needs a (code,value) pair
   lookup and a byte-packing layer)?
6. **Scales:** confirm per-tensor, public fp32 scales (as in the RedHatAI-style checkpoints);
   per-row/per-token scales change G5 from constant-mul to committed-mul (slightly bigger, still
   B·N-sized). And output dtype for the real target: fp32 or bf16?
7. **Padding rule:** K padded to K0 multiples with zero codes (zero blocks are exact no-ops) —
   confirm, and confirm the non-pow2-K handling should follow whatever the current llama driver
   does for 768.

---

## Appendix: file map

- `/root/fable-eval/fp8_ref_experiments.py` — decode table (E1), pure-integer RNE convert/add
  models + hardware validation (E2), block widths (E3), order/ulp comparisons (E4), stock-kernel
  determinism probes (E5), gadget counts (E6). All results quoted in §1/§4 are reproducible by
  running it (~4 min, needs the 4090 for the `_scaled_mm` section).
- `/root/fable-eval/hawkeye_vs_ada.py` — Hopper Hawkeye model vs Ada `_scaled_mm` (86.5%).
- `/workspace/projects/int-model-approximation/src/int_model_approximation/hawkeye.py` — the
  Hopper QGMMA integer-replay semantics (read-only; evidence for §2's option-(c) analysis).
- `/root/zkllm/p3_matmul.cuh`, `p3_private_fc.cuh`, `p3_zksumcheck.cuh`, `p3_basefold.cuh` — the
  machinery G2/G6 extend; `zkob_lookup.cuh` — logup logic to port (not the commitment backend).
