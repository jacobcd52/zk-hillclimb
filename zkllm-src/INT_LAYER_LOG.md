# INT_LAYER_LOG — composed integer-layer ZKP on the Goldilocks/hash substrate

Working log, newest entries at the bottom.  Deliverable spec: int_layer_prompt.md.

## 2026-07-15 session 1 — design locked after substrate read

Read: COMPARISON_AUDIT.md (INT_BASELINE.md does not exist on disk; the audit doc
carries the context), p3_transformer.cuh (fp8 composed template), p3_matmul.cuh,
p3_logup.cuh, p3_basefold.cuh (via usage), p3_quant.cuh (gadget template),
p3_hawkeye.cuh (sc5z/claimc machinery), p3_zkc.cuh (mechanisms 1–4),
p3_zkopen/p3_zk/p3_zksumcheck/p3_zkmatmul, p3_transformer_bench.cu,
p3_transformer_zk_test.cu (hiding battery), zkob_{rescale,rmsnorm,softmax8,
glu,rope}.cu headers (integer semantics).

### Semantics (int_layer_ref.py is the normative reference)

Fixed-point llama block, residual scale 2^16 (zkob-style), all values Goldilocks
field elements representing signed ints; ranges enforced by logUp range lookups
so no in-field wraparound is adversarially reachable:

- activations |a| < 2^19 (±8 real), enforced at every rescale output (RS20
  signed range table);
- weights at scale 2^16, |w| < 2^19 (same table);
- matmul accumulator scale 2^32, |acc| < 2^35 honest (in-field bound 2^48 ≪ p);
- rescale y = floor((x + 2^(sf-1))/2^sf): x + 2^(sf-1) = y*2^sf + rem,
  rem ∈ [0,2^sf) via 16-bit limb lookups (exact tables for non-16 widths);
- RMSNorm: M = Σx² + CEPS (CEPS = 2^14), normalize M = m*4^e + r4
  (m ∈ [2^14,2^16) window via two R16 lookups; (e,4^e,2^e,h) bound by a
  32-row EXP4 table, rows ≥ 17 duplicate row 16 to cap magnitudes),
  R' = ISQ[m] (2^16-row table, round(2^32*sqrt(d)/sqrt(m))), R = (R'+h)>>e via
  dynamic-power rescale, W = rescale(R⊗g,16), y = rescale(W∘x,16).
  DEVIATION from zkob: table-based inverse-sqrt instead of the (R±1)²M 2^80
  bracket (doesn't fit Goldilocks); same ±1-2 ulp tolerance class, exact
  bitwise vs the reference which implements the identical table computation.
- softmax (int, zkob softmax8-style): scores z at scale 2^8 (QK rescale with
  pow2 temp 2^ceil(ldh/2) folded in, |z| < 2^15 RS16-checked); rowmax mx by
  SEL-attainment + dominance-via-lookup; DM = mask*(mx−z)+(1−mask)*2^16;
  E = EXPT[DM] (2^17-row table, round(2^16*exp(−t/2^8)), sentinel region 0);
  S = Σ_j E rowsum; P = round_half_up(2^16E/S) by the zkob bracket
  r1 = 2^17E + S − 2PS ∈ [0,2S), r2 = 2S−1−r1, both 2×16-bit limbs; P ∈ R17.
- SwiGLU: (G, SIL) 2-column mapping lookup vs SILU table (2^20 rows, signed
  col0 = j−2^19 — signed table values avoid shift columns), M = rescale(SIL*U,16).
- RoPE: y64 = q*C + σ*q_flip*SN (C/SN public int cos/sin at scale 2^14),
  rotate-half via the flipped-point MLE claim on the SAME commitment,
  y = rescale(y64, 14).
- residual adds: plain zero-check + RS20.

### Proof architecture (all on ONE transcript + ONE p3lu::XCtx ledger,
one lu_flush, one p3bo batched-opening pass — the p3tf composition pattern)

- Every gadget = committed witness columns + p3hwl::sc5z eq-weighted
  zero-checks (Libra blinds free in zk) + p3lu::defer_v lookups + claimc
  terminal claims. Template: p3_quant.cuh.
- **Integer matmul (the new piece, p3imm)**: cheap classic-style contraction
  WITHOUT product-domain commitments: draw (z, zex), hiding claim
  yY = Y~(xpt(z,zex)), then ONE cubic sumcheck over the product domain
  (k,j,i,ex) of EQ·Xb·Wb with Xb/Wb VIRTUAL broadcasts of the committed
  operands (index maps only, bc_aug/expt contract). Terminal claims land on
  the operand commitments at points with random ex coords (hiding). zk: Y's
  mask slice 1 is LINKED = matmul of the operands' mask slice 1s so the claim
  algebra holds slice-by-slice (p3_zkc mechanism-1 linkage, exactly the seam
  rule). Data committed per matmul: only Y (the accumulator column).
- Layout dividend: operand index maps make transposes/head-slices/concat FREE
  (claims at partially-fixed points on the producer commitment). The int layer
  therefore has NO per-head rope/quant instances, NO V^T transpose commitment,
  NO concat seam: rope/rescale/softmax run on full (T×d / A·seq×seq) grids;
  per-instance QK/PV matmuls claim slices of the shared commitments directly.
  All chaining = shared commitments (root equality) + partial-point claims.
- Contraction dims are pow2 by config → no k-padding seams.
- Public IO binding: x0 and out bound at zpt (public values), like p3tf.
- Weight roots pinned by the caller (bench recomputes independently non-zk,
  pins prover's salted commits in zk — p3_transformer_bench pattern).
- Field/soundness parity with fp8 side: same Goldilocks base-field challenges,
  same Basefold PCS, R=2 Q=24 at bench time (same as fp8 sweep).

### Range/no-wrap ledger (soundness of in-field integer identities)
x,w ∈ ±2^19; acc ≤ 2^19·2^19·2^10 = 2^48; R < 2^32 (limb-checked),
R·2^e ≤ 2^48, R·g ≤ 2^49... wait R·g with R<2^32, g<2^19 → 2^51 ✓;
EXP4 rows capped (p4 ≤ 2^32) so m·p4 ≤ 2^48; bracket 2PS ≤ 2^17·2^27 = 2^44;
rope 2·2^19·2^14 = 2^34. All ≪ p ≈ 2^64. Every advice column is range-bound
by a lookup before entering a product.

### Plan / status
1. [x] tables + p3imm matmul + p3irs rescale + selftest (22/22 both modes)
2. [x] p3irms rmsnorm, p3irope, p3iadd + selftests
3. [x] p3ismx softmax, p3iswg swiglu + selftests — GADGET BATTERY 60/60
   (honest accepts + per-gadget adversarial teeth, non-zk AND zk modes;
   binary /root/p3_int_selftest).  Note: the rope "unrotated" tamper must
   target a position t>0 (t=0 has sin=0).  P needs NO range lookup: the
   bracket + r1+r2=2S-1 limb identity FORCES P in-field (see p3ismx header).
4. [x] p3_int_layer.cuh compose + battery: 34/34 (honest accept both modes,
   zk prove 1.81s at seq16/d64; every chain tamper, per-gadget forgery,
   public-IO tamper and weight-root tamper rejected at its owning check;
   binary /root/p3_int_layer_test).  int_layer_ref.py: REF_OK -- all 30
   dumped intermediates match the C++ witness replay BITWISE (tables shared
   via int_tables.bin; the python chain math is independent code).
   Gotcha fixed on the way: gadget Operands now hold const Col* (a brace-
   temporary Operands copy dies while the ledger holds &col.v -- dangling).
5. [x] zk hiding battery: 18/18 (binary /root/p3_int_zk_test).  Same
   methodology as the fp8 battery (12000 draws at fixed challenges on the
   REAL int column set): (1) all 14 column classes uniform (chi-sq 232-309,
   threshold 400); (2) finite-difference coeffs uniform blinded, control
   spikes to 3.07e6; (3) matmul mask-linkage (Ym1 = Xm1*Wm1) and row-sum
   linkage claims uniform AND agree; (4) batch blinder OTP; (5) witness-
   recovery: control collapses to 1 distinct value, hidden 12000/12000
   distinct, posterior flat; (6) HVZK simulator same law + accepts;
   (7) GKR mask siblings uniform with teeth.  ZK CLAIM SUPPORTED.
6. [x] p3_int_layer_bench.cu + FULL zk=1 sweep COMPLETE at the fp8 grid, all
   12 points verify_ok=1, peak RSS 4.97 GB (<< 41 GB cap).  Matmul sumcheck
   moved to the GPU functor path (p3hwl::sc5rz<FImmGpu>, identical
   transcript; batteries re-run green 60/60 + 34/34 after the switch).
   Results: /root/zk_int_layer_results.json (+ /root/int_sweep.log).
   ANCHOR s128 d=64: int prove 4.25 s (fp8 18.0 s -> measured premium 4.2x;
   zk=0 anchor 1.21 s -> int ZK premium 3.5x).  Measured fp8/int premium
   across the grid: 3.7x (s64) .. 14.8x (b64, p512), matmul-dominated
   (53-85% mm share) unlike the audit's circular estimate (92-98% copied
   gadget floor).  Old estimate was wrong BOTH ways as the audit predicted:
   too big at token-heavy configs (s1024 est 24.7 vs measured 15.7), too
   small at wide ones (p512 est 5.77 vs measured 16.2).
   Deliverables written: INT_LAYER.md (report + measured table + caveats),
   int_layer_done.flag = OK / 4.25 s anchor.  TASK COMPLETE.

Files: p3_int_gadgets.cuh (namespace p3ig), p3_int_layer.cuh (p3itf),
p3_int_selftest.cu, p3_int_zk_test.cu, p3_int_layer_bench.cu, int_layer_ref.py.
NO edits to any fp8 source.
