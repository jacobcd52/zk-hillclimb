# SOFTMAX_DESIGN.md — final design for `zkob_softmax.cu`

Status: DESIGN FINAL (2026-06-10). Companion to the rmsnorm design in HANDOFF.md
("IMMEDIATE NEXT STEP" section, since implemented as `zkob_rmsnorm.cu`, selftest ALL PASS).
Read PHASE0_NOTES.md first for all pinned conventions (FS transcript, IPA, logUp,
Montgomery rules, -dlto gotchas). This document is written so the implementer makes
**no design decisions**: every relation, scale, bound, absorb, file, CLI argument and
selftest case is pinned here.

## 0. Executive summary and deviations from the HANDOFF sketch

The HANDOFF sketch was: *"E = exp mapping lookup of (z − mx) reusing glu machinery;
row-sum binding (SS-style); inverse advice R_i bracket-bound (reuse rmsnorm machinery);
P = R⊙E hadamard + rescale. mx need not be exact max (shift-invariance; table domain
bounds it)."* Two parts of that sketch hit concrete problems:

**Problem 1 — the final rescale is unbuildable.** With inverse advice R at scale 2^k,
`P = rescale(R⊙E, 2^k)` needs `zkob_rescale` with sf = 2^k. The rem range table has
N = sf entries; for any useful inverse precision k ≥ 32 (k = 32 already gives 2^-7
worst-case relative error on P), a 2^32-entry table is impossible (128 GB of Fr_t).
Multi-stage rescaling (2^16 + 2^16) works but adds double-rounding to the spec, keeps
the R ±1 covert channel (log2(3) ≈ 1.585 bits/row → ≈ 39 Kbit per forward pass over
24,576 rows), and costs 2–3 extra obligations per head.

**Problem 2 — "table domain bounds mx" leaves a wide covert channel.** The table
domain only forces mx ≥ rowmax(z) and mx ≤ rowmin(z) + N − 1. Every shift
s = mx − rowmax in that window (up to N ≈ 2^20 values) produces a *distinct* accepted
E tensor (dE/ds ≈ E/2^16 ≈ 1 ulp per step near the max entry), i.e. **~16–20 covert
bits per row** (≈ 400–490 Kbit per forward pass) unless mx is pinned exactly — and an
exact row-max proof needs one-hot/argmax machinery we do not have.

**The fix (this design): eliminate both advice values.**
- The pipeline's effective softmax temperature is 128 (§1): honest exponents are
  |z_real|/128 ≲ 4.5. At score scale 2^9 a single 2^20-entry exp table covers
  exponents in [−8, +8) — the **entire honest range with ≥ 5× margin in real score
  units** — so no max-shift is needed at all. Shift-invariance of softmax means the
  real-valued target function is unchanged by dropping the pipeline's shift.
- The normalization `P[i,j] = round(2^16·MK[i,j]·E[i,j] / S[i])` is bound **exactly**
  (round-half-up, unique integer) by a per-entry residual bracket
  `r1 = 2^17·MK·E + S_bcast − 2·P·S_bcast ∈ [0, 2S)`, with r1 ≥ 0 from a 20-bit limb
  range lookup and r1 < 2S from the committed complement r2 = 2S − 1 − r1 ≥ 0.
  No inverse advice R exists.

Resulting covert capacity of this obligation: **0 bits/row** (§3). Machinery used:
fs_phase1/fs_phase2 logUp, fs_hadamard, open_prove/open_verify (FS-IPA),
build_eq_tensor, k_bcast_rows, k_fr_emul, k_fr_fold, h_* host G1 helpers —
**no new CUDA kernels at all** (constraint satisfied in the strongest form).
`zkob_lookup.cuh` and `vrf_common.cuh` are used as-is; **do not edit either header**
(edits require re-running every driver's selftest).

One instance per head: B = 1024 rows × NCOL = 1024 cols, 12 heads × 2 layers = 24
instances (justification in §4.0).

---

## 1. Pipeline semantics (m68-pipeline.py, quoted, every scale explicit)

Constants (m68-pipeline.py lines 29–32):

```python
LOG_SF = 16          # weight/activation fixed-point scale 2^16
LOG_OFF_FACTOR = 5
VALUE_LOGSF = 16     # scale used to integerize the QK^T scores
ACCU_LOGSF = 20      # scale at which the score integers are READ inside softmax
```

llama-68m: embed 768, 12 heads, head_dim d = 64, sqrt(d) = 8, seq = 1024, 2 layers.

The attention block (lines 137–159, the part our obligation must bind):

```python
Q = load_int("temp_Q.bin").reshape(seq, embed) / (1 << 16)     # int32, scale 2^16
K = load_int("temp_K.bin").reshape(seq, embed) / (1 << 16)
V = load_int("temp_V.bin").reshape(seq, embed) / (1 << 16)
...rotary applied to Q, K in float...
A = to_int64(Q @ K.transpose(-2, -1), VALUE_LOGSF)             # line 147
mask = torch.triu(torch.ones(seq, seq, device=0, dtype=bool), diagonal=1)   # line 148
A -= torch.max(A * ~mask, dim=-1, keepdim=True).values         # line 149
shift = math.sqrt(head_dim) * torch.log(
    (torch.exp((to_float(A, ACCU_LOGSF) / math.sqrt(head_dim))) * ~mask)
    .sum(axis=-1, keepdim=True))                               # lines 150-152
A -= to_int64(shift, ACCU_LOGSF)                               # line 153
attn = (torch.exp(to_float(A, ACCU_LOGSF, torch.float64)
        / math.sqrt(head_dim)).float()) * ~mask                # line 154
attn = fromto_int64(attn @ V, VALUE_LOGSF)                     # line 155
...
save_int(attn, 1 << 16, attn_out)                              # line 158-159
```

Reading off the exact semantics:

1. **Scores.** `A = round(QK^T_real · 2^16)` per head — int64, scale 2^16
   (`to_int64(x, k) = round(x·2^k)`). Shape per head: 1024 × 1024.
2. **Causal mask.** `mask` is True strictly above the diagonal (`diagonal=1`), i.e.
   position (i, j) is **masked iff j > i**; the diagonal is unmasked; row i has
   i+1 unmasked entries (row 0 has exactly one).
3. **Max-shift (line 149).** Subtracts `max_j (A·~mask)[i,j]` — note `A*~mask` zeroes
   masked entries first, so the subtracted value is `max(0, max_{j≤i} A[i,j])`
   (the 0 sneaks in whenever the row has at least one masked entry and all unmasked
   scores are negative). Pure shift advice; softmax is shift-invariant in real
   arithmetic, so this quirk does not change the target function.
4. **The temperature.** Lines 150–154 read A through `to_float(A, ACCU_LOGSF) =
   A / 2^20` although A was built at scale 2^16. The exponent actually computed is
   `(A/2^20)/sqrt(64) = (QK^T_real·2^16)/(2^20·8) = QK^T_real / 128`.
   **The pipeline's softmax temperature is 128 — this is the load-bearing fact that
   lets one lookup table cover the whole domain.** (Standard attention would divide
   by 8; the extra /16 is the pipeline's scale-reinterpretation. We bind what the
   pipeline computes, not what a textbook would.)
5. **Log-sum shift (lines 150–153).** The second shift is `sqrt(d)·log(Σ_unmasked
   exp(·))` rounded at 2^20 — i.e. the pipeline normalizes in the log domain (the
   upstream zkAttn approach: `zksoftmax_shift` in zksoftmax.cu does the same in
   float64 inside a kernel). After it, line 154's exp output is already ≈ normalized.
6. **Probabilities are float.** Line 154's `attn` is a float tensor (rows sum to ≈ 1
   over unmasked entries; masked entries exactly 0). It is **never integerized**; only
   `attn @ V` is rounded, at scale 2^16 (line 155, `fromto_int64(·, 16)`), and saved
   as int32 at scale 2^16 (lines 158–159).

Because steps 5–6 are float64 `exp`/`log`, **no integer proof can bind lines 150–155
literally**. Exactly as with rmsnorm (where the python computes `1/sqrt` in float and
the proof binds an integer bracket instead), the obligation binds an *integerized
softmax specification* of the same real-valued function:
`P_real[i,j] = exp(z_real[i,j]/128) / Σ_{j'≤i} exp(z_real[i,j']/128)` for j ≤ i,
0 for j > i — with every integerization step (two rescale roundings, one table
rounding, one division rounding) exact and proven. The orchestrator's prover computes
P with this integer spec (replacing the python float path for downstream continuity),
the same move already made for rmsnorm's R.

### 1.1 The integer chain feeding this obligation

Per head h ∈ [0,12), layer l ∈ {0,1}:

- `zkob_fc` (HANDOFF item (c)) proves the scores matmul
  `z = Q_h @ K_h^T` (B=1024, IN=64, OUT=1024), Q_h, K_h int32 at scale 2^16
  (post-RoPE integerization is item (c)'s concern, not ours). Chain file `z.i64`
  (int64, unpadded 1024×1024, **scale 2^32**), commitment `com_Y`.
- `zkob_rescale` stage 1: sf = 2^13 → `z13` at scale 2^19 (int32 chain file).
- `zkob_rescale` stage 2: sf = 2^10 → `z_` at scale **2^9** (int32 chain file).
  (Two stages because a single sf = 2^23 table violates the lookup layout
  C_pad ≤ N ≤ D: sf must lie in [2^10, 2^20] for the 1024×1024 grid. Stage order is
  pinned 2^13-then-2^10: the intermediate at scale 2^19 stays < 2^31 and survives
  `save_int`; the reverse order would overflow int32 at scale 2^22. See §7.3 for the
  int32→int64 widening shim between the stages.)
- **`zkob_softmax` (this design): z_ (scale 2^9) → P (scale 2^16).**
- `zkob_fc` values matmul: `out = P @ V_h` (B=1024, IN=1024, OUT=64) at scale 2^32,
  then `zkob_rescale` sf 2^16 → scale 2^16 — matching line 155's
  `fromto_int64(attn@V, 16)` (item (c) again).

Exponent bookkeeping check: exponent = z_real/128 = (z/2^32)/2^7 = z/2^39
= (z_·2^23)/2^39 = **z_/2^16** (up to the two proven rescale roundings). So with z_
at scale 2^9, the exp table is indexed by v = z_ with
`E(v) = round(2^16 · exp(v / 2^16))`.

---

## 2. Statement to prove (exact integer relation)

Public constants (absorbed into the transcript and recorded in dims.bin):
- B = NCOL = 1024 (require B == NCOL, both powers of two; throw otherwise).
- Causal mask MK[i,j] = 1 if j ≤ i else 0 (derived from B; never committed — public).
- Exp table: LOW_E = −2^19 = −524288, LEN_E = 2^20.
  `X_E[k] = round(2^16 · exp((LOW_E + k)/2^16))` for k ∈ [0, LEN_E)
  — domain v = z_ ∈ [−2^19, 2^19), exponent ∈ [−8, +8). Values: X_E[0] = 22,
  max X_E = round(2^16·e^(8−2^-16)) = 195,376,xxx < 2^28. Generated once by a pinned
  python script (§7.4), stored as int32, registered by sha256 like the swiglu table.
- Limb range table: tLookupRange(0, LEN_R), LEN_R = 2^20 (limb width 20 bits,
  LOG_R = log2(LEN_R) = 20).
- Output scale LOG_OUT = 16 (P at scale 2^16).

Committed tensors (all FrTensors PLAIN form, all grids committed row-wise with the
single generator set `gen` of size 1024; no padding anywhere — every dimension is
already a power of two):
- `z_` — input grid (chained; com_z byte-identical with rescale stage-2's com_Xr).
- `E`  — exp values grid, E[i,j] = X_E[z_[i,j] − LOW_E] over **all** (i,j) including
  masked positions (masked scores are real matmul outputs in the same range).
- `S`  — row sums, length B (committed with the same gen, 1 commitment row):
  `S[i] = Σ_{j: MK[i,j]=1} E[i,j]`.
- `P`  — probabilities grid: `P[i,j] = round_half_up(2^16 · MK[i,j] · E[i,j] / S[i])`
  i.e. the unique integer with `r1[i,j] := 2^17·MK[i,j]·E[i,j] + S[i] − 2·P[i,j]·S[i]
  ∈ [0, 2·S[i])`. (Proof of uniqueness: r1 ∈ [0,2S) ⟺ 2PS ≤ 2^17·MK·E + S < 2(P+1)S
  ⟺ P ≤ 2^16·MK·E/S + 1/2 < P + 1 ⟺ P = floor(2^16·MK·E/S + 1/2).)
  Masked entries: MK = 0 forces r1 = S − 2PS ∈ [0,2S) ⟺ P = 0 exactly.
- `L`  — limb tensor, 4 planes of B×NCOL each (flat index = plane·2^20 + i·2^10 + j;
  4096 commitment rows, plane index = the two MSBs of the row index):
  plane 0 = r1 mod 2^20, plane 1 = floor(r1 / 2^20),
  plane 2 = r2 mod 2^20, plane 3 = floor(r2 / 2^20), where r2[i,j] = 2·S[i] − 1 − r1[i,j].
  (r1, r2 are never committed standalone; their MLEs are reconstructed from plane
  openings of com_L — no homomorphic affine link needed.)
- `m_E`, `A_E` — exp-lookup multiplicities (LEN_E/1024 = 1024 rows) and logUp inverse
  grid (1024 rows); `m_L`, `A_L` — same for the limb lookup (1024 and 4096 rows).

The relation proven (∀ i, j ∈ [0,1024)):

```
(R1)  z_[i,j] ∈ [LOW_E, LOW_E + LEN_E)  and  E[i,j] = X_E[z_[i,j] − LOW_E]
(R2)  S[i] = Σ_{j ≤ i} E[i,j]
(R3)  r1[i,j] = 2^17·MK[i,j]·E[i,j] + S[i] − 2·P[i,j]·S[i],
      r1[i,j] ≥ 0,  r2[i,j] = 2·S[i] − 1 − r1[i,j] ≥ 0,
      r1, r2 each = lo + 2^20·hi with lo, hi ∈ [0, 2^20)
      ⟹ P[i,j] = round_half_up(2^16·MK[i,j]·E[i,j]/S[i]) exactly, P[masked] = 0
```

**Tolerances: none.** Every committed tensor is a deterministic function of
(z_, MK, X_E). R1 also serves as the range proof on z_ (the table domain *is* the
range check); it simultaneously kills any need for a max-shift (see §3).

Well-definedness: every row has the diagonal unmasked, X_E ≥ 22 ≥ 1 on the whole
domain, so S[i] ≥ 22 ≥ 1 always (division never degenerate; this is why LOW_E must
satisfy round(2^16·e^(LOW_E/2^16)) ≥ 1 — at −2^19 the value is 22; pinned invariant
if anyone retunes the domain). E ≤ S (E is a summand of S, all summands ≥ 0) gives
P ≤ 2^16; row sums Σ_j P[i,j] ∈ [2^16 − 512, 2^16 + 512] automatically (510 ≤ i
rounding errors of ≤ 1/2 each; no separate constraint needed).

Honest-prover throws (completeness guards, mirroring rmsnorm's "M ≥ 2^62" throw):
z_ out of [LOW_E, LOW_E+LEN_E); r1 ∉ [0, 2S) (cannot happen if P computed per spec —
defensive); r1 ≥ LEN_R^2 (cannot happen, §6); B ≠ NCOL or not a power of two;
gen.size ≠ NCOL.

---

## 3. Advice-binding analysis

**This design has zero prover advice.** For the record, and because the task of this
section is to quantify what the alternatives leak, here is the full accounting.

### 3.1 The max-shift mx (eliminated)

What freedom would a committed mx have had under the sketch?

- The lookup domain forces, for every unmasked j: 0 ≤ mx[i] − z_[i,j] ≤ N − 1.
  The left inequality forces mx[i] ≥ max_j z_[i,j] (the table bound is genuinely a
  one-sided max proof — for free). The right forces mx[i] ≤ min_j z_[i,j] + N − 1.
  So the dishonest shift s = mx[i] − rowmax ranges over [0, N − 1 − spread_i],
  spread_i = rowmax − rowmin. With N = 2^20 and typical spread ≪ N, that is ≈ 2^20
  accepted values.
- Distinctness: E_j(s) = round(2^16·e^{−(s+δ_j)/2^16}); near the max entry
  dE/ds ≈ −E/2^16 ≈ −1 per unit s, so adjacent s values produce distinct committed E
  tensors until E decays, then distinct values thin out logarithmically. Capacity
  ≈ log2(#accepted s) ≈ **16–20 bits/row**, i.e. 16–20 × 24,576 rows ≈
  **400–490 Kbit per forward pass** — two orders of magnitude beyond the rmsnorm
  floor (1.6 bits/row). The semantic damage is bounded (output P perturbed by ≈ ±1
  ulp per entry, plus tail entries flipping to 0 as s pushes them past the table
  cutoff), but as a covert channel it is enormous.
- Constraining it: an S ≥ 2^16 floor (one extra range check) caps s at
  2^16·ln σ_i (σ_i = Σ e^{−δ_j/2^16}): 0 bits for peaked rows but still up to
  ≈ 18.8 bits/row for flat rows. Only an exact "∃ j: y_j = 0" (row-max) proof kills
  it, which needs one-hot/argmax machinery we don't have.
- **Resolution: no shift at all.** The pipeline temperature is 128 (§1.4), so honest
  exponents satisfy |z_real|/128 ≤ ~4.5 (worst-case |QK^T| ≲ 576 for head_dim 64
  with |q|,|k| ≲ 3); the table domain [−8, +8) in exponent units covers it with
  ≥ 5× margin in score units (|scores_real| up to 1024). Shift-invariance of the
  real target means dropping the shift changes nothing about the function being
  approximated; it only changes which integerization is proven. Leftover covert
  freedom: **0 bits**. The pipeline's max(0, rowmax) quirk (line 149) becomes
  irrelevant for the same reason.
- What replaces the "range proof on z − mx": the exp-table domain itself is the
  range proof **on z_ directly** — a dishonest z_ cannot even be out of [−2^19, 2^19)
  (and z_ is chained by commitment anyway, so it is not free to begin with).

### 3.2 The inverse advice R (eliminated)

Under the sketch, R[i] ≈ 2^k/S[i] bracket-bound to ±1 leaks log2(3) ≈ 1.585 bits/row
(rmsnorm's measured floor): 1.585 × 1024 × 24 = **38,955 bits ≈ 39 Kbit per forward
pass**, on top of mx. It also costs accuracy (±0.5 absolute on R = up to 2^-7
relative on P at k = 32) and an unbuildable or multi-stage final rescale (§0).

The per-entry rounding bracket (R3) replaces it. P is the *unique* integer satisfying
the proven inequalities — round-half-up needs no tolerance because the open/closed
bracket [0, 2S) breaks the tie deterministically. Leftover freedom: **0 bits**.

### 3.3 Total covert capacity

| design | per row | per forward pass (2 layers × 12 heads × 1024 rows) |
|---|---|---|
| sketch (mx domain-bound + R ±1) | ~17.6–21.6 bits | ~430–530 Kbit |
| sketch + S-floor on mx | 1.6–20.4 bits (data-dependent) | 39–500 Kbit |
| **this design** | **0 bits** | **0 bits** |

The only nondeterminism left in the whole softmax obligation is *completeness*
failure (out-of-domain scores → no proof exists), never *choice*. For the full
attention path, the chained rescale obligations contribute 0 bits as well (rem is
exactly range-bound; the affine link is exact).

---

## 4. Proof obligations (one transcript, 16 IPA openings)

### 4.0 Instance granularity

**One obligation instance per (layer, head)** — 24 instances:
- matches `zkob_fc` per-head scores/values matmuls (item (c)), so the chain
  byte-equalities are per-head commitment files with no reshuffling;
- the grid is exactly 2^20 = 1024×1024 with zero padding, every layout constraint
  (C_pad ≤ N ≤ D, N | D) satisfied as equalities or naturally;
- per-head GPU footprint stays < 1 GB (L and A_L at 2^22 Fr_t = 128 MB each
  dominate); stacking 12 heads would put single tensors at 2^25.5 Fr_t ≈ 1.5 GB each
  and break nothing else, but gains nothing — the work is linear either way.

Obligation id naming: `layer{l}.attn.softmax.h{hh}` (hh = 00..11), composed with the
two score-rescale instances `layer{l}.attn.scores_rescale13.h{hh}` and
`...scores_rescale10.h{hh}` under the manifest id `layer{l}.attn.softmax` by the
orchestrator (composition precedent: mlp.swiglu = glu + hidden-rescale).

### 4.1 Obligation 1 — exp mapping lookup (binds R1)

glu pattern, verbatim: every pair (z_[i,j], E[i,j]) must be a row of the public
table {(v, X_E[v − LOW_E]) : v ∈ [LOW_E, LOW_E + LEN_E)}.
- Combined witness comb = z_ + r·E vs combined table T_comb = table + r·mapped,
  r an FS challenge derived after all commitments.
- The verifier forms com_comb = com_z + r·com_E **homomorphically** (1-thread
  h_mul/h_add — the chained commitment com_z participates directly in the lookup;
  no separate link sumcheck for z_ is needed).
- D = 2^20, N = LEN_E = 2^20 ⟹ n1 = 0, **pure phase2** (validated shape: glu
  selftest case b; rescale "n1=0" case).
- Terminals: A_f vs com_A_E, S_f vs com_comb, m_f vs com_m_E (3 IPA openings);
  B_f, T_f recomputed by the verifier from the public table + r (k_fr_fold chain,
  exactly the glu verifier block).
- No padding ⟹ no (0, mapped(0)) table requirement (glu's `mapped(0)==0` check does
  NOT apply here and must not be copied; mapped(0) = X_E[2^19] = 65536).

### 4.2 Obligation 2 — limb range lookup (binds r1, r2 ≥ 0 and < 2^40)

logUp on L vs tLookupRange(0, LEN_R = 2^20): D_L = 4·2^20 = 2^22, N = 2^20, n1 = 2.
Terminals: A_f vs com_A_L, S_f vs com_L, m_f vs com_m_L (3 IPA openings, opening
points = reversed round challenges; m at the phase-2 suffix). Committed with the same
gen (1024 cols): com_L and com_A_L have 4096 rows, com_m_L has 1024 rows.

### 4.3 Obligation 3 — row-sum sumcheck (binds R2)

`ev_S = S̃(u_b) = Σ_b W_rs(b)·E(b)·1(b)` over logD = 20 variables, where the weight
tensor `W_rs = k_bcast_rows(build_eq_tensor(u_b)) ⊙ MK` (k_fr_emul; MK uploaded as a
plain FrTensor from a host int buffer `mk[i·1024+j] = (j<=i)`).
- Reuses fs_hadamard with E = W_rs, S = E, U = 𝟙 (all-ones grid).
- Verifier: recomputes the folded weight W_f by building W_rs itself (it knows u_b
  and MK) and folding with k_fr_fold over the 20 round challenges — the same
  recompute-the-public-side pattern as B_f/T_f in every lookup. (The rmsnorm eq_acc
  shortcut does NOT apply: W_rs is not a pure eq tensor. Cost: one 2^20 fold chain,
  negligible.)
- Verifier requires U_f2 == 1 (MLE of all-ones is 1 at any point — no opening),
  checks the terminal `cur == W_f · S_f2 · U_f2`, then opens:
  E at pt_rs = reverse(ws_rs) vs com_E; ev_S at u_b vs com_S (single-row opening,
  u_row empty — the rmsnorm com_M pattern). 2 IPA openings.

### 4.4 Obligation 4 — bracket sumcheck V1 (the MK·E term of R3 at u_r)

After challenge u_r (20 vars): `c1 = Σ_b eq(u_r,b)·MK(b)·E(b)` — the MLE of MK⊙E
at u_r. fs_hadamard with E = build_eq_tensor(u_r) ⊙ MK (k_fr_emul), S = E, U = 𝟙.
Verifier: recompute the weight fold (same as 4.3, with u_r), require U_f2 == 1,
terminal check, open E at pt1 = reverse(ws_v1) vs com_E. 1 IPA opening.

### 4.5 Obligation 5 — bracket sumcheck V2 (the P·S term of R3 at u_r)

`c2 = Σ_b eq(u_r,b)·P(b)·S_bcast(b)`. fs_hadamard with E = build_eq_tensor(u_r)
(pure eq — verifier uses the my_eq accumulator over all 20 rounds, exactly the glu
hadamard verifier), S = P, U = S_bcast (prover materializes the broadcast grid with
k_bcast_rows; it is **never committed**).
Verifier terminal openings: P at pt2 = reverse(ws_v2) vs com_P, and U_f2 against
**com_S opened at the row-bit suffix of pt2** — the MLE of a column-broadcast tensor
equals the row vector's MLE at the row bits (independent of column bits), so
S̃_bcast(pt2) = S̃(pt2[10..20)). This binds the broadcast without an extra commitment.
2 IPA openings.

### 4.6 Obligation 6 — residual reconstruction + the two bracket identities

Four plane openings of com_L at u_r with Boolean plane bits, plus one more S opening:
- v00 = L̃ at u_pt = (u_r[0..10) cols ‖ u_r[10..20) rows ‖ 0 ‖ 0)   (plane 0, r1 lo)
- v10 = … ‖ 1 ‖ 0   (plane 1, r1 hi);  v01 = … ‖ 0 ‖ 1 (plane 2, r2 lo);
  v11 = … ‖ 1 ‖ 1 (plane 3, r2 hi).
  (Plane p contributes bits (p&1, p>>1) as u_pt[20], u_pt[21]; me_weights pairs
  u_pt[i] with bit i of the flat index, and the flat index's bits 20,21 are the
  plane index — consistent with flat = plane·2^20 + i·2^10 + j. Constant 0/1
  challenges are ordinary field elements for fold_chain/IPA; soundness unaffected.)
- S_id = S̃(u_r[10..20)) vs com_S (single-row opening).

Verifier computes r̃1 = v00 + 2^20·v10, r̃2 = v01 + 2^20·v11 and checks (plain field
arithmetic, h_scalar):

```
(I1)  2^17·c1 + S_id − 2·c2 == r̃1          "bracket r1 identity"
(I2)  r̃1 + r̃2 + 1 == 2·S_id               "bracket sum identity"
```

I1 is the MLE at u_r of `r1 = 2^17·MK·E + S_bcast − 2·P·S_bcast`; I2 is the MLE of
`r1 + r2 + 1 = 2·S_bcast`. Both are tensor identities ⟺ they hold at the random u_r
(Schwartz–Zippel over multilinears, the standard argument used everywhere else).
Combined with obligation 2 (r1, r2 ≥ 0 componentwise as 20-bit limb pairs), R3 holds
exactly. 5 IPA openings (4 × L planes + S_id).

### 4.7 Obligation 7 — chain byte-equalities (orchestrator-level, not in the driver)

- `<softmax obdir>/com_z.bin == <scores_rescale10 obdir>/com_Xr.bin` (byte-identical).
- `<scores_rescale10>/com_X.bin == <scores_rescale13>/com_Xr.bin`,
  `<scores_rescale13>/com_X.bin == <scores zkob_fc obdir>/com_Y.bin`.
- `<values zkob_fc obdir>/com_X.bin == <softmax obdir>/com_P.bin`.

The driver re-commits z_ from its input file with the same gens and no padding, so
byte-identity holds automatically for an honest run.

Soundness summary — which check catches what: lookup 1 pins (z_, E) pairs to the
public exp spec and z_ to the domain; sumcheck 3 pins S to E and the public mask;
sumchecks 4+5 + identities I1/I2 + lookup 2 pin P to (E, S, MK) uniquely; the IPA
openings pin every terminal to commitments; com_z and com_P pin the instance into
the chain; the FS transcript (one per obligation, all commitments absorbed before
any challenge) prevents splice/replay.

---

## 5. FS schedule (one transcript; seed = "<run_seed>:<obligation_id>")

Absorb-by-absorb. Labels in quotes are the exact absorb labels; `→ x` means "derive
challenge x". Every IPA internally absorbs its L/R points before each of its 10 round
challenges (gen size 1024 ⟹ 10 rounds), continuing this same transcript — pinned
behavior of open_prove/open_verify.

```
absorb_u32  "B" B, "NCOL" NCOL, "LOW_E" (uint32)LOW_E, "LEN_E" LEN_E,
            "LEN_R" LEN_R, "LOG_OUT" 16
absorb_g1_tensor "com_z" com_z          (1024 rows)
absorb_g1_tensor "com_E" com_E          (1024 rows)
absorb_g1_tensor "com_P" com_P          (1024 rows)
absorb_g1_tensor "com_S" com_S          (1 row)
absorb_g1_tensor "com_L" com_L          (4096 rows)
absorb_g1_tensor "com_m_E" com_m_E      (1024 rows)
absorb_g1_tensor "com_m_L" com_m_L      (1024 rows)
→ r                                      (exp pair combiner)
→ beta_E
   [prover: comb = z_ + r·E;  A_E = 1/(comb + beta_E)]
absorb_g1_tensor "com_A_E" com_A_E      (1024 rows)
→ alpha_E
→ u_E = fs_challenge_vec(20)
exp lookup, 20 rounds (pure phase2):     per round absorb "p0","p1","p2","p3" → w
absorb_fr "A_f","S_f","m_f"             (exp terminals)
IPA(A_E)   at u_ptE = reverse(ws_E)         → ipa_A_E.bin
IPA(comb)  at u_ptE                          → ipa_comb.bin   (vs com_z + r·com_E)
IPA(m_E)   at u_mE  = reverse(ws_E[n1_E..]) → ipa_m_E.bin    (n1_E = 0 at real scale)
→ beta_L
   [prover: A_L = 1/(L + beta_L)]
absorb_g1_tensor "com_A_L" com_A_L      (4096 rows)
→ alpha_L
→ u_L = fs_challenge_vec(22)
limb lookup, 22 rounds (n1_L = 2):       per round absorb "p0".."p3" → w
absorb_fr "A_f","S_f","m_f"             (limb terminals)
IPA(A_L) at u_ptL = reverse(ws_L);  IPA(L) at u_ptL;  IPA(m_L) at reverse(ws_L[2..])
→ u_b = fs_challenge_vec(10)
absorb_fr "ev_S"                         (= S̃(u_b))
row-sum hadamard, 20 rounds:             per round absorb "hp0".."hp3" → w
absorb_fr "S_f2","U_f2"
IPA(E) at pt_rs = reverse(ws_rs)        → ipa_E_rs.bin
IPA(S) at u_b                            → ipa_S_rs.bin
→ u_r = fs_challenge_vec(20)
absorb_fr "c1"                           (V1 claim)
V1 hadamard, 20 rounds:                  absorb "hp0".."hp3" → w
absorb_fr "S_f2","U_f2"                 (V1 terminals)
IPA(E) at pt1 = reverse(ws_v1)          → ipa_E_v1.bin
absorb_fr "c2"                           (V2 claim)
V2 hadamard, 20 rounds:                  absorb "hp0".."hp3" → w
absorb_fr "S_f2","U_f2"                 (V2 terminals)
IPA(P) at pt2 = reverse(ws_v2)          → ipa_P_v2.bin
IPA(S) at pt2[10..20)                    → ipa_S_v2.bin
absorb_fr "v00","v10","v01","v11"
IPA(L) at (u_r ‖ 0 ‖ 0)                 → ipa_L00.bin
IPA(L) at (u_r ‖ 1 ‖ 0)                 → ipa_L10.bin
IPA(L) at (u_r ‖ 0 ‖ 1)                 → ipa_L01.bin
IPA(L) at (u_r ‖ 1 ‖ 1)                 → ipa_L11.bin
absorb_fr "S_id"
IPA(S) at u_r[10..20)                    → ipa_S_id.bin
[verifier-only final checks, no absorbs: I1, I2 of §4.6]
```

Notes for the implementer:
- The three fs_hadamard instances share the fixed "hp0..3" labels — fine, the
  transcript state is positional (rmsnorm already runs two such instances on one
  transcript); the interleaved "ev_S"/"c1"/"c2" absorbs disambiguate further.
- Round count checks on the verifier: exp ev = 4·20, limb ev = 4·22,
  each hadamard ev = 4·20; commitment row counts as listed; reject otherwise.
- The verifier re-derives every challenge itself; dims.bin is cross-checked against
  the CLI arguments exactly as in zkob_rmsnorm.

---

## 6. Numeric bounds (with the arithmetic)

All committed values are plain (non-Montgomery) Fr; host math fits in int64
(`long long`) throughout — no __int128 needed:

- z_ ∈ [−2^19, 2^19) by R1 (the table domain). Honest z_: |z_real| ≲ 576 ⟹
  |z_| ≲ 576·2^9 = 294,912 < 2^19 ✓ (margin 1.78×; measurement gate in §9.1).
- E = X_E[·] ∈ [22, 195,376,482]. Upper: 2^16·e^((2^19−1)/2^16) = 65536·e^(8−2^-16)
  = 65536·2980.91 ≈ 1.9538·10^8 < 2^28. Lower: 65536·e^(−8) = 21.98 → 22 ≥ 1.
- S = Σ over ≤ 1024 unmasked entries ≤ 1024·195,376,482 ≈ 2.0007·10^11 < 2^38;
  S ≥ 22 (diagonal always unmasked).
- P = round_half_up(2^16·MK·E/S) ∈ [0, 2^16] (E ≤ S componentwise ⟹ E/S ≤ 1).
- r1 ∈ [0, 2S) ⊂ [0, 2^39); r2 = 2S−1−r1 ∈ [0, 2S) ⊂ [0, 2^39).
  Both < LEN_R^2 = 2^40 ⟹ exactly two 20-bit limbs each (lo + 2^20·hi, hi < 2^19).
  The limb lookup only enforces r1, r2 ∈ [0, 2^40); the *exact* bound r1 ≤ 2S−1 comes
  from I2 + r2 ≥ 0 (this is why limb headroom is harmless here, unlike a rem-style
  decomposition where the headroom would loosen the rescale semantics — one of the
  reasons §0 rejects the R-route).
- Intermediates: 2^17·E ≤ 2^17·1.954·10^8 ≈ 2.56·10^13 < 2^45;
  2·P·S ≤ 2·65536·2.0007·10^11 ≈ 2.62·10^16 < 2^55 — all int64-safe (< 2^63).
- Field: every committed value < 2^40 ≪ p ≈ 2^255; sumcheck round evaluations are
  field elements by construction. Exp-table mapped values < 2^28 fit the int32
  table-file format (cf. glu's mapped-int32.bin).
- Multiplicities m_E, m_L ≤ D = 2^20 resp. D_L = 2^22 — trivially in range.
- Commitment row counts: com_z/E/P/A_E = 1024; com_S = 1; com_L/A_L = 4096;
  com_m_E = LEN_E/1024 = 1024; com_m_L = LEN_R/1024 = 1024.
- Lookup layouts: exp: C_pad = 2^10 ≤ N = 2^20 ≤ D = 2^20, N | D, n1 = 0 ✓.
  limb: C_pad = 2^10 ≤ N = 2^20 ≤ D_L = 2^22, N | D_L, n1 = 2 ✓.

---

## 7. CLI, files, chain interface

### 7.1 CLI (same style as the other drivers)

```
zkob_softmax prove  <obdir> <seed> <z-int32.bin> <B> <NCOL> <LOW_E> <LEN_E>
                    <expmap-int32.bin> <LEN_R> <gen.bin> <q.bin> [P-int32-out.bin]
zkob_softmax verify <obdir> <seed> <B> <NCOL> <LOW_E> <LEN_E>
                    <expmap-int32.bin> <LEN_R> <gen.bin> <q.bin>
zkob_softmax selftest
```

- `<z-int32.bin>`: unpadded B×NCOL int32 at scale 2^9 — the chain file written by
  scores_rescale10 (`zkob_rescale ... [Xr-int-out.bin]` writes int32 via save_int).
- `<expmap-int32.bin>`: the LEN_E mapped values X_E (int32; the (LOW_E, LEN_E) range
  is implicit, exactly the glu table convention). Loaded with a check that
  LEN_E is a power of two; do **not** port glu's `mapped(0)==0` check (§4.1).
- `<LEN_R>`: limb range table length (2^20 real scale); LOG_R = log2(LEN_R) derived;
  driver checks both layout constraint sets of §6 and r1 < LEN_R^2.
- One `<gen.bin>` (size NCOL = 1024) serves every tensor including com_S, because
  B == NCOL is required. `<q.bin>` = the 1-element IPA generator, as everywhere.
- `[P-int32-out.bin]`: unpadded B×NCOL **int32** chain file at scale 2^16.
  Deliberate deviation from the blanket "int64 chain files" rule: P ≤ 2^16 is proven
  (§6), the downstream consumer `zkob_fc prove` reads activations with
  load_int_tensor (int32), and zkob_rescale's own chain output is already int32 —
  int64 here would force a converter for no information.
- Prover writes dims.bin = {u32 B, u32 NCOL, i32 LOW_E, u32 LEN_E, u32 LEN_R,
  u32 LOG_OUT}; verifier cross-checks against argv.
- The driver does NOT mkdir the obdir (orchestrator creates it — standing gotcha).

### 7.2 Files in <obdir>

```
dims.bin
com_z.bin com_E.bin com_P.bin com_S.bin com_L.bin
com_m_E.bin com_m_L.bin com_A_E.bin com_A_L.bin
lookup_E.bin lookup_L.bin            (LookupProof serialization)
hp_rs.bin hp_v1.bin hp_v2.bin        (HadamardProof; claim_H = ev_S / c1 / c2)
lvals.bin                            (5 × Fr_t: v00, v10, v01, v11, S_id)
ipa_A_E.bin ipa_comb.bin ipa_m_E.bin
ipa_A_L.bin ipa_L_lk.bin ipa_m_L.bin
ipa_E_rs.bin ipa_S_rs.bin
ipa_E_v1.bin ipa_P_v2.bin ipa_S_v2.bin
ipa_L00.bin ipa_L10.bin ipa_L01.bin ipa_L11.bin ipa_S_id.bin
```

31 files, ≈ 1.42 MB/head (dominated by com_L + com_A_L at 4096·96 B = 384 KB each;
six 1024-row commitments at 96 KB; proofs+IPAs ≈ 45 KB). 24 heads ≈ 34 MB; com_z
duplicates the rescale chain commitment and can be deduped by the orchestrator later.

### 7.3 Chain wiring per head (orchestrator)

```
zkob_fc prove  ... scores: B=1024 IN=64 OUT=1024 → z.i64 (scale 2^32), com_Y
zkob_rescale prove <ob13> <seed:...rescale13.hNN> z.i64 1024 1024 13 gen q z13.i32
   [widen z13.i32 → z13.i64: orchestrator-side numpy int32→int64 copy; data files
    are not trust-carrying — the commitments are; see §9.3]
zkob_rescale prove <ob10> <seed:...rescale10.hNN> z13.i64 1024 1024 10 gen q z_.i32
zkob_softmax prove <ob>  <seed:...softmax.hNN>  z_.i32 1024 1024 -524288 1048576
                   expmap.bin 1048576 gen q P.i32
zkob_fc prove ... values: X = P.i32, B=1024 IN=1024 OUT=64 (com_W = chained V com)
```

Byte-equality checks (§4.7): com_Y==com_X(13), com_Xr(13)==com_X(10),
com_Xr(10)==com_z(softmax), com_P(softmax)==com_X(values-matmul).

### 7.4 Exp table generation (pinned)

One python script, run once, output registered by sha256 in public.json next to the
swiglu table:

```python
import numpy as np
LOW_E, LEN_E = -(1 << 19), 1 << 20
v = np.arange(LOW_E, LOW_E + LEN_E, dtype=np.float64)
np.rint(65536.0 * np.exp(v / 65536.0)).astype(np.int32).tofile("softmax-exp-table.bin")
```

(float64 exp of |arg| ≤ 8 is well within 0.5-ulp territory of the int32 rounding —
the table is bit-reproducible across machines for this domain; nevertheless the
sha256 registration, not regeneration, is the source of truth. The C++ driver never
generates the real table.)

---

## 8. Selftest plan

Structure copied from zkob_rmsnorm: `selftest_small()` (3 sizes) +
`selftest_real()`; semantic evil modes checked against the *expected reject string*;
byte tampers over every proof file with restore-and-reverify at the end.

### 8.1 Small honest cases

Toy mapped table mapped[k] = k + 1 (values ≥ 1 keep S ≥ 1; arbitrary mapping is fine
— the driver is table-agnostic like glu). Causal mask square. LOG_OUT stays 16.

| case | B=NCOL | LOW_E | LEN_E | LEN_R | shapes exercised |
|---|---|---|---|---|---|
| a | 8 | −8 | 16 | 32 | n1_E = 2 (phase1+2 exp), n1_L = 3, D_L = 256 |
| b | 4 | −8 | 16 | 32 | n1_E = 0 (pure phase2 — the real-scale shape), D = LEN_E |
| c | 16 | −32 | 64 | 64 | bigger grid, LEN_R = 64 (r1 < 2·16·64 = 2048 < 64²) |

Bounds check per case: r1 < 2S ≤ 2·B·max(mapped) < LEN_R² ✓ (a: 256 < 1024;
b: 128 < 1024; c: 2048 < 4096). Random z_ uniform in the domain (masked positions
included), S/P/r/limbs computed per §2 by the host, exact.

### 8.2 Semantic evil modes (one per check; why nothing else catches it)

Prover runs the honest *procedure* on an inconsistent witness (strict=false only for
the targeted recursion, rmsnorm-style); each must be rejected by exactly the named
check.

- **evil=1: E[idx] += 1 at an unmasked idx; S, P, r1/r2, limbs all recomputed from
  the evil E.** → **exp lookup round 0** rejects (the pair (z_, E+1) has no table
  row; recomputed anchor α+α² cannot match the multiplicities). Nothing else can
  catch it: row-sum, V1, V2, identities and the limb lookup are all consistent with
  the evil E by construction.
- **evil=2: P[idx] += 1 at a MASKED idx (also certifies mask enforcement); limbs
  stored as the low 40 bits of (r1 mod 2^40) where true r1 = −S < 0; everything else
  honest.** → **"bracket r1 identity" (I1)** rejects: the limb-reconstructed r̃1(u_r)
  is the MLE of the truncated values, while 2^17·c1 + S_id − 2·c2 is the MLE of the
  true (negative-as-field-element) r1 — they differ as tensors, so at random u_r
  the equality fails whp. The limb lookup passes (limbs in range), V2 passes (run
  honestly on the evil P), I2 is *constructed* to pass (r2 limbs set to
  2S−1−(truncated r1), which is in [0,2^40) here) — pin that in the evil-mode setup.
- **evil=3: P[idx] −= 1 at an unmasked idx with honest P[idx] ≥ 1 (the selftest
  asserts and picks e.g. a diagonal entry); r1' = r1 + 2S < 4S < 2^40 is
  representable, limbs honest for r1'; r2 limbs = low 40 bits of (2S−1−r1' mod 2^40)
  (true value negative).** → **"bracket sum identity" (I2)** rejects. I1 holds by
  construction (r1' really is 2^17·MK·E + S − 2P'S), the limb lookup passes, so I2 is
  the *only* line of defense for the upper bracket — this evil certifies it.
- **evil=4: S[row] += 1; P, r1/r2, limbs recomputed consistently from the evil S.**
  → **row-sum round 0** rejects (ev_S = S̃(u_b) no longer equals Σ W_rs·E; recomputed
  anchor breaks immediately). Brackets, lookups, V1/V2 and identities are consistent
  with the evil S by construction, so nothing else fires.
- **evil=5: V2 run with a corrupted broadcast buffer Sb[idx] += 1 (P, S, limbs all
  honest).** → **"IPA opening of V2 U_f2 vs com_S"** rejects: the V2 rounds are
  self-consistent and the terminal identity holds for the corrupted fold, P opens
  fine, but U_f2 = corrupted-S̃b(pt2) ≠ S̃(pt2 rows) from com_S. This certifies the
  never-committed broadcast tensor is genuinely pinned to com_S.

(z_-tampering inside the obligation is *deliberately* not an evil mode: com_z is
self-committed and internally consistent; the defense is the orchestrator's
byte-equality (§4.7), tested at the chain level like matmul→rescale was.)

### 8.3 Byte tampers

For every file in §7.2: tamper one byte (offsets per the rmsnorm table: lookup/hp
files at 4+32 → a round-0 evaluation; lvals.bin at 4; ipa_* at −32 → a_final;
com_* at 24 → first point x-limbs; dims.bin at 0), verify must REJECT; restore;
final full verify must ACCEPT. (lvals.bin has no count header — offset 4 hits v00's
second byte; any offset in [0,160) is fine, pin 4.)

### 8.4 Real-scale case

B = NCOL = 1024, gen = /tmp/gen1024.bin (ppgen 1024, shared with rmsnorm), the real
exp table (LOW_E = −2^19, LEN_E = 2^20) loaded from softmax-exp-table.bin if present,
else generated in-driver via host double exp **for the selftest only** (flagged
non-authoritative). Scores: z_ ~ round(N(0, 2^13)) clipped to the domain — matches
the realistic |z_| ≈ 600·2^9 envelope. Measure prove/verify wall-clock and proof
bytes; one byte tamper (lookup_E.bin @ 4+32) must reject.

**Expected cost (extrapolation, to be replaced by measurement):** reference points —
glu at D = 2^22 grid + 2^22 table: prove 11.4 s / verify 13.8 s; rescale at D = 2^20,
N = 2^16: 2.6 s / 3.3 s; rmsnorm (B=1024, C=768, ~17 IPAs): 9.4 s. This driver's
volume: commits ≈ 14·2^20 Fr-MSM rows (two 2^22 tensors + six 2^20-class), recursions
= one 2^20 + one 2^22 lookup + three 2^20 hadamards, 16 IPAs over gen-1024 ⟹
**≈ 8–12 s prove, ≈ 10–14 s verify per head**; 24 heads ≈ 4–5 min each side, plus
48 score-rescale runs ≈ 2.6/3.3 s each ≈ 2–3 min each side, plus the per-head
matmuls (item (c), not this obligation). Proof+commitments ≈ 1.4 MB/head ≈ 34 MB
total (before com dedup).

### 8.5 After validation

Persist driver to zkllm-src/, document as the next § in PHASE0_NOTES.md, re-run ALL
drivers' selftests (header untouched ⟹ rebuilds unnecessary, but they are cheap and
the rule stands).

---

## 9. Open questions / risks (explicit, none load-bearing for starting implementation)

1. **Domain margin vs real scores (the one real unknown).** The ±8-exponent domain
   assumes |scores_real| < 1024 (estimate of worst case ≈ 576). **Gate before
   freezing LOW_E:** run the integer pipeline once and record max|z_| over all 24
   heads. If margin < 2×, the pinned knob is the total score-rescale factor: shifting
   2^23 → 2^24 (stages 2^13·2^11; z_ scale 2^8) doubles the real-unit domain to ±16
   exponents at the same LEN_E = 2^20, at the cost of E_max ≈ 2^16·e^16 < 2^40 ⟹
   S < 2^50, r1 < 2^51 ⟹ limbs become 3 × 17-bit (LEN_R = 2^17, 6 planes padded to
   8, D_L = 2^23) — workable but uglier; only do it if the measurement demands it.
2. **Exp-table reproducibility.** Treated as a registered public input (sha256), not
   regenerated independently (§7.4). Residual risk: none for verification (both sides
   load the same file); only table *generation* is environment-sensitive in
   principle.
3. **int32→int64 widening between the two rescale stages.** zkob_rescale emits int32
   (save_int) but consumes int64 (load_long_tensor). Pinned resolution: orchestrator
   widens the data file (lossless; commitments — the trust carriers — are unaffected).
   Alternative (don't, for now): teach zkob_rescale an int32-input flag; that edits a
   validated driver for zero soundness gain.
4. **Two same-labeled fs_hadamard instances back-to-back (V1, V2).** Positionally
   distinct in the transcript and separated by the "c1"/"c2" absorbs; rmsnorm
   precedent already runs multiple instances on one transcript. No action; noted so
   nobody "fixes" it by editing the shared header.
5. **Verifier-side public-weight folds.** The verifier folds three public 2^20
   tensors (W_rs, W1, and conceptually MK inside them) with k_fr_fold/k_fr_emul —
   the same recompute-public-side pattern as B_f/T_f but a new *usage*; the small
   selftest cases (a)–(c) exercise it at toy scale where it is cross-checkable
   against multi_dim_me (add the evil==0 convention sanity checks exactly like the
   other drivers: every fold terminal == multi_dim_me of the corresponding tensor).
6. **Mask MLE closed form.** [j ≤ i] has an O(log) closed-form MLE (bitwise
   recursion), which would let the verifier skip building the 2^20 mask tensor.
   Rejected for now: the fold is validated machinery and costs milliseconds; the
   closed form is new algebra. Optimization note only.
7. **The pipeline's /128 temperature is load-bearing.** If anyone "fixes"
   m68-pipeline.py's ACCU_LOGSF/VALUE_LOGSF mixing (making the temperature 8), the
   honest exponent range grows 16× and this design's single-table premise breaks —
   it would need the upstream-style multi-segment decomposition (zksoftmax.cu's
   bs = {2^8, 2^20, 2^20} product-of-segments). Pinned: the pipeline is frozen as-is
   for task #12; revisit only with a pipeline change.
8. **P@V matmul orientation** (values matmul consumes P as activation X with
   com_W = chained V commitment over gen-64 columns) is item (c)'s design, including
   how V's per-head commitment is produced; this document only fixes com_P and the
   P chain file format (int32, scale 2^16, §7.1).
9. **S = 0 impossibility is domain-dependent.** Currently guaranteed (X_E ≥ 22 on the
   whole domain + diagonal unmasked). If the domain knob of (1) is ever turned so far
   that round(2^16·e^(LOW_E/2^16)) = 0, rows whose unmasked entries all sit at the
   table floor could reach S = 0 and become unprovable (honest throw). Invariant to
   re-check whenever LOW_E changes: X_E[0] ≥ 1.
