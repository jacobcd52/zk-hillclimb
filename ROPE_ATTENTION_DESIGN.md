# ROPE_ATTENTION_DESIGN.md — final design for the attention chain's missing links

Status: DESIGN FINAL (2026-06-10). Companion to SOFTMAX_DESIGN.md (implemented as
`zkob_softmax.cu`, PHASE0 §15, selftest 122/122 + audit SOUND) and to the chain/edge
formalism of orchestrator/ORCHESTRATOR_DESIGN.md §3–4. Read PHASE0_NOTES.md first for
all pinned conventions (FS transcript, FS-IPA, logUp, Montgomery/integer-view rules,
-dlto gotchas). This document is written so the implementer makes **no design
decisions**: every relation, scale, bound, absorb, file, CLI argument and selftest case
is pinned here.

Scope: the last unproven segment of the per-layer attention chain —

```
q/k/v_proj zkob_fc + rescale (validated drivers, instances not yet run)
  → RoPE on Q,K                      [UNBOUND — designed here: zkob_rope + rescale]
  → per-head slicing of Qr, Kr, V    [UNBOUND — designed here: zkob_headslice]
  → scores matmul per head           [zkob_fc, unmodified — operand semantics pinned here]
  → scores rescale 2^13, 2^10        [zkob_rescale; pinned in SOFTMAX_DESIGN §1.1/§7.3]
  → zkob_softmax                     [done]
  → values matmul per head           [zkob_fc, unmodified — operand semantics pinned here]
  → values rescale 2^16              [zkob_rescale instance, pinned here]
  → head merge + line-157 permutation [UNBOUND — designed here: zkob_headmerge]
  → com_attn_out                     [closes ORCHESTRATOR_DESIGN §4 edge S1's OPEN BOUNDARY]
```

**There is no output projection in this chain.** The task framing lists "output
projection (zkob_fc)" as the final link, but m68-pipeline.py never applies o_proj
(its python attention math, lines 137–159, goes straight from the values matmul to
`save_int(attn, …, attn_out)`; the pipeline itself records
`"note_o_proj": "zkLLM's released per-layer pipeline does not prove o_proj"`, line 205),
and the frozen harness manifest **waives** all three `layer{l}.attn.o_proj.*` ids with
reason "zkLLM upstream omits o_proj". The chain therefore closes at the attention
output tensor itself — which is *not* the naive head-concat: lines 156–157 apply a
nontrivial public permutation to it (§1.3). Binding that permutation is part of
deliverable (B); without it the chain cannot reach `com_attn_out` byte-identically.

## 0. Executive summary

Three new obligation drivers, **zero new CUDA kernels** (constraint satisfied in the
strongest form — same claim, same justification as zkob_softmax: every prover/verifier
step is composed from `fs_hadamard`, `build_eq_tensor`, `k_fr_emul`, `k_fr_fold`,
`open_prove`/`open_verify` (FS-IPA), `me_weights`, `fold_chain`, the h_* 1-thread
helpers, and `Commitment::commit`; `zkob_lookup.cuh` and `vrf_common.cuh` are used
as-is and **must not be edited**):

1. **`zkob_rope`** — binds the integerized RoPE relation
   `Y64 = Q ⊙ C + (σ ⊙ Q∘flip) ⊙ S` on the full 1024×768 tensor at once (RoPE is
   head-independent, §1.2), where C, S are **public sha256-registered integer cos/sin
   tables** at scale 2^16 and `flip` is the rotate_half permutation `e ↦ e⊕32`. The
   permuted factor needs **no extra commitment**: rotate_half is a signed
   bit-structured permutation, so its MLE at a point is the original MLE at the point
   with column-coordinate 5 replaced by its complement — the terminal opens the SAME
   chained `com_Q` at the modified point (§4.1, checked against the pinned LSB-first
   conventions in §1.4). One `zkob_rescale` run (sf = 2^16, the validated 2.6 s/3.3 s
   shape) closes the single rounding. Exact integer relation per element, every
   rounding proven exact, zero advice, **0 covert bits** (§3).

2. **`zkob_headslice`** — binds, per layer, all 36 per-head operand commitments
   (`Q_h`, `K_h^T`, `V_h` × 12 heads) to the chained full-tensor commitments
   `com_Qr`, `com_Kr`, `com_Vr` by paired IPA openings at one FS challenge: a head
   slice of a 64-aligned column block is the full tensor's MLE with the four
   head-selector column bits (bits 6–9, LSB-first) fixed to the Booleans of h — the
   softmax L-plane-opening pattern — and the K-transpose is free (a coordinate
   reordering of the same opening point, §4.3). This is what makes the per-head
   scores/values matmuls runnable on **unmodified zkob_fc**: com_W there is exactly
   the slice commitment this obligation pins (§4.4–4.5). No orientation is
   impossible; no driver changes anywhere.

3. **`zkob_headmerge`** — binds the 12 per-head outputs to `com_attn_out` *including
   the pipeline's line-156/157 double transpose+reshape*, which is a genuine entry
   permutation π of the (1024, 768) grid (derivation in §1.3 — this is the
   off-by-reshape trap; the permutation is in the frozen pipeline and must be bound,
   not "fixed"). One 20-variable sumcheck whose public weight is the π-gathered eq
   tensor splits into 12 per-head public-weight hadamards — concat and permutation
   bound in one shot, no intermediate commitment (§4.6).

Instance granularity: rope per (layer, tensor∈{Q,K}) = 4 instances; headslice and
headmerge per layer = 2 + 2; scores/values fc and values rescale per (layer, head) =
24 + 24 + 24. Composed under the frozen manifest ids `layer{l}.attn.scores_matmul`
and `layer{l}.attn.values_matmul` (composition precedent: mlp.swiglu, §4.0). Total
attention segment: **168 driver transcripts per forward pass**, predicted
**≈ 8.3 min prove / ≈ 9.5 min verify** (§8.6).

---

## 1. Pipeline semantics (m68-pipeline.py, quoted, exact)

Constants (lines 29–32): `LOG_SF = 16`, `VALUE_LOGSF = 16`, `ACCU_LOGSF = 20`.
llama-68m (config.json verified): embed = 768, n_heads = 12, head_dim = 64,
n_kv = n_heads (MHA), seq = 1024, 2 layers, **no `rope_theta` key and no
`rope_scaling` key** ⟹ transformers defaults: θ = 10000.0, rope_type "default",
attention_scaling = 1.0.

### 1.1 The RoPE code path (lines 50–53 and 137–146, quoted)

```python
def rotate_half(x):                                   # lines 50-53
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)
...
rotary = model_gpu.model.rotary_emb                   # line 109
...
Q = load_int("temp_Q.bin").reshape(seq, embed) / (1 << 16)    # line 137: int32@2^16 → float32
K = load_int("temp_K.bin").reshape(seq, embed) / (1 << 16)
V = load_int("temp_V.bin").reshape(seq, embed) / (1 << 16)
Q = Q.view(seq, n_heads, head_dim).transpose(0, 1)            # lines 140-142: (12, 1024, 64)
K = K.view(seq, n_heads, head_dim).transpose(0, 1)
V = V.view(seq, n_heads, head_dim).transpose(0, 1)
pos = torch.arange(seq, device=0).unsqueeze(0)                # line 143
cos, sin = rotary(torch.randn(1, seq, embed, device=0), pos)  # line 144: (1, 1024, 64) float32
Q, K = Q * cos + rotate_half(Q) * sin, K * cos + rotate_half(K) * sin   # line 145
Q, K = Q.to(torch.float64), K.to(torch.float64)               # line 146
```

Reading off the exact semantics:

1. **Rotate convention: LLaMA rotate_half (block-halved), NOT interleaved.**
   `rotate_half(q)[d] = −q[d+32]` for d ∈ [0,32), `= +q[d−32]` for d ∈ [32,64).
   Equivalently, with `flip(d) = d ⊕ 32` (an involution on [0,64)) and sign
   `σ(d) = −1 if d < 32 else +1`: `rotate_half(q)[d] = σ(d)·q[flip(d)]`.
2. **Frequency/position math** (transformers 4.57 `LlamaRotaryEmbedding`, default
   rope): `inv_freq[k] = θ^(−2k/64) = 10000^(−k/32)` for k ∈ [0,32);
   `freqs[t,k] = t·inv_freq[k]`; `emb = cat(freqs, freqs)` ⟹ the angle at lane d is
   `ang(t,d) = t·inv_freq[d mod 32]`; `cos = emb.cos()·1.0`, `sin = emb.sin()·1.0`,
   computed in **float32** (autocast disabled, `.float()` casts), shape
   (1, seq, 64) — the `x` argument (line 144's randn) supplies only device/dtype.
3. **Position ids are exactly 0..1023** (line 143). The tables depend only on
   (seq, head_dim, θ); they are **the same for every head and both layers** (cos/sin
   broadcast over the head axis of the (12, 1024, 64) Q in line 145).
4. **RoPE is applied in float32 at the real-value scale** (Q was divided by 2^16 at
   line 137), then cast to float64 for the scores matmul. The pipeline never
   integerizes roped Q/K: the only rounding on this path is at the *scores*,
   `A = to_int64(Q @ K.transpose(-2,-1), 16)` (line 147). An integer proof therefore
   cannot bind line 145 literally; exactly as with rmsnorm's R and softmax's P, the
   obligation binds an **integerized RoPE specification** of the same real function
   (§2), and the integer spec replaces the float path for downstream continuity (the
   pinned witness-authority rule, ORCHESTRATOR_DESIGN §3).

### 1.2 Head layout of Q/K/V (lines 140–142) — which columns belong to head h

`view(seq, 12, 64).transpose(0,1)` means head h of position t reads the (seq, embed)
tensor at columns `e = 64h + d`, d ∈ [0,64). So in the flat (row-major, B×C = 1024×768,
padded to 1024×1024) layout used by every commitment in the chain:

- flat index = `t·1024 + e` (after padding); LSB-first bits: **e = bits 0–9
  (d = bits 0–5, h = bits 6–9), t = bits 10–19**. Padded columns e ∈ [768, 1024) are
  h ∈ {12..15} — never sliced, zero in every honest tensor.
- `flip(e) = e ⊕ 32` flips **bit 5 only** — it stays inside the head (and inside the
  padding block when e ≥ 768, since 64 | 768 and 64 | 1024). The head-selector bits
  6–9 are untouched. This is what makes both coordinate hints of §0 true *in the
  pinned bit order*; verified against `open_verify`'s convention in §1.4.

### 1.3 The output assembly (lines 154–159) and the line-157 permutation

```python
attn = (torch.exp(...)).float() * ~mask                       # line 154: (12,1024,1024) float
attn = fromto_int64(attn @ V, VALUE_LOGSF)                    # line 155: (12,1024,64), integer-valued floats @2^16
attn = attn.transpose(0, 1).contiguous().view(seq, embed)     # line 156: (1024, 768) head-concat M
attn = attn.transpose(0, 1).reshape(seq, embed)               # line 157: PERMUTATION (see below)
save_int(attn, 1 << 16, "temp_attn_out.bin")                  # line 158
save_int(attn, 1 << 16, attn_out)                             # line 159
```

Line 156 produces the natural head-concat `M[t, 64h+d] = out_h[t, d]`, shape
(1024, 768). Line 157 is **not** a no-op: `M.transpose(0,1)` is the (768, 1024)
matrix `M^T`; `.reshape(1024, 768)` flattens `M^T` row-major (forcing a copy — the
transpose view is non-contiguous) and re-chops into 1024 rows of 768. With
seq = 1024 ≠ embed = 768 this is a genuine entry permutation:

```
O2[i, j] = M^T.flatten()[i·768 + j] = M[(i·768 + j) mod 1024, (i·768 + j) div 1024]
```

i.e. with m = i·768 + j ∈ [0, 786432): row t = m mod 1024, column e = m div 1024
(< 768 always ✓). Define **π : (i,j) ↦ (t,e)** on the real index set and
**π⁻¹ : (t,e) ↦ (i,j)** by `m = e·1024 + t; i = m div 768; j = m mod 768`. Sanity:
O2[0,0] = M[0,0]; O2[0,1] = M[1,0] (walks down M's column 0). This is almost
certainly an upstream accident (the double transpose), but it is **in the frozen
pipeline**, the orchestrator already replicates it verbatim (ORCHESTRATOR_DESIGN §3:
"including the … line-156/157 double transpose+reshape"), and the saved
`attn_out` — the tensor `attn_skip.add` consumes and the open-boundary
`com_attn_out` commits — is the **post-π** tensor. We bind what the pipeline
computes (precedent: the /128 temperature, SOFTMAX_DESIGN §1.4). §4.6 binds π.

Note `fromto_int64(x,16) = round(x·2^16)/2^16` (float64), so line 155's `attn` holds
exact integers/2^16 and line 158/159's `save_int(·, 2^16)` recovers them exactly:
attn_out's integers are the values-matmul integers, permuted by π. In the integer
chain those integers are produced by `zkob_rescale(sf 2^16)` on the exact
P@V products — rounding semantics `floor((x+2^15)/2^16)` (round-half-up) versus
torch's round-half-even on exact .5 ties: the same pinned authority deviation already
accepted for q/k/v (ORCHESTRATOR_DESIGN §3) and softmax's P. Not a new decision.

### 1.4 Coordinate-hint verification against the pinned conventions

Both hints from the task were checked against the actual machinery
(`open_prove`/`open_verify` in zkob_lookup.cuh, `me_weights`/`fold_chain` in
vrf_common.cuh):

- `open_verify(com, gen, G, Q, u_pt, eval, …)`: `u_pt[0..logG)` are **column**
  coordinates (paired LSB-first with the intra-row index by `me_weights`: bit i of
  the index pairs with `u_col[i]`); `u_pt[logG..)` fold the commitment **rows**, also
  LSB-first (`fold_chain` orientation 0 pairs the first row challenge with the row
  index's bit 0). So flat-index bit i ↔ u_pt[i], exactly the convention zkob_softmax
  used for its Boolean L-plane bits (SOFTMAX_DESIGN §4.6).
- **rotate_half hint** ✓: for the padded grid tensor `Qx[t,e] := Q[t, e⊕32]` (a
  bijection of the padded index set, §1.2), the multilinear extension satisfies
  `Q̃x(u) = Q̃(u₀,…,u₄, 1−u₅, u₆,…,u₁₉)` — substitute the involution into the MLE sum;
  only the bit-5 eq-factor changes, `(b₅ ? u₅ : 1−u₅)` ↦ `(b₅ ? 1−u₅ : u₅)`. The
  constant/affine modification `1−u₅` is an ordinary field element for
  `fold_chain`/IPA. **Sign handling:** the σ sign is folded into the *public* sin
  weight (§2), so the witness-side factor is the unsigned permuted tensor — no
  signed permutation machinery needed anywhere.
- **head-slice hint** ✓: `Q̃_h(v_d, v_t) = Q̃(v_d ‖ bits(h) ‖ v_t)` with
  bits(h) = (h₀,h₁,h₂,h₃) ∈ {0,1}⁴ LSB-first at u_pt positions 6–9, because the
  Boolean eq-factors select exactly the columns with head bits = h. For the slice
  tensors' own layouts: `Qh` (1024×64) has flat = t·64+d ⟹ u_pt = (v_d[0..6) ‖
  v_t[0..10)); `KhT` (64×1024) has flat = d·1024+t ⟹ u_pt = (v_t[0..10) ‖ v_d[0..6))
  — **the transpose is a pure reordering of the opening-point coordinates**, no
  extra machinery (§4.3).

### 1.5 The integer chain (replacing lines 137–159), per layer

All scales explicit; `≡` marks the chain byte-equality edges (full table in §7.4):

```
attn_in (input_norm output, int32@2^16, com ≡ N/yrescale/com_Xr)
→ zkob_fc q/k/v_proj (B=1024, IN=768, OUT=768; com_W = REGISTERED)   → Q64/K64/V64 @2^32 (i64)
→ zkob_rescale sf 2^16 (D=2^20, N=2^16, gen1024)                     → Q/K/V int32 @2^16
→ zkob_rope on Q and on K (public tables C,S @2^16; §2, §4.1)        → Qr64/Kr64 @2^32 (i64)
→ zkob_rescale sf 2^16                                               → Qr/Kr int32 @2^16
→ zkob_headslice (Qr, Kr, V → 12× {Qh, KhT, Vh}; §4.3)               → slice files + slice coms
→ per head: zkob_fc scores (B=1024, IN=64, OUT=1024; X=Qh, W=KhT)    → z.i64 @2^32
→ zkob_rescale 2^13 then 2^10 (widening shim between; SOFTMAX §7.3)  → z_ int32 @2^9
→ zkob_softmax                                                       → P int32 @2^16
→ per head: zkob_fc values (B=1024, IN=1024, OUT=64; X=P, W=Vh)      → out64.i64 @2^32
→ zkob_rescale sf 2^16 (D=2^16, N=2^16, gen64)                       → out_h int32 @2^16
→ zkob_headmerge (12× out_h → O2 = π(concat); §4.6)                  → attn_out int32 @2^16,
                                                                        com_O2 ≡ com_attn_out
```

This closes SOFTMAX_DESIGN §9.8 (the P@V orientation question) and discharges
HANDOFF item (c) ("per-head attention matmuls via zkob_fc … com_W = prover-supplied
chained activation commitment") with the slicing binding that makes "chained
activation commitment" actually mean something.

---

## 2. Integer RoPE specification + table generation

### 2.1 Public tables (registered like the exp table)

Two int32 files of seq × head_dim = 1024 × 64 = 65,536 entries (256 KB) each:

```
C_tab[t, d] = rint(2^16 · cos(t · 10000^(−(d mod 32)/32)))
S_tab[t, d] = rint(2^16 · sin(t · 10000^(−(d mod 32)/32)))
```

t ∈ [0, 1024), d ∈ [0, 64). Values in [−65536, +65536] (cos(0) = 1 ⟹ exactly
2^16 = 65536 at t = 0 and wherever the angle rounds to 1.0 — int32 is required, the
extremes do not fit int16). Scale **SCALE_R = 2^16**, pinned.

Generation script (the ONLY code artifact specified by this document), run once,
output registered by sha256 in public.json next to the swiglu and softmax-exp tables
— `gen_rope_tables.py`:

```python
# Pinned cos/sin table generator for zkob_rope (ROPE_ATTENTION_DESIGN.md section 2).
# Run ONCE; register both outputs by sha256 in public.json. The sha256 registration,
# not regeneration, is the source of truth; the C++ driver never generates the real
# tables (the selftest's in-driver fallback is flagged non-authoritative).
import numpy as np
SEQ, HEAD_DIM, THETA, SCALE = 1024, 64, 10000.0, 1 << 16
half = HEAD_DIM // 2
inv_freq = THETA ** (-np.arange(half, dtype=np.float64) / half)        # 10000^(-k/32)
ang = np.arange(SEQ, dtype=np.float64)[:, None] * inv_freq[None, :]    # (1024, 32)
ang = np.concatenate([ang, ang], axis=1)                               # (1024, 64) = cat(freqs, freqs)
np.rint(SCALE * np.cos(ang)).astype(np.int32).tofile("rope-cos-table.bin")
np.rint(SCALE * np.sin(ang)).astype(np.int32).tofile("rope-sin-table.bin")
```

Deviation note (same class as the exp table's §7.4 note): the script computes angles
and cos/sin in **float64**; the pipeline computes them in **float32** (§1.1.2). The
integer spec is the registered table — pipeline-float fidelity is an approximation-
quality question (a few table entries may differ by ±1 ulp at 2^16 from a
float32-faithful table), not a soundness question: both prover and verifier load the
same sha256-pinned file. The tables depend on (seq, head_dim, θ) only — **one pair
serves all 12 heads, both layers, and both of Q and K**; regenerate only if seq
changes (they are position-indexed by t = row index, valid because line 143 pins
position ids to 0..seq).

### 2.2 The exact integer relation (the spec the orchestrator and driver both compute)

Inputs: T ∈ {Q, K} as the chained int32 tensor at scale 2^16, shape 1024×768
(padded to 1024×1024 with zeros for commitment). Define on the padded grid, for
t ∈ [0,1024), e ∈ [0,1024):

```
W1[t,e] = [e < 768] · C_tab[t, e mod 64]                       (public, int32)
W2[t,e] = [e < 768] · σ(e) · S_tab[t, e mod 64],  σ(e) = (e & 32) ? +1 : −1
flip(e) = e ⊕ 32                                                (involution; §1.2)

(General driver rule, needed for the §8.1 toy shapes: the flip value is HD/2, the
flipped column bit is fb = log2(HD) − 1, σ(e) = (e & (HD/2)) ? +1 : −1, and
"mod 64" reads "mod HD"; at real scale HD = 64 ⟹ flip value 32, fb = 5.)

(R-ROPE)   Y64[t,e] = T[t,e]·W1[t,e] + T[t, flip(e)]·W2[t,e]    (exact int64, scale 2^32)
(R-RND)    Tr[t,e]  = floor( (Y64[t,e] + 2^15) / 2^16 )         (scale 2^16, int32)
```

R-RND is exactly `zkob_rescale` semantics (X = sf·X̂ + rem, rem ∈ [−2^15, 2^15),
sf = 2^16) — the one rounding on this path, proven exact by the validated rescale
machinery. R-ROPE has **no rounding at all** (exact integer multiply-add). Element
check against line 145 (d = e mod 64): d < 32 ⟹ `q'= q[d]·cos − q[d+32]·sin` ✓;
d ≥ 32 ⟹ `q' = q[d]·cos + q[d−32]·sin` with cos/sin at lane d = lane d−32's angle ✓.

Padding self-consistency: for e ∈ [768, 1024), flip(e) ∈ [768, 1024) (64 | 768), so
Y64's padding is 0·W1 + 0·W2 = 0 regardless — and W1 = W2 = 0 there anyway (belt and
suspenders; both prover and verifier build the weights with the same [e<768] rule).

Witness authority (pinned, ORCHESTRATOR_DESIGN §3 rule): the integer spec above
**replaces** line 145's float path — the same move as rmsnorm's R and softmax's P.
The driver computes Y64 from (T, C_tab, S_tab) itself (it must, to commit it) and
emits the chain file; the orchestrator may cross-check with the identical formula.
Downstream (scores) consumes Tr; the drift vs the float pipeline is one ≤0.5-ulp
rounding per Q/K entry at 2^16 — negligible against the 3.7× score-domain margin
(SCORES_RANGE.md headline: max|score| = 276.7 vs domain 1024; the RoPE rounding
perturbs scores by ≲ 64·(0.5/2^16)·(|q|+|k|) ≪ 1).

---

## 3. Advice-binding analysis (covert capacity)

**All three new obligations have zero prover advice; 0 covert bits.** Accounting:

- `zkob_rope`: Y64 is a deterministic function of the chained T and the registered
  tables (R-ROPE has no rounding); the rescale's rem is exactly range-bound (the
  validated rescale closes the ±sf compensation channel — PHASE0 §11's semantic
  forgery). The absorbed scalars ev, c1, c2 are forced by IPA openings/sumchecks
  against com_Y64 and com_T (§4.1) — no choice anywhere. **0 bits.**
- `zkob_headslice`: the 36 slice tensors are deterministic gathers of chained
  tensors. Each claimed eval eQ/eK/eV is checked against **two** commitments (the
  slice's and the full tensor's) at challenge-derived points — forced on both sides.
  **0 bits.**
- `zkob_headmerge`: O2 is a deterministic permutation of the chained out_h; ev and
  the 12 c_h are forced by openings/sumchecks against com_O2 / com_O{hh}; the
  sumcheck identity even forces O2's padding columns to exact zero (§4.6). **0 bits.**
- The zkob_fc and zkob_rescale instances contribute 0 bits as before (exact affine
  links, exactly range-bound rem, terminals pinned by openings).

The only nondeterminism in the whole segment is *completeness* failure (the §6
honest-prover throws), never *choice*. Combined with zkob_softmax's 0 bits, the
attention chain adds **0 covert bits per forward pass**.

---

## 4. Proof obligations

### 4.0 Instance granularity and manifest composition

The frozen manifest has **no ids for rope/slice/merge** (and o_proj is waived). They
compose as sub-obligations under existing non-waived ids, exactly the mlp.swiglu
precedent (= glu + hidden-rescale) and the softmax precedent (= softmax + two score
rescales under `layer{l}.attn.softmax`):

| manifest id (frozen) | composed driver runs (seed suffix per run) |
|---|---|
| `layer{l}.attn.{q,k,v}_proj.matmul` + `.commitment_opening` | one zkob_fc run (`…{q,k,v}_proj.matmul`; IPA(W) vs REGISTERED com discharges the opening id) |
| `layer{l}.attn.{q,k,v}_proj.rescaling` | zkob_rescale sf 2^16 (`…{q,k,v}_proj.rescaling`) |
| `layer{l}.attn.scores_matmul` (per_head 12) | zkob_rope ×2 (`…rope.q`, `…rope.k`) + zkob_rescale ×2 (`…rope.q.rescale`, `…rope.k.rescale`) + zkob_headslice (`…slice`) + zkob_fc ×12 (`…scores.h{hh}`) |
| `layer{l}.attn.softmax` | zkob_rescale ×24 (`…scores_rescale13.h{hh}`, `…scores_rescale10.h{hh}`) + zkob_softmax ×12 (`…softmax.h{hh}`) — pinned in SOFTMAX_DESIGN §4.0 |
| `layer{l}.attn.values_matmul` (per_head 12) | zkob_fc ×12 (`…values.h{hh}`) + zkob_rescale ×12 (`…values_rescale.h{hh}`) + zkob_headmerge (`…merge`) |
| `layer{l}.attn.o_proj.*` | WAIVED (manifest: "zkLLM upstream omits o_proj") — nothing runs |
| `layer{l}.attn_skip.add` | already covered; its `com_attn_out` OPEN BOUNDARY (edge S1) is **closed** by edge A15 (§7.4) |

RoPE composes under scores_matmul because it produces that matmul's operands and the
manifest is frozen (no new ids may be added; composition is the sanctioned
mechanism). The headmerge composes under values_matmul as the consumer of its
outputs. Obdir layout: `proofs/<manifest_id>/<sub>/` with sub names
`rope.q`, `rope.q.rescale`, `rope.k`, `rope.k.rescale`, `slice`, `fc.h{hh}` (scores),
`fc.h{hh}` / `rescale.h{hh}` / `merge` (values), matching ORCHESTRATOR_DESIGN §1.

Granularity rationale: rope on the **full tensor** (not per head) because RoPE is
head-independent (§1.2) — per-head rope would multiply work ×12 for nothing.
Headslice/headmerge **per layer** (not per head) so com_Qr/com_Kr/com_Vr are
re-committed once, not 12 times, and the chain has a single slice/merge edge fan-out;
the per-head fc/rescale/softmax instances keep the per-head granularity already
pinned by SOFTMAX_DESIGN §4.0.

### 4.1 zkob_rope — the RoPE obligation (binds R-ROPE)

One instance per (layer, T ∈ {Q, K}); B = 1024, C = 768, HD = 64, padded grid
2^20, single gen set gen1024, logD = 20.

Committed tensors (PLAIN form, committed row-wise with gen1024, zero-padded):
- `com_T` — the input (re-committed from the chain file; byte-identical with the
  {q,k}_proj rescale's com_Xr — orchestrator edge A3).
- `com_Y64` — the RoPE output at scale 2^32 (chains into the rope rescale's com_X).

The relation is bound as a two-term MLE identity at an FS challenge u (20 vars):

```
ev := Ỹ64(u)  ==  c1 + c2,   where
c1 = Σ_b eq(u,b)·W1(b)·T(b)          (hadamard 1: E = eq(u) ⊙ W1, S = T,  U = 𝟙)
c2 = Σ_b eq(u,b)·W2(b)·Tx(b)         (hadamard 2: E = eq(u) ⊙ W2, S = Tx, U = 𝟙)
```

with Tx(b) = T(flip-bit-5(b)) materialized by the prover as a host-permuted int
buffer (never committed). Both weights are **public** (verifier rebuilds W1, W2 from
its own registered table copies with the §2.2 formula — sign folded in host-side as
negative ints, mod-p via the FrTensor int ctor — then `build_eq_tensor(u)`,
`k_fr_emul`, and folds with the round challenges via the `k_fr_fold` chain — the
recompute-the-public-side pattern validated in softmax §4.3/4.4). Per hadamard the
verifier REQUIRES `U_f2 == 1` (load-bearing, as in softmax) and checks the terminal
`cur == W_f · S_f2 · U_f2`.

Terminal openings (3 IPAs):
- hadamard 1: `S_f2` opens T at `pt1 = reverse(ws1)` vs com_T (standard).
- hadamard 2: `S_f2` = T̃x(pt2), pt2 = reverse(ws2). By §1.4, T̃x(pt2) =
  T̃(pt2′) with `pt2′ = (pt2[0..5), 1 − pt2[5], pt2[6..20))`. The verifier computes
  pt2′ itself (h_scalar(F_ONE, pt2[5], 1)) and runs a **standard open_verify of
  com_T at pt2′ with eval = S_f2** — the rotate_half permutation is bound by opening
  the SAME commitment at the modified point; no commitment to Tx exists. The prover
  side is `open_prove(T_padded, …, pt2′)` (and asserts T̃(pt2′) == S_f2 —
  completeness guard).
- `ev` opens Y64 at u vs com_Y64.

Final verifier check (plain field): `c1 + c2 == ev` — by Schwartz–Zippel over
multilinears at the random u this is the tensor identity R-ROPE, given that c1/c2
are bound to com_T's actual content by the sumchecks + openings.

Soundness summary: hadamard 1 + IPA(T@pt1) pin c1 to (com_T, registered C table);
hadamard 2 + IPA(T@pt2′) pin c2 to (com_T∘flip, registered S table) — the flipped
opening is what makes a wrong permutation unprovable (selftest evil=2); IPA(Y64@u)
pins ev to com_Y64; `c1+c2==ev` ties them; the chain edges pin com_T upstream and
com_Y64 downstream; the rescale run closes R-RND exactly.

### 4.2 The rope rescale (existing driver, new instances)

`zkob_rescale` on Y64.i64, B = 1024, C = 768, sf = 2^16, gen1024 — bit-for-bit the
validated q/k/v-proj rescale shape (D = 2^20, N = 2^16, C_pad = 2^10 ≤ N ≤ D, N | D,
n1 = 4; measured 2.6 s / 3.3 s). Edges: `rope.{q,k}/com_Y64 ≡ rescale/com_X`,
`rescale/com_Xr ≡ slice/com_{Q,K}` (§7.4). No widening shim needed (zkob_rope emits
int64 like zkob_fc does).

### 4.3 zkob_headslice — the per-head slicing obligation

One instance per layer. Inputs: Qr, Kr, V chain files (1024×768 int32 @2^16).
NH = C/HD = 12 heads; nhb = log2(C_pad) − log2(HD) = 4 head-selector bits;
logB = 10, logHD = 6.

Slice tensors, derived (deterministically, no advice) and committed exactly as the
downstream zkob_fc will commit them (so the chain checks are byte-equalities /
same-file path bindings):

| tensor | shape | data | committed with | rows | = zkob_fc's |
|---|---|---|---|---|---|
| `Qh{hh}` | 1024×64 | `Qr[t, 64h+d]` | gen64 (`pad({1024,64})` = no-op) | 1024 | scores com_X |
| `KhT{hh}` | 64×1024 | `Kr[t, 64h+d]` at (d,t) | gen1024 (`pad({64,1024})` = no-op) | 64 | scores com_W |
| `Vh{hh}` | 1024×64 | `V[t, 64h+d]` | gen64 | 1024 | values com_W |

plus re-commits of the full `com_Q`, `com_K`, `com_V` (1024×1024-padded, gen1024,
1024 rows each — byte-identical with the upstream rescale com_Xr files).

The binding: one FS challenge `v` of logB + logHD = 16 variables, split
`v_d = v[0..6)`, `v_t = v[6..16)`. Per head h, three claimed evaluations, each
discharged by **two** IPA openings that must both verify against the SAME eval:

```
eQ_h:  open com_Qh{hh}  at (v_d ‖ v_t)            and  com_Q at (v_d ‖ bits(h) ‖ v_t)
eK_h:  open com_KhT{hh} at (v_t ‖ v_d)            and  com_K at (v_d ‖ bits(h) ‖ v_t)
eV_h:  open com_Vh{hh}  at (v_d ‖ v_t)            and  com_V at (v_d ‖ bits(h) ‖ v_t)
```

Point layouts verified in §1.4: the full-tensor points put bits(h) (LSB-first
Booleans of h) at u_pt positions 6–9; the KhT point is the same (v_d, v_t) pair with
the coordinate blocks swapped because KhT's flat index is d·1024 + t — **the
transpose costs nothing**. Equality of the two openings at the random (v_d, v_t)
forces (Schwartz–Zippel in 16 variables) the slice tensor to equal the head-h column
block of the full tensor, as integer tensors (both commitments are over plain ints
< 2^31 ≪ p; the MLEs agree as polynomials iff the tensors agree).

Per layer: 36 slice commitments + 3 full re-commits + **72 IPA openings**
(36 × gen-1024-sized [the com_Q/K/V and KhT openings], 24 × gen-64-sized
[Qh/Vh], wait — per head: com_Q/K/V side = 3 gen-1024 IPAs, KhT side = 1 gen-1024
IPA, Qh/Vh side = 2 gen-64 IPAs ⟹ 48 gen-1024 + 24 gen-64). The IPA count is the
cost driver — flagged with a measurement gate in §9.1; there is no validated
batching primitive, and per-head splitting would only redistribute the same work
while re-committing com_Q/K/V 12× each. The driver also writes the 36 slice data
files (`Qh{hh}.i32.bin`, `KhT{hh}.i32.bin`, `Vh{hh}.i32.bin`) for the fc runs —
prover-only witness files under `data/`, not trust-carrying.

Why this obligation exists / what it defeats: without it, the scores/values fc
instances accept ANY com_X/com_W the prover supplies — per-head tensors would be
completely unchained (the attention analog of rmsnorm's MINOR-5: a standalone fc
ACCEPT proves multiplication, not provenance). With it, every per-head operand is
pinned to the roped/chained full tensors at 4 random field coordinates' worth of
S-Z security per pair.

### 4.4 Scores matmul per head — unmodified zkob_fc, operand semantics pinned

`zkob_fc prove` with **B = 1024, IN = 64, OUT = 1024; X = Qh{hh}.i32 (activation),
W = KhT{hh}.i32; gen_in = gen64, gen_out = gen1024**; emits z.i64 (scale 2^32,
unpadded 1024×1024) and com_Y (1024 rows, gen1024) chaining into scores_rescale13.

**What com_W means here (the task's explicit question):** com_W is NOT a registered
weight — it is the **chained activation commitment of K_h^T**: 64 G1 points, point
d = ⟨gen1024, (Kr[t, 64h+d])_{t∈[0,1024)}⟩, i.e. the row-wise commitment of the
64×1024 transposed slice under gen1024, exactly what `gen_out.commit(W.pad({64,1024}))`
produces — and exactly what zkob_headslice committed and pinned to com_Kr (§4.3).
The orchestrator passes `proofs/…/slice/com_KhT{hh}.bin` as zkob_fc verify's
`<com_W.bin>` path argument (the same mechanism that passes registered weight
commitments; the verifier absorbs that file, so a prover whose K_h^T differs
diverges the transcript). **No orientation is impossible and no driver change is
needed**: the only cost of zkob_fc's row-major W layout is that K's slice must be
materialized *transposed* in the data file (a witness-file concern), and that its
commitment is column-orientation w.r.t. K — which is precisely what the headslice
obligation knows how to bind (§1.4's coordinate swap). The rejected alternative —
teaching zkob_fc a transposed-W mode — would edit a validated driver and force a
revalidation pass for zero soundness gain (same verdict as PHASE0 §9.3's rescale
flag): **do not do it**.

X-side: com_X (1024 rows, gen64) is byte-identical with `slice/com_Qh{hh}.bin`
(edge A6) because both commit the same unpadded 1024×64 ints with the same gens.

### 4.5 Values matmul per head + values rescale

`zkob_fc prove` with **B = 1024, IN = 1024, OUT = 64; X = P.i32 (softmax output,
com_X ≡ softmax com_P — the edge already pinned in SOFTMAX_DESIGN §4.7), W =
Vh{hh}.i32; gen_in = gen1024, gen_out = gen64**; com_W = the chained activation
commitment of V_h: 1024 points, point t = ⟨gen64, (V[t, 64h+d])_{d∈[0,64)}⟩ — the
headslice's com_Vh{hh}.bin, passed as the com_W path argument. Emits out64.i64
(1024×64, scale 2^32) → `zkob_rescale` sf 2^16, **gen64** (D = 2^16, N = 2^16,
C_pad = 64 ≤ N = D, n1 = 0 — the validated "pure phase2" lookup shape), output
out_h.i32 (1024×64 @2^16), com_Xr (1024 rows, gen64) chaining into the headmerge.

This requires **one new registration artifact: `gen64.bin` (`ppgen 64`)** — added to
ORCHESTRATOR_DESIGN §2 step 1 and hash-pinned in public.json like gen1024/gen4096.

### 4.6 zkob_headmerge — concat + line-157 permutation (binds π)

One instance per layer. Inputs: 12 out_h chain files (1024×64 int32 @2^16); output:
attn_out = O2 (1024×768 int32 @2^16, computed by the §1.3 formula exactly — host
integer gather, no arithmetic, no rounding).

Statement: `O2[i,j] = out_h[t,d]` where (t, e=64h+d) = π(i,j) per §1.3, for all real
(i,j); O2's padding = 0.

Committed: `com_O2` (1024×1024-padded, gen1024, 1024 rows — byte-identical with the
orchestrator's `com_attn_out`, closing edge S1's open boundary) and re-commits
`com_O{hh}` of each out_h (1024 rows, gen64 — byte-identical with the values-rescale
com_Xr, edge A14).

Binding: FS challenge u (20 vars over the padded O2 grid), claim `ev = Õ2(u)`, then
**12 public-weight hadamards** (16 vars each), one per head:

```
c_h = Σ_{(t,d)} Wm_h[t·64+d] · out_h[t·64+d],
Wm_h[t·64+d] = E_u[ π⁻¹(t, 64h+d) ]        (E_u = build_eq_tensor(u), 2^20 entries;
                                             π⁻¹(t,e): m = e·1024+t; i = m div 768;
                                             j = m mod 768; index = i·1024 + j)
```

run as fs_hadamard(E = Wm_h, S = out_h, U = 𝟙); terminal opens out_h at
reverse(ws_h) vs com_O{hh} (12 IPAs); the verifier rebuilds each Wm_h itself
(build_eq_tensor on device, one host copy, 12 gathers of 2^16 entries, upload, fold)
and requires U_f2 == 1 per head. Final: IPA of O2 at u vs com_O2 (eval = ev) and the
plain-field check **Σ_h c_h == ev**.

Why this binds everything at once: Σ_h c_h is the MLE at u of the tensor "π(concat)
on real entries, 0 on padding" (each real (i,j) is hit by exactly one (h,t,d) — π is
a bijection of the real index sets); ev is the MLE of the actual committed O2_pad.
Equality at random u ⟹ tensor equality whp ⟹ O2's real entries are the π-gathered
head outputs AND its padding columns are exactly zero. Concat order, the
permutation, and padding hygiene are all certified by one check; there is no
intermediate "M" commitment (line 156's tensor never materializes in the proof).

13 IPAs, 13 commitments, 12 cheap 16-round hadamards per instance.

### 4.7 Chain byte-equalities (orchestrator-level)

Full table in §7.4. The drivers re-commit every chained tensor from its input file
with the same gens and padding, so byte-identity holds automatically for honest runs
(validated mechanism, PHASE0 §12). As with softmax's com_z (SOFTMAX REVIEW MINOR-5),
**standalone ACCEPTs of rope/slice/merge bind their tensors only internally** — the
edges are the defense that pins them into the chain; the selftests therefore do not
duplicate cross-obligation tamper cases (chain-level tampering is the orchestrator
selftest's job, as in ORCHESTRATOR_DESIGN §7(b)).

---

## 5. FS schedules (one transcript per obligation; seed = "<run_seed>:<obligation_id>")

Absorb-by-absorb; labels in quotes are exact; `→ x` = derive challenge. Every IPA
internally absorbs its "L","R" points before each of its round challenges on the
same transcript (gen1024 ⟹ 10 rounds, gen64 ⟹ 6 rounds) — pinned open_prove/
open_verify behavior. The shared "hp0..hp3" hadamard labels across instances are
positionally disambiguated (rmsnorm/softmax precedent; the interleaved claim absorbs
disambiguate further) — do NOT "fix" the shared header.

### 5.1 zkob_rope (ids `layer{l}.attn.rope.{q,k}`)

```
absorb_u32  "B" 1024, "C" 768, "HD" 64, "SCALE_R" 16
absorb_g1_tensor "com_T"   com_T     (1024 rows)
absorb_g1_tensor "com_Y64" com_Y64   (1024 rows)
→ u = fs_challenge_vec(20)
absorb_fr "ev"                        (= Ỹ64(u))
absorb_fr "c1"                        (= Σ eq(u)·W1·T, prover-computed exactly)
hadamard 1, 20 rounds:                per round absorb "hp0".."hp3" → w
absorb_fr "S_f2","U_f2"              (h1 terminals)
IPA(T)   at pt1 = reverse(ws1)               → ipa_T1.bin   (vs com_T)
absorb_fr "c2"
hadamard 2, 20 rounds:                "hp0".."hp3" → w
absorb_fr "S_f2","U_f2"              (h2 terminals)
IPA(T)   at pt2' = flipbit5(reverse(ws2))    → ipa_T2.bin   (vs com_T; eval = h2 S_f2)
IPA(Y64) at u                                → ipa_Y.bin    (vs com_Y64; eval = ev)
[verifier-only, no absorbs: W1/W2 fold recomputes; U_f2 == 1 twice;
 terminal cur_i == W_f_i·S_f2_i·U_f2_i; c1 + c2 == ev]
```

Verifier round-count checks: each hadamard ev = 4·20; commitment rows 1024/1024;
dims.bin = {u32 B, u32 C, u32 HD, u32 SCALE_R} cross-checked against argv.

### 5.2 zkob_headslice (id `layer{l}.attn.slice`)

```
absorb_u32  "B" 1024, "C" 768, "HD" 64
absorb_g1_tensor "com_Q","com_K","com_V"            (1024 rows each)
for hh = 00..11:
  absorb_g1_tensor "com_Qh{hh}"  (1024 rows, gen64)
  absorb_g1_tensor "com_KhT{hh}" (64 rows, gen1024)
  absorb_g1_tensor "com_Vh{hh}"  (1024 rows, gen64)
→ v = fs_challenge_vec(16)        (v_d = v[0..6), v_t = v[6..16))
for hh = 00..11:
  absorb_fr "eQ{hh}" ; IPA(Qh)  at (v_d ‖ v_t)           → ipa_Qh{hh}.bin
                       IPA(Q)   at (v_d ‖ bits(h) ‖ v_t) → ipa_Qf{hh}.bin
  absorb_fr "eK{hh}" ; IPA(KhT) at (v_t ‖ v_d)           → ipa_Kh{hh}.bin
                       IPA(K)   at (v_d ‖ bits(h) ‖ v_t) → ipa_Kf{hh}.bin
  absorb_fr "eV{hh}" ; IPA(Vh)  at (v_d ‖ v_t)           → ipa_Vh{hh}.bin
                       IPA(V)   at (v_d ‖ bits(h) ‖ v_t) → ipa_Vf{hh}.bin
[verifier: each pair verifies BOTH IPAs against the SAME absorbed eval; bits(h) are
 the field constants {0,1} of h's 4 LSB-first bits at point positions 6..9]
```

{hh} = two-digit decimal head index baked into the label string. dims.bin =
{u32 B, u32 C, u32 HD}.

### 5.3 zkob_headmerge (id `layer{l}.attn.merge`)

```
absorb_u32  "B" 1024, "C" 768, "HD" 64
absorb_g1_tensor "com_O2"            (1024 rows, gen1024)
for hh: absorb_g1_tensor "com_O{hh}" (1024 rows each, gen64)
→ u = fs_challenge_vec(20)
absorb_fr "ev"                        (= Õ2(u))
for hh = 00..11:
  absorb_fr "c{hh}"
  hadamard hh, 16 rounds:             "hp0".."hp3" → w
  absorb_fr "S_f2","U_f2"
  IPA(out_h) at reverse(ws_hh)        → ipa_O{hh}.bin   (vs com_O{hh}; eval = S_f2)
IPA(O2) at u                          → ipa_O2.bin      (vs com_O2; eval = ev)
[verifier-only: 12 × (Wm_h gather+fold recompute; U_f2 == 1; terminal check);
 Σ c_hh == ev]
```

dims.bin = {u32 B, u32 C, u32 HD}. The eq tensor E_u is built ONCE (device), copied
to host once, gathered 12× — prover and verifier use the identical pinned π⁻¹
formula (§4.6); any divergence is caught by the weight-terminal check.

### 5.4 Existing drivers (for completeness; schedules unchanged)

zkob_fc and zkob_rescale run their validated schedules verbatim (PHASE0 §11–12,
zkob_fc.cu header). Seeds per §4.0's table. The scores rescales and softmax follow
SOFTMAX_DESIGN §5 verbatim.

---

## 6. Numeric bounds (with the arithmetic)

All committed values plain (non-Montgomery) Fr; host math int64-safe throughout:

- **rope inputs**: |T| < 2^31 (int32 file format — no tighter bound is assumed or
  needed). |W1|, |W2| ≤ 2^16. Each product |T·W| < 2^47; |Y64| < 2^48 — int64-safe
  (≪ 2^63) and ≪ p ≈ 2^255 **unconditionally**. Honest magnitudes: |T_real| =
  |q|,|k| ≲ tens ⟹ |T| ≲ 2^21, |Y64| ≲ 2·2^21·2^16 = 2^38.
- **rope rescale output**: |Tr| = |Y64|/2^16 < 2^32 only under the worst int32 input
  — honest-prover throw `|Y64| ≥ 2^47` (guards save_int's int32 output; honest data
  is ~2^38, margin 2^9; mirrors rmsnorm's M-throw style).
- **rope real-function bound**: |q'_real| ≤ |q[d]|·|cos| + |q[d̄]|·|sin| ≤ 2·max|q| —
  RoPE at most doubles entry magnitude; the downstream score bound is the measured
  one (next item), not derived from entry bounds.
- **scores**: |z| = |score_real|·2^32 with measured max|score_real| = 276.72
  (SCORES_RANGE.md, real-text envelope; 3.70× inside the exp-table domain 1024, and
  the integer-RoPE rounding shifts scores by ≪ 1 — §2.2) ⟹ |z| < 2^40.2, int64 ✓;
  rescale13 intermediate at scale 2^19: < 277·2^19 < 2^28 < int32 ✓ (the pinned
  stage order, SOFTMAX_DESIGN §1.1).
- **slices**: pure gathers of int32 values — no arithmetic; field fit trivial.
- **values matmul**: |out64| ≤ Σ_j P[i,j]·max|V| ≤ (2^16 + 512)·max|V_int|; honest
  |V_int| ≲ 2^21 ⟹ |out64| ≲ 2^37.5, int64 ✓; rescaled |out_h| ≲ 2^21.5, int32 ✓.
- **merge**: pure permutation of int32 values; field fit trivial.
- Sumcheck/IPA values are field elements by construction; eq/weight tensors are
  field-valued. Commitment row counts: rope 1024/1024; slice 1024×3 + (1024, 64,
  1024)×12; merge 1024 + 1024×12 — all powers of two, no odd-level padding anywhere.

Honest-prover throws (completeness guards): B ≠ 1024-style violations — pinned
checks per driver: B, C_pad powers of two; HD a power of two with HD | C (so flip
and head blocks stay aligned, §1.2) and HD | C_pad; gen sizes match (gen_big =
C_pad, gen_small = HD); NH = C/HD ≥ 2; rope: |Y64| ≥ 2^47; slice/merge: input file
sizes exact (short-read throws); merge: B ≠ C required for π to be the §1.3 formula
— NOT thrown (the formula is total; B = C makes π the plain transpose and is a
selftest case).

---

## 7. CLI, files, chain interface

### 7.1 CLI (same style as the other drivers; none of them mkdir the obdir)

```
zkob_rope prove   <obdir> <seed> <T-int32.bin> <B> <C> <HD>
                  <cos-int32.bin> <sin-int32.bin> <gen.bin> <q.bin> [Y64-i64-out.bin]
zkob_rope verify  <obdir> <seed> <B> <C> <HD>
                  <cos-int32.bin> <sin-int32.bin> <gen.bin> <q.bin>
zkob_rope selftest

zkob_headslice prove   <obdir> <seed> <Q-int32.bin> <K-int32.bin> <V-int32.bin>
                       <B> <C> <HD> <gen_big.bin> <gen_small.bin> <q.bin> [slice-out-dir]
zkob_headslice verify  <obdir> <seed> <B> <C> <HD> <gen_big.bin> <gen_small.bin> <q.bin>
zkob_headslice selftest

zkob_headmerge prove   <obdir> <seed> <Oh-prefix> <B> <C> <HD>
                       <gen_big.bin> <gen_small.bin> <q.bin> [O2-int32-out.bin]
zkob_headmerge verify  <obdir> <seed> <B> <C> <HD> <gen_big.bin> <gen_small.bin> <q.bin>
zkob_headmerge selftest
```

- `<T-int32.bin>`: unpadded B×C int32 @2^16 (the {q,k}_proj rescale's Xr output).
- `<cos/sin-int32.bin>`: the §2.1 registered tables, B×HD int32 each; the verifier
  loads its own registered copies (hash-checked against public.json by the
  orchestrator before any driver runs) and rebuilds W1/W2 itself. The driver checks
  file size == B·HD exactly.
- `[Y64-i64-out.bin]`: unpadded B×C **int64** chain file @2^32 (pad-stripped, like
  rmsnorm's W.i64) — feeds zkob_rescale's load_long_tensor directly, no shim.
- `[slice-out-dir]`: zkob_headslice writes `Qh{hh}.i32.bin` (B×HD), `KhT{hh}.i32.bin`
  (HD×B, transposed layout), `Vh{hh}.i32.bin` (B×HD) — prover-only witness files for
  the fc runs (orchestrator passes `data/layer{l}.attn.slice/`).
- `<Oh-prefix>`: zkob_headmerge reads `<Oh-prefix>{hh}.i32.bin` for hh = 00..NH−1
  (B×HD int32 @2^16 — the values-rescale Xr outputs). `[O2-int32-out.bin]`: unpadded
  B×C int32 @2^16 = the attn_out chain file (consumed by the orchestrator's skip
  add; the verifier never reads it — O2's integers flow through com_O2).
- gens: `gen_big` = C_pad (= 1024, the existing gen1024.bin), `gen_small` = HD
  (= 64, **NEW: gen64.bin via `ppgen 64`**, registered + hash-pinned). `q.bin` as
  everywhere.
- Each prover writes dims.bin (§5 layouts); each verifier cross-checks against argv.

### 7.2 Files in <obdir>

```
zkob_rope (9 files, ≈ 295 KB):
  dims.bin  com_T.bin  com_Y64.bin  ev.bin (1 Fr_t)
  hp1.bin hp2.bin (HadamardProof; claim_H = c1/c2)
  ipa_T1.bin ipa_T2.bin ipa_Y.bin

zkob_headslice (113 files, ≈ 4.0 MB):
  dims.bin  com_Q.bin com_K.bin com_V.bin            (144 KB each)
  com_Qh{00..11}.bin (144 KB)  com_KhT{00..11}.bin (9 KB)  com_Vh{00..11}.bin (144 KB)
  evals.bin (36 Fr_t: eQ00,eK00,eV00, eQ01, …)
  ipa_Qh{hh}.bin ipa_Qf{hh}.bin ipa_Kh{hh}.bin ipa_Kf{hh}.bin ipa_Vh{hh}.bin ipa_Vf{hh}.bin

zkob_headmerge (40 files, ≈ 1.9 MB):
  dims.bin  com_O2.bin (144 KB)  com_O{00..11}.bin (144 KB each)
  ev.bin (1 Fr_t)  hp{00..11}.bin  ipa_O{00..11}.bin  ipa_O2.bin
```

(Serialized G1Jacobian_t = 144 B ⟹ 1024-row com = 144 KB, 64-row com = 9 KB.
evals.bin/ev.bin have no count header — raw Fr_t sequences, same as lvals.bin.)

### 7.3 Chain wiring per layer (orchestrator; one head hh shown)

```
zkob_fc prove …q_proj  X=attn_in B=1024 IN=768 OUT=768 gen1024 gen1024 → Q64.i64, com_Y
zkob_rescale prove …q_proj.rescaling Q64.i64 1024 768 16 gen1024 q → Q.i32
zkob_rope prove <ob> <seed:…rope.q> Q.i32 1024 768 64 rope-cos-table.bin
               rope-sin-table.bin gen1024 q Qr64.i64
zkob_rescale prove <ob> <seed:…rope.q.rescale> Qr64.i64 1024 768 16 gen1024 q Qr.i32
   [same two runs for K; V has no rope]
zkob_headslice prove <ob> <seed:…slice> Qr.i32 Kr.i32 V.i32 1024 768 64
               gen1024 gen64 q data/layer{l}.attn.slice/
zkob_fc prove <ob> <seed:…scores.h{hh}> Qh{hh}.i32 KhT{hh}.i32 1024 64 1024
              gen64 gen1024 q z.i64
   [zkob_fc verify gets <com_W.bin> = proofs/…/slice/com_KhT{hh}.bin]
zkob_rescale …scores_rescale13.h{hh} z.i64 1024 1024 13 gen1024 q z13.i32
   [widen_i32_to_i64 shim]                                    (SOFTMAX_DESIGN §7.3)
zkob_rescale …scores_rescale10.h{hh} z13.i64 1024 1024 10 gen1024 q z_.i32
zkob_softmax …softmax.h{hh} z_.i32 1024 1024 -524288 1048576 softmax-exp-table.bin
             1048576 gen1024 q P.i32
zkob_fc prove <ob> <seed:…values.h{hh}> P.i32 Vh{hh}.i32 1024 1024 64
              gen1024 gen64 q out64.i64
   [zkob_fc verify gets <com_W.bin> = proofs/…/slice/com_Vh{hh}.bin]
zkob_rescale …values_rescale.h{hh} out64.i64 1024 64 16 gen64 q out{hh}.i32
zkob_headmerge prove <ob> <seed:…merge> data/…/out 1024 768 64 gen1024 gen64 q attn_out.i32
   [attn_out.i32 → skip add; com_O2 ≡ com_attn_out]
```

### 7.4 The chain byte-equality map (every new edge; ORCHESTRATOR_DESIGN §4 format)

`≡` = byte-identical files; per layer l; hh ranges over 00..11; N = the layer's
input_norm site. Sub-paths under `proofs/layer{l}.attn.scores_matmul/` (= `SM/`),
`…softmax/` (= `SX/`), `…values_matmul/` (= `VM/`):

| # | edge | binds |
|---|---|---|
| A1q/k/v | `layer{l}.attn.{q,k,v}_proj.matmul/com_X.bin ≡ N/yrescale/com_Xr.bin` | attn input = input_norm output, 3 consumers |
| A2q/k/v | `{q,k,v}_proj fc/com_Y.bin ≡ {q,k,v}_proj rescaling/com_X.bin` | pre-rescale projection product |
| A3q/k | `SM/rope.{q,k}/com_T.bin ≡ {q,k}_proj rescaling/com_Xr.bin` | RoPE input chained to the projection |
| A3v | `SM/slice/com_V.bin ≡ v_proj rescaling/com_Xr.bin` | V (unroped) into the slicer |
| A4q/k | `SM/rope.{q,k}/com_Y64.bin ≡ SM/rope.{q,k}.rescale/com_X.bin` | exact RoPE product → its rounding proof |
| A5q/k | `SM/slice/com_{Q,K}.bin ≡ SM/rope.{q,k}.rescale/com_Xr.bin` | roped+rescaled Q/K into the slicer |
| A6.hh | `SM/slice/com_Qh{hh}.bin ≡ SM/fc.h{hh}/com_X.bin` | per-head Q operand |
| A7.hh | `SM/slice/com_KhT{hh}.bin` IS the `<com_W.bin>` argv of `SM/fc.h{hh}` verify | per-head K^T operand (path binding; the verifier absorbs the file) |
| A8.hh | `SM/fc.h{hh}/com_Y.bin ≡ SX/rescale13.h{hh}/com_X.bin` | scores @2^32 |
| A9.hh | `SX/rescale13.h{hh}/com_Xr.bin ≡ SX/rescale10.h{hh}/com_X.bin` | (SOFTMAX_DESIGN §4.7) |
| A10.hh | `SX/rescale10.h{hh}/com_Xr.bin ≡ SX/softmax.h{hh}/com_z.bin` | scores @2^9 into softmax |
| A11.hh | `SX/softmax.h{hh}/com_P.bin ≡ VM/fc.h{hh}/com_X.bin` | probabilities into the values matmul |
| A12.hh | `SM/slice/com_Vh{hh}.bin` IS the `<com_W.bin>` argv of `VM/fc.h{hh}` verify | per-head V operand (path binding) |
| A13.hh | `VM/fc.h{hh}/com_Y.bin ≡ VM/rescale.h{hh}/com_X.bin` | head output @2^32 |
| A14.hh | `VM/rescale.h{hh}/com_Xr.bin ≡ VM/merge/com_O{hh}.bin` | rescaled head output into the merge |
| A15 | `VM/merge/com_O2.bin ≡ attn_skip/com_attn_out.bin` | **closes the S1 OPEN BOUNDARY** — the residual stream is now fully chained input → output |

Per layer: 13 layer-level byte edges + 84 per-head byte edges + 24 path bindings.
Weight bindings for q/k/v_proj need no edges (registered-com argv absorbs, as for
the MLP). The widening shim (A8→A9 path) is data-file-only, not trust-carrying
(ORCHESTRATOR_DESIGN §3, already wired and unit-checked).

Registration additions (ORCHESTRATOR_DESIGN §2): `ppgen 64 → gen64.bin`;
`rope-cos-table.bin` + `rope-sin-table.bin` sha256-registered; (q/k/v weights are
already registered "for forward-compat with the attention stage").

---

## 8. Selftest plan

Structure copied from zkob_rmsnorm/zkob_softmax: `selftest_small()` (3 shape cases
per driver) + `selftest_real()`; semantic evil modes checked against the *expected
reject string*; byte tampers over every verifier-read file with restore-and-reverify
at the end; evil==0 convention sanity checks (every verifier fold terminal ==
multi_dim_me of the corresponding tensor at toy scale).

### 8.1 Small honest cases (toy tables: random int32 in [−2^7, 2^7] — the drivers
are table-agnostic like glu; the relations are exact whatever the table)

| driver | case a | case b | case c | shapes exercised |
|---|---|---|---|---|
| zkob_rope | B=8, C=6, HD=2 | B=4, C=8, HD=4 | B=16, C=12, HD=4 | a: padding cols + flip bit 0; b: C == C_pad (no padding); c: padded 16×16 grid |
| zkob_headslice | B=8, C=6, HD=2 (NH=3) | B=4, C=8, HD=4 (NH=2) | B=16, C=12, HD=4 (NH=3) | head bits 1–2 wide; padded heads present (a,c) and absent (b) |
| zkob_headmerge | B=8, C=6, HD=2 | B=4, C=4, HD=2 | B=16, C=12, HD=4 | b: B == C (π = plain transpose — degenerate case must still pass); a,c: generic π |

All witnesses computed host-side per the §2.2 / §4.3 / §4.6 specs, exact.

### 8.2 Semantic evil modes (one per check; why nothing else catches it)

The evil prover runs the honest *procedure* on an inconsistent witness
(strict=false on the targeted recursion only, rmsnorm-style); each must be rejected
by exactly the named check.

zkob_rope:
- **evil=1: Y64[idx] += 1; ev computed (honestly) from the evil Y64; hadamards
  honest from T.** → **"c1 + c2 != ev"** rejects (S-Z at u). The Y64 IPA passes
  (consistent with com_Y64), both hadamards pass (consistent with com_T) — the sum
  identity is the only line of defense for the output tensor.
- **evil=2: hadamard 2 run with the UNPERMUTED T (Tx := T); Y64 computed
  consistently with that wrong relation (so evil=1's check passes by
  construction).** → **"IPA opening of h2 terminal vs com_T at the flipped point"**
  rejects: S_f2 = T̃(pt2) but the verifier opens at pt2′, expecting T̃(pt2′) ≠ S_f2
  whp. This is the evil that certifies the rotate_half binding is load-bearing —
  with it passing, a prover could ship un-rotated RoPE.
- **evil=3: prover uses a bumped cos table (W1[idx] += 1) consistently throughout
  (Y64, weight, hadamard).** → **"h1 weight terminal"** (cur == W_f·S_f2·U_f2 with
  the verifier's W_f rebuilt from the REGISTERED table) rejects. Certifies the
  public-table fold recompute.
- **evil=4: h1 run with a corrupted ones-buffer U[idx] += 1 (everything else
  honest).** → **"h1 U_f2 != 1"** rejects. (Same load-bearing role as softmax
  evil=5's broadcast check.)

zkob_headslice (the evil prover computes the claimed eval from its evil slice, so
the slice-side IPA passes and the full-tensor side must catch it):
- **evil=1: Qh{1} filled from head 2's columns.** → **"IPA opening of eQ01 vs com_Q
  (head-selector point)"** rejects.
- **evil=2: Vh{0} sliced with a one-column offset (cols 1..HD+1) — the classic
  off-by-one.** → **"IPA opening of eV00 vs com_V"** rejects.
- **evil=3: KhT{0} filled by reinterpreting K_h's row-major buffer as HD×B without
  transposing — the classic transpose bug.** → **"IPA opening of eK00 vs com_K"**
  rejects. (Honest-case pass + this evil jointly pin the (v_t ‖ v_d) coordinate
  swap: if the VERIFIER had the swap wrong, the honest case would fail.)
- (Tampering the full tensors Q/K/V themselves is deliberately NOT an evil mode:
  com_Q/K/V are self-committed and internally consistent; the defense is the A3/A5
  byte-equality, tested at chain level — the softmax com_z precedent.)

zkob_headmerge:
- **evil=1: O2 assembled WITHOUT the line-157 permutation (identity layout, i.e.
  O2 := M) — the natural implementation bug; ev honest from the evil O2; hadamards
  honest.** → **"sum of head claims != ev"** rejects. THE certifying evil for π.
- **evil=2: O2 honest on real entries but with junk in a padding column; ev honest
  from it.** → **"sum of head claims != ev"** rejects (the weights' real-index
  support forces padding = 0; same named check, different certified property).
- **evil=3: prover gathers Wm_3 with h off by one (weight-construction bug),
  runs hadamard 3 honestly on it, c03 absorbed from that run.** → **"merge hadamard
  03 terminal"** rejects (verifier-rebuilt Wm_3 fold ≠ prover's).
- **evil=4: U-buffer bump in hadamard 7.** → **"merge hadamard 07 U_f2 != 1"**.
- (Swapping two heads' out_h files wholesale is internally consistent here by
  construction — the defense is edge A14's byte-equality per head; chain-level test.)

### 8.3 Byte tampers

For every file in §7.2, tamper one byte, verify must REJECT, restore; final full
verify must ACCEPT. Pinned offsets: com_* at 24 (first point x-limbs); hp*.bin at 36
(claim_H is bytes 0–31, count header 32–35 ⟹ 36 hits the round-0 p(0)); ipa_* at −32
(a_final); ev.bin/evals.bin at 4 (no count header — raw Fr sequence); dims.bin at 0.
113 files for headslice makes this the longest tamper loop — keep it (every file the
verifier reads must be load-bearing).

### 8.4 Real-scale cases (and predicted costs, to be replaced by measurement)

Reference points: rescale D=2^20/N=2^16: 2.6 s / 3.3 s; softmax (2^20 grid + 2^20
table, 16 IPAs): 10.18 / 11.60 s; glu (D=2^22): 11.4 / 13.8 s; rmsnorm (17 IPAs +
2^16-class lookup): 9.1 / 7.0 s. Volume reasoning: a 2^20-row commit MSM ≈ 0.3–0.5 s;
a 20-round 2^20 hadamard ≈ 0.5–1 s (softmax ran three inside 10.2 s alongside two
2^20-table lookups); a gen-1024 FS-IPA ≈ 0.1–0.3 s (softmax's 16 fit in budget), a
gen-64 IPA ≈ trivial.

- **zkob_rope** (B=1024, C=768, HD=64, real tables; T ~ round(N(0, 2^18)) — the
  realistic |q| envelope): 2 grid commits + 2 × 2^20 hadamards + 3 IPAs ⟹
  **predict ≈ 3 s prove / ≈ 4 s verify** (verifier adds two 2^20 weight
  builds+folds, ~ms each). Byte tamper at scale (hp1.bin @ 36) must reject.
- **zkob_headslice** (real dims): 3 × 2^20-row commits + 36 small commits + 48
  gen-1024 IPAs + 24 gen-64 IPAs ⟹ **predict ≈ 8–15 s prove / ≈ 8–15 s verify** —
  the widest error bars in this design (IPA-dominated, a regime no measured driver
  isolates; §9.1 measurement gate). Byte tamper (ipa_Qf05.bin @ −32) must reject.
- **zkob_headmerge** (real dims): 1 + 12 commits + 12 × 2^16 hadamards + 13 IPAs +
  verifier's 12 gathers of one 2^20 eq build ⟹ **predict ≈ 2.5 s prove / ≈ 3.5 s
  verify**. Byte tamper (hp07.bin @ 36) must reject.

### 8.5 Whole-segment cost prediction (per forward pass = 2 layers)

| instance type | runs | prove s (each) | verify s (each) | prove total | verify total |
|---|--:|--:|--:|--:|--:|
| q/k/v_proj zkob_fc (768×768) | 6 | ~1.5 | ~1.5 | 9 | 9 |
| q/k/v_proj rescale (D=2^20) | 6 | 2.6 | 3.3 | 15.6 | 19.8 |
| zkob_rope | 4 | ~3 | ~4 | 12 | 16 |
| rope rescale (D=2^20) | 4 | 2.6 | 3.3 | 10.4 | 13.2 |
| zkob_headslice | 2 | ~12 (8–15) | ~12 | 24 | 24 |
| scores zkob_fc (64-wide) | 24 | ~1.0 | ~1.0 | 24 | 24 |
| scores rescale 13+10 | 48 | ~2.5 | ~3.0 | 120 | 144 |
| zkob_softmax (measured) | 24 | 10.2 | 11.6 | 244.8 | 278.4 |
| values zkob_fc | 24 | ~0.8 | ~0.8 | 19.2 | 19.2 |
| values rescale (D=2^16) | 24 | ~0.5 | ~0.6 | 12 | 14.4 |
| zkob_headmerge | 2 | ~2.5 | ~3.5 | 5 | 7 |
| **attention segment total** | **168** | | | **≈ 496 s ≈ 8.3 min** | **≈ 569 s ≈ 9.5 min** |

(±30% honest error bars on the non-measured rows; softmax + rescale rows are
measured.) Proof+commitment bytes ≈ 60 MB/layer ≈ **120 MB/forward** (softmax 24.4
MB/layer + values/scores fc obdirs ~7 MB + score rescales ~14 MB + slice 4 MB +
merge 1.9 MB + rope 1.2 MB + proj ~5 MB, before any com dedup — com_X/com_P/com_Y
duplicates across adjacent obdirs are the obvious later dedup, as noted for softmax).

Manifest ids newly covered: 24 non-waived ids (q/k/v_proj ×3 obligations ×2 layers
= 18, scores_matmul ×2, softmax ×2, values_matmul ×2 — softmax already implemented)
+ the S1 open-boundary closure. After this stage the only remaining non-waived gaps
are embedding/lm_head/logit-binding (out of scope here).

### 8.6 After validation

Persist drivers to zkllm-src/, document as the next §§ in PHASE0_NOTES.md, re-run
ALL drivers' selftests (headers untouched ⟹ rebuilds unnecessary, but the rule
stands), then wire the orchestrator stage-2 manifest (attention ids move from
`skipped` to `checked`; check_transcript re-run against both manifests per
ORCHESTRATOR_DESIGN §6).

---

## 9. Open questions / risks (explicit; none load-bearing for starting implementation)

1. **zkob_headslice cost is the one real unknown.** 72 FS-IPAs per instance is a
   regime no measured driver isolates (softmax's 16 IPAs hide inside lookup-dominated
   10 s). The 8–15 s estimate could be off either way; `me_weights` is a host loop
   with one device round-trip per element (10·1024 h_scalar calls per gen-1024 IPA)
   and is the plausible hot spot. **Gate:** measure case-real early in
   implementation; if a single instance exceeds ~30 s, the sanctioned fallbacks are
   (i) accept it (it is 2 instances/forward, <10% of segment cost), or (ii) memoize
   me_weights host-side per point (pure host code, no kernel change). Do NOT invent
   an opening-batching protocol — that is new cryptography, not new engineering.
2. **Float32-vs-float64 table fidelity** (§2.1): the registered tables are float64-
   generated; the pipeline's cos/sin are float32. A handful of ±1-ulp@2^16 table
   entries may differ from a float32-faithful regeneration. Soundness is unaffected
   (sha256 registration is the source of truth, both sides load the same bytes);
   the only impact is approximation drift vs the float pipeline, which the
   witness-authority rule already accepts (the integer spec *defines* the proven
   function). If exact float32-fidelity is ever demanded, regenerate with float32
   angle math and re-register — a table swap, not a design change.
3. **|Y64| ≥ 2^47 honest throw** (§6): unconditionally int64-safe, but the int32
   chain-file format after the rope rescale assumes |Tr| < 2^31. Honest margin is
   ~2^9 (measured-scale q/k magnitudes); the throw converts a violated assumption
   into a completeness failure, never an unsound accept. Re-check only if the
   pipeline's activation scales ever change.
4. **The line-157 permutation is frozen, including its probable accidental origin.**
   If anyone "fixes" m68-pipeline.py's double transpose (making attn_out the plain
   head-concat M), zkob_headmerge's π⁻¹ gather must become the identity — a
   one-line pinned-formula change plus re-registration of the public statement.
   Same class as SOFTMAX_DESIGN §9.7's temperature warning: the pipeline is frozen
   for this task; bind what it computes.
5. **Composition under scores_matmul/values_matmul** (§4.0): rope/slice/merge have
   no manifest ids of their own; they ride the matmul ids the way swiglu's rescale
   rides mlp.swiglu. check_transcript only counts manifest ids, so this is
   accounting-clean, but the transcript.json `details` strings must list every
   sub-run so a missing rope run cannot hide behind a passing fc — the orchestrator
   marks the manifest id ok only if ALL composed sub-runs and ALL §7.4 edges pass.
6. **com_attn_out provenance switch.** Stage-1 commits attn_out from the
   python-replicated float attention; once this design lands, attn_out comes from
   the integer chain (≤1-ulp-per-entry drift through rope and the values rescale,
   §1.3/§2.2). The skip add and everything downstream re-runs on the new integers —
   a witness change, with difr impact expected ≪ the rmsnorm/softmax replacements
   already absorbed. Measure difr once end-to-end before declaring the stage done.
7. **gen64 is new.** `ppgen 64`, register, hash-pin. The IPA requires pow2 gen
   sizes (PHASE0 §10) — 64 ✓. The 6-round gen-64 IPA shape is smaller than any
   measured case (toy cases ran GEN=8, so the shape itself is validated).
8. **Slice/merge instance failure isolation is per layer, not per head** (§4.0
   trade-off). A single corrupt head fails the whole layer's slice obligation. The
   per-head alternative (12× re-commit of com_Q/K/V) costs ~×4 the slice work for
   isolation we don't need — the selftests and chain edges localize failures well
   enough. Revisit only if debugging at scale proves painful.
9. **No range proof on Q/K/V or out_h entries is added** — none is needed: every
   bound in §6 is either unconditional (int64 safety from the int32 file format) or
   measured-with-margin and guarded by honest throws (completeness, not soundness).
   The exp-table domain remains the only semantic range gate in the chain
   (SOFTMAX_DESIGN §3.1), and SCORES_RANGE.md's 3.7× margin stands with integer
   RoPE (§2.2).
