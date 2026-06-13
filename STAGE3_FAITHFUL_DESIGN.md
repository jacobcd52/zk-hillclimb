# STAGE3_FAITHFUL_DESIGN.md — exact row-max primitive, lm_head/logit binding (stage 3), and the faithful-architecture revision (submission: faithful-arch-v1)

Status: DESIGN FINAL (2026-06-11). Companions: SOFTMAX_DESIGN.md, ROPE_ATTENTION_DESIGN.md
(read both as registers — this document follows their conventions exactly), PHASE0_NOTES.md
(all pinned machinery conventions: FS transcript, FS-IPA, logUp, Montgomery/integer-view
rules, -dlto gotchas), ORCHESTRATOR_DESIGN.md + ORCHESTRATOR_REPORT.md stage-2 section,
DIFR_BASELINE_NATIVE.md (the measurement this design answers), THREAT_MODEL_NOTES.md
(greedy-argmax binding pinned). Written so the implementer makes **no design decisions**:
every relation, scale, bound, absorb, file, CLI argument and selftest case is pinned here.
Design only; no code was written; `m68-pipeline.py`, `harness/`, `zkob_lookup.cuh`,
`vrf_common.cuh` and all nine validated drivers are consumed AS-IS (one pinned exception:
a flag added to `zkob_headmerge`, §4.2, with full revalidation).

## 0. Executive summary

DIFR_BASELINE_NATIVE.md measured the stage-2 chain at **DiFR 8.988 vs the FP8 teacher**,
and decomposed it: **2.4e-6 nats is integerization; the remaining ~9 nats is entirely three
frozen-pipeline quirks** (no o_proj, the line-157 head scramble, softmax temperature 128
instead of 8). Its §7 caveat 6 is the strategy this document executes: the Pareto work that
buys DiFR is architectural, submitted as an explicit pipeline-authority revision with the
baseline as the before/after yardstick. Three deliverables, one new primitive:

**Part A — `zkob_rowmax`** (§2): a new obligation driver proving, with zero advice freedom
on the value, that a committed per-row scalar `mx[i]` equals `max_{j ∈ allowed(i)} z[i,j]`
over a committed grid. Mechanism: a committed one-hot selector S — binarity by an
eq-weighted sumcheck of S⊙(S−𝟙) = 0, per-row one-hot by two broadcast-eq sumchecks with
*constant* claims (Σ_allowed S = 1, Σ_masked S = 0), attainment by a three-factor sumcheck
⟨S,z⟩-row = mx, and dominance by a limb-range-checked residual grid AL⊙(mx_bcast − z) ≥ 0
bound to (z, mx) by an MLE identity. Two masking regimes: `causal` (allowed = j ≤ i; the MK
machinery) and `vpad` (vocab 32000 inside 32768; allowed = j < V, pads excluded by the mask
weights and forced out of S — pinned in §2.2). **Zero new G1 kernels; one new Fr-only
kernel** (`k_pp_expand`, driver-local, §2.8 — the glu precedent for Fr kernels) so the
gen-32768 IPAs do not inherit the `me_weights` host-loop hot spot. mx carries **0 covert
bits** (the value is forced); the selector carries log2(#argmax ties) bits/row in proof
bytes only — quantified, gated, and consistent with TOKEN_CAPACITY.md's tie accounting
(§2.4).

**Part B — stage 3, close the manifest** (§3): `final_norm.rmsnorm` = the existing
zkob_rmsnorm trio at the final-norm site (advice R switches to the integer-exact bracket;
C_eps = 3298535 — same eps/C as the per-layer sites, verified §3.1); `lm_head` = one
registered 768×32000 weight (NEW `gen32768` via ppgen; single-instance zkob_fc + rescale —
feasibility arithmetic in §3.2, the 4×8192 split REJECTED), discharging
`lm_head.commitment_opening` by the standard IPA(W)-vs-registered-com pattern;
`statement.logit_binding` = one Part-A `vpad` instance on the 1024×32768 logits grid
binding `t*[i] = argmax_v logits[i,v]` at **all 1024 positions** (score.py scores every
position), with t* sha256-pinned in public.json before sealing (ordering pinned §3.4).
Manifest lands at **56/56 non-waived checked, 0 stage-skipped**, plus three covered-waived
ids (final_norm.rmsnorm, lm_head.matmul, lm_head.rescaling — check_transcript's
covered-waived NOTE path, verified §3.5). The only remaining waived-and-uncovered id is
`embedding.lookup`.

**Part C — submission `faithful-arch-v1`** (§4): one coherent re-registration that makes
the proven function a faithful llama-68m: (1) o_proj added per layer (registered weight —
the pipeline already dumps it, so the byte provenance guard applies; fc + rescale slotted
between headmerge and attn_skip); (2) line-157 fixed: zkob_headmerge gains a pinned
`<perm>` CLI flag (`pi157`/`concat`, absorbed into the transcript) — a validated-driver
edit, full revalidation pinned, NO new driver; (3) temperature 8: a NEW driver
`zkob_softmax8` (structural diff justified §4.3) consuming Part A's causal mx — exponents
land in (−∞, 0], table E(v) = rint(2^16·e^{v/4096}) over v ∈ [−2^20+2, +1] with a pinned
sentinel row for masked positions; the full SOFTMAX_DESIGN §6 arithmetic is redone (S drops
to ≤ 2^26, limbs shrink to 2×14-bit). Predicted end state: **DiFR ≈ 0.008** (the
linears-only floor — argued §5.2), prove ≈ +60–90 % over the 743 s baseline, the
before/after Pareto table in §5.3.

Covert capacity of everything added: 0 bits everywhere except the rowmax tie channel
(§2.4, measured gate, expected ≈ 0) — the system keeps its zero-advice property.

---

## 1. Pipeline semantics (quoted, with authority noted)

### 1.1 What m68-pipeline.py actually contains — and does not

The frozen pipeline (read-only, `zkllm-src/m68-pipeline.py`) **ends at the per-layer
loop**. There is no final norm, no lm_head, and no logits anywhere in it: after the last
`skip-connection` it writes `m68_layer_output.bin` and the summary JSON. The three
manifest ids `final_norm.rmsnorm`, `lm_head.matmul`, `lm_head.rescaling` are waived
exactly because of this ("not in current m68 pipeline scope" / "zkLLM pipeline proves
through final hidden states"). The semantics Part B binds are therefore the
**pipeline-authority extensions already pinned and measured** in
DIFR_BASELINE_NATIVE.md §3 and `measure/int_chain.py::logits()` (validated byte-exact
against the stage-2 witness chain):

```python
# int_chain.py logits() — the pinned pipeline-authority completion
Xr = X.astype(np.float64) / SF                                  # X = final residual, int32@2^16
rms_inv = 1.0 / np.sqrt((Xr * Xr).mean(axis=1) + self.final_eps)
R = np.rint(rms_inv * SF).astype(np.int64)        # the float advice (replaced in §3.1)
normed = self._rmsnorm_body(X, R, gain=self.g_final)            # W=R×g, rescale, ⊙X, rescale
logits_i = rescale(imatmul(normed, self.w_lm), LOG_SF)          # int @2^16
```

with `g_final = round(model.model.norm.weight.float()·2^16)` and
`w_lm = round(model.lm_head.weight.float().T·2^16)` (int32, 768×32000) — the same
`round(w.float().T·2^16)` convention as every registered weight (m68-pipeline.py lines
93–94). The float-advice line is the analog of the per-layer advice convention the
pipeline DOES contain (lines 124–127):

```python
X = torch.tensor(np.fromfile(cur_input, ...), dtype=torch.float64) / (1 << 16)   # line 124-125
eps = layer.input_layernorm.variance_epsilon                                      # line 126
save_int(1 / torch.sqrt(torch.mean(X ** 2, dim=1) + eps), 1 << 16, "rms_inv_temp.bin")  # 127
```

Per the pinned witness-authority rule (ORCHESTRATOR_DESIGN §3, applied at all four
per-layer norm sites), the proof binds the **integer-exact bracket** R instead of the
float advice — §3.1 applies the same switch at the final-norm site. llama-68m's
`rms_norm_eps = 1e-6` (PHASE0 §14; `model.norm.variance_epsilon` is the same constant —
registration asserts it, §3.1) gives C_eps = round(1e-6·768·2^32) = **3298535**, identical
to the per-layer sites.

### 1.2 The three quirks Part C revises (quoted)

1. **No o_proj.** Lines 137–159 go straight from the values matmul to
   `save_int(attn, 1<<16, attn_out)`; line 205 records
   `"note_o_proj": "zkLLM's released per-layer pipeline does not prove o_proj"`; the
   frozen manifest waives all six `layer{l}.attn.o_proj.*` ids ("zkLLM upstream omits
   o_proj"). **But the commit loop (lines 91–101) iterates ALL
   `layer.named_parameters()`** — `self_attn.o_proj.weight` IS exported and committed by
   the pipeline (`layer-{i}-self_attn.o_proj.weight-int.bin`), it is just never used. So
   the registration provenance guard (byte-compare vs the pipeline's own dump) applies to
   o_proj exactly as to q/k/v (§4.1).

2. **Line 156–157** (the head scramble — ROPE_ATTENTION_DESIGN §1.3 derivation):

   ```python
   attn = attn.transpose(0, 1).contiguous().view(seq, embed)    # line 156: head-concat M
   attn = attn.transpose(0, 1).reshape(seq, embed)              # line 157: permutation π
   ```

   π is a genuine entry permutation of the (1024, 768) grid, "almost certainly an
   upstream accident", bound faithfully by zkob_headmerge's π⁻¹ gather. The fix note
   (ROPE design §9.4) pre-pinned the change shape: "zkob_headmerge's π⁻¹ gather must
   become the identity — a one-line pinned-formula change plus re-registration of the
   public statement." §4.2 executes exactly that, as a flag.

3. **Temperature 128** (SOFTMAX_DESIGN §1.4): lines 150–154 read the scale-2^16 scores
   `A` through `to_float(A, ACCU_LOGSF=20)`, so the computed exponent is
   `QK^T_real/(16·8) = score_real/128` instead of the faithful `score_real/8`.
   SOFTMAX_DESIGN §9.7 pre-pinned the consequence of fixing it: "the honest exponent
   range grows 16× and this design's single-table premise breaks". §4.3 is the redesign
   that warning demanded: a max-shift with the shift bound **exactly** by Part A (the §3.1
   elimination argument was "an exact row-max proof needs one-hot/argmax machinery we do
   not have" — Part A is that machinery, so the max-shift returns with 0 covert bits).

### 1.3 Authority statement for this document

- Part B binds the existing function (m68 quirks intact) — it changes coverage, not
  semantics, except the final-norm advice switch (float → exact bracket, same move as
  stage 2; its drift is inside the measured 2.4e-6 integerization floor).
- Part C **redefines the registered statement** (new public.json, new run, new
  registration — a submission per HARNESS.md, which explicitly allows "zkLLM fork /
  integerization changes / pipeline script" changes). m68-pipeline.py itself is NOT
  edited; the proven function is defined by the integer specs in this document, exactly
  as stage 2's function is defined by SOFTMAX_DESIGN §2 / ROPE_ATTENTION_DESIGN §2.2.
  The pipeline-authority decision THREAT_MODEL/DIFR caveat 6 called for is made here, in
  writing: **faithful-arch-v1 proves the faithful llama-68m architecture** (o_proj
  applied to the plain head-concat, softmax temperature 8), integerized with the same
  conventions as stage 2.
- The scored-positions protocol (for Part B's logit binding): `harness/score.py` computes
  `post_gumbel_margin` over the full `forced_logits` tensor — **every one of the 1024
  positions** is scored. The binding therefore covers all 1024 rows (no subset).

---

## 2. Part A — `zkob_rowmax`: the exact row-max obligation driver

### 2.1 Statement to prove

Public constants (absorbed; recorded in dims.bin): B, NCOL (both powers of two), MODE ∈
{causal = 0, vpad = 1}, V (real column count, vpad only; 0 in causal), LEN_R (limb table
length, power of two), NPL ∈ {1, 2} (limb planes). The allowed-set mask:

```
causal:  AL[i,j] = 1 if j ≤ i else 0          (requires B == NCOL; V = 0; the MK mask)
vpad:    AL[i,j] = 1 if j < V else 0          (requires 0 < V ≤ NCOL; column padding)
```

AL is public, never committed (derived from B/NCOL/MODE/V — exactly the softmax MK
convention). Every row has a nonempty allowed set (causal: the diagonal; vpad: V ≥ 1).

Committed tensors (PLAIN form; grid tensors committed row-wise with `gen_grid` of size
NCOL; mx committed as ONE row of B values with `gen_mx` of size B):

- `z`  — the input grid (chained; re-committed from the chain file, byte-identical with
  the upstream commitment — §2.7 edges). Input file holds the UNPADDED B×V (vpad) or
  B×NCOL (causal) int32 values; the driver zero-pads columns V→NCOL (vpad).
- `S`  — the selector grid: S[i,j] = 1 at the selected argmax of row i, else 0
  (including masked and pad positions). **Pinned canonical honest witness: the LOWEST
  allowed j attaining the max** (np.argmax convention; see §2.4 on ties).
- `mx` — B values, mx[i] = max_{j: AL[i,j]=1} z[i,j].
- `L`  — the dominance-residual limb tensor, NPL planes of B·NCOL each
  (flat = plane·D + i·NCOL + j, D = B·NCOL):
  the residual is `Df[i,j] = AL[i,j]·(mx[i] − z[i,j])`;
  NPL = 1 (causal): plane 0 = Df itself (single limb, Df ∈ [0, LEN_R));
  NPL = 2 (vpad):  plane 0 = Df mod 2^20, plane 1 = floor(Df / 2^20), LEN_R = 2^20.
  Df is never committed standalone; its MLE is reconstructed from plane openings.
- `m_L` — limb-lookup multiplicities (LEN_R values; LEN_R/NCOL commitment rows);
  `A_L` — logUp inverse grid (NPL·D values; NPL·D/NCOL rows), committed after β_L.

The relation proven (∀ real i, j):

```
(X1)  S[i,j] ∈ {0,1} on the whole padded grid
(X2)  Σ_j AL[i,j]·S[i,j] = 1   and   Σ_j (1−AL[i,j])·S[i,j] = 0     (per row)
      ⟹ with X1: S is one-hot per row, with its 1 at an ALLOWED position
(X3)  Σ_j S[i,j]·z[i,j] = mx[i]                                      (attainment)
(X4)  Df[i,j] = AL[i,j]·(mx[i] − z[i,j]),  Df[i,j] ∈ [0, LEN_R^NPL) componentwise
      ⟹ mx[i] ≥ z[i,j] for every allowed j                           (dominance)
(X5)  [vpad with t* supplied]  S[i, t*[i]] = 1                       (argmax = t*)
```

X1–X4 together: mx[i] = z[i, j*] for an allowed j* (X1+X2+X3) and mx[i] ≥ all allowed
z[i,j] (X4), hence mx[i] is exactly the allowed row max. **Tolerances: none; mx is the
unique integer satisfying the relation.** X5 (public t*, absorbed) binds the served token.

Field-wrap soundness note (the chain-composition argument, rmsnorm-M / rescale
precedent): X4's "≥" reads Df's field values as integers in [0, LEN_R^NPL). A wrap would
need committed |z| or |mx| values within ~LEN_R^NPL of the field modulus; com_z is
byte-chained to the upstream driver outputs, which are exact deterministic functions of
the registered input/weights (the whole chain has zero witness freedom), and mx = some
z value by X1–X3. So wrap is impossible for any z the chain can accept. Pinned as a
documented reliance on the §2.7 edges (exactly softmax's com_z MINOR-5 posture).

Honest-prover throws (completeness guards): B, NCOL, LEN_R not powers of two; causal
with B ≠ NCOL or V ≠ 0; vpad with V ∉ (0, NCOL]; gen sizes ≠ (NCOL, B); any allowed
Df ≥ LEN_R^NPL (causal: spread ≥ 2^20 — cannot happen for in-domain scores; vpad guard
|z| ≥ 2^25, see §2.5); input file short-read; t* supplied in causal mode; any
t*[i] ∉ [0, V).

### 2.2 Why the vpad regime is pinned this way (the padding decision)

The task's two candidate treatments for pad columns were (a) zero selector weight + range
proof only over real columns, or (b) a −∞ sentinel committed as a public constant.
**Pinned: (a).** Concretely: pads are excluded by the PUBLIC mask AL in every weight
(X2's sums, X4's residual — Df is identically 0 at pads, in range trivially), and S is
forced to 0 at pads by X2's second sum + X1. A pad column therefore cannot "win the max"
because it can neither be selected (S = 0 there) nor constrain dominance (AL = 0 there).
(b) is rejected: a sentinel requires committing or deriving a modified grid z′ with
−∞-entries, i.e. an extra committed tensor plus a binding identity, for a property the
public mask already gives for free; and "−∞" as a large-negative field constant
reintroduces exactly the wrap-around analysis (a) avoids. The same argument excludes pad
positions in the causal regime's padded grid — except the causal real-scale shape
(1024×1024) has no padding at all.

### 2.3 Proof obligations (one transcript)

All sumchecks are `fs_hadamard` instances (k_hp3_step: three multilinear factors, 4 evals
per round). "Rebuild+fold" = the verifier constructs the public weight tensor itself and
folds it with the round challenges via k_fr_fold (the softmax §4.3/§4.4 validated
pattern); "eq_acc" = the rmsnorm SS accumulator (eq factor only for the row-bit rounds —
valid exactly when the weight is a pure column-broadcast eq tensor, rmsnorm-validated).
Constant claims (X1's 0, X2's 1 and 0, X5's 1) are **protocol constants: not absorbed,
not read from disk** — the verifier imposes the constant at round 0 and additionally
requires the serialized claim_H field to equal it (reject otherwise). Data-dependent
claims (ev_mx, c1, c2) are absorbed, softmax-style.

1. **LIMB — limb range lookup** (binds X4's range): logUp on L vs tLookupRange(0, LEN_R).
   D_L = NPL·D, N = LEN_R; layouts (§2.5) give n1 = 0 (causal) / 6 (vpad). Terminals
   A_f vs com_A_L, S_f vs com_L, m_f vs com_m_L (3 IPAs; opening points = reversed round
   challenges, m at the phase-2 suffix). Verifier recomputes B_f/T_f from the public
   range table.

2. **BIN — binarity sumcheck** (X1): claim 0 = Σ_b eq(u_bin,b)·S(b)·(S(b)−1) over logD
   vars. E = build_eq_tensor(u_bin) (pure eq → verifier uses the my_eq accumulator over
   all rounds, glu-hadamard style), S = S, U = S − 𝟙 (prover materializes the int buffer
   S−1 — values −1/0, FrTensor int ctor handles the mod-p — never committed). Verifier
   REQUIRES **U_f2 == S_f2 − 1** (the MLE of S − 𝟙 at any point is S̃ − 1; load-bearing —
   this is what makes U bound to com_S), checks the terminal, opens S at
   pt_bin = reverse(ws_bin) vs com_S. 1 IPA. Without BIN, a fractional selector with
   row-sum 1 makes ⟨S,z⟩ an arbitrary field value — X3 would bind nothing (selftest
   evil=3 certifies this).

3. **SUM — one-hot-over-allowed** (X2 first sum): claim **1** = Σ_b W_sum(b)·S(b)·𝟙 over
   logD vars, W_sum = k_bcast_rows(build_eq_tensor(u_s)) ⊙ AL. The row-sum vector
   ⟨AL-row, S-row⟩ has MLE ≡ 1 iff every row sums to 1; its value at random u_s is the
   claim. Rebuild+fold; U_f2 == 1 required; opens S at pt_s. 1 IPA.

4. **MASK — nothing-selected-outside** (X2 second sum): claim **0** =
   Σ_b W_mask(b)·S(b)·𝟙, W_mask = k_bcast_rows(build_eq_tensor(u_m)) ⊙ (𝟙 − AL).
   Rebuild+fold; U_f2 == 1; opens S at pt_m. 1 IPA. (In vpad, W_mask's support includes
   the pad columns — that is what forces S's pads to zero.)

5. **ATT — attainment** (X3): absorb ev_mx = m̃x(u_a), u_a = logB vars; claim ev_mx =
   Σ_b W_a(b)·S(b)·z(b), W_a = k_bcast_rows(build_eq_tensor(u_a)) (pure broadcast eq →
   verifier uses the rmsnorm eq_acc shortcut: eq factor for the first logB rounds only).
   Terminals: S_f2 opens S at pt_a vs com_S; U_f2 opens z at pt_a vs com_z; ev_mx opens
   vs com_mx at u_a (single-row opening, u_row empty — the softmax com_S pattern, with
   gen_mx). 3 IPAs. Equality of Σ W_a·S·z with m̃x(u_a) at random u_a forces the row-dot
   vector ⟨S-row, z-row⟩ to equal mx as tensors whp.

6. **DOM — dominance binding** (X4's identity): challenge u_r (logD vars), then:
   - absorb c1; sumcheck c1 = Σ_b [eq(u_r)⊙AL](b)·z(b)·𝟙 (rebuild+fold; U_f2 == 1);
     opens z at pt_c1 vs com_z. 1 IPA.
   - absorb c2; sumcheck c2 = Σ_b [eq(u_r)⊙AL](b)·mx_bcast(b)·𝟙 (rebuild+fold;
     U_f2 == 1). mx_bcast is materialized by the prover (k_bcast_rows) and **never
     committed**: its terminal S_f2 opens against com_mx at the row-bit suffix
     pt_c2[logNCOL..logD) — the softmax V2 broadcast-MLE binding. 1 IPA.
   - absorb v0 (and v1 if NPL = 2); open com_L at (u_r ‖ plane-bits): NPL = 1: one
     opening at u_r; NPL = 2: at (u_r ‖ 0) and (u_r ‖ 1) (Boolean plane bit = flat-index
     bit logD, the softmax §4.6 L-plane convention). NPL IPAs.
   - Verifier identity (plain field, h_scalar): **c2 − c1 == v0 [+ 2^20·v1]**.
     LHS is the MLE at u_r of AL⊙(mx_bcast − z) (linearity of the two eq⊙AL-weighted
     sums); RHS is the MLE of the committed limb reconstruction. Equality at random u_r
     ⟹ Df = AL⊙(mx_bcast − z) as tensors whp; LIMB then gives componentwise
     Df ∈ [0, LEN_R^NPL), i.e. X4.

7. **T-BIND — served-token binding** (X5; vpad only, when t* is supplied): t* (B int32
   values, public) is absorbed in the preamble. Claim **1** = Σ_b W_t(b)·S(b)·𝟙 over
   logD vars, W_t[i·NCOL+j] = eq(u_t)[i]·[j == t*[i]] (u_t = logB vars; verifier gathers
   W_t from its own t* copy — one 2^logD device upload, rebuild+fold). U_f2 == 1; opens
   S at pt_t. 1 IPA. Forces S[i, t*[i]] = 1 ∀i whp; with X1–X4, t*[i] is a maximizer of
   row i.

Soundness summary — which check catches what: LIMB pins the residual limbs to [0, LEN_R);
BIN pins S to {0,1} (kills fractional-selector forgeries — unbounded mx otherwise); SUM +
MASK pin one-hot-on-allowed; ATT pins mx to an attained z value through com_S/com_z/com_mx
openings; DOM's identity + LIMB pin mx ≥ z on allowed; T-BIND pins the public t* into S;
the IPA openings pin every terminal to commitments; com_z/com_mx pin the instance into the
chain (§2.7); the single FS transcript (every commitment absorbed before any challenge)
prevents splice/replay.

IPA count: causal 12 (3 lookup + 1 BIN + 1 SUM + 1 MASK + 3 ATT + 3 DOM);
vpad 13 (+1 L plane) or 14 with T-BIND.

### 2.4 Advice-binding analysis (covert capacity)

- **mx: 0 bits.** The value is the unique allowed row max — X1–X4 leave no choice.
  Everything downstream of mx (softmax8's E/S/P in Part C; the t* statement in Part B)
  therefore inherits 0 bits from this obligation.
- **The selector tie channel (the one honest caveat).** When row i's allowed max is
  attained at k_i > 1 positions (exact integer ties), any of the k_i one-hot selectors
  passes X1–X5* (*if t* is tied, any tied t* passes too). Capacity: Σ_i log2(k_i) bits
  per instance, **observable only in proof bytes** (com_S and its openings) — never in
  any served tensor, never in mx. This is the same tie-break artifact class
  TOKEN_CAPACITY.md already accounts for (its measured exact ties: 3/3072 positions, all
  BOS artifacts of tiled prompts; "a tie-break, not a channel bit"). Pinned handling:
  (i) the honest prover always emits the lowest-index argmax (canonical witness);
  (ii) the orchestrator measures and reports Σ log2(k_i) over all instances of a run in
  prove_manifest.json (cheap: it has the witness); (iii) the capacity-budget table gets a
  "rowmax selector ties" row with that measured number. Expected magnitude: ≈ 0–10 bits
  per forward pass total (ties require exact int equality of row maxima; the vpad
  instance's analog is exactly TOKEN_CAPACITY's 3-BOS-ties class). A strict
  first-argmax enforcement was designed and **rejected**: it needs a committed prefix
  tensor T = S·U_tri with U_tri the 32768² public triangular matrix — the prover-side
  sumcheck would have to materialize 2^30 field elements (32 GiB). Documented as the
  upgrade path if the measurement ever shows material capacity (open question §6.1).
- Everything else (limbs, residuals, multiplicities, A tensors) is a deterministic
  function of (z, mx, S, AL): **0 bits**. The only nondeterminism is completeness
  failure, never choice — except the tie row above.

### 2.5 Numeric bounds (with the arithmetic)

Causal shape (B = NCOL = 1024, z = attention scores at scale 2^9, LEN_R = 2^20, NPL = 1):
- z ∈ [−2^19, 2^19) honest (witness envelope; measured |z| ≤ ~1.42e5 = 277·2^9,
  SCORES_RANGE.md headline, 3.7× margin). In the faithful-arch chain this bound is a
  completeness envelope guarded by the upstream rescale chain + the driver throw, not a
  proven range (§4.3 note); in any chain that also runs baseline zkob_softmax on the same
  com_z, its R1 lookup proves it outright.
- mx ∈ [−2^19, 2^19) (attained value). Df = mx − z over allowed ∈ [0, 2^20 − 1] —
  exactly one 20-bit limb, never overflows for in-domain z. LEN_R = 2^20, NPL = 1,
  D_L = D = 2^20, N = 2^20: C_pad = 2^10 ≤ N ≤ D_L, N | D_L, n1 = 0 (pure phase2) ✓.
- Commitment rows: com_z/S/L/A_L = 1024; com_m_L = 2^20/2^10 = 1024; com_mx = 1.

Vpad shape (B = 1024, NCOL = 32768, V = 32000, z = logits at scale 2^16, LEN_R = 2^20,
NPL = 2):
- z honest: |logit_real| ≤ 36.2 measured (DIFR_BASELINE §1) ⟹ |z| ≤ 36.2·2^16 < 2^22.
  Driver completeness guard: throw if |z| ≥ 2^25 (margin 8×; converts any future
  envelope violation into a loud failure, rmsnorm-M style).
- Df ∈ [0, 2·2^25) ⊂ [0, 2^26) honest; forced ∈ [0, 2^40) by two 20-bit limbs
  (lo + 2^20·hi). Headroom is harmless — the load-bearing bound is Df ≥ 0 (the softmax §6
  argument verbatim). D = 2^25, D_L = 2^26, N = 2^20: C_pad = 2^15 ≤ 2^20 ≤ 2^26,
  N | D_L, n1 = 6 ✓.
- Commitment rows: com_z/S = 2^25/2^15 = 1024; com_L/A_L = 2^26/2^15 = 2048;
  com_m_L = 2^20/2^15 = 32; com_mx = 1 (gen_mx = 1024-wide — gen1024, both modes).
- All committed values < 2^40 ≪ p; every host intermediate < 2^45 — int64-safe, no
  __int128.
- GPU memory at vpad scale (Fr_t = 32 B; 2^25 Fr = 1 GiB): peak during the limb lookup ≈
  L (2 GiB) + A_L (2 GiB) + fold buffers (≤ 2 GiB) + z, S resident (2 GiB) ≈ 8 GiB;
  during the grid hadamards ≈ z + S + eq (1 GiB) + weight (1 GiB) + folds ≈ 6 GiB.
  Fits the 24 GiB RTX 4090 with ≥ 2× headroom **provided the prover frees each
  obligation's tensors before the next block** (pinned implementation requirement;
  measured gate + the blocked fallback in §6.3).

### 2.6 FS schedule (one transcript; seed = "<run_seed>:<obligation_id>")

Absorb-by-absorb; labels exact; → x = derive challenge; every IPA absorbs its "L","R"
internally (gen_grid: 10 rounds causal / 15 vpad; gen_mx: 10 rounds).

```
absorb_u32  "B" B, "NCOL" NCOL, "MODE" mode, "V" V, "LEN_R" LEN_R, "NPL" NPL
[vpad+t*]   absorb "TSTAR" (the B int32 values, raw bytes)
absorb_g1_tensor "com_z" com_z      "com_S" com_S      "com_mx" com_mx
absorb_g1_tensor "com_L" com_L      "com_m_L" com_m_L
→ beta_L     [prover: A_L = 1/(L + beta_L)]
absorb_g1_tensor "com_A_L" com_A_L
→ alpha_L  → u_L = fs_challenge_vec(log2(NPL·D))
limb lookup rounds ("p0".."p3" → w each); absorb_fr "A_f","S_f","m_f"
IPA(A_L) at reverse(ws_L); IPA(L) at reverse(ws_L); IPA(m_L) at reverse(ws_L[n1..])
→ u_bin = fs_challenge_vec(logD)
BIN rounds ("hp0".."hp3" → w); absorb_fr "S_f2","U_f2"     [claim = 0, NOT absorbed]
IPA(S) at pt_bin = reverse(ws_bin)
→ u_s = fs_challenge_vec(logB)
SUM rounds; absorb_fr "S_f2","U_f2"                         [claim = 1]
IPA(S) at pt_s
→ u_m = fs_challenge_vec(logB)
MASK rounds; absorb_fr "S_f2","U_f2"                        [claim = 0]
IPA(S) at pt_m
→ u_a = fs_challenge_vec(logB)
absorb_fr "ev_mx"
ATT rounds; absorb_fr "S_f2","U_f2"
IPA(S) at pt_a; IPA(z) at pt_a; IPA(mx) at u_a              [vs com_mx, gen_mx]
→ u_r = fs_challenge_vec(logD)
absorb_fr "c1";  c1 rounds; absorb_fr "S_f2","U_f2";  IPA(z) at pt_c1
absorb_fr "c2";  c2 rounds; absorb_fr "S_f2","U_f2";  IPA(mx) at pt_c2[logNCOL..logD)
absorb_fr "v0" [, "v1"]
IPA(L) at (u_r ‖ 0) [; IPA(L) at (u_r ‖ 1)]                 [NPL=1: at u_r, no plane bit]
[vpad+t*] → u_t = fs_challenge_vec(logB)
           T-BIND rounds; absorb_fr "S_f2","U_f2"; IPA(S) at pt_t   [claim = 1]
[verifier-only, no absorbs: U_f2 checks per §2.3; c2 − c1 == v0 [+ 2^20·v1]]
```

Verifier round-count checks: lookup ev = 4·log2(NPL·D); BIN/c1/c2/T-BIND-class
hadamards ev = 4·logD each... (SUM/MASK/ATT/T-BIND are also logD-round instances — their
u challenges have logB vars but the sumcheck runs over the full grid, broadcast-weighted,
exactly softmax's row-sum: 4·logD evals each). Commitment row counts per §2.5; dims.bin =
{u32 B, u32 NCOL, u32 MODE, u32 V, u32 LEN_R, u32 NPL} cross-checked against argv.

### 2.7 CLI, files, chain interface

```
zkob_rowmax prove  <obdir> <seed> <z-int32.bin> <B> <NCOL> <MODE> <V> <LEN_R> <NPL>
                   <gen_grid.bin> <gen_mx.bin> <q.bin> [mx-int32-out.bin] [tstar-int32.bin]
zkob_rowmax verify <obdir> <seed> <B> <NCOL> <MODE> <V> <LEN_R> <NPL>
                   <gen_grid.bin> <gen_mx.bin> <q.bin> [tstar-int32.bin]
zkob_rowmax selftest
```

- MODE: literal strings `causal` / `vpad` (parsed to 0/1 for dims/absorbs).
- `[mx-int32-out.bin]`: unpadded B int32 chain file (causal consumer: zkob_softmax8's
  mx input; same scale as z). vpad runs may omit it.
- `[tstar-int32.bin]`: B int32 public token ids; vpad only; both prove AND verify take it
  (the verifier loads its own registered copy — hash-pinned via public.json, §3.4).
- The driver does NOT mkdir the obdir; gens: gen_grid size NCOL, gen_mx size B.

Files in <obdir> (26 causal / 29 vpad+t*): dims.bin; com_z, com_S, com_mx, com_L,
com_m_L, com_A_L (.bin); lookup_L.bin; hp_bin.bin, hp_sum.bin, hp_mask.bin, hp_att.bin
(claim_H = ev_mx), hp_c1.bin, hp_c2.bin [, hp_tbind.bin]; lvals.bin (NPL Fr_t: v0 [, v1]);
ipa_A_L, ipa_L_lk, ipa_m_L, ipa_S_bin, ipa_S_sum, ipa_S_mask, ipa_S_att, ipa_z_att,
ipa_mx_att, ipa_z_c1, ipa_mx_c2, ipa_L_p0 [, ipa_L_p1] [, ipa_S_tbind] (.bin).
Causal proof+commitment bytes ≈ 1.1 MB/instance (four 1024-row coms at 144 KB dominate);
vpad ≈ 1.6 MB (two 2048-row coms at 288 KB).

Chain edges (orchestrator byte-equalities, ≡):
- causal (per layer l, head hh — Part C wiring §4.3):
  `RM1.hh: SX/rescale10.h{hh}/com_Xr.bin ≡ SX/rowmax.h{hh}/com_z.bin`
  `RM2.hh: SX/rowmax.h{hh}/com_mx.bin ≡ SX/softmax8.h{hh}/com_mx.bin`
- vpad (Part B wiring §3.4):
  `L1: lm_head.rescaling obdir com_Xr.bin ≡ statement.logit_binding/rowmax/com_z.bin`
  (the lm_head rescale commits the zero-padded 1024×32768 grid with gen32768; the rowmax
  prover pads identically — byte-identity holds for honest runs, PHASE0 §12 mechanism).

### 2.8 The one new kernel (and why it is allowed)

`me_weights` and `ipa_verify`'s s-vector are host loops with one 1-thread kernel
round-trip per element (ROPE §9.1's measured hot spot: ≈ 60 µs/element — headslice's
29.6 s is exactly this). At gen-32768 that is ≈ 30 s per IPA prove-side and ≈ 65 s
verify-side — unacceptable for rowmax-vpad's 13–14 IPAs. Pinned fix, **driver-local to
zkob_rowmax.cu** (the shared headers are NOT edited): one new Fr-only kernel

```
KERNEL void k_pp_expand(GLOBAL Fr_t* in, Fr_t a, Fr_t b, GLOBAL Fr_t* out, uint n)
// out[gid] = a·in[gid]; out[gid+n] = b·in[gid]   (generalizes k_eq_expand's (1−c, c))
```

plus two host helpers built on it: `fast_me_weights(u)` (doubling with (1−u_i, u_i) —
build_eq_tensor's recurrence, one device→host copy at the end) and
`fast_s_vector(xs, xis)` (doubling with (xi_r, x_r), MSB-first order matching
ipa_verify's pinned s_i product). Both are cross-checked element-exact against the slow
header versions in the selftest (evil==0 convention checks, at toy AND gen-1024 scale).
Precedent: glu added k_eq_expand/k_hp3_step when needed; "Fr kernels are not in the
-dlto miscompile family" (zkob_lookup.cuh header note). The IPA protocol itself is
untouched — only the construction of two public vectors moves to the device. The
existing drivers (zkob_fc, zkob_rescale) are NOT modified; their gen-32768 cost is
quantified honestly in §5.1 and the header-lift is a gated follow-up (§6.2).

### 2.9 Selftest plan

Structure copied from zkob_softmax: selftest_small (4 shape cases) + selftest_real
(both real shapes); semantic evil modes against expected reject strings (strict=false on
the targeted recursion only); byte tampers over every verifier-read file with
restore-and-reverify; evil==0 convention checks (every verifier fold terminal ==
multi_dim_me of the corresponding tensor; fast_me_weights/fast_s_vector == slow header
versions).

Small cases (toy z random int32 in [−2^6, 2^6] so Df fits the toy LEN_R):

| case | mode | B | NCOL | V | LEN_R | NPL | exercises |
|---|---|--:|--:|--:|--:|--:|---|
| a | causal | 8 | 8 | 0 | 32 | 1 | n1=0 toy; diagonal-only row 0 |
| b | vpad | 8 | 16 | 11 | 32 | 2 | pad cols + t* + plane bit; n1=1 |
| c | causal | 16 | 16 | 0 | 64 | 1 | bigger grid |
| d | vpad | 4 | 8 | 8 | 32 | 2 | V == NCOL (no pads; MASK weight ≡ 0 must still pass) |

Semantic evil modes (one per check; the witness-engineering that isolates each is part
of the pinned setup):

- **evil=1: mx[row] += 1; S honest at the true argmax; Df/limbs recomputed from the evil
  mx.** → **"ATT round 0"** rejects (absorbed ev_mx = evil m̃x(u_a); the sumcheck runs on
  honest (S, z) → round-0 claim mismatch). LIMB/DOM pass (all residuals shifted +1, still
  ≥ 0 and identity-consistent), BIN/SUM/MASK pass, mx IPA passes (consistent with
  com_mx). ATT is the sole catcher — certifies attainment is load-bearing against
  too-high mx.
- **evil=2: mx[row] −= 1, at a selftest-constructed row whose second-distinct value is
  max − 1; S moved to that j (attainment consistent); the true argmax position's residual
  is −1 — limbs stored as the low 20·NPL bits of its field representative.** →
  **"DOM bracket identity"** rejects (c2 − c1 = MLE of the true residual tensor incl. the
  field rep of −1; the limb reconstruction differs as a tensor). LIMB passes (limbs in
  range), ATT/BIN/SUM/MASK pass. Certifies dominance is load-bearing against too-low mx.
- **evil=3: fractional selector.** Row with allowed j1 ≠ j2, z1 ≠ z2: S[j1] = c,
  S[j2] = 1 − c with c = (mx − z2)·inv(z1 − z2) (field), mx honest, everything else
  recomputed consistently. SUM passes (sums to 1), MASK passes, ATT passes (by
  construction), DOM passes (mx honest). → **"BIN round 0"** sole catcher. This is THE
  certifying evil: without BIN, this construction proves an arbitrary mx ≥ max.
- **evil=4: two-hot selector.** S[j2] += 1 at an allowed j2 with z[j2] = 0
  (selftest constructs a zero entry), mx/Df honest. BIN passes (binary), ATT passes
  (adds 0·1 to the dot), MASK/DOM/LIMB pass. → **"SUM round 0"** sole catcher.
- **evil=5: selector on a pad column** (vpad cases only): S[i, V] += 1 (pad z = 0 ⟹ ATT
  unchanged; AL = 0 there ⟹ SUM and Df unchanged; binary ⟹ BIN passes). →
  **"MASK round 0"** sole catcher — certifies pad/masked exclusion. (The causal-mode
  variant entangles ATT — masked z ≠ 0 — and is deliberately not a selftest case;
  MASK's coverage there follows from the same check.)
- **evil=6: out-of-range limb with compensating carry** (vpad): lo += 2^20, hi −= 1
  (reconstruction unchanged ⟹ DOM identity passes; all commitments consistent). →
  **"limb lookup round 0"** sole catcher — proves the range lookup is load-bearing for
  Df ≥ 0.
- **evil=7: corrupted broadcast buffer in c2** (mx_bcast[idx] += 1; hadamard run honestly
  on it; c2 absorbed from that run; everything else honest). → **"IPA opening of c2
  terminal vs com_mx"** rejects — certifies the never-committed broadcast is pinned to
  com_mx (softmax evil=5 analog).
- **evil=8: wrong served token** (vpad+t*): t*[row] set to a non-argmax allowed token;
  S honest. → **"T-BIND round 0"** sole catcher.

Byte tampers: every file in §2.7 (offsets per the softmax table: hp_* at 36, lookup at
4+32, ipa_* at −32, com_* at 24, lvals at 4, dims at 0); restore; final full verify
ACCEPT.

Real-scale cases: (i) causal B = NCOL = 1024, LEN_R = 2^20, NPL = 1, z ~ the softmax
selftest's score distribution (round(N(0, 2^13)) clipped to ±(2^19−1)); (ii) vpad
B = 1024, NCOL = 32768, V = 32000, LEN_R = 2^20, NPL = 2, z ~ round(N(0, 2^21)) clipped
to ±(2^25−1), t* = the true argmax. Measure prove/verify wall and proof bytes; one byte
tamper each must reject. **Predicted (to be replaced by measurement):** causal ≈ 8–10 s
prove / 9–12 s verify per instance (volume ≈ softmax minus one 2^22 lookup plus three
extra 2^20 hadamards); vpad ≈ 140–220 s prove / 160–260 s verify (≈ 190 grid-commit
2^20-units at ~0.4 s/unit + a 2^26 lookup + six 2^25 hadamards + 13–14 fast IPAs;
±40 % bars — the largest single instance in the system).

### 2.10 Cost at both consumer shapes (summary table)

| shape | instances | commits (2^20-units) | recursions | IPAs | predicted prove | predicted verify |
|---|--:|--:|---|--:|--:|--:|
| causal 1024×1024 (Part C) | 24 | ~5 | 1×2^20 lookup + 6×2^20 hadamard | 12 | ≈ 8–10 s each ⟹ **3.2–4.0 min total** | ≈ 9–12 s each ⟹ 3.6–4.8 min |
| vpad 1024×32768 (Part B) | 1 | ~192 | 1×2^26 lookup + 6×2^25 hadamard | 14 | **≈ 140–220 s** | ≈ 160–260 s |

---

## 3. Part B — stage 3: close the manifest (final norm, lm_head, the final statement)

Part B binds the SAME function the stage-2 chain already computes (DIFR_BASELINE §3's
table rows "final norm" and "lm_head", currently NOT proven), with one pinned authority
switch (final-norm R). It is orchestrator + registration + one Part-A instance work; the
only new driver it needs is zkob_rowmax.

### 3.1 final_norm.rmsnorm — the existing trio at the final-norm site

- **Instance:** zkob_rmsnorm + zkob_rescale(W.i64, sf 2^16) + zkob_rescale(Y.i64,
  sf 2^16) — bit-for-bit the validated per-layer shape (B = 1024, C = 768, gen_C =
  gen_B = gen1024). Composed under manifest id `final_norm.rmsnorm` (sub-seeds
  `final_norm.rmsnorm`, `…rmsnorm.wrescale`, `…rmsnorm.yrescale`).
- **X** = the terminal residual (layer1.mlp_skip output). Edge F0:
  `final_norm/rmsnorm/com_X.bin ≡ proofs/layer1.mlp_skip/com_Z.bin` (byte — both are
  fresh commitments of the same int32 tensor with gen1024; the skip's com_Z is already
  produced and point-checked by edge S2).
- **Advice R: integer-exact bracket** (`prove_walk.compute_R`, python big-int isqrt) —
  the witness-authority switch, replacing int_chain.py's float `np.rint(rms_inv·2^16)`
  advice. Same authority rule as the four per-layer sites; drift vs the float path is
  ±1 ulp on R in rare rows, inside the measured 2.4e-6 integerization floor (honest
  note: this changes the witness logits by ≤ the same class of drift the stage-2
  switches did; the §3.4 t* is computed from the SWITCHED chain, so statement and
  witness stay consistent by construction).
- **C_eps = 3298535** (eps = 1e-6, C = 768 — identical constant to the per-layer
  sites). PINNED registration assert: `model.model.norm.variance_epsilon == 1e-6`
  (export fails loudly otherwise; do NOT silently inherit the per-layer value).
- **New registered weight `final_norm.g`** = round(model.model.norm.weight.float()·2^16)
  int32 (768,), committed as (1, 768) with gen1024 (the rmsnorm-gain pattern,
  ORCHESTRATOR_DESIGN §2 step 3). Provenance: the pipeline never dumps model.norm
  (its commit loop only iterates `model.model.layers[*]`), so the guard is re-export
  comparison only — documented deviation from the byte-compare-vs-pipeline-dump rule.
- Trio-internal edges: W1/W2/Y1 pattern verbatim (com_Wr.bin naming — PHASE0 §14
  MINOR-7). Downstream edge F5: `final_norm/yrescale/com_Xr.bin ≡ lm_head fc com_X.bin`.
- Cost (measured shape): ≈ 14.3 s prove / ≈ 13.9 s verify (the input_norm trio's
  numbers).

### 3.2 lm_head — registered weight, one fc + one rescale (the gen32768 decision)

**Registration additions:**
- `ppgen 32768 → gen32768.bin` (4.7 MB; one-time, seconds) — hash-pinned in public.json
  next to gen1024/gen4096/gen64. The IPA requires pow2 gen sizes — 32768 = 2^15 ✓.
- Weight id `lm_head`: `round(model.lm_head.weight.float().T · 2^16)` int32, 768×32000
  (97.7 MB file). Note llama-68m may tie lm_head to the embedding
  (`config.tie_word_embeddings`); HF materializes `model.lm_head.weight` either way and
  the export reads it directly — semantics identical, noted in public.json
  (`"lm_head_tied": <bool>`). Provenance: the pipeline never dumps lm_head (same class
  as final_norm.g — re-export guard only).
- Commitment: `zkob_fc commit lm_head-int.bin 768 32000 gen32768 lm_head-com.bin` —
  rows = IN_pad = 1024 G1 points (144 KB; W is padded {1024, 32768} per the §8 glue,
  padding transparent). One-time MSM volume = 768 × 32768 ≈ 2.5e7 point-muls ≈ 10–16 s —
  registration-time, reported separately per HARNESS.md timing rules (genuinely
  input-independent).

**lm_head.matmul (+ commitment_opening):** one `zkob_fc prove` — B = 1024, IN = 768,
OUT = 32000 (OUT_pad = 32768 internal), X = final-norm yrescale output (int32 @2^16),
gen_in = gen1024, gen_out = gen32768. Emits `logits64.i64` (unpadded 1024×32000, scale
2^32, 256 MB) and com_Y (1024 rows, gen32768). The fc verify takes the REGISTERED
`lm_head-com.bin` as its com_W argument and absorbs it — **its IPA(W) vs the registered
commitment discharges `lm_head.commitment_opening`**, exactly the gate/up/down pattern
(ORCHESTRATOR_DESIGN §3 table). This closes one of the two remaining non-waived ids.

**lm_head.rescaling:** one `zkob_rescale prove`, sf = 2^16, B = 1024, C = 32000
(C_pad = 32768), gen32768. Lookup layout: D = 2^25, N = 2^16, C_pad = 2^15 ≤ N ≤ D,
N | D, n1 = 9 ✓. Input = logits64.i64 (int64, direct — no widening shim). Emits
`logits.i32` (1024×32000 @2^16, 128 MB) and com_Xr (1024 rows) — **the committed logits
grid**. Round-half-up rescale semantics = int_chain.py's `rescale(·, 16)` exactly
(byte-validated convention, probe_semantics check class).

**Single instance vs 4×8192 column blocks — DECIDED: single instance, gen32768.**
The arithmetic:
- *Memory:* fc peak ≈ W_pad (2^25 Fr = 1 GiB) + Y (1 GiB) + X (32 MB) + sumcheck halves
  (≤ 1 GiB) ≈ 3.3 GiB. rescale peak ≈ X + Xr + rem + A (4 × 1 GiB) + B/T/m (N = 2^16,
  ~6 MB) + fold halves ≈ 5.5 GiB. Both fit the 24 GiB card with ≥ 4× headroom.
- *MSM/commit volume:* com_Y + com_X/com_Xr/com_rem/com_A ≈ 5 × 2^25 ≈ 160 2^20-units ≈
  60–80 s total — large but linear; blocking does not reduce it.
- *Prove/verify time predictions:* fc ≈ 30–45 s prove core (Y commit 12–16 s + 25-round
  sumcheck over 2^25 ≈ 10–20 s) **+ ≈ 60 s of slow gen-32768 IPA weight-building**
  (me_weights host loop, §2.8 — zkob_fc is not modified) ⟹ ≈ 90–110 s prove,
  ≈ 100–150 s verify (two gen-32768 IPAs at ≈ 65 s verify-side each + folds). rescale ≈
  110–140 s prove core + ≈ 90 s slow IPAs (three) ⟹ ≈ 200–230 s prove, ≈ 230–290 s
  verify. Wide ±40 % bars; the slow-IPA share is itemized because the §6.2 header-lift
  removes it.
- *The rejected split:* 4 × (1024×768×8192) fc blocks via the headslice double-opening
  pattern would need 4 block commitments PLUS a new binding obligation pinning each
  block to a slice of the registered full-width commitment (the registered com_W is
  gen32768-row-structured; block gens would be gen8192 — the openings do not line up
  without a headslice-style paired-opening instance per block), 4× the IPA count, a new
  gen8192, and a combine step for the logits grid — strictly more machinery to avoid a
  memory ceiling we are nowhere near. REJECTED; revisit only via the §6.3 gate.

### 3.3 statement.logit_binding — the greedy pinned statement

> **Follow-up flagged (2026-06-13): decoding-regime mismatch — DOCUMENTED, not silently
> fixed.** This binding proves greedy argmax (`served = argmax(logits)`), but
> `THREAT_MODEL_NOTES.md §0/§1` was corrected: the protocol runs in the **verifiable
> sampled-decoding regime** (`served = argmax_v(logits + T·g_σ)`, committed seed σ), of
> which greedy is the T→0 case. Two honest readings: **(a)** for the treaty protocol the
> served token is *observed at the network tap* and checked against the proven logits
> *externally* (the DiFR margin check), so the in-proof greedy binding is **superfluous** to
> that external check; or **(b)** it should be recast as a **verifiable-sampled binding**
> `served = argmax_v(logits + T·g_σ)` — which the existing `zkob_rowmax` argmax-driver
> extends to almost directly: feed it `logits + T·g_σ` (the committed-seed Gumbel grid added
> to the proven logits) instead of `logits`, with **no new driver**. This is an identified
> design follow-up; the text below is left as-is and describes the current greedy binding.

THREAT_MODEL_NOTES §1 (pinned): "The final statement obligation must bind served-token
== argmax." The binding:

- **One zkob_rowmax vpad instance** on the committed logits grid: B = 1024,
  NCOL = 32768, V = 32000, LEN_R = 2^20, NPL = 2, gen_grid = gen32768, gen_mx =
  gen1024, with `tstar-int32.bin` supplied (T-BIND on). Obligation id
  `statement.logit_binding`, sub `rowmax`, seed `"<run_seed>:statement.logit_binding"`.
- **Positions: all 1024** (§1.3 — score.py's metric covers every position; no subset).
- **Edges:** L1 (com_z ≡ lm_head rescale com_Xr, §2.7); L2: registration hash check
  covers tstar.i32.bin (it is a registered artifact, §3.4) — the verifier's MINOR-5
  structural assertion sees it like any pinned path.
- **What is proven:** for every position i, t*[i] ∈ [0, 32000) and
  logits[i, t*[i]] = max_{v < 32000} logits[i, v] — argmax over the REAL vocabulary,
  pad columns excluded by construction (§2.2). Ties: t* is proven to be A maximizer;
  exact-tie rows admit any tied token (the TOKEN_CAPACITY tie-break class, counted in
  the capacity table per §2.4 — for the greedy threat model this is the same ambiguity
  the teacher-side check already has at margin 0).
- `statement.logit_binding` enters `checked` iff: rowmax verify ACCEPTs AND edge L1
  holds AND the registration hash check covered tstar.i32.bin.

**How t* enters the public statement (pinned ordering).** t* must be inside
public.json's hash (so run_seed = sha256(public.json) seeds every FS transcript with it
— the same mechanism that binds the input digest), but t* is only known after a forward
pass. Pinned registration order (register.py):

1. Phase A (unchanged): gens (now incl. gen32768), weight exports + commitments (now
   incl. final_norm.g, lm_head), input generation + commitment, tables.
2. Phase B (new): run the **deterministic integer-chain head pass** — the validated
   numpy forward (int_chain.py semantics, with the final-norm R switched to
   compute_R per §3.1) from the registered input through the registered weights to
   integer logits; t*[i] = np.argmax(logits_i[i, :32000]) (lowest-index tie-break —
   matches §2.1's canonical witness). Write `registration/tstar.i32.bin`.
3. Phase C: write public.json including `"served_tokens": {"file":
   "registration/tstar.i32.bin", "sha256": …}` and the new artifact hashes
   (gen32768, final_norm.g-com, lm_head-com, tstar). Seal; run_seed := sha256(bytes).

The later driver walk reproduces the same logits byte-exactly (validate_chain.py
precedent: the numpy chain reproduces driver-emitted tensors exactly); prove_walk
asserts `argmax(driver logits) == registered t*` and throws on mismatch (completeness
guard — a mismatch means the chain and the head pass diverged, which is a bug, never a
soundness event). Also recorded: `transcript.json.details["statement.logit_binding"]`
lists the rowmax sub-run + both edges; t* additionally echoed into transcript.json
(`"served_tokens_sha256"`) for the harness reader. For the current random-input runs t*
is "the served tokens of the registered random input" — the binding machinery is
identical when real prompts land (embedding stage); noted, not blocking.

### 3.4 Orchestrator wiring (walk order, edges, accounting)

Walk order appends after `layer1.mlp_skip.add`:

```
final_norm.rmsnorm   (trio: rmsnorm + wrescale + yrescale)        §3.1
lm_head.matmul       (zkob_fc; discharges lm_head.commitment_opening)
lm_head.rescaling    (zkob_rescale sf 2^16, gen32768)
statement.logit_binding (zkob_rowmax vpad + t*)                   §3.3
```

New edges (≡ byte; format of ORCHESTRATOR_DESIGN §4):

| # | edge | binds |
|---|---|---|
| F0 | `final_norm/rmsnorm/com_X.bin ≡ layer1.mlp_skip/com_Z.bin` | terminal residual → final norm |
| F1 | `final_norm/rmsnorm/com_g.bin ≡ reg/weights/final_norm.g-com.bin` | registered gain (defense in depth; the driver also absorbs the registered path) |
| F2/F3/F4 | trio-internal W1/W2/Y1 edges (com_Wr.bin naming) | as every norm site |
| F5 | `final_norm/yrescale/com_Xr.bin ≡ lm_head.matmul/com_X.bin` | normed hidden → lm_head |
| F6 | `lm_head.matmul/com_Y.bin ≡ lm_head.rescaling/com_X.bin` | pre-rescale logits |
| L1 | `lm_head.rescaling/com_Xr.bin ≡ statement.logit_binding/rowmax/com_z.bin` | committed logits → argmax binding |

Registration additions: gen32768.bin, final_norm.g (+com), lm_head (+com),
tstar.i32.bin — all sha256-pinned (public.json hash count 26 → 31). verify_walk's
registration phase re-hashes all of them fail-closed before any driver runs (the
stage-2 mechanism extends without code-shape changes — it iterates public.json's
pinned dicts).

### 3.5 Manifest accounting after Part B

`checked` = stage-2's 54 + `lm_head.commitment_opening` + `statement.logit_binding`
= **56/56 non-waived, `skipped` = {} (0 stage-waived)** — recounted from the manifest
by the selftest, stage-2 style. Additionally three WAIVED ids are now genuinely covered
and reported in `checked`: `final_norm.rmsnorm`, `lm_head.matmul`, `lm_head.rescaling`.
check_transcript.py handles this exactly (verified against its source): `known =
required ∪ waived`, unknown counts only ids outside the manifest, and covered waived ids
produce the "NOTE: … covered (candidates for waiver removal)" line with **exit 0**. So
vs the FROZEN manifest: `required: 56 checked: 59 missing: 0 unknown: 0` → PASS, with
the covered-waived NOTE. The only remaining waived-and-uncovered ids: `embedding.lookup`
(no integer path exists for the random-input pipeline; prompt binding remains the input
digest via run_seed — exactly the stage-2 posture) and, until Part C, the six
`o_proj.*` ids. Nothing else remains waived or skipped; no new manifest ids are invented
(rowmax sub-runs compose under `statement.logit_binding` / `layer{l}.attn.softmax`, the
swiglu precedent).

Selftest additions (selftest.sh): (d) tamper one byte of
`statement.logit_binding/rowmax/com_S.bin` → REJECT localized to
`statement.logit_binding` (transcript divergence — com_S absorbed before challenges);
restore. (e) flip one token in registration/tstar.i32.bin → REJECT at the registration
hash check (fail-closed stop, no drivers run); restore. (f) re-run (a)'s full honest
walk → ACCEPT with checked = 59, skipped = {}.

---

## 4. Part C — submission `faithful-arch-v1` (the first hill-climb submission)

One coherent re-registration. The three changes ship TOGETHER (they interact: o_proj
must consume the un-scrambled concat; the DiFR prediction is for the bundle), as
`submissions/faithful-arch-v1/` per HARNESS.md: claim.json, prove.sh (registration +
prove_walk on a fresh run id), verify.sh (verify_walk + check_transcript), student.py
(§4.5). The frozen harness manifest and m68-pipeline.py are untouched.

### 4.1 o_proj (per layer): registered weight, fc + rescale

- **Registration:** weight ids `layer{l}.attn.o_proj` =
  `round(layer.self_attn.o_proj.weight.float().T · 2^16)` int32 (768×768), committed
  with gen1024 (rows = IN_pad = 1024). **Provenance guard applies in full**: the
  pipeline's commit loop already dumps `layer-{l}-self_attn.o_proj.weight-int.bin`
  (§1.2.1) — byte-compare at export, fail on mismatch.
- **Chain slot (per layer):** between headmerge and attn_skip:

  ```
  zkob_headmerge (concat mode, §4.2)  → M.i32 (1024×768 @2^16), com_O2 (= com_M)
  zkob_fc  …o_proj.matmul   X = M.i32, B=1024 IN=768 OUT=768, com_W = REGISTERED,
                            gen1024/gen1024 → O64.i64 @2^32, com_Y
  zkob_rescale …o_proj.rescaling  O64.i64 sf 2^16 gen1024 → attn_out.i32 @2^16, com_Xr
  attn_skip.add consumes attn_out.i32; com_attn_out := o_proj rescale's com_Xr
  ```

  The fc + rescale are the validated 768×768 shapes (q/k/v-proj instances bit-for-bit;
  measured ≈ 2.2 s + 2.6 s prove each). The fc's IPA(W) vs the registered com discharges
  `layer{l}.attn.o_proj.commitment_opening`.
- **Edges (replacing A15's direct wiring):**
  `O1: VM/merge/com_O2.bin ≡ o_proj.matmul/com_X.bin`;
  `O2: o_proj.matmul/com_Y.bin ≡ o_proj.rescaling/com_X.bin`;
  `O3: o_proj.rescaling/com_Xr.bin ≡ attn_skip/com_attn_out.bin` (the S1-closure edge,
  now through o_proj).
- **Semantics:** out_real = M_real · W_o^T with one round-half-up rescale at 2^16 —
  the faithful HF o_proj, integerized with the standard convention. Witness authority:
  driver-emitted files, as everywhere.
- **Manifest:** the six `o_proj.*` ids are waived in the frozen manifest; the submission
  covers them → covered-waived NOTE, exit 0 (§3.5 mechanism). The submission's own
  scope manifest (§4.4) un-waives them.

### 4.2 line-157 fix: zkob_headmerge gains a pinned `<perm>` mode (no new driver)

- **CLI change (both prove and verify, positional, after `<HD>`):** `<perm>` ∈
  {`pi157`, `concat`}. dims.bin gains a u32 PERM field (0 = pi157, 1 = concat);
  transcript absorbs it: `absorb_u32 "PERM" perm` immediately after `"HD"` — REQUIRED,
  else a proof made in one mode could be replayed against a verifier in the other.
- **Formula change (the only algorithmic delta):** the public weight gather and the O2
  assembly use, in concat mode, the identity layout
  `Wm_h[t·HD + d] = E_u[t·C_pad + (HD·h + d)]` and `O2[t, HD·h + d] = out_h[t, d]`
  (i.e. π⁻¹ = identity on the real index set; padding columns still forced to 0 by the
  same Σ c_h == ev identity). pi157 mode keeps the §4.6 formula verbatim. Everything
  else (commitments, FS schedule shape, IPAs, file set) is unchanged.
- **Why a flag and not a constant or a new driver:** ROPE design §9.4 pre-pinned the
  change as "a one-line pinned-formula change plus re-registration"; a new driver would
  duplicate 95 % of a validated driver (worse review surface), and a hard constant would
  destroy reproducibility of the stage-2 baseline registration (the LEDGER's
  before-point must remain re-verifiable). The flag keeps one binary serving both
  registered statements, disambiguated in-transcript.
- **Revalidation (pinned, non-negotiable):** editing a validated driver ⟹ full
  zkob_headmerge selftest re-run, extended: all §8.1 small cases × both modes; the
  §8.2 evil set in both modes; NEW evil=5: prover assembles O2 with the WRONG mode's
  layout (concat-mode run with π157 gather — the exact regression this flag could
  introduce) → "sum of head claims != ev"; NEW cross-mode splice test: honest concat
  proof verified with `pi157` argv → reject (dims.bin cross-check + "PERM" absorb
  divergence). Plus a re-audit of the diff (ROPE_REVIEW addendum) before the
  orchestrator consumes it, and a re-run of ALL drivers' selftests (the standing rule —
  no header was touched, so this is cheap insurance, not rebuilds).
- public.json gains the constant `"headmerge_perm": "concat"`; verify_walk passes it as
  argv (a prover that ran pi157 diverges at the absorb).

### 4.3 temperature 8: `zkob_softmax8` (new driver) + the rowmax max-shift

**Real-valued target (the revised statement):**
`P_real[i,j] = exp(z_real[i,j]/8) / Σ_{j'≤i} exp(z_real[i,j']/8)` for j ≤ i, 0 for
j > i — the faithful √d = 8 temperature. With max-shift mx[i] = max_{j≤i} z_[i,j] bound
EXACTLY by Part A (covert capacity stays 0 — the §3.1 objection to max-shift is
dissolved), shift-invariance keeps the target unchanged while every exponent lands in
(−∞, 0].

**Why allowed-set max (causal), not global max:** the pipeline's own line-149 quirk
(`max(A·~mask)`) and a global-row max were both considered and REJECTED: with
temperature 8, a masked-region score can exceed the allowed max by hundreds of real
units (SCORES_RANGE: mask_max ≈ 43–113 vs all_max ≈ 235 on the same head), and shifting
by a masked max can drive EVERY allowed exponent below the E = 0 floor ⟹ S = 0 ⟹ the
row becomes unprovable. Shifting by the ALLOWED max guarantees the argmax entry has
diff = 0 ⟹ E = 2^16 ⟹ **S ≥ 2^16 always** — division never degenerate, by construction.

**Chain per (layer, head)** (replacing the §7.3 softmax step; scores path unchanged —
the temperature lives in the table, not the score scaling):

```
… zkob_rescale 2^13 then 2^10 → z_.i32 (scale 2^9)                 [unchanged]
zkob_rowmax prove <ob> <seed:…rowmax.h{hh}> z_.i32 1024 1024 causal 0 1048576 1
            gen1024 gen1024 q mx.i32
zkob_softmax8 prove <ob> <seed:…softmax8.h{hh}> z_.i32 mx.i32 1024 1024
            -1048574 1048576 softmax8-exp-table.bin 16384 gen1024 q P.i32
zkob_fc values … X = P.i32                                          [unchanged]
```

Edges: RM1.hh, RM2.hh (§2.7) + `SX8a.hh: rescale10/com_Xr ≡ softmax8/com_z` +
`SX8b.hh: softmax8/com_P ≡ VM/fc.h{hh}/com_X`. Manifest composition: id
`layer{l}.attn.softmax` = 12 × (rescale13 + rescale10 + **rowmax** + **softmax8**).

**Exp table (pinned, registered by sha256 — `gen_softmax8_exp_table.py`):**

```python
import numpy as np
LOW8, LEN8 = -(1 << 20) + 2, 1 << 20          # domain v ∈ [−1048574, +1]
v = np.arange(LOW8, LOW8 + LEN8, dtype=np.float64)
tab = np.rint(65536.0 * np.exp(v / 4096.0))    # exponent = v/2^12  (scale 2^9 · temp 8)
tab[v > 0] = 0.0                               # sentinel row(s): v = +1 maps to 0
np.rint(tab).astype(np.int32).tofile("softmax8-exp-table.bin")
```

Exponent bookkeeping: z_real = z_/2^9, exponent = z_real/8 = z_/2^12, so
E(v) = rint(2^16·e^{v/4096}) for v = z_ − mx ≤ 0. X[v=0] = 65536 = 2^16 (the shifted
argmax); X[v] = 0 for v ≤ −48,266 (rint(65536·e^{v/4096}) < 0.5 ⟺ v < −4096·ln(2^17));
the sentinel value X[+1] := 0 is the masked-position row.

**Committed tensors** (gen1024, B = NCOL = 1024, D = 2^20; PLAIN form):
- `z_` (chained), `mx` (chained from rowmax: 1024 values, 1 row, gen1024),
- `Dm` — the shifted-diff grid WITH sentinel:
  `Dm[i,j] = MK[i,j]·(z_[i,j] − mx[i]) + (1 − MK[i,j])·SENT`, SENT = +1.
- `E` — E[i,j] = X_E8[Dm[i,j] − LOW8]. By the sentinel, **E = 0 at masked positions by
  the table itself** — MK disappears from every downstream constraint.
- `S` — row sums S[i] = Σ_j E[i,j] (no mask weight needed), 1 row.
- `P` — P[i,j] = round_half_up(2^16·E[i,j]/S[i]); masked entries 0 automatically
  (E = 0 there).
- `L` — 4 planes of 14-bit limb pairs for r1 = 2^17·E + S_bcast − 2·P·S_bcast and
  r2 = 2S − 1 − r1 (plane layout = softmax §2's, with 2^14 limbs:
  plane 0 = r1 mod 2^14, 1 = r1 div 2^14, 2/3 same for r2).
- `m_E8`, `A_E8`, `m_L`, `A_L` — lookup auxiliaries.

**Obligations** (one transcript; deltas vs SOFTMAX_DESIGN §4 marked):

1. Exp mapping lookup: comb = Dm + r·E vs T_comb = table + r·mapped; D = N = 2^20,
   n1 = 0; com_comb formed homomorphically (com_Dm + r·com_E). *(Delta: indexes Dm, not
   z_ — the table no longer range-binds z_; see the soundness note below.)*
2. **Dm-binding block (NEW):** challenge u_d (20 vars); absorb cD1, cD2:
   cD1 = Σ_b [eq(u_d)⊙MK](b)·z_(b)·𝟙 (rebuild+fold; opens z_ at its pt);
   cD2 = Σ_b [eq(u_d)⊙MK](b)·mx_bcast(b)·𝟙 (broadcast never committed; terminal opens
   com_mx at the row-bit suffix — the §2.3-DOM/c2 mechanism). Verifier computes the
   public scalar k_MK = Σ_b eq(u_d,b)·MK(b) directly (build eq(u_d), k_fr_emul with MK,
   device sum — milliseconds), opens D̃m(u_d) vs com_Dm, and checks
   **D̃m(u_d) == cD1 − cD2 + SENT·(1 − k_MK)** (plain field). S-Z ⟹ Dm is exactly the
   masked-diff-with-sentinel tensor. 3 IPAs.
3. Limb range lookup: L vs tLookupRange(0, 2^14); D_L = 2^22, N = 2^14, n1 = 8. *(Delta:
   LEN_R8 = 2^14, was 2^20 — r1 < 2^27 needs only 14-bit pairs; m_L shrinks to 16 values
   → 1 commitment row.)*
4. Row-sum: ev_S = S̃(u_b) = Σ_b bcast(eq(u_b))(b)·E(b)·𝟙. *(Delta: NO MK factor — the
   weight is the pure broadcast eq, so the verifier uses the rmsnorm eq_acc shortcut
   instead of the rebuild+fold.)* Opens E at pt_rs, S at u_b.
5. Bracket V1: c1 = Σ_b eq(u_r,b)·E(b) (pure eq — my_eq accumulator). *(Delta: no MK.)*
   Opens E at pt1.
6. Bracket V2: c2 = Σ_b eq(u_r,b)·P(b)·S_bcast(b) — verbatim softmax §4.5 (S_bcast
   opened vs com_S at the row-bit suffix). Opens P, S.
7. Residual reconstruction + identities — softmax §4.6 verbatim with 2^14 limb weight:
   four plane openings + S_id; **I1: 2^17·c1 + S_id − 2·c2 == v00 + 2^14·v10;
   I2: r̃1 + r̃2 + 1 == 2·S_id.**

Composition soundness note (the MINOR-5 class, pinned loudly): within softmax8 alone,
nothing forbids Dm = SENT at an ALLOWED position (it would zero that probability) or
binds z_ − mx ≤ 0 — those are exactly what the chained rowmax instance proves (mx is
the allowed max ⟹ diffs ≤ 0 ⟹ allowed Dm ∈ [−spread, 0], never SENT) on the SAME
com_z/com_mx (edges RM1/RM2/SX8a). A standalone softmax8 ACCEPT binds (Dm, E, S, P)
internally; the rowmax edges are the defense — same posture as softmax's com_z.

**Numeric bounds (the §6 arithmetic redone for temp 8):**
- z_, mx ∈ [−2^19, 2^19) honest envelope (measured |z_| ≤ 1.42e5, 3.7× margin). NOT
  proof-bound anymore (delta from softmax R1); guarded by driver throws + chain
  determinism (§2.1's wrap note applies verbatim).
- Dm ∈ [LOW8, +1] = [−1048574, +1]; honest allowed diffs ≥ −283,000 (measured spread
  ≤ 553 real ⟹ ≤ 553·2^9), margin 3.7×. Corner: the absolute-worst in-envelope spread
  2^20 − 1 = 1,048,575 exceeds |LOW8| by 1 — unreachable honestly (needs |z_| at the
  full ±2^19 envelope simultaneously, 3.7× beyond real data); honest throw if
  diff < LOW8 (§6.8 open question records it).
- E ∈ [0, 2^16] (was [22, 2^28]); X_E8 fits int32 trivially.
- **S ∈ [2^16, 2^26]** (≥ 2^16: the allowed-argmax entry maps v = 0 → 2^16; ≤ 1024·2^16).
  The softmax S ≥ 1 invariant is structural here, not domain-dependent.
- P ∈ [0, 2^16] (E ≤ S still componentwise: E ≥ 0 and E is a summand of S).
- r1 ∈ [0, 2S) ⊂ [0, 2^27); r2 likewise; both < LEN_R8² = 2^28 ⟹ exactly two 14-bit
  limbs (hi < 2^13). Intermediates: 2^17·E ≤ 2^33; 2·P·S ≤ 2^43 — int64-safe.
- Lookup layouts: exp D = N = 2^20, n1 = 0 ✓; limb D_L = 2^22, N = 2^14,
  C_pad = 2^10 ≤ N ≤ D_L, N | D_L, n1 = 8 ✓.
- Commitment rows: com_z/Dm/E/P/A_E8 = 1024; com_S = com_mx = 1; com_L/A_L = 4096;
  com_m_E8 = 1024; com_m_L = 1 (16 values, one padded row).

**FS schedule:** softmax §5 verbatim with these deltas, in order: preamble absorbs
"B","NCOL","LOW8","LEN8","LEN_R8","LOG_OUT",“SENT”; base commitments now
com_z, com_mx, com_Dm, com_E, com_P, com_S, com_L, com_m_E8, com_m_L; → r, β_E →
com_A_E8 → α_E, u_E → exp lookup (pure phase2) + 3 IPAs (A_E8, comb vs com_Dm + r·com_E,
m_E8); → u_d → absorb "cD1", cD1 rounds + terminals + IPA(z_); absorb "cD2", cD2 rounds
+ terminals + IPA(mx at row-bits); absorb "vDm" + IPA(Dm at u_d); → β_L → com_A_L → α_L,
u_L → limb lookup + 3 IPAs; → u_b → "ev_S" → row-sum + IPA(E), IPA(S); → u_r → "c1" →
V1 + IPA(E); "c2" → V2 + IPA(P), IPA(S); "v00".."v11" + 4 plane IPAs; "S_id" + IPA(S);
verifier-only I1/I2 + the Dm identity. **19 IPA openings** (was 16; +3 for the Dm
block).

**CLI:**

```
zkob_softmax8 prove  <obdir> <seed> <z-int32.bin> <mx-int32.bin> <B> <NCOL>
                     <LOW8> <LEN8> <expmap8-int32.bin> <LEN_R8> <gen.bin> <q.bin>
                     [P-int32-out.bin]
zkob_softmax8 verify <obdir> <seed> <B> <NCOL> <LOW8> <LEN8>
                     <expmap8-int32.bin> <LEN_R8> <gen.bin> <q.bin>
zkob_softmax8 selftest
```

B == NCOL required (pow2); SENT = LOW8 + LEN8 − 1 derived (require the table's last
entry == 0 at load — the sentinel check, replacing glu's mapped(0) check). mx re-committed
from its chain file (byte-equal to rowmax's com_mx — edge RM2). P chain file int32 @2^16
(unchanged convention).

**Why a NEW driver, not a zkob_softmax revision (the task's flag, decided):** the diff
is structural, not parametric — one new committed tensor (Dm) + a three-IPA binding
block + a second chained input (mx) + different weights in three sumchecks (MK removed)
+ sentinel table semantics + new bounds. A revision would edit ~40 % of a validated,
audited driver and still require full revalidation, while DESTROYING the baseline's
bit-reproducibility (the stage-2 registration must stay verifiable as the LEDGER
before-point). zkob_softmax remains frozen; zkob_softmax8 is reviewed independently
(SOFTMAX8_REVIEW, same bar as SOFTMAX_REVIEW).

**Selftest deltas (vs softmax §8, pinned):** small cases reuse softmax's a/b/c shapes
with toy tables that include a sentinel last row = 0 and toy mx from a toy rowmax-style
host max; evil modes:
- evil=1 (E[idx] += 1, all downstream recomputed) → exp lookup round 0 [unchanged];
- **evil=2 (NEW, the Dm certifier): Dm at an allowed idx set to SENT (E = 0 there,
  S/P/limbs recomputed consistently — the "silently dropped probability" forgery)** →
  "Dm identity" rejects (cD1/cD2 are MLEs of the true masked diffs; D̃m differs).
  Nothing else fires: the lookup sees a valid (SENT, 0) table row, row-sum/brackets are
  consistent with the evil E.
- evil=3 (P += 1 at a masked idx, limbs truncated) → I1 [softmax evil=2 analog];
- evil=4 (P −= 1 at an unmasked idx, r1' = r1 + 2S) → I2 [softmax evil=3];
- evil=5 (S[row] += 1, all consistent) → row-sum round 0 [softmax evil=4];
- evil=6 (V2 broadcast buffer bump) → IPA of V2 U_f2 vs com_S [softmax evil=5];
- **evil=7 (NEW): cD2's mx_bcast buffer bump** → IPA of cD2 terminal vs com_mx;
- evil=8 (out-of-range limb with compensating carry, 14-bit) → limb lookup round 0
  [softmax hardening evil=6].
Byte tampers over all files (now incl. com_Dm, com_mx, the cD1/cD2 hp files, ipa_Dm);
real-scale case with the real table + mx from a real rowmax run on the same z_;
mx-tampering inside the obligation is NOT an evil mode (com_mx is chained; edge RM2 is
the defense — the softmax com_z precedent).

### 4.4 Submission / re-registration mechanics (HARNESS.md compliance, pinned)

- **What the submission IS:** a new registration + run under a new run id
  (`faithful1`), with: new public.json constants (`"headmerge_perm": "concat"`,
  `"softmax_temperature": 8`, `"o_proj": "applied"`), new registered artifacts
  (o_proj weights ×2 + coms, softmax8-exp-table.bin, plus Part B's artifacts — the
  submission INCLUDES stage 3), and the revised walk. The baseline registration and
  binaries remain intact and re-verifiable (driver coexistence: zkob_softmax /
  zkob_softmax8; headmerge's mode flag).
- **Manifest accounting:** the frozen `harness/manifest_llama68m.json` is NEVER edited.
  check_transcript vs the FROZEN manifest: required 56, checked 65 (56 non-waived + 6
  o_proj + final_norm + 2 lm_head), missing 0, unknown 0 → **PASS with the
  covered-waived NOTE** (§3.5 mechanism — this is exactly what that NOTE path is for:
  "candidates for waiver removal"). The submission also ships
  `manifest_faithful_scope.json` (a generated copy with the 9 covered waivers removed;
  harness file untouched — the make_stage1_manifest.py precedent) and check_transcript
  vs it must show `required: 65 checked: 65 missing: 0 unknown: 0`. claim.json's
  `covers` lists all 65. Only `embedding.lookup` remains waived, with the unchanged
  reason.
- **Forgery suite:** the submission must reject every existing forgery in
  `harness/forgeries/` (unchanged contract) — and this design hands the coordinator
  three NEW forgery candidates to add to the suite (suite only grows): (i) logit grid
  with one bumped logit at a non-argmax position + honest t* (must reject via rowmax
  DOM/chain edges); (ii) t* file swapped for second-best tokens (rejects at registration
  hash or T-BIND); (iii) softmax8 proof with a masked-sentinel at an allowed position
  (the evil=2 class, as a serialized forgery).
- **Scoring:** prove timing per HARNESS protocol (cold, exclusive GPU, median of 3 —
  all numbers in §5 are sequential-honest predictions, NOT official). DiFR scored by
  the coordinator on fresh held-out prompts; claim.json carries the §5.2 prediction
  with provenance "composed-estimate".

### 4.5 student.py (the DiFR scoring contract — flagged decision)

score.py's student contract is `replace(model) → int` (an in-place linears swap).
The integer chain cannot be expressed as a linears swap (DIFR_BASELINE §2 deviation 1).
Pinned submission shape: student.py `replace(model)` monkeypatches `model.forward` to
run the faithful-arch integer chain (the validated numpy semantics, torch-wrapped on
CUDA) over the input ids' embeddings — `forced_logits(student, ids)` then returns the
chain's logits; the function returns 22 (the number of integerized linears incl. o_proj,
lm_head). Whether a forward-replacement satisfies the harness's student contract is a
**coordinator ruling** (HARNESS.md gives the coordinator review authority over every
submission); flagged in §6.4 — if rejected, the fallback is scoring via the replicated
protocol (the DIFR_BASELINE measure/ path) with the deviation documented in the LEDGER.

---

## 5. Cost & Pareto predictions

All baseline rows are MEASURED (ORCHESTRATOR_REPORT stage 2, sequential-honest); all new
rows are composed estimates from measured per-shape reference points (rescale 2^20:
2.6/3.3 s; softmax: 10.6/12.2 s median; 768×768 fc: 2.2/1.65 s; rmsnorm trio: 14.3/13.9 s;
2^20-row commit ≈ 0.3–0.5 s; the me_weights host loop ≈ 60 µs/element — headslice-derived).
±30–40 % honest bars on every estimate. NOTHING here is an official timing (HARNESS.md
protocol: cold process, exclusive GPU, coordinator-run, median of 3 — still pending,
flagged since stage 2).

### 5.1 Added prove/verify cost

Part B (stage 3 — binds the same function):

| instance | runs | prove s | verify s | notes |
|---|--:|--:|--:|---|
| final_norm trio | 1 | ≈ 14.3 | ≈ 13.9 | measured shape |
| lm_head fc (1024×768×32768) | 1 | ≈ 90–120 | ≈ 140–180 | incl. ≈ 60/130 s slow gen-32768 IPAs |
| lm_head rescale (D = 2^25) | 1 | ≈ 180–230 | ≈ 230–290 | incl. ≈ 90/195 s slow IPAs |
| rowmax vpad (logit binding) | 1 | ≈ 140–220 | ≈ 160–260 | fast IPAs (§2.8) |
| **Part B total** | 4 | **≈ 420–580** | **≈ 540–740** | slow-IPA share ≈ 150/325 s, removable (§6.2) |

Part C (faithful-arch deltas; Part B included in the submission):

| instance | runs | prove s | verify s | notes |
|---|--:|--:|--:|---|
| o_proj fc + rescale | 2+2 | ≈ 10 | ≈ 11 | measured shapes |
| rowmax causal | 24 | ≈ 190–240 | ≈ 215–290 | the dominant Part C add |
| softmax8 − softmax delta | 24 | ≈ +12–36 | ≈ +25–50 | +1 binding block +3 IPAs, −2^20→2^14 limb table |
| headmerge (concat mode) | 2 | ±0 | ±0 | same volume |
| **Part C total** | — | **≈ +215–290** | **≈ +250–350** | |

End state (stage-2 base 743/1357* s + B + C): **prove ≈ 1380–1610 s ≈ 23–27 min**;
verify ≈ exclusive-GPU-median basis (~17–18 min stage 2) + 13–18 min ≈ **30–36 min**
(*stage-2 verify was contention-inflated; medians used). With the §6.2 header-lift
landed: prove ≈ −150 s, verify ≈ −325 s. Proof+commitment bytes: stage-2 143.6 MiB +
Part B ≈ 5.5 MB + Part C ≈ 35 MB ⟹ **≈ 185 MiB** (commitments are row-counts, so the
2^25-class grids stay cheap on disk: 1024–2048 rows = 144–288 KB each). Witness data/
grows ≈ +0.4 GiB (logits64.i64 256 MB + logits.i32 128 MB). One-time registration adds
≈ 30–45 s (ppgen 32768, lm_head export+commit, head pass for t*).

### 5.2 DiFR prediction for faithful-arch-v1 (the ~0.0077–0.008 argument)

Decompose exactly as DIFR_BASELINE did, both legs now favorable:

1. **Integer chain vs its own float replica** (integerization only): the stage-2
   machinery measured **2.4e-6 mean / 0.0 p99** for the same rounding classes this
   revision uses. The new segments add one rescale (o_proj), one rescale + matmul
   (lm_head, already inside that measurement's chain), and the softmax8 bracket
   (identical unique-integer mechanism, plus an EXACT max-shift — shift-invariant, no
   new approximation). Expected: stays ≤ 1e-5 — three orders below the next term.
2. **Float replica vs the FP8 teacher** (architecture + weight-grid only): the replica
   is now the TRUE llama-68m function (o_proj applied, plain concat, temperature 8) with
   weights on the 2^-16 grid. The closest measured anchor is
   `zkllm_native_fixed_point`: **difr_mean 0.0077 / p99 0.213 / top1 0.947** — a student
   that also runs the faithful architecture with 2^-16-grid linears. Residual
   differences, with expected signs: (a) it kept float norms/softmax/lm_head where we
   integerize — adds ≤ the leg-1 floor, negligible; (b) its weights are FP8-dequant
   values regridded to 2^-16 vs our round(checkpoint·2^16) — both are 2^-16-grid
   quantizations of nearly identical tensors; small, sign unknown; (c) it was measured
   on the repo dev prompt vs held-out dolly — TOKEN_CAPACITY saw input-dependence of
   ~2× in margin-sensitive metrics. **Prediction: difr_mean ≈ 0.008, honest band
   0.004–0.016 (×/÷2); difr_p99 ≈ 0.21–0.25; top1 ≈ 0.95.** Provenance:
   composed-estimate; the coordinator's held-out round is the test.
3. Stage 3 alone (Part B without C) does not move DiFR — it binds the same function
   (the final-norm R switch perturbs logits within the leg-1 class). Its Pareto value is
   coverage: logits and the served token become part of the proven statement.

Capacity-budget consequence (THREAT_MODEL §2 table, recomputed at the frontier): with
honest τ_p99 back at ≈ 0.21, the DiFR token channel returns to ≈ 0.047 bits/token
(TOKEN_CAPACITY's measured curve) and again dominates; the ZK side contributes rmsnorm
≈ 0.001 b/tok (now 5 sites incl. final norm), softmax8/rowmax/rescale/matmul/rope/
o_proj/lm_head all 0, plus the rowmax tie rows (§2.4, expected ≈ 0, measured per run).
Contrast the baseline-native point: τ_p99 ≈ 24 nats made the token check vacuous
(DIFR_BASELINE §7.6). **This submission is what moves the system from "proof of the
wrong function" to "checkable function with a meaningful token-level tolerance."**

### 5.3 Before/after Pareto points

| point | difr_mean | difr_p99 | prove (s, seq-honest) | coverage (frozen manifest) | provenance |
|---|--:|--:|--:|---|---|
| baseline-native (stage 2) | 8.988 | 23.996 | 743 | 54/56, logits/argmax unproven | measured |
| + stage 3 (Part B) | 8.988 (≈) | 24.0 (≈) | ≈ 1160–1320 | **56/56 + 3 covered-waived; t* = argmax bound** | composed |
| **faithful-arch-v1 (B+C)** | **≈ 0.008** | ≈ 0.21–0.25 | ≈ 1380–1610 | 56/56 + 9 covered-waived (scope 65/66) | predicted |

Reading: Part B buys the statement (necessary for ANY meaningful Pareto point — without
logit binding the DiFR axis is not attached to the proof); Part C buys ~3 orders of
magnitude of DiFR for ≈ +60–90 % prove time. Both are one frontier step; the LEDGER
records whatever the coordinator measures.

---

## 6. Open questions / risks (explicit; none blocks starting implementation)

1. **The rowmax selector tie channel** (§2.4) is the only nonzero covert term this
   design adds. Expected ≈ 0 (exact integer ties of row maxima; TOKEN_CAPACITY measured
   3/3072 tie positions, all tiled-prompt artifacts), but it is data-dependent.
   Pinned gate: prove_walk reports Σ log2(#maximizers) per run; the capacity table gets
   the measured row. The strict first-argmax upgrade (committed prefix tensor
   T = S·U_tri) is specified in concept but INFEASIBLE at the vocab shape (2^30-entry
   public matrix on the prover side) — if ties ever matter, the realistic mitigation is
   a public tie-breaking statement amendment, a coordinator decision.
2. **The me_weights/s-vector host loop** costs ≈ 150 s prove / ≈ 325 s verify across the
   lm_head fc + rescale gen-32768 IPAs (§3.2, §5.1) because zkob_fc/zkob_rescale are
   consumed unmodified. Sanctioned follow-up (NOT in v1): lift §2.8's fast helpers into
   vrf_common.cuh — a shared-header edit, hence the full all-driver selftest re-run rule
   applies. Do it as its own hardening pass with its own review, never bundled into this
   submission's diff.
3. **rowmax-vpad GPU memory** is predicted ≈ 8 GiB peak (§2.5) on the 24 GiB card —
   comfortable, but the 2^26 limb-lookup tensors are the largest the machinery has run.
   Gate: measure at selftest-real; fallback (only if > ~18 GiB observed): two
   1024×16384 column-block instances (V split 16384 + 15616) plus a 1024×2 vpad combine
   instance over the two block maxima, with block edges into the same lm_head com via
   block-slice openings — a headslice-pattern extension that needs its own §2-grade
   spec; deliberately NOT designed further because the primary path has 3× headroom.
4. **student.py contract** (§4.5): whether a forward-monkeypatch satisfies score.py's
   "swaps linears in-place" student contract is a coordinator ruling. Fallback: the
   replicated-protocol scoring path (DIFR_BASELINE §2 deviation 1), LEDGER-documented.
5. **z_'s domain is no longer lookup-bound in the faithful-arch chain** (softmax8's
   table indexes Dm, not z_). Soundness rests on chain determinism (§2.1 wrap note) +
   completeness throws — the accepted rmsnorm-M/rescale posture, but now load-bearing
   for the dominance bracket too. Optional belt if instance-local soundness is ever
   demanded: one tLookupRange(−2^19, 2^20) lookup on z_ per head (+~1.5 s/head).
   Not included in v1.
6. **No official timings exist anywhere yet** (LEDGER empty; the 743 s baseline is
   sequential-honest). Every number in §5 inherits that caveat; the coordinator's
   HARNESS-protocol run is the source of truth, and claim.json must carry provenance
   tags per the honesty rules.
7. **The LOW8 off-by-one corner** (§4.3 bounds): a spread of exactly 2^20 − 1 between
   two in-envelope scores is unrepresentable in the softmax8 table domain (sentinel
   occupies the top slot). Honest margin 3.7× (measured spread ≤ 283 K vs 1,048,574);
   guarded by an honest throw. If the envelope ever tightens the wrong way, the knob is
   LEN8 = 2^21 with D padded to match (N ≤ D forces stacking two heads per instance) —
   recorded, not designed.
8. **lm_head weight tying**: if `config.tie_word_embeddings`, the registered lm_head
   tensor IS the embedding matrix. Registration asserts and records the flag
   (`"lm_head_tied"`); semantics unaffected (the export reads model.lm_head.weight
   either way). Becomes interesting only when the embedding obligation lands (the same
   registered commitment could serve both — noted for the embedding stage).
9. **Table generation fidelity**: softmax8-exp-table is float64-generated; nonzero
   entries need e^{v/4096} for |v/4096| ≤ 11.8 — the same well-within-0.5-ulp regime as
   the softmax table's ±8 claim. sha256 registration, not regeneration, is the source of
   truth (standing rule; the C++ drivers never generate real tables).
10. **com_z triplication**: per head, the score grid commitment now exists in the
    rescale10, rowmax, and softmax8 obdirs (and com_mx twice). That duplication IS what
    the byte edges check (standing posture); the dedup opportunity (~0.5 MB/head)
    remains future work alongside the existing com dedup note.
11. **Headmerge flag revalidation scope** (§4.2): the edit touches a validated,
    audited driver. The pinned mitigation (both-mode selftest + cross-mode splice test +
    diff re-audit + all-driver selftest re-run) is process, not proof; the residual risk
    is the same class every flag-bearing driver carries and is why the diff is
    constrained to the gather formula + one absorb.
12. **Embedding stays waived** — prompt binding remains the input-digest sense until an
    embedding obligation exists. statement.logit_binding is real from Part B on, but for
    random-input runs t* binds "the argmax of the registered random input's logits";
    the same machinery binds real served tokens the day embeddings land.
