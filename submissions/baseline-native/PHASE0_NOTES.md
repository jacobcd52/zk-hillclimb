# Phase 0 — serialized proof + separate verifier for the llama-68m zkLLM pipeline

Author: implementation agent (baseline-native submission).
Status date: 2026-06-10.

## 0. TL;DR / honest headline

zkLLM (jvhs0706/zkllm-ccs2024) as shipped is **prover-only research code**. There
is **no serialized proof, no separate verifier, and no Fiat–Shamir transcript** in the
upstream codebase. The "verification" that exists is the prover checking *its own*
intermediate sumcheck identities in-process, and — critically — closing every sumcheck
by recomputing the final multilinear evaluation **from the witness** (`X`, `W`,
`rem`, …). That is not a verifier in the soundness sense: a malicious prover that
controls those tensors can make every check pass.

So Phase 0 is not "wire up the existing verifier". It is "build the missing verifier".
What is genuinely reusable from upstream:

- The **sumcheck round polynomials** themselves (the `vector<Polynomial>` / `vector<Fr_t>`
  objects). These are real and, once serialized together with the transcript randomness,
  can be checked **witness-free**: `p_i(0)+p_i(1) == claim_{i-1}` and `claim_i = p_i(r_i)`.
  This is the heart of sumcheck soundness and it is exactly what upstream throws away.
- The **commitment** primitive (`Commitment::commit_int`, BLS12-381 Pedersen/MSM) is real
  and binding. A weight commitment opening *can* be made witness-free by checking the
  opening proof (the `me_open` G1 transcript) against the committed point — but upstream's
  `me_open` produces that transcript and **never checks it**, and `verifyWeightClaim`
  re-opens using the full weight. So the opening-verify equation has to be written fresh.
- `FrTensor::save / from_int_bin / from_long_bin` + `ioutils.{savebin,loadbin}` give us a
  binary serialization for field elements. `Fr_t` is 8×uint32 = 32 bytes little-endian.

What is **prove-only with no witness-free check available without new circuit work**
(documented honestly in §4): zkSoftmax internal structure, rmsnorm, skip-connection,
the SwiGLU mapping lookup's *membership* binding to a committed activation, and the
cross-component Fiat–Shamir binding (there is none upstream — randomness is fresh
`random_vec` per component, so C5/C6-style splice/replay are not even defendable by the
upstream structure; a real transcript has to be introduced).

This submission therefore delivers a **real but partial** verifier: it makes the
sumcheck recursion and the commitment opening genuinely checkable end-to-end for the
matmul + rescaling-lookup + commitment obligations, with a Fiat–Shamir transcript that
binds the public statement and chains components. It does **not** claim coverage of the
obligations it cannot yet check; those are listed as gaps, not faked. A verifier that
"covers" them by accepting unconditionally is the exact failure this project exists to
catch (HARNESS.md hack #1), so they are left explicitly uncovered.

## 1. Component-by-component: what has real, checkable verify logic

Reading `proof.cu`, `zkfc.cu`, `tlookup.cu`, `rescaling.cu`, `zksoftmax.cu`,
`commitment.cu`, `self-attn.cu`, `ffn.cu`, `main.cu`:

| component | upstream "verify" | witness-free checkable? | notes |
|---|---|---|---|
| zkFC matmul (`zkip`/`zkip_stacked`) | inlined `if(claim!=p(0)+p(1))throw`; final claim = `X_red·W_red` from **witness** | **YES for the round recursion**; final claim needs an opening | round polys are the real sumcheck transcript. Final eval must be tied to a *commitment opening* of W and to the input-activation claim, not recomputed from witness. |
| commitment opening (`Commitment::open`/`me_open`) | produces G1 `proof` transcript, **never checked**; `verifyWeightClaim` re-opens with full weight | **YES** — the `me_open` fold identity `C = temp + r·(temp0+temp1-...)` is checkable against the committed point | upstream literally discards the proof vector. We serialize + check it. |
| rescaling (`Rescaling::prove`) | inlined `if(X(u)!=X_(u)*sf+rem(u))throw` using **witness** X,X_,rem; then `tl_rem.prove` | **PARTIAL** — the lookup *recursion* (tLookup phase1/phase2 round polys) is checkable witness-free; the `X = sf·X_ + rem` link needs the matmul-output claim + rescaled-output claim to be tied by opening | the range table membership is enforced by the lookup; the affine link is the missing glue. |
| tLookup (`tLookup_phase1/2`) | inlined `if(claim!=p(0)+p(1))throw` | **YES for the recursion** | logUp-style; round polys real. Final `B·m`/table eval needs the committed table (public) — table is deterministic from `(low,len)`, so verifier can recompute it. m (multiplicity) is witness → must be committed or folded. |
| zkSoftmax (`zksoftmax.cu`) | inlined prover checks | **NO (not in phase 0)** | multi-segment shifted-exp construction; verify side would need its own re-derivation. Honestly out of scope for phase 0; left UNCOVERED. |
| rmsnorm driver | python computes 1/sqrt, C++ proves via lookups | **NO (not in phase 0)** | inv-sqrt is a lookup-mapping; same situation as softmax. UNCOVERED. |
| skip-connection | elementwise add | trivially checkable but no proof object upstream | UNCOVERED in phase 0 (cheap to add later as an eq-claim). |
| Fiat–Shamir / transcript | **absent**; `random_vec` is fresh OS randomness | n/a | we INTRODUCE a transcript (§3) so verifier randomness is reproducible and bound to the statement. Without this, prove→serialize→verify cannot even agree on the challenge points. |

**Honest fraction estimate.** Of the 56 non-waived obligations in
`manifest_llama68m.json`, the ones whose verification this phase-0 design can make
*genuinely* witness-free and checkable are: the **sumcheck matmuls** (q/k/v/gate/up/down
+ scores/values), the **rescaling/range lookups**, the **commitment openings**, and the
three **statement** obligations (via the transcript). That is the large majority of the
*field-arithmetic* obligations. The ones it cannot yet check honestly are **softmax,
rmsnorm, swiglu-membership, skip-add** (≈ 8 obligation ids across 2 layers + lm_head
which is already waived). Phase 0 lands the matmul+rescaling+commitment+statement core
end-to-end (prove→serialize→verify→ACCEPT, and REJECT on all wired forgeries) and marks
the rest UNCOVERED rather than faking them.

## 2. Why the upstream "checks" are not a verifier (the load-bearing point)

In `zkfc.cu::zkip`:

```
if (claim != p(TEMP_ZERO) + p(TEMP_ONE)) throw;     // round check — REAL, witness-free
...
return zkip(p(u.back()), new_a, new_b, ...);          // folds the WITNESS a,b
```
and the caller closes with
```
auto opening = X.multi_dim_me(...) * W.multi_dim_me(...);   // <-- from witness X,W
if (final_claim != opening) throw;
```
The round checks are sound, but the *terminal* equation compares the folded sumcheck
value against an evaluation recomputed from the witness tensors `X` and `W`. A cheating
prover supplies whatever `X,W` make this hold. The only thing that pins `W` to reality is
the **commitment** — and upstream never verifies the opening of that commitment against
the folded point. So end-to-end soundness is missing precisely at the witness boundary.

Our verifier fixes the boundary for matmul+rescaling: the terminal `W`-evaluation is
checked by a **commitment opening proof** against the *registered* commitment, and the
terminal input-activation evaluation is carried as a claim into the upstream chain
(or, in phase 0's standalone scope, committed too). Softmax/rmsnorm terminals remain
unbound → those obligations stay UNCOVERED.

## 3. Serialized proof format (this submission's contract)

A proof is a **directory** `proof_dir/` containing:

- `public.json` — the public statement:
  ```json
  {
    "model": "JackFram/llama-68m",
    "seq_len": 1024,
    "prompt_token_ids": [...],            // binds A3/prompt_binding
    "output_digest": "<sha256 hex>",      // sha256 of committed output logits/int tensor
    "output_file": "output_int.bin",      // the committed per-position outputs (A1/A2/F3)
    "registered_weight_commitments": {    // binds B1..B4/registered_weight_hash
        "layer0.attn.q_proj": "<sha256 of commitment.bin>",
        ...
    }
  }
  ```
- `meta.json` — Fiat–Shamir transcript info:
  ```json
  {
    "fs_transcript_seed": "<sha256 hex>",   // = H(public.json bytes) ; the ROOT challenge
    "challenge_log": {"<obl_id>": {"u":[...],"v":[...],"r":[...]}},  // derived, reproducible
    "component_order": ["layer0.attn.q_proj.matmul", ...]  // chaining order (C5/C6)
  }
  ```
  The transcript seed is `H(public.json)`. Each component's challenge points are derived
  by hashing (prev_transcript_state ‖ component_id ‖ serialized round polys so far) →
  field elements. The verifier RE-DERIVES these; it does not trust meta.json's values,
  it only uses `component_order`. (This is what defeats C2/C5/C6: tampered polys change
  the hash, so the re-derived challenges diverge and the recursion check fails.)
- one subdirectory **per obligation id** (ids exactly from the manifest), e.g.
  `layer0.attn.q_proj.matmul/`, each containing the obligation's proof object:
  - `kind` (in a small `obl.json`): `sumcheck_matmul | rescaling_lookup | commitment_opening | statement`
  - `sumcheck_matmul/`: `round_polys.bin` (concatenated `Fr_t` coeffs; degree per round
    is fixed by the protocol so layout is `n_rounds × (deg+1)` field elements),
    `n_rounds.txt`, `claim0.bin` (the initial claim Fr_t), `dims.txt`, plus the terminal
    `W_eval.bin` / `X_eval.bin` (claimed evaluations to be discharged by the opening).
  - `rescaling_lookup/`: phase1 + phase2 round polys (`phase1.bin`, `phase2.bin`),
    `m.bin` reference (committed multiplicity), `scaling_factor.txt`, `claim0.bin`.
  - `commitment_opening/`: `open_proof.bin` (the G1 `me_open` transcript, each step =
    3 G1Jacobian points), `commitment.bin` (the committed point, must hash to the
    registered value in public.json), `point.bin` (the evaluation point u), `eval.bin`
    (claimed opening value — must equal the matmul terminal `W_eval`).
  - `statement/`: nothing extra; checked purely from public.json + meta.json.

`Fr_t` on disk = 8 × little-endian uint32 (matches `FrTensor::save`). G1Jacobian =
3 × Fr_t (x,y,z) = 96 bytes (matches `G1TensorJacobian` save layout).

Transcript consumed by `harness/check_transcript.py`:
```json
{"verdict": "ACCEPT"|"REJECT",
 "checked": ["layer0.attn.q_proj.matmul", "layer0.attn.q_proj.commitment_opening", ...],
 "details": {"<id>": {"ok": true, "reason": "..."}}}
```
`checked` lists only obligations that PASSED a real check. A failed obligation flips
`verdict` to REJECT and is recorded in `details` with `ok:false`.

## 4. Honest gaps (UNCOVERED in phase 0 — do NOT score as covered)

- softmax (`layer{0,1}.attn.softmax`), rmsnorm (`*.input_norm`, `*.post_attn_norm`),
  swiglu (`layer{0,1}.mlp.swiglu`), skip-adds (`*_skip.add`): verify side not yet
  written. These remain prover-only. Marked UNCOVERED.
- Cross-component Fiat–Shamir is INTRODUCED by us (upstream has none). Until every
  component feeds the same transcript, C5 (splice) / C6 (replay) are only defended for
  the components that DO chain (matmul→rescaling→opening within q_proj). Documented.
- The input-activation terminal of each matmul is, in standalone phase-0 scope, bound by
  committing the activation too (so the verifier checks it like a weight opening). Tying
  it to the *previous* layer's output instead of a fresh commitment is layer-chaining
  work deferred past phase 0.

## 5. Build / run

See `build.sh` (compiles the new `zkprove_dump` and `zkverify` drivers against the
existing `.o` files on /root), `prove.sh`, `verify.sh`. Sources are mirrored to
`/workspace/projects/zk-hillclimb/zkllm-src/` for persistence.

## 6. Coverage status (updated as implementation lands — see end of file)
</content>

## 7. COORDINATOR ADDENDUM (2026-06-10): commitment-opening verify algebra PINNED

I (coordinator) rebuilt and ran the toy ground-truth harness (`zkllm-src/vrf_toy_open.cu`,
runs the REAL `Commitment::me_open` prover-side). **TOY-OPEN-VERIFY: ALL PASS** on three
cases: (GEN=8,N=32), (GEN=6,N=24) [odd level inside the me_open fold], (GEN=8,N=24)
[odd commitment-row count]. The witness-free verify chain, confirmed bit-exact:

1. `C_0` = ME of the public row commitments at `u_out` computed with **raw challenge
   limbs** (integer view). Do NOT use `G1TensorJacobian::operator()(u)`: its `G1_me_step`
   kernel `unmont`s the challenge, which is inconsistent with `me_open_step` /
   `Fr_partial_me_step` (both use raw limbs). Upstream never noticed because `open()`'s
   `g_temp = com(u_out)` is dead code. **This was the bug the first implementation agent
   died on.**
2. Per round i (proof triple T,T0,T1; challenge u): check `T == C_i`, then
   `C_{i+1} = (1-u)^2*T0 + u(1-u)*T + u^2*T1`. Coefficients in the integer view:
   plain modmul(a,b) = `mont(montmul(a,b))`.
3. `G_final` recomputed from the PUBLIC generators only, fold `g' = g1 + u*(g0-g1)`
   (opposite orientation from the scalar fold!), never read from the proof.
4. Accept iff `C_L == G_final * eval`. Forged eval (+1) rejected.

**Miscompilation gotcha (affects all new verifier kernels):** under the project's
`-dlto` build, a kernel with two `G1Jacobian_mul`-bearing branches is silently
miscompiled (bisection: `zkllm-src/vrf_toy_debug2.cu`). All fold kernels must be
single-branch; odd levels are handled by padding with the identity point (all-zero
bytes), which exactly reproduces upstream's odd-branch semantics in both orientations.
G1 point equality: `a - b` has z == 0 (Jacobian coords aren't unique).

## 8. COORDINATOR ADDENDUM (2026-06-10): sumcheck (zkip) verify algebra PINNED

Second toy ground-truth harness (`zkllm-src/vrf_toy_matmul.cu`, runs the REAL upstream
`zkip` prover + `Commitment::me_open`). **TOY-MATMUL-VERIFY: ALL PASS** on four cases:
(B=4,IN=8,OUT=4), (B=4,IN=6,OUT=3) [both dims non-pow2], (B=2,IN=8,OUT=3), (B=8,16,16).

Pinned, witness-free sumcheck verification for one matmul claim:

1. **Serialization of round polys:** `Polynomial`'s coefficients are private; ship each
   degree-2 round poly as THREE EVALUATIONS p(0), p(1), p(2) extracted with
   `Polynomial::operator()` (same integer-view arithmetic as the kernels).
2. **Round order:** zkip consumes challenges from the END — round r uses
   `u_input[L-1-r]`. Per round: check `claim == p(0) + p(1)` (integer-view add), then
   `claim' = Lagrange3(p0,p1,p2)(u_r)` with
   `p(u) = p0*(u-1)(u-2)/2 + p1*u(2-u) + p2*u(u-1)/2`.
   INV2 = (r+1)/2 hardcoded as integer-view limbs
   `{2147483649, 2147483647, 2147429887, 2849952257, 80800770, 429714436, 2496577188, 972477353}`
   with a startup self-check `2*INV2 == 1`. Lagrange3 cross-checked bit-exact against
   `Polynomial::operator()(u)` every round.
3. **Terminal:** `claim_final == claim_X * claim_W` (plain integer-view product).
   `claim_X` chains to the previous layer / committed input; `claim_W` is discharged by
   the §7 opening verify.
4. **Glue (verifyWeightClaim mapping, confirmed empirically):** `u_cat = u_output ++
   u_input`; `W_padded = W.pad({IN, OUT})` (per-dim pow2). Commitment rows = IN_pad side,
   folded by `u_input` (C-chain); generators = OUT_pad side, folded by `u_output`
   (G_final + round challenges). **The me_open eval equals claim_W bit-exactly**, and the
   final opening check can be run directly against claim_W: `C_L == G_final * claim_W`.
5. **Padding is transparent:** `W_padded.multi_dim_me({u_input,u_output},{IN_pad,OUT_pad})
   == W.multi_dim_me(..., {IN,OUT})` bit-exact (partial_me's odd branch == implicit
   zero-fill). zkip's internal zero-fill (`N_out = (1<<ceilLog2(size))>>1`) likewise
   needs no special handling.
6. **Forgeries rejected at toy scale:** tampered round poly (p(1)+1) breaks the
   p(0)+p(1) check; claim_W+1 breaks BOTH the terminal product check and the opening
   final check.

Remaining to pin: tLookup phase1/phase2 verification, Fiat-Shamir challenge derivation,
then the real `zkprove_dump`/`zkverify` drivers (agent drafts are reference-only).

## 9. COORDINATOR ADDENDUM (2026-06-10): tLookup (logUp) verify algebra PINNED

Third toy harness (`zkllm-src/vrf_toy_lookup.cu`). **TOY-LOOKUP-VERIFY: ALL PASS** on
(D=32,N=8,low=-3,len=5), (D=16,N=16 — v1 empty, pure phase2), (D=64,N=4).

Key upstream fact: **`tLookup::prove`'s `proof` parameter is NEVER written** — upstream
serializes nothing for lookups; every round poly is recomputed from witness. Our prover
driver must replicate the phase1/phase2 recursion and serialize each round poly as
FOUR evaluations p(0..3) (**round polys are degree 3** here: deg-2 product x deg-1 eq).

Pinned witness-free verification:
1. Anchor claim RECOMPUTED by verifier: `claim_0 = alpha + alpha^2` (never trusted).
2. Rounds: phase1 (|v1| = log(D/N) rounds, v1 end-first) then phase2 (|v2| = logN
   rounds, v2 end-first); u pairs end-first across both phases: round k uses
   u[logD-1-k]. Per round: `claim == p(0)+p(1)`, then `claim' = Lagrange4(p(0..3))(v_k)`
   (needs INV6 = inv(6); runtime self-check 6*inv6 == 1).
3. Terminal (derived and empirically confirmed):
   `claim_L == alpha_acc*A_f*(S_f+beta) + inv_ratio*alphasq_acc*B_f*(T_f+beta)
               + A_f - inv_ratio*m_f*B_f`
   where inv_ratio = N/D (verifier recomputes as N*inv(D));
   alpha_acc = alpha * prod over ALL rounds of eq(u_k, v_k);
   alphasq_acc = alpha^2 * prod over PHASE2 rounds only;
   eq(u,v) = 2uv - (u+v) + 1 (upstream eqEvalKernel), integer view.
4. **B_f and T_f are verifier-recomputable from the PUBLIC table**: B = pointwise
   unmont(inverse(mont(T+beta))), then fold both with the upstream front/back-half
   reduce (new[i] = a[i] + mont(v)*(a[i+half]-a[i])) over v2 end-first. Confirmed
   bit-exact vs the prover's folded values.
5. Remaining obligations: A_f and m_f are commitment-opening claims (use §7 verify);
   S_f chains to the layer tensor claim. For tLookupRangeMapping::prove (the variant
   zkrelu/zksoftmax/rescaling actually use), S_com = S_in + r*S_out and
   T_com = table + r*mapped_vals with mapped_vals PUBLIC (model definition), so T side
   stays verifier-recomputable; same algebra otherwise.
6. Forgeries rejected: tampered round poly; m_f+1; A_f+1; and the SEMANTIC forgery
   (one out-of-range value in S with honest-procedure A/m) — fails the recomputed
   round-0 anchor, as it must.
7. Cross-TU `extern "C" __global__` declarations of upstream kernels link and run fine
   under -dc -dlto (used for tlookup_inv_kernel + both reduce kernels).

Status: opening (§7) + matmul sumcheck (§8) + lookup (§9) algebra all pinned with
ground-truth harnesses. Next: Fiat-Shamir transcript, then the real
zkprove_dump/zkverify drivers over llama-68m's layer obligations.

## §10. me_open is UNSOUND (steering attack) → replaced with a Fiat-Shamir IPA (vrf_toy_ipa.cu, ALL PASS)

**The hole.** In upstream's `Commitment::me_open`, the fold coefficients are the
evaluation point coordinates — known to the prover BEFORE it constructs the
per-round T0/T1 points, which are otherwise unconstrained. So the prover can
*steer* the chain: at the last round set `T1' = T1 + u_last^{-2} * Delta` with
`Delta = G_final * (eval' - eval)`; then `C_L` shifts by exactly Delta and the
§7 verify chain ACCEPTS the forged `eval'`. **Demonstrated live** in
vrf_toy_ipa.cu on all cases: old verify accepts forged eval+1. This is not an
implementation bug — the protocol is unsound even interactively, because
`open()` takes the whole challenge vector as an input. (Same class of issue:
upstream `zkip` receives all sumcheck challenges upfront; in the NI setting
every round's challenge must be derived AFTER absorbing that round's message.)

**The fix (pinned, ALL PASS at GEN=8/N=32, GEN=16/N=64, GEN=8/N=24):**
1. `fs_transcript.hpp`: dependency-free SHA-256 Fiat-Shamir transcript.
   `state = SHA256(seed)`; absorb(label, data) rehashes state||label||data;
   challenges via SHA256(state||"chal") with state ratchet. Challenge → Fr:
   8 LE uint32 limbs, top limb % 1944954707 (same distribution as random_vec,
   < r guaranteed), re-derive on zero.
2. Opening replaced by a Bulletproofs-style **IPA** over the GEN-sized
   within-row vector (hybrid: the COM-sized row fold stays the verifier's own
   deterministic §7 step-1 computation):
   - Statement: `C_0` = raw-limb com fold at u_out (== `<g, t_row>`, asserted),
     `b` = ME weights of u_in with **bit i of index k pairing with u_in[i]**
     (pinned: `<t_row, b> == me_open eval` bit-exact), claim `eval = <a, b>`,
     `P_0 = C_0 + eval*Q` (Q = one extra pp generator, Commitment::random(1)).
   - Round (lo = front half, hi = back half):
     `L = <a_lo, g_hi> + <a_lo, b_hi>*Q`, `R = <a_hi, g_lo> + <a_hi, b_lo>*Q`;
     absorb L,R → x; fold `a' = x*a_lo + x^{-1}*a_hi`,
     `b' = x^{-1}*b_lo + x*b_hi`, `g' = x^{-1}*g_lo + x*g_hi`;
     `P' = x^2*L + P + x^{-2}*R`.
   - Final: prover sends `a_f`; verifier recomputes `g_f` (s-vector MSM with
     `s_i = prod_r (bit_{MSB-r}(i) ? x_r : x_r^{-1})`, cross-checked bit-exact
     against the explicit fold), folds `b_f` itself, checks
     `P_L == a_f*g_f + (a_f*b_f)*Q`. All Fr ops integer view.
3. Transcript binds the full statement (com bytes, u_out, u_in, eval) before
   any round challenge.

**Forgeries rejected:** forged eval+1 with honest transcript; **adaptive**
forged eval+1 where the prover re-runs the whole IPA honestly on the false
claim (the steering-attack analog — full freedom over L/R, still rejected);
tampered round L; tampered a_f.

**Side findings:**
- A straight-line kernel with TWO `G1Jacobian_mul` calls is CLEAN under the
  -dlto build — the §7 miscompilation is specific to the two-BRANCH shape.
  (Probe kept in vrf_toy_ipa.cu; single-branch fold kernels remain the rule.)
- GEN must be a power of two for the IPA. pp generators already are; weight
  matrices are padded (§8).
- Documented and accepted for phase 0: pp generators are known-dlog
  (Commitment::random = G*r — the auditing side runs setup, binding is what we
  need); the IPA leaks a_f and eval (no blinding) — weight privacy of opened
  rows deferred.
- IMPLICATION FOR DRIVERS: every sumcheck (zkip, lookup phases) must derive its
  round challenge from the transcript after absorbing that round's poly evals.
  My replicating recursions (§8, §9) compute rounds one at a time, so this drops
  in directly; never hand upstream provers a pre-derived full challenge vector.

Status: §7 opening chain is now DEMOTED to consistency-checking honest
transcripts; soundness comes from the IPA (§10). All algebra pinned. Next: the
real zkprove_dump / zkverify drivers over llama-68m's layer obligations.

## §11. Rescaling obligation driver (zkob_rescale.cu) — DONE, validated at scale

Real driver for one `rescaling_lookup` obligation: X = sf·X̂ + rem with
rem ∈ [−sf/2, sf/2). Two parts:

1. **Homomorphic affine link** — checked directly on public commitments,
   per row: `com_X[j] == sf·com_X̂[j] + com_rem[j]`. No opening needed;
   zero-padding satisfies it trivially. (A batched CUDA kernel for this was
   ANOTHER -dlto miscompile victim — one straight-line G1Jacobian_mul + adds +
   z-test, wrong on all rows; the check now uses the proven 1-thread helpers.)
2. **logUp range proof** on rem vs the public table tLookupRange(−sf/2, sf)
   (§9 algebra), per-round Fiat-Shamir (absorb 4 evals → round challenge),
   anchor α+α² recomputed, B_f/T_f recomputed from the public table, and the
   terminals bound by THREE FS-IPA openings (§10): A_f vs com_A (committed
   after β), S_f vs com_rem, m_f vs com_m. Opening points are the reversed
   round-challenge sequences (full for D-sized A/rem, phase-2-only for m).

FS schedule: absorb B, C, LOG_SF, com_X, com_X̂, com_rem, com_m → β →
absorb com_A → α → u (logD) → per-round 4 evals → w_r → absorb A_f,S_f,m_f →
IPA(A), IPA(rem), IPA(m).

Selftest ALL PASS on (8,4,sf=16), (4,16,sf=16: m fits one commitment row),
(8,6,sf=64: zero phase-1 rounds). Forgeries rejected: round-poly byte, m_f
terminal, ipa a_final, com_A/com_rem/com_m point tampering, and the SEMANTIC
covert-channel forgery — X̂[i] += 1 compensated by rem[i] −= sf, which keeps
the affine link exactly valid and is caught by the range lookup (this is
precisely the attack channel the obligation exists to close; the verifier
rejects at round 0 because the multiplicities can't match an out-of-range rem).

Real scale (B=1024 seq, C=768, sf=2^16 → D=2^20 lookup, N=2^16 table):
prove 2.6 s, verify 3.3 s, ACCEPT; byte-tamper at scale REJECTED.
Proof dir ~600 KB (four 144 KB row-commitment files dominate; com_X/com_X̂
are shared with the adjacent matmul obligations once chained).
15 rescaling obligations ≈ 40 s prove / 50 s verify total.

## 12. Chaining validated (matmul → rescale), int64 activation files

- `zkob_fc prove ... Y_out.bin` now writes Y as **int64** (`save_long`); values are
  pre-rescale products up to ~2^42 at sf=2^16, so int32 on disk would overflow.
- `zkob_rescale prove` consumes the int64 file via a driver-local `load_long_tensor`
  (host fread + `FrTensor(uint, const long*)` ctor). **Upstream `FrTensor::from_long_bin`
  is buggy** — it mallocs/loads `sizeof(int)*size` bytes for long data, silently producing
  a garbage tensor. Upstream left pristine; fix lives in the driver. (GOTCHAS updated.)
- Chain test at real scale (B=1024, IN=OUT=768, sf=2^16), same generators both sides:
  - `zkob_fc prove /tmp/ob_chain_fc ... /tmp/Y_chain.i64.bin` → `zkob_fc verify` **ACCEPT**
  - `zkob_rescale prove /tmp/ob_chain_rs ... /tmp/Y_chain.i64.bin` → `verify` **ACCEPT**
  - `cmp com_Y.bin com_X.bin` → **byte-identical** — the orchestrator's chaining check
    (commitment-file equality between adjacent obligations) works as designed.
- Note: `zkob_rescale prove` expects the obligation dir to already exist (orchestrator
  must mkdir; the driver does not).

## 13. zkob_glu — nonlinearity (swiglu) obligation driver: SELFTEST ALL PASS + real-scale chain

Covers manifest id `*.mlp.swiglu` (nonlinearity_lookup), composed of three parts:
1. **Mapping lookup** — every pair (G[i], S[i]) must be a row of the PUBLIC table
   {(x, silu(x))}: logUp on the combined witness G + r·S vs combined table
   table + r·mapped, r an FS challenge derived AFTER all commitments. The
   verifier forms the combined commitment HOMOMORPHICALLY (com_G + r·com_S,
   1-thread helpers) and opens the lookup terminal against it. This also
   range-binds G (out-of-range gate values have no table row). ZERO covert
   freedom: the mapping is exact (the table IS the integerized silu spec).
2. **Hadamard sumcheck** — H = S ⊙ U via eq-weighted degree-3 sumcheck
   (claim_H = H~(u_h) = Σ_b eq(u_h,b)S(b)U(b)); new Fr-only kernels
   k_eq_expand (eq-tensor doubling) and k_hp3_step (4-point round polys) —
   Fr kernels are not in the -dlto miscompile family, and the 22-round
   p(0)+p(1)==claim chain against an independently computed claim_H is a
   built-in runtime probe of them.
3. **Hidden rescale** — H → H_ (sf 2^16) is a SEPARATE zkob_rescale run; the
   orchestrator checks com_H (glu) == com_X (rescale) byte-identically, and
   com_Xr (rescale) chains into down_proj.matmul's com_X.

Shared machinery now lives in `zkob_lookup.cuh` (extracted verbatim from the
validated zkob_rescale.cu: fs_phase1/2, open_prove/verify, my_eq, lagrange4,
k_fr_fold, k_bump, LookupProof, load_long_tensor). zkob_rescale selftest
re-run after the refactor: ALL PASS.

Selftest (3 cases incl. n1=0 and padded-C/multi-row m): 9 byte-tamper
forgeries + 2 SEMANTIC forgeries per case, all rejected:
- wrong-mapping (S[i]+=1, hadamard kept consistent) → lookup rejects
- wrong-product (H[i]+=1, lookup kept consistent) → hadamard rejects

Real scale (llama-68m MLP: B=1024, C=3072→4096, D=2^22; silu table
low=−2^21, len=2^22, in-scale 2^12, out-scale 2^16, table pinned by file):
prove 11.4s, verify 13.8s, proof+commitments 904KB, honest ACCEPT,
tamper REJECT, chain into zkob_rescale byte-identical and both ACCEPT.
Note: the H chain file is the UNPADDED B×C int64 tensor (pad stripped).

Per-layer cost: 2 layers × (11.4 + rescale 9s-ish) — swiglu is the priciest
obligation so far (D is 4× the matmul/rescale D).

## 14. zkob_rmsnorm — inverse-RMS obligation driver: SELFTEST ALL PASS (48/48) + independent audit (SOUND)

Covers the rmsnorm obligation: binds the inverse-RMS advice R to the committed
activation X within ±1 integer tolerance, closing the unbound `rms_inv_temp`
covert channel (residual freedom ≤ log2(3) ≈ 1.6 bits/row — the documented
floor, ~6.5 Kbit/forward over 4 norms). With M[s] = Σ_j X[s,j]² + C_eps
(C_eps = round(eps·C·2³²), u64 CLI arg), the verifier forces
(R−1)²·M ≤ 2⁶⁴·C ≤ (R+1)²·M per row. Implementation passed an independent
soundness audit (RMSNORM_REVIEW.md, VERDICT: SOUND — no critical/major
findings), plus a hardening round adding the audit's three requested semantic
evil modes (see below). One FS transcript, 17 IPA openings, seven
sub-obligations:

1. **Limb range lookup** — P1 = 2⁶⁴C − (R−1)²M and P2 = (R+1)²M − 2⁶⁴C
   decomposed into 5×16-bit limbs each, packed into the limb matrix L
   (max(16, 65536/B_pad) rows × B_pad cols, gen_B), logUp vs
   tLookupRange(0, 65536) (§9 algebra); B_f/T_f recomputed by the verifier
   from the public table.
2. **Homomorphic affine limb links** — com_P1 == Σ_{i<5} 2^{16i}·com_L_row[i]
   (and com_P2 vs rows 5–9), checked with the proven 1-thread helpers
   (h_mul/h_add/g1_eq). No new G1 kernels anywhere in the driver (-dlto rule).
3. **SS sumcheck** binding M to X: M̃(u_b) − C_eps = Σ eq·X·X over logD vars;
   eq factor only for the logB row rounds; S_f2 == U_f2 REQUIRED
   (load-bearing). Openings: X at the terminal point, M̃(u_b) vs com_M.
4. **Bracket quartics** (Lagrange-5, tags "q1"/"q2"): claim_q1 =
   2⁶⁴C − P̃1(u_b2) over (eq, R−1, R−1, M); claim_q2 = P̃2(u_b2) + 2⁶⁴C over
   (eq, R+1, R+1, M). No commitment for R∓1: R opens at pt1 expecting
   q1.A_f + 1 and at pt2 expecting q2.A_f − 1; A_f == B_f REQUIRED
   (load-bearing). Together with 1+2 (P1, P2 ∈ [0, 2⁸⁰) as integers), any
   integer R off by ≥2 makes one bracket negative → no in-range limb
   decomposition of the true value (affine link fails) and no consistent
   in-range substitute (quartic round chain fails) — both directions
   forgery-tested.
5. **Outer product** W = R×g via single-point MLE factorization:
   val_W == val_R·val_g at fresh (u_b3, u_c3); val_g opens against the
   REGISTERED com_g path (discharging the norm-weight commitment_opening id).
   B == B_pad required in BOTH prove and verify.
6. **Internal rescale** W_ = rescale(W, 2¹⁶) — committed here (com_W_), proven
   by a SEPARATE zkob_rescale run (chain interface below).
7. **Hadamard** Y = W_ ⊙ X. Openings: Y@u_h, W_@pt, X@pt.

FS schedule: absorb B, C, C_eps(lo,hi), com_X, com_g, com_R, com_M, com_P1,
com_P2, com_L, com_m_L, com_W, com_W_, com_Y → β → com_A_L → α → u_L →
lookup rounds + terminals + 3 openings → u_b → ev_M → SS rounds + 2 openings
→ u_b2 → ev_P1, ev_P2 → q1/q2 rounds + 6 openings → u_b3, u_c3 →
val_R/val_g/val_W + 3 openings → u_h → claim_Y → hadamard rounds + 3 openings.
Audit confirmed prove/verify schedules absorb-for-absorb identical, every
challenge squeezed only after the message it binds.

CLI:
```
zkob_rmsnorm prove  <obdir> <seed> <X-int32> <R-int32> <g-int32> <B> <C>
                    <C_eps-u64> <gen_C> <gen_B> <q> [W-i64-out Y-i64-out]
zkob_rmsnorm verify <obdir> <seed> <B> <C> <C_eps> <com_g-path>
                    <gen_C> <gen_B> <q>
zkob_rmsnorm selftest
```

Files in <obdir> (36, all byte-tamper-tested): dims.bin; lookup.bin, hpss.bin,
qp1.bin, qp2.bin, outer.bin, hp.bin; com_X / com_g / com_R / com_M / com_P1 /
com_P2 / com_L / com_m_L / com_A_L / com_W / **com_Wr** / com_Y .bin; 17
ipa_*.bin (AL, L, mL, X_ss, M, P1, P2, R_q1, M_q1, R_q2, M_q2, R_o, g, W, Y,
Wr, X_h). **NOTE: the spec's com_W_ is saved as `com_Wr.bin`** (the transcript
label stays "com_W_"); the orchestrator's manifest must use the on-disk name.

Chain interface: writes W.i64 and Y.i64 (both UNPADDED B×C int64; W from the
exact host ints, Y via save_long + column-pad strip). The internal rescale is
closed by a separate zkob_rescale run on W.i64, with commitment byte-equality
com_W (here) == com_X (rescale) and com_W_ (here, file com_Wr.bin) == com_Xr
(rescale). com_Y chains downstream; eps must match the pipeline (llama-68m:
rms_norm_eps = 1e-6 → C_eps = 3298535 at C=768, NOT 1e-5).

**Two orchestrator obligations pinned by the audit:**
- **(MINOR-5) com_X here MUST be chained byte-identical to the upstream
  activation commitment.** The ±1 bound on R relies on M being the honest
  bounded integer derived from a chained X; with a prover-chosen X the prover
  can craft M tiny and the two 2⁸⁰-wide windows admit ~2³⁹ integer values of
  R — a wide-open covert channel. A standalone ACCEPT of this obligation
  proves much less than the chained one.
- **(MINOR-7) The com_W_ byte-equality check against zkob_rescale must look
  for `com_Wr.bin`** — wiring the manifest with the spec name com_W_.bin will
  fail to find the file.

Selftest (after the hardening round): **48 PASS / 0 FAIL, exit 0**. Small case
(B=8, C=5): honest ACCEPT; EIGHT semantic evil modes, each rejected by exactly
the named check (R+2 mod-p P1 → affine link P1; R+2 limb-reconstruction P1 →
q1 round 0; M+1 → SS round 0; Y+1 → hadamard round 0; W+1 → outer product;
and from the audit's MINOR-1/2: R−2 mod-p P2 → affine link P2; R−2
limb-reconstruction P2 → q2 round 0; out-of-range limb L[0,s] += 2¹⁶ with a
compensating borrow — affine links, quartics and all other commitments stay
consistent — → limb lookup round 0, proving the range lookup itself is
load-bearing); 36 byte tampers (every file the verifier reads) + restored
ACCEPT. Real scale (B=1024, C=768, C_eps=3298535): honest ACCEPT,
**prove 9.1 s, verify 7.0 s, proof+commitments 678,876 bytes (~663 KB)**;
byte tamper at scale rejected.

Per-norm cost: ~9 s prove / ~7 s verify, plus the chained zkob_rescale run on
W.i64; the manifest determines the number of rmsnorm sites per forward.

## 15. zkob_softmax — attention softmax obligation driver: SELFTEST ALL PASS (122/122) + independent audit (SOUND)

Covers manifest id `layer{l}.attn.softmax.h{hh}`: binds, per head, the B×NCOL
int32 score grid z_ (scale 2⁹, chained from scores_rescale stage 2) to the
softmax output P = round_half_up(2¹⁶·MK·E/S) with P[masked] = 0, where
E[i,j] = X_E[z_[i,j] − LOW_E] is the registered integer exp table and
S[i] = Σ_{j≤i} E[i,j] the causal row-sum. **Zero prover advice — 0-bit covert
capacity**: every committed tensor (E, S, P, the residual limbs L) is a
deterministic function of (z_, the public causal mask MK, the registered
table X_E); the rounding is exact (unique-integer bracket, both sides forced),
so an accepting proof leaves the prover no freedom anywhere in the head.
Passed an independent soundness audit (SOFTMAX_REVIEW.md, VERDICT: SOUND — no
critical/major findings; FS schedules absorb-for-absorb identical; every
disk value anchored; both bracket directions independently forgery-tested),
plus a hardening round (audit MINOR-1/MINOR-2; see SOFTMAX_REPORT.md). One FS
transcript, 16 IPA openings, six sub-obligations:

1. **Exp mapping lookup (R1)** — glu pattern: comb = z_ + r·E vs the combined
   public table table + r·mapped (r squeezed after ALL base commitments); the
   verifier forms com_comb = com_z + r·com_E HOMOMORPHICALLY (1-thread
   h_mul/h_add; no new G1 kernels anywhere — the -dlto rule) and recomputes
   B_f/T_f from the public table. Also domain-binds z_ ∈ [LOW_E, LOW_E+LEN_E).
2. **Limb range lookup** — L = 4 planes (r1 lo/hi, r2 lo/hi; 20-bit limbs at
   real scale, r2 = 2S−1−r1) vs tLookupRange(0, LEN_R), D_L = 4D. Sole
   enforcer of r1, r2 ∈ [0, LEN_R²) as integers — load-bearing for the
   bracket, and semantically forgery-tested (evil=6, hardening round).
3. **Row-sum sumcheck (R2)** — ev_S = S̃(u_b) = Σ W_rs·E·𝟙 with
   W_rs = bcast_rows(eq(u_b)) ⊙ MK; the verifier rebuilds and folds the
   public weight ITSELF and REQUIRES U_f2 == 1 (load-bearing).
4. **Bracket V1** — c1 = MLE(MK⊙E)(u_r), weight eq(u_r)⊙MK recomputed by the
   verifier; U_f2 == 1 required.
5. **Bracket V2** — c2 = MLE(P⊙S_bcast)(u_r), pure-eq weight; S_bcast is
   NEVER committed: its terminal U_f2 opens against com_S at the row-bit
   suffix of pt2 (broadcast-MLE = row-vector MLE at the row bits; LSB-first
   columns make the suffix exactly the row coordinates).
6. **Residual reconstruction** — four Boolean-plane openings of com_L at u_r
   (v00/v10/v01/v11) + S_id = S̃(u_r rows) vs com_S, then two plain-field
   identities: I1: 2¹⁷·c1 + S_id − 2·c2 == v00 + LEN_R·v10 ("bracket r1
   identity"); I2: r̃1 + r̃2 + 1 == 2·S_id ("bracket sum identity"). With the
   limb lookup (2·LEN_R² = 2⁴¹ < p, 2S < 2⁴⁰) these force r1 ∈ [0, 2S) as an
   integer ⇒ P is the exact round-half-up, masked entries exactly 0.

FS schedule (seed = run_seed:obligation_id): absorb B, NCOL, LOW_E, LEN_E,
LEN_R, LOG_OUT, the 7 base commitments (com_z, com_E, com_P, com_S, com_L,
com_m_E, com_m_L) → r, β_E → com_A_E → α_E, u_E → exp-lookup rounds +
terminals + 3 openings → β_L → com_A_L → α_L, u_L → limb-lookup rounds +
terminals + 3 openings → u_b → ev_S → row-sum rounds + 2 openings → u_r → c1
→ V1 rounds + 1 opening → c2 → V2 rounds + 2 openings → v00..v11 + 4 plane
openings → S_id + 1 opening → I1, I2 (verifier-only). Audit confirmed
prove/verify absorb-for-absorb identical; every challenge squeezed only after
the message it binds.

CLI (LOG_OUT = 16 pinned in-driver):
```
zkob_softmax prove  <obdir> <seed> <z-int32.bin> <B> <NCOL> <LOW_E> <LEN_E>
                    <expmap-int32.bin> <LEN_R> <gen.bin> <q.bin> [P-int32-out.bin]
zkob_softmax verify <obdir> <seed> <B> <NCOL> <LOW_E> <LEN_E>
                    <expmap-int32.bin> <LEN_R> <gen.bin> <q.bin>
zkob_softmax selftest
```
Layout: B == NCOL required (both pow2), single generator set gen (size NCOL),
no padding anywhere. Real scale per head: B = NCOL = 1024, LOW_E = −2¹⁹,
LEN_E = LEN_R = 2²⁰.

Files in <obdir> (32, all byte-tamper-tested): dims.bin; com_z / com_E /
com_P / com_S / com_L / com_m_E / com_m_L / com_A_E / com_A_L .bin;
lookup_E.bin, lookup_L.bin; hp_rs.bin, hp_v1.bin, hp_v2.bin; lvals.bin; 16
ipa_*.bin (A_E, comb, m_E, A_L, L_lk, m_L, E_rs, S_rs, E_v1, P_v2, S_v2,
L00, L10, L01, L11, S_id). The driver does NOT mkdir the obdir.

Chain interface: com_z is re-committed from the input score file; the
obligation is closed in the pipeline by the orchestrator's byte-equality
**com_z == scores_rescale stage-2 com_Xr.bin**. Downstream,
**com_P == values-matmul com_X.bin** (byte-identical). [P-int32-out.bin] is
the UNPADDED B×NCOL int32 chain file at scale 2¹⁶ (real scale: 4,194,304 B);
the verifier never reads it — P's integer interpretation flows through
com_P. Sanity at real scale: P ∈ [0, 65536], masked entries exactly 0,
row 0 = {65536, 0, …}, row sums within ±21 of 65536.

Exp-table registration rule: run `gen_softmax_exp_table.py` ONCE
(np.rint(65536·exp(v/65536)) for v ∈ [−2¹⁹, 2¹⁹), int32) and register
`softmax-exp-table.bin` by **sha256** in public.json next to the swiglu
table. The sha256 registration, not regeneration, is the source of truth —
the C++ driver never generates the real table (its selftest fallback is
flagged NON-AUTHORITATIVE).

Real-scale numbers (per head): prove 10.18 s, verify 11.60 s,
proof+commitments 2,124,988 B (~2.03 MB; serialized G1Jacobian_t is 144 B, so
commitments dominate — com_L = com_A_L = 576 KB each). Audit reproduced
10.29 s / 11.81 s / identical bytes. 24 heads ⟹ ≈ 4.1 min prove / 4.6 min
verify for the softmax obligations proper (score rescales and the two
matmuls accounted separately).

**Orchestrator obligations pinned by the audit:**
- **(MINOR-5) The §4.7 chain byte-equalities MUST be enforced by the
  orchestrator**: com_z == scores_rescale stage-2 com_Xr and com_P ==
  values-matmul com_X, byte-identical. com_z is only used homomorphically
  (never opened standalone), so a standalone ACCEPT binds (z_, E, S, P)
  internally consistently but does NOT pin z_ to the upstream score tensor —
  the chain check is the defense, by design.
- **(MINOR-4) Missing-proof-file behavior**: a missing/short proof file makes
  verify() throw (uncaught) rather than print REJECT — the process exits
  nonzero, so it is fail-closed (never a false ACCEPT); the orchestrator
  must treat ANY nonzero exit as reject, not parse for the REJECT line.

## 19. zkob_rowmax — row-max (argmax-binding) obligation driver: SELFTEST ALL PASS (170/170) + independent audit (SOUND)

Covers STAGE3_FAITHFUL_DESIGN §2 Part A — the row-max obligation, in both
masking regimes. Proves, **with zero advice freedom on the value**, that a
committed per-row scalar mx[i] equals the max over the *allowed* set of a
committed B×NCOL int32 grid z:

- **causal**: AL = lower triangle, B == NCOL, V = 0 — per-head attention
  row-max (com_mx feeds softmax8's mx input).
- **vpad**: AL = first V columns, 0 < V ≤ NCOL — the logit-binding statement,
  zero-padding z columns V→NCOL before commitment (the |z| < 2²⁵ envelope
  guard); when `tstar-int32.bin` is supplied, **T-BIND** additionally proves
  S[i, t*[i]] = 1, i.e. each served token t*[i] is an actual row maximizer.

The only protocol freedom is the **selector tie channel** (§2.4): all witness
tensors are deterministic functions of (z, mx, S, AL) — the limbs are the
unique base-LEN_R digits of Df, m_L is logUp-forced at random β, A_L = 1/(L+β)
elementwise — *except* which tied argmax position carries the 1 (and a tied
t*). The honest prover uses the canonical **lowest-index** witness
(strictly-greater scan on an ascending j loop, np.argmax convention;
`host_tstar` is the same scan). Visible only in com_S/its openings; the
per-row duty of this channel is the measured-gate quantity the orchestrator
reports (tie-count reporting, below). Passed an independent soundness audit
(ROWMAX_REVIEW.md, 2026-06-11, **VERDICT: SOUND — 0 CRITICAL / 0 MAJOR**; FS
schedules absorb-for-absorb identical; every disk value anchored; the
memory-fix byte-identity reproduced from preserved artifacts), plus a
hardening round dispositioning all 9 MINORs (ROWMAX_REPORT.md §8). One FS
transcript, 12 (causal) / 14 (vpad+t*) IPA openings, seven sub-obligations:

1. **LIMB** — logUp range lookup of the NPL·B·NCOL limb tensor L (base-LEN_R
   digits of the dominance gap Df) vs tLookupRange(0, LEN_R), com_A_L
   committed AFTER β_L; the verbatim-softmax logUp terminal, table rebuilt
   verifier-side. Sole enforcer of Df ∈ [0, LEN_Rᴺᴾᴸ) as integers.
2. **BIN** — S binary on the whole padded grid; claim 0; the verifier
   **REQUIRES `U_f2 == S_f2 − 1`** (load-bearing — this binds the
   never-committed U = S−𝟙 to com_S). Kills the fractional-selector forgery
   (evil=3) that passes SUM/MASK/ATT/DOM by construction.
3. **SUM** (one-hot over allowed, claim 1) / 4. **MASK** (nothing outside
   allowed, claim 0): weights `bcast(eq(u_s))⊙AL` / `bcast(eq(u_m))⊙(𝟙−AL)`
   rebuilt verifier-side from its own AL; `U_f2 == 1` forced. In vpad, MASK's
   support is exactly the pad columns, so S's pads are forced to 0.
5. **ATT** (attainment) — ev_mx absorbed at u_a; pure-column-broadcast eq_acc
   shortcut (row-bits-only accumulator) on the verifier; all three openings
   (S, z at pt_a; ev_mx vs com_mx at u_a). Forces ⟨S-row, z-row⟩ = mx rowwise.
6. **DOM** — the c1/c2 bracket: each forced by its round chain to the
   verifier's own eq(u_r)⊙AL weight fold × an anchored terminal (z̃(pt_c1) vs
   com_z; the **never-committed mx broadcast pinned to com_mx at pt_c2's
   row-bit suffix** pt_c2[logC..logD)); NPL plane openings of L at (u_r‖plane);
   plain-field identity `c2 − c1 == v0 [+ LEN_R·v1]`. With LIMB ⟹ mx ≥ every
   allowed z entry.
7. **T-BIND** (vpad+t* only) — W_t gathered from the verifier's OWN t*, every
   t*[i] range-forced into [0, V); claim 1 ⟹ S[i, t*[i]] = 1.

**Constant-claim discipline** (the found-and-fixed evil=3 gap, audited complete
at ALL FOUR instances): BIN 0, SUM 1, MASK 0, T-BIND 1 are protocol constants —
imposed by the verifier at round 0 AND required equal to the serialized claim_H
(both halves; the imposition is load-bearing per evil=3, the equality is
belt-and-suspenders, negatively exercised since the hardening round). ev_mx,
c1, c2 are data-dependent, absorbed per §2.6, each independently anchored.

**The ONE new kernel (§2.8)** — `k_pp_expand` (zkob_rowmax.cu:71–78), Fr-only,
driver-local, generalizing k_eq_expand's (1−c, c) doubling to arbitrary (a, b)
pairs; powers `fast_me_weights`/`fast_s_vector` so the gen-32768 IPAs avoid the
me_weights 1-thread host-loop hot spot (ROPE §9.1). No other new kernels; no G1
kernels (the -dlto rule). `crosscheck_fast_helpers` runs in EVERY honest prove
(toy + gen-1024 real), element-exact vs the slow header path, STOP on mismatch.

FS schedule (seed = run_seed:obligation_id): absorb B, NCOL, MODE, V, LEN_R,
NPL [, "TSTAR" (raw int32) in vpad+t*] → com_z, com_S, com_mx, com_L, com_m_L
→ β_L → com_A_L → α_L, u_L → LIMB rounds + terminals + 3 openings → u_bin →
BIN rounds + opening (U_f2 == S_f2−1) → u_s → SUM → u_m → MASK → u_a → ev_mx →
ATT rounds + 3 openings → u_r → c1 → c1 rounds + ipa_z_c1; c2 → c2 rounds +
ipa_mx_c2 (suffix); v0[,v1] + plane openings → [u_t → ev_mx-style; T-BIND
rounds + ipa_S_tbind] → verifier-only U_f2 checks + the `c2 − c1 == v0 [+
LEN_R·v1]` identity. Audit confirmed prove/verify absorb-for-absorb identical,
matching §2.6 label-for-label including the TSTAR preamble.

CLI (MODE = literal `causal`/`vpad`):
```
zkob_rowmax prove  <obdir> <seed> <z-int32.bin> <B> <NCOL> <MODE> <V> <LEN_R> <NPL>
                   <gen_grid.bin> <gen_mx.bin> <q.bin> [mx-int32-out.bin] [tstar-int32.bin]
zkob_rowmax verify <obdir> <seed> <B> <NCOL> <MODE> <V> <LEN_R> <NPL>
                   <gen_grid.bin> <gen_mx.bin> <q.bin> [tstar-int32.bin]
zkob_rowmax selftest
```
Layout: B/NCOL/LEN_R pow2, NPL ∈ {1,2}, NCOL ≤ LEN_R ≤ NPL·D with LEN_R | NPL·D;
gen_grid size NCOL, gen_mx size B; the driver does NOT mkdir the obdir.
`[mx-int32-out.bin]` is the UNPADDED B int32 chain file (causal: softmax8's mx
input, same scale as z; `-` / omit to skip). `[tstar-int32.bin]` is taken by
BOTH prove and verify (the verifier loads its own registered, hash-pinned copy
— §3.4); supplying it in causal mode is a guarded throw.

Files in <obdir> — **27 causal / 30 vpad+t*** (design §2.7's 26/29 headline is
a counting slip; its own file LIST and this driver both yield 27/30 — see
report §8 MINOR-1; all byte-tamper-tested): dims.bin; com_z, com_S, com_mx,
com_L, com_m_L, com_A_L (.bin); lookup_L.bin; hp_bin, hp_sum, hp_mask, hp_att
(claim_H = ev_mx), hp_c1, hp_c2 [, hp_tbind] .bin; lvals.bin (NPL Fr_t: v0
[, v1]); ipa_A_L, ipa_L_lk, ipa_m_L, ipa_S_bin, ipa_S_sum, ipa_S_mask,
ipa_S_att, ipa_z_att, ipa_mx_att, ipa_z_c1, ipa_mx_c2, ipa_L_p0 [, ipa_L_p1]
[, ipa_S_tbind] .bin.

Chain edges (§2.7; orchestrator byte-equalities ≡):
- **causal** (per layer l, head hh — Part C §4.3):
  `RM1.hh: rescale10.h{hh}/com_Xr.bin ≡ rowmax.h{hh}/com_z.bin`;
  `RM2.hh: rowmax.h{hh}/com_mx.bin ≡ softmax8.h{hh}/com_mx.bin`.
- **vpad** (Part B logit-binding §3.4):
  `L1: lm_head.rescaling/com_Xr.bin ≡ statement.logit_binding/rowmax/com_z.bin`
  (lm_head commits the zero-padded 1024×32768 grid with gen32768; the rowmax
  prover pads identically — byte-identity for honest runs, §12 mechanism);
  `L2: registration hash over tstar.i32.bin` (t* is a registered artifact,
  served_tokens in public.json; public.json hash count 26 → 31).

Real-scale numbers (gen1024 grid / gen32768 vpad):

| case | prove | verify | proof+commitments | GPU peak |
|---|---|---|---|---|
| causal 1024×1024, LEN_R=2²⁰, NPL=1 | 5.28 s | 1.92 s | 791,140 B | 0.70 GiB |
| vpad 1024×32768, V=32000, NPL=2, +t* | 44.19 s | 3.99 s | 974,264 B | **10.99 GiB** |

The vpad peak is **WITHIN the ~18 GiB §6.3 gate** (≥ 2.2× headroom on a 24 GiB
RTX 4090; the §6.3 column-block fallback is NOT needed). It required a
memory-discipline pass (driver-local, no header edits, no protocol change,
byte-identity reproduced): the first implementation peaked at 18.21–18.24 GiB
in the shared header `fs_hadamard`/`fs_phase1` frames; an iterative
`lean_hadamard` clone (same kernels/labels/challenges/terminals, only buffer
lifetime differs) + raw `k_pp_expand` eq builds + commit-then-free/re-upload of
z, S, L brought it to 10.5–11 GiB (ROWMAX_REPORT.md §4; the 10.99 full-selftest
vs 10.48 standalone-profile peak is the documented fragmentation delta). The
residual is the header `fs_phase1` frame-0 burst, irreducible without editing
`zkob_lookup.cuh`/`tlookup.cu` (out of scope, unnecessary at this headroom).

Selftest (after the hardening round): **170 PASS / 0 FAIL, exit 0**
(`/root/zkllm/rowmax_selftest_hardened.log`; up from the audit's 160 — the +10
are the four audit coverage findings). Four toy shapes (causal 8×8 n1=1,
vpad 8×16 V=11 NPL=2 +t* n1=3, causal 16×16 n1=2, vpad 4×8 V=NCOL=8 MASK
weight≡0 n1=1); EIGHT evil modes each rejected by exactly the named check
(evil=3 → "BIN round 0", the certifying fractional-selector forgery; evil=2 →
"DOM bracket identity" at the very last check; etc.); byte tampers on every
proof file + restored ACCEPT; all 15 §2.1 guard throws; the chunked-vs-unchunked
commit byte test; both real-scale runs.

**Orchestrator obligations pinned by the audit:**
- **(MINOR-7) Standalone-ACCEPT caveat (inherited, by design).** A standalone
  rowmax ACCEPT binds (z, S, mx, L, t*) **internally** only; it does NOT pin z
  to the upstream logits/scores nor mx downstream. The orchestrator MUST
  enforce the §2.7 byte-edges — **RM1/RM2** (causal) and **L1** (vpad) — and
  the t* registration hash (**L2**, §3.4). The §2.1 field-wrap reliance (X4
  reads Df field values as small integers; mx inherits the chained z's range)
  is part of the same scoped assumption and is documented in the file header.
  Same posture as softmax's com_z (§15 MINOR-5) and rope MINOR-6.
- **Tie-count reporting (§2.4 ii–iii).** The selector tie channel — which tied
  argmax position carries the 1, and tied t* — is the only protocol freedom
  and is **out of this driver's scope by design**. The orchestrator's tie
  measurement hook reports the per-row duty (the measured-gate quantity); the
  driver emits the canonical lowest-index witness deterministically.
- **(MINOR-8) Fail-closed on missing/short files** (`open_or_die` etc. throw,
  nonzero exit, never false ACCEPT): treat ANY nonzero exit as reject.
