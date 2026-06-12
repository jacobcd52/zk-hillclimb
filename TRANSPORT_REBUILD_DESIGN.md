# TRANSPORT_REBUILD_DESIGN — claim accumulation + one batched opening, and the weight-privacy endgame

Status: DESIGN, 2026-06-11. No code exists yet. Implements BACKEND_DECISION §2's decision
(KEEP ours, rebuild the proof-transport layer along DeepProve's architectural lines) and
§3's three next steps. Inputs read for this design: BACKEND_DECISION.md, vrf_common.cuh,
zkob_lookup.cuh, fs_transcript.hpp, zkob_fc.cu (representative driver, full read),
ORCHESTRATOR_DESIGN.md, PHASE0_NOTES §8–§21, STAGE3_FAITHFUL_DESIGN, ROPE_ATTENTION_DESIGN,
and the measured faithful-arch-v1 run at `/root/zkorch/stage3v2-fa/` (prove_manifest.json,
transcript.json, the 3,664 proof files). Hard rules inherited: what each obligation PROVES
does not change; byte-equality chaining, homomorphic links, the registered integerized
model, and the zero-advice closures are invariants; every FS-tail change triggers the
PHASE0 §13 re-validation rule (every selftest + both audit walkthroughs re-run).

Honesty header — measurement corrections made by this document:

- **Verify is 999.5 s, not 1999 s.** transcript.json's `total_verify_wall_s` = 999.51
  (sum of its 234 per-driver entries = 999.2 s; the walk is strictly serialized).
  transcript_restored.json = 998.25 s, transcript_tamper_commx.json = 995.8 s.
  999.51 + 998.25 ≈ 1998 — BACKEND_DECISION's "1999 s" appears to have summed TWO
  selftest verify walks. All speedup ratios below use the correct 999.5 s baseline
  (DeepProve verify advantage is therefore ≈430×, not 860×; the decision is unaffected).
- **Proof bytes are 96.8 % commitments, not openings.** 175.6 MB total = 169.9 MB
  com_*.bin (1,242 files, 1,179,868 G1 Jacobian points at 144 B) + 4.28 MB ipa_*.bin
  (1,535 files) + 1.4 MB sumcheck/lookup/aux. Killing the 1,535 inline IPAs wins verify
  TIME but only ~4 MB of SIZE. The ≤30 MB target is a commitment-payload problem and is
  treated separately (§2.6) so it cannot silently gate the verify rebuild.
- **~80–150 s of verify is process overhead, not protocol.** 234 verify subprocesses,
  each with CUDA-context init + generator-file loads (smallest entries ~0.35 s; rowmax
  causal measured 1.92 s standalone in PHASE0 §19 but 4.4–4.6 s in the walk). Batching
  openings alone cannot reach 10–60 s; §2.7 adds single-process verification as an
  explicit, separately-validatable part of the rebuild.

---

## 0. One-paragraph summary

Every driver keeps its sumchecks, lookups, commitments, homomorphic links, and chain
edges bit-for-bit; the ONLY change is the tail: instead of discharging 12–72 inline
FS-IPA openings per run (1,535 across faithful-arch-v1, ≈90 % of the 999.5 s verify),
each driver EMITS its terminal claims `(commitment-id, point, eval, domain)` — all of
which it already absorbs into its transcript today — into a per-run accumulator. A new
`zkob_batchopen` driver then runs ONE batch-evaluation sumcheck (Spartan/Jolt-style
RLC-of-eq reduction, the same architecture as DeepProve's claim aggregation) that
reduces all claims to a single random point r, followed by ONE homomorphically-combined
IPA per generator domain (4 domains: gen64/gen1024/gen4096/gen32768). Soundness is
Schwartz–Zippel over the RLC plus the already-audited Pedersen/IPA binding; no new
assumptions, no trusted setup, no curve change, zero new prover advice. Expected verify
35–95 s (target window 10–60 s reachable with the single-process verifier; the final
IPA is NOT the post-rebuild bottleneck, which is why HyperKZG is documented but NOT
recommended). Proof size drops to ~171 MB protocol-only, ~44 MB with canonical-affine +
content-dedupe commitment storage (S1), ~6–12 MB with row-chunked commitments (S2,
optional follow-on). The weight-privacy endgame (§4) — hiding Pedersen rows + masked
weight-claim sub-batch + ZK final opening — is designed now and the accumulator format
carries the variant tag from day one, so the rebuild cannot preclude it.

---

## 1. CURRENT TRANSPORT, PRECISELY

### 1.1 The opening primitive

All commitments are row-wise Pedersen over BLS12-381: a padded tensor T (R_pad × G,
G = generator-domain size, flat index = row·G + col) commits as R_pad row points
`com[r] = Σ_c T[r,c]·g[c]` (deterministic MSM; no blinding anywhere). An opening of the
tensor's MLE at point `u_pt = u_col[0..logG) ‖ u_row` is discharged by
`open_verify` (zkob_lookup.cuh:127-138):

1. `C0 = fold_chain(com, u_row)` — fold the R_pad row points down to one point with the
   k_com_me pair-fold (≈ R_pad G1 muls, vrf_common.cuh:144-163);
2. `P0 = C0 + eval·Q`;
3. `ipa_verify(gen, G, Q, P0, u_col, proof, tr)` — the §10-pinned Fiat–Shamir IPA
   (vrf_common.cuh:338-371): logG rounds, each absorbing L,R and deriving x_r; per-round
   b-vector fold (host h_scalar loops); final s-vector build (G·logG host h_scalar
   calls) + one G-point GPU MSM (`dev_msm`, naive scale-and-tree-reduce) + the final
   point equation `P_L == a_f·g_f + (a_f·b_f)Q`.

`zkob_fc` calls `ipa_verify` directly with the same structure (zkob_fc.cu:271-300);
everything else goes through `open_prove`/`open_verify`. The IPA proof on disk is
`L[], R[], a_final` (~96·logG + 32 B; measured average ipa file 2,788 B).

Critical property reused by the rebuild: **every opened eval is already absorbed into
the driver transcript before its IPA starts** (fc absorbs `claim_X, claim_W` at
zkob_fc.cu:197/269; lookups absorb `A_f, S_f, m_f` per the §11 schedule; rope absorbs
`ev`, headslice absorbs `eQ/eK/eV`, etc.). The opening tail consumes FS-bound claims;
it does not create them. That is exactly the seam we cut at.

### 1.2 Where the openings are: 1,535 inline IPAs

Counted directly: `find proofs -name 'ipa_*.bin' | wc -l` = **1,535** in the
faithful-arch-v1 run (65/65 ACCEPT). Per-driver dynamic counts (per instance; from the
design docs and confirmed against call sites):

| driver | inline IPAs / instance | instances in faithful-arch-v1 | opened tensors (domain) |
|---|--:|--:|---|
| zkob_fc | 3 | 15 full + 48 per-head | W (gen_out), X (gen_in), Y (gen_out); domains 64–32768 |
| zkob_rescale | 3 | 15 + 58 sub-runs | A, rem, m (lookup terminals; gen up to 32768 for lm_head) |
| zkob_glu | 6 | 2 | lookup terminals + hadamard terminals (gen4096/gen1024) |
| zkob_rmsnorm | 17 | 5 | AL,L,mL,X_ss,M,P1,P2,R_q1,M_q1,R_q2,M_q2,R_o,g,W,Y,Wr,X_h (gen1024) |
| zkob_skip | 0 | 4 | — (pure homomorphic point check) |
| zkob_rope | 3 | 4 | T (twice, one at flipped point), Y64 (gen1024) |
| zkob_headslice | 72 | 2 | 36 paired openings: com_Q/K/V (gen1024) vs 36 slice coms (gen64/gen1024) |
| zkob_rowmax | 12 causal / 14 vpad | 48 + 1 | z, S(×6), mx(×2), L, m_L, T-claims (gen1024 / gen32768) |
| zkob_softmax8 | 19 | 48 | z, mx, Dm, E, P, S, L-planes, m_E8 terminals (gen1024) |
| zkob_headmerge | 13 | 2 | 12 out_h (gen64) + O2 (gen1024) |
| zkob_softmax (stage-2) | 16 | 0 in this run | superseded by rowmax+softmax8 in faithful-arch; driver still maintained |

Generator domains in registration/: gen64 (9 KB), gen1024 (144 KB), gen4096 (576 KB),
gen32768 (4.6 MB), plus Q (q.bin).

### 1.3 Where the 999.5 s verify goes

There is no instrumented per-phase split inside a driver verify (flagged: Stage A adds
one). The accounting below combines the measured per-entry timings in transcript.json
with the cost model of §1.1; treat the percentages as ±10 points until profiled.

Measured anchors (transcript.json, RTX 4090):

| entry class | total | n | avg | openings/run |
|---|--:|--:|--:|--:|
| softmax8 heads | 619 s | 48 | 12.9 s | 19 |
| rescale (incl. lm_head 42.8 s @gen32768) | 102.1 s | 15 | 6.8 s | 3 |
| fc full-size | 61.1 s | 15 | 4.1 s | 3 |
| headslice (`slice`) | 50.5 s | 2 | 25.3 s | 72 |
| rmsnorm | 36.9 s | 5 | 7.4 s | 17 |
| rowmax heads + vpad | 113 s | 49 | 2.3/4.4 s | 12/14 |
| everything else (per-head fc/rescale, rope, merge, glu, skips) | ~216 s | 100 | ~2 s | 3–13 |

Cost model per opening (basis: ROPE §9.1's measured ~60 µs/element host-loop figure,
the gen32768 measurements — ≈130 s verify-side per gen-32768 IPA pre-fast-helpers — and
the per-driver averages above): the dominant per-opening costs are the HOST-side scalar
loops (`me_weights`: G·logG 1-thread h_scalar round-trips; the per-round b-fold; the
s-vector build) plus the O(G) `dev_msm` and the O(rows) `fold_chain`. At gen1024 an
opening lands ≈0.3–0.6 s, at gen4096 ≈0.7–1.5 s, at gen32768 ≈13 s. Multiplying by the
opening census reproduces the measured totals to within ~15 %:

- **inline IPA openings: ~850–900 s ≈ 85–90 %** (1,535 openings),
- **subprocess overhead (CUDA init + gen loads + tensor absorbs): ~80–150 s** (234
  processes; visible as the gap between PHASE0 standalone timings and walk timings,
  e.g. rowmax causal 1.92 s → 4.5 s),
- **sumcheck/lookup round checks + terminal identities + chain edges + registration
  hashing: ~30–60 s** (h_scalar host round-trips per round; 100+ rounds in softmax8).

Conclusion: the rebuild must remove (a) AND (b); (c) is already cheap and is untouched.

### 1.4 Where the 175.6 MB goes

| kind | files | bytes | share |
|---|--:|--:|--:|
| com_*.bin (Jacobian, 144 B/point) | 1,242 | 169.9 MB | 96.8 % |
| ipa_*.bin | 1,535 | 4.28 MB | 2.4 % |
| sumcheck/lookup/hp/ev/dims | 887 | 1.4 MB | 0.8 % |

Of the 169.9 MB of commitments: 38.5 MB are byte-duplicate files (285 of 1,242 — the
chain-edge pairs com_Y ≡ next com_X etc. are stored twice by design); distinct payload
131.4 MB = 1.18 M G1 points. Size distribution: 951 files × 1024 rows (the per-head
softmax8/rowmax tensors dominate by count), 48 files × 4096 rows (softmax8 limb planes,
28.3 MB). The opening rebuild (§2.1–2.5) and the size plan (§2.6) are therefore
deliberately separable.

---

## 2. THE BATCHED-OPENING PROTOCOL

### 2.1 Claim objects and the accumulator

A **claim** is the tuple a driver today passes to `open_verify`/`ipa_verify`:

```
Claim {
  id        : string   // canonical: "<manifest_id>[/<sub>]:<tensor>"  e.g. "layer0.mlp.gate_proj.matmul:W"
  comref    : string   // path of the commitment file it opens against (registered/ or proofs/)
  domain    : u32      // generator domain size G ∈ {64, 1024, 4096, 32768}
  n_rows    : u32      // R_pad (row count of the commitment vector)
  point     : Fr[]     // u_col[0..logG) ‖ u_row[0..logR)   (existing layout, col bits first)
  eval      : EvalVar  // v0: Plain(Fr) | v1: Committed(G1)   ← privacy forward-compat, §4.4
}
```

Prover side: each driver, at the exact program point where it currently calls
`ipa_prove`/`open_prove`, instead appends the Claim to `<obdir>/claims.bin` and does
nothing else. **No absorb schedule changes before that point**: the eval was already
absorbed (§1.1), the point is already FS-derived. The driver transcript simply ends
earlier.

Verifier side (per driver): the driver verify runs all its round checks, terminal
identities (e.g. fc's `cur == claim_X·claim_W`, rope's `c1+c2 == ev`, softmax8's I1/I2,
rmsnorm's U_f2 checks) and link checks exactly as today, then — instead of running
`ipa_verify` — RE-DERIVES each expected Claim itself (it has the point from its own
transcript and the eval from the proof files it already read) and writes them to its
own output. The orchestrator byte-compares the prover's accumulated claim list against
the verifier-recomputed list; any mismatch (missing, extra, reordered, altered) is a
REJECT at a new check `opening_batch.claims_match`. **The accumulator is therefore
verifier-recomputed content, never trusted** — same discipline as dims.bin.

A driver verify that passes its local checks now returns **ACCEPT-conditional**; only
the orchestrator emits ACCEPT, after `opening_batch` passes. (Selftest expectation
lines change accordingly — §5.)

Canonical claim order = manifest walk order, sub-runs in their documented sub-seed
order, claims within a run in the order the driver's schedule emits them. Pinned in
the accumulator file; any deviation is a claims_match REJECT.

### 2.2 The reduction: batch-evaluation sumcheck + per-domain RLC opening

Notation: n claims C_1..C_n; claim i opens tensor j = tensor(i) (m distinct tensors —
claims may share a tensor: rope opens T twice, rowmax opens S six times). Tensor j has
vars_j = logG_j + logR_j variables; its MLE P_j is zero-extended to the global
m_max = max vars (25: lm_head W and rowmax-vpad tensors, 1024×32768) by
`P̂_j(x_low, x_high) = P_j(x_low)·Π_h (1 − x_high_h)`. Claim point embeds as
`û_i = point_i ‖ 0^{m_max − vars_i}`; note `P̂_j(û_i) = P_j(point_i)`, so claim i is
equivalent to `P̂_{tensor(i)}(û_i) = v_i` on the common domain.

**Phase G (one per run), transcript seeded `"<run_seed>:opening_batch"`:**

```
G0  absorb "n_claims" n; for each claim i in canonical order:
        absorb "claim" ( id_i ‖ domain_i ‖ n_rows_i ‖ point_i ‖ eval_i )
    for each distinct tensor j: absorb "comref" ( id_j ‖ sha256(com file bytes) )
    for each driver run d in walk order: absorb "drvstate" (final 32-byte transcript
        state of d's transcript — binds the batch to every per-obligation transcript)
G1  ρ ← fs_challenge_fr; weights ρ_i := ρ^i           (single challenge, powers)
G2  batch-evaluation sumcheck, m_max rounds, over the identity
        Σ_i ρ_i v_i  ==  Σ_{x ∈ {0,1}^m_max}  Σ_j M_j(x)·P̂_j(x),
        M_j(x) := Σ_{i: tensor(i)=j} ρ_i · eq(û_i, x).
    Per round: prover absorbs p(0),p(1),p(2) (degree 2 per variable: M_j and P̂_j are
    each multilinear) → challenge r_t; standard front/back fold of every M_j, P̂_j
    (k_fr_fold orientation). Verifier: round check p(0)+p(1) == cur, cur ← lagrange3.
G3  absorb per-tensor terminal evals: for each distinct tensor j (canonical order):
        absorb "vfin" v'_j        // claimed P_j(r_low_j), r_low_j = r[0..vars_j)
    Verifier checks the sumcheck terminal:
        cur  ==  Σ_j M_j(r)·κ_j·v'_j ,   κ_j := Π_{h ≥ vars_j} (1 − r_h),
    where M_j(r) and κ_j are VERIFIER-computed from (ρ, û_i, r) — no prover input.
G4  ρ' ← fs_challenge_fr; weights ρ'_j := ρ'^j
G5  per generator-domain group g ∈ {64, 1024, 4096, 32768} (ascending):
        C*_g := Σ_{j ∈ g} ρ'_j · κ_j' · fold_chain(com_j, r[logG .. vars_j))
        v*_g := Σ_{j ∈ g} ρ'_j · κ_j' · v'_j          (κ_j' = κ_j; public scalar)
        P0_g := C*_g + v*_g·Q
        ONE ipa_verify(gen_g, G, Q, P0_g, r[0..logG), ipa_g, tr)   // existing §10 IPA
```

Why grouping by domain works with zero new machinery: the flat layout puts **column
bits first**, so every tensor in group g opens at the SAME column point r[0..logG) —
one shared b-vector, one IPA — while each tensor's row-commitment vector folds by its
own prefix r[logG..vars_j) of the SAME r. Pedersen linearity makes C*_g a binding
commitment to the ρ'-combination of the row-folded tensors over the SAME generator
vector gen_g. (This is why grouping by domain is load-bearing: an RLC across different
generator vectors is not a commitment to anything.)

Proof artifacts of phase G: `batch_sumcheck.bin` (3·m_max evals + claim count),
`batch_vfin.bin` (m' tensor evals), `ipa_batch_{64,1024,4096,32768}.bin` — ~50–60 KB
total, replacing 4.28 MB of inline IPAs.

Prover cost of G2 (estimate, flagged for Stage-A measurement): one streaming pass over
all distinct tensors per round with sizes halving — total ≈ 2·Σ_j 2^{vars_j} ≈ 2.4 G
field mults for P̂ folding plus ≈ Σ_i 2^{vars_tensor(i)} ≈ 1.6 G for building/folding
the M_j masks, all flat GPU kernels. Round-0/1 residency exceeds VRAM (Σ sizes ≈ 38 GB
as Fr) → stream per-tensor through host RAM (251 GB; data/ witness is 1.5 GB on disk,
expanded per tensor on GPU). Wall estimate **+60–180 s prove**, against which the
removal of 1,535 `ipa_prove` calls (each logG rounds × 2 MSMs + host loops) claws back
a substantial fraction. Net prove: roughly unchanged, ±20 %. Honest flag: UNMEASURED.

### 2.3 Soundness argument (written, to be audited like §10's IPA)

Setting: random-oracle Fiat–Shamir over fs_transcript.hpp's SHA-256 ratchet; challenge
space |C| ≈ 2^254.86 (7 full 32-bit limbs + top limb < 1,944,954,707 — the random_vec
distribution, < r). Pedersen vector commitments over BLS12-381 G1, binding under DLOG.
All claims below are about a PPT prover P* that outputs accepting artifacts.

**Lemma 1 (driver-local claim binding — unchanged).** For each driver, the tuple
(point_i, v_i) of every emitted claim is bound by that driver's transcript: the point
is squeezed from the transcript after the messages it binds, and the eval is absorbed
before the driver tail ends (§1.1). The existing per-driver soundness arguments
(PHASE0 §8–§21 audits) prove: if the driver's relation does not hold for the committed
tensors, then either some round check / terminal identity fails, or at least one
emitted claim is FALSE, i.e. v_i ≠ P̃_{tensor(i)}(point_i) for the tensor actually
bound by com file bytes. (Today "false claim" is caught by the inline IPA; the rebuild
moves exactly that catch into Lemmas 3–5. Nothing upstream of the tail changed, so the
audited arguments transfer verbatim.)

**Lemma 2 (claim transfer).** The accumulator is recomputed by the verifier from its
own per-driver transcripts and compared byte-exactly (§2.1), and the batch transcript
absorbs every claim, every com-file hash, and every driver's final transcript state
BEFORE ρ is squeezed (G0/G1). Hence P* has zero freedom in the batch statement: the
multiset of (comref, point, eval) entering G2 is a deterministic function of artifacts
already FS-bound to public.json (run_seed seeds every driver transcript). Omission,
addition, reordering, or cross-run replay of a claim is caught by the byte-compare
(or, if the verifier-side list itself were forged, by the driver transcript divergence
that Lemma 1 already covers).

**Lemma 3 (RLC batching soundness).** Suppose some claim is false: δ_i :=
v_i − P̂_{tensor(i)}(û_i) ≠ 0 for some i. The polynomial g(Z) = Σ_i δ_i Z^i is nonzero
of degree ≤ n = 1,535. ρ is squeezed after all δ_i are fixed (G0 ≺ G1), so by
Schwartz–Zippel Pr_ρ[g(ρ) = 0] ≤ n/|C| ≈ 2^{-243.6}. Except with that probability, the
batch sumcheck starts from a FALSE sum: Σ ρ_i v_i ≠ Σ_x Σ_j M_j(x)P̂_j(x).

**Lemma 4 (batch sumcheck soundness).** Standard sumcheck: each round polynomial has
degree ≤ 2, challenges squeezed after the round message (the §10 discipline), so a
false sum survives to the terminal with probability ≤ 2·m_max/|C| ≤ 50/|C|. The
terminal check (G3) uses M_j(r) and κ_j computed by the VERIFIER; the only prover
input is the vector v'_j. Therefore, except with the union-bound probability, the
prover is forced into Σ_j M_j(r)κ_j v'_j == (true) Σ_j M_j(r)κ_j P_j(r_low_j) with at
least one v'_j ≠ P_j(r_low_j) — OR all v'_j are correct, in which case the original
false claim has been reduced to a false v'_j... formally: either some v'_j is false,
or the G3 identity fails. (Both branches continue below or reject.)

**Lemma 5 (same-point RLC + IPA binding).** Fix the group g containing a false v'_j.
ρ' is squeezed after the v'_j are absorbed (G3 ≺ G4). By Schwartz–Zippel on
Σ_j ρ'_j κ_j (v'_j − P_j(r_low_j)), the combined target v*_g differs from the true
combined evaluation except with probability ≤ m'/|C|. By Pedersen linearity, C*_g is a
binding commitment (under DLOG, same generator vector gen_g) to the tensor
`A_g := Σ_j ρ'_j κ_j · rowfold(P_j, r_rows_j)`, and the audited §10 IPA is a sound
argument for `<A_g, me_weights(r_col)> == v*_g`; a false v*_g therefore makes the IPA
reject except with the IPA's knowledge error (logG rounds, ≤ 2·logG/|C| per the §10
argument, plus DLOG). Note binding is precisely why the RLC cannot be "re-opened" to a
different tensor: a P* that opens C*_g to anything other than A_g yields two distinct
Pedersen openings of the same point over gen_g — a DLOG break.

**Composition.** Union bound over Lemmas 3–5 and 4 groups: total added soundness error
≤ (1,535 + 50 + 957 + Σ_g 2·logG_g)/|C| + ε_DLOG < 2^{-240}. The per-driver errors are
unchanged. **Zero new advice:** every prover message in phase G (round evals, v'_j) is
a deterministic function of the witness and prior transcript — same zero-freedom class
as the existing sumchecks; the capacity layer gains no new channel (and the claim list
itself is verifier-recomputed, so it carries 0 bits of prover choice).

Convention risk (flagged): the embedding/orientation details — col-bits-first, MSB
fold order of k_fr_fold vs LSB-first me_weights, κ ordering — are exactly the class of
bit-order bugs §8/§10 caught at toy scale. Stage A pins ALL of them in a
`vrf_toy_batchopen.cu` against brute-force MLE evaluation before any driver is touched.

### 2.4 What is untouched (the reasons we didn't fork Basefold)

- **Byte-equality chaining**: com_*.bin files, their deterministic MSM production, and
  every `≡` edge in ORCHESTRATOR_DESIGN §4 / the stage-2/3 edge maps are not touched by
  the protocol change. (The optional size plan S1 canonicalizes the FILE FORMAT —
  §2.6 — which preserves the edges but is a format migration; it is severable.)
- **Homomorphic links**: zkob_skip's ⊕ point checks, the rescale affine limb link
  `com_X == sf·com_X̂ + com_rem`, rmsnorm's affine links — all operate on commitments,
  not openings. Unchanged. (And the batch itself now RELIES on the same Pedersen
  linearity — the design is more homomorphic, not less.)
- **Registered integer model & statement**: registration, public.json, run_seed
  derivation, weight export semantics, DiFR floor — untouched. The 15
  `*.commitment_opening` manifest ids stop having standalone IPA transcripts and are
  instead discharged by the presence + verification of the registered-com claim in the
  batch (transcript.json `details` will say `discharged via opening_batch claim <id>`;
  the manifest keeps the ids — BACKEND_DECISION §2 "the manifest keeps the ids; the
  proofs merge").
- **Zero-advice closures**: rowmax/softmax8/rmsnorm advice-freedom arguments are about
  WHAT is proven; unchanged. §2.3 shows the batch adds 0 bits.

### 2.5 PCS decision: batched-IPA (RECOMMENDED) vs HyperKZG

| | (a) keep Pedersen/BLS12-381 + batched-IPA | (b) add HyperKZG (MIT arkworks/jolt lineage) for the batch |
|---|---|---|
| verify cost of final openings | 4 IPAs: s-vector MSMs of 64+1024+4096+32768 ≈ 38 k G1 muls (GPU, well under 1 s) + host loops (≈2–4 s unless lifted to GPU — the §6.2 fast-helpers item) | 1–4 pairings + O(log) G1 — saves ~2–4 s |
| does NOT change | — | the per-claim fold/RLC (1.18 M G1 muls, the real post-batch crypto cost), the driver walks, the com payload |
| proof bytes for openings | ~50–60 KB | ~10–20 KB |
| assumptions / setup | DLOG only, no trusted setup (status quo) | pairing + trusted setup (SRS policy, ceremony provenance) |
| curve | BLS12-381 (all 11 drivers, ppgen, registration unchanged) | BN254 in the mature implementations → either run TWO curves with a cross-curve claim bridge (new, unaudited protocol) or port HyperKZG to BLS12-381 (new code, no upstream test vectors) |
| code provenance | vrf_toy_ipa-pinned, §10-audited, -dlto-probed kernels | foreign Rust crate + FFI into a CUDA codebase with a known LTO miscompiler — every new G1 kernel shape is miscompile bait (vrf_common.cuh header rules) |
| homomorphic links | native | KZG is additively homomorphic too, but only if the WHOLE commitment layer moves — out of scope of "transport only" |

**Recommendation: (a) batched-IPA.** After batching, the final opening is ~3–5 % of the
verify budget; (b) optimizes the wrong term while importing a trusted setup, a second
curve or an unaudited port, and FFI surface into a build environment with a documented
LTO miscompilation problem. Decision gate (same as BACKEND_DECISION §3.2): if Stage B
measurement shows the four batched IPAs + RLC folds > 30 s despite GPU-side eq/s-vector
helpers, revisit (b) behind the SAME accumulator interface — the interface in §2.1/2.2
is PCS-agnostic by construction (G5 is the only PCS-aware step).

### 2.6 Proof-size plan (separable from the protocol rebuild)

- **S0 (protocol-only, Stage A–C default):** proof = 169.9 com + 1.4 aux + 0.06 batch
  ≈ **171 MB**. Verify rebuilt, size barely moved. This is the honest statement that
  opening batching alone does NOT approach 30 MB.
- **S1 (canonical-affine + content-dedupe, recommended in Stage C):** store com files
  as compressed affine (48 B/point — canonical, unlike Jacobian) and store the 285
  byte-duplicate files once (content-addressed, manifest maps both paths to one blob).
  131.4 MB × 48/144 ≈ **43.8 MB + 1.5 MB aux ≈ 45 MB**. Byte-equality edges survive
  (canonical form is still deterministic bytes; arguably stronger — Jacobian equality
  was relying on identical MSM execution order). Cost: every com producer/consumer and
  `absorb_g1_tensor` reads/writes the new format → it changes every transcript's
  absorbed bytes → full §13 re-validation (which Stage C triggers anyway — this is why
  S1 is scheduled INSIDE Stage C, not after). Risk: one new affine-conversion kernel
  (batch inversion) — -dlto probe required.
- **S2 (row-chunking, optional follow-on stage):** commit rows in chunks of 16
  (gens of size 16·G; gen file ≤ 75 MB for lm_head): com payload ÷16 ≈ **3–9 MB**,
  total proof ≈ **6–12 MB**. Changes commitment layout (registration re-issued, every
  fold/IPA split shifts by 4 bits) — a second §13 campaign. Only if ≤30 MB is a HARD
  requirement; S1's ~45 MB misses the 30 MB target and this is stated plainly rather
  than papered over.
- **S3 (KZG flat commitments):** single point per tensor (~60 KB total!) but kills the
  per-row structure the drivers' fold logic and the affine limb links are written
  against, needs 2^25-point SRS (4.8 GB), trusted setup — rejected with §2.5's
  reasoning; recorded for completeness.

Note for §7 comparisons: DeepProve's 10.25 MB excludes its one-time weight-commitment
context (107 s context generation). Our comparable per-proof payload after S1 is the
~45 MB (activation commitments are inherently per-run); after S2, ~6–12 MB.

### 2.7 Expected verify budget after rebuild

| component | today | after | basis |
|---|--:|--:|---|
| 1,535 inline IPAs | ~850–900 s | 0 | removed |
| batch: RLC + folds (1.18 M G1 muls, flat GPU kernels) | — | 1–3 s | 2^20-MSM ≈ 0.3–0.5 s measured (STAGE3 §5.0) |
| batch: 4 IPAs + eq/κ/M_j(r) terms | — | 2–6 s | 38 k G1 muls + 1,535·25 host muls (GPU-lift if needed) |
| batch: sumcheck rounds + absorbs (incl. hashing claim list) | — | < 2 s | 75 evals; SHA-256 over ~0.5 MB |
| driver round checks, terminal identities, links, registration hash | ~30–60 s | ~30–60 s | unchanged |
| subprocess overhead (234 × CUDA init + gen loads) | ~80–150 s | ~5–15 s | **requires the single-process verifier** below |
| **total** | **999.5 s** | **≈ 40–85 s** | mid-estimate ~55 s |

**Single-process verifier (part of the rebuild, Stage C):** a `zkverify_walk` mode that
links the existing per-driver verify routines into one binary (one CUDA context, gens
loaded once), driven by the same verify_walk.py order; per-driver binaries remain for
selftests. This is verification PACKAGING, not protocol — FS schedules identical — but
without it the 10–60 s window is arithmetically unreachable (234 × ~0.4 s floor).
Honest statement: hitting the LOW end (10–20 s) additionally needs the §6.2
fast-helpers (h_scalar/me_weights GPU-side batching) applied to the remaining
round-check host loops; the design treats that as stretch, not plan-of-record.

---

## 3. PER-DRIVER CHANGE LIST

The uniform change, for every driver: **the tail only.** Replace each
`open_prove(...)`/`ipa_prove(...)` call with `emit_claim(...)`; replace each
`open_verify(...)`/`ipa_verify(...)` block with `expect_claim(...)` (derive point,
read eval, append to verifier-side claim list). Everything before the tail —
commit phases, absorb schedules, sumcheck/lookup rounds, terminal-identity checks,
homomorphic links, dims cross-checks — is INVARIANT, byte-for-byte.

| driver | tail change (claims emitted) | invariant (untouched) | notes |
|---|---|---|---|
| zkob_fc | 3: W vs registered com (point u_output‖u_input), X vs com_X (u_input‖u_batch), Y vs com_Y (u_output‖u_batch) | zkip sumcheck, claim/claim_X/claim_W absorbs, terminal product check, com absorbs | W-claim discharges the `.commitment_opening` manifest id via the batch (§2.4); W-claim is the privacy-variant claim in Stage D |
| zkob_rescale | 3 lookup terminals (A, rem-plane, m) | affine limb link `com_X == sf·com_X̂ + com_rem` (point check), logUp phases, β/α schedule | lm_head instance is the gen32768 stress case — Stage B includes it |
| zkob_glu | 6 (lookup + hadamard terminals) | both sumchecks, table absorbs | |
| zkob_rmsnorm | 17 | 7 sub-obligations, all affine links, bracket quartics, U_f2 checks | largest per-driver claim count; good Stage-C canary |
| zkob_skip | 0 — **no change at all** | the ⊕ point check | only driver with no tail |
| zkob_rope | 3 (T@pt1, T@pt2′ flipped, Y64@u) | both hadamards, verifier weight recompute, c1+c2==ev | two claims on the SAME tensor at different points — exercises multi-claim-per-tensor |
| zkob_headslice | 72 (36 pairs: full-tensor point vs slice point) | the head-selector bit construction of the paired points | biggest single-run claim emitter; the pair structure is two ordinary claims whose evals the driver checks equal — equality check stays driver-side, both claims go to the batch |
| zkob_rowmax | 12 causal / 14 vpad | all 7 sub-obligations, verifier-recomputed eq weights, dominance identity, T-BIND | vpad claims are gen32768-domain, 25-var — the m_max drivers |
| zkob_softmax8 | 19 | Dm-binding block, sentinel check, I1/I2, row-sum | 48 instances → 912 claims, the bulk of the batch |
| zkob_headmerge | 13 (12 out_h gen64 + O2) | 12 public-weight hadamards, Σc_h==ev, PERM absorb | gen64 group population |
| zkob_softmax (stage-2) | 16 | all six sub-obligations | not in faithful-arch runs but maintained; must be converted + selftested for stage-2 walk compatibility |
| orchestrator | new obligation id `opening_batch`; claims_match check; ACCEPT gated on batch; prove_walk runs zkob_batchopen last; verify_walk consumes conditional verdicts | registration, edges, walk order, transcript.json format (one added id + per-id reasons) | manifest: `opening_batch` added as covered id; check_transcript count goes 65 → 66 |

**The real cost — the PHASE0 §13 rule, stated explicitly.** Every driver's FS tail
changes (absorbs end earlier; the batch transcript is new). Therefore:

1. **Every per-driver selftest re-runs and must pass** with updated expectation lines
   (honest ACCEPT-conditional + batch ACCEPT; every existing forgery still REJECTED,
   now possibly at a different locus — each relocation must be REVIEWED, not just
   re-recorded: a forgery that used to die in `ipa_verify` must die in the batch or in
   claims_match, and the new locus must be named in the selftest output).
2. **Both independent audit FS walkthroughs re-run**: the per-driver
   prove/verify absorb-for-absorb walkthroughs (the *_REVIEW.md discipline — every
   challenge squeezed only after the message it binds) for all 11 drivers PLUS the new
   zkob_batchopen, and the orchestrator-level walkthrough (run_seed → per-obligation
   seeds → drvstate binding → batch seed).
3. **vrf_common.cuh / zkob_lookup.cuh edits** (emit/expect_claim helpers, any new
   kernels): header-edit rule applies — every dependent driver's selftest re-runs; any
   new G1 kernel shape gets the -dlto miscompile probe (the §11-noted batched
   affine-check kernel ALREADY miscompiled once; assume new shapes are bait until
   probed).

Estimated re-validation effort: 1–2 weeks of §14–§21-style work (BACKEND_DECISION's
estimate, unchanged by this design). This is the dominant cost of the rebuild and it
is scheduled as its own stage (§6 Stage C), not amortized into hope.

---

## 4. WEIGHT-PRIVACY ENDGAME

The thing none of the evaluated stacks built (BACKEND_DECISION §1: DeepProve leaks
weight-MLE evals at FS points, JOLT Atlas hands the verifier the model, zkGPT's
Pedersen has no blinding term). Designed now; implemented as Stage D; the §2.1
accumulator carries the `EvalVar` tag from day one so the transport rebuild cannot
preclude it.

### 4.1 Leakage accounting of the CURRENT scheme (and of S0–S2 without Stage D)

Threat model (THREAT_MODEL_NOTES): weights are the secret; the input activation,
public.json, and all proof artifacts are adversary-visible. Activations stay
non-hiding — they are the public statement and the chaining substrate.

Per zkob_fc proof against weight W (IN×OUT):

- `claim_W = W̃(u_input, u_output)` — one field element, a known-coefficient **linear
  functional of W** (the ME weights are public functions of FS challenges).
- The 3·log(IN) sumcheck round evals p_r(0/1/2): each equals Σ_k X-fold_k·W-fold_k.
  Where the relevant X is PUBLIC (layer-0: input.i32.bin is registered), the X-folds
  are adversary-computable and each round eval is another linear functional of W —
  ≈ 3·log(IN)+1 ≈ **31 linear functionals per proof per weight matrix** (gate/up/down,
  IN=768/3072). For deeper layers X is only committed, so coefficients are not fully
  known; conservatively assume known (chained activations are low-entropy functions of
  public input + weights).
- The inline `ipa_W` adds the a_final and L/R points — further folded combinations
  (the batch rebuild REMOVES these per-weight artifacts; the batched IPA's a_final
  mixes weights with hundreds of activation tensors — strictly less informative, but
  NOT zero, and the batch sumcheck round evals still contain W-functional summands).

Recovery arithmetic (the honest number): gate_proj has 768·3072 ≈ 2.36 M unknowns; at
~31 functionals/proof, full linear-algebra recovery after ≈ **76,000 proofs** on the
same registered weights; partial information from proof #1. The sharpest leak is the
rmsnorm gains: g is 768-dim, the rmsnorm tail opens g and its sub-protocols leak
several functionals per proof → g recoverable after ≈ **a few hundred proofs**. This
is the same leakage CLASS as DeepProve (their eprint avoids claiming ZK; ours should
too until Stage D lands): not invertible from one proof, unboundedly accumulating.

### 4.2 The design: hiding commitments + masked weight sub-batch

Scope: ONLY the registered weight commitments (the 15 `commitment_opening` ids' tensors:
gate/up/down/q/k/v/o_proj per layer, lm_head, the 4+1 rmsnorm gains). Activations,
tables, and all chain commitments stay deterministic non-hiding.

**(D1) Hiding Pedersen rows.** Registration gains one generator H (ppgen extension,
hash-pinned in public.json; H independent of g[] and Q). Registered weight commitments
become `com_W[r] = Σ_c W[r,c]·g[c] + s_r·H`, blinds s_r drawn once at registration
from a sealed CSPRNG seed stored prover-side (`registration/private/blinds.bin`, never
read by the verifier — same authority split as data/). Commitments remain
deterministic-after-registration → public.json hash pinning, byte-equality, and the
"absorb registered com" discipline are unchanged. Homomorphic interplay: weight coms
appear in NO affine link (links are activation-side — checked across ORCHESTRATOR §4 +
stage-2/3 edge maps), so blinds break nothing; row-folds of com_W simply carry a
folded blind `s(u_row)`.

**(D2) Masked weight-touching sumchecks.** The fc zkip sumcheck on (X_fold, W_fold)
and the rmsnorm sub-protocols touching g leak round evals (§4.1). Fix: Libra-style ZK
sumcheck (Xie et al. 2019 / CFS17): prover samples a small mask polynomial
`q(x) = Σ_r q_r(x_r)` (univariate per variable, degree ≤ 2; ~3·L coefficients),
commits to it with a HIDING Pedersen vector commitment, absorbs the commitment,
squeezes λ, and runs the sumcheck on `f + λ·q` instead of f; the verifier checks the
masked rounds and, at the end, the claimed q-terms via one ZK opening of the mask
commitment folded at the round challenges. Round messages become uniformly distributed
(perfectly masked for λ≠0); terminal becomes `claim_X·claim_W + λ·q(r)` with claim_W
now hidden (D3).

**(D3) Committed weight claims in the accumulator.** Weight claims enter with
`EvalVar = Committed(C_v)`, `C_v = v·Q + t·H` (fresh t per proof, prover-private).
The driver terminal check `cur == claim_X·claim_W` becomes a Schnorr-class sigma
proof: claim_X is public, so the verifier forms `claim_X·C_v − (cur − λ·q(r))·Q` and
the prover proves knowledge of an H-discrete-log opening (one Schnorr proof: 1 G1 + 2
Fr). The same trick covers rmsnorm's g-terminal identities (all linear or
product-with-public in the hidden eval).

**(D4) The weight sub-batch.** Weight claims do NOT enter the main (public) batch —
the main batch's sumcheck round evals are sums including P̂_W folds and would re-leak.
Instead a second, small accumulator (19 matmul W-claims + 5 g-claims ≈ 24 claims, all
gen1024/gen4096/gen32768) runs the SAME reduction (§2.2) with two changes: (i) the
batch-eval sumcheck itself is D2-masked; (ii) the per-group final IPA is the ZK
variant — blinds folded alongside (`P0_g = C*_g + V*_g` where V*_g is the homomorphic
RLC of the C_v's; the relation acquires an H-component), and the final round replaces
the plaintext `a_final` reveal with a Schnorr proof of opening of
`P_L − a_f·g_f − (a_f·b_f)Q` ... precisely: the prover sends Pedersen commitments to
a_final and runs the standard Bulletproofs-style ZK final round (3 G1 + 3 Fr) instead
of revealing a_final. The IPA L/R points in the weight sub-batch carry folded blinds
and are uniformly distributed.

**(D5) What this achieves and does NOT.** Achieves: zero weight-functional leakage
from claims, round evals, and openings (statistical ZK for D2/D4 messages; hiding for
D1/D3 under DLOG). Does NOT achieve — flagged honestly: (i) **candidate-confirmation
via deterministic activation commitments**: layer-0's com_Y commits Y = X·W with X
public and com deterministic, so an adversary who GUESSES W exactly (e.g. "is this
stock llama-68m?") can confirm by recomputing com_Y. Hiding against extraction, not
against confirmation of a fully-known candidate. Closing that requires hiding
activations too — a different project (it breaks byte-equality chaining as currently
conceived; recorded as an open question, not promised). (ii) Output logits/served
tokens are the statement — anything inferable from input/output behavior is outside
any proof-layer fix (model-stealing via queries is orthogonal).

### 4.3 Overhead quantification (estimates, Stage-D gate re-measures)

| | prover | verifier | proof bytes |
|---|--:|--:|--:|
| D1 blinds (registration, one-time) | +R_pad G1 muls/weight ≈ +2 s total | 0 | 0 (same file size) |
| D2 masks (19+5 sumchecks) | +O(3L) commit + evals ≈ negligible | +1 ZK open each ≈ +0.1–0.3 s total | +~3 KB/sumcheck ≈ +75 KB |
| D3 sigma terminals (24) | negligible | 24 Schnorr verifies ≈ ms | +~100 B each |
| D4 weight sub-batch | one extra small batch (Σ weight tensor sizes ≈ 33 M elements) ≈ +5–15 s | +3 ZK-IPAs ≈ +2–5 s | +~40 KB |
| **total Stage D** | **≈ +1–2 %** | **≈ +3–6 s** | **≈ +0.2 MB** |

Comparable to Artemis's reported ~1.1–1.2× commitment-consistency overhead — and ours
rides on machinery (hiding Pedersen, sigma protocols, the existing IPA) that needs no
new proof system, vs their halo2 CP-SNARK. Artemis (arXiv:2409.12055) remains the
reference for the alternative formulation (prove commitment-consistency of the witness
polynomial inside the existing proof); we take the masking route because our sumchecks
are already per-round-FS and the masks drop in per driver without restructuring.

### 4.4 Forward-compat obligations on Stages A–C (so the rebuild doesn't preclude D)

1. `EvalVar` tag in the Claim struct and accumulator serialization from day one
   (Stage A); plain batch REJECTS Committed claims until Stage D (explicit check, so
   the format is exercised but the path is closed).
2. TWO accumulators plumbed (weight-tagged claims routed by comref into the weight
   accumulator) from Stage B — in Stages B/C the weight accumulator is discharged by
   the SAME public protocol (evals plain); Stage D swaps its discharge to D4 without
   touching the routing or the drivers again.
3. ppgen/q.bin format gains the H slot in Stage A's registration touch (generated,
   absorbed into public.json hashing, UNUSED until D) — avoids a second registration
   format migration.
4. The G5 grouping keys on (domain, comref-class) so weight groups never homomorphically
   mix with activation groups.

### 4.5 Stage-D AS-BUILT amendment (resolves TRANSPORT_REVIEW F7) — 2026-06-12

F7 flagged that D4's "same reduction with two changes" was one change short: with
Committed evals the batch's initial claim exists only inside commitments, so
Libra-masked rounds have no public running claim to check against. AS BUILT
(zkob_wpriv.cuh; STAGE_D_REPORT.md for the full protocol + arguments), the missing
component is exactly the one the review named, and it REPLACES the Libra masks
entirely:

- **Committed round messages with homomorphic round checks** (weight batch AND the
  fc driver zkip sumcheck): prover sends C_p0/C_p1/C_p2 = p_t·Q + τ_t·H with τ_0, τ_2
  fresh and τ_1 = τ_cur − τ_0; verifier checks C_p0 + C_p1 == C_cur in G1 and folds
  C_cur' = lagrange3(C_p0,C_p1,C_p2)(x) homomorphically (lagrange3 is linear; blind
  tracking is the scalar lagrange3 of the τ's). No mask polynomial q, no mask
  commitment, no extra mask opening: the round messages are perfectly-hiding Pedersen
  commitments, which subsumes D2's goal for these sumchecks.
- **Committed per-tensor terminals**: C_vfin_j = v'_j·Q + t'_j·H, the G3 check is
  Σ_j M_j(r)κ_j·C_vfin_j == C_cur in G1; the prover solves the last nonzero-coefficient
  t'_j so the H-components balance.
- **Driver terminals as sigma proofs** (D3 as designed): fc's `cur == claim_X·claim_W`
  becomes a Schnorr PoK of the H-component of C_cur − claim_X·C_W; rmsnorm's
  `val_W == val_R·val_g` becomes a Schnorr PoK of the H-component of
  val_R·C_val_g − val_W·Q.
- **ZK final opening** (D4's IPA): blinds β_L, β_R on L/R, β' = β + x²β_L + x⁻²β_R,
  and the a_final reveal replaced by a 2-base Schnorr PoK of (a_f, β_f) for
  P_fin = a_f·(g_f + b_f·Q) + β_f·H.

Honest accounting addition to §4.1/§4.2-D5, found during the build: the rmsnorm
gain's privacy is bounded by the PUBLIC chain tensor W = R⊗g — val_g = val_W/val_R is
computable from public claims even with the g claim hidden (and Y = W_∘X exposes g
multiplicatively in the activation statement itself). Stage D closes the proof-LAYER
leakage for g (no val_g byte appears in any artifact; regression-checked); the
activation-statement channel is the same class as D5(i) and needs hiding activations.
For the matmul weights the same class survives only as layer-0 public-X claim
functionals (claim_Y at known points), per §4.1's existing accounting.

---

## 5. THE TEST HARNESS

### 5.1 Invariant battery — everything that exists must still pass

Inventory (what "everything" is, pinned so nothing silently narrows):

- **Per-driver selftests** (small shapes × honest + forgeries + byte-tampers + real
  scale): fc (3 cases × 6 tampers + honest/restored), rescale (incl. the semantic
  wrong-value case), glu (9 tampers + 2 semantic), rmsnorm (48 checks, 8 evil modes),
  rope (4 evil + tampers), headslice (3 evil + 70+ tampers), rowmax (170 checks, 8 evil
  modes × modes causal/vpad/t*), softmax (122 checks, 6 evil), softmax8 (170 checks,
  8 evil incl. sentinel/Dm), headmerge (166 checks, 4 evil, both perm modes), skip.
  ~40 semantic forgeries + ~hundreds of byte-tampers total.
- **Orchestrator selftest.sh** phases (a)–(j): honest walks (stage-2 59 ids,
  faithful-arch 65 ids), commitment tamper, registration-hash tamper, served-token
  tamper, lm_head-hash tamper, restoration re-ACCEPTs, the double-detection case (i)
  (rowmax com_mx: transcript divergence + edge RM2.h05).
- **check_transcript.py** both runs (frozen full manifest + stage manifest), counts
  updated 65 → 66 (`opening_batch`).
- **Both audit FS walkthroughs** (§3 item 2).
- **harness/FORGERIES.md** suite: still PLANNED (forgeries/ dir is empty; activation
  blocked on serialized-proof format finalization). The rebuild FINALIZES the format —
  Stage C includes activating classes B (weight/commitment) and C (proof-object)
  against the new transport; IDs stay stable, suite grows only.

Rule for relocated rejections (restating §3): every existing forgery keeps a named
expected locus; a forgery whose locus moves (e.g. from `REJECT: IPA opening of claim_W`
to `REJECT: opening_batch`) gets its new locus REVIEWED for "is this still the check
that semantically catches it" and recorded in the selftest expectations. A forgery that
starts passing driver-local checks and is only caught by the batch is acceptable ONLY
if the batch rejection is deterministic for that forgery (no probabilistic-miss
hand-waving — at our challenge space, SZ misses are 2^-240-class, fine).

### 5.2 NEW forgery battery — targeting the batched-opening protocol itself

Each case must be rejected by EXACTLY the named check (loci discipline). Implemented in
zkob_batchopen's selftest (toy scale: 3+ shape cases over ≥2 domains, multi-claim
tensors, single-row tensors, max-vars tensors) and re-run at real scale in the
orchestrator selftest.

| id | forgery (prover behavior) | must be rejected by |
|---|---|---|
| BO-1 | inconsistent claim: driver emits eval v+δ but driver-local terminal identities arranged to pass (e.g. rope: bump ev AND c1 compensatingly — passes c1+c2==ev) | batch: G3 terminal or final IPA of the claim's group (the claim is false vs the commitment) |
| BO-2 | omit one claim from claims.bin (and from the batch) | orchestrator `opening_batch.claims_match` byte-compare |
| BO-3 | swap comrefs of two same-shape claims (e.g. two softmax8 heads' com_z) | group IPA (folded C* no longer matches v*); claims_match if the swap is in the list only |
| BO-4 | forge the RLC: derive ρ from a doctored claim list, present honest list to verifier | batch transcript divergence (G0 absorbs the list before ρ) |
| BO-5 | tamper batch sumcheck round eval p(1) | batch round check `p(0)+p(1) != cur` at that round |
| BO-6 | tamper one v'_j in batch_vfin.bin | G3 terminal identity (`Σ M_j κ_j v'_j != cur`) |
| BO-7 | wrong κ/padding: claim emitted with point padded by junk instead of zeros (vars_i lie) | claims_match (verifier recomputes points); if forged consistently in both lists → G3/G5 mismatch (κ is verifier-computed) |
| BO-8 | byte-tampers: claims.bin (each field of one claim), batch_sumcheck.bin @4+32, batch_vfin.bin @-32, each ipa_batch_* @-32 and @8+16, drvstate list | claims_match / round check / terminal / the group IPA respectively — each named |
| BO-9 | cross-run replay: claims.bin + batch artifacts from another run (same model, different input) | batch transcript divergence (seed = run_seed:opening_batch; drvstate absorbs differ) |
| BO-10 | duplicate claim appended (same tuple twice, batch run honestly over n+1) | claims_match (count + order pinned) |
| BO-11 | claim against an unregistered/substituted commitment file (comref hash honest for the substituted file) | the producing driver's transcript divergence (com absorbed there) — pinned to confirm the existing catch still fires with the new tail |
| BO-12 | omit one domain group's IPA / reorder groups | batch verify structural check (groups derived from claim list, ascending; missing file = REJECT) |
| BO-13 (Stage D) | weight claim emitted Plain instead of Committed; mask commitment reused across proofs; sigma replay | routing check / mask-freshness absorb / sigma transcript binding |

Plus the SEMANTIC end-to-end: re-run orchestrator phases (b), (d), (i) — the existing
commitment/registration tampers — and confirm the double-detection property of (i)
survives (edge RM2.h05 still fires even though the transcript-divergence locus moved).

### 5.3 End-to-end before/after

Same llama-68m faithful-arch-v1 configuration (seq 1024, same registration, same input
seed), proven and verified under OLD transport and NEW transport on the same box:

| metric | OLD (measured) | NEW (gate) |
|---|--:|---|
| verdict | ACCEPT 65/65 | ACCEPT 66/66 (65 + opening_batch) |
| DiFR | 0.0156 (unchanged by construction — same integer pipeline) | identical bytes in data/ (spot-check hash) |
| prove wall | 1062 s | ≤ 1.3× (gate; measure) |
| verify wall | 999.5 s | **≤ 100 s hard gate; target ≤ 60 s** |
| proof bytes | 175.6 MB | S0: ~171 MB; with S1: **≤ 50 MB gate** |
| every selftest | ALL PASS | ALL PASS (updated expectations) |
| forgery suites | all REJECT | all REJECT incl. BO-1..12 |

### 5.4 Pass/fail matrix (the campaign's definition of done)

| gate | stage | pass criterion |
|---|---|---|
| T1 toy pin | A | vrf_toy_batchopen: batch reduction == brute-force MLE evals bit-exact, all orientations; -dlto probes clean |
| T2 single-driver | A | zkob_fc claim-mode selftest ALL PASS; fc+batch on real-scale gate_proj: verify(fc)+verify(batch) ≤ 1/20 of old fc verify-with-IPAs (BACKEND_DECISION's ≥20× gate) |
| T3 two-driver chain | B | fc→rescale (incl. gen32768 lm_head pair) one accumulator; BO-1..BO-10 toy battery ALL REJECT |
| T4 full walk | C | §5.3 table, all rows |
| T5 selftest campaign | C | all 11 driver selftests + orchestrator (a)–(j) + BO battery at real scale: ALL PASS/REJECT as specified |
| T6 audits | C | both FS walkthrough audits re-signed for all drivers + batch + orchestrator |
| T7 size | C(S1) | ≤ 50 MB, edges/all checks pass on canonical-affine format |
| T8 privacy | D | D-battery (BO-13 + leakage regression: no plaintext weight-functional in any artifact — grep-style transcript audit + statistical test on masked rounds); overhead within §4.3 ±50 % |

---

## 6. STAGED IMPLEMENTATION PLAN

Ordered so each stage is independently validatable; no stage starts until the previous
stage's gates pass. Estimates assume the §14–§21 working cadence; they are estimates.

- **Stage A — the primitive + one driver (4–6 days).**
  `vrf_toy_batchopen.cu` (pin embedding/orientation/κ against brute force, multi-claim
  tensors, 2 domains, single-row tensors; -dlto probe every new kernel shape);
  Claim/accumulator serialization (with EvalVar tag + H-slot registration touch, §4.4);
  `zkob_batchopen` prove/verify; zkob_fc claim mode behind a flag (old tail kept
  compilable until Stage C flips the default); fc selftest updated; per-phase verify
  instrumentation (closing §1.3's profiling gap). Gates T1, T2.
- **Stage B — the chain + the evil battery (2–3 days).**
  zkob_rescale converted; two-driver accumulator through the orchestrator on the
  validated fc→rescale pair INCLUDING the lm_head gen32768 instances; BO-1..BO-10 at
  toy + pair scale; measure prover batch-sumcheck overhead (the §2.2 unmeasured flag)
  and the verify split. Gate T3 + a go/no-go on the §2.5 PCS contingency.
- **Stage C — everything (8–12 days; the §13 campaign is most of it).**
  Remaining 9 drivers' tails; orchestrator opening_batch id + conditional verdicts;
  single-process zkverify_walk; S1 canonical-affine + content-dedupe (inside this
  campaign, §2.6); ALL selftests + orchestrator phases + BO battery at real scale;
  both audit walkthroughs; the §5.3 before/after measurement; activate harness
  FORGERIES classes B/C against the finalized format. Gates T4–T7.
- **Stage D — weight privacy (8–12 days).**
  D1 blinded registration (+H), D2 masks on fc/rmsnorm, D3 sigma terminals, D4 ZK
  weight sub-batch; scoped re-validation: fc + rmsnorm selftests, batch selftest,
  orchestrator, audits for the touched schedules; leakage regression battery. Gate T8.
  (Separable: the system is fully functional and improved after C; D changes only the
  weight-claim discharge path designed-in since A.)
- **Contingencies (pinned, from BACKEND_DECISION §3):** T2 misses 20× → profile, apply
  fast-helpers, re-gate; still missing → PCS spike (b) behind the same interface.
  Stage C verify > 100 s → fast-helpers on round-check host loops before declaring.

Total ≈ 4–6 weeks. int-model-approximation untouched throughout; no git pushes.

---

## 7. HONEST COMPARISON (ours-after-rebuild vs the field)

Sources: BACKEND_DECISION measurements (our box, RTX 4090) except where flagged
not-built/not-run. "ours-after" = post-Stage-C estimates (mid-case), post-Stage-D for
privacy; estimates italicized in spirit — they are NOT measurements yet.

| | ours TODAY | ours AFTER rebuild | DeepProve | JOLT Atlas | Artemis |
|---|---|---|---|---|---|
| prove | 1062 s (68m@1024 ≈ 1.04 s/tok, GPU) | ~1060–1280 s (batch adds, IPA-prove removal claws back) — UNMEASURED | 179 s (gpt2@64 ≈ 2.8 s/tok; CPU sumcheck, ~0 % GPU) | 7.7 s (nanoGPT 0.25M, CPU); GPT-2 claim ~15 s not run | NOT BUILT; halo2 lineage suggests 1–2 orders slower at GPT-2 scale |
| verify | 999.5 s | **~40–85 s** (≤60 s target; floor needs single-process walk) | **2.33 s** | 0.30 s (nanoGPT) | not measured |
| proof size | 175.6 MB | S0 ~171 / **S1 ~45 / S2 ~6–12 MB** | 10.25 MB (excl. one-time weight context) | small (not recorded) | KZG-class small (paper) |
| weight privacy | evals leak (≈31 linear functionals/matmul/proof; g in ~10² proofs) | **Stage D: hiding + masked — no candidate stack has this built** | NONE (non-hiding, evals in plaintext; "ZK" is README marketing) | NONE — verifier holds the model | **real (CP-SNARK, per paper)** — not independently run |
| integerization fidelity | registered int model, DiFR floor 1.35e-6, statement-bound hashes | identical (untouched) | float-calibrated 12-bit quant; would destroy the floor | exact integer lookups (good) but model public | quantized halo2 circuits; fidelity unevaluated |
| covert-channel closure | zero-advice softmax (0 bits), ±1 rmsnorm bracket, measured capacity layer | identical + batch adds 0 bits (§2.3) | error-band softmax = per-row channel ~2·err·scale by construction | deterministic semantics (good) | unknown |
| chaining/auditability | per-tensor commitments, byte-equality edges, 2 independent audits | identical (one opening protocol to audit instead of 1,535 transcripts) | GKR claim chaining only; no per-tensor commitments — our edge map has no home there | shared-preprocessing model | halo2 monolith |
| license | ours | ours | proprietary since 2026-05-12 (MIT fork `4cb8b9b6` lacks HyperKZG/GPU) | proprietary (ICME eval-only) | Apache-2.0 |
| GPU maturity | 11 CUDA drivers, but -dlto landmines + host-loop debt | same + fast-helpers debt explicitly tracked | PCS-only GPU; sumcheck CPU | none | none |
| extensibility | 9 obligation kinds, registry-free but OURS | + PCS-agnostic accumulator seam | closed 17-variant enum, 6 traits per op | ONNX op set | circuit-per-model |

**What we are better at (after rebuild):** weight privacy (the only stack with a
designed+built path: hiding commitments, masked sumchecks, ZK batch opening); the
registered integerized model with a measured 1.35e-6 DiFR floor bound to the statement;
zero-advice nonlinearities with a capacity-measurement layer; per-tensor auditable
chaining (byte-equality + homomorphic links) that survives the rebuild BECAUSE we kept
Pedersen; a GPU prover at ≈1 s/token; full-stack auditability (every absorb written by
us, two independent walkthroughs).

**What we remain worse at (stated without flinching):** raw verify latency — 40–85 s
vs DeepProve's 2.33 s; closing THAT gap needs the KZG-class batch (rejected §2.5 for
assumption/licensing/LTO reasons) AND killing the per-tensor commitment walk, i.e.
giving up the chaining design — a trade we are explicitly not making. Proof size at S1
(~45 MB) still 4× DeepProve unless we take the S2 re-layout. Prover RAM/streaming
complexity grows (38 GB Fr-expanded batch pass). Ecosystem maturity: their PCS code is
exercised by many parties; every line of ours is exercised by us and our audits.

---

## 8. OPEN QUESTIONS / UNCERTAINTY REGISTER (complete, per the no-hiding rule)

1. The 1999 s vs 999.5 s correction (header) — accepted here; BACKEND_DECISION should
   gain an erratum line when this design is reviewed.
2. §1.3's 85–90 % opening share is a model fit to per-driver aggregates, not an
   instrumented split — Stage A instrumentation replaces it.
3. Prover cost of the batch sumcheck (§2.2): 2.4 G + 1.6 G mult estimate and the
   38 GB streaming plan are paper numbers. Stage B gate measures; contingency = chunk
   the batch into 2–4 sub-batches (costs extra IPAs, stays sound — claims partition
   freely across batches by Lemma 3 applied per batch).
4. Subprocess-overhead estimate (~80–150 s) inferred from PHASE0-standalone vs walk
   deltas, not isolated; the single-process verifier is justified regardless (it
   removes a term that cannot otherwise go below 234 × CUDA-init).
5. -dlto: every new kernel (k_eq accumulate for M_j, affine batch-inversion for S1,
   any batched fold) is presumed miscompile-bait until probed (the §11 precedent).
6. headslice's 72 paired claims: the pair-equality check moves driver-side trivially,
   but it is the only place a driver consumes TWO claims' evals jointly — the toy pin
   must include this shape (claims on same tensor, different points, equality checked
   outside the batch).
7. S1's 45 MB misses the stated ≤30 MB; only S2 (second §13 campaign) or S3 (rejected)
   reach it. Decision needed from the project owner on whether 30 MB is hard.
8. Stage D's masked-sumcheck distribution claims (uniform rounds) need the same audit
   treatment as §10 — a toy pin with a statistical test, plus an explicit write-up of
   the simulator argument. Also the candidate-confirmation residual (§4.2 D5) needs a
   THREAT_MODEL_NOTES entry and an owner decision (documented-residual vs
   hide-activations-someday).
9. zkob_softmax (stage-2 driver) conversion is required for stage-2 walk
   compatibility but has no faithful-arch instance to e2e-test against — its selftest
   + a stage-2 orchestrator walk is the only coverage; scheduled in Stage C.
10. rowmax selector ties: tie-count reporting flows through prove_manifest, untouched —
    confirmed no interaction with the batch (ties affect WHAT is proven, not openings).
11. Whether `opening_batch` needs a manifest waiver gymnastics for the FROZEN harness
    manifest (check_transcript against frozen manifest will see an extra checked id —
    believed fine since extra coverage is not narrowing; verify against
    check_transcript.py semantics in Stage C).
