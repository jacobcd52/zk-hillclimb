# STAGE_D_REPORT — weight privacy (hiding registration + hidden weight claims + ZK weight sub-batch)

Date: 2026-06-12. Builds on Stage C2 (STAGE_C2_REPORT.md). Design basis:
TRANSPORT_REBUILD_DESIGN §4 with the TRANSPORT_REVIEW **F7 amendment implemented**
(committed-round-message sumcheck; §4.5 as-built note appended to the design doc).
Reference formulation: Artemis (arXiv:2409.12055) commit-and-prove — we take the
in-protocol route (hiding Pedersen + sigma protocols + a blinded IPA) because every
sumcheck here is already per-round-FS; no new proof system, no new assumptions
beyond DLOG in the existing group.

## 0. Executive summary

| sub-step | implemented | validated by |
|---|---|---|
| D1 blinded weight registration (hiding Pedersen rows) | **YES** | wselftest + fc/rmsnorm wpriv selftests + hash-pin tamper cases |
| D2 hidden weight claims + hidden weight-touching sumcheck rounds | **YES** (committed round messages — the F7 mechanism, subsumes Libra masks; see §3.2) | fc wpriv selftest (driver sumcheck), wselftest (batch sumcheck) |
| D3 ZK final opening (blinded batched IPA, no a_final reveal) | **YES** | wselftest + driver wpriv selftests |
| D4 leakage regression (no weight-MLE eval in any artifact) | **YES — CLEAN** | scans inside all three selftests + standalone bench regression (§6) |

Honest ACCEPT and every-forgery-REJECT preserved everywhere: full 13-TU selftest
re-validation after the change: **C2 HEADER-EDIT REVALIDATION: ALL 13 PASS** (§4). The public batch path is bit-identical
to Stage C2 (zkob_claims.cuh NOT edited; all new code in a new header zkob_wpriv.cuh
included only by zkob_batchopen.cu / zkob_fc.cu / zkob_rmsnorm.cu).

NOT done (documented, §8): orchestrator/walk wiring (pair_walk + DriverPool flags for
the 15 registered tensors), so the full llama-68m wpriv walk is pending; the rmsnorm
activation-statement residual is structural (§5.3).

## 1. What was built (files)

- **zkob_wpriv.cuh** (NEW, ~700 lines; the only header added — the 3 protected
  headers untouched, zkob_claims.cuh untouched):
  - `wp_rand()` — /dev/urandom blinds with fs_challenge_fr's limb distribution
    (max point mass ≤ 3·2⁻²⁵⁶; statistical distance from uniform < 2⁻³²).
  - `wp_hide_rows()` — D1: com[r] += s_r·H via the long-probed k_g1_scale /
    k_g1_add_pairs shapes (**no new G1 kernel**; cross-checked against the 1-thread
    h_mul/h_add path in `wpriv_probe()`).
  - blind stores (all prover-private, never read by a verifier): `*.blinds.bin`
    (registration row blinds), `waccdir/cblinds.bin` (per-claim (v,t)),
    `waccdir/blindrefs.txt` (comref → blinds file).
  - `ped_qh`, `lagrange3_g1` (+ homomorphism probe), `h_sub`.
  - `SchnorrH` (PoK of dlog base H), `Schnorr2` (2-base PoK) with FS challenges
    from the caller's transcript.
  - `ZkIpaProof` + `zk_ipa_prove/verify` — D3 (blinded L/R, Schnorr2 final).
  - `wbatch_prove/wbatch_verify` — the weight sub-batch (D4 protocol, §3).
  - `wp_leak_scan` — D4 regression helper.
- **zkob_batchopen.cu**: `wprove` / `wverify` / `wselftest` CLI; the public
  prove/verify/selftest paths unchanged.
- **zkob_fc.cu**: `--wpriv <waccdir> <W-blinds>` prove/verify mode (committed
  driver-sumcheck rounds, C_W terminal + Schnorr, claim routing);
  `commit ... --hiding <q.bin> <blinds_out>` hiding registration; 2 new selftest
  cases. Plain and claim-mode paths unchanged.
- **zkob_rmsnorm.cu**: `--wpriv <waccdir> <g-blinds>` (committed val_g + Schnorr
  outer-product terminal, g-claim routing); 2 new selftest cases. Unchanged otherwise.
- **TRANSPORT_REBUILD_DESIGN.md §4.5**: as-built amendment resolving F7.

Artifact inventory of the weight path (all verifier-consumed; everything else the
prover holds is private): `waccdir/claims.bin` (Committed blobs: id, comref, shape,
point, tag=1, **C_v** — no scalar eval), `wbatch_sumcheck.bin` (3 G1 round
commitments per round), `wbatch_vfin.bin` (G1 terminal commitments),
`wipa_batch_<G>.bin` (blinded L/R + Schnorr2), `drvstates.bin`; per-driver:
`wsc.bin` (fc: round commitments + claim, claim_X public + C_W + Schnorr),
`outer_w.bin` (rmsnorm: val_R, val_W public + C_val_g + Schnorr).

## 2. D1 — blinded weight registration

`com_W[r] = Σ_c W[r,c]·g[c] + s_r·H`, fresh per-row blinds from urandom, H = the
q.bin slot-1 generator registered (hash-pinned) since Stage A's §4.4 touch —
independent of g[] and Q (all drawn by the same trusted-setup discipline as the
existing pp; public.json pins the file hashes). Blinds live prover-side only.

Hash-pin + chain still work, two ways, both tested:
1. **Driver transcript**: the prover RECOMPUTES the hiding com from (W, blinds)
   rather than loading the registered file, so a tampered/substituted registration
   file diverges prover/verifier transcripts → driver REJECT (fc + rmsnorm selftest
   cases "registered com tamper").
2. **Batch comref hash**: G0w absorbs the registered file's SHA-256 before ρ;
   post-driver tampering REJECTs at the named FS-binding locus (wselftest
   "substituted (re-blinded) commitment file" → wround0; fc case → wround1 — with a
   single weight claim C_cur0 = ceval is ρ-independent, so the divergence first
   bites at the round-0 challenge).

Soundness: Pedersen with an extra independent generator stays binding under DLOG —
a second opening (w', s') ≠ (w, s) of any row yields a nontrivial relation among
(g[], H). Hiding: each row carries an independent uniform s_r·H → rows are uniform
group elements. Registered-weight rows appear in **no affine link** (re-checked: the
links are activation-side — rescale links com_X/com_X̂/com_rem, rmsnorm links
com_L rows/com_P1/com_P2), so blinds break no existing homomorphic identity; the
batch fold simply carries the folded blind (§3.3).

## 3. D2/D3/D4 — the private discharge path (as built)

### 3.1 Claim routing (the §4.4 dual accumulators, realized)

Weight claims (fc `:W`, rmsnorm `:g`) are emitted with `EvalVar = Committed`,
`ceval = C_v = v·Q + t·H` (fresh t per proof) into a SEPARATE weight accumulator
(waccdir); X/Y and all other claims stay plain in the public accumulator. Both
accumulators get the driver's drvstate. The verifier-side driver emits its own
recomputed lists; each batch does its own byte-compare (`claims_match` /
`wclaims_match`). Routing forgeries: a plain claim inside the weight batch REJECTs
at `wevalvar`; a Committed claim inside the public batch still REJECTs at the
Stage-A `evalvar` gate (both tested = BO-13a/b).

### 3.2 Hidden sumcheck rounds — committed round messages (the F7 fix; replaces Libra masks)

Wherever round evals carry weight functionals — the fc zkip sumcheck on
(X_fold, W_fold) and the weight sub-batch's batch-eval sumcheck — the prover now
sends Pedersen commitments instead of scalars:

```
C_p0 = p(0)·Q + τ0·H,  C_p1 = p(1)·Q + τ1·H,  C_p2 = p(2)·Q + τ2·H
τ0, τ2 fresh;  τ1 = τ_cur − τ0
verifier:  C_p0 + C_p1 == C_cur   (in G1)
fold:      C_cur' = l0(x)·C_p0 + l1(x)·C_p1 + l2(x)·C_p2   (lagrange3 is linear)
prover:    τ_cur' = lagrange3(τ0, τ1, τ2)(x)
```

Initial claims: fc C_cur0 = claim·Q (public claim, zero blind); weight batch
C_cur0 = Σ ρⁱ·C_v_i (verifier-homomorphic — exactly the F7 observation that the
initial claim exists only inside commitments). Terminals: committed per-tensor
C_vfin_j with the homomorphic G3 check Σ_j M_j(r)κ_j·C_vfin_j == C_cur; the prover
solves the last nonzero-coefficient blind so H-components balance (all-zero
coefficients → prover throws; probability negligible, prover cannot induce it —
r is squeezed after everything it controls).

**Deviation from the task's D2 wording, flagged:** the task (and design §4.2-D2)
described Libra-style mask polynomials. The F7 audit already mandated committed
round messages for the weight batch (a hidden initial claim leaves no public `cur`
for masked rounds to check against). Committed round messages make the round
messages *perfectly hiding on their own*, so a mask polynomial — and its commitment
and its extra ZK opening per sumcheck — would be dead weight. One mechanism now
covers both D2 sites. Goal achieved is identical (no plaintext round evaluation
anywhere on a weight-touching sumcheck); mechanism is the audit's own.

Driver terminal identities become sigma proofs:
- fc: `E := C_cur − claim_X·C_W` is δ·H for an honest prover
  (δ = τ_fin − claim_X·t_W); Schnorr PoK of dlog_H(E).
- rmsnorm: `E := val_R·C_val_g − val_W·Q = (val_R·t_g)·H`; same proof. (val_R,
  val_W are public claims of public tensors.)

### 3.3 D3 — the ZK batched IPA

Relation per domain group: `P0 = ⟨gen, a⟩ + ⟨a,b⟩·Q + β·H` where the verifier forms
`P0 = C*_g + V*_g`, with C*_g = Σ_j ρ'_j κ_j · rowfold(com_j) (hiding rows → carries
S*_g = Σ_j ρ'_j κ_j ⟨s_j, me_weights(u_row_j)⟩ on H) and V*_g = Σ_j ρ'_j κ_j ·
C_vfin_j (carries T*_g on H). Prover-side β = S* + T*. Rounds are the pinned header
IPA plus `L += β_L·H`, `R += β_R·H` (fresh), `β' = β + x²β_L + x⁻²β_R`; the verifier's
P-fold equation is unchanged (H components ride inside the points). Final round: NO
a_final reveal — a 2-base Schnorr PoK of (a_f, β_f) for
`P_fin = a_f·(g_f + b_f·Q) + β_f·H` (3 G1 + 2 Fr instead of 1 Fr).
Activations' public batch keeps the fast non-hiding opening, untouched.

### 3.4 Soundness argument (to be audited)

All Stage-D checks are linear G1 equations over (gen[], Q, H) plus two Schnorr
classes. The lifting lemma we rely on: **an accepting transcript either satisfies
the corresponding scalar identities on the Q-components, or yields a nontrivial
discrete-log relation among (gen[], Q, H).** Concretely:

1. *Round check*: C_p0 + C_p1 == C_cur with p(0)+p(1) ≠ cur forces
   (p0+p1−cur)·Q = (τ_cur−τ0−τ1)·H — a Q/H relation.
2. *Terminal checks* (G3w, fc/rmsnorm sigma): same shape. For the sigma proofs,
   forking extracts dlog_H(E); if the Q-component of E is nonzero that dlog is again
   a Q/H relation. (Knowledge soundness of Schnorr in the FS/ROM model.)
3. *ZK IPA*: standard Bulletproofs-with-blinding extraction — Schnorr2 extracts
   (a_f, β_f); the round equations then extract a full (a, β) with
   P0 = ⟨gen,a⟩ + ⟨a,b⟩Q + βH or a relation among (gen, Q, H). Binding of the hiding
   rows pins a to the registered row-fold RLC.
4. With all Q-component identities in hand, the remaining argument is **verbatim the
   audited public §2.2 reduction** (TRANSPORT_REVIEW §0–§2: completeness identity,
   degree bound, G3 terminal composition, G5 consistency, ρ/SZ bounds in
   max-point-mass form — ρ is squeezed after G0 absorbs every C_v blob, every comref
   hash, and every drvstate, so the cancellation argument carries over with evals
   replaced by cevals).

So: weight-batch soundness = public-batch soundness + DLOG hardness of the
registered group with the independent H. Blinding weakens nothing — it only ADDS the
H dimension, and every check tolerates it exactly on the prover-known blind.

FS schedule pins kept: own seed suffix `:opening_batch_w`; own claims_match; F3 com
row-count check before any fold; F6 redundant-header cross-checks + n_claims==0
reject; F4 length-prefixed claim blobs (ceval inside the blob).

### 3.5 Zero-knowledge argument (to be audited)

Simulator for the entire weight path, given only public inputs (registered hiding
com files, public claims, transcript seeds):

- C_v, C_W, C_val_g: uniform G1 elements (fresh t per proof) — simulate as uniform.
- Round messages: C_p0, C_p2 uniform; C_p1 := C_cur − C_p0 (the honest C_p1 is
  exactly this group element; its "content" is determined by revealed elements, so
  it carries no new information). Fold homomorphically.
- Terminals: C_vfin_j uniform for j ≠ J, C_vfin_J := c_J⁻¹·(C_cur − Σ_{j≠J} c_j·C_vfin_j)
  — identical to the honest distribution (the honest prover solves t'_J the same way).
- ZK IPA: L/R uniform (fresh β_L, β_R); Schnorr/Schnorr2 simulated per standard HVZK
  (uniform responses, derived A).

Every weight-path message is therefore uniform or determined-by-revealed-elements;
distribution gap vs honest ≤ the wp_rand sampling bias (< 2⁻³² statistical distance
per blind) — **statistical ZK in the FS heuristic**, hiding of the registration
under no assumption (information-theoretic given uniform blinds). The
re-randomization sanity check (re-prove same claims → byte-different round
commitments and IPA, still ACCEPT) is in wselftest.

## 4. Selftest verdicts (run on this box, RTX 4090)

- `zkob_batchopen wselftest` (NEW — weight sub-batch battery):
  **26/26 ALL PASS** — honest accept; D4 mini-scale leak scan CLEAN + scanner
  positive control; ZK re-randomization (2 checks + re-accept); WBO-1a false
  committed eval → wround0; WBO-1b adaptive (forged C_vfin past G3) → wipa8;
  doctored-ρ → wround0; β-shift blind lie → wipa4; byte tampers on round
  commitments (wround0/wround1), n_claims field (wxcheck), C_vfin (wterminal),
  Schnorr response (wipa4), L point (wipa8), ceval bytes (wclaims_match);
  drvstate divergence → wround0; claim omission → wclaims_match; BO-13a plain
  claim in weight batch → wevalvar; BO-13b committed claims at the PUBLIC batch →
  evalvar (path still closed); empty → wempty; substituted re-blinded com →
  wround0; missing group IPA → wgroup_missing; restores → accept.
- `zkob_batchopen selftest` (public batch, unchanged code path):
  **33/33 ALL PASS**.
- `zkob_fc selftest`: **ALL PASS** — 3 legacy + 3 claim-mode (9/9 each) + 2 NEW
  wpriv cases (**15/15 each**): honest; D4 claim_W scan CLEAN over 17 artifacts +
  plain-mode positive control; committed-round tampers C_p0/C_p1/C_p2 → driver;
  Schnorr z and C_W tampers → driver; hiding-registration tamper → driver
  (transcript divergence); com_Y tamper → driver; public/weight claims_match;
  weight ZK-IPA tamper → wipa; post-driver registration tamper → wround1
  (hash-pin at the batch); restore → accept.
- `zkob_rmsnorm selftest`: **ALL PASS** — small + real-scale + 2 claim-mode +
  2 NEW wpriv cases (**11/11 each**): honest; D4 val_g scan CLEAN over 31
  artifacts; documented residual val_g = val_W/val_R pinned explicitly (§5.3);
  C_val_g + Schnorr tampers → driver "outer product Schnorr"; hiding com_g
  tamper → driver; semantic evil=5 (W[i]+1) → driver "outer product Schnorr";
  wclaims_match / wipa / public claims_match; restore → accept.
- Full 13-TU re-validation (`selftests_c2.sh`, after the 3-TU rebuild):
  ```
  vrf_toy_batchopen  PASS (4 s):   TOY-BATCHOPEN: ALL PASS
  zkob_batchopen     PASS (5 s):   ZKOB-BATCHOPEN SELFTEST: 33/33 ALL PASS
  zkob_fc            PASS (11 s):  ZKOB-FC SELFTEST: ALL PASS
  zkob_rescale       PASS (5 s):   ZKOB-RESCALE SELFTEST: ALL PASS
  zkob_skip          PASS (1 s):   ZKOB-SKIP SELFTEST: ALL PASS
  zkob_glu           PASS (24 s):  ZKOB-GLU SELFTEST: ALL PASS
  zkob_rope          PASS (16 s):  ZKOB-ROPE SELFTEST: ALL PASS
  zkob_headmerge     PASS (42 s):  ZKOB-HEADMERGE SELFTEST: ALL PASS
  zkob_headslice     PASS (126 s): ZKOB-HEADSLICE SELFTEST: ALL PASS
  zkob_rmsnorm       PASS (73 s):  ZKOB-RMSNORM SELFTEST: ALL PASS
  zkob_softmax       PASS (108 s): ZKOB-SOFTMAX SELFTEST: ALL PASS
  zkob_softmax8      PASS (162 s): ZKOB-SOFTMAX8 SELFTEST: ALL PASS
  zkob_rowmax        PASS (158 s): ZKOB-ROWMAX SELFTEST: ALL PASS
  C2 HEADER-EDIT REVALIDATION: ALL 13 PASS
  ```
  (The wselftest is additionally run standalone: `zkob_batchopen wselftest` →
  **ZKOB-WBATCH SELFTEST: 26/26 ALL PASS**.)

## 5. D4 — leakage regression

### 5.1 What is scanned

The known weight-MLE evals (fc `claim_W = W̃(u_output,u_input)` read back from the
prover-private cblinds stash; rmsnorm `val_g = g̃(u_c3)`) are byte-pattern-scanned
(32-byte little-endian Fr) across EVERY verifier-consumed artifact of the wpriv
pipeline: obdir files (dims/com_X/com_Y/wsc/outer_w/com_* + registered com),
public accumulator (claims.bin, drvstates, batch_sumcheck, batch_vfin, ipa_batch_*),
weight accumulator (claims.bin, drvstates, wbatch_sumcheck, wbatch_vfin,
wipa_batch_*), and the verifier-side recomputed accumulators.

### 5.2 Outcome

- wpriv path: **0 hits** in every run (fc selftest ×2 shapes, rmsnorm selftest ×2
  shapes, wselftest mini-batch over 4 hidden evals, and the bench-scale regression
  in §6).
- Scanner positive control: the Stage-C2 plain claim-mode artifacts DO contain
  claim_W (sumcheck.bin and acc/claims.bin) — found in both, as expected; the
  prover-private cblinds.bin contains the secret (it must — it is not shipped).
- Pre-D vs post-D, per fc proof: plaintext weight functionals went from
  ≈ 3·log(IN)+1 ≈ 31–37 (round evals + claim_W; §4.1) **to 0**. The batch-side
  weight terminal v'_j (a direct weight functional in the clear pre-D, per the
  review's honest line) is now a hiding commitment.

### 5.3 What still leaks (quantified, honest)

1. **rmsnorm structural residual**: W = R⊗g is a public CHAIN tensor; val_W/val_R
   = val_g is computable from public claims (demonstrated as an explicit selftest
   check) — ≈1 g-functional per proof, ~C_pad proofs to recover g, UNCHANGED by any
   proof-layer fix because Y = W_∘X exposes g multiplicatively in the activation
   statement itself. Closing it = hiding activations (design §4.2-D5's "different
   project"). The g-claim hiding still matters: it removes the *additional* clean
   functionals and keeps the weight batch uniformly committed.
2. **Layer-0 activation claims**: with public input X, the public Y claim is a
   known-coefficient W functional (≈2 per proof vs ≈33 pre-D); same
   activation-statement class, out of scope by design (activations are the
   statement and the chaining substrate).
3. **Candidate confirmation** (design D5(i)): deterministic activation commitments
   still allow confirming a fully-guessed W. Unchanged, flagged.
4. wp_rand bias: < 2⁻³² statistical distance per blind — negligible, stated.

## 6. Overhead (bench at llama-68m gate_proj scale: B=64, IN=768, OUT=3072)

`stage_d_bench.py` (this box, RTX 4090; each step is one CLI invocation, so every
number CARRIES the ~0.3 s process+CUDA-init floor — the C2 serve-mode pool removes
that at walk scale):

```
                      plain (C2)   wpriv (D)
registration              0.91 s      0.91 s     (hiding adds < 0.01 s: GPU-side blind add)
driver prove              1.09 s      1.38 s     (+0.29 s: 36 committed rounds + C_W + Schnorr)
driver verify             0.29 s      0.44 s     (+0.15 s: homomorphic round folds + Schnorr)
public batch prove        1.39 s      1.07 s     (W tensor left the public batch)
public batch verify       0.57 s      0.56 s
weight batch prove           —        1.76 s     (22-round committed sumcheck + 1 ZK IPA @4096)
weight batch verify          —        0.74 s
TOTAL prove               2.48 s      4.21 s
TOTAL verify              0.85 s      1.74 s
```

Proof-side bytes (this one obligation): plain 11,815 → wpriv 28,506 (**+16.7 KB**).
The growth is exactly the commitment widening: wsc.bin 4,708 vs sumcheck.bin 1,060
(round messages 144-byte G1 instead of 32-byte Fr), wbatch_sumcheck 9,520
(22 rounds × 3 × G1) + wbatch_vfin 152 + wipa 3,672 (Schnorr2 in place of a_final,
+blinds change no L/R count); claims.bin SHRINKS on both sides (the weight claim's
scalar eval is gone from the public list; the weight list carries one C_v).

Walk-scale projection (NOT measured — walk wiring pending, §8.1): per walk this adds
19 fc committed-round sumchecks (+~0.3 s each prover-side at these L; the G1 round
commitments are 1-thread host muls, ~70/sumcheck), 5 rmsnorm sigma terminals
(negligible), and ONE weight batch over the 15 registered tensors (Σ ≈ 33 M
elements, 25 rounds, 3 ZK-IPAs @1024/4096/32768). Against C2's 636 s prove / 27.1 s
verify this lands inside the design's ≈ +1–2 % prove envelope; verify gains the
weight batch (≈ the cost profile of the existing public batch's weight share plus
3 Schnorr-final IPAs) — estimate +2–5 s, to be replaced by the official number when
the walk is wired. Proof-size delta at walk scale: ≈ +0.2–0.3 MB
(19×3.6 KB wsc deltas + one weight batch ≈ 25 KB + 3 wipa files), matching the
design's ≈ +0.2 MB line.

Standalone D4 regression at bench scale (also in `stage_d_bench.py` output):
```
scanned 20 artifacts for claim_W: CLEAN (0 hits)
positive control (plain path): claim_W found in 2/2 expected artifacts
prover-private blind stash contains claim_W: True
D4 BENCH REGRESSION: PASS
```

## 7. Deviations from the design/task text (all flagged above)

1. D2 mechanism: committed round messages instead of Libra mask polynomials
   (mandated by F7 for the batch; adopted for the fc driver too — strictly simpler,
   cheaper, same hiding goal; §3.2).
2. The fc driver sumcheck masking was REQUIRED, not optional: with plaintext round
   evals, hiding claim_W alone is vacuous (cur is publicly recomputable and
   claim_W = cur/claim_X). The task's D2 scoping ("batch-eval sumcheck rounds")
   would have left this hole; we closed both sites.
3. rmsnorm g: proof-layer leakage closed; statement-layer residual documented
   (§5.3) — full g privacy is impossible under public activations, stated rather
   than papered over.
4. Committed rounds change the wpriv driver FS schedule (G1 absorbs instead of Fr
   absorbs) — wpriv is a distinct, flag-selected mode; plain and claim-mode
   transcripts are byte-identical to Stage C2.

## 8. What's left (clean handoff)

1. **Walk wiring**: pair_walk.py / coordinator must (a) register the 15 weight
   tensors with `--hiding` (one `blinds.bin` per tensor under registration/private/),
   (b) pass `--wpriv <waccdir> <blinds>` to the 8 fc-type and 1 rmsnorm-type
   obligations per layer-walk, (c) run `zkob_batchopen wprove/wverify` beside the
   public batch, (d) add `wclaims_match` + wbatch verdict to the run gate, and
   (e) extend public.json pinning to assert q.bin's 2-slot H. No driver or batch
   code should need changes (the CLIs are in place); selftests validate every
   protocol path at driver+batch scale, but the FULL llama-68m wpriv walk has not
   been run.
2. Official walk-scale timings of the weight batch over the real 15-tensor set
   (Σ ≈ 33M elements; §6's single-tensor bench suggests the design's +1–2% prove
   envelope holds, dominated by the one extra 25-round batch).
3. Simulator write-up to a publishable standard (§3.5 is the argument sketch the
   design's §8.8 demands; an external audit pass like TRANSPORT_REVIEW's is the
   right next gate).
4. T6 audits / proof-size S1 work from the C2 backlog are orthogonal and untouched.

## 9. Build/validation discipline

No new G1 kernel (the -dlto rule): blind arithmetic rides k_g1_scale,
k_g1_add_pairs, k_g1_mul/k_g1_addsub 1-thread helpers, and the probed Stage-B fast
IPA helpers; `wpriv_probe()` cross-checks wp_hide_rows element-exact against the
1-thread path and the lagrange3_g1 homomorphism identity at every selftest startup.
zkob_claims.cuh was not edited → no rebuild-all-includers obligation; the 3 edited
TUs were rebuilt with the pinned sm_89 -dc -dlto line and the full 13-TU battery
re-run regardless. Protected headers untouched. No git commits; nothing pushed;
int-model-approximation untouched.
