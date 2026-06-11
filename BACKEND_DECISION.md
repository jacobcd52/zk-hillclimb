# BACKEND_DECISION — prior zkML codebases vs. our stack, and the path to an efficient weight-private ZKP

Date: 2026-06-11. Hardware for all measurements: RTX 4090 (24 GB), 128-core EPYC-class CPU,
251 GB RAM, Ubuntu 22.04, CUDA 12.x. All clones/builds in `/root/backend-eval/` (local disk).
Evaluated commits: deep-prove `9d1a53e2` (2026-05-26, v2.0.1), jolt-atlas `ce0a265e`
(2026-06-03), zkGPT `8fcde848` (2025-08-27). Our reference point: faithful-arch-v1
(llama-68m, seq 1024, 65/65 obligations ACCEPT, DiFR 0.0156): **prove 1062 s wall,
verify 1999 s (sum of 235 per-driver verifies), proof+commitments 175.6 MB**
(`/root/zkorch/stage3v2-fa/prove_manifest.json`, `transcript.json`).

Each candidate got the five checks: (i) weight privacy real or marketing, (ii) commitment
determinism, (iii) field / CPU-GPU / measured numbers, (iv) extensibility for our
zero-advice obligations, (v) license & maintenance. Evidence is file:line in the local
clones; everything not built/measured is flagged.

---

## 0. The one-paragraph summary

**Nobody has implemented weight privacy.** All three systems we built and ran (DeepProve,
JOLT Atlas, zkGPT) use deterministic, non-hiding commitments and send weight-polynomial
evaluations at Fiat–Shamir points to the verifier in plaintext — the same leakage class as
our own stack. JOLT Atlas hands the verifier the whole model. The only system with real
commit-to-hidden-weights machinery (Artemis, ETH, Apache-2.0) is halo2-based and was not
benchmarkable in this window. Meanwhile DeepProve's measured *prove throughput* is in the
same ballpark as ours (≈1–3 s/token vs our ≈1.0 s/token), and its decisive advantage —
**verify 2.3 s vs our 1999 s, proof 10 MB vs our 176 MB** — comes from *batching
architecture* (claim chaining + one batched PCS opening), not from a fundamentally faster
protocol family. Its nonlinearities accept error-*bands* (the exact prover-freedom class
our project exists to close), its quantization is float-calibrated (would destroy our
1.35e-6 DiFR integerization floor), and its license flipped to proprietary eval-only on
2026-05-12. **Decision: KEEP OURS and rebuild the proof-transport layer along DeepProve's
architectural lines (claim accumulation + batched openings), with the MIT-licensed
deep-prove fork point `4cb8b9b6` recorded as contingency.** Details in §2.

---

## 1. Per-candidate findings

### 1.1 DeepProve (Lagrange-Labs/deep-prove) — PRIMARY; built, run, code-audited

Built with `--features cuda` (rust nightly-2026-01-27, builds clean in 2m52s).

**(i) Weight privacy: architecture yes, cryptography no.**
- The verifier holds weight *commitments*, never plaintext weights:
  `VerifierContext { commitment_ctx: CommitmentVerifierCtx }` with
  `model_commitments: BTreeMap<CommitmentId, VerifierCommitment>` —
  `zkml/src/iop/context.rs:75-84`, `zkml/src/poly_commit/context.rs:349-364`. This is the
  right commit-and-prove shape (unlike JOLT Atlas).
- But the commitment is **not hiding** and openings **leak evaluations**: weight-MLE
  evaluations at FS points are appended to the transcript in plaintext
  (`zkml/src/poly_commit/prover.rs:165-175` `transcript.append_scalars(&[eval])`;
  verifier consumes them at `verifier.rs:226-240`). The PCS opening call passes no
  blinding (`prover.rs:115-131`), and in dp-crypto itself
  (`~/.cargo/git/checkouts/dp-crypto-*/22e8e93/dp-crypto/src/arkyper/hyperkzg_gpu.rs:384-394`)
  `prove()` is literally `eval = poly.evaluate(opening_point)` + open. Zero hits for
  hiding/blinding/masking anywhere in dp-crypto or zkml.
- The eprint (2026/1112) is careful: its formal claims are verifiability/succinctness;
  "zero-knowledge" appears almost only in the bibliography. The README's "zero-knowledge
  proof system" is marketing. Third parties (ICME's 2025 zkML guide) independently rate it
  "not likely privacy preserving".
- Leakage magnitude, for honesty: a handful of field elements per weight matrix per proof
  (~hundreds of 254-bit evals/proof for GPT-2). Not invertible to weights in practice,
  accumulates across proofs (FS points differ per proof). Identical class to our stack's
  per-proof IPA openings of registered weight commitments.

**(ii) Commitment determinism: yes.** `commit()` has no RNG — commitment is a function of
(weights, SRS) only (`zkml/src/poly_commit/context.rs:299-321`; the GPU MSM at
`hyperkzg_gpu.rs:370-382`). Byte-determinism would survive in any fork. BUT intermediate
activations are **not individually committed** — layer-to-layer binding is GKR-style claim
chaining (output claim of layer N at a random point becomes input claim of layer N+1,
`zkml/src/iop/claim.rs:19-31`); only weights + lookup witnesses get commitments, and one
batched PCS opening discharges everything. This is exactly where the 860× verify advantage
comes from, and exactly why our byte-equality chaining concept has no native home there.

**(iii) Field / hardware / measured numbers.** BN254 scalar field (`ark_bn254::Fr`,
`zkml/src/bin/bench/llm.rs:34`), HyperKZG PCS, Blake3 transcript. The `cuda` feature moves
inference (burn) and the PCS opening to GPU; **the sumcheck prover is CPU** (GPU sat at
~0% util through proving; the dp-crypto sumcheck even warns it wants power-of-2 CPU thread
counts).

Measured on our box, GPT-2 (124M params) seq 64, GGUF Q2_K path, defaults (18 threads):

| phase | time |
|---|--:|
| one-time context generation (commit weights etc.) | 107.0 s |
| inference (witness gen) | 63.5 s |
| **prove_full** (witness commit 83.7 + claims 78.1 + PCS open 13.8) | **178.9 s** |
| **verify_full** | **2.33 s** |
| proof size | 10.25 MB |
| peak prover RAM | 26.2 GB |

(64-thread rerun: see addendum at end of §1.1.) README/paper numbers (7.6 min prove /
1.3 s verify, GPT-2 seq 512, 24-core EPYC) are consistent with what we see: ≈0.9–2.8
s/token prove, seconds verify. **For comparison our stack proves llama-68m seq 1024 in
1062 s ≈ 1.04 s/token.** The prove gap to us is small-to-none per token (their model is
1.8× larger; our integer pipeline does no float inference); the verify and proof-size gaps
are enormous (≈860× / ≈17×) and architectural.

Reproduction warts (code quality signal, both worked around):
- Stock HF `openai-community/gpt2` safetensors **crashes** at
  `zkml/src/iop/context.rs:329` "Found different MLE for polynomial wte.weight" — the tied
  embedding/lm-head tensor is quantized twice under different scales; their release
  profile force-enables `debug-assertions` (Cargo.toml:77-79), and **without that assert
  the mismatch would be silently accepted, committing one of two inconsistent MLEs**.
  The GGUF path avoids it.
- `--model llama2 --hf JackFram/llama-68m` (our exact reference model) **fails** in graph
  construction: einsum shape error ("axis d: input tensor 2 has size 64, output tensor 0
  has size 1") at any sequence length, after we supplied the missing
  `num_key_value_heads`/`rope_theta` config keys. Their llama2 path appears tested only
  against 7B-shaped configs. So no apples-to-apples llama-68m number on DeepProve.
- Repo ships git-LFS pointer stubs in `model_cache/` that the binary happily "loads".

**(iv) Extensibility: closed enum, lossy nonlinearity semantics.**
- All ops live in `enum Layer<T>` (`zkml/src/layers/mod.rs:74-98`, 17 variants) with
  parallel `LayerCtx`/`LayerProof` enums and giant match dispatch; adding an op means
  editing 5+ core files and implementing 6 traits (`ProvableOp` etc.,
  `zkml/src/layers/provable/mod.rs:423-456`). No registry/plugin mechanism. Feasible but
  invasive — fork-level work per obligation.
- Nonlinearities are lookup-based with **error-band acceptance**: softmax keeps an
  explicit `error_bound: f32` (`zkml/src/layers/transformer/softmax/mod.rs:89-106`) and
  the verifier accepts any output whose row sums fall in
  `normalised_sum_value ± error_bound·output_scale` via a range lookup
  (`softmax/lookup.rs:28-33`); rmsnorm commits a prover-computed normalization tensor as
  advice with float `round_ties_even` and per-row shift selection
  (`normalisation/rmsnorm/evaluate.rs:82-184`, `mod.rs:159-165`). **This is precisely the
  advice-freedom class our zero-advice softmax (0 bits) and ±1 rmsnorm bracket
  (≤1.6 bits/row) close.** Adopting their layers as-is would re-open covert channels our
  capacity layer currently measures in the 0.001 b/tok range — by construction their
  softmax band is a per-row covert channel of width 2·error_bound·scale.
- Quantization is 12-bit default, float-calibrated per run from representative inputs
  (`zkml/src/quantization/mod.rs:28-77`, `ScalingStrategy`). The proven function is *not*
  our registered integerized model; our DiFR integerization floor (1.35e-6 nats) and
  registered-weight-hash statement do not survive adoption without porting our entire
  integer pipeline into their graph.

**(v) License: the killer.** LICENSE at HEAD is the proprietary "Lagrange License":
internal testing/evaluation only, no derivative works, no redistribution, revocable at
will, $100 liability cap. It replaced MIT/Apache-2.0 on **2026-05-12** (commit `107e01d9`);
`Cargo.toml` still falsely says `MIT OR Apache-2.0`. Two mitigating facts we verified in
git history:
- Only 6 commits exist after the flip (serialization fix, release chores) — the MIT tree
  is functionally HEAD.
- **Fork point `4cb8b9b6` (2026-04-29, "feat: llama2 model safetensors support") is
  MIT/Apache INCLUDING the PCS** (ceno `mpcs`/Basefold — the proprietary dp-crypto
  HyperKZG backend only replaced it in `100ee5b5`, May 2026; dp-crypto itself was born
  2025-11-19 and has been proprietary since 2025-12-09, so no MIT tree ever had the
  HyperKZG/GPU backend). A legal fork therefore gets: full LLM layer machinery + Basefold
  hash commitments (deterministic, NOT homomorphic, no trusted setup), and none of the
  HyperKZG perf numbers.

### 1.2 JOLT Atlas (ICME-Lab/jolt-atlas) — built, run

- **(i) Weight privacy: none, by design.** The ONNX model (weights included) is *shared
  preprocessing*: `AtlasSharedPreprocessing { model: Model }` cloned into the verifier
  (`jolt-atlas-core/src/onnx_proof/preprocessing.rs:25-32`, `verifier.rs:44-50`); constant
  tensors are verified by the verifier evaluating the *plaintext* tensor's MLE itself
  (`ops/constant.rs:35-36`). The arxiv paper (2602.17452) claims witness-ZK via
  "BlindFold", but weights are public regardless. **Fails our hard requirement (a).**
- (ii) HyperKZG over BN254, Blake2b transcript, deterministic commits.
- (iii) CPU only, no GPU anywhere. Measured on our box: **nanoGPT (0.25M params, 4
  layers): prove 7.73 s, verify ~0.30 s** (tracing spans `ONNXProof::prove`/`::verify`),
  ~10 s wall including model load. README claims GPT-2 ~15 s — plausible, not run.
- (iv) The interesting part: **deterministic integer semantics** — softmax/exp via
  two-level decomposed lookup tables with exact quotient/remainder range checks
  (`atlas-onnx-tracer/src/ops/softmax.rs:91-152`, `ops/rsqrt.rs:46-49`); witness values
  fully determined by input + fixed-point scale. Philosophically the closest to our
  zero-advice discipline — worth mining for table-decomposition tricks for our exp table.
- (v) **Proprietary ICME eval-only license from day one** (same template as Lagrange's),
  while forking a16z's MIT Jolt. Active development. Code quality good.

### 1.3 zkGPT (security-Anonymous/zkGPT, USENIX Sec'25 artifact) — built, partially run

- **(i) Weight privacy: claimed in paper, absent in artifact.** Hyrax-style Pedersen
  vector commitments with **no blinding term**: `perdersen_commit` is a bare
  `Σ f[i]·g[i]` MSM (`src/hyrax.cpp`, `G1::mulVec`), no `h^r`, no ZK sumcheck masking.
  Same class as DeepProve and us.
- (ii) Deterministic commits (no randomness in commit paths). BN254 via mcl.
- (iii) CPU-only, C++. Demo is a hardcoded synthetic GPT-2 shape (12 layers / 768 / seq
  30, `src/main_demo_llm.cpp:26-37`). Measured: range-prove phase 17.9 s, weight commit
  38.3 s, then **OOM-killed at circuit init on our 251 GB box, twice (exit 137), even
  with ~200 GB free**. Their "<25 s GPT-2 prove" claim is **not reproducible under
  256 GB**; we cannot confirm or deny it. Verify path untested by us.
- (iv) Research-artifact quality: hardcoded dimensions, `#undef NDEBUG`, frozen since
  2025-08-27, anonymous repo, no model-loading path (synthetic weights). Not a base to
  build on; possibly a parts-bin (its attention-matmul GKR and range-prover are MIT).
- (v) **MIT license** — the only permissive one among the transformer-scale systems.

### 1.4 Polyhedra zkPyTorch / Expander — web + repo inspection only

zkPyTorch (the ML compiler with the Llama-3-8B ≈150 s/token single-thread claim and the
"without revealing model parameters" blog language) is **not public** — no repo exists as
of 2026-06. What is public: Expander (GKR prover, AGPL-3.0, active) and
ExpanderCompilerCollection (AGPL-3.0) — a generic GKR stack with prove+verify CLI but no
transformer pipeline. Nothing to adopt; AGPL would also be problematic. Dropped.

### 1.5 Artemis (pps-lab/artemis, arXiv 2409.12055) — web only, NOT built (honesty flag)

The one genuine commit-and-prove weight-privacy design with code: CP-SNARKs proving
inference against *hidden committed weights*, PCS-agnostic, Apache-2.0, repo includes
`gpt2` configs; reduces commitment-check overhead to ~1.1–1.2× on VGG-class models. Built
on Daniel Kang's halo2 zkml (slow prover family, KZG params up to 64 GB). Last push
2026-02. We did not build it in this window — proving-time at GPT-2 scale is the open
question (halo2 lineage suggests it loses to sumcheck/GKR stacks by 1–2 orders on prove).
**Relevance: not as a backend, but as the reference design if/when we add formal hiding
to weight commitments** (their Poly/Apollo trick: prove commitment-consistency of the
witness polynomial inside the existing proof rather than re-committing).

### Summary table

| | weight privacy (code) | det. commit | field / HW | measured (our box) | obligations injectable | license |
|---|---|---|---|---|---|---|
| **ours** | commit-and-prove, evals leak at FS points | yes (Pedersen) | BLS12-381 / GPU | 68m@1024: P 1062 s, V 1999 s, 176 MB | native (9 kinds, audited) | ours |
| **DeepProve** | same leakage class; verifier holds commitments | yes (HyperKZG) | BN254 / CPU+partial GPU | gpt2@64: P 179 s, **V 2.33 s**, 10.3 MB | closed enum; error-band nonlinearities | **proprietary** (MIT fork point `4cb8b9b6`) |
| JOLT Atlas | **weights public to verifier** | yes (HyperKZG) | BN254 / CPU | nanoGPT: P 7.7 s, V 0.3 s | exact-integer lookups (nice) but no privacy | proprietary |
| zkGPT | claimed; no blinding in code | yes (Pedersen) | BN254 / CPU | OOM >200 GB at circuit init | hardcoded GPT-2 | MIT |
| zkPyTorch | claimed; **closed source** | — | — | — | — | n/a |
| Artemis | **real (CP-SNARK)** per paper | hiding variant | BN254-ish / CPU | not built | halo2 circuits | Apache-2.0 |

---

## 2. THE DECISION

**KEEP OURS, and port DeepProve's transport architecture into it** (claim accumulation +
batched commitment openings + log-time-verify PCS for the batched opening). Not ADOPT, not
FORK+AUGMENT, and "HYBRID" only in the sense that the *architecture* — not the code — of
DeepProve's commitment layer is what we take.

### Why not ADOPT or FORK DeepProve

1. **It doesn't deliver the one thing we'd be adopting it for.** Weight privacy in
   DeepProve is the same deterministic-commitment-plus-leaky-openings story as ours
   (§1.1.i). Adopting buys zero privacy progress; formal hiding has to be built either way,
   and it is strictly easier to add blinding/masking to a stack whose every FS absorb we
   wrote and audited than to a foreign one.
2. **The prove-speed gap to us is roughly nil per token** (≈1 s/tok both, §1.1.iii). The
   headline "54–158× vs EZKL" is real but EZKL is the wrong baseline; against our stack
   the measured advantage is verify time and proof size — which are batching properties we
   can port (see plan), not protocol superiority.
3. **Its operator semantics would undo the project.** Error-band softmax, advice-tensor
   rmsnorm, float-calibrated quantization (§1.1.iv): adopting means re-opening
   covert channels we spent the project closing (zero-advice softmax: 0 bits; their
   softmax: ~per-row band) and losing the registered integerized model + 1.35e-6 DiFR
   floor + statement.registered_weight_hash. Replacing their nonlinearity layers with
   exact ones = rewriting our obligations in their framework = the FORK cost with none of
   the KEEP benefits.
4. **License.** HEAD is unusable beyond evaluation (no derivatives, revocable). The legal
   fork point `4cb8b9b6` loses the HyperKZG+GPU backend (proprietary dp-crypto) — we'd
   inherit Basefold (no homomorphic links — our affine limb links and skip-connection
   adds rely on commitment linearity) and an unbenchmarked configuration.
5. **Code-quality findings** (silent tied-weight MLE inconsistency masked only by
   debug-asserts; llama2 graph broken for non-7B shapes) do not inspire confidence for a
   soundness-critical adoption.

JOLT Atlas fails requirement (a) outright; zkGPT doesn't fit in our RAM and is a frozen
artifact; zkPyTorch is vaporware-grade for our purposes; Artemis is a privacy design
reference, not a transformer-scale backend.

### What KEEP+REBUILD-TRANSPORT costs us, obligation by obligation

Our 9 obligation kinds (66 manifest ids): sumcheck_matmul ×19, commitment_opening ×15,
rescaling_lookup ×15, rmsnorm ×5, skip_connection ×4, statement ×3, zkattn_softmax ×2
(softmax8+rowmax+headmerge composite), nonlinearity_lookup ×2 (swiglu), table_lookup ×1.

- **Port trivially (unchanged semantics, new opening discipline):** all of them, by
  construction — the change is WHERE openings are discharged, not WHAT is proven. Each
  driver currently ends sub-protocols with immediate IPA openings (12–17 per driver); under
  the rebuild each driver instead *emits* its terminal claims (point, eval, commitment-id)
  into a shared accumulator, and one batched opening (RLC over claims, single MSM or
  HyperKZG multi-open) discharges the lot. Sumcheck round verification (cheap) is
  untouched; FS schedules gain one global accumulation phase — every driver's absorb
  schedule changes at the tail, so **every selftest and both independent-audit FS
  walkthroughs must be re-run/re-checked** (the §13 rule). That is the real cost: a
  re-validation campaign over 9 drivers (~1–2 weeks of the kind of work sections 14–21
  document), not new cryptographic design.
- **Byte-equality chaining: survives unchanged.** Commitments stay deterministic Pedersen
  per-tensor; the orchestrator's com_X == com_Xr checks and `chain_edges` are untouched.
  (This is a thing we KEEP that DeepProve cannot do at all — per-tensor commitments are
  what make our chain auditable and our capacity layer instrumentable.)
- **Homomorphic links: survive unchanged** (Pedersen linearity; would have died under a
  Basefold fork — a key reason not to fork).
- **Become unnecessary:** possibly some of the 15 standalone commitment_opening ids — under
  a global accumulator, a registered weight's opening is just another accumulated claim;
  the separate per-weight opening transcripts dissolve into the batch. The manifest keeps
  the ids; the proofs merge.
- **Capacity-measurement layer: needs nothing new from the backend.** Acceptance
  predicates per obligation are unchanged; tie-count reporting (rowmax) unchanged. It
  additionally gains a *smaller* attack surface to reason about (one opening protocol
  instead of ~235 transcripts).
- **What we adopt as new work (the actual price):** (1) the claim-accumulator + RLC
  batched-opening protocol and its security argument (standard, but must be written and
  audited like §10's IPA was); (2) optionally a log-verify PCS (HyperKZG via MIT
  arkworks/jolt code, BN254 or keep BLS12-381 with arkworks KZG) for the single batched
  opening if batched-IPA's one O(N)-MSM verify is still too slow; (3) the weight-privacy
  endgame (blinded Pedersen + masked openings for weight commitments only) which NO
  candidate would have given us anyway — Artemis is the design reference.

### Quantified expectation for the rebuild

Our verify is 1999 s ≈ 235 transcripts × (rounds + several linear-time IPA-opening MSMs
over up to 2²⁰-generator domains). Collapsing to ONE batched opening leaves: cheap round
checks (seconds total) + homomorphic link/equality checks (seconds) + one MSM at the
largest domain (~2²⁰–2²² points ≈ 1–5 s GPU, tens of s CPU) + per-claim RLC folding.
Realistic target: **verify 10–60 s without changing the commitment scheme; low single-digit
seconds with a KZG-class PCS for the batch**. Proof size: dominated today by per-driver
commitments + 12–17 IPA proofs each (~log N G1 points per opening × ~235); batching cuts
the opening payload ~100× → target **≤30 MB**, further with KZG. Prove time 1062 s is a
separate (GPU kernel) workstream — unaffected by this decision, and already competitive.

---

## 3. Next 3 steps (chosen path)

1. **Design + prototype the claim accumulator on one driver pair** (zkob_fc → zkob_rescale,
   the validated chain): emit terminal claims instead of opening; a new `zkob_batchopen`
   driver does the RLC + single IPA opening per generator domain; orchestrator gains an
   `opening_batch` obligation id. Deliverable: design note (FS schedule, security
   reduction sketch — standard RLC-binding argument), selftest with the full evil-mode
   battery (claim tamper, RLC tamper, cross-driver claim swap), measured verify delta on
   the real-scale pair. Acceptance gate: ≥20× verify reduction on the pair.
2. **PCS endgame decision spike:** behind the same accumulator interface, implement the
   batched opening twice — (a) batched IPA (current Pedersen, no new assumptions) and
   (b) arkworks KZG/HyperKZG (MIT — a16z jolt or arkworks, NOT dp-crypto) on a curve we
   pick; measure verify ms + proof bytes, check byte-determinism of commitments and
   linearity for the affine links; pick. Record SRS/trusted-setup policy if KZG wins.
3. **Weight-privacy memo (the gap nobody filled):** leakage accounting of the current
   scheme (evals/proof × proofs/epoch vs weight entropy), then the hiding design: blinded
   Pedersen commitments + masked/ZK openings for the 15 registered-weight ids only
   (activations stay deterministic for chaining), Artemis(2409.12055) as the
   commitment-consistency reference. Decide with the threat-model owner whether formal
   hiding is in-scope for stage 3 or documented-leakage suffices.

Contingency: if step 1 misses its 20× gate or step 2 stalls, re-evaluate FORK of
deep-prove @ `4cb8b9b6` (MIT, Basefold) with our obligations as new enum variants —
costed in §2 as strictly worse today, but the fork point is pinned and the audit of their
layer traits (§1.1.iv file:line map) is reusable.

---

## 4. Honest unverified-items list

- DeepProve at seq 512 / 1024 on our box (their README scale): not run — the seq-64 run +
  their published scaling is what we have. Their GPU story beyond the PCS (the
  marketing "GPU acceleration supported today") measured at ~0% GPU during sumcheck.
- DeepProve CPU-only build (no `cuda`): not separately benchmarked; 18-vs-64-thread
  sensitivity run was attempted (results in `/root/backend-eval/gpt2-gguf-seq64-t64.log`
  if it completed after this writing; the 18-thread numbers in §1.1 are the official ones).
- deep-prove llama-68m: blocked by their einsum shape bug — no apples-to-apples llama
  number; the per-token comparison uses GPT-2@64 vs our llama-68m@1024.
- zkGPT's <25 s claim: unfalsifiable under 251 GB; phases measured up to OOM.
- JOLT Atlas GPT-2 (~15 s claim): not run (nanoGPT measured instead).
- Artemis: not built; privacy mechanism taken from paper + repo README only.
- Whether the last MIT deep-prove commit (`4cb8b9b6`) actually builds and proves GPT-2
  with the mpcs/Basefold backend: not attempted (fork-point claim is from git history +
  Cargo.toml inspection, not a build).
- All license interpretations here are engineering reads, not legal advice.

Artifacts: logs and clones under `/root/backend-eval/` (gpt2-gguf-seq64-gpu.log,
jolt-nanogpt-trace.log, zkgpt-demo*.log, llama68m-seq64-gpu.log, gpt2-seq64-gpu.log);
deep-prove CSVs at `/root/backend-eval/deep-prove/bench{,-llm}.csv`.
