# Option A — fast + fully-private single-layer prover (plan)

Decision (2026-06-19): build on our own zkLLM-derived prover (this repo), add **activation
privacy** (weights are already private — Stage D), and pursue **both speed levers**. Single-layer
proofs are the dev unit; at runtime we sample a fraction of layers (statistical/economic soundness,
quantified by the covert-capacity curves) — see the layer-sampling discussion.

## Where we start (current state of the repo)
- **Weight privacy: DONE + audited** (STAGE_D_REPORT). Hiding Pedersen rows (`com_W[r]=⟨g,W[r]⟩+s_r·H`),
  committed-round (hidden) sumcheck, blinded ZK-IPA. 13/13 selftests pass; full `--wpriv` walk works.
- **Activation privacy: NOT done — deliberate gap** (STAGE_D_REPORT §5.3). Activation commitments are
  deterministic/binding-only → guess-and-confirm possible. This is "D5 / different project". OUR JOB.
- Field: BLS12-381 scalar (256-bit), hand-coded CUDA. Commitment: row-wise Pedersen over G1.
  Bottleneck: G1 MSM (hand-coded, ~64 threads, no NTT) + per-op IPA openings. Proof 176 MB @ walk.
- Build: sm_89, system CUDA, `-dc -dlto`. Binaries live on LOCAL disk `/root/zkllm` (FUSE rule).
  Per-op selftests (e.g. `zkob_fc selftest` ~11 s) = the fast single-layer dev loop.

## Why hashes aren't enough (recorded for future me)
A plain hash/commitment is **binding but not hiding**: an adversary who guesses the activation can
hash it and confirm (guess-and-confirm; inputs/model partly known). Activation privacy needs
**hiding commitments** (blinded, e.g. Pedersen `g^x·h^r`) **+ a zero-knowledge proof** of the layer
relation. Both, not either. The Stage-D weight machinery already does exactly this for weights.

## Phases
**P0 — Rebuild env + baseline (this session).** Reconstruct `/root/zkllm` build + python env;
`zkob_fc selftest` green; record `stage_d_bench.py` single-FC baseline (plain + wpriv) prove/verify/
proof-size. [task #14, #15]

**P1 — Activation privacy on one layer (core deliverable).** [task #16]
Extend `zkob_wpriv.cuh` machinery from W-claims to X/Y activation claims on `zkob_fc`:
hiding (blinded) commitments on input & output activations; make the activation-touching sumchecks
ZK (committed round messages, already the F7 mechanism); route activation claims as Committed.
Boundary chaining: layer l's output commitment == layer l+1's input commitment (reuse blind or add
an equality proof — same idea as the EZKL phase-3 boundary binding). Validate: honest ACCEPT +
extend the D4 leak scan to activation functionals (CLEAN) + forgery loci + ZK re-randomization.

**P2 — Speed lever 1: Icicle GPU MSM (cheap, privacy-preserving).** [task #17]
Swap hand-coded G1 MSM/commitment for Ingonyama Icicle BLS12-381 (+NTT). Same field/scheme → privacy
untouched. Measure vs P0 baseline.

**P3 — Speed lever 2: small-field/hash PCS (the 10-100x ceiling).** [task #18]
Migrate commitment to small-field + hash-based PCS (Basefold/Binius). Pedersen homomorphism is lost
→ re-derive hiding + the linear-relation binding inside the sumcheck. Largest crypto lift; informed
by P2. Read Expander (open source) for fast-GKR techniques and reimplement here (license-clean).

## Honest scope
P1 is real crypto (weeks); P2 weeks; P3 months. Multi-session — suited to the headless/autopilot
loop. Each step gates on the selftest battery (never weaken soundness). The 3 protected headers are
not edited without full 13-TU re-validation. Privacy claims stay "hash-like→now real ZK on the
Pedersen path"; remind before any external ZK claim/writeup (see memory zk-ezkl-privacy-caveat).
