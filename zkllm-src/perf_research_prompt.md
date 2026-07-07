# Task: web survey — how do the fastest zkML/proof systems get their speed, and what can we steal?

You have WebSearch and WebFetch. Research the state of the art (2024-2026) in FAST
zero-knowledge proving for ML inference and produce a cited, numbers-first report.

OUR SYSTEM (context for relevance): zkLLM-derived CUDA prover on an RTX 4090 — sumcheck +
logUp lookups over the BLS12-381 scalar field (256-bit), Pedersen/Hyrax row commitments,
linear-time IPA openings, Fiat-Shamir. Measured: llama-68m forward pass (seq 1024) proves
~17.7 min, verifies ~16.7 min (~10^5x plain inference). Constraints on any redesign:
(a) deterministic commitments (byte-equality chaining must survive), (b) a way to bind
linear relations between committed tensors (today: Pedersen homomorphism), (c) weight
privacy (commit-and-prove).

ANSWER THESE, with concrete numbers and primary sources (papers/repos/benchmarks; flag
self-reported vs reproduced):
1. Fastest sumcheck/GKR provers (Polyhedra Expander, Lagrange DeepProve, JOLT, others):
   reported speeds on transformer/matmul workloads, hardware, and WHICH design choices
   buy the speed (small fields M31/BabyBear/Goldilocks vs 256-bit; hash commitments
   Brakedown/Basefold/Binius/FRI vs elliptic curve; GKR layering vs per-op sumchecks).
2. Small-field + hash-commitment migration: measured prover speedups vs 256-bit EC
   baselines; verification cost (succinct?); existing CUDA implementations.
3. GPU ZK libraries (Ingonyama Icicle, cuZK, GZKP, Tachyon, sppark): consumer-GPU
   MSM/NTT/sumcheck support, tensor-core field-arithmetic tricks, measured speedups.
4. zkLLM (Sun et al., CCS 2024) itself: the PAPER's prover times per model size/hardware
   — so we can separate protocol overhead from our additions.
5. Faster lookup arguments than logUp on GPU (cq, Lasso, LogUp-GKR): measured numbers.
6. Getting VERIFICATION to seconds: recursion/wrapping (Plonky3/SP1-style), FRI verify
   costs, batch openings — and the prover-side cost of wrapping a sumcheck transcript.

Deliverable: /workspace/projects/zk-hillclimb/PERF_SURVEY.md — per question: findings
with numbers + citations (URLs), then a final "STEALABLE TECHNIQUES RANKED" section:
each candidate with expected speedup for OUR system, what breaks (constraints a-c),
and rough port effort. Honest about uncertainty; no padding.
