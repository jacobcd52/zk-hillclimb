# P3 — small-field + hash-commitment migration (speed lever 2)

Goal: shed the two dominant costs of the BLS12-381 prover while keeping weight+activation
privacy: (#1) 256-bit field for int8 data, (#2) elliptic-curve commitments/openings.
Strategy: Goldilocks field + Basefold/FRI hash commitment. Built as an isolated module
(p3_*), existing prover untouched, validated stage by stage.

## P3.1 — Goldilocks field core  [DONE 2026-06-21]
p3_goldilocks.cuh (field), p3_field_bench.cu (correctness + throughput).
- Host correctness: 2M random mul/add/sub + 100k inverses + roots of unity order checks
  cross-checked vs independent __int128 reference -> ALL PASS.
- GPU throughput (RTX 4090, 4.19e9 multiplies):
    Goldilocks 64-bit:   5.49 ms   763.6 Gmul/s
    BLS12-381 256-bit:  65.30 ms    64.2 Gmul/s
    => 11.9x faster per multiply.  (validates culprit #1)

## P3.2 — NTT + Merkle commit; commit-cost vs EC Pedersen   [DONE 2026-06-21]
p3_ntt.cuh (Goldilocks NTT), p3_merkle.cuh (device SHA-256 + GPU Merkle), p3_commit_bench.cu.
- NTT correctness: naive-DFT cross-check (n=8) + forward/inverse round-trip (n=2^16) -> PASS.
- Commit cost, same N=2^22 (4.19M) elements on RTX 4090:
    hash-PCS (RS blowup2 NTT -> 2^23 + SHA-256 Merkle):  15.5 ms
    EC Pedersen (Pippenger MSM, 2048x2048):             700.9 ms
    => 45.1x faster commit.  (validates culprit #2; SHA-256 is conservative vs Poseidon)
## P3.3 — FRI low-degree test (proximity engine)   [DONE 2026-06-21]
p3_fri.cuh (host: Merkle, fold, Fiat-Shamir prove/verify), p3_fri_selftest.cu.
- Subgroup domain, k=logN fold rounds, Q Merkle-authenticated queries.
- Selftest 11/11: honest accept (3 sizes); reject on tampered value / path / final /
  mid-round / root / wrong-seed; high-degree word rejected (final not constant).
- Host-side for correctness clarity; GPU acceleration deferred to perf pass (P3.6).

## P3.3b/P3.4a — Basefold multilinear evaluation opening   [DONE 2026-06-21]
p3_basefold.cuh (eq-weighted sumcheck coupled to the MLE codeword fold), p3_basefold_selftest.cu.
- Generalized the fold to coeff (FRI LDT) vs MLE (Basefold) via fold_pair; shared check_queries.
- Opens h(z)=c~(z) for committed coeff vector c: v-round sumcheck of sum_b c[b]*eq(b,z) with
  challenges = MLE fold challenges; tie  claim == C*eq(alpha,z),  C = final folded constant.
- Selftest 10/10: honest opens (3 sizes) accept; reject on wrong value / tampered sumcheck /
  codeword / path / final / opening point.
- Soundness ~2^-58 (base-field challenges); degree-2 extension is the flagged production change.

## P3.4b — sumcheck matmul argument for the FC layer (Y = X.W)   [DONE 2026-06-21]
p3_matmul.cuh, p3_matmul_selftest.cu.
- Y~(r_i,r_k)=sum_j X~(r_i,j)W~(j,r_k): sumcheck over IN contraction vars -> 3 Basefold opens
  (X@(r_i,r_j), W@(r_j,r_k), Y@(r_i,r_k)), tied  final == X~*W~, initial == Y~.
- Selftest 8/8: honest accept (3 shapes); reject on wrong product / tampered sumcheck /
  opened value / opening codeword.
- This is the complete INTEGRITY core (not yet zero-knowledge / private-evaluation).

## P3.5 — ZK/hiding + weight & activation privacy   [next; the hard part]
## P3.6 — GPU-accelerate + end-to-end FC bench vs BLS
## P3.4 — sumcheck over Goldilocks + FC matmul argument
## P3.5 — ZK/hiding + weight & activation privacy (hard requirement)
## P3.6 — wire FC-layer prover + selftest battery + end-to-end bench


## Integrity-core summary (2026-06-21)
Validated end to end (selftests green at every stage). Real protocol numbers (host,
B=16 IN=1024 OUT=16): verify 14 ms, proof 657 KB  vs  BLS verify ~1.7 s, proof 176 MB.
Prove wall is dominated by the placeholder host Horner encoder (5.6 s/poly); GPU NTT
(P3.2) does the same encode in ms and host Merkle is 71 ms, so protocol prove cost is ~1 s
host and fully GPU-accelerable. Per-primitive speedups already prove the thesis: 12x field,
45x commit. Remaining: P3.5 ZK/privacy (hard), P3.6 GPU end-to-end number.


## End-to-end (integrity, GPU encode + host Merkle/fold) 2026-06-21
Same shape B=16 IN=1024 OUT=16, R=2, Q=32:
  prove  461 ms   (GPU NTT encode + host Merkle/fold; was 12.2 s with host Horner encode)
  verify  14 ms
  proof  657 KB
Reference BLS single-FC prove ~3.7 s / verify ~1.7 s / proof 176 MB.
=> already ~8x prove, ~120x verify, ~270x proof, with Merkle+fold STILL ON HOST.
GPU Merkle-with-paths + GPU fold (P3.6) remove the remaining host bottleneck.
gpu-encode commitment is byte-identical to host-encode -> openings unchanged.
