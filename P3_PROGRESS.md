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

## P3.2 — NTT + Merkle commit; commit-cost vs EC Pedersen   [in progress]
## P3.3 — Basefold/FRI low-degree test + evaluation opening
## P3.4 — sumcheck over Goldilocks + FC matmul argument
## P3.5 — ZK/hiding + weight & activation privacy (hard requirement)
## P3.6 — wire FC-layer prover + selftest battery + end-to-end bench
