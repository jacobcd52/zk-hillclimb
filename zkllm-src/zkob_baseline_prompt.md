# Task: resurrect the REAL end-to-end integer prover (zkob/BLS) and produce an honest integer baseline

Context: we need honest "integerized model" ZKP numbers. Our recent comparison
used a synthetic estimate that an audit (zkllm-src/COMPARISON_AUDIT.md — READ IT
FIRST) found circular. But this repo contains a REAL end-to-end integer prover,
built BEFORE the fp8 work, with tailor-made integer gadgets: the zkob/BLS12-381
stack (zkLLM-derived; Pedersen + sumcheck + tlookup) plus the orchestrator walk
over real llama-68m. Your job: dig out its old results, rebuild it, rerun it at
shapes as comparable as possible to our fp8 sweep, and produce the corrected
integer baseline.

Environment rules (hard): work dir /workspace/projects/zk-hillclimb (repo) but
COPY sources to /root/zkob_build and build/run there (NEVER execute binaries
from /workspace — FUSE). Build via build_zkob.sh conventions (see repo root;
sm_89, system CUDA, -dc -dlto). ONE GPU job at a time. 41 GB host cap. No git
commit/push. No other claude agents. Long runs: setsid + oom_score_adj 1000.

## Part 1 — archaeology (read, don't run)

Collect ALL previously-measured zkob/orchestrator numbers with their configs:
- orchestrator/ORCHESTRATOR_REPORT.md: stage-1 walk timings (per-obligation
  table, 148.4 s prove for 2 llama-68m layers, MLP+norms only, attention
  UNPROVEN at that stage), proof sizes.
- STAGE_D_REPORT.md (repo root): single-FC benchmarks (plain vs weight-private).
- Any later end-to-end walk results: search repo .md files (and orchestrator/)
  for a "C2" baseline (~522 s prove / 27 s verify / 176 MB proof) and a
  "~17.7 min prove / 16.7 min verify" full llama-68m seq-1024 figure — find
  what scope each covered (attention proven or not? which stages waived?) and
  reconcile them.
- OVERNIGHT_SUMMARY.md if present; zkob_* selftest expectations.

## Part 2 — rebuild + rerun (measure, matched shapes)

1. Rebuild zkob binaries per build_zkob.sh (zkob_fc, ppgen, zkob_batchopen at
   minimum; also the rmsnorm/swiglu/rescale gadget binaries if separate).
   Verify: `zkob_fc selftest` ALL PASS before benchmarking.
2. Single-op benchmarks at shapes MATCHED to our fp8 sweep grid where the
   system allows (it may require specific pp sizes — ppgen as needed):
   - FC/matmul: (B,IN,OUT) matching our layer shapes at d=64..512, e.g.
     (64,64,64), (128,256,256), (64,512,2048), and the llama shape
     (16,768,3072) as anchor to STAGE_D_REPORT.
   - rmsnorm, swiglu, rescaling gadgets at matching widths/token counts.
   Report prove/verify seconds + proof bytes per op. State whether each run is
   weight-private/ZK or plain, and use the mode that matches "full ZK" closest.
3. If feasible without the missing llama data env (/root/int-model-env may not
   exist — check; reconstruct ONLY if cheap), rerun the 2-layer walk or a
   1-layer subset for an end-to-end anchor. If not feasible, say so and rely
   on the documented numbers.
4. Assemble "integer layer" totals at our sweep shapes from the zkob gadget
   measurements (matmuls + norms + swiglu + rescales for one transformer
   layer; state what attention costs and whether it is included — the old
   walk left attention unproven; if zkob has no attention gadget, construct
   the attention matmuls from FC calls and say so).

## Part 3 — the corrected comparison

- Table: fp8 zk=1 layer prove (from /root/zkllm/bench_sat.json) vs zkob
  integer layer at matched shapes; premium column.
- SUBSTRATE CAVEAT, quantified: zkob is BLS12-381 (256-bit; our old
  measurement: Goldilocks field mul ~11.9x faster; hash commit ~45x faster
  than Pedersen at 2^22). Give a stated-assumptions estimate of what the same
  integer pipeline would cost ported to the Goldilocks substrate (divide
  field-op-bound stages by the measured factors — label clearly as an
  ESTIMATE, not a measurement) — this brackets the honest premium from above
  (measured vs BLS zkob) and below (estimated vs a ported integer prover).
- Reconcile with COMPARISON_AUDIT.md's zkLLM-calibrated 10^2-10^3x claim: does
  the real zkob data support it?

## Deliverables

/workspace/projects/zk-hillclimb/zkllm-src/INT_BASELINE.md with parts 1-3
(numbers-first, sources cited, honest caveats); raw logs in /root/zkob_*.log;
/root/zkllm/zkob_baseline_done.flag with OK/PARTIAL + one-line summary.
Do not touch the fp8/P3 sources or benches. If your session nears its limit,
write the flag as PARTIAL and leave a handoff note at the top of INT_BASELINE.md.
