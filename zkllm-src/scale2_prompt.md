# Task: memory + speed scaling study of the P3 full-ZK prover (measure, diagnose, implement, validate)

You are a senior CUDA/ZKP performance engineer working on OUR OWN defensive
verifiable-inference codebase (Goldilocks sumcheck+logUp+Basefold stack, "p3_*").
Work dir `/root/zkllm` (build tree; canonical repo `/workspace/projects/zk-hillclimb/zkllm-src`
— do NOT git commit or push; the coordinator reviews, gates and commits).
Environment rules (hard):
- Build binaries to `/root/` (NEVER execute binaries from /workspace — FUSE mount).
- Always `nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp -I. <x>.cu -o /root/<x>`
  (omitting -fopenmp silently makes the prover 4x slower).
- ONE GPU job at a time. NEVER launch other claude agents/workflows.
- Host memory: 41 GB container cgroup cap (read-only; exit 137 = OOM kill). GPU 24 GB.
- Long runs: `timeout 7200 ... > /root/zkrun_<tag>.log 2>&1`, sequential.

## Current state (2026-07-12, git ced1fa5+3f34c99 — read scale_debug_report.md and BENCH_ZK_AT_SCALE.md first)

`/root/p3_tb_c24r <seq> <d> <nh> <dh> <dff> <batch> <zk> <tables.bin>` = one composed
transformer layer, `/root/p3_model_bench3 <nlayers> <seq> <d> <nh> <dh> <dff> <vocab> <zk> <tables>`
= composed model. Tables: tables_ld6.bin (d=64), tables_ld8.bin (d=256), tables_ld9/ld10 exist.
P3_MEMLOG=1 gives RSS + "bo class" lines; P3_ZKPROF=1 gives STAGES + "hwl prof" lines.
Verified zk=1 results: seq1024/d64 298.8s/31.3GB; d256/seq128 259.5s/29.0GB;
b16/seq128 (2048 tok) 269.4s/30.1GB; models N=1/2/4. ZK premium ~2-2.2x over zk=0.

Known walls and facts:
- d=256 seq=256 and 4096+-token configs die exit=137: the P=2^27 dff chain's ~24 GB
  of committed HOST witness columns land on a ~23 GB baseline. This is the #1 wall.
- STAGES at d256/seq128 zk1: mm=89.8 lug=83.4 batch=80.1 (of 259.5s) vs d64/seq256:
  mm≈? lug≈? (measure). mm+lug+batch dominate everywhere.
- Native forward at these toy dims is launch-bound flat (~2.2 ms batched mode), so
  overhead ratios inflate with size even where the prover is linear in work.
- A Binius/GF(2^128) stack exists (p3_binius_*.cuh, ~20x faster than GL on the
  standalone hawkeye product gadget, 46x less committed data) but is NOT integrated
  into the layer prover; a naive swap is UNSOUND (cross-field binding seams). Do NOT
  attempt that integration here — but DO quantify (on paper, from measured stage
  shares) what it would buy, as a roadmap item.

## Objectives, in priority order

1. **Diagnose overhead growth with model size** (user question). Run a d-sweep at
   fixed tokens (e.g. seq=128 b=1, d in {64,128,256} with matching tables/dff=4d)
   zk=1 + zk=0, collect STAGES + hwl prof + bo class + RSS, and answer with numbers:
   which stages grow superlinearly in d and why (P=tokens*d*dff products? logUp table
   sizes? batch-open class count/size? per-chain fixed costs?). Separate "prover does
   more work" from "forward baseline is flat" — compute prove-per-product and
   prove-per-token trends. Also sample `nvidia-smi --query-gpu=utilization.gpu,memory.used
   --format=csv -l 1` during one big run to show GPU idle share (suspected large:
  host<->device round-trips and host folds).
2. **Memory levers to break the witness wall** (goal: d=256 seq=256 zk=1 and a
   4096-token config proving under 41 GB). Study where the ~24 GB committed-column
   retention comes from (packed-witness cpk machinery in p3_zkc.cuh/p3_hawkeye.cuh —
   the materializer already rebuilds packed columns transiently; find what is NOT
   packed or is retained when it need not be: logUp helper columns? ledger (PLedger/
   keep deques)? per-chain aug columns? opening-phase rematerialization inputs?).
   Candidate directions (verify, don't assume): commit-then-drop with rematerialize-
   on-open (prior art exists — "strCol" in p3_batchopen and the flush/remat levers),
   slicing the giant dff matmul into K-chunks with chained partial claims (changes
   transcript — only if a gadget for it already exists; otherwise prefer
   transcript-identical levers), spilling committed columns to /workspace (FUSE ok
   for DATA) or local disk with mmap.
3. **Speedups.** The three dominant stages are mm (hawkeye zero-checks), lug (logUp),
   batch (batch-open). Profile inside them (hwl prof fields: cwit/zcdp/zcdg/chain/
   gsum/zcdo; bo timing lines) and implement what is safe and big. Known cheap ideas
   to evaluate: reduce host<->device ping-pong in sc5z_gpu streamed/resident paths;
   GPU-side round-message reduction batching; overlapping witness gen with commits
   (streams); eq-table rebuild caching in batch-open rounds; OpenMP thread-count
   tuning at big N (p3bf::nthr). Measure each candidate's win on ONE mid config
   (e.g. d64 seq256 zk1) before adopting.
4. **Implement the winning levers** (memory first, then speed), keeping proofs
   VALID: after each change run `/root/p3_tb_c24r`-style quick checks
   (seq=64 zk1 verify_ok=1 + proof_mb unchanged if the lever claims transcript-
   identity), and at the end the full gates:
   `bash run_gates.sh`-equivalent = 26-suite battery (rebuild .bat tests), compact
   teeth, forced-stream pairs (P3_SC5ZG_CAP, both P3_SBLIND_MIN modes), then rerun
   the headline configs incl. the newly-unlocked ones. Levers that change the
   transcript are acceptable ONLY with full battery + hiding teeth green and a note.

## Deliverables

1. `/root/zkllm/SCALING_STUDY.md`: (a) the overhead-growth diagnosis with measured
   tables (per-stage seconds vs d; per-product and per-token normalization; GPU
   utilization trace summary), (b) memory accounting of the witness wall (what holds
   the 24 GB, with numbers), (c) ranked lever list with measured or bounded wins,
   (d) what was implemented + gate results, (e) roadmap items NOT done (incl. the
   Binius integration estimate from measured stage shares).
2. Source changes in /root/zkllm (smallest correct diffs, matching style).
3. New BENCH lines appended to /root/zk_scale_results5.log for every config you
   prove (full logs /root/zkrun_<tag>.log).
4. `/root/zkllm/scale2_done.flag` containing one line: OK or PARTIAL <reason>.

Budget guidance: this is a long autonomous task (hours). Prefer measured small steps
over big rewrites; keep a running log in SCALING_STUDY.md as you go so partial
progress survives interruption. If you approach a session/token limit, write the
flag file with PARTIAL and a handoff paragraph at the top of SCALING_STUDY.md.
