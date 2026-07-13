# Task: improvement-loop iteration — implement the roadmap, then propose the next round

You are a senior CUDA/ZKP engineer improving OUR OWN defensive verifiable-inference
codebase (Goldilocks P3 stack). This is one iteration of an ongoing improvement
loop; a coordinator reviews, commits, and relaunches between iterations.

Read FIRST, in order:
1. `/root/zkllm/scale2_prompt.md` — environment rules (they all apply: binaries to
   /root, `nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp -I.`, ONE GPU job at
   a time, no other claude agents, no git commit/push, 41 GB host cgroup cap, 24 GB
   GPU).
2. `/root/zkllm/ROADMAP.md` — the work list. You implement it and extend it.
3. `/root/zkllm/SCALING_STUDY.md` — current performance baseline + stage profiles.
4. `/root/zkllm/IMPROVEMENT_LOG.md` if it exists — what previous iterations did.

OPERATIONAL RULES (learned the hard way — follow exactly):
- Launch every heavy bench DETACHED and OOM-SHIELDED so the kernel kills the bench,
  never you, and it survives you exiting:
    setsid nohup bash -c 'echo 1000 > /proc/self/oom_score_adj; exec env <ENV> timeout 7200 <cmd>' > /root/zkrun_<tag>.log 2>&1 &
  then poll its log with tail/grep (never cat whole big logs).
- If YOU are about to run out of session/tokens: write the flag file (below) with
  PARTIAL and update IMPROVEMENT_LOG.md first. Never leave undocumented state.
- Reference binaries that must keep working: /root/p3_tb_s2 (layer bench),
  /root/p3_model_bench3 (model bench). Name your new build /root/p3_tb_i<N>.

## The iteration

1. Pick up EVERY item in ROADMAP.md marked TODO, in rank order. For each:
   a. Implement the smallest correct version in /root/zkllm source.
   b. Quick identity check: seq=64 pairs (default + P3_SC5ZG_CAP=800000000 +
      P3_SBLIND_MIN=10 + both) — verify_ok=1 and proof_mb byte-identical
      (42.658 / 41.569) for transcript-identical levers. Transcript-CHANGING
      items additionally require the FULL battery + compact teeth + the ZK
      hiding suites green before keeping them (and say so in the log).
   c. Measure the win on the relevant config (d256 s128 for speed levers:
      baseline 214.8 s / 16.2 GB; the wall config for memory levers). KEEP if it
      wins ≥3% time or ≥10% memory or unlocks a config; otherwise REVERT and
      record why (keep the diff in a .rej file for the record).
   d. Update ROADMAP.md status + IMPROVEMENT_LOG.md (append an "## Iteration N —
      <item>" section with the measured numbers) as you go, not at the end.
2. After all items: run the FULL gates (bash run_gates2.sh — check
   /root/gates2_result.log for BATTERY: ALL GREEN, COMPACT: OK, identity pairs,
   guards) and rerun the headline configs that your changes affect
   (d256 s256; s256 b16; s128 b64 if the disk-backed store landed; 16384 tokens
   if 8192 landed). Append BENCH lines to /root/zk_scale_results6.log.
3. PROPOSE the next round: study the new profiles (STAGES + hwl prof + bo class +
   lu group lines from your final runs) and append to ROADMAP.md "Proposed" every
   improvement you can defend with numbers: estimated win, mechanism, risk
   (transcript-identical or not), and a confidence. Include honest "this is
   near-exhausted" statements if a stage has no meaningful headroom left. Rank them.
4. Write `/root/zkllm/impl_loop_done.flag`: first line OK or PARTIAL <reason>;
   second line a one-sentence summary; third line MAJOR or MINOR — your judgment
   of whether the NEW proposals in ROADMAP.md are collectively major (any single
   item ≥10% end-to-end speed or ≥20% memory or unlocks a config class) or minor.

Honesty rules: never fake or extrapolate a number; failed runs are reported as
failures with their logs; ZK claims only with the hiding gates green. If an item
is infeasible, mark INFEASIBLE with the mechanism, don't force it.
