# Task: CONTINUE the scaling study (predecessor was OOM-killed mid-run)

You are continuing another engineer's nearly-complete work. Read these FIRST:
1. `/root/zkllm/scale2_prompt.md` — the original brief (all environment rules apply:
   binaries to /root, -Xcompiler -fopenmp, ONE GPU job at a time, no other claude
   agents, no git commit/push, 41 GB host cgroup cap, GPU 24 GB).
2. `/root/zkllm/SCALING_STUDY.md` — the predecessor's running log: the d-sweep
   diagnosis (done), the three-stacked-walls memory accounting (done), levers 1-5
   (IMPLEMENTED in /root/zkllm source: p3_zkc.cuh, p3_logup.cuh, p3_hawkeye.cuh,
   p3_batchopen.cuh, p3_basefold.cuh), speed levers (done, partially measured).

State when the predecessor died:
- Best verified: d256 seq=128 zk=1 with all levers = 221.2 s / 16.2 GB, proof
  BYTE-IDENTICAL to base (77.527 MB), verify_ok=1. Small-config guard clean.
- Last blocker seen on d256 seq=256: `device alloc failed at bo_encode:out
  (8589934592 bytes)` (/root/zkrun_d256s256_e5.log) — the batch-open encode of the
  v=28 class needs an 8 GB device output; likely needs chunked/host-staged encode or
  pool trim before it. There may have been later attempts — check the newest
  /root/zkrun_*.log files and binaries in /root (ls -t) before assuming.
- s256b16 (4096 tokens) attempts exist (zkrun_s256b16_e*.log) — check status.

CRITICAL OPERATIONAL RULE — the predecessor was killed by the cgroup OOM killer
while a 41-GB bench ran: launch EVERY heavy bench through an OOM-preference shield
so the kernel kills the BENCH, not you:
  bash -c 'echo 1000 > /proc/self/oom_score_adj; exec <bench cmd...>' > log 2>&1
and keep your own working set small (don't cat huge logs; use tail/grep).

## Remaining work (in order)

1. Finish d256 seq=256 zk=1 under 41 GB: fix the bo_encode device OOM (chunked
   encode / host staging / pool release — transcript-identical only), then run it
   with all levers. Target: BENCH verify_ok=1 line.
2. Run s256b16 (4096 tokens) zk=1 with all levers (+P3_PK_SPILL if needed).
   If it lands, try s128b64 (8192 tokens) once.
3. GATES (predecessor never ran them on the final source): 26-suite battery
   (run_battery.sh), compact teeth (run_compact_teeth.sh), forced-stream pairs at
   seq=64 (all combos: default / P3_SC5ZG_CAP / P3_SBLIND_MIN / both, with the new
   levers at their defaults) — proof_mb must equal 42.658 / 41.569 and verify_ok=1.
   Also one guard rerun each: d64 seq=256 (expect ~62 s / 4.9 GB) and d256 seq=128
   (expect ~221 s / 16.2 GB).
4. Complete SCALING_STUDY.md: fill the "(fills in)" cells and sections D results,
   E gates, F roadmap (finish the Binius bound paragraph from the measured stage
   shares; also keep the ≥8k-token mmap-spill roadmap note).
5. Append all new BENCH lines to /root/zk_scale_results5.log (full logs
   /root/zkrun_<tag>.log). Write /root/zkllm/scale2_done.flag: OK or
   PARTIAL <reason>. If a wall is genuinely unbreakable under the 41 GB cap,
   document it honestly with the RSS trace — do not fudge.
