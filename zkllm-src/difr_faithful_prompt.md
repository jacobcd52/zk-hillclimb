# Task: measure end-to-end DiFR of the faithful-arch-v1 chain (the before/after yardstick)

The faithful-arch-v1 submission is validated (orchestrator run /root/zkorch/stage3v2-fa:
65/65 ids ACCEPT). Measure its DiFR exactly the way DIFR_BASELINE_NATIVE.md measured the
baseline — same protocol, same 8 dolly prompts, same seed 20260611, same teacher — so the
two numbers are directly comparable. Predicted: difr_mean ≈ 0.008 (STAGE3 §5.2; band
0.004-0.016).

How:
1. Read /workspace/projects/zk-hillclimb/DIFR_BASELINE_NATIVE.md and the scripts in
   /workspace/projects/zk-hillclimb/measure/ (int_chain.py, difr_baseline.py,
   probe_semantics.py, validate_chain.py) — your job is the faithful-arch analog.
2. Extend measure/int_chain.py (new class or flag — keep the baseline path intact) with
   the faithful semantics per STAGE3_FAITHFUL_DESIGN §4: per-head exact rowmax max-shift
   + temperature-8 softmax8 (the §4.3 integer spec: E8 table, masked sentinel, the
   rounding bracket P), headmerge plain concat (NO line-157 permutation), o_proj
   fc+rescale between merge and skip, lm_head/final-norm as in stage 3.
3. VALIDATE the faithful chain implementation byte-exactly against the driver-emitted
   witness files in /root/zkorch/stage3v2-fa/data/ (the validate_chain.py pattern:
   registered input -> every chain tensor -> final logits; 100% byte-equality required —
   if any tensor mismatches, STOP and report the first divergence precisely).
4. Score: chain logits vs the FP8 teacher per the harness protocol (difr_baseline.py
   replicated verbatim) on the same 8 prompts/seed; also the chain-vs-float-replica
   decomposition row (decompose_difr.py pattern, replica = faithful architecture).
5. Deliverable /workspace/projects/zk-hillclimb/DIFR_FAITHFUL_V1.md: headline table
   (faithful difr_mean/p99/top1/flips vs the baseline's 8.988/23.996/0.038), the
   decomposition row, the §5.2 prediction vs measured, scripts/commands, honest caveats
   (same seed caveat as baseline: documented, not a coordinator round).
GPU free. No git commits; no GitHub; int-model-approximation read-only.
