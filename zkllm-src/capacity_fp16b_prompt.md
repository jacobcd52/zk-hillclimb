# Task: finish the FP16 capacity sweep (dumps mostly done; need baseline + full analysis + report + plots)

A prior agent produced the FP16/bf16-reference per-position dumps for FAITHFUL and CODEBOOK
(/workspace/projects/zk-hillclimb/capacity/capacity_dump_fp16_{faithful,codebook}_seed20260611.npz)
and the scripts (capacity_dump_fp16.py, capacity_analyze_fp16.py, fp16_ref_logits.py) but
stopped before: (a) the BASELINE dump (see capacity/dump_fp16_baseline.log for why it didn't
finish — fix + rerun, or document the blocker), (b) the full b-sweep / K-sweep / 5-term top-K
breakdown analysis, (c) writing CAPACITY_FP16.md + plots.

Partial results already confirmed (margin means vs bf16): faithful 0.000486 (p0 0.0107, huge
drop from FP8-ref), codebook 0.0163 (p0 0.0665, rose) — schemes reorder, as predicted.

Do:
1. Finish/repair the baseline FP16 dump (capacity/dump_fp16_baseline.log shows the issue).
2. Run the SAME analysis as CAPACITY_CORRECTED.md (min_b C simple + top-K K=16 + best-K, the
   5-component breakdown at the top-K optimum, argmax agreement, b=0/b->inf limit checks) on
   all 3 schemes against the FP16/bf16 reference, using the existing dumps + capacity_analyze_fp16.py.
3. Write /workspace/projects/zk-hillclimb/CAPACITY_FP16.md: headline table FP16-ref vs FP8-ref
   side-by-side per scheme (min capacity simple + top-K, argmax agreement, delta), the 5-term
   breakdowns, U-curve + K-sweep plots (capacity_fp16_*.png), and the verdict: faithful dropped /
   codebook rose / did they reorder, with the precise "what FP16 reference means" note.
Reuse existing dumps; do NOT re-run the model except for the missing baseline dump.
/root/int-model-env/bin/python. No git commits; no pushes; int-model-approximation READ ONLY.
