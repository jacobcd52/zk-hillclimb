# Task: produce a SUCCINCT technical write-up of the whole project, with plots embedded

Write a single, succinct, accurate technical report that ties together everything in
/workspace/projects/zk-hillclimb/. Target ~4-6 pages (succinct — lead with findings, don't
pad). Pull EVERY number from the committed source docs; do NOT invent or estimate — if a
number isn't in a doc, omit it or say "not measured". Embed the existing PNG plots.

## Read these source docs for the real numbers
- THREAT_MODEL_NOTES.md (the threat model: datacenter proves served tokens came from a
  committed model; integerization error = covert-channel capacity; greedy decoding pinned)
- BACKEND_DECISION.md (zkLLM was prover-only/leaky/slow; we built the rest; nobody else has
  real weight privacy; we keep ours + steal batching architecture)
- DIFR_BASELINE_NATIVE.md + DIFR_FAITHFUL_V1.md (the DiFR decomposition: integerization floor
  1.35e-6 nats; baseline 8.99 -> faithful 0.0156 for +21% prove; the gap is architecture not
  integerization)
- CAPACITY_CORRECTED.md + CAPACITY_TOPK.md + CAPACITY_FP16.md (the b/K capacity sweeps:
  baseline ~12.5, faithful 0.36, codebook 0.23 bits/tok vs FP8; the FP16 reorder - faithful
  0.065, codebook 0.37; the U-curves; ~70% of capacity is the model's own near-ties)
- BUFFER_FPR.md (mu=0.306 bits/tok afforded; cumulative variance ~7.5x i.i.d. due to
  within-prompt autocorrelation, N_eff~N/7.5; buffer for FPR<1e-10 falls ~1/sqrt(N);
  per-token buffer < mu only after N~4-16k tokens; single-prompt audits certify nothing)
- TRANSPORT_REBUILD_DESIGN.md + the STAGE_*_REPORT.md / REBUILD_AUDIT.md headlines (the
  efficient rebuild: verify 999s->27s/36.9x, prove ~636s, weight-private, audited SOUND;
  proof 176MB; S1 proof-shrink tried + dropped - 3x smaller but 4.4x slower verify)

## Structure (succinct)
1. One-paragraph executive summary (what the project is + the 3-4 headline results).
2. Threat model & framing (covert capacity = integerization gap; afforded vs detected;
   greedy/served-tokens-from-quantized-model setup).
3. The ZKP system: what zkLLM was vs what we built (full prove/serialize/verify, weight
   privacy, zero-advice obligations closing covert channels); the efficiency rebuild
   (27s verify, the table of before/after); audited SOUND. Keep it tight - a paragraph +
   the before/after table, not a driver-by-driver list.
4. Results (the meat, with plots):
   - DiFR decomposition: proof machinery is ~free (1.35e-6); the gap is architecture; the
     8.99->0.0156 Pareto move (embed pareto_v1.png).
   - Covert capacity sweeps: the U-curves (embed capacity_combined.png), the scheme ordering,
     the FP16 reorder (embed capacity_fp16_combined.png), the "~70% is the model's own
     near-ties" diagnosis.
   - The treaty / buffer analysis: afforded capacity, the autocorrelation finding (embed
     buffer_vs_N.png and/or buffer_rt_dist_acf.png), the operational bottom line (cumulative
     auditing over thousands of tokens needed).
5. Open questions / next steps (brief): excess-over-honest capacity (defuses the DoS +
   strips intrinsic uncertainty), the sequential decision rule, tighter 1e-10 data, scaling.

Embed plots with relative paths (e.g. ![](pareto_v1.png)) - they live in the repo root.
Available PNGs: pareto_v1.png, capacity_combined.png, capacity_fp16_combined.png,
capacity_topk_ksweep.png, buffer_vs_N.png, buffer_rt_dist_acf.png, difr_delta_curve.png.

Deliverable: /workspace/projects/zk-hillclimb/WRITEUP.md. Accurate, succinct, plot-rich,
honest about caveats (afforded-not-detected; the 1e-10 extrapolation; the dropped S1).
No git commits; no pushes; int-model-approximation untouched. READ-ONLY except WRITEUP.md.
