# Task: covert-capacity sweep with an FP16 (full-precision) reference instead of FP8 — the sanity check

The capacity sweeps so far used the FP8 teacher (bf16 base + 14 FP8Linear, fp8_scaled_mm)
as the reference model M_true. The user wants the SAME analysis with a full-precision
**FP16/bf16** reference (NO FP8 linear swap) — the genuine "true" model, since FP8 is itself
a lossy quantization. Expected: faithful's capacity drops a lot (faithful M_int = integerized
bf16, so it's CLOSE to fp16); codebook may RISE (codebook M_int = integerized FP8, so vs
fp16 it inherits the FP8-vs-fp16 gap). The reordering is itself an interesting result.

Read: /workspace/projects/zk-hillclimb/CAPACITY_CORRECTED.md (the corrected-orientation
sweep + its scripts in capacity/ and measure/), DIFR_BASELINE_NATIVE.md (the teacher setup).
The capacity formula, b-sweep, K-sweep, 5-term top-K breakdown, and limit self-checks are
ALL UNCHANGED — only the reference model changes from FP8 to FP16/bf16.

## Do
1. Produce FP16/bf16 reference logits on the SAME 8 dolly prompts + seed 20260611: load
   JackFram/llama-68m in bf16 (or fp16) with NO FP8Linear swap (i.e. the plain model,
   IMA_TEACHER_KERNEL unset / standard matmul) — confirm exactly how the measure pipeline
   selects the teacher kernel and produce the no-FP8 variant. Document precisely what "FP16
   reference" means here (bf16 model as-loaded vs a forced fp16 cast) and whether nonlinears
   are bf16 (match the existing protocol's choices, just drop the FP8 linears).
2. Use the CORRECTED orientation (reference = the model the verifier checks against = FP16;
   served = M_int's behavior — but here M_int is the SERVED quantized model and FP16 is the
   reference, per the threat-model fix in CAPACITY_CORRECTED.md). Margins computed under the
   FP16 reference's post-Gumbel logits, for each served scheme's tokens.
3. Recompute for all 3 served schemes (baseline, faithful, codebook): min_b C(b) simple +
   top-K (K=16 and the best-K), the 5-term breakdown at the top-K optimum, argmax-agreement
   vs FP16, and the limit self-checks (b=0, b->inf). Reuse the existing per-position dump +
   sweep machinery; do NOT re-train/re-prove.

## Deliverable
/workspace/projects/zk-hillclimb/CAPACITY_FP16.md: headline table FP16-ref vs FP8-ref side by
side (per scheme: min capacity, argmax agreement, and the delta), the 5-term breakdowns, the
U-curve + K-sweep plots, and a short verdict: did faithful drop / codebook rise as predicted?
did the schemes reorder? Honest about what "FP16 reference" precisely is. Scripts under
capacity/. /root/int-model-env/bin/python. No git commits; no pushes; int-model-approximation
READ ONLY.
