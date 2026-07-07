# Task: measure the real attention-score range in the llama-68m integer pipeline

Context: we are designing a ZK proof for the softmax in an integerized llama-68m forward
pass. The proof uses a fixed exp-lookup table whose domain assumes the real-valued
attention scores satisfy |QK^T_real| < 1024 per head (estimate was ~576 worst case).
Before freezing the table domain we need the MEASURED maximum.

## What to do
1. Find the pipeline: `/workspace/projects/int-model-approximation` contains
   `m68-pipeline.py` (search for it). Read its attention section: per head it computes
   `A = to_int64(Q @ K.transpose(-2,-1), VALUE_LOGSF)` with VALUE_LOGSF=16, i.e. A is the
   real score times 2^16. There are 2 layers × 12 heads, seq 1024.
2. DO NOT modify anything inside that repo, and never push it anywhere. Copy whatever you
   need to /tmp/scores-measure/ and work there. Figure out how the pipeline is normally
   invoked (look for a README, driver scripts, or how it loads its inputs — there may be
   existing dumped tensors like temp_Q.bin; if the pipeline needs the HF model
   JackFram/llama-68m it is likely already in the HF cache, HF_HOME is set).
3. Instrument the copied pipeline to record, for every layer and head, BEFORE any
   max-subtraction or masking: max(A)/2^16, min(A)/2^16, and max(|A|)/2^16 — over ALL
   positions (masked positions included). Use whatever real input the pipeline normally
   runs with (its standard prompt/sample input). If it supports multiple inputs easily,
   run 2-3 different inputs.
4. Also record, with the causal mask applied (j ≤ i only): the per-(layer,head) max and
   min of A/2^16 — so we can see whether masked positions are more extreme.
5. The GPU is an RTX 4090 (24 GB); the model is tiny (68M params). A foreground run is fine.

## Deliverable
Write `/workspace/projects/zk-hillclimb/SCORES_RANGE.md`:
- a table: layer, head, max score, min score, max |score| (all positions), and the
  masked-only max/min;
- the global max |score| and the margin factor vs 1024 (the table-domain bound) and vs 512
  (the 2x-margin threshold: if global max |score| > 512 the design's domain knob must turn);
- exactly which input(s) the numbers come from, and the exact commands you ran;
- any surprises (NaNs, scores hugging the bound, asymmetry).
Be precise and honest; if you cannot run the pipeline, say exactly what is missing rather
than guessing numbers.
