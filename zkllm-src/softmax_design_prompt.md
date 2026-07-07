# Task: produce the FINAL design document for `zkob_softmax.cu` (design only — NO implementation)

You are designing the last missing proof obligation for a ZK-verified llama-68m forward pass.
Output: a design document so complete that a separate implementer can write the driver without
making a single design decision. Model it on the "IMMEDIATE NEXT STEP: write zkob_rmsnorm.cu"
section of `/workspace/projects/zk-hillclimb/HANDOFF.md` — that is the bar for completeness
(exact proof obligations, exact FS schedule, exact CLI, exact selftest evil modes, exact
real-scale parameters, exact numeric bounds with proofs they fit).

## Read first (all of these)
1. `/workspace/projects/zk-hillclimb/HANDOFF.md` — overall context; the softmax sketch is in
   "After rmsnorm, remaining for task #12", item (a). Your design should follow that sketch
   unless you find a concrete reason it cannot work (then document the problem AND the fix).
2. `/workspace/projects/zk-hillclimb/submissions/baseline-native/PHASE0_NOTES.md` — ALL pinned
   conventions: FS rules, IPA/registration layout, logUp lookup layout, Montgomery rules,
   chaining via byte-identical commitment files, known machinery in zkob_lookup.cuh.
3. The integer pipeline: find it with `ls /workspace/projects/int-model-approximation/` and
   locate `m68-pipeline.py` (search for it if not at top level). Read the ATTENTION section
   carefully: exactly how scores → softmax probabilities are computed in integers (scales,
   the max-shift, the exp table/mapping, the row-sum, the inverse, rescales). The proof must
   bind EXACTLY that computation. Quote the relevant pipeline lines in your design and state
   every scale (2^k) explicitly. DO NOT modify anything in that repo, and never push it anywhere.
4. `/root/zkllm/zksoftmax.cu` + `.cuh` (upstream zkLLM) — how upstream handles softmax; we do
   NOT have to reuse it, but its shift-invariance trick and table bounds are relevant.
5. `/root/zkllm/zkob_glu.cu` (mapping-lookup pattern), `/root/zkllm/zkob_rmsnorm.cu` (bracket
   pattern for inverse advice, quartic sumcheck usage), `/root/zkllm/zkob_lookup.cuh` +
   `/root/zkllm/vrf_common.cuh` (available machinery — fs_phase1/fs_phase2 logUp, fs_hadamard,
   fs_quartic, open_prove/open_verify, build_eq_tensor, k_bcast_rows, host G1 helpers).

## Design constraints (pinned, non-negotiable)
- Everything the verifier checks must be anchored in commitments + openings + public constants.
  Any prover "advice" (max-shift mx, inverse of row-sum, etc.) must be BOUND with explicit,
  quantified tolerance; state the covert-channel capacity in bits/row that the tolerance leaves,
  like the rmsnorm ±1 bracket analysis (≤ log2(3) bits/row).
- Shift advice mx: per HANDOFF it need not be the exact row max — softmax is shift-invariant.
  But the table domain must bound what dishonest shifts can do: analyze precisely what freedom
  a dishonest mx gives (saturation/clipping at table edges) and what constraint (range proof on
  z − mx, or on mx itself) eliminates or bounds it. Quantify leftover freedom in bits.
- Attention is per-head: scores are B=1024 rows × 1024 cols per head, 12 heads × 2 layers.
  Decide and justify: one obligation instance per head (like zkob_fc per-head) — keep it simple.
- Causal mask: check how m68-pipeline.py applies it (masked positions). The proof must handle
  masked entries exactly as the pipeline does. State explicitly how mask positions enter the
  lookup/table and the row-sum.
- Reuse existing validated machinery wherever possible; NO new G1 CUDA kernels (known
  miscompilation); new Fr kernels allowed but must follow the Montgomery convention
  (mont-ify all factors except one). Prefer no new kernels at all if achievable.
- All FrTensors PLAIN form. Bounds analysis for every committed quantity (must fit field,
  must fit the chosen limb/lookup widths; show the arithmetic like the rmsnorm P1 < 2^80 proof).
- One transcript per obligation; exact FS schedule listed absorb-by-absorb like the rmsnorm one.
- Chain interface: int64 unpadded chain files in, int64 unpadded out; which commitments must be
  byte-identical with neighbor obligations (scores from zkob_fc; output P feeding the V-matmul).
- CLI signatures for prove/verify/selftest in the same style as the other drivers.
- Selftest plan: small honest case; a semantic evil mode for EVERY check (state which check
  catches each and why nothing else does); byte tampers per proof file; real-scale case
  (B=1024 per head) with expected rough cost (extrapolate from: glu mapping lookup at
  B=1024,C=3072,table 2^22 → prove 11.4s; rmsnorm at B=1024,C=768 → prove 9.4s).

## Deliverable
Write `/workspace/projects/zk-hillclimb/SOFTMAX_DESIGN.md`. Sections:
1. Pipeline semantics (quoted from m68-pipeline.py, every scale explicit).
2. Statement to prove (the exact integer relation, including mask and tolerances).
3. Advice-binding analysis (mx and inverse: tolerance, leftover covert bits/row, totals
   per forward pass over 2 layers × 12 heads × 1024 rows).
4. Proof obligations (numbered, like HANDOFF rmsnorm §1-7).
5. FS schedule (absorb-by-absorb).
6. Numeric bounds (with arithmetic).
7. CLI + chain files + which commitment byte-equalities the orchestrator checks.
8. Selftest plan (evil modes + tampers + real-scale).
9. Open questions / risks (anything you could not fully pin down — be honest; an explicit
   open question is worth more than a hidden assumption).
This is a READ-ONLY task except for writing that one file. Do not write any code.
