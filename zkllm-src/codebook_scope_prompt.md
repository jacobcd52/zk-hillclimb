# Task: SCOPE proving the codebook model end-to-end through our ZKP (investigate + plan, do NOT implement yet)

Our ZKP currently proves the FIXED-POINT integerization (faithful-arch-v1: weights =
round(w*2^16), full walk verifies in 27s, weight-private). We want to prove the CODEBOOK
model end-to-end instead. The drivers are scheme-agnostic (they prove int matmuls over
committed int weights), so we expect NO new drivers/crypto — but we need the exact change-list.
Investigate and produce a concrete plan; DO NOT implement or modify anything yet.

## Investigate (READ ONLY; never modify or push int-model-approximation)
1. The codebook integerization in /workspace/projects/int-model-approximation (the scheme
   used for the codebook DiFR/capacity numbers — find it; the capacity docs call it
   "FP8->int32 codebook linears, lm_head float"). Determine EXACTLY:
   - Are the codebook linears a CLEAN integer matmul (int32 x int32 -> int64) over codebook
     integer values, i.e. structurally identical to the fixed-point path the ZKP already
     proves? Or is there per-group scaling / non-matmul structure (Hawkeye-like) that the
     current drivers can't express? This is the make-or-break question.
   - The SCALES: what fixed-point scales does the codebook path use for weights, activations,
     and each rescale (vs our orchestrator's hardcoded LOG_SF=16, GATE_RESCALE_LOG=20,
     UP/HIDDEN/DOWN/QKV/ROPE/VALUES rescales, scores 2^13/2^10, lm_head 2^16 in common.py)?
     List any that DIFFER.
   - lm_head: codebook keeps it float. Options to prove the model: (a) integerize lm_head
     fixed-point (minor deviation, what we do now) or (b) leave lm_head out of the proven
     scope. Recommend one.
   - rmsnorm/softmax/rope/attention: does codebook change any of these vs the fixed-point
     pipeline, or only the linear weights? (The nonlinearity drivers are scheme-independent;
     confirm.)
2. Our orchestrator (orchestrator/register.py, prove_walk.py, common.py): what exactly
   sources the weights + per-op activations today, and what would have to change to source
   the CODEBOOK weights + codebook forward-pass activations instead. Map each change to a
   file:line and classify it: (config/scale constant) vs (witness-generation wiring) vs
   (genuinely new code) vs (driver/protocol change — expected NONE).

## Deliverable
/workspace/projects/zk-hillclimb/CODEBOOK_PLAN.md:
- The make-or-break finding: is codebook a clean int-matmul model the existing drivers can
  prove as-is? (yes/no + evidence file:line). If NO, what's the minimal extra obligation.
- The exact change-list to prove codebook end-to-end, classified (config vs witness-wiring vs
  new code vs crypto), each with file:line and effort estimate.
- The scale-reconciliation table (codebook scales vs orchestrator constants; which differ).
- The lm_head recommendation.
- A go/no-go: "no major code changes" true or false, and the realistic effort (agent-cycles).
- Any risks (e.g. codebook activations exceed the int64/field bounds the drivers assume;
  codebook scales break a lookup-table-size constraint like the swiglu/exp/rescale table widths).
READ-ONLY except writing CODEBOOK_PLAN.md. No git commits; no pushes; no code changes anywhere.
