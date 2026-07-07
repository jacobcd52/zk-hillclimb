# Task: FINAL design for binding RoPE + per-head slicing in the attention chain (design only)

The attention chain is the last unproven segment: QKV projections (zkob_fc, done) →
**RoPE applied to Q,K (currently float in the pipeline — UNBOUND)** → per-head slicing →
scores matmul per head (zkob_fc) → two rescales → zkob_softmax (done) → values matmul per
head (zkob_fc) → rescale → output projection (zkob_fc). Your job: a design document for the
missing links — (A) an integerized, proof-bound RoPE obligation, and (B) the per-head
slicing binding — complete enough that an implementer makes zero design decisions. The bar
is /workspace/projects/zk-hillclimb/SOFTMAX_DESIGN.md (read it first as the style/quality
register; its §4.3-4.6 patterns are directly reusable here).

## Read
1. SOFTMAX_DESIGN.md (style + the public-weight-tensor sumcheck and boolean-bit-opening
   patterns); PHASE0_NOTES.md §1-15 (all conventions + all validated machinery).
2. m68-pipeline.py in /workspace/projects/int-model-approximation (READ ONLY): the exact
   RoPE code path (which lines, what rotate convention — LLaMA rotate_half vs interleaved,
   what position/frequency math, applied at what scale/dtype), and the Q/K/V + scores
   integerization (lines ~137-159).
3. /workspace/projects/zk-hillclimb/orchestrator/ORCHESTRATOR_DESIGN.md — the chain/edge
   formalism your obligations must slot into (§3-4), and the pinned witness-authority rule
   (the orchestrator computes integer witnesses exactly; the integer spec replaces the
   float path — RoPE gets the same treatment as rmsnorm's R and softmax's P).
4. /root/zkllm/zkob_lookup.cuh + vrf_common.cuh — available machinery. NO new G1 kernels;
   prefer no new kernels at all; any new Fr kernel needs the Montgomery rule + justification.

## Design constraints (pinned)
- Integer RoPE spec: public integer cos/sin tables (state scale, generation script,
  sha256-registered like the exp table), exact integer multiply-add, one rescale back to
  scale 2^16 via zkob_rescale. Every rounding proven exact (rescale machinery). State the
  exact integer relation per element, with bounds arithmetic (int64 safety, field fit).
- Zero prover advice; covert capacity 0 bits. If any tolerance is unavoidable, quantify it
  in bits/row and justify why it cannot be removed.
- Coordinate hints: rotate_half is a signed permutation — its MLE at a point is ±the
  original MLE at a bit-flipped point, so it may be bindable via openings of the SAME
  commitment at modified points (no extra commitment). A head slice of a 64-aligned
  column block is the full tensor's MLE with the head-selector column bits fixed to
  booleans — the softmax L-plane-opening pattern. CHECK these against the actual pinned
  bit-order conventions (LSB-first column bits) and the actual head layout in the
  pipeline (which columns belong to head h AFTER the pipeline's reshape/transpose at
  lines ~156-157 — get this exactly right, it is the classic off-by-reshape bug).
- Per-head scores/values matmuls via UNMODIFIED zkob_fc: specify exactly what com_W means
  there (chained activation commitment of K_h^T / V_h — how does a 1024×64 tensor commit
  under which gens, and how does that interoperate with the slicing binding?). If zkob_fc's
  layout makes some orientation impossible without driver changes, say so explicitly and
  design the minimal-change alternative (flag it — driver edits need a revalidation pass).
- One transcript per obligation; exact FS schedules absorb-by-absorb; CLI signatures;
  obdir file lists; chain byte-equality edges in ORCHESTRATOR_DESIGN §4 table format;
  selftest plan with a semantic evil mode per check (+ byte tampers + real-scale with
  predicted costs extrapolated from: softmax 10.2s/11.6s at 2^20 grid; fc/rescale/rmsnorm
  numbers in PHASE0_NOTES).
- 12 heads × 2 layers; B=1024; head_dim 64. Count total new obligations per forward pass
  and predicted added prove/verify seconds.

## Deliverable
/workspace/projects/zk-hillclimb/ROPE_ATTENTION_DESIGN.md with sections: 1. Pipeline
semantics (quoted, exact, incl. the reshape/transpose head-layout analysis); 2. Integer
RoPE spec + table generation; 3. Statement(s) to prove; 4. Proof obligations; 5. FS
schedules; 6. Numeric bounds; 7. CLI + files + chain edges; 8. Selftest plan; 9. Open
questions/risks (honest). Design only — write NO code outside the table-gen script spec.
READ-ONLY task except that one file.
