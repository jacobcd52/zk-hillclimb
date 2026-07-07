# Task: design + build the orchestrator (registration → prover walk → separate verifier → transcript.json)

You are building the conductor for a ZK-verified llama-68m forward pass. Five proof drivers
exist, are validated, and are BINARIES READY TO RUN in /root/zkllm: zkob_fc, zkob_rescale,
zkob_skip, zkob_glu, zkob_rmsnorm (a sixth, zkob_softmax, is being built by another agent
RIGHT NOW — do not touch zkob_softmax* files or rebuild ANY driver; use existing binaries).
The orchestrator is the trust boundary: its VERIFIER side must trust nothing but
commitments, openings, public constants, and byte-equality.

## Read first (all)
1. /workspace/projects/zk-hillclimb/submissions/baseline-native/PHASE0_NOTES.md — every
   driver's CLI, file conventions, chaining rules. §14 (rmsnorm) contains two PINNED
   orchestrator obligations: (a) com_X of each obligation MUST be chained byte-identical to
   the upstream activation commitment (a standalone ACCEPT proves much less); (b) rmsnorm
   saves com_W_ as com_Wr.bin — byte-equality checks must use that name.
2. /workspace/projects/zk-hillclimb/harness/HARNESS.md + harness/manifest_llama68m.json +
   harness/check_transcript.py — the FROZEN contract: what obligation ids must appear in
   transcript.json and the exact format check_transcript.py expects.
3. /workspace/projects/zk-hillclimb/SOFTMAX_DESIGN.md §7 (CLI + chain wiring you must be
   forward-compatible with) and §1.1 (the attention chain: fc → rescale 2^13 → rescale 2^10
   → softmax → fc; the int32→int64 widening shim between rescale stages is YOURS).
4. /workspace/projects/zk-hillclimb/THREAT_MODEL_NOTES.md — framing; the final-statement
   obligation (served token = argmax within tolerance) is future scope but design the
   transcript so it has a slot.
5. The integer pipeline /workspace/projects/int-model-approximation/m68-pipeline.py
   (READ ONLY, never modify/push) — how integer weights and activations are produced and
   dumped. The orchestrator's prover must produce its witnesses from THIS pipeline's
   integer semantics. NOTE one pinned change of authority: the rmsnorm advice R must be
   computed INTEGER-EXACTLY by the orchestrator's witness generator (the __int128/python-int
   bracket: largest r with r²·M ≤ 2^64·C, then ±1 fix-ups — see zkob_rmsnorm.cu's
   exact_R for reference), NOT by the pipeline's float path. Same authority rule will apply
   to softmax's P later (SOFTMAX_DESIGN §1: the integer spec replaces the float path).

## Deliverables (work in /workspace/projects/zk-hillclimb/orchestrator/)
1. ORCHESTRATOR_DESIGN.md — short and precise: directory layout per run, registration
   format (public.json: gens, registered weight-commitment hashes, table hashes), the
   manifest walk order, the chain byte-equality map (which obdir file == which obdir file,
   for EVERY edge in the covered subgraph), how SKIPPED ids are recorded, and the
   verifier's independence argument (one paragraph: everything it checks and what it never
   trusts).
2. `register.py` — one-time setup for a run: generate/locate gens (ppgen), export integer
   weights from the pipeline (to a workdir OUTSIDE the int-model-approximation repo),
   commit registered weights (drivers have a commit/registration mode — check their CLIs in
   PHASE0_NOTES; if a needed standalone commit mode is missing, commit via a tiny driver
   invocation pattern that already exists rather than writing new CUDA), write public.json
   with sha256 of every registered commitment + table.
3. `prove_walk.py` — walks manifest_llama68m.json over real pipeline data for the COVERED
   SUBGRAPH (stage 1 = the full MLP path of both layers + all rmsnorm sites + skip
   connections: rmsnorm → gate/up fc → rescales → glu(swiglu) → rescale → down fc →
   rescale → skip; embeddings/attention/softmax/lm-head are SKIPPED for now), creating
   obdirs (drivers do NOT mkdir — that is the orchestrator's job, a known gotcha), running
   each prove with the seed convention "<run_seed>:<obligation_id>", wiring chain files
   (including any int32→int64 widening shims), and recording a manifest of what ran.
4. `verify_walk.py` — a SEPARATE process that re-runs every covered obligation's verify
   with the registered public inputs, checks EVERY chain byte-equality edge, checks
   registered-commitment hashes against public.json, and emits transcript.json in the
   harness format (ACCEPT/REJECT per obligation id + SKIPPED markers + overall verdict).
   It must never read prover-side witness files — list in the design doc exactly what it
   reads. transcript.json must pass harness/check_transcript.py (run it).
5. A selftest: `selftest.sh` that (a) runs register → prove_walk → verify_walk on real
   llama-68m data end-to-end for the covered subgraph and gets overall ACCEPT with all
   covered ids ACCEPTED; (b) tampers ONE chained commitment file and shows verify_walk
   REJECTS with the right id; (c) tampers one registered-weight hash in public.json and
   shows REJECT at registration check. Print PASS/FAIL lines and a final ALL PASS.
6. ORCHESTRATOR_REPORT.md — what ran, timings per obligation and total (prove and verify
   separately), total proof bytes for the covered subgraph, problems hit, honest status.

## Rules
- Python for the orchestrator (stdlib + numpy + torch only as needed); NO new CUDA, no
  edits to any .cu/.cuh, no rebuilds of drivers (another job is compiling in /root/zkllm —
  binaries for the five validated drivers are already there; invoke them as subprocesses).
- GPU is shared with that job: serialize your driver invocations (one at a time), retry
  once on failure.
- Never push anything to GitHub; never modify int-model-approximation; do not git-commit
  (the coordinator commits).
- Honest reporting: if a chain edge or obligation cannot be completed, the report and
  transcript must say so explicitly — no quiet narrowing of scope.
