# Task: implement, build, and validate `zkob_rowmax.cu`

Implement ONE new driver from a FINAL design: §2 (Part A) of
/workspace/projects/zk-hillclimb/STAGE3_FAITHFUL_DESIGN.md. Follow it EXACTLY — the
statement (§2.1), padding regime (§2.2), obligations LIMB/BIN/SUM/MASK/ATT/DOM/T-BIND
(§2.3), bounds (§2.5), FS schedule (§2.6), CLI/files/edges (§2.7), the one new kernel
(§2.8), and the selftest plan (§2.9) are all normative. No design decisions are yours.

Also read: PHASE0_NOTES.md (conventions), zkob_lookup.cuh + vrf_common.cuh (machinery —
DO NOT MODIFY EITHER), zkob_softmax.cu (closest patterns: constant-claim discipline,
broadcast-MLE binding, L-plane openings, public-weight rebuild+fold, selftest style),
zkob_rmsnorm.cu (eq_acc shortcut for pure-broadcast weights).

## Hard rules
- Create ONLY zkob_rowmax.cu (and the report). All FrTensors PLAIN; host math int64
  (no __int128 needed per §2.5); every §2.1 honest-prover throw implemented.
- The ONE new kernel k_pp_expand is Fr-only, driver-local, per §2.8 EXACTLY — and the
  two fast helpers (fast_me_weights, fast_s_vector) MUST be cross-checked element-exact
  against the slow header versions in evil==0 convention checks at toy AND gen-1024
  scale (the design pins this; a mismatch is a STOP-and-report, not a workaround).
  Montgomery convention: mont-ify all factors except one. NO other new kernels; NO G1
  kernels ever.
- Constant claims (BIN 0, SUM 1, MASK 0, T-BIND 1) are protocol constants: imposed by
  the verifier at round 0 AND required equal to the serialized claim_H (reject
  otherwise); never absorbed. Data-dependent claims (ev_mx, c1, c2) absorbed per §2.6.
- Load-bearing verifier checks (do not forget): BIN's U_f2 == S_f2 − 1; U_f2 == 1 in
  SUM/MASK/c1/c2/T-BIND; ATT's three openings; DOM's c2 − c1 == v0 [+ 2^20·v1] in plain
  field; mx broadcast bound via com_mx opened at pt_c2's row-bit suffix; round-count and
  com-row-count guards; dims.bin vs argv; vpad: t* loaded by the VERIFIER from its own
  path argument and absorbed before any commitment.
- Both modes (causal, vpad) fully implemented; prover frees each obligation block's
  tensors before the next (§2.5 memory requirement).
- Build: pinned -dc -dlto compile + standard link list. Driver does NOT mkdir obdir.
  No GitHub; no edits to anything else.

## Deliverables
1. zkob_rowmax.cu — compiles clean; `./zkob_rowmax selftest` runs the FULL §2.9 plan:
   toy shapes both modes, every evil mode rejected by EXACTLY the named check, byte
   tampers on every §2.7 file, real-scale causal (1024×1024, scores-like data) AND
   real-scale vpad (1024×32768, V=32000, logits-like data, with t*) — generate gen32768
   via `./ppgen 32768 /tmp/gen32768.bin` if missing (and gen1024 as usual). Print
   timings; report the vpad GPU memory peak vs the §6.3 gate (~18 GiB). Final line
   "ZKOB-ROWMAX SELFTEST: ALL PASS" only if everything passed.
2. /root/zkllm/ROWMAX_REPORT.md — what was implemented, selftest summary, real-scale
   timings both modes (prove/verify/bytes), the memory-gate measurement, the
   fast-vs-slow helper cross-check outcome, deviations (none expected), concerns.
   Honest reporting.

GPU: RTX 4090, free. Iterate until ALL PASS; fix only your own file; if the design seems
impossible somewhere, STOP and write the precise blocker to the report.
