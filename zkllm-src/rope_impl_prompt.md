# Task: implement, build, and validate THREE drivers: zkob_rope, zkob_headslice, zkob_headmerge

Implement the three attention-binding drivers from a FINAL design. Follow it EXACTLY —
every relation, FS absorb, file name, CLI argument and selftest case is pinned. No design
decisions are yours to make.

## Read in this order
1. /workspace/projects/zk-hillclimb/ROPE_ATTENTION_DESIGN.md — THE spec (all sections
   normative; §5 FS schedules absorb-by-absorb, §8 selftest plan including the toy-shape
   general rule in §2.2's parenthetical).
2. /workspace/projects/zk-hillclimb/submissions/baseline-native/PHASE0_NOTES.md — pinned
   conventions.
3. /root/zkllm/zkob_lookup.cuh + vrf_common.cuh — shared machinery (**DO NOT MODIFY**).
4. /root/zkllm/zkob_softmax.cu — the closest patterns (public-weight hadamard with
   verifier-side weight folds, U_f2==1 checks, boolean-bit openings, selftest discipline);
   /root/zkllm/zkob_rope* does not exist yet — you create all three .cu files.

## Hard rules
- Create ONLY zkob_rope.cu, zkob_headslice.cu, zkob_headmerge.cu, gen_rope_tables.py
  (§2.1, verbatim from the design), and the report. No edits to anything else.
- ZERO new CUDA kernels (the design composes entirely from existing machinery — if you
  find yourself writing one, re-read the design). All G1 host work via h_mul/h_add/g1_eq.
- All FrTensors PLAIN form; negative table values via the int ctor (mod-p); host math
  int64; every §6 honest-prover throw implemented.
- FS: every challenge derived only AFTER absorbing what it binds; §5 schedules are
  normative including labels.
- Load-bearing verifier requirements (easy to forget): U_f2 == 1 in every public-weight
  hadamard; the verifier REBUILDS W1/W2 (rope) and each Wm_h (merge) itself from
  registered tables / the π⁻¹ formula; rope's second opening point pt2' computed by the
  VERIFIER (1−pt2[5]); headslice's paired openings must BOTH verify against the SAME
  absorbed eval; round-count and com-row-count guards; dims.bin vs argv.
- Build (sm_89): same -dc -dlto compile + link object list as the other drivers, one
  binary per driver.
- Drivers do NOT mkdir obdirs; selftests create their own /tmp dirs.
- No GitHub pushes; never touch int-model-approximation.

## Order of work (one driver fully done before the next)
1. zkob_rope: implement → build → full §8 selftest (toy shapes per §2.2 general rule +
   evil modes per §8 + byte tampers + real-scale B=1024 C=768 HD=64 with in-driver
   non-authoritative table fallback) → ALL PASS.
2. zkob_headslice: same; **measure the real-scale instance time and report it against
   the §9.1 gate (30 s)** — if exceeded, apply the sanctioned me_weights memoization
   fallback (host-side only) and re-measure; do NOT invent batching.
3. zkob_headmerge: same (including the B == C plain-transpose toy case §6 mentions).

## Deliverables
- Three .cu files + three binaries, each selftest printing per-case PASS/FAIL and a final
  "ZKOB-<NAME> SELFTEST: ALL PASS" only if true.
- gen_rope_tables.py (§2.1 verbatim).
- /root/zkllm/ROPE_IMPL_REPORT.md: per driver — what was implemented, selftest summary,
  real-scale timings (prove s / verify s / proof bytes), the headslice gate measurement
  and which fallback (if any) was applied, deviations (should be none), concerns. Honest
  reporting: FAIL stated prominently if anything fails.

GPU: RTX 4090, may be shared with a python audit job (light). Iterate until everything
passes; fix only your own files. If the design seems genuinely impossible somewhere,
write the precise blocker to the report and exit — do not silently change the design.
