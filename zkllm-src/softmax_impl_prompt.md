# Task: implement, build, and validate `zkob_softmax.cu`

Implement one new CUDA driver, `/root/zkllm/zkob_softmax.cu`, from a FINAL design document.
Follow the design EXACTLY — every relation, FS absorb, file name, CLI argument and selftest
case is already pinned. Do not redesign anything.

## Read in this order
1. `/workspace/projects/zk-hillclimb/SOFTMAX_DESIGN.md` — THE spec. Sections 2 (statement),
   4 (obligations), 5 (FS schedule), 6 (bounds), 7 (CLI/files), 8 (selftest) are normative.
   The domain-margin gate in §9.1 has ALREADY been resolved by measurement
   (/workspace/projects/zk-hillclimb/SCORES_RANGE.md: global max |score| = 277, margin 3.7x)
   — the constants are frozen as designed (LOW_E = −2^19, LEN_E = 2^20, LEN_R = 2^20).
2. `/workspace/projects/zk-hillclimb/submissions/baseline-native/PHASE0_NOTES.md` — pinned
   conventions.
3. `/root/zkllm/zkob_lookup.cuh` + `/root/zkllm/vrf_common.cuh` — shared machinery
   (**DO NOT MODIFY EITHER**; if you believe one has a bug, stop and write the finding to
   the report instead).
4. `/root/zkllm/zkob_glu.cu` — the mapping-lookup pattern (obligation 1) and hadamard
   verifier replay; `/root/zkllm/zkob_rmsnorm.cu` — multi-sumcheck single-transcript
   structure, single-row openings, evil-mode/selftest discipline.

## Hard rules
- Only create `zkob_softmax.cu` (and the report). No edits to any existing file.
- All FrTensors PLAIN form. No new G1 CUDA kernels (host h_mul/h_add/g1_eq only for G1).
  The design needs NO new kernels at all (§0) — k_fr_emul, k_fr_fold, k_bcast_rows,
  build_eq_tensor and the upstream kernels cover everything. If you find yourself writing
  a kernel, re-read the design.
- Host math fits int64 per design §6 — use long long; throw on every §2 completeness guard.
- Every FS challenge derived only AFTER absorbing what it binds — the §5 schedule is
  absorb-by-absorb normative, labels included.
- Verifier requirements that are load-bearing and easy to forget (from the design):
  U_f2 == 1 in obligations 3 and 4 (all-ones MLE); the verifier RECOMPUTES the public
  weight folds (W_rs, eq⊙MK) itself; V2's U_f2 opens against com_S at the row-bit suffix
  of pt2; identities I1/I2 checked in plain field arithmetic; round-count and
  commitment-row-count guards; dims.bin cross-checked against argv.
- Do NOT port glu's mapped(0)==0 check (design §4.1/§7.1 explains why).
- Build commands (sm_89), same object list as the other drivers:
  `nvcc -arch=sm_89 -std=c++17 -I/usr/local/cuda/include -dc -dlto zkob_softmax.cu -o zkob_softmax.o`
  `nvcc -arch=sm_89 -std=c++17 -dlto zkob_softmax.o bls12-381.o ioutils.o commitment.o fr-tensor.o g1-tensor.o proof.o zkrelu.o zkfc.o tlookup.o polynomial.o zksoftmax.o rescaling.o timer.o -o zkob_softmax`
- Drivers do NOT mkdir the obdir; selftest creates its own /tmp dirs.
- Do not push anything to GitHub; never touch /workspace/projects/int-model-approximation.

## Deliverables
1. `/root/zkllm/zkob_softmax.cu` — compiles clean, selftest passes IN FULL:
   the three small cases of §8.1 (with the evil==0 fold-vs-multi_dim_me convention sanity
   checks of §9.5), all five semantic evil modes of §8.2 rejected by EXACTLY the named
   check, byte tampers on every §7.2 file (§8.3 offsets), real-scale case §8.4 (B=1024,
   gen via ./ppgen 1024 /tmp/gen1024.bin if missing, in-driver host-double exp table
   flagged non-authoritative) with one tamper, timings printed. Final line
   "ZKOB-SOFTMAX SELFTEST: ALL PASS" only if everything passed.
2. The pinned table-generation script from §7.4, saved as
   /root/zkllm/gen_softmax_exp_table.py (do not run it as authority; selftest may generate
   in-driver — the script is for the orchestrator's registration step).
3. `/root/zkllm/SOFTMAX_REPORT.md` — what was implemented, selftest summary, real-scale
   timings (prove s / verify s / bytes), any deviations (should be none) and concerns.
   Honest reporting: a FAIL is reported as FAIL, prominently.

## Working style
Iterate until the selftest fully passes; fix only your own file. The GPU is an RTX 4090.
If the design seems genuinely impossible somewhere (not a bug in your code), write the
precise blocker to the report and exit — do not silently change the design.
