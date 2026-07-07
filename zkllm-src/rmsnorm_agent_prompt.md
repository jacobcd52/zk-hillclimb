# Task: implement, build, and validate `zkob_rmsnorm.cu`

You are working on a ZK-proof infrastructure project. Your ONLY job is to implement one new
CUDA driver, `/root/zkllm/zkob_rmsnorm.cu`, following a design that is already FINAL, then
build it and make its selftest pass completely.

## Read these first (in this order)
1. `/workspace/projects/zk-hillclimb/HANDOFF.md` — section "IMMEDIATE NEXT STEP: write
   zkob_rmsnorm.cu" is the complete final design spec. Follow it EXACTLY. Do not redesign.
2. `/workspace/projects/zk-hillclimb/submissions/baseline-native/PHASE0_NOTES.md` — pinned
   conventions (Fiat-Shamir schedule rules, IPA layout, Montgomery conventions, build commands).
3. `/root/zkllm/zkob_lookup.cuh` — the shared header. It already contains EVERYTHING you need:
   logUp lookup (fs_phase1/fs_phase2, LookupProof), IPA open_prove/open_verify, eq-weighted
   hadamard sumcheck (fs_hadamard, HadamardProof), build_eq_tensor, k_bcast_rows, and the
   degree-4 quartic sumcheck (fs_quartic, QuarticProof, lagrange5, k_hp4_step) — all compiled
   and validated. **DO NOT MODIFY THIS HEADER.** If you believe it has a bug, stop and write
   your finding to the report file instead of editing it.
4. `/root/zkllm/zkob_glu.cu` — the closest existing driver; copy its structure (prove/verify/
   selftest modes, FS absorb schedule style, homomorphic com helpers via h_mul/h_add/g1_eq,
   chain-file strip pattern, tamper_byte selftest helper, fr_eq sanity checks).
5. `/root/zkllm/zkob_rescale.cu` — for the rescaling_kernel usage and tLookupRange usage.
6. `/root/zkllm/vrf_common.cuh` — transcript + IPA + host G1/Fr helpers.

## Hard rules (violating any of these makes the work worthless)
- DO NOT modify ANY existing file in /root/zkllm. Only create `zkob_rmsnorm.cu` (and the report).
- All FrTensors hold PLAIN (non-Montgomery) Fr values. For values exceeding int64, build host
  Fr_t arrays: fr_from_u128(v) = {(uint32_t)v, (uint32_t)(v>>32), (uint32_t)(v>>64),
  (uint32_t)(v>>96), 0,0,0,0} and use the FrTensor(uint, const Fr_t*) ctor (raw memcpy).
- Host bracket math in __int128; require M < 2^62 and C < 2^16 (throw otherwise); require B == B_pad.
- NEVER write a NEW G1 CUDA kernel (known -dlto miscompilation on new G1 kernel shapes).
  All G1 arithmetic outside the proven header/upstream kernels must use the 1-thread host
  helpers h_mul/h_add/g1_eq from vrf_common.cuh. Fr-only kernels are safe to write.
- Use `load_long_tensor` / `load_int32_tensor` from the header for file input (upstream
  FrTensor::from_long_bin is broken).
- Every Fiat-Shamir challenge must be derived only AFTER absorbing the message it binds
  (follow the exact FS schedule in HANDOFF.md).
- Build commands (sm_89):
  compile: `cd /root/zkllm && nvcc -arch=sm_89 -std=c++17 -I/usr/local/cuda/include -dc -dlto zkob_rmsnorm.cu -o zkob_rmsnorm.o`
  link:    `nvcc -arch=sm_89 -std=c++17 -dlto zkob_rmsnorm.o bls12-381.o ioutils.o commitment.o fr-tensor.o g1-tensor.o proof.o zkrelu.o zkfc.o tlookup.o polynomial.o zksoftmax.o rescaling.o timer.o -o zkob_rmsnorm`
- Drivers do NOT mkdir the obdir — in selftest mode create temp dirs yourself (mkdir via std::filesystem or system()).
- Do not push anything to GitHub. Do not touch /workspace/projects/int-model-approximation.

## Deliverables
1. `/root/zkllm/zkob_rmsnorm.cu` — compiles and links clean.
2. `./zkob_rmsnorm selftest` runs the FULL selftest suite from the HANDOFF spec:
   honest small case (B=8, C=5) ACCEPT; all 5 semantic evil modes rejected by exactly the
   check the spec says should catch each; byte tampers on every proof/commitment/ipa file
   rejected. Then the real-scale case B=1024, C=768 (generate gens with `./ppgen 1024
   /tmp/gen1024.bin` if missing; C_eps for eps=1e-5: C_eps = round(1e-5 * 768 * 2^32)).
   Print clear PASS/FAIL lines for every case and a final "ALL PASS" only if everything passed.
3. `/root/zkllm/RMSNORM_REPORT.md` — what you implemented, exact selftest output summary,
   timings for the real-scale case (prove s / verify s / proof bytes), any deviations from the
   spec (there should be none) and any concerns. Be honest: if something does not pass, the
   report must say FAIL prominently — never claim success that didn't happen.

## Working style
- Iterate until the selftest fully passes. Compile errors, runtime errors: diagnose and fix
  your own driver file (never the shared sources).
- The GPU is an RTX 4090 (sm_89), currently idle, all yours.
- If after many attempts something in the DESIGN seems genuinely impossible (not a bug in your
  code), write the precise blocker to the report and exit — do not silently change the design.
