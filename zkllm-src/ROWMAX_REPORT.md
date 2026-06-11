# ROWMAX_REPORT — zkob_rowmax.cu (STAGE3_FAITHFUL_DESIGN §2, Part A)

Date: 2026-06-11. Driver: `/root/zkllm/zkob_rowmax.cu` (the only file created/edited).
Build: pinned `nvcc -arch=sm_89 -std=c++17 -I/usr/local/cuda/include -dc -dlto` +
standard link list. Final selftest log: `/root/zkllm/rowmax_selftest_final.log`.

## 1. What was implemented

One new driver proving, with zero advice freedom on the value, that a committed
per-row scalar mx[i] equals max over the allowed set of a committed B×NCOL int32
grid z, exactly per design §2:

- **Statement & guards (§2.1)** — both masking regimes (`causal`: AL = lower
  triangle, B == NCOL, V = 0; `vpad`: AL = first V columns, 0 < V <= NCOL) with
  every honest-prover throw implemented (power-of-two shapes, NPL ∈ {1,2}, gen
  sizes, t* range/mode checks, Df ∈ [0, LEN_R^NPL), the vpad |z| < 2^25
  envelope guard, limb-layout divisibility).
- **Obligations (§2.3)** — LIMB (logUp range lookup of the NPL·B·NCOL limb
  tensor L against tLookupRange(0, LEN_R), with com_A_L committed after beta_L),
  BIN (S binary on the whole padded grid; verifier requires U_f2 == S_f2 − 1),
  SUM (one-hot over allowed, claim 1), MASK (nothing outside allowed, claim 0),
  ATT (attainment, claim ev_mx absorbed; pure-broadcast eq_acc shortcut on the
  verifier), DOM (c1/c2 bracket with the never-committed mx broadcast pinned to
  com_mx at pt_c2's row-bit suffix, NPL plane openings of L at (u_r‖plane), and
  the plain-field identity c2 − c1 == v0 [+ 2^20·v1]), T-BIND (vpad+t*:
  S[i, t*[i]] = 1, claim 1, with the verifier loading and absorbing its OWN t*
  before any commitment).
- **Constant-claim discipline** — BIN 0, SUM 1, MASK 0, T-BIND 1 are protocol
  constants: imposed by the verifier at round 0 AND required equal to the
  serialized claim_H; never absorbed. Data-dependent claims (ev_mx, c1, c2)
  absorbed per the §2.6 schedule, byte-for-byte in the pinned label order.
- **The ONE new kernel (§2.8)** — `k_pp_expand`, Fr-only, driver-local,
  generalizing k_eq_expand's (1−c, c) doubling to arbitrary (a, b) pairs;
  powers `fast_me_weights` / `fast_s_vector` so the gen-32768 IPAs avoid the
  me_weights host-loop hot spot. No other new kernels; no G1 kernels.
- **CLI/files (§2.7)** — prove/verify/selftest; 26 (causal) / 29 (vpad+t*)
  proof files; dims.bin cross-checked against argv; driver does not mkdir;
  unpadded mx-int32 chain output (`-` to skip).

## 2. Selftest summary (final run: ALL PASS, 160 PASS / 0 FAIL)

- **Toy cases ×4**: causal 8×8 (n1=1), vpad 8×16 V=11 NPL=2 +t* (n1=3), causal
  16×16 (n1=2), vpad 4×8 V=NCOL=8 (MASK weight ≡ 0, n1=1).
- **Evil modes** — each rejected by EXACTLY the named check: evil=1 → "ATT
  round 0"; evil=2 → "DOM bracket identity"; evil=3 → "BIN round 0" (the
  certifying evil; see §6); evil=4 → "SUM round 0"; evil=5 → "MASK round 0"
  (vpad with pads only); evil=6 → "limb lookup round 0" (NPL=2 only); evil=7 →
  "IPA opening of c2 terminal vs com_mx"; evil=8 → "T-BIND round 0" (+t* only).
- **Byte tampers** on every §2.7 proof file (commitments, lookup, all hp files,
  lvals, every IPA, dims): all rejected (fail-closed on parse throws), and the
  restored directory re-verifies ACCEPT.
- **Guard throws (§2.1)**: all 12 setups throw with the expected message.
- **Real-scale causal and vpad**: both PASS (numbers below).

## 3. Real-scale timings

| case | prove | verify | proof+commitments | GPU peak |
|---|---|---|---|---|
| causal 1024×1024, LEN_R=2^20, NPL=1 | 5.28 s | 1.94 s | 791,140 B | 0.70 GiB |
| vpad 1024×32768, V=32000, NPL=2, +t* | 44.24 s | 4.02 s | 974,264 B | 10.99 GiB |

## 4. Memory-gate measurement — and the fix that was needed

**Final: 10.99 GiB vs the ~18 GiB §6.3 gate → WITHIN GATE** (also under the
14 GiB target this hardening pass aimed for; ≥ 2.2× headroom on the 24 GiB
RTX 4090). The §6.3 column-block fallback is NOT needed.

The first complete implementation passed everything *except* this gate: the
vpad prover peaked at **18.21–18.24 GiB**. A per-phase profile (env-guarded
`mem_mark` diagnostics fed by the selftest's 50 ms `cudaMemGetInfo` monitor)
located it precisely:

| prover phase | peak before (GiB) | peak after (GiB) |
|---|---|---|
| witness commits | 5.74 | 3.74 |
| LIMB (limb lookup) | 12.24 | **10.48 ← new global peak** |
| BIN | 15.90 | 9.21 |
| SUM | 17.15 | 8.33 |
| MASK | 17.90 | 7.96 |
| ATT | 15.93 | 6.46 |
| c1 | 17.41 | 7.96 |
| **c2** | **18.24 ← old global peak** | 8.21 |
| DOM planes | 8.21 | 6.46 |
| T-BIND | 15.96 | 6.21 |

(The full-selftest gate line reads 10.99 GiB vs the 10.48 standalone profile —
same allocations, slightly different fragmentation after the toy cases.)

**Diagnosis.** §2.5 predicted ≈ 8 GiB assuming per-obligation freeing. The
witness tensors WERE freed per obligation, but two costs hid in the shared
header machinery (`zkob_lookup.cuh`, which this task may not edit):

1. `fs_hadamard` is recursive and every frame's buffers stay alive through the
   entire descent: o0..o3 (4×h) for the round polynomial PLUS the three fold
   halves (3×h) per frame, summing to ≈ 7× the grid (7 GiB at D = 2^25) on top
   of three full-size head COPIES (`FrTensor Sc(S_t), Uc(ones_t)`). During c2
   this ran with z, S, L (2 GiB), AL, ones, mxb and eq_r all resident → 18.2 GiB.
2. `fs_phase1`'s first frame materializes five DL/2 temporaries (5 GiB at
   DL = 2^26) plus ME-fold chains, while z, S (dead weight there), L and A_L
   were resident → 12.2 GiB.

**Fix (memory discipline only; driver-local; no header edits; no new kernels;
no protocol change):**

- `lean_hadamard` — an iterative clone of the header's `fs_hadamard` using the
  SAME kernels (`k_hp3_step`, `k_fr_fold`), the same `FrTensor::sum` reduction,
  absorb labels, challenge schedule and terminal reads. Every emitted value is
  the identical field element; the only difference is lifetime: each round's
  o-buffers are freed before the fold halves are allocated, the previous
  round's halves are freed immediately after folding, and the round-1 inputs
  are read in place (no head copies). Working set ≈ 2.75×(D/2) above the
  inputs instead of ≈ 7×D + 3 copies. Used for all seven prover hadamard
  instances (BIN/SUM/MASK/ATT/c1/c2/T-BIND).
- Full-grid eq tensors (BIN's E, c1/c2's eq(u_r)) built as raw device buffers
  via the existing `k_pp_expand` doubling (`fast_me_weights_dev`, identical
  values to `build_eq_tensor` — same recurrence, same Montgomery convention)
  → 2× instead of 3× transient, freed immediately after the weight emul.
- Lifetime discipline: z and S are committed then freed until after the limb
  lookup (S's exact post-edit device bytes stashed host-side and re-uploaded,
  so the evil-3 device edits survive bit-exactly; z re-uploaded from the same
  int buffer); L freed after the lookup IPAs and re-uploaded from the
  unchanged host limbs for the DOM plane openings; z freed after ipa_z_c1;
  AL freed after c2's weight; Wtmp/NAL/eq scoped to die before each sumcheck.
- The previously-added `commit_chunked` (row-block commits through the same
  upstream kernels, bit-identical per row) is unchanged.

The remaining 10.5–11 GiB peak is the header `fs_phase1` frame-0 burst
(A_L 2 + L 2 + five 1 GiB temporaries + ME-fold transients), which cannot be
reduced further without editing `zkob_lookup.cuh`/`tlookup.cu` or changing the
round-polynomial computation path — out of scope and unnecessary at 2.2×
headroom.

**Byte-identity confirmation (required).** TOY-scale proofs were generated
with the pre-fix binary (saved as `zkob_rowmax.baseline`) and re-generated
with the final binary on identical inputs and seeds, then compared with
`diff -r`:

- causal 8×8 (seed `bidentest:causal`, fixed z, ppgen gens, mx-out):
  `/tmp/biden_old_causal` vs `/tmp/biden_new_causal` → **byte-identical**
  (all 27 files), mx chain files identical (`cmp`).
- vpad 8×16 V=11 NPL=2 +t* (seed `bidentest:vpad`):
  `/tmp/biden_old_vpad` vs `/tmp/biden_new_vpad` → **byte-identical**
  (all 30 files incl. hp_tbind/ipa_S_tbind/ipa_L_p1), mx files identical.

So the FS transcript, all commitments, every proof file and every verifier
byte are unchanged; old proofs verify under the new binary and vice versa.

## 5. Fast-vs-slow helper cross-check (§2.8, pinned)

`crosscheck_fast_helpers` runs in every evil==0 prove at the ATT block:
`fast_me_weights` vs the slow header `me_weights`, element-exact, and
`fast_s_vector` vs the slow MSB-first s_i product, element-exact, at toy
scale AND at gen-1024 real scale (real-causal and real-vpad runs both execute
it; u from logB=10-bit challenges, xs from the first 10 BIN challenges).
A mismatch throws (STOP-and-report). Outcome: **no mismatch ever observed**;
all selftest runs pass with the checks live. Additionally `fast_ipa_verify`'s
b-fold/s-vector algebra is exercised by every accepting verify (13–14 IPAs at
vpad) and by every IPA byte-tamper rejection.

## 6. evil=3 constant-claim fix (in the code; described from source)

evil=3 is the certifying forgery: a **fractional selector** that satisfies
SUM/MASK/ATT exactly and must be killed by BIN alone. The implementation in
the source (the fix from the earlier session, validated here):

- **Genuinely fractional c.** The setup scans for a row with two allowed
  positions j1 ≠ j2 whose values are distinct and BOTH ≠ mx, then sets
  c = (mx − z2)/(z1 − z2) in the field — c = 1 iff z1 = mx and c = 0 iff
  z2 = mx, so excluding max-valued positions guarantees c ∉ {0,1} (a naive
  choice touching the argmax degenerates to a binary selector that BIN cannot
  and should not reject). The honest one-hot at the argmax is zeroed and the
  row's mass moves entirely to S[j1] = c, S[j2] = 1 − c; attainment stays
  exact (c·z1 + (1−c)·z2 = mx), so ATT passes and only binarity is violated.
- **Field-true U buffer.** BIN's U = S − 1 is built from the int buffer
  Sh − 1, which cannot represent the fractional entries; the three affected
  entries are overwritten on device with the exact field values −1 (zeroed
  argmax), c − 1, and −c, so the prover runs the honest PROCEDURE on the
  inconsistent witness (strict=false on the BIN recursion only).
- **Constant claim does the catching.** The prover serializes
  hp_bin.claim_H = 0 unconditionally (protocol constant, never absorbed, never
  witness-derived); the verifier independently (a) rejects unless the
  serialized claim_H equals the constant and (b) imposes cur = 0 at round 0
  regardless of the file. The fractional row makes Σ eq·S·(S−1) ≠ 0, so the
  prover's honest round-0 evaluations satisfy p(0)+p(1) ≠ 0 and the verifier
  rejects at exactly **"BIN round 0 p(0)+p(1) != claim"** — the named check.
  Had the claim been absorbed from the prover, round 0 would pass and the
  rejection would surface elsewhere (or, against a malicious transcript, not
  at all): pinning the constants is load-bearing, not cosmetic.
- **t* interaction.** In the vpad+t* toy case, evil=3 runs with T-BIND
  disabled (the selftest passes tp = nullptr): strict=false applies only to
  the targeted recursion, and the fractional selector makes the T-BIND
  recursion inconsistent too. The same selftest case covers T-BIND's own
  rejection separately via evil=8.

## 7. Deviations and concerns

- **No protocol deviations.** FS schedule, file set, claims, checks and
  round counts are exactly §2.6/§2.7; byte-identity across the memory fix is
  proven in §4.
- **Implementation-note deviations from the literal reference code** (all
  value-identical, all forced by the §2.5/§6.3 memory gate, all confirmed by
  the byte-diff): chunked commits, the iterative `lean_hadamard`, raw
  `k_pp_expand` eq builds, and the commit-then-free/re-upload lifetime of
  z, S and L.
- **Concerns.** (i) The prover peak is now dominated by the shared header's
  `fs_phase1` frame-0 temporaries; any future grid growth (e.g. NCOL = 2^16)
  would need either the §6.3 column-block fallback or a header-side fix.
  (ii) The 50 ms memory monitor can under-sample very short allocation bursts;
  this is the gate's own metric and consistent across before/after runs.
  (iii) The argmax tie-freedom measurement hook (§2.4) remains an orchestrator
  task, unchanged by this driver.
