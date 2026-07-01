# Fable Independent Adversarial Review — zkLLM / P3 Hash-Based Prover

**Reviewer:** Fable (Claude), independent adversarial technical review
**Date:** 2026-07-01
**Hardware:** RTX 4090 (Ada, SM89), CUDA 12.4, nvcc 12.4.131
**Scope:** `/root/zkllm/p3_*.cuh`, `p3_*_test.cu`, `p3_*selftest*.cu`

> Status legend: **[VERIFIED]** = I ran code and observed it. **[REASONED]** = analysis/argument, not executed.

---

## 0. Executive summary

I built and ran everything on the RTX 4090. **All 11 self-tests (91 checks) and the 9-check
capstone soundness battery pass**, and I reproduced the headline performance on a real llama-68m FC
layer: **prove 303 ms, verify 34 ms, proof 2.47 MB**. The test suite is genuinely good (honest /
tamper / hiding trichotomy with real negative controls). I could **not** forge a false `Y=X·W`; the
committed-mask (`rootQ`/`qr`) fix defeats every soundness tamper I tried.

Four findings, in priority order:

- **F2 (privacy, confirmed & sharpened):** *Zero-knowledge is not achieved.* The Basefold opening's
  **last** sumcheck round (the ex-slice bit, bound last) emits `s0 = c0·w0` where `c0` is an
  **unmasked, publicly-known linear functional of the hidden operand** — I measured its
  distribution as a spike (chi-sq 5.12e6, constant across 20k mask draws) while rounds 0–7 are
  uniform. This is not "0.3 bit of leakage": it is one exact linear equation per operand per proof,
  so **≈|W| proofs of the same weights recover W by Gaussian elimination.** Fix (design-level, not
  applied): blind the opening sumcheck with the `p3_zksumcheck` machinery that already exists.

- **F3 (soundness calibration, verified):** the advertised **2^-58 is only the algebraic term**; the
  dominant term is FRI query soundness, which I measured at **≈2^-24** at the sweep defaults
  (R=2,Q=24) and ≈2^-32…2^-44 at the llama driver (R=1,Q=64). GL2 helps the algebraic term but not
  this one. Raise `Q` (~100 at R=2) for cryptographic soundness; proof/verify scale linearly in Q.

- **F1 (robustness, verified & FIXED):** the prover ignored **all** CUDA return codes; above
  ~16–17M-param W it silently OOMs (mempool never releases), scribbles a host pointer into a Merkle
  root, and returns a corrupt proof (I caught `cudaGetLastError = illegal memory access`, 0 GB free,
  non-deterministic roots). It happens to fail verification (no false-accept), but the prover gave no
  signal. **I applied a fix**: `prove()` now checks for CUDA errors and returns an empty proof
  (cleanly rejected) with a diagnostic instead of garbage — verified on the two failing shapes.

- **F4 (scope, verified):** no code binds `rootW` to a *published* weight commitment, and inputs/
  outputs aren't tied to public values (`Y` is hidden). This is a sound private-matmul gadget, **not**
  yet end-to-end verifiable inference; non-linearities aren't in the P3 path at all.

**Comparison (§8):** transparent + post-quantum + GPU-fast-per-layer, uniquely *aiming* at weight
privacy — but today matmul-only, ZK-broken, large proofs (hash-PCS tax), under-parameterized
soundness. WHIR would upgrade the PCS (fewer queries, smaller proofs) and DeepProve/Expander
out-throughput it on full models. **fp8-exact (§9):** cheapest path is to serve with an exact/Kulisch
(or MXFP8 within-block-integer) accumulator so *served == proven* reuses today's integer proof at
**~1×** (Design 1) or **~2–4×** (Design 2, MXFP8); faithfully matching arbitrary fp32-order
accumulation costs **~10–50×** (Design 3). Keep matmul in sumcheck, use lookups only for fp8 decode
+ RNE rounding.

---

## 1. Build + test reproduction  [VERIFIED]

All 13 targets compile clean with `nvcc -arch=sm_89 -std=c++17 -O3 -I.`.

**Self-tests (all pass, 91 checks total):**

| harness | result |
|---|---|
| p3_gl2_selftest (ext field axioms) | 4/4 PASS |
| p3_basefold_gl2_selftest (GL2 opening) | 8/8 PASS |
| p3_zk_selftest (salted hiding Merkle) | 8/8 PASS |
| p3_zkopen_test (query-open hiding) | 4/4 PASS |
| p3_matmul_selftest (FC matmul argument) | 8/8 PASS |
| p3_maskslice_test (mask-slice mechanism) | 5/5 PASS |
| p3_opening_zk_test (transcript privacy) | 9/9 PASS |
| p3_basefold_selftest (eval opening) | 10/10 PASS |
| p3_fri_selftest (FRI LDT) | 11/11 PASS |
| p3_zksumcheck_test (ZK sumcheck) | 10/10 PASS |
| p3_zkmatmul_test (ZK matmul sumcheck) | 9/9 PASS |

Every harness includes negative controls with "teeth" (unmasked reveal → chi-sq ≈ 5.12e6 vs uniform ≈ 256), and every tamper case rejects with a sensible reason. This is a genuinely well-constructed test suite — the honest/tamper/hiding trichotomy is exactly what you want.

**Capstone soundness battery `p3_private_fc_test` (bb=4,ii=8,oo=6,R=2,Q=24):** all 10 checks PASS
(honest-accept, wrong-product reject, tampered-sumcheck reject, tampered-opened-value reject,
tampered-codeword reject, determinism, mask-active, Q=0-forgery reject, mismatched-R reject).
Note: this test calls `prove()` with the default `gpu=false`, so its commit runs the **host**
`rs_encode` (O(N·M) Horner) — this is why the battery takes minutes, not the ~0.3 s of the GPU path.
Correctness is unaffected.

---

## 2. Finding F1 — prover swallows all CUDA errors; silent corruption / OOM at scale  [VERIFIED]

**Severity: robustness / liveness (NOT a demonstrated soundness break).**

Running the intended GPU path (`./p3_sweep bb ii oo R Q`, which passes `gpu=true`):

```
B,IN,OUT,prove_ms,verify_ms,proof_kb,ok
64,1024,1024, 149.9, 14.1,  995.1, 1     # 1M-param W
64,4096,4096,1901.3, 19.2, 1274.7, 1     # 16.7M-param W  (headline claim holds)
256,4096,4096,2086.5,19.0, 1411.4, 1
16,2048,2048, 488.9, 14.1, 1011.6, 1
128,8192,4096,3846.5, 0.0, 1418.9, 0     # <-- FAILS
64,4096,4096,1157.3, 16.0, 1180.1, 1     # R=1
64,4096,4096,2963.6, 0.0, 1369.2, 0      # R=3  <-- FAILS
64,4096,4096,1892.0, 28.6, 2121.4, 1     # Q=40
```

Two shapes produce `ok=0`. Both fail with `why="Y opening bind"`. Digging in:
`openY.roots[0]` comes back as `10a83bf3ff7f0000` — **a host pointer** (`…7f0000` = Linux
user-space address), not a SHA-256 hash — and `rootY` is **non-deterministic across two identical
`prove()` calls**. Instrumenting the failing shape:

```
GPU mem free BEFORE prove: 24.37 / 25.26 GB
GPU mem free AFTER  prove:  0.00 GB ; cudaGetLastError = an illegal memory access was encountered
```

**Root cause.** The prover contains *zero* CUDA error checking — every `cudaMalloc`,
`cudaMemcpy`, kernel launch, and `cudaMallocAsync` return code is ignored. At large shapes the
device Merkle tree for `W` alone is ~`2·2^(ii+oo+1+R)·32` bytes (≈8.6 GB at M0=2^27), and the
async mempool is deliberately set to **never release memory to the driver**
(`cudaMemPoolAttrReleaseThreshold = UINT64_MAX`, `p3_enable_mempool`). Memory for X/W/Y commits
accumulates, the device eventually returns errors, the kernels scribble/return garbage, and
`prove()` returns a structurally-complete but corrupt proof. Here it happens to fail `verify`
(garbage root ≠ committed root), so there is **no false-accept** — but the prover gives the caller
no signal that anything went wrong, and `p3_sweep` reports `ok=0` as if it were a normal negative
result.

**Why it matters.** (a) The linear-scaling performance claim is only valid below the OOM cliff
(~16–17M-param W at R=2 on a 24 GB card); above it the system silently breaks. (b) A prover that
ignores all device errors can, in principle, emit wrong results in other paths; you never want a
proof system whose prover can't tell success from memory corruption. (c) The never-release mempool
makes OOM arrive *sooner* under repeated proving (long-running server).

**Fix I recommend (did not apply — invasive, spans hot paths):** wrap CUDA calls in a checked
macro that aborts (or returns an explicit `ok=false` sentinel) on error; free `DeviceMerkle` trees
between operands rather than only at the end; and for the largest operand stream the Merkle tree
(only keep the ~O(log M) frontier + roots) instead of materializing every level. A one-line
mitigation to at least surface the bug: check `cudaDeviceSynchronize()`/`cudaGetLastError()` at the
end of `prove()` and refuse to return a proof on error.

---

## 3. Finding F2 — the ZK gap is real, and it is a *linear* leak enabling full weight recovery  [VERIFIED]

**Severity: privacy — ZERO-KNOWLEDGE IS NOT ACHIEVED. Confirmed and characterized more sharply than "a small leak."**

The self-assessment (OVERNIGHT_SUMMARY §2D) says the Basefold opening's last sumcheck round
leaks the real-slice evaluation. I verified this and pinned down *what* leaks and *how bad* it is.

**Mechanism.** Each operand is augmented as `[real(N) | random(N)]`, so the ex slice-selector is
the **top** bit of the index. The opening sumcheck binds variables **LSB-first**, so the ex bit is
bound **last**. One round before the end the folded coefficient array is exactly `[c0, c1]` indexed
by ex, where `c0 = MLE(real)(α')` and `c1 = MLE(random)(α')` at the folded point
`α' = (α_0,…,α_{v-2})`. The final round's message is `s0 = c0·w0`. The mask lives entirely in `c1`
(→ `s1`); it never enters `s0`. So `s0` of the last round reveals `c0`, an **unmasked, known linear
functional of the real witness**, in the clear.

**Experiment** (`leak_lastround.cu`, replicates the opening fold at fixed challenges, 20 000 mask
draws per round):

```
round :  chisq(real)  masked_uniform?
  0   :       236       YES         <- rounds 0..7: s0 uniform over mask draws (hidden)
  ...
  7   :       210       YES
  8   :   5120000       NO          <- LAST round (ex bit): s0 CONSTANT across mask draws
```

chi-sq = 5.12e6 = every sample in one histogram bin = the value does **not move** when the mask
changes. It is a deterministic function of the real witness only. Rounds 0–7 sit at the uniform
baseline (~256). This is exactly the predicted leak, and it is decisive.

**Why this is worse than "leaks ~0.3 bit."** The leaked quantity is a *known linear functional*
`⟨a_k, w⟩` of the hidden operand (the coefficients `a_k = w0·eq(α'_k,·)` are computable by anyone
from the public Fiat-Shamir transcript). One proof leaks one such equation per opened operand
(X, W, Y). The persistent secret is the weight matrix **W**, opened afresh in every inference proof.
Across `K` proofs the attacker collects `K` independent linear equations in W's entries; once
`K ≳ |W|` (number of weight coordinates in the opened block) the system is full-rank whp and **W is
recovered exactly by Gaussian elimination.** This is not a statistical/entropy argument — it is
linear algebra with known coefficients. For a weight-privacy threat model (the whole point of the
system), repeated serving of the *same* weights is the normal case, so this is a practical break.

**Fix I recommend (did not apply — design-level).** The project already has the right tool for the
matmul sumcheck: `p3_zksumcheck.cuh` blinds each round message with a random polynomial of known
cube-sum (its tests show simulated, witnessless transcripts that accept and match the real
distribution). That masking must also be applied to the **Basefold opening sumcheck** (the piece
that is currently *not* zero-knowledge): add a per-opening random blinding polynomial `g` with
published `Σg`, send `s(t)+ρ·g` messages, and open `g` at the fold point (a committed mask, exactly
like `rootQ`/`qr` in the capstone). Alternatively, bind the ex bit **first** (MSB→LSB) so it mixes
into every later round and no round's `s0` is a pure real functional. Either way this reuses code
that already exists and is tested; it is contained work, not a rewrite.

---

## 4. Finding F3 — the "~2^-58" margin is the *algebraic* term only; system soundness is bottlenecked by FRI queries (~2^-24 at defaults)  [VERIFIED + REASONED]

The soundness of this construction is the sum of two independent error terms:

1. **Algebraic / Fiat-Shamir term** (sumcheck + Basefold fold consistency). Each round is a
   degree-2 identity checked at a base-field challenge; error ≈ `2·(#rounds)/|F|`. With `|F|≈2^64`
   and up to ~26 rounds this is ≈ `2^-58`. **This is the figure the write-up cites, and it is
   correct for what it measures.** The flagged GL2 (degree-2 extension) upgrade lifts it to
   ≈ `2^-116`.

2. **FRI / Basefold query (proximity) term.** To pass the queries with a codeword `δ`-far from the
   RS code, the adversary evades detection with probability `(1-δ)^Q` per the checked coset pairs.
   I measured the per-query detection rate directly (`forge_probe.cu`, honest RS-encoded word of
   degree < 2^8, rate `ρ=2^-R`, corrupt a `δ` fraction, count inconsistent folds over 200k random
   queries):

   ```
   delta=0.05  per-query detect=0.096   (1-detect)^24 = 2^-3.5
   delta=0.10  per-query detect=0.185   (1-detect)^24 = 2^-7.1
   delta=0.25  per-query detect=0.382   (1-detect)^24 = 2^-16.6
   delta=0.50  per-query detect=0.632   (1-detect)^24 = 2^-34.7
   ```

   At `R=2` (rate 1/4) the unique-decoding radius is `δ=(1-ρ)/2=0.375`, where detection ≈ 0.5, so
   the **query soundness at the sweep default `R=2, Q=24` is ≈ 2^-24** (theoretical `(√ρ)^Q = 2^-24`
   agrees). The real-layer driver (`p3_private_fc_llama`, `R=1, Q=64`) is stronger:
   `ρ=1/2`, `(√ρ)^Q = 2^-32` (≈ 2^-44 under the more optimistic unique-decoding bound).

**Bottom line.** Overall soundness = term1 + term2 ≈ **term2** (it dominates):
`~2^-24` at the p3_sweep defaults, `~2^-32…2^-44` at the llama driver's settings. The advertised
`2^-58` describes only the algebraic term and **overstates the system soundness by 15–35 bits.**
Moving to GL2 fixes term1 but does *nothing* for term2 — the query bottleneck is a function of the
code rate and query count, not the challenge field. To reach a ~2^-100 statistical soundness you
need roughly `Q ≈ 100/(-log2 √ρ)`: `Q ≈ 100` at `R=2` or `Q ≈ 200` at `R=1`. This is not a bug (Q is
a knob, and the privacy `safe()` guard permits large Q on real-size layers), but the shipped
configurations should be re-labeled and, for production, `Q` raised to the target security level
(proof size and verify time scale linearly in `Q`; see §7 cost table).

**Forgery attempts (mask-sufficient shape B=16,IN=256,OUT=64, `forge2.cu`) — all correctly rejected:**

| attack | verifies? (want 0) |
|---|---|
| honest prover on a FALSE `Y≠X·W` | 0 ✓ |
| `qr` desynced from committed `rootQ` | 0 ✓ |
| `Sq` (claimed mask-sum) tampered | 0 ✓ |
| `pf.qr` field tampered | 0 ✓ |
| drop W-opening queries 24→10 | 0 ✓ |
| honest TRUE statement | 1 ✓ |

The committed-mask fix (`rootQ`/`qr`, opened at the sumcheck point) genuinely closes the
"free-scalar `qr`" forge that a prior red-team found. I could not break the *algebraic* soundness of
the protocol; the residual soundness concern is purely the query-count calibration above.

---

## 5. Finding F4 — scope: this is a private-matmul gadget, not an end-to-end verified-inference system  [VERIFIED]

`p3pfc::verify(pf, Q_pub, R_pub)` takes the matrix dimensions **and all three commitments from the
proof itself**. `grep` confirms there is nowhere in `p3_*` that pins `rootW` to a *published* weight
commitment, nor that ties `X` to public input tokens or `Y` to public output tokens (the only match
for "pinned/committed weight" is a code comment about `Q,R` in the test). So what this code proves
is:

> "I know `X, W, Y` of the claimed shape with `Y = X·W`, consistent with these three fresh Merkle roots."

That is a genuine, sound (modulo §4) **zero-ish-knowledge private matmul**. But the threat-model
claim — *"the public output tokens came from a **pre-committed** model on public inputs"* — requires
three bindings that are **not present** in the reviewed code:

- `rootW` ≟ the model's published weight commitment (so it's *the* model, not any weights);
- `X` ↔ public input activations (an equality/opening against public data);
- `Y` ↔ the public output (currently `Y` is *hidden*, so it cannot equal a public token vector
  without an extra opening; and per §3 that opening would itself leak).

`p3_private_fc_llama.cu` runs one real layer but still generates `rootW` fresh and never checks it
against a commitment. **Verdict:** the matmul relation is proven; the "came-from-the-committed-model"
and input/output binding are unimplemented. This is fine as a research building block but must not be
described as end-to-end verifiable inference yet. (This is orthogonal to F2/F3 and cheaper to fix:
pass the expected `rootW` into `verify` and reject on mismatch; add public-value opening for the
first/last layer.)

---

## 6. State of the system today (Task 2)  [VERIFIED numbers]

**What it is:** a GPU, transparent (no trusted setup), hash-based (Goldilocks + Basefold/FRI +
SHA-256 Merkle) argument that a single fully-connected layer `Y = X·W` over the field was computed
correctly, with the operands committed. Fiat-Shamir over a clean SHA-256 transcript.

**Soundness of the matmul relation: YES (with a parameter caveat).** Every self-test and the
capstone battery pass; I could not forge a false `Y=X·W`, and the committed-mask fix defeats the
`qr` forge. The *algebraic* soundness is ~2^-58 (base field). **But the binding system soundness is
the FRI-query term**, ≈ 2^-24 at the p3_sweep defaults (R=2,Q=24) and ≈ 2^-32…2^-44 at the llama
driver (R=1,Q=64) — real but well short of cryptographic 100+ bits; raise Q for production (§4).

**Zero-knowledge: NO.** Confirmed (§3). The Basefold opening's last sumcheck round reveals an
unmasked, publicly-known linear functional of each hidden operand. One proof → one linear equation
per operand; ~|W| proofs of the same weights → full weight recovery by linear algebra. The
augmentation/mask hides the *eval value* and rounds 0..v-2, but not the final (ex-bit) round.

**End-to-end verifiable inference: NO (§5).** No binding of `rootW` to a published weight
commitment; inputs/outputs not tied to public values; `Y` is hidden. It is a private-matmul gadget,
not yet a "these tokens came from the committed model" system. Non-linearities (softmax/RMSNorm/
RoPE/GLU) exist only in the older curve-based `zkob_*` code, **not** in the P3 hash-based path — so
the P3 system as reviewed proves one FC layer, not a transformer block.

**Measured cost (RTX 4090, reproduced by me):**

| what | shape | prove | verify | proof |
|---|---|---|---|---|
| real llama-68m up_proj | 768→3072, B=16, R1 Q64 | **303 ms** | **33.9 ms** | **2.47 MB** |
| 16.7M-param W | 4096→4096, B=64, R2 Q24 | 1.90 s | 19 ms | 1.27 MB |
| 1M-param W | 1024→1024, B=64, R2 Q24 | 150 ms | 14 ms | 0.97 MB |

Prove is dominated by SHA-256 Merkle (commit 159 ms + opening-merkle 49 ms of the 303 ms). Verify
and proof size scale with `Q`. Above ~16–17M-param W at R=2 on 24 GB the prover OOMs (F1); the fix I
applied now reports this instead of returning garbage.

---

## 7. How to make it much faster (Task 4)  [REASONED, grounded in measured breakdown]

The measured bottleneck is **SHA-256 Merkle hashing** (~35–50 % of prove), then the RS-encode NTT.
Prioritized:

1. **Higher FRI/Basefold folding factor (fold-by-4 or -8), and block/cap the Merkle leaves**
   (hash `k` codeword elements per leaf, `k` = query granularity). Fold-by-4 halves the number of
   round-trees and roots; block leaves cut leaf-hash count by `k`. Together ~**2–3× less hashing**
   and materially smaller proofs (fewer paths). *Highest ROI, standard, low risk.*
2. **Batch all operands (and all layers) under one commitment + one opening.** Combine X/W/Y (and
   every layer's polynomials) via a random-linear-combination batched Basefold so a *single* set of
   `Q` FRI queries authenticates everything. Cuts the current "3 commits + 3 openings per layer" to
   ≈1, and shares query cost across the whole model — near **N×** fewer query paths for an N-poly
   model. *Highest ROI for multi-layer.*
3. **Swap FRI/Basefold for WHIR (or STIR).** Same hash-PCS family, but far fewer queries for equal
   soundness and a much cheaper verifier + **smaller proofs** (the 2.5 MB → likely a few hundred KB).
   Also directly mitigates F3 (query soundness). *High ROI, moderate effort.*
4. **Faster hash.** SHA-256 is not the natural FRI hash. On GPU, a Blake3 or a tuned Keccak leaf/
   node kernel, or a field-native Poseidon2 (only if you later want recursion), can be **1.5–3×**
   faster than the current single-block SHA-256 compress. Fuse the RS-encode NTT output directly
   into the leaf-hash kernel (no separate pass / no D2H).
5. **Free device Merkle trees between operands and cap the mempool** (also fixes F1's OOM cliff):
   lets much larger layers fit, and enables multi-stream concurrent commits of X/W/Y (3× overlap on
   the commit phase).
6. **Keep the opening fully device-resident** (the non-`_dev` `prove_eval` path still does host
   `build_eq`/sumcheck for small `z`; route everything through `prove_eval_dev`), and replace the
   per-round 3× tiny D2H of the sumcheck partials with a single fused reduction.
7. **Lower rate but batch queries:** R=1 (already used by the llama driver) minimizes commit blow-up;
   pair with WHIR to keep query count sane.

Rough cumulative: (1)+(4) ≈ 2–4× on a single layer (prove ~80–150 ms); (2)+(3) turn a full model
from "sum of per-layer proofs" into one amortized proof with a fraction of the paths and a
sub-MB total proof. None of these change the soundness/ZK story — F2 and F3 are orthogonal.

---

## 8. Where this sits vs known zkML / ZK-inference systems (Task 3)  [REASONED — from public literature, knowledge cutoff Jan 2026]

This repo is a descendant of **zkLLM** (Sun–Zhao–Zhang, CCS 2024): the `bls12-381.cuh`, `zkfc.cuh`,
`zkob_*`, `tlookup.cuh` files are the original curve-based zkLLM; the **P3 layer is a rewrite that
replaces the elliptic-curve (Hyrax/IPA-style) PCS with a transparent hash-based Goldilocks
Basefold/FRI PCS** and adds the augmentation/mask ambition of *hiding weights*. That reframing is
the interesting delta.

- **vs zkLLM (original):** zkLLM commits weights and proves correctness but does **not** try to hide
  the weights cryptographically the way this does; it uses tlookup for nonlinearities and a
  curve-based commitment (not transparent, not post-quantum). This repo is transparent/PQ and aims
  at weight-privacy, but (a) only covers the matmul so far, (b) ZK is broken (F2). Per-layer prove
  time is the same order (hundreds of ms–seconds on a GPU).
- **vs EZKL (halo2 + KZG):** EZKL has tiny proofs (few KB) and cheap verify but a **trusted setup**,
  a slower prover, and struggles at LLM scale. This system is transparent and GPU-fast per layer but
  has **~1000× larger proofs** (hash-PCS tax) — the classic FRI-vs-KZG trade.
- **vs Lagrange DeepProve / Polyhedra Expander-zkPyTorch (sumcheck+GKR):** these are the current
  speed leaders for ML inference (GKR keeps the prover near the cost of inference, transparent,
  large-but-manageable proofs). This repo is in the same algebraic family (sumcheck) but less
  mature: single-layer, no GKR layer-chaining, no batched commitment. Expander/DeepProve would
  out-throughput it on a full model today.
- **vs Jolt / zkVMs (Lasso/Twist-Shout):** zkVMs prove *arbitrary* execution via lookups; running a
  GEMM inside a VM is far more expensive than a native sumcheck matmul. This system's native-matmul
  approach is the right call for the linear layers; a zkVM would only make sense for control-flow /
  glue.
- **vs Binius / WHIR / Ligero-Brakedown (hash-PCS frontier):** same transparent family as this repo.
  **WHIR** in particular would be a strict upgrade to the PCS here (fewer queries, smaller proofs,
  faster verify). **Binius** (binary tower fields) is attractive if the computation is re-expressed
  bit-wise (relevant to the fp8 question, §9), but Goldilocks is the pragmatic choice for
  integer-matmul-heavy work.

**Summary placement:** transparent + PQ + GPU-fast-per-layer + *aspires* to weight-privacy (unique),
but today: matmul-only, ZK-broken, large proofs, no end-to-end binding, soundness under-parameterized.
Its genuinely novel angle is weight-hiding via augmentation; that angle needs the opening made ZK
(§3 fix) to be real, at which point it would occupy a niche none of the above fully cover (transparent
*weight-private* inference).

---

## 9. An efficient ZKP for the EXACT fp8 computation (Task 5)  [REASONED / DESIGN]

**The gap.** Serving uses fp8 (E4M3) products with **lossy, order-dependent fp32 accumulation**; the
current proof asserts an **integer, exact-accumulation** matmul. The two differ by per-add rounding
and accumulation order — the residual ~0.3–0.4 bit/token covert channel. To make *served == proven*
you must prove the computation the hardware actually does, or change the hardware's computation to
one that is cheap to prove. Three designs, cheapest first.

### Design 1 (recommended when you control serving): align serving to an exact accumulator → reuse today's integer proof. Overhead ≈ **1×**.
E4M3 has 256 codes; a product of two E4M3 values is an exact dyadic rational with a bounded exponent
(product magnitudes span ≈ 2^-18…2^18). A **Kulisch / fixed-point exact accumulator** of ≈ 40–50
bits holds the full inner product with **zero intermediate rounding**, order-independently. If you
serve with such an accumulator (or equivalently pre-scale each MX block to a common integer grid and
accumulate in int64), the served result is *defined by* an exact integer inner product — which is
**exactly what `p3_private_fc` already proves.** Marginal proof cost over today: only the fp8→integer
**decode lookup** (256-entry table, one `logup`/tlookup per operand element; the machinery exists in
`zkob_lookup.cuh`). This closes the channel by construction. Cost: a small serving-accuracy change
vs "real" fp32-order accumulation (usually negligible and often *more* accurate). **If you can move
serving, do this.**

### Design 2 (recommended when you must match MXFP8 hardware): block-structured proof. Overhead ≈ **2–4×**.
OCP **MXFP8** shares one E8M0 scale per block of 32 elements. Within a block every element has the
same scale, so after factoring the shared exponent the 32 products are **integer-exact and sum with
no intermediate rounding** — i.e. the within-block inner product is exactly the sumcheck matmul we
already have (cheap, native). Only the **block-partial-sum combine** is done in fp32 with
round-to-nearest-even. So:
- **fp8 decode:** 256-entry lookup per operand element → sign/significand/exp.
- **within-block:** integer multiply-accumulate proven by the existing sumcheck (no new cost class).
- **cross-block combine:** `N/32` fp32 RNE additions per output (128 per element at N=4096 vs 4096).
  Each RNE proven with a small gadget: commit significand + guard/round/sticky, one range-check /
  tiny lookup for the rounding decision and the exponent alignment shift.

The expensive class (per-add rounding proofs) drops **32×** relative to proving every add. Net
overhead over the integer proof ≈ **2–4×** (decode lookups + `N/32` round gadgets), and served==proven
if the deployment is MXFP8 — which is where Blackwell-class fp8 serving is heading anyway.

### Design 3 (only if forced to match arbitrary fp32-order fp8): full per-add RNE circuit. Overhead ≈ **10–50×**.
If accumulation is plain fp32 in a fixed hardware tree order with a rounding after **every** add,
you must prove `N` RNE roundings per output element (≈ 3.8e7 for the llama layer). Structure it as a
**layered GKR circuit over the accumulation tree** (each fp32 add = one gate proven with a
round/align lookup), keeping the fp8 decode and rounding as lookups. This is dominated by the `N`
per-element round gadgets → ~an order of magnitude more committed data than the operands → prove
**10–50×** the integer proof, larger proofs, similar verify. Not recommended unless the exact
hardware order is a hard requirement.

### Substrate choice.
Keep the matmul in **sumcheck/GKR** (its native, cheap form) — a pure lookup-circuit / zkVM
rendering of GEMM would be far worse. Use **lookups only** for the two genuinely table-shaped pieces:
the 256-entry fp8 decode and the RNE rounding decision. **Binius (binary towers)** is worth a look
specifically for the bit-level rounding/mantissa work if Design 3 is ever required, but for
Designs 1–2 stay on Goldilocks and reuse everything here.

### Recommendation.
Pursue **Design 1** if serving is under your control (channel closed at ~1× cost), else **Design 2**
(MXFP8, ~2–4×). Both keep the fast sumcheck matmul and add only lookups; Design 3's 10–50× is a last
resort. Note: none of this fixes ZK — the fp8-exact proof must *also* carry the §3 opening fix, or it
leaks the (now fp8) weights just the same.

---
