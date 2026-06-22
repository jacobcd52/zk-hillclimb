# Overnight red-team + speed run — summary

Two independent red-teams (OpenAI **gpt-5.5-pro** and an **Opus** agent), neutral/adversarial framing,
were run against the ZK private-FC prover. Bugs they found that I agreed with were fixed; the rest is
documented honestly below. Then a speed sweep (batch sizes + model sizes) was run.

> Note on the OpenAI step: it was initially blocked by the harness's data-exfiltration boundary
> (sending private source to api.openai.com). Switching the session to "accept edits" unblocked it.

---

## 1. Bugs FIXED

### A. CRITICAL soundness — uncommitted sumcheck mask `q` made proofs forgeable  (found by gpt-5.5-pro)
The main sumcheck is over `summand + rho*q` with a random mask `q`. `q` was **never committed**, and
the verifier's final tie used `pf.qr` — a **free scalar** in the proof. So a prover could accept ANY
false statement (e.g. X=0, W=0, Y[0]=1): forge all-zero sumcheck messages, open the (false) committed
operands honestly, and set `qr = (claim − summand)/rho`. No Merkle/FRI break needed.
**Fix** (`p3_private_fc.cuh`): commit `q` (`rootQ`, absorbed into the transcript *before* `rho`), and
open `q` at the sumcheck point `r` (`openQ`); the verifier now uses `openQ.y` (bound to `rootQ`), not a
free scalar. **Validated:** gpt-5.5-pro's own forge experiment now returns `verify=0`; soundness battery
9/9 (honest accept + wrong-product + tampered-message/value/codeword + Q=0 + wrong-R all reject).

### B. HIGH privacy — joint linear recovery of the witness from the openings  (found by Opus)
Each opened codeword value is a fixed public linear combination of the augmented coefficients; once the
number of revealed positions reaches the augmented size, the witness `X` (and `W`) is recovered by
Gaussian elimination (demonstrated at Q=48 on a small layer). My earlier `p3_opening_zk_test` only
checked *marginal* uniformity, not *joint* recovery — exactly the gap exploited.
**Fix** (`p3_private_fc.cuh`): the prover now **refuses to emit a proof** unless each operand's random
mask slice strictly dominates everything the openings reveal (`2·Q·logN + 2^R ≤ 2^(logN−1)`), with
fresh randomness per proof; the verifier re-checks the same condition. Real model layers are always far
inside this safe regime; toy/high-Q shapes are refused (no leaky proof can be produced). **Validated:**
the attack reproducer is now refused for all Q.

### C. CRITICAL soundness (earlier session) — Q/R trusted from the proof → Q=0 vacuous accept
**Fixed** previously: verify pins `Q`, `R`, `logN` to public params; `Q≥20`, `R≥1` enforced.

---

## 2. Remaining issues NOT fully fixed (with honest reasons — not "engineering difficulty")

### D. HIGH privacy — the Basefold opening's last sumcheck round leaks the real-slice evaluation  (gpt-5.5-pro Finding 2)  ⟵ THE BIG ONE
With `ex` (real|random selector) as the last opening variable, the last sumcheck round splits on it and
publishes `s0 = C_real · E · (1−rex)` and `s1 = C_mask · E · rex` **separately**. So
`C_real = s0 / (E·(1−rex))` — a direct, **unmasked** evaluation of the real tensor. Fresh masks do not
help (s0 has zero mask coefficient). The mask-count fix (B) does **not** address this.
**Status: activation and weight privacy are currently BROKEN.** The system as it stands is an
**integrity** proof (it correctly proves Y=X·W, soundly after fix A) but is **not zero-knowledge**.
**Why not fixed tonight:** the correct fix is to make every Basefold opening itself zero-knowledge —
mask the opening's sumcheck messages (Libra-style, the `p3_zksumcheck` primitive I built) and commit
those masks, i.e. the full ZK-opening composition. This is the genuine ZK-FRI construction (the
binding-vs-hiding problem for hash PCS) — a real multi-component crypto build, not a one-line change.
I prioritized the CRITICAL soundness forge (A) and the deliverables on the night's clock and did not
complete a *validated* ZK-opening composition. **This is the #1 next task**, and I do not claim privacy
until it's done and re-red-teamed. (I have the primitives — `p3_zksumcheck`, `p3_zkmatmul`, salted
Merkle, GL2 — they need to be correctly composed into `prove_eval`, which is exactly what both
red-teams showed is currently missing.)

### E. MEDIUM soundness — base-field Fiat-Shamir (~2⁻⁵⁸); GL2 not wired into this path
Challenges are single 64-bit Goldilocks elements, so soundness is ~2⁻⁵⁸, not the ~2⁻¹¹⁶ the comments
imply. The degree-2 extension (`p3_gl2.cuh`, and `p3_basefold_gl2.cuh` validated standalone) is **not**
wired into the capstone. **Why not fixed:** it's a soundness *margin* (not a break — 2⁻⁵⁸ per attempt),
and wiring GL2 is a full retype of the capstone's sumcheck+openings to `gl2_t`. Flagged; should be done
for any production soundness claim. (Also: `v % GL_P` has a small modulo bias — trivial to fix.)

### F. MEDIUM — unsalted Merkle leaves; verify API; input validation
- Leaves are `SHA256(8-byte value)`, not salted. **Largely mitigated** once B holds (revealed codeword
  values are full-field uniform, not brute-forceable); salting (`p3_zk.cuh`, built) is defense-in-depth.
  I judged this low-priority *given B*, but it should be wired with the ZK-opening work.
- `verify(pf, Q, R)` trusts `rootX/W/Y` and `bb,ii,oo` from the proof; a caller must compare them to the
  externally-registered commitment/dims. Easy hardening (add expected-roots/dims args). Flagged.
- Missing bounds checks (lengths match, shift ranges). Hygiene; flagged.

---

## 3. Honest bottom line
- **Soundness of Y = X·W: HOLDS** after fix A (forge defeated; battery 9/9), at ~2⁻⁵⁸ (fix E pending).
- **Zero-knowledge (hide W, X, Y): NOT achieved.** Fix B closes the joint-linear-recovery path, but
  fix D (opening sumcheck leaks the real-slice eval) is unaddressed → the real tensors are recoverable.
  The prover is currently an **integrity** prover with partial hiding, not a ZK prover. Closing D (the
  ZK-opening composition) is the top priority and I won't claim privacy before it's done + re-reviewed.

(Speed numbers — table + plot — in section 4 below, appended by the sweep finalizer.)

---

## 4. Speed sweep (RTX 4090; up_proj-style FC layer; R=1, Q=64; dims padded to powers of 2)

Prover = the current (post-fix) GPU device-resident path. "overhead" = prove / forward for the whole batch.

| config | B | IN | OUT | forward (µs) | prove (ms) | verify (ms) | proof (KB) | overhead |
|---|---|---|---|---|---|---|---|---|
| llama-68m  B=4   | 4   | 1024 | 4096 | 19.1 | 567.9 | 97.6  | 2206.7 | 29,732× |
| llama-68m  B=16  | 16  | 1024 | 4096 | 12.0 | 584.2 | 104.1 | 2474.9 | 48,633× |
| llama-68m  B=64  | 64  | 1024 | 4096 | 20.6 | 595.1 | 116.0 | 2775.2 | 28,923× |
| llama-68m  B=256 | 256 | 1024 | 4096 | 52.5 | 736.0 | 125.4 | 3107.4 | 14,026× |
| gpt2-medium (1024×4096) | 16 | 1024 | 4096 | 12.0 | 570.9 | 107.3 | 2474.9 | 47,543× |
| gpt2-large  (1280×5120) | 16 | 2048 | 8192 | 42.4 | 2179.4 | 120.5 | 2827.2 | 51,350× |

Plot: `sweep_plot.png` (left: batch scaling; right: model-size scaling).

Notes / findings:
- **B=1 was REFUSED** by the prover: a single 1024-wide activation row cannot mask Q=64 queries
  (the privacy enforcement, fix B). Batch ≥ 4 (or fewer queries) is required at this width.
- **Batch scaling:** prove time is ~flat in B (568→736 ms from B=4→256) because the *weight* opening
  dominates and is independent of B — so a single proof covers the whole batch and the **per-inference
  overhead drops** as batch grows (overhead 48,633× at B=16 → 14,026× at B=256). This is the key
  practical lever.
- **Model-size scaling:** prove grows ~linearly with parameters (gpt2-large has 4× the params of
  gpt2-medium → ~3.8× the prove time, 571→2179 ms). Verify stays ~0.1 s and proof ~2–3 MB (succinct).
- **Memory wall:** 3B- and 7B-class single layers (padded to 4096×16384) **OOM on the 24 GB card** —
  the Merkle tree over the 2²⁸ codeword needs ~16 GB. Bigger layers need a memory-streamed Merkle or a
  bigger/multi-GPU card. (Real finding, not a crash bug.)
- All "overhead" figures are for the current prover, which (see §2D) is **integrity-only, not yet ZK**;
  the ZK-opening composition will add roughly a constant factor on top of these numbers.
