# Threat-model pins and the capacity-budget framing

Status: PINNED 2026-06-10 (coordinator); **§1 decoding regime CORRECTED 2026-06-13** (see
the reconciliation summary below). These decisions shape what the experiment measures and
how results are reported. Change only deliberately, in writing, here.

## 0. Regime reconciliation (2026-06-13) — READ FIRST

An earlier version of §1 pinned **greedy** decoding (served = argmax(logits)). That was an
oversimplification and is corrected here. The protocol — and the entire covert-capacity
analysis — actually operates in the **verifiable sampled-decoding regime** (shared-seed
Gumbel-max, the DiFR setting): the datacenter samples each token as
`argmax_v(logits[v] + T·g_σ[v])`, where `g_σ` is a public function of a **committed** seed
σ. **The committed seed is what closes the sampling-randomness channel** (an *uncommitted*
sampler RNG would be a steganographic channel that dominates everything). The covert
capacity we measure is the *residual* after that channel is closed: the verifier
reconstructs post-Gumbel scores (proven `M_int` logits + committed-seed Gumbel) and bounds
each served token's margin from the post-Gumbel argmax; capacity = bits hidden by deviating
within the tolerated margin. **Greedy (T→0) is the degenerate special case** of this regime
(no sampler freedom), not a separate model. Temperature T is a parameter; the headline
numbers are at **T=1**, and `CAPACITY_TEMPERATURE.md` shows the faithful worst-case capacity
is **nearly T-independent** (0.38–0.45 bits/tok over T ∈ [0.05, 2.0], within ±9 % of the
T=1 value) — it does *not* vanish at greedy, because the served FP8 model and the proven
`M_int` deterministically disagree at ~5.5 % of positions (an integerization gap, not a
sampling channel). See also the `statement.logit_binding` follow-up flagged in
`STAGE3_FAITHFUL_DESIGN.md §3.3`.

## 1. The decoding/sampling assumption (CORRECTED 2026-06-13)

The system proves that committed weights applied to public inputs produce specific
**logits**. Tokens are what the network observer sees. The link between logits and tokens
is therefore part of the threat model, not a detail:

- **Regime: verifiable sampled decoding (shared-seed Gumbel-max).** The served token is
  `argmax_v(logits[v] + T·g_σ[v])` with `g_σ` a public, reproducible Gumbel draw from a
  **committed** seed σ. The verifier reconstructs the post-Gumbel scores from the *proven*
  logits and the committed seed and checks the served token's DiFR logit-margin against
  them. Greedy decoding is exactly the **T→0** limit of this rule.
- **The committed seed is load-bearing.** With a *free* (uncommitted) sampler, the RNG is
  itself a covert channel of ~entropy(distribution) bits per token (steganographic
  sampling), which dwarfs everything the ZK layer controls and no proof of logits
  constrains. Committing σ at registration and deriving `g_σ` deterministically — proving
  `tokens = argmax(logits + T·g_σ)` — **closes that channel**; this is the regime we
  analyse, not future work. (The PRF-seed "verifiable sampling" framing is the same
  mechanism.)
- Consequence for the experiment: covert-capacity numbers are the **residual within the
  DiFR margin** after the sampling channel is closed, reported per served token at **T=1**
  (sampled regime); the channel is exercised through margin abuse / argmax flips of the
  post-Gumbel scores. `CAPACITY_TEMPERATURE.md` reports the full T-sensitivity.

## 2. The headline output: one capacity-budget table (PINNED)

The experiment's primary deliverable is a single table that puts every tolerance in the
system on the same footing, converting each into **bits per served token actually
extractable by the network observer**:

| channel | where the freedom lives | tensor-level freedom | token-level capacity (measured) |
|---|---|---|---|
| rmsnorm inverse advice | ±1 bracket on R, per row, 4-5 norm sites | ~1.6 bits/row (proven ceiling) | TBD — measurement running (TOKEN_CAPACITY.md) |
| softmax | none (zero-advice design) | 0 (proven) | 0 |
| rescale remainders | none (exact range + affine link) | 0 (proven) | 0 |
| matmul / skip / glu | none (exact) | 0 (proven) | 0 |
| RoPE binding | design pending | TBD | TBD |
| **DiFR logit-margin tolerance δ** | acceptance threshold of the token check | n/a (defined at the logit level) | TBD as a CURVE in δ — same measurement run |
| sampling randomness | **closed by the committed seed σ** (verifiable sampled regime, §1) | — | **0** (would dominate if σ were free; committing it removes the channel — §0/§1) |

Reporting rules:
- **All token-level capacities are reported in the verifiable sampled regime at T=1**
  (post-Gumbel score = `logits + 1·g_σ`, committed seed). T is a parameter; the headline
  numbers are at T=1 and `CAPACITY_TEMPERATURE.md` shows they are ~T-insensitive
  (±9 % over T ∈ [0.05, 2.0]). They are the residual *within the DiFR margin* after the
  committed seed closes the sampling channel — not a greedy-only number.
- Tensor-level freedom (what the ZK verifier tolerates) and token-level capacity (what an
  adversary can push through argmax) are DIFFERENT numbers and both get reported. The
  token-level one is the threat-model-relevant result; the tensor-level one is the proven
  ceiling.
- The DiFR tolerance is a channel like any other and competes in the same table. If it
  dominates the ZK-side floors, that IS the result (and motivates tightening DiFR, not the
  proofs).
- The Pareto curve (DiFR score vs proving overhead) is the secondary deliverable; each
  hill-climbing submission lands one point, and the capacity table is recomputed at the
  frontier.

## 3. Known caveats to carry into any writeup
- Weight privacy: DONE and audited SOUND (Stage-D rebuild, REBUILD_AUDIT.md 2026-06-12).
  The WEIGHT-path openings are hiding/ZK — per-row blinded Pedersen (s_r·H), masked weight
  claims in a separate accumulator, and a blinded (Bulletproofs-style ZK) IPA on the final
  weight opening; the leak-scan finds no image of the hidden weight-MLE evals in the proof
  surface (positive control passes). So the property achieved is a *weight-hiding* succinct
  argument. CAVEAT (still true): ACTIVATIONS are NOT hidden — activation-path commitments
  are deterministic and their IPA openings are UNBLINDED, so guess-and-confirm against them
  is possible; inputs/logits are public by design; and the final-norm gain `g` is recoverable
  from public claims. Full input/activation privacy is out of scope (would need hidden
  activations, not just hidden weights).
- Scale: all numbers are llama-68m @ seq 1024 (prefill framing, not autoregressive
  decode). The writeup gets a cost-model extrapolation section, not a claim of measured
  large-scale numbers.
