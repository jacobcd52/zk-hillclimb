# Threat-model pins and the capacity-budget framing

Status: PINNED 2026-06-10 (coordinator). These two decisions shape what the experiment
measures and how results are reported. Change only deliberately, in writing, here.

## 1. The decoding/sampling assumption (PINNED)

The system proves that committed weights applied to public inputs produce specific
**logits**. Tokens are what the network observer sees. The link between logits and tokens
is therefore part of the threat model, not a detail:

- **Pinned assumption: greedy decoding.** The served token is argmax(logits), and the
  verifier's token-level check (DiFR logit-margin) is interpreted against that. The final
  statement obligation must bind served-token == argmax within the DiFR tolerance.
- **Sampled decoding is explicitly OUT OF SCOPE, and we say why loudly:** with free
  sampling, the sampler's randomness is itself a covert channel of ~entropy(distribution)
  bits per token (steganographic sampling), which dwarfs everything the ZK layer controls.
  No proof of logits constrains it. The known mitigation (commit to a PRF seed at
  registration; prove tokens = sample(logits, PRF(seed, position)) — "verifiable
  sampling") is compatible with our architecture but is future work. One paragraph in the
  writeup; no engineering now.
- Consequence for the experiment: all covert-capacity numbers are reported **per greedy
  token**; any channel is exercised only through argmax flips (or DiFR-margin abuse).

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
| sampling randomness | out of scope (greedy pinned, §1) | — | — (would dominate; see §1) |

Reporting rules:
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
- IPA openings are currently UNBLINDED: the "weights stay private" claim needs either the
  blinding fix (deferred, tracked) or honest wording ("succinct verification with weight
  commitment; formal ZK property pending blinding").
- Scale: all numbers are llama-68m @ seq 1024 (prefill framing, not autoregressive
  decode). The writeup gets a cost-model extrapolation section, not a claim of measured
  large-scale numbers.
