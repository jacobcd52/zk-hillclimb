# How to make the prover faster — synthesis of two independent brainstorms (gpt-5.5-pro + Opus)

Both were given the timing breakdown (prover dominated by COMMIT + OPEN of the weight matrix W:
SHA-256 Merkle + NTT + FRI fold; matmul sumcheck ~free). They converged hard. Ranked by
speedup-per-effort:

## CONSENSUS TOP WINS (both ranked these #1–#3)

1. **Amortize / preprocess the W commitment; batch many tokens per proof.**  (BOTH #1)
   W doesn't change between forward passes — commit it ONCE (NTT+Merkle), cache the codeword + tree,
   reuse the root every proof; skip the per-proof W NTT + initial Merkle. And B=16 leaves huge
   amortization on the table — the W opening is ~independent of batch, so larger B (prompt/multi-request
   batching) gives near-linear per-token throughput.
   Win: ~2–5× single-proof immediately; 8–100× per-token amortized. Tradeoff: needs the reusable
   commitment to be hiding + query-masked (so repeated openings don't leak W). Effort: days. **Free-ish,
   biggest lever.**

2. **Kill the SHA-256 hashing cost** (the dominant sub-cost). Three stacking sub-ideas:
   - **Packed/wide Merkle leaves** (8–16 field elements per leaf, verifier samples a *symbol* index):
     leaf count ÷8–16, tree height −3–4, hash work ÷several. Opus found we currently SHA-256 an
     **8-byte** leaf into a 64-byte block → ~87% of the work is padding. Win 2–6×, ~free.
   - **Replace SHA-256** with Blake3 (conservative, ~3–5×) or Poseidon2/Monolith over Goldilocks
     (native field, ~5–10×, also enables cheap recursion later). Tradeoff: algebraic-hash needs review;
     Blake3 is the safe interim. Effort days–2 weeks.
   - **Checkpointed / streamed Merkle** (store only subtree roots + top tree; recompute queried paths):
     full-tree memory 16 GiB → <1 GiB → **breaks the 24 GB OOM** for 7B-class layers. ~free, days.
   Combined: ~5–15× on the dominant cost AND removes the memory wall.

3. **NTT engineering** (shared-memory radix-2^k / Stockham four-step, fused bit-reverse, cached
   twiddles — currently rebuilt per call, skip the trivial padded-zero first stage). 2–5× on the NTT
   term. ~free, ~1 week (cheap wins in 1 day).

## HIGH-VALUE, mostly free

4. **Avoid power-of-two padding by tiling/chunking dimensions** (gpt-5.5-pro, unique & important):
   real LLM dims aren't powers of 2, so we pad 768→1024 (1.78×) and 11008→16384 (1.49×) — pure waste
   hitting NTT+Merkle+FRI+memory. Split into power-of-two chunks (11008 = 8192+2048+512+256) and prove
   tiles; extra sumchecks are ~free. **1.3–1.8× free.** (Or mixed-radix NTT — Goldilocks has factors
   3,5,17,257,65537 — but chunking is simpler.) Effort: days.

5. **Move host "prep" contractions to GPU + kernel fusion + CUDA graphs + preallocated buffers.**
   ~70 ms prep → ~10 ms; kills launch/alloc overhead. ~10–15% total, free, days.

6. **Batch the 4 openings (X, W, Y, mask) into one FRI** via random linear combination. 1.5–4× on the
   opening phase, ~free (standard), days.

7. **Higher-arity Basefold** (fold k variables/round → folded Merkle work ~N/(2^k−1) instead of ~N).
   2–7× on the opening's per-round Merkle. Tradeoff: more siblings per query, FRI-soundness constants.
   1–3 weeks.

## REAL TRADEOFFS (do later / with care)

8. **Shrink/redesign the 2× privacy augmentation** (query-masked FRI + O(#openings) blinding instead of
   doubling the witness). Up to **2×** on the whole dominant path + half the memory. But soundness/ZK
   sensitive (this is exactly the privacy hole the red-teams found) — weeks + audit.
9. **Lower blowup / higher code rate** (2→1.5 via factor-3 root = 1.33×; 1.25 = 1.6× but many more
   queries). Real soundness/proof-size tradeoff; mixed-radix NTT/FRI; 1–3 weeks. Note: base-field
   challenges need an extension field for true 128-bit soundness (the GL2 item).
10. **Quantization + lookups** (gpt-5.5-pro, unique): commit packed int8/int4 weights (8–16× smaller
    than 64-bit field elements) + prove decompression via a lookup. Only wins if the lookup is cheaper
    than the saved PCS work; real arithmetic-semantics complexity. Weeks.
11. **Smaller 31/32-bit field + extension challenges** (32-bit arithmetic faster on the 4090; half the
    codeword memory). Big rewrite; needs smooth domains + extension fields. Weeks–months.

## ARCHITECTURAL (for the whole-transformer goal)
12. **GKR-chained layers** (BOTH): commit only W per layer + the model input once; carry activation
    eval-claims layer→layer via sumchecks — removes per-layer X/Y commitments. The fastest zkML stacks
    (zkLLM/Libra lineage) are built this way; "commit X,W,Y per layer, repeat" is suboptimal for full
    models. Weeks.
13. **WHIR** (FRI-family successor): better constants + faster verifier, keeps the multilinear-eval
    structure — the one same-family PCS upgrade worth evaluating. Weeks. (Brakedown/Ligero: both
    caution it trades NTT for MORE hashing + 10× bigger proofs — wrong trade since hashing dominates.)
14. **Multi-GPU** (shard W by tiles, independent tile proofs — no NVLink needed). Near-linear on the
    embarrassingly-parallel hashing/encode; also breaks OOM. Days–weeks.

## Headline (both agree)
The prover (a) **re-commits the weights on every proof** and (b) **SHA-256-hashes 8-byte leaves**.
Fixing just those two — cache/preprocess W (+batch) and Poseidon2/Blake3 + packed leaves — plus
chunking away the padding, plausibly takes the 34M-param layer from ~4.6 s to **well under 1 s**, with
per-token proving nearly free on the W term, and **no cost to transparency or post-quantum posture**
(only the standard algebraic-hash assumption, avoidable with Blake3). The real tradeoffs (lower blowup,
removing the 2× augmentation, smaller field) come after the free wins.
