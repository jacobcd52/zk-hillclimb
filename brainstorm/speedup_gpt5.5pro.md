## Short version: biggest wins first

Your prover is a PCS prover for a huge private dense vector. The matmul argument is already cheap. So the best wins are the ones that reduce the number of expanded weight symbols you hash/encode, or amortize that work across many tokens/proofs.

| Priority | Idea | Attacks | Expected win | Tradeoff / risk | Difficulty |
|---|---|---:|---:|---|---:|
| 1 | **Preprocess/reuse the weight commitment; batch many tokens/rows per W opening** | W NTT + initial W Merkle; per-token W cost | Online layer proof: often **2–5×** wall-time reduction if W fixed. Per-token throughput can improve **8–50×** by increasing batch/sequence because W opening is nearly fixed. | Only helps when W is reused. Need a hiding reusable model commitment and query-masked FRI so repeated openings do not leak W. Opening W is still linear per proof. | days |
| 2 | **Packed Merkle leaves + checkpointed/streamed Merkle trees** | SHA-256 Merkle time and 24 GB OOM | Merkle build **2–6× faster**; full tree memory **8–64× lower**, or near-eliminated with checkpoints. For 2²⁸ symbols: tree goes from ~16 GiB to ~2 GiB with 8-symbol leaves, or <<1 GiB with checkpoints. | Mostly free if verifier samples symbol indices, not leaf indices. Slightly more opened field elements per query. | 2–7 days |
| 3 | **Avoid power-of-two padding by tiling/chunking matrix dimensions** | O(padded W) NTT+hash+FRI | LLaMA-style 4096×11008 currently padded to 4096×16384: **1.49×** less W work. 768×3072 padded to 1024×4096: **1.78×** less. Also helps OOM. | More roots/openings/sumchecks, but sumcheck is ~free and openings can be batched. Soundness unchanged. | days–1 week |
| 4 | **Replace SHA-256 with a GPU-friendly Merkle hash**: BLAKE3, Poseidon2/Monolith over Goldilocks, or optimized SHA compression | Merkle hashing | Merkle **2–8×**, total prover **1.3–3×** depending NTT share. | Non-SHA hashes require security review. Keep 256-bit digest for PQ 128-bit collision security. | days–2 weeks |
| 5 | **Raise code rate / reduce blowup**: e.g. blowup 2 → 1.5 | NTT + Merkle + FRI length | **1.33×** less codeword work for blowup 1.5. Blowup 1.25 could give **1.6×**, but needs many more queries. | Real soundness/proof-size tradeoff. Need mixed-radix NTT/FRI and soundness recalculation. | 1–3 weeks |
| 6 | **Higher-arity Basefold + batched openings** | FRI folded Merkle rounds | Opening time **2–7×** lower; total layer proof maybe **1.1–1.5×** unless W commit is preprocessed. | More symbols per query; FRI soundness constants change. | 1–3 weeks |
| 7 | **Remove global 2× ZK augmentation** using proper query-masked FRI / random-tail blinding | Everything, because current privacy doubles W | Potential **2×** speed/memory win over current augmented design. | Not free. ZK/soundness proof is subtle. Needs audit. | weeks |
| 8 | **Move host prep contractions to GPU; fuse kernels; CUDA graphs** | 70 ms prep + launch/alloc overhead | For 4M layer, save **50–80 ms**, i.e. **~10–15%**. | Free lunch engineering. | hours–days |

---

# 1. Preprocess and reuse W aggressively

If W is fixed across tokens/requests, stop treating W commitment as online work.

### What to do

For each layer/tile of W:

1. Offline:
   - Convert W to field.
   - Apply the eventual ZK blinding/masking.
   - RS/NTT encode.
   - Build packed/checkpointed Merkle commitment.
   - Publish/store the root as the model commitment.

2. Online:
   - The proof references the existing W root.
   - Skip W NTT + initial W Merkle.
   - Still do the Basefold opening for the random W evaluation point, unless you batch many uses together.

For your 4M layer, X and Y are tiny versus W, so most of the 0.36 s commit is W. Reusing W root should move a 0.57 s proof closer to ~0.2–0.3 s before other optimizations. For the 33.6M case, expect seconds saved.

### Important caveat

Precomputing only the root is not enough if you later need Merkle auth paths. Store either:

- the full packed tree,
- or better, the encoded W codeword plus Merkle checkpoints/top tree, so paths can be regenerated cheaply after Fiat-Shamir queries are known.

### Batch tokens/sequence positions

Your matmul sumcheck opens W once per layer proof, independent of batch size B. Current B=16 leaves huge amortization on the table.

For a 1024×4096-ish layer:

- B=16: W dominates.
- B=128: X/Y grow 8×, but are still much smaller than W.
- B=512/1024: W is finally amortized over many rows.

For prompt processing or multi-request batching, increasing B can give near-linear per-token throughput improvement until activation commitments become comparable to W.

Tradeoff: latency/memory. For autoregressive decoding, future tokens are sequential, but prompt tokens and multiple users batch well.

---

# 2. Packed Merkle leaves + checkpointed Merkle trees

This is the most obvious engineering/protocol “free lunch”.

Your OOM example is exactly a Merkle-layout problem:

- W = 4096×16384 = 2²⁶ field values.
- Current privacy 2× and rate-1/2 blowup 2× gives codeword size 2²⁸.
- Codeword itself: 2²⁸ × 8 B ≈ 2 GiB.
- Binary Merkle tree with 32-byte nodes: ~2 × 2²⁸ × 32 B ≈ 16 GiB.

The codeword fits. The per-symbol Merkle tree kills you.

## 2.1 Pack multiple field symbols per leaf

Instead of one Goldilocks symbol per Merkle leaf, use e.g.

```text
leaf_i = H(domain || layer_id || round_id || leaf_index || c[8i .. 8i+7])
```

Use `LEAF_ELEMS = 4, 8, or 16`.

For the 2²⁸ codeword:

| Leaf packing | Leaf count | Full binary tree memory |
|---:|---:|---:|
| 1 symbol/leaf | 2²⁸ | ~16 GiB |
| 4 symbols/leaf | 2²⁶ | ~4 GiB |
| 8 symbols/leaf | 2²⁵ | ~2 GiB |
| 16 symbols/leaf | 2²⁴ | ~1 GiB |

Hash work drops similarly. With SHA-256 specifically, packing 4 or 8 symbols amortizes padding/compression overhead.

Soundness is unchanged if the verifier samples a random **symbol index** and then opens the containing leaf. Do not sample random leaf indices unless you compensate query count.

Proof-size cost is small: opening a leaf reveals extra symbols. With Q=64, log sizes ~25–28, and 8-byte symbols, this is usually hundreds of KB, not tens of MB. Path lengths also shrink by `log2(LEAF_ELEMS)`, so proof size may barely grow.

## 2.2 Store checkpoints/top tree, not the full tree

Build the Merkle root in two levels:

1. Divide leaves into subtrees of height h, e.g. h=10–12.
2. Store only each subtree root.
3. Build and store the top tree over subtree roots.
4. Discard all lower internal nodes.

When queries are known, recompute only the queried local subtrees to recover auth paths.

Example: with 2²⁵ packed leaves and h=12:

- Number of subtree roots: 2¹³.
- Top tree memory: tiny.
- Per queried leaf local recomputation: 2¹² leaf blocks.
- Q=64: ~262k leaf-block hashes per Merkle layer worst case, tiny versus 33M leaf blocks.

This breaks the memory wall without changing the protocol. It may even speed up because you avoid writing/reading 16 GiB of digest nodes.

## 2.3 Align leaves with FRI/Basefold cosets

Choose codeword layout so symbols that are opened together by FRI folds land in the same packed leaf when possible. If current NTT output is bit-reversed or FRI siblings are separated by half-domain offsets, add a logical index mapping or reorder the codeword after NTT.

This matters for path sharing and for high-arity folding later.

---

# 3. Avoid dimension padding with tiling/chunking

Padding is silently multiplying your W cost.

Examples:

- 768×3072 actual = 2.36M.
- Padded to 1024×4096 = 4.19M.
- Overhead = **1.78×**.

For LLaMA-like FFN:

- 4096×11008 actual = 45.1M.
- Padded to 4096×16384 = 67.1M.
- Overhead = **1.49×**.

That overhead hits NTT, Merkle, Basefold, and memory.

## Immediate solution: power-of-two chunks

Instead of padding 11008 to 16384, split:

```text
11008 = 8192 + 2048 + 512 + 256
```

For 3072:

```text
3072 = 2048 + 1024
```

For 768:

```text
768 = 512 + 256
```

Then prove each tile/chunk exactly.

Column chunks are easiest: each chunk proves one slice of Y.

Row/contraction chunks require summing partial products, e.g.

```text
Y_c = X_0 W_{0,c} + X_1 W_{1,c}.
```

But your matmul sumcheck is ~1 ms, so several extra sumchecks are cheap.

Tradeoff:

- More commitments/roots.
- More openings, unless batched.
- Slightly more transcript complexity.

But soundness and ZK do not fundamentally change. For non-power-of-two LLM dimensions, this is one of the cleanest wins.

Longer-term alternative: mixed-radix domains. Goldilocks supports roots with factors 3, 5, 17, 257, 65537, so dimensions like 768=3·2⁸ and 3072=3·2¹⁰ can be represented more naturally. But mixed-radix Basefold/FRI is more work than chunking.

---

# 4. Replace or optimize SHA-256 Merkle hashing

SHA-256 is not ideal on RTX 4090. Your Merkle tree is hashing millions of tiny inputs and millions of internal nodes. Try three options behind a hash trait and benchmark.

## Option A: BLAKE3 Merkle compression

Pros:

- Very fast ARX hash.
- Good CPU verifier performance.
- Mature cryptanalysis relative to algebraic hashes.
- Natural tree compression.

Expected: Merkle **2–5×** faster than generic SHA-256 GPU code.

Use 256-bit chaining values if you want PQ 128-bit collision security.

## Option B: Poseidon2 / Monolith over Goldilocks

Pros:

- Native field hash.
- Can use high arity, e.g. absorb many Goldilocks elements per permutation.
- Often excellent in STARK provers.

Cons:

- Less conservative than SHA/BLAKE.
- CPU verifier may get slower, though verifier hash count is small.

Expected: Merkle **3–8×** if well tuned, but benchmark on 4090 because 64-bit integer throughput can be tricky.

Use at least 4 Goldilocks elements of digest for 256-bit output.

## Option C: Keep SHA but use fixed-length compression and packing

If you keep SHA-256:

- Avoid generic padded-message SHA for every leaf/internal node.
- Use a domain-separated fixed-input compression function.
- Pack 4–8 field elements per leaf.
- Consider 4-ary or 8-ary internal nodes only if proof-size tradeoff is acceptable.

Even staying with SHA, packing plus fixed compression can give a large constant-factor win.

---

# 5. Reduce blowup / increase code rate

Current rate 1/2 means 2× codeword blowup. Since prover time is linear in codeword length, reducing blowup is valuable.

For a message length n=2ᵏ:

- Blowup 2.0: codeword 2n.
- Blowup 1.5: codeword 3·2ᵏ⁻¹. Goldilocks supports factor 3. This gives **1.33×** less codeword work versus blowup 2.
- Blowup 1.25: codeword 5·2ᵏ⁻². Goldilocks supports factor 5. This gives **1.6×** less codeword work versus blowup 2, but soundness per query worsens a lot.

The right direction is not “fewer queries”. Query/path extraction is only ~5 ms. Instead, spend more queries to buy lower blowup.

Very rough query intuition:

- Rate 1/2, Q=64 gives around 64-bit proximity-style security.
- Rate 2/3 may need Q≈100–128.
- Rate 4/5 may need Q≈200+ and stronger DEEP/WHIR-style analysis.

Tradeoff:

- Less prover hashing/NTT/memory.
- Larger proof.
- Slower verifier.
- More delicate FRI soundness.

Difficulty: mixed-radix NTT and FRI domain handling, likely 1–3 weeks.

Also note: if you need true 128-bit algebraic soundness, Goldilocks base-field challenges are not enough by themselves; use extension-field challenges. That can affect folding cost.

---

# 6. Higher-arity Basefold / fewer folded Merkle rounds

Binary Basefold commits a folded codeword at every halving. The total folded-codeword size committed after the initial codeword is roughly:

```text
N/2 + N/4 + ... ≈ N.
```

Fold k variables at a time instead:

```text
arity = 2^k
```

Then folded committed size becomes approximately:

```text
N/2^k + N/2^(2k) + ... = N / (2^k - 1).
```

Examples:

| Fold variables per round | Arity | Folded Merkle work after initial |
|---:|---:|---:|
| 1 | 2 | ~N |
| 2 | 4 | ~N/3 |
| 3 | 8 | ~N/7 |
| 4 | 16 | ~N/15 |

This directly attacks your opening cost, especially the “per-round Merkle build ~52 ms” part.

Tradeoff:

- Query must reveal more sibling symbols per round.
- Proof may grow modestly.
- Need updated FRI/Basefold soundness analysis.
- Works best with packed leaves aligned to fold cosets.

Implementation difficulty: moderate protocol change, likely 1–3 weeks.

---

# 7. Batch Basefold openings

You currently have multiple Basefold openings: X, W, Y, mask, etc. Do not run independent FRI folding for every polynomial if the domains match.

For polynomials `f_1, ..., f_m` committed on the same domain and opened at compatible points:

1. Commit roots first.
2. Fiat-Shamir sample batching coefficients `β_i`.
3. Form a virtual combined oracle:

```text
F = β_1 f_1 + ... + β_m f_m.
```

4. Run one folded FRI/Basefold proof for F.
5. At queried positions, open the original codewords at round 0 so verifier can check the linear combination.

This is standard and usually safe.

Expected:

- If current “4 Basefold incl. mask” are independent, batching can make opening **1.5–4×** faster.
- Overall single-layer wall-time improvement is smaller unless W commit is already preprocessed, but still worthwhile.

Caveat: batching multiple **points** for the same MLE is harder than batching multiple polynomials at the same point, because the deterministic evaluation folds depend on the point. Group what is easy first: W with W-mask, X with X-mask, same-size operands, same-point openings.

---

# 8. Replace the 2× privacy augmentation

The current `[real | random]` doubling is extremely expensive: it doubles the W message before the 2× RS blowup, so it doubles NTT, Merkle, FRI, memory.

A better ZK PCS should avoid global 2× domain expansion. Options:

1. **Query-masked FRI**  
   Make the values revealed in FRI queries one-time padded, while preserving linear consistency checks.

2. **Random-tail / low-degree blinding**  
   Add only O(security) or O(number of openings) random low-degree degrees of freedom rather than N random entries.

3. **Masked evaluation sumcheck**  
   Keep evaluation claims masked through the matmul sumcheck and unmask only safe aggregates.

Potential win: almost exactly **2×** on the dominant W path.

But this is a real protocol change, not an engineering free lunch. The danger is subtle leakage from repeated FRI query values, or giving the prover freedom to fake masks. I would treat this as a weeks-long implementation plus proof/audit item.

---

# 9. Move prep contractions to GPU and fuse them

Your host prep is ~70 ms for the 4M layer, which is large now that sumcheck is ~1 ms.

Move these to CUDA:

- Build equality/factor arrays on device.
- Compute `A[j] = X~(r_i, j)` on device.
- Compute `B[j] = W~(j, r_k)` on device.
- Or better: fuse B-contraction directly into the first sumcheck reductions and avoid materializing B fully.

Expected: 70 ms → 5–15 ms, saving **~10%** of the current 0.57 s proof.

This is mostly free.

Also:

- Use CUDA Graphs for the repeated NTT/hash/fold kernels.
- Preallocate all max-size buffers; avoid the remaining ~10 ms alloc/upload.
- Keep transcript challenge derivation from forcing unnecessary device-host syncs.

---

# 10. Kernel fusion and NTT-specific work

The fold kernel is only ~2 ms, so do not over-optimize it first. But there are still useful fusions:

### Fold + leaf hash

Currently likely:

```text
fold kernel writes folded codeword
hash kernel reads folded codeword and builds leaves
```

Fuse so the fold kernel writes folded values and immediately hashes packed leaves. You still store folded field values for later queries/rounds, but you avoid an extra full read and kernel launch.

Expected: opening **5–15%** improvement.

### Final NTT stage + leaf packing

Harder, but valuable:

- Produce NTT output in Merkle leaf order.
- Pack `LEAF_ELEMS` consecutive field elements.
- Hash leaves immediately.

Expected: commit **5–15%** improvement.

### NTT improvements

Use/verify:

- Stockham or four-step NTT avoiding expensive global bit reversal.
- Radix-4/8 where possible.
- Twiddle layout optimized for coalesced reads.
- Batched NTT kernels for X/Y/masks to improve occupancy.
- Pruned NTT only as a stopgap for padded zeros; chunking is better because it also reduces Merkle.

Expected: NTT **1.2–1.5×**, total prover less depending hash share.

---

# 11. Alternative PCS/code choices

## Brakedown / Orion / Ligero-style linear-time codes

These avoid a huge NTT by using linear-time encodable codes.

Potential benefits:

- Remove or reduce NTT bottleneck.
- Can use high-rate codes.
- Transparent and hash-based.

Costs:

- Merkle hashing remains.
- Proof size and verifier work may grow.
- Engineering/protocol rewrite.
- Soundness/ZK details need care.

Expected: maybe **1.5–3×** if NTT is a large share and the code is implemented well. Not a guaranteed 10× because hashing the encoded W still dominates.

Difficulty: weeks to months.

## WHIR / Blaze / newer FRI-style PCS

Worth evaluating because they target exactly lower PCS constants and better batching. But they will not magically avoid reading/hashing a dense private W. Expect better constants, not a different asymptotic under hash-based transparent constraints.

Difficulty: weeks.

## Pure “sumcheck-only PCS” caution

A tempting idea is: commit to raw W with Merkle, then prove MLE evaluations by iterative folding without RS/FRI encoding.

Be careful: without a distance-amplifying code, a cheating prover can localize errors along a tiny set of positions and evade random checks. To get soundness with Q≈64, you need a code/proximity layer. So do not simply drop RS/FRI unless replacing it with Brakedown/Ligero/WHIR-style proximity.

---

# 12. Whole-transformer architecture: avoid per-layer activation commitments

For one FC layer, W dominates. For a whole transformer, committing X/Y at every layer boundary will become wasteful.

A better architecture is a monolithic or streaming GKR-style proof:

- Commit fixed weights once.
- Commit only external input/output activations, not every intermediate activation.
- Carry random evaluation claims from layer to layer via sumchecks.
- Use lookups/range arguments for nonlinearities/quantization as needed.

This does not remove the need to touch W, but it avoids turning every intermediate tensor into a fresh FRI commitment.

If your plan is “layer proof = commit X,W,Y; repeat for every layer”, I would consider that fundamentally suboptimal for full-transformer proving.

Difficulty: weeks/months, but architecturally important.

---

# 13. Quantization / lookups

If W is actually 4-bit or 8-bit quantized, committing it as Goldilocks field elements is wasteful.

Possible direction:

- Commit packed quantized weights.
- Prove decompression/range correctness via lookup.
- Prove integer/fixed-point matmul semantics.

Potential W commitment shrink:

- int8: up to **8×** smaller than 64-bit field elements.
- int4: up to **16×** smaller.

But this is only a win if the lookup/decompression proof is cheaper than the saved PCS work. For arbitrary Goldilocks W, lookups do not help the dense matmul commitment bottleneck.

Difficulty: weeks/months; real arithmetic-semantics complexity.

---

# 14. Field choice / tensor cores

Goldilocks is good for 64-bit STARK-style work and has high 2-adicity. But on RTX 4090, 64-bit integer arithmetic is not as cheap as 32-bit.

Possible future direction:

- Move to a 31/32-bit field plus extension-field challenges.
- Halve codeword memory.
- Improve NTT/fold arithmetic throughput.

Risks:

- Need sufficient smooth domain sizes.
- Need extension fields for soundness.
- May complicate representing large integer activations/products.
- Big rewrite.

Tensor cores generally do not help exact Goldilocks modular arithmetic. They only become relevant if you redesign around small-prime CRT, quantized arithmetic, or approximate-but-proven-correct fixed-point pipelines.

---

# 15. Multi-GPU

This violates your one-GPU invariant, but if relaxed:

- Shard W by tiles.
- Each GPU commits/opens its tile.
- Combine roots and sumcheck claims in transcript.

Because the workload is embarrassingly linear in W, speed can scale close to GPU count for large layers. RTX 4090 lacks NVLink, so avoid cross-GPU communication; use independent tile proofs.

Difficulty: days–weeks.

---

# What I would implement first

### Week 1

1. **Packed Merkle leaves, `LEAF_ELEMS=8`**, with symbol-level query sampling.
2. **Checkpointed Merkle tree**: store top tree/subtree roots only.
3. **No full Merkle tree allocation** anywhere.
4. **GPU prep contractions** and remove remaining alloc/upload overhead.
5. **Chunk non-power-of-two dimensions**, especially 11008 and 3072/768 cases.

This should both break the 7B-layer OOM and likely cut current single-layer wall-time substantially.

### Week 2–3

6. Add hash abstraction and benchmark:
   - optimized SHA compression,
   - BLAKE3,
   - Poseidon2/Monolith.
7. Add blowup 1.5 / rate 2/3 with mixed-radix domain.
8. Add batched openings for same-domain polys/masks.
9. Prototype higher-arity Basefold.

### Protocol track

10. Replace the 2× privacy augmentation with true query-masked ZK FRI.
11. For full transformer, move away from standalone per-layer activation commitments toward a streaming GKR/global proof.

The most “free” wins are packed/checkpointed Merkle, chunking away padding, GPU prep, and W preprocessing/batching. The real tradeoffs are lower blowup, non-SHA hashes, high-arity FRI, and removing the 2× ZK augmentation.