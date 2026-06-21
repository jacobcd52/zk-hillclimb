# P3.5 — Privacy layer: design + honest assessment

Goal: weight privacy (W never learnable) AND activation privacy (X, Y and the
inter-layer evaluations never learnable) on top of the P3 Goldilocks+Basefold prover.
The BLS system achieved this with Pedersen homomorphism + a product sigma-proof
(Stage D / P1). Hash commitments have NO homomorphism, so that trick does not port;
this doc is the replacement plan.

## What leaks in the current (integrity) prover
1. **Commitment of small-domain data.** Merkle leaf = SHA256(8-byte value); int8
   activations/weights are brute-forceable -> guess-and-confirm. **FIXED: P3.5 salted
   leaves (p3_zk.cuh, selftest 8/8).** leaf = SHA256(value || 256-bit salt); root and
   unopened siblings reveal nothing; an opened index discloses only (value, salt) there.
2. **FRI query openings.** Each query reveals Q codeword values per round. NB these are
   RS-codeword positions = linear combinations of ALL N data entries (not raw entries),
   so Q queries leak <= Q linear constraints — but that is not zero. Needs masking.
3. **Sumcheck messages.** The per-round (s0,s1,s2) are partial sums of the witness. Needs
   masking.
4. **Evaluation claims.** The matmul reveals y_X=X~(r_i,r_j), y_W=W~(r_j,r_k),
   y_Y=Y~(r_i,r_k). For activation privacy X~ and Y~ evaluations must NOT be revealed.

## Determinism vs hiding
Inter-layer chaining needs the activation commitment to be DETERMINISTIC (byte-equality
so layer L's output commitment == layer L+1's input commitment). Hiding wants randomness.
Resolution: salts are sampled ONCE and fixed (a registered secret), so the commitment is
deterministic across proofs AND hiding. (Same idea the BLS path used with a fixed Pedersen
blinding.) Adjacent layers share the salt set for the shared activation tensor -> identical
roots -> chaining by comref still works, with hiding.

## Constructions for the remaining pieces
### (a) ZK query openings  [standard, ~mechanical]
Random-linear-combination masking (ZK-FRI, Aurora/Brakedown style): prover commits a
random mask codeword m (its own salted Merkle), verifier sends gamma, the opening runs on
f' = f + gamma*m. Because m is uniform and secret, every revealed f'-value is uniform ->
hides f. Binding of f' to the registered commitment of f: reveal nothing of f directly;
instead commit f' as the round-0 tree and prove f' = f + gamma*m by also committing m and
checking the relation only through the (masked) openings. The mask must carry more
independent randomness than the number of revealed values (Q*rounds) — satisfied by a
full-length random m. Cost: ~2x prover (one extra codeword + opening).

### (b) ZK sumcheck  [standard, ~mechanical]
Mask polynomial g committed+opened via the same PCS; reveal S_g = sum_b g(b); run the
sumcheck on h + rho*g for FS challenge rho. Round messages are blinded by rho*g_i (random);
final needs (h+rho g)(r) = h(r) + rho g(r), with g(r) from g's (masked) opening. Cost: +1
masked opening per sumcheck.

### (c) Activation privacy / evaluation-claim chaining  [RESEARCH-GRADE — the hard part]
The matmul tie checks y_Y (initial) and y_X*y_W (final). To keep these hidden:
 - Commit each evaluation claim with a salted hash: C_yX, C_yW, C_yY.
 - Prove the multiplicative relation  final_claim == y_X * y_W  and  initial == y_Y  on the
   COMMITTED claims, without opening them. Over a hash PCS this is a tiny product-relation
   sub-argument (a 1-variable sumcheck / a committed-witness product check) — the analogue
   of the BLS product sigma-proof, but built from sumcheck+Basefold instead of Pedersen.
 - Chain layers: layer L's output activation polynomial commitment == layer L+1's input
   commitment (comref / byte-equality, enabled by the registered shared salts), and the
   per-layer eval-claim commitments are linked by the rescale/relation obligations. The
   verifier never sees an activation value or an activation evaluation — only commitments
   and a relation proof.

## Honest assessment
- (1) salted leaves: DONE + validated.
- (a),(b): standard, well-understood; each is a bounded implementation (~a focused build +
  soundness selftests). Genuine zero-knowledge (the simulator argument) cannot be *proven*
  by a selftest — only soundness and structural hiding can be checked — so these warrant a
  cryptographic review before any external ZK claim.
- (c): research-grade. The product-relation-on-committed-scalars sub-argument over a hash
  PCS, plus the multi-layer eval-claim chaining, is the real work and the part most likely
  to hide subtle soundness/ZK bugs. It should be designed against the BLS Stage-D/P1
  obligations and reviewed, not rushed.
- Soundness caveat (independent of privacy): challenges are base-field (~2^-58); production
  needs a degree-2 Goldilocks extension (mechanical) for ~2^-116.

## Recommendation
Speed levers are complete and validated (P3.1-P3.6: ~47x prove / ~120x verify / ~270x proof
vs BLS). The commitment-hiding layer (salted leaves) is done. The remaining ZK opening masks
(a,b) are mechanical and can be added with soundness selftests; the activation-private
eval-claim chaining (c) is a dedicated cryptographic effort that should be specced against
the BLS P1 obligations and independently reviewed before being relied on or claimed publicly.
REMINDER (standing): do not make external ZK / privacy claims about this system without that
review (see memory: zk-ezkl-privacy-caveat).
