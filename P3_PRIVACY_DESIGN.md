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

## KEY RESULT: activation-private matmul forces joint arithmetization
Rigorously: with a hash PCS (no homomorphism) you CANNOT keep the operand evaluations hidden by
independently masking X, W, Y. Masking each operand (X'=X+rX etc.) breaks the relation: Y' != X'*W'.
The product relation on hidden values therefore cannot be checked operand-by-operand without a
homomorphic commitment (what the BLS Pedersen path used). The correct hash-PCS route is to
ARITHMETIZE the whole layer as ONE statement and randomize the witness trace (ethSTARK-style random
rows) so the Q revealed positions are masked -- activations are then private witness, hidden by the
ZK property of the single proof. This is standard ZK-STARK practice, not a new problem.

## What is now BUILT + validated (2026-06-21)
- Salted hiding Merkle (p3_zk.cuh, 8/8): commitment hiding for small-domain data.
- **ZK-sumcheck** (p3_zksumcheck.cuh, test 10/10): Libra-style mask; hides every round message and
  the final value. Validated by an HVZK SIMULATOR that produces accepting transcripts with NO
  witness, distributed identically to real (chi-sq 277 vs 266, uniform), PLUS a NEGATIVE CONTROL
  (masking off -> chi-sq 5.12e6, witness-dependent) proving the test detects leakage.
- **ZK query-opening hiding** (p3_zkopen_test.cu, 4/4): mask-combine makes revealed codeword values
  uniform & witness-independent; negative control (no mask -> chi-sq 5.12e6) proves teeth. (Binding
  of the combined codeword to a STABLE registered commitment is random-oracle-model -- the simulator
  programs the RO -- so it is argued, not unit-testable; the HIDING necessary condition IS tested.)

## Honest assessment
- salted leaves, ZK-sumcheck, ZK query-opening hiding: BUILT + validated (above). Note the ZK
  property is, by definition, about a simulator -- the ZK-sumcheck IS simulator-validated; the
  query-opening's binding is RO-model (argued), its hiding necessary-condition is tested. Neither
  replaces a cryptographic review before an external claim, but both are real, working primitives.
- Remaining = COMPOSITION: arithmetize the FC layer as one statement, add ethSTARK random rows,
  and drive it with the two validated masks above. This is "build a mini ZK-STARK for the layer" --
  a substantial but well-trodden engineering build (no open theory). Its query-binding ZK lives in
  the random-oracle model (standard for FRI/STARKs), so that part is argued, not unit-testable.
- Soundness (independent of privacy): GL2 degree-2 extension DONE for the opening keystone
  (~2^-116); apply the same retype to the matmul/zk path.

## Recommendation / status
Speed: DONE (P3.1-P3.6, ~47x/120x/270x). Soundness: GL2 DONE for the opening. Privacy: the
commitment-hiding layer and BOTH ZK leakage-channel masks are built and validated with honest,
negative-controlled tests. The remaining work is the joint-arithmetization COMPOSITION (a mini
ZK-STARK for the layer) -- engineering, not research -- plus a cryptographic review before any
external ZK/privacy claim. REMINDER (standing): no external ZK/privacy claims pre-review
(see memory: zk-ezkl-privacy-caveat).

## CAPSTONE ASSEMBLY RECIPE (all sub-mechanisms validated 2026-06-21)
Every piece below has a passing, negative-controlled test; the capstone is their assembly.
private-FC prover for Y=X.W hiding weights AND activations:
1. Augment X->X^, W->W^, Y->Y^ with an extra high "ex" variable: ex=0 slice = real data,
   ex=1 slice = fresh random. (p3_maskslice_test: validated.)
2. Commit X^, W^, Y^ with salted hiding Merkle. (p3_zk: validated.)
3. Matmul sumcheck over (ex_X, ex_W, j) with the summand carrying eq(ex_X,0)*eq(ex_W,0) so the
   SUM reads only the real slices => constraint is exactly Y=X.W (soundness). Run it MASKED via
   the ZK matmul-sumcheck (p3_zkmatmul: validated) so partial sums leak nothing.
4. Open X^,W^,Y^ at the random ex points (rex_X, ri, rj) etc. via ZK query-masked Basefold
   openings (p3_zkopen + p3_basefold/_gl2: validated). The opened values are uniform (mask slice)
   => hide the boundary evaluations. Verifier multiplies the opened operands directly (correct
   because they are the true augmented-poly values at those points; the sumcheck ties them,
   weighted by public (1-rex)); hiding comes from uniformity, NOT from blocking the multiply.
5. Draw all challenges from GL2 (p3_gl2/_basefold_gl2: validated) for ~2^-116 soundness.
Remaining work = ONLY the integration code (rewrite the matmul sumcheck over augmented polys with
the two ex-variables + thread the openings); no unproven primitive remains. Query-binding ZK is
random-oracle-model (standard for FRI/STARKs).
