# HANDOFF — state as of 2026-06-10 (session ending, new API-billed session takes over)

Read this FIRST, then `submissions/baseline-native/PHASE0_NOTES.md` (full pinned conventions),
then `harness/HARNESS.md`. Working build lives at **/root/zkllm** (local disk — REBUILD it on a
fresh pod by copying `zkllm-src/` there and running the build commands in PHASE0_NOTES). All
sources here in `zkllm-src/` are the canonical copies and are in sync with /root/zkllm.

## Where we are (task #12: coordinator-built prove→serialize→verify infrastructure)

DONE and validated (each binary has a `selftest` mode — ALL PASS as of this commit):
- `vrf_common.cuh` — FS transcript, IPA prove/verify, host G1/Fr helpers (h_mul/h_add/g1_eq/h_scalar).
- `zkob_fc.cu` — matmul obligation (emits int64 Y chain file, unpadded B×C).
  verify CLI order: `verify <obdir> <seed> <B> <IN> <OUT> <com_W> <gen_in> <gen_out> <q>`.
- `zkob_rescale.cu` — rescaling obligation: homomorphic affine link + logUp range proof on rem.
  Takes int64 X input (NOTE upstream `FrTensor::from_long_bin` is BROKEN — use
  `load_long_tensor` from the header; see GOTCHAS).
- `zkob_skip.cu` — skip connection, pure homomorphic adds.
- `zkob_glu.cu` — mlp.swiglu: mapping lookup (comb = G + r·S vs T_comb = table + r·mapped,
  homomorphic combined commitment) + eq-weighted hadamard sumcheck (H = S⊙U).
  Validated at real scale (B=1024, C=3072, silu table 2^22): prove 11.4s, verify 13.8s, 904KB.
- `zkob_lookup.cuh` — SHARED header (lookup recursion, IPA openings, hadamard sumcheck,
  eq-tensor builder, **degree-4 sumcheck machinery: k_hp4_step / lagrange5 / QuarticProof /
  fs_quartic — just added + compile-validated, the k_hp4_step Montgomery bug is FIXED**
  (mont-ify all factors except one; both glu and rescale selftests re-ran ALL PASS after).
  RULE: any edit to this header requires re-running EVERY driver's selftest.
- Chaining validated end-to-end at real scale: matmul→rescale and swiglu→rescale, chain
  commitment files byte-identical, all verifies ACCEPT.

## IMMEDIATE NEXT STEP: write `zkob_rmsnorm.cu` (design is FINAL, just implement it)

Purpose: close the **unbound rms_inv_temp covert channel** (upstream rmsnorm never binds the
inverse-RMS advice R). Binds R to X within ±1 integer tolerance (≤ log2(3) ≈ 1.6 bits/row
covert freedom — document as measured floor, ~6.5 Kbit/forward over 4 norms).

Pipeline semantics (m68-pipeline.py): X int32 (B×C, scale 2^16), R = round(2^16/sqrt(mean(X_real²)+eps))
(B), g int32 (C, registered weight). W = R×g outer (scale 2^32) → rescale 2^16 → W_ →
Y = W_ ⊙ X (scale 2^32) → separate zkob_rescale → out. Let M[s] = Σ_j X[s,j]² + C_eps,
C_eps = round(eps·C·2^32) (u64 CLI arg). Exact R satisfies (R−1)²·M ≤ 2^64·C ≤ (R+1)²·M.

Proof obligations (one transcript, ~17 IPA openings):
1. **Limb range lookup**: P1 = 2^64·C − (R−1)²M ≥ 0, P2 = (R+1)²M − 2^64C ≥ 0, both < 2^80.
   L = limb matrix, n_rows = max(16, 65536/B_pad) rows × B_pad cols (rows 0-4 = P1 limbs
   16-bit, 5-9 = P2 limbs, rest zero), committed with gen_B. logUp vs tLookupRange(0, 65536)
   (table contains 0 → zero pad rows fine; for B=1024: D_L = N = 65536 → n1 = 0, pure phase2).
2. **Homomorphic affine limb links** (host h_mul/h_add/g1_eq, NO openings):
   com_P1 == Σ_{i<5} 2^{16i}·com_L_row[i]; com_P2 == Σ_{i<5} 2^{16i}·com_L_row[5+i].
3. **SS sumcheck** (binds M to X): claim = M̃(u_b) − C_eps = Σ eq·X·X over logD vars with
   E_bcast = k_bcast_rows(build_eq_tensor(u_b)); reuse fs_hadamard with S=U=X_pad.
   Verifier eq_acc: factor my_eq(u_b[logB−1−k], w_k) ONLY for rounds k < logB (row bits are
   the MSBs; later rounds bind column bits, eq factor 1). Require S_f2 == U_f2; one opening
   of X at reverse(ws); opening of M̃(u_b) vs com_M (single row, gen_B, u_row empty).
4. **Bracket quartics** (two fs_quartic instances, tags "q1"/"q2", Lagrange-5):
   Σ_s eq(u_b2,s) = 1 ⇒ claim_q1 = 2^64C − P̃1(u_b2) over factors (E2, T1, T1, M), T1 = R−1;
   claim_q2 = P̃2(u_b2) + 2^64C over (E2, T2, T2, M), T2 = R+1.
   NO commitment for T1/T2: verifier opens R at pt1 expecting qp1.A_f + 1, at pt2 expecting
   qp2.A_f − 1 (MLE of constant 1 is 1). Verifier requires A_f == B_f. Openings:
   P1@u_b2, P2@u_b2, R@pt1, M@pt1, R@pt2, M@pt2.
5. **Outer product** W = R×g via MLE factorization — NO sumcheck: absorb val_R = R̃(u_b3),
   val_g = g̃(u_c3), val_W = W̃(u_pt3); check val_W == val_R·val_g; three openings.
   g opens vs the REGISTERED com_g (this also discharges the norm-weight commitment_opening id).
   u_pt3 = concat(u_c3 [low logC bits], u_b3 [high]). Padding exact: g zero-padded ⇒
   W_pad = R⊗g_pad on the whole grid (needs B == B_pad — REQUIRE it, throw otherwise).
6. **Internal rescale** W_ = rescale(W, 2^16) computed with rescaling_kernel on the UNPADDED
   B*C tensor then padded (byte-identical with the separate zkob_rescale run; orchestrator
   checks com_W==com_X and com_W_==com_Xr there). Driver commits com_W_.
7. **Hadamard** Y = W_ ⊙ X — exactly glu part 2. Openings Y@u_h, W_@pt, X@pt.
Chain outputs: W.i64 and Y.i64, UNPADDED int64 (W from host vector directly; Y via
save_long + strip-padding, copy the glu pattern).

Implementation notes (decided):
- Host math in __int128 (P1,P2 < 2^80; products (R+1)²·M < 2^124 — require M < 2^62, throw).
- Build P1/P2/M FrTensors from host Fr_t limb arrays (values exceed int32/int64 ctors):
  fr_from_u128(v) = {(u32)v, (u32)(v>>32), (u32)(v>>64), (u32)(v>>96), 0,0,0,0} — tensors
  hold PLAIN (non-Montgomery) form, same as all drivers (FrTensor(uint, const Fr_t*) ctor
  is a raw memcpy — verified).
- Fr constants: 2^64·C = {0,0,(u32)C,0,...} (require C < 2^16); limb weights 2^{16i} for
  i=0..4 = {1},{65536},{0,1},{0,65536},{0,0,1}.
- prove CLI: `prove <obdir> <seed> <X-int32> <R-int32> <g-int32> <B> <C> <C_eps-u64> <gen_C>
  <gen_B> <q> [W-i64-out Y-i64-out]`; verify: `verify <obdir> <seed> <B> <C> <C_eps>
  <com_g-path> <gen_C> <gen_B> <q>`. Prover sanity-checks the bracket on the input R and
  throws loudly if the pipeline advice is out of tolerance.
- FS schedule: absorb B, C, C_eps(lo,hi), com_X, com_g, com_R, com_M, com_P1, com_P2, com_L,
  com_m_L, com_W, com_W_, com_Y → β → com_A_L → α → u_L → limb lookup rounds + terminals +
  3 openings → u_b → ev_M → SS rounds + terminals + 2 openings → u_b2 → ev_P1, ev_P2 →
  q1 rounds, q2 rounds, terminals + 6 openings → u_b3, u_c3 → val_R/val_g/val_W + 3 openings
  → u_h → claim_Y → hadamard rounds + terminals + 3 openings.
- Selftest: B=8, C=5, X random ±2^12, exact R via __int128 bracket search (largest r with
  r²M ≤ 2^64C; double-sqrt seed + fix loops). Semantic forgeries (R'=R+2 violates bracket
  always, since honest R > r_true−1):
  evil=1 R bumped, P1 recomputed mod p (negative→wraps), limbs = low 80 bits → AFFINE LINK rejects
         (everything else stays consistent — lookup/quartics/SS/hadamard all pass);
  evil=2 R bumped, limbs honest-truncated, P1 = limb reconstruction → quartic q1 round-0 rejects
         (strict=false only for q1; q2 stays consistent);
  evil=3 M[i] += 1 with brackets recomputed from new M → SS sumcheck rejects;
  evil=4 Y bumped → hadamard rejects;  evil=5 W bumped → outer point check rejects
         (skip prover's val_W==val_R·val_g sanity for evil=5);
  plus byte tampers on lookup.bin/hpss.bin/qp1.bin/qp2.bin/outer.bin/hp.bin/com_*/ipa_*.
- Real-scale test: B=1024, C=768 → gen_C = gen_B = /tmp/gen1024.bin (regenerate with
  `./ppgen 1024 /tmp/gen1024.bin` on a fresh pod), C_eps for eps=1e-5 (CHECK the eps actually
  used in m68-pipeline.py before wiring), R from a real pipeline dump or recomputed exactly.
- After validation: persist to zkllm-src/, document as §14 in PHASE0_NOTES.md, re-run ALL
  selftests (header untouched ⇒ glu/rescale/fc/skip should not need rebuilds, but cheap).

## After rmsnorm, remaining for task #12 (in order)
(a) `zkob_softmax` — read upstream zksoftmax.cu + m68-pipeline.py attention section FIRST.
    Sketch: E = exp mapping lookup of (z − mx) reusing glu machinery; row-sum binding
    (SS-style); inverse advice R_i bracket-bound (reuse rmsnorm machinery); P = R⊙E hadamard
    + rescale. mx need not be exact max (shift-invariance; table domain bounds it).
(b) Orchestrator: registration script (ppgen gens, registered weight coms + hash), prover
    walking harness/manifest_llama68m.json over real llama-68m data, SEPARATE verifier process
    emitting transcript.json per harness/check_transcript.py, chaining via commitment-file
    byte-equality, statement obligations. mlp.swiglu is ONE manifest id = glu + hidden-rescale
    composed. NOTE: drivers do NOT mkdir the obdir — orchestrator must create it (silent no-op
    otherwise).
(c) Per-head attention matmuls via zkob_fc (scores 1024×64×1024, values 1024×1024×64,
    12 heads × 2 layers; com_W = prover-supplied chained activation commitment).
(d) commitment_opening ids emitted explicitly; (e) forgery wiring into harness/forgeries/;
(f) baseline-native passes the full soundness gate end-to-end.
Then: hill-climbing loop (Opus subagents for NON-trust-critical submissions only — the
governing directive is that the coordinator personally builds anything trust-critical).

## Standing constraints (do not violate)
- NEVER push `int-model-approximation` (or its contents) to GitHub.
- LLaMA-land only (llama-68m @ seq 1024; embed 768, hidden 3072, 12 heads, 2 layers). No Qwen.
- User is not a SWE: plain language, do everything yourself, one command at a time if they
  must act. Ask before publishing anything publicly.
- Budgets: RunPod ≤ $100 total (balance was ~$78), Anthropic key ≤ $1000.
- Pod claude-il (u7nqmd9rgvhmxy, RTX 4090, sm_89). Hourly check-in cron exists (id 34c56141).
- GOTCHAS.md: -dlto miscompiles NEW G1 kernel shapes (use 1-thread host helpers); upstream
  from_long_bin broken; never mix non-dlto objects with upstream LTO objects.
