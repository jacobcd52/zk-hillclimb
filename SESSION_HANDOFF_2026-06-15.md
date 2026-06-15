# SESSION HANDOFF (2026-06-15, plaintext)

Detailed state dump for a fresh agent taking over. Plaintext on purpose (terminal
doesn't render LaTeX). Everything below is committed and pushed unless noted.

============================================================
0. ONE-LINE STATUS
============================================================
A sound, weight-private zero-knowledge proof of a full llama-68m forward pass is BUILT,
audited, and pushed (verify ~27s). The active work is the CAPACITY RESEARCH thread:
quantifying the covert-channel capacity an integerization leaves, and designing the
treaty-enforcement protocol + its false-positive (buffer) calibration. Right now we are
working through the *theory* of the capacity function R (method-of-types meaning).

============================================================
1. PROJECT & GOAL
============================================================
Threat model: a datacenter proves (in zero knowledge) that the tokens it served came from
a pre-committed model (architecture public; WEIGHTS PRIVATE, only a hash public), applied
to public inputs. Because the proven model is an INTEGERIZED approximation of the real
model, the gap between them is a COVERT CHANNEL: a malicious datacenter can deviate within
the tolerance and exfiltrate bits. The research question is: how many bits/token can leak,
and how do you build a treaty (audit protocol) that bounds it with a tiny false-positive
("tear up the treaty / shut down the datacenter") rate.

Two halves:
 (A) the ZKP system (engineering) -- DONE.
 (B) the covert-capacity science + treaty protocol (research) -- IN PROGRESS.

Base codebase: zkLLM (jvhs0706/zkllm-ccs2024), which was PROVER-ONLY (no verifier, no
serialization, leaky advice, slow). We built the rest. Model: JackFram/llama-68m, seq 1024,
2 layers, embed 768, 12 heads, head_dim 64.

============================================================
2. HOW TO OPERATE (environment + delegation) -- READ THIS
============================================================
- Heavy work is delegated to headless API-billed agents, NOT done in the main session
  (main session tokens are rate-limited). Launch pattern (background bash):
    set -a && source /workspace/.env && set +a && export IS_SANDBOX=1 \
      CLAUDE_CONFIG_DIR=/workspace/.claude-api && \
      /usr/bin/claude -p --model <MODEL> --dangerously-skip-permissions \
      --output-format json "$(cat /root/zkllm/PROMPT.md)" > RESULT.json 2> ERR.log < /dev/null
  The isolated CLAUDE_CONFIG_DIR=/workspace/.claude-api guarantees API-KEY billing (verify
  total_cost_usd>0 in RESULT.json). User authorized --dangerously-skip-permissions for these.
- MODELS: claude-fable-5 = smartest, for trust-critical crypto. claude-opus-4-8 = medium,
  for analysis/stats/writing. **FABLE 5 IS CURRENTLY DOWN** ("currently unavailable",
  reportedly a US-government-related issue, expected down a while). USE OPUS for everything
  meanwhile; DEFER trust-critical crypto that needs Fable.
- Agents reliably EXIT MID-RUN on long tasks (turn limit) -- they often launch a long
  selftest/walk via nohup then end. DRIVE VALIDATION YOURSELF: watch the run to completion,
  run the acceptance selftest yourself, don't trust the agent's "ALL PASS" claim.
- Transient launch failures happen (cost 0, instant death). Relaunch. (If "Credit balance
  too low" -> tell the user to top up; don't burn retries.)
- Kill stray agents by PID, never by command-line pattern (the pattern matches your own
  wrapper shells and the agents' prompts).
- Budgets: Anthropic key <= $1000 (≈$800 spent so far). RunPod <= $100.

STORAGE (important):
- /workspace = 671 TB MooseFS network volume (persistent, survives pod termination). Put
  DATA / run outputs / dumps here. It is FUSE: you CANNOT execute large binaries from it
  (they segfault).
- /root and / = 50 GB container overlay disk (ephemeral, wiped on pod re-create). BINARIES
  must live/run here (/root/zkllm). Keep run outputs OFF it (use /workspace/zkorch via
  ZKORCH_RUN_ROOT) or clean between runs.
- The pod was NOT restarted; user will restart + grow disk only if needed (it isn't, given
  /workspace).

============================================================
3. THE ZKP SYSTEM (DONE) -- what, numbers, where
============================================================
Integerization used by the PROVEN system: FIXED-POINT, weights = round(w * 2^16) -> int32,
clean int32 x int32 -> int64 matmuls + uniform power-of-2 rescales. (NOT codebook -- see §6.)

11 proof drivers, each implemented -> independently audited SOUND -> hardened -> accepted,
all with selftests incl. ~semantic forgeries that must reject at exactly the named check:
  zkob_fc (matmul), zkob_rescale, zkob_skip, zkob_glu (swiglu), zkob_rmsnorm,
  zkob_softmax, zkob_softmax8 (temp-8 variant), zkob_rope, zkob_headslice, zkob_headmerge,
  zkob_rowmax (exact argmax/row-max, causal+vpad).
Plus zkob_batchopen (the batched-opening transport) and shared headers:
  vrf_common.cuh, zkob_lookup.cuh  (PROTECTED -- never edit without full all-driver
  re-validation), zkob_claims.cuh, zkob_wpriv.cuh, zkob_serve.cuh.
Build dir: /root/zkllm (local disk). Canonical source copies: zkllm-src/ in the repo.
Build: nvcc -arch=sm_89 -std=c++17 -I/usr/local/cuda/include -dc -dlto X.cu -o X.o ; then
  link with: bls12-381.o ioutils.o commitment.o fr-tensor.o g1-tensor.o proof.o zkrelu.o
  zkfc.o tlookup.o polynomial.o zksoftmax.o rescaling.o timer.o
zkllm-src/ also bundles the needed upstream .cu/.cuh/.hpp (bls12-381, commitment, fr-tensor,
g1-tensor, proof, fs_transcript.hpp, etc.) so a clean checkout builds.

Orchestrator (orchestrator/): register.py, prove_walk.py, verify_walk.py, common.py,
selftest.sh. Registration commits weights+hashes (public.json); prove_walk runs every
driver in claim mode into ONE accumulator; verify_walk is a SINGLE-PROCESS batched verifier.
Covers the full forward pass (rmsnorm, full attention incl. RoPE/headslice/softmax/headmerge,
MLP, skips, final norm, lm_head, served-token argmax binding). 65 manifest obligations.

KEY NUMBERS (full faithful-arch-v1 forward pass, RTX 4090, llama-68m seq 1024):
  prove   1062 s -> 636 s (claim-mode) [~522 s exclusive-GPU re-measure]
  verify  999.5 s -> 27.1 s  (36.9x; the headline win, from the batched-opening rebuild)
  proof   176 MB
  weight privacy: hiding Pedersen commitments + masked weight claims + ZK opening;
    leakage scan CLEAN (no weight-MLE eval in any artifact). Audited SOUND (REBUILD_AUDIT.md).
The verify rebuild was Stages A->D (batched claim accumulation + flattened-fold kernel +
me_weights GPU lift + single-process verifier + weight privacy). All committed.

S1 (proof-size shrink via canonical-affine G1 encoding) was TRIED and DROPPED: it made
proofs 176->58MB (3x) but verify 27->119s (4.4x worse). User chose to keep fast verify.
The system was reset clean to the 27s/176MB state. (The graceful-reject verify wrapper and
the F-2 production-verify kernel-probe were part of that work and were also reverted; they
are small hardening items still on the deferred list.)

============================================================
4. CAPACITY RESEARCH (results so far, all committed)
============================================================
Method: for an integerization M_int (proven) vs the served model, compute per-position
post-Gumbel margins (logits + shared-seed Gumbel; SAMPLED regime, NOT greedy) and the
within-margin token counts N_b, then the per-token capacity R(b,K) and its min over the
margin threshold b and top-K param K.

The capacity function (five terms):
  R(b,K) = H(p) + (1-p)*E[log2 N_b | within-margin]
           + p*( H(q) + (1-q)*log2 K + q*log2(V-K) )
  p = violation fraction (served token outside margin b);
  q = fraction of violations in the tail (outside top-K);
  N_b = # within-margin tokens; V = vocab (32000); K = top-K param.
"simple" variant uses p*log2(V) for the violation term (conservative / free adversary).
"top-K" variant uses measured benign q (optimistic; assumes adversary mimics benign split).

RESULTS:
- DiFR decomposition (DIFR_BASELINE_NATIVE.md, DIFR_FAITHFUL_V1.md): the integerization+proof
  machinery costs essentially nothing -- DiFR 1.35e-6 nats (p99 = 0) vs its own float target.
  A faithful-architecture fix moved DiFR 8.988 -> 0.0156 (/575) for +21% prove. The 9-nat
  gap was ARCHITECTURE (no o_proj, the line-157 head-scramble, softmax temperature 128), not
  integerization. First Pareto point measured.
- Capacity vs FP8 reference (CAPACITY_CORRECTED.md, CAPACITY_TOPK.md): worst-case min-over-b/K
  bits/token: baseline ~12.5 ; faithful 0.36 (top-K) / 0.43 (simple) ; codebook 0.23 / 0.28.
  Five-term breakdown: ~60-72% of capacity is the MODEL'S OWN near-ties, not violations.
  K-sweep U-shaped, min around K=4.
- Capacity vs FP16 (full-precision) reference (CAPACITY_FP16.md): the schemes REORDER --
  faithful collapses to 0.065 (it's integerized-from-bf16, so close to fp16), codebook RISES
  to 0.37 (built from FP8, inherits the FP8<->fp16 gap). "Best scheme depends on which model
  the threat is defined against."
- Temperature sweep (CAPACITY_TEMPERATURE.md): capacity is NEARLY T-INVARIANT (0.38-0.45
  bits/tok over T in [0.05,2.0]) and does NOT vanish at greedy (T->0) -- it's dominated by
  the temperature-independent M_int-vs-served argmax disagreement (~5.5%), not sampling
  near-ties. (This was triggered by the user catching a regime inconsistency; the capacity
  math was verified Gumbel-correct, but THREAT_MODEL_NOTES had wrongly said "greedy" --
  now corrected to verifiable sampled decoding / shared-seed Gumbel-max.)
- Buffer / treaty-threshold (BUFFER_FPR.md, THRESHOLD_CURVE.md): benign mean capacity
  mu = 0.306 (faithful) / 0.191 (codebook), on 128 prompts (~131k tokens). Cumulative
  variance grows ~7.5x faster than i.i.d. due to within-prompt AUTOCORRELATION (effective
  sample size N_eff ~ N/7.5). The treaty THRESHOLD(N) = mu + buffer(N), buffer sized so the
  false-treaty-tear-up rate <= 1e-10, is a DECREASING curve toward mu as audited tokens N
  grow (~1/sqrt(N)). threshold_curve.png: single panel (K=4), log-x powers of 10 from 1e4,
  faithful+codebook, SOLID where variance is directly observed (N<=131072) and DASHED where
  extrapolated, validated sub-exp tail (Bernstein = text-only conservative cross-check).
  Operational upshot: a single-prompt audit certifies nothing at 1e-10; need cumulative
  auditing over thousands of tokens / several prompts.

Consolidated write-up: WRITEUP.md (succinct, with embedded plots).

============================================================
5. THE TREATY PROTOCOL (design, as discussed with user)
============================================================
Regime: VERIFIABLE SAMPLED DECODING (shared-seed Gumbel-max). Served token =
argmax(logits + T*g_sigma), g_sigma a public function of a COMMITTED seed sigma. The
committed seed CLOSES the sampling-randomness channel (an uncommitted RNG would dominate).
Greedy is the T->0 degenerate case. Headline numbers at T=1.

Protocol (user's framing, broadly endorsed):
1. Calibrate: on a benign workload, find b,K minimizing R; that's the benign per-token
   capacity. The PROVER DECLARES b,K up front (solves the model-mismatch: they calibrate on
   their own deployment). 
2. Audit: the ZKP provides M_int's logits; the network tap provides served tokens; the
   verifier computes the realized per-token capacity r from served-vs-proven-logits at the
   declared b,K. 
3. Tear up the treaty if CUMULATIVE r over the audited tokens exceeds (benign mu + buffer),
   with the buffer sized for FPR <= 1e-10 (from the variance/autocorrelation analysis above).
   This is a sequential / cumulative test, not per-message.
Notes / open: r bounds AFFORDED capacity (worst case), not detected usage -- the within-margin
near-tie freedom is undetectable, so the protocol is a capacity BUDGET, not a leak detector.
There is a known "DoS attack" (an outside user feeds high-near-tie inputs to inflate r and
trip an honest datacenter) -- deferred. "Excess over honest capacity" idea was CONSIDERED and
DROPPED (user correctly noted: if proven model == served model the tokens are exactly
reproducible from logits+seed, so there is no honest capacity to subtract; capacity IS the
M_int-vs-served gap).

============================================================
6. CURRENT THEORY THREAD (where the live discussion is)
============================================================
We are nailing the method-of-types meaning of R. Established this session:
- For a MEMORYLESS model with fixed per-position alphabet L_i: #messages = product L_i =
  exp(n * (1/n) sum log L_i) EXACTLY; that's term (b).
- BUT the real model is AUTOREGRESSIVE: L_i = N_b(i ; x_{<i}) depends on the served prefix.
  So it's a TREE, not fixed L. #messages = #leaves = sum over paths of product of branching
  factors. (1/n) log #leaves -> entropy rate R; by AEP a typical message has
  (1/n) sum log N_b -> R, so #messages ~ exp(nR). Term (b)'s "E[log N_b]" is a PROCESS /
  branching ENTROPY RATE, not a fixed-position average.
- Types CAN be defined by the empirical distribution of log N_b along the path (method of
  types for sources with memory / Markov-tree types). Allowed.
- The five-term statement: (1/n) log #{messages of macro-type (p,q)} = R(p,q) + o(1);
  #{messages : R <= R*} ~ exp(nR*) (boundary type dominates).
- Subtlety in (b): the clean H(p)+(1-p)E[...] assumes RANDOM violation placement (typical
  message); the COUNT-maximizing adversary violates the LOWEST-N_b positions, so the
  capacity rate at fixed p is slightly higher (uses the largest-N_b positions for
  compliance / the elementary-symmetric / sorted form). Also the entropy rate = E[log N_b]
  under the MAXENT (uniform-over-messages) measure, whereas we estimated it via the
  honest-path average; these coincide when branching is fairly homogeneous / deviations small
  (our p~1e-3 regime). For a strict capacity bound, use the maxent / sorted form.

NEXT THEORY/ANALYSIS STEPS (all Opus-runnable):
- Possibly formalize R as the maxent/sorted (count-maximizing) rate vs the typical rate.
- The SEQUENTIAL DECISION RULE: turn THRESHOLD(N) into the actual cumulative random-walk /
  SPRT tear-up boundary (FPR 1e-10 + a target detection power). This was the agreed next
  build step on the protocol.
- Optionally: make the conservative "simple" R the headline instead of top-K (user was
  probing this; simple uses p*log2 V -- the free-adversary bound; both are computed).

============================================================
7. DEFERRED (needs Fable / bigger effort)
============================================================
- Prove the GENUINE codebook model end-to-end (CODEBOOK_PLAN.md). Make-or-break finding:
  the codebook MATMUL core is clean int, but genuine codebook uses per-token x
  per-output-channel NON-power-of-2 FLOAT dequant scales (per-token scale is data-dependent),
  which our uniform-pow2 zkob_rescale CANNOT express. Needs a NEW affine-dequant obligation +
  binding the dynamic per-token scale (itself a new covert channel to close). ~4-6 cycles,
  new crypto-adjacent code, needs Fable. Cheap alternative (round(fp8-dequant w * 2^16) on
  the existing fixed-point pipeline) is NOT the genuine codebook and won't match the measured
  codebook numbers. PARKED until Fable returns.
- F-2 hardening (run the -dlto kernel-miscompile probe + fold cross-check in production
  verify, fail-closed) and the graceful-reject verify wrapper -- small, were reverted with S1.
- statement.logit_binding currently binds GREEDY argmax; should be recast as verifiable
  sampled-argmax (served = argmax(logits + T*g_sigma)); the rowmax driver extends to it.
  Flagged in STAGE3_FAITHFUL_DESIGN.md §3.3.

============================================================
8. REPO / KEY FILES
============================================================
Repo: github.com/jacobcd52/zk-hillclimb (PRIVATE), local at
/workspace/projects/zk-hillclimb. Working tree clean, in sync with origin/main.
NEVER push or modify /workspace/projects/int-model-approximation (the integer pipeline;
READ ONLY; it produces weights/activations + the codebook/FP8 reference logits).

Docs: WRITEUP.md (summary), THREAT_MODEL_NOTES.md (regime corrected to sampled),
CAPACITY_CORRECTED.md, CAPACITY_TOPK.md, CAPACITY_FP16.md, CAPACITY_TEMPERATURE.md,
TOKEN_CAPACITY.md, BUFFER_FPR.md, THRESHOLD_CURVE.md, DIFR_BASELINE_NATIVE.md,
DIFR_FAITHFUL_V1.md, CODEBOOK_PLAN.md, BACKEND_DECISION.md (zkLLM/DeepProve/JOLT/zkGPT eval:
NOBODY else has real weight privacy), TRANSPORT_REBUILD_DESIGN.md, TRANSPORT_REVIEW.md,
REBUILD_AUDIT.md, STAGE_A/B/C1/C2/D_REPORT.md, ROPE_ATTENTION_DESIGN.md,
STAGE3_FAITHFUL_DESIGN.md, submissions/baseline-native/PHASE0_NOTES.md (the driver register,
sec 11-21), HANDOFF.md (older).
Plots (repo root): pareto_v1.png, capacity_combined.png, capacity_fp16_combined.png,
capacity_topk_ksweep.png, capacity_vs_T.png, buffer_vs_N.png, buffer_rt_dist_acf.png,
threshold_curve.png, difr_delta_curve.png.
Capacity scripts: capacity/ and buffer/ ; measure/ (DiFR + chain validation).
Agent prompts + results from this session: /root/zkllm/*_prompt.md, *_result.json.

============================================================
9. GOTCHAS (carried)
============================================================
- -dlto miscompiles NEW G1 kernel shapes -> use 1-thread host helpers (h_mul/h_add/g1_eq)
  for G1; Fr-only kernels are safe; always runtime-probe new kernels.
- upstream FrTensor::from_long_bin is broken -> use load_long_tensor.
- drivers do NOT mkdir obdirs (orchestrator must).
- all FrTensors hold PLAIN (non-Montgomery) values.
- the 3 protected headers (vrf_common.cuh, zkob_lookup.cuh) must stay byte-identical, else
  re-run EVERY driver selftest.
- agents working in /root/wpriv-dev or similar separate dev copies: their edits aren't in the
  repo until synced -- always verify the committed tree builds (the REBUILD_AUDIT caught this).
