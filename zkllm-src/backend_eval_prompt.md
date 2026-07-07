# Task: evaluate prior zkML codebases and decide the best path to an efficient, weight-private ZKP for a fully integerized model

GOAL (the project's new phase): produce an efficient zero-knowledge proof of a FULL
integerized-LLM forward pass with (a) FULL WEIGHT PRIVACY (prove against committed
weights, never reveal them), (b) a fast implemented VERIFIER, (c) the ability to AUGMENT
the proof with our custom obligations that close prover freedom (zero-advice bindings —
see /workspace/projects/zk-hillclimb/PHASE0_NOTES sections 14-21 for what these are:
exact rmsnorm bracket, zero-advice softmax, exact rowmax, chained commitments between
operations). Our current zkLLM-derived stack works and is sound but slow (llama-68m:
17.7 min prove / 16.7 min verify on an RTX 4090, ~1e5x inference). Decide: ADOPT a
prior codebase / FORK+AUGMENT one / HYBRID (their backend under our obligations) /
KEEP OPTIMIZING OURS. Evidence over vibes: clone, build, run, measure where possible.

## Candidates (start here; add others you find)
1. **DeepProve** (github.com/Lagrange-Labs/deep-prove) — PRIMARY candidate. Sumcheck +
   logUp-GKR + Basefold hash commitments (same protocol family as ours, modern backend).
   Claims: full GPT-2 proof, ~0.5s verify, 54-158x faster than EZKL. CHECK SPECIFICALLY:
   (i) Is weight privacy real (commit-and-prove against a weight commitment, ZK hiding)
   or marketing (just "weights not in the proof")? Read the code, not the README.
   (ii) Is the commitment deterministic (needed for our byte-equality chaining)?
   (iii) What field? CPU or GPU prover? Build it and prove something small (their
   examples; measure on our 4090 box — note if CPU-only, measure anyway).
   (iv) EXTENSIBILITY: can custom constraints/obligations be injected per-operation
   (our brackets/rowmax/zero-advice bindings), or is it a closed ONNX->proof pipeline?
   How are nonlinearities handled (lookups? which quantization? does it leave prover
   advice freedom — the EXACT issue our project exists to close)?
   (v) License, maintenance, code quality.
2. **JOLT Atlas** (search github; arxiv 2602.17452) — lookup-centric, nanoGPT ~14s claim.
   Same checks, lighter depth.
3. **zkGPT** (GPT-2 under 25s claim) — is there code at all?
4. **Expander / Polyhedra zkPyTorch** (github.com/PolyhedraZK) — GKR family, GPU story.
5. Anything else with: implemented verifier + weight privacy + transformer-scale numbers.

## Method
- WebSearch/WebFetch for docs/papers; CLONE the repos to /root/backend-eval/ (NOT on
  /workspace — FUSE; binaries/builds need local disk); build (Rust toolchain may need
  installing — rustup to local disk; uv/cargo caches fine); run smallest end-to-end
  example; record real numbers from OUR hardware where feasible (time-box: if a build
  fights you >45 min, record the blocker and move on).
- For the leading candidate, answer the five checks with file/line evidence.

## Deliverable
/workspace/projects/zk-hillclimb/BACKEND_DECISION.md:
1. Per candidate: the five checks, measured numbers (ours vs claimed), evidence.
2. THE DECISION (adopt/fork/hybrid/keep-ours) with reasoning, and what it costs us:
   which of our 11 obligations port trivially / need redesign / become unnecessary;
   what happens to byte-equality chaining and the homomorphic links; what the
   capacity-measurement layer needs from the backend.
3. A concrete next-3-steps plan for the chosen path.
Honest about everything unverifiable in the time available. No git commits; no pushes;
never touch int-model-approximation.
