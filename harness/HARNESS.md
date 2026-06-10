# ZK hill-climb harness — frozen rules ("the constitution")

Version 1.0 — 2026-06-10. Owned by the COORDINATOR (main Claude session).
Implementation agents MUST NOT edit anything under `harness/` or `private/`.
The coordinator records sha256 hashes of every harness file and re-verifies
them before each scoring round; a mismatch voids the round.

## Objective

Improve the Pareto frontier of **(DiFR vs FP8 serving stack) x (ZK proof cost)**
for verifiable inference of llama-68m @ seq 1024 (later: bigger models), built
on zkLLM (sumcheck/lookup, BLS12-381) + the int-model-approximation
integerizations. A submission improves the frontier if it (a) lowers prove time
at equal-or-better DiFR, (b) lowers DiFR at equal-or-better prove time, or
(c) strengthens the security binding at bounded cost — WITHOUT failing any
soundness check below.

## What a submission is

A directory under `submissions/<name>/` containing:
1. Code (zkLLM fork / integerization changes / pipeline script). Builds happen
   on LOCAL disk (`/root/`), never executed from `/workspace` (FUSE).
2. `claim.json`: { what changed, expected effect, prove-time claim, DiFR claim,
   which obligations in the manifest its verifier covers }.
3. A **prover** entrypoint: produces a SERIALIZED proof artifact to disk.
4. A **verifier** entrypoint: reads ONLY {serialized proof, public inputs
   (prompt token ids, output logits/tokens), weight commitment} and returns
   ACCEPT/REJECT with a machine-readable transcript of which obligations it
   checked. The verifier must be a separate process from the prover.

NO SUBMISSION IS SCORED WITHOUT A WORKING SEPARATE VERIFIER. "It proves fast
but nothing checks it" is exactly hack #1 (hollowed-out proofs).

## Scoring (coordinator-run only)

Score = (prove_time_s, difr_mean_heldout, soundness_pass: bool). Soundness is
a GATE, not a tradeoff: any forgery acceptance or manifest gap = round score
INVALID, regardless of speed.

1. **Soundness gate**
   a. Verifier ACCEPTS the honest proof.
   b. Verifier REJECTS every forgery in `harness/forgeries/` (suite only ever
      GROWS; passing an old round does not exempt a submission from new
      forgeries).
   c. Obligation-manifest check: the verifier transcript covers EVERY
      obligation in `harness/manifest_llama68m.json`. Missing obligation =
      FAIL, even if all forgeries are rejected.
2. **Timing** (`harness/score.py --timing`): cold process, fixed shapes,
   exclusive GPU (no concurrent jobs; coordinator serializes), median of 3
   runs, one discarded warmup for JIT only when the submission declares it.
   Agents never run the official timer on their own results; they may
   self-time informally during development.
3. **DiFR** (`harness/score.py --difr`): teacher = FP8-dynamic stack
   (fp8_scaled_mm on this 4090; Hopper QGMMA logits from
   results/llama_pareto/h100_teacher_logits.pt where applicable). Prompts =
   HELD-OUT set drawn fresh each round by the coordinator from a corpus with a
   round-specific seed (stored in `private/round<N>_seed.txt` only after the
   round closes). Gumbel seeds fresh per round. Tuning against the public dev
   prompt is allowed; tuning against held-out prompts is impossible by
   construction since they don't exist until scoring time.

## The four hack vectors and their countermeasures

1. **Hollowed-out proofs** (prover skips work, verifier doesn't notice)
   -> forgery suite: coordinator injects corrupted proofs/outputs/weights; the
   verifier must reject ALL. See `harness/FORGERIES.md` for the catalog.
2. **Coverage shrinkage** (proof quietly stops covering a component)
   -> obligation manifest generated from the model architecture by
   `harness/manifest.py`; verifier transcript must match it exactly.
3. **Timing games** (warm caches, precomputation smuggled out of the timed
   region, background contention on baselines)
   -> frozen timing protocol above; one-time costs (ppgen, weight commit) are
   reported separately and may NOT be moved into "one-time" unless they are
   genuinely input-independent (coordinator audits the claim).
4. **DiFR overfitting** (tuning to the fixed prompt/seed)
   -> held-out prompts + fresh seeds each round, drawn at scoring time.

## Periodic transfer checks (anti-overfitting to llama-68m)

Every ~5 accepted rounds, or before declaring any "milestone": re-run the
current best scheme on TinyLlama-1.1B (locally, bigger batch) and — when
warranted — a brief H100 rental for Hopper-teacher checks. A scheme whose win
evaporates on transfer is flagged, not reverted automatically.

## Roles

- COORDINATOR (main session): owns harness/private, draws seeds, runs official
  scoring, reviews diffs of every submission before scoring (a submission that
  modifies scoring-adjacent code is rejected outright), maintains the ledger
  `results/LEDGER.md`.
- IMPLEMENTERS (Opus subagents): build submissions; GPU work is serialized via
  the coordinator.
- CREATIVE AGENT: periodically proposes non-obvious ideas (accuracy gains,
  overhead reductions, new forgery classes); proposals are triaged by the
  coordinator into implementation tasks.

## Honesty rules

All reported numbers carry provenance (measured / composed-estimate / assumed).
A submission's claim.json may be optimistic; the LEDGER records only
coordinator-measured values. Negative results are recorded too.

## Threat-model note (do not optimize it away)

The DiFR tolerance floor IS the covert-channel capacity. Any change that
lowers prove cost but widens the honest-student error floor is trading
security for speed; the ledger must say so explicitly (record bits/token at
p99 via results/llama_pareto/acceptance_bits.py methodology).
