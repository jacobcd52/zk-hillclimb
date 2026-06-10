# Forgery catalog — what every submission's verifier must REJECT

Rule: the suite only GROWS. Each forgery is implemented in
`harness/forgeries/` as a mutation of an honest (proof, public inputs,
commitment) triple, with expected verdict REJECT. IDs are stable.
Status: PLANNED until wired against the serialized-proof format (phase 0),
then ACTIVE.

## A. Output / statement tampering
- A1 flip-one-token: same proof, one output token id changed.
- A2 logit-nudge: per-position logits perturbed by +0.1 at a non-argmax slot
  (must fail because the proof binds logits, not just argmax).
- A3 prompt-swap: proof generated for prompt P presented with prompt P'.
- A4 truncation: last k positions of the output dropped/duplicated.

## B. Weight / commitment tampering
- B1 wrong-weights: honest proof, commitment replaced by commitment to
  perturbed weights (one tensor, one element += 1 ulp at FP8 grid).
- B2 commitment-reuse: commitment from a DIFFERENT model (llama-68m vs a
  re-initialized clone) with otherwise honest proof.
- B3 scale-tamper: per-row FP8 weight scales modified, FP8 codes unchanged.
- B4 self-chosen-commitment: prover commits to weights it actually used, but
  they differ from the registered public weight hash (binding to the
  REGISTERED hash is the point; verifier must check against it, not against
  whatever the proof ships).

## C. Proof-object tampering (hollowed-out proofs)
- C1 zeroed-polys: random subset of sumcheck-round polynomials zeroed.
- C2 random-bytes: same length, random field elements.
- C3 dropped-round: one sumcheck round removed (length-checks alone must not
  pass; the recursion must actually be checked).
- C4 dropped-component: proof section for one obligation (e.g. layer-1
  down_proj rescaling lookup) removed; tests manifest enforcement.
- C5 spliced-proof: component proofs from two DIFFERENT honest runs combined
  (tests Fiat-Shamir/transcript binding across components).
- C6 stale-proof replay: yesterday's honest proof for today's claimed run with
  different randomness/seed context.

## D. Semantic / circuit-level forgeries (the subtle ones)
- D1 cheaper-architecture: honest-looking proof of a SMALLER model (fewer
  layers / narrower FFN) presented as the full model.
- D2 out-of-range-remainder: rescaling remainder outside [-2^15, 2^15) with
  compensating quotient (tests the range lookup is actually binding).
- D3 non-codebook-value: an activation/weight value off the FP8 codebook grid
  (tests codebook membership lookup).
- D4 wrong-rounding: floor instead of round-half-away in one rescale (small
  systematic drift an inattentive verifier accepts).
- D5 lookup-multiplicity: tampered multiplicity vector m in a tLookup proof.
- D6 padded-garbage: padding regions (e.g. seq 685 -> 1024) carry values that
  influence committed outputs (tests padding is constrained).

## E. Process / timing forgeries (scored by protocol, not verifier)
- E1 warm-cache timing: submission timed with hot JIT/weights when claim says
  cold. Countermeasure: coordinator-run cold protocol only.
- E2 one-time smuggling: per-input work relabeled "one-time setup".
  Countermeasure: coordinator audits input-independence (re-run with a second
  prompt; "one-time" artifacts must be byte-identical).
- E3 concurrent-contention: baseline timed under load, submission timed quiet.
  Countermeasure: exclusive-GPU serialization for all official timing.

## F. Evaluation-gaming (scored by protocol)
- F1 difr-seed-mining: tuning to the public Gumbel seed. Countermeasure:
  fresh seeds at scoring.
- F2 prompt-overfit: tuning to the public dev prompt. Countermeasure:
  held-out prompts drawn at scoring time.
- F3 vocab-clip: student that clips/quantizes logits toward the teacher's
  argmax only (improves DiFR metric while corrupting the distribution).
  Countermeasure: logit_l2 + full-logit binding in the proof statement (A2).

Creative-agent standing brief: propose new forgery classes; anything that
fools the current verifier gets added here and the verifier must be fixed
before any further submissions are scored.
