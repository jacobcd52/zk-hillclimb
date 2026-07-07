# Task: temperature sensitivity check + reconcile the decoding-regime framing (greedy -> verifiable sampled) across docs

We found a real inconsistency: the COVERT-CAPACITY analysis correctly uses post-Gumbel
scores (logits + Gumbel noise, shared seed) per the DiFR paper — VERIFIED in
capacity/capacity_dump_corrected.py:103-139 (z_ref+g, z_fp8+g, margins & N_b all post-Gumbel,
no bug). BUT THREAT_MODEL_NOTES.md and the ZKP's statement.logit_binding (STAGE3_FAITHFUL_DESIGN
§3.3) pin GREEDY decoding (served = argmax(logits)). Greedy (T->0) has NO sampling channel —
the sampled/post-Gumbel regime is the correct one and what the whole capacity analysis assumes.
Two jobs:

## Job 1 — temperature: verify + sensitivity sweep
The capacity dumps use "metric temperature 1.0" (post-Gumbel score = logits + g, T=1). The
DiFR margin is logits + T*g. Confirm T=1 is used CONSISTENTLY (the Gumbel scale is the same in
the served-token sampling AND the margin/N_b). Then run a SENSITIVITY SWEEP: recompute the
worst-case min-over-(b,K) capacity for the FAITHFUL scheme (corrected orientation, reference =
M_int post-Gumbel, served = FP8 argmax(logits + T*g)) at several sampling temperatures
T in {0.3, 0.5, 0.7, 1.0, 1.3, 2.0} (reuse the existing logit dumps z_int / z_fp8; just rescale
the Gumbel term by T, i.e. post-Gumbel score = logits + T*g, and re-derive served token, margins,
N_b, and the capacity). Report capacity(T) — expect it to RISE with T (more spread -> more
near-ties / violations) and -> ~0 as T->0 (greedy, deterministic). This quantifies how
temperature-dependent the headline numbers are. Plot capacity-vs-T. Keep the seed/prompts fixed
(seed 20260611, 8 dolly prompts) so it's comparable to the existing numbers.

## Job 2 — reconcile the docs (precise framing, NOT word-swapping)
The CORRECT framing to write everywhere the docs currently say "greedy":
- The protocol operates in the VERIFIABLE SAMPLED-DECODING regime (shared-seed Gumbel-max,
  the DiFR setting): the datacenter samples each token as argmax_v(logits[v] + T*g_sigma[v])
  with g_sigma a public function of a COMMITTED seed sigma. The committed seed is what closes
  the sampling-randomness channel (otherwise it dominates everything).
- The ZKP proves the model's LOGITS; the verifier reconstructs post-Gumbel scores
  (proven logits + committed-seed Gumbel) and measures each served token's margin from the
  post-Gumbel argmax. Covert capacity = bits hidden by deviating within the tolerated margin.
- Greedy (T->0) is the DEGENERATE special case (no sampling channel); the earlier "greedy
  pinned" note was an oversimplification and is corrected.
- Sampling temperature T is a parameter; capacity scales with T (cite the Job-1 sweep). The
  reported headline numbers are at T=1.
- The ZKP's statement.logit_binding currently binds GREEDY argmax (served = argmax(logits)).
  Flag this HONESTLY as a now-inconsistent design point: for the treaty protocol the served
  token is OBSERVED at the network tap and checked against the proven logits externally, so the
  in-proof greedy binding is either superfluous OR should be recast as a verifiable-sampled
  binding served = argmax(logits + T*g_sigma) — which the existing rowmax argmax-binding driver
  extends to almost directly (feed it logits+T*g_sigma instead of logits; no new driver). Do
  NOT silently "fix" the ZKP design — DOCUMENT it as an identified follow-up.

Update: THREAT_MODEL_NOTES.md (the core fix — rewrite the greedy assumption + the capacity-table
note + the sampling-randomness row), WRITEUP.md (its threat-model/framing section + add the T
note + the capacity(T) plot), and add a short "decoding regime + temperature" clarification note
to CAPACITY_CORRECTED.md and BUFFER_FPR.md (the numbers are at T=1, sampled regime). Leave
STAGE3_FAITHFUL_DESIGN.md's text but add ONE note at §3.3 flagging the logit_binding regime
follow-up. Keep edits accurate and tight; preserve all existing correct numbers.

Deliverable: the doc edits + a new capacity/CAPACITY_TEMPERATURE.md with the T-sweep table +
plot (capacity_vs_T.png), and a one-paragraph summary of the regime reconciliation at the top of
THREAT_MODEL_NOTES.md. /root/int-model-env/bin/python for the sweep. No git commits; no pushes;
int-model-approximation READ ONLY. Be honest and precise; this is correcting the record.
