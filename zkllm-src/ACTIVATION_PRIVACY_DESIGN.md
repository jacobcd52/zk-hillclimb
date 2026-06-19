# Activation privacy (D5) — design, grounded in zkob_fc.cu / zkob_wpriv.cuh

Goal: hide the layer's INPUT activation X and OUTPUT activation Y (currently public), reusing the
Stage-D weight-privacy machinery. Closes STAGE_D_REPORT §5.3 (deterministic activation commitments →
guess-and-confirm). Single layer first (`zkob_fc`); compose via boundary chaining for layer-sampling.

## What leaks today (zkob_fc.cu)
1. `com_X` (L195) and `com_Y` (L206) are PLAIN Pedersen (no H-blind). com_W gets hidden via
   `wp_hide_rows` (L197–204) in `--wpriv`; X/Y never do. → commitments are guess-and-confirmable.
2. The activation claims are emitted as PLAINTEXT evals into the PUBLIC accumulator:
   `mk("X",…,claim_X)` (L343) and `mk("Y",…,claim)` (L345), tag `BO_EVAL_PLAIN`. The public batch
   then opens them with the fast NON-hiding IPA. → claim_X = X̃(u), claim_Y = Ỹ(u) revealed in clear.
3. (Weight side already fixed: W routed `BO_EVAL_COMMITTED` → waccdir, ZK batch. L311–330.)

Note the X·W product sumcheck round messages (L244–282) are ALREADY committed in `--wpriv`
(`ped_qh(p0,τ0,Q,H)`, L264) because they touch W — so the round-message channel is already ZK; the
remaining activation leaks are the COMMITMENTS (1) and the CLAIM EVALS (2).

## The clean mirror (mostly reuse)
For X and Y, replicate the W treatment:
- **D1 hide**: generate row blinds `s_X` (len B_pad), `s_Y` (len B_pad); `wp_hide_rows(com_X,s_X,H)`,
  `wp_hide_rows(com_Y,s_Y,H)`; save prover-private (`wp_blinds_save`). com_X/com_Y become hiding.
- **D2 route committed**: emit X- and Y-claims with `tag=BO_EVAL_COMMITTED`, `ceval=C_X=claim_X·Q+t_X·H`
  (resp. C_Y) into an ACTIVATION accumulator `aacc` (new, mirrors waccdir), discharged by the
  existing `wbatch_prove/verify` (committed-round sumcheck + ZK-IPA). `wp_cblind_emit` the (v,t).
- Reuse verbatim: `wp_rand`, `wp_hide_rows`, `ped_qh`, `lagrange3_g1`, `zk_ipa_prove/verify`,
  `wbatch_prove/verify`, `schnorr_h_*`, `schnorr2_*`, the cblind/blindref bookkeeping.

## The ONE new piece of crypto — the FC terminal product
fc proves `claim = claim_X · claim_W` (L296). Stage D Schnorr (L303–309) covers `E = C_cur −
claim_X·C_W = δ·H` — works ONLY because `claim_X` is a PUBLIC scalar (one hidden factor → linear).
Hiding X makes claim_X committed (C_X), so the terminal is a **product of two committed secrets**
`claim_Y ?= claim_X · claim_W` with all three in commitments (C_Y, C_X, C_W). Over our DLOG group
(no pairing) this needs a **multiplication/product sigma-proof**:
  given C_x=x·Q+t_x·H, C_w=w·Q+t_w·H, C_y=y·Q+t_y·H, prove y = x·w in ZK.
Standard gadget (Bulletproofs-style multiplication / a 3-move sigma): commit auxiliary, challenge e,
respond; verifier checks two linear G1 equations. ~constant size (a few G1 + a few Fr), no new
assumption beyond DLOG. THIS is the core implementation work; everything else is reuse.

## Boundary chaining (for layer-sampling soundness)
com_Y[l] must equal com_X[l+1] as the SAME committed object so a bad transition is catchable. Two
options: (a) reuse the same row-blinds at the boundary so the two hiding commitments are byte-identical
(simplest; the orchestrator already comref-matches com files), or (b) a cheap commitment-equality
proof if blinds must differ. Mirrors the EZKL phase-3 boundary-binding we already validated.

## Phasing (each gated on `./zkob_fc selftest`, then full 13-TU + walk)
- **P1a**: hide com_X/com_Y + route the Y-claim committed (X still public). Validates the hiding +
  aacc discharge pipeline with the EXISTING linear terminal (claim_X public). Low risk, pure mirror.
- **P1b**: hide X too → add the product sigma-proof for the terminal. The real new crypto.
- **P1c**: boundary chaining (shared-blind or equality proof) + a 2-layer compose selftest.
- **Validation**: extend the D4 leak scan to scan for claim_X AND claim_Y (must be CLEAN); add
  honest-ACCEPT, forgery-locus, and ZK re-randomization selftest cases mirroring the wpriv ones;
  then the full `orchestrator/selftest.sh` walk (needs the python env + llama-68m data reconstructed).

## Discipline
zkob_fc.cu and zkob_wpriv.cuh are COORDINATOR-BUILT/protected. We (coordinator) may edit, but every
edit re-runs the selftest battery; no new G1 kernel (ride k_g1_scale/k_g1_add_pairs per §9). Keep
plain + claim-mode transcripts byte-identical; activation-privacy is a new flag-selected mode
(`--apriv`, mirroring `--wpriv`). Privacy remains real ZK on the Pedersen path — remind before any
external ZK claim (memory zk-ezkl-privacy-caveat).
