# Task: Stage C-part-1 — convert the remaining 9 drivers to claim mode (the mechanical bulk of Stage C)

Stages A+B are DONE: the batched-opening protocol (zkob_batchopen), the flattened fold
kernel + me_weights GPU lift (in zkob_claims.cuh), and claim mode on zkob_fc + zkob_rescale
are all validated (T1/T2/T3 PASS; see /root/zkllm/STAGE_A_REPORT.md, STAGE_B_REPORT.md).
The verify projection is now ~2-3 s full-walk. Stage C finishes the job. THIS task is the
mechanical bulk: convert the REMAINING 9 drivers' opening tails to claim mode, exactly the
way zkob_fc and zkob_rescale were converted.

Read: STAGE_B_REPORT.md (the conversion pattern: emit terminal claims at the exact old
open_prove sites with witrefs+drvstate, verify recomputes claims into the accumulator,
old inline tail kept behind absence of --claims and DEFAULT; ZKOB_FOLD_CROSSCHECK on in
selftests), zkob_fc.cu + zkob_rescale.cu (the two done examples), zkob_claims.cuh (the
shared claim/accumulator + batched fold + fast IPA), TRANSPORT_REBUILD_DESIGN §3 (per-driver
change list) + TRANSPORT_REVIEW.md F3-F6 pins.

## Drivers to convert (each: claim mode behind --claims, old tail default+compilable,
## dual-mode selftest ALL PASS — old honest+forgeries unchanged, claim-mode honest ACCEPT +
## same forgeries reject via the batch):
1. zkob_skip      (pure homomorphic adds — point checks; may emit ZERO opening claims, just
                   confirm it routes through the batch correctly / has nothing to open)
2. zkob_glu       (mapping lookup + hadamard — ~the rescale+fc pattern)
3. zkob_rmsnorm   (17 openings — the trio; the registered com_g claim must carry the
                   REGISTERED file path as comref per F5/F6)
4. zkob_softmax   (16 openings)
5. zkob_softmax8  (similar to softmax)
6. zkob_rope      (3 openings)
7. zkob_headslice (the paired-opening driver — eval_full==eval_slice checked driver-side
                   BEFORE the batch, per TRANSPORT_REVIEW §4; both become batch claims)
8. zkob_headmerge (13 openings)
9. zkob_rowmax    (12-14 openings, both modes; already has the fast-IPA kernels — reuse)

## Hard rules
- DO NOT edit the 3 protected headers (vrf_common.cuh, zkob_lookup.cuh) — diff-verify
  untouched (sha256 before/after, print them). zkob_claims.cuh edits allowed IF needed,
  but if you touch it, REBUILD + re-run ALL its includers' selftests (the Stage-B rule).
- Each driver: the F3 pin (commitment-size check before fold — already in batchopen verify;
  ensure each driver's claims carry correct n_rows so batchopen enforces it), F4-F6
  (claims absorbed before challenge, verifier recomputes claim list, registered comref for
  weight claims). Keep every sumcheck/lookup/commitment/link byte-identical — ONLY the
  opening tail changes.
- Build with pinned -dlto commands; probe any new kernel shape; no new G1 kernels.
- Work one driver at a time; a driver isn't done until its dual-mode selftest ALL PASS.
- No GitHub; no git commits; int-model-approximation untouched. GPU free.

## Deliverable
/root/zkllm/STAGE_C1_REPORT.md: per driver — converted Y/N, dual-mode selftest result
(claim counts, forgeries still caught), any per-driver subtlety (skip's zero-claims,
rmsnorm's registered-comref, headslice's pre-batch equality), header-untouched sha256
proof, deviations. Copy converted .cu to /workspace/projects/zk-hillclimb/zkllm-src/ as
each passes. If a driver resists conversion, document precisely and move on (don't block
the others). Honest reporting — a driver whose claim-mode selftest fails is reported FAIL.
