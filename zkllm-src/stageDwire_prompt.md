# Task: wire weight privacy into the full walk + measure walk-scale weight-private timings

Stage D protocol is DONE and committed: hiding weight registration (D1), hidden weight
claims (D2), ZK blinded IPA opening (D3), leakage regression CLEAN (D4) — all validated at
single-driver/single-obligation scale (wbatch 26/26, fc+rmsnorm wpriv ALL PASS). The CLIs
exist (--hiding registration, --wpriv driver mode, wprove/wverify on the batch). What's
left is PLUMBING per STAGE_D_REPORT.md "What's left": wire it through the orchestrator and
get the real walk-scale numbers.

Read: STAGE_D_REPORT.md (the as-built interface: register 15 weight tensors --hiding;
prove_walk passes --wpriv through the DriverPool so fc/rmsnorm emit hidden weight claims;
verify adds wprove/wverify of the weight sub-batch to the gate; the blinds live in
*.blinds.bin / cblinds.bin, prover-private), TRANSPORT_REBUILD_DESIGN §4, the orchestrator
(common.py, register.py, prove_walk.py, verify_walk.py, selftest.sh).

## Work
1. register.py: register the 15 weight tensors (q/k/v/o_proj/gate/up/down per layer +
   lm_head + the 5 rmsnorm gains — confirm the exact set) with HIDING commitments; store
   blinds privately; hash-pin the hiding commitments in public.json; register the
   independent generator H.
2. prove_walk.py / common.py: a --wpriv run mode routing fc + rmsnorm weight claims into
   the HIDDEN weight sub-batch (activations stay the fast public batch); run wprove over it.
3. verify_walk.py: wverify the weight sub-batch in the single-process verifier; gate
   ACCEPT requires it. Keep the public activation batch path bit-identical.
4. selftest.sh: a --wpriv full-walk phase. Gates:
   - honest faithful-arch-v1 walk in weight-private mode ACCEPTs (65 ids + opening_batch +
     weight sub-batch); every existing forgery still REJECTs at its named locus; a
     weight-claim tamper / a wrong-blind REJECTs.
   - D4 at walk scale: scan ALL proof artifacts of the full weight-private run — NO
     weight-MLE evaluation in plaintext (the 15 tensors). Report CLEAN or the leak.
   - timings: full weight-private walk prove/verify/proof vs the non-private C2 numbers
     (prove 636s, verify 27.1s, 176MB). Confirm the ~+1-2% prove / small verify envelope.

## Deliverable
Append a "Walk-scale" section to STAGE_D_REPORT.md: the gate results, the walk-scale D4
scan, the weight-private before/after timing table, deviations, what (if anything) still
blocks a fully weight-private end-to-end proof. RUN selftest.sh --wpriv yourself, paste the
verdict + numbers. Copy changed orchestrator/*.py to /workspace/projects/zk-hillclimb/
orchestrator/ when passing. Don't edit the 3 protected headers. NO git commits. No GitHub;
int-model-approximation untouched. GPU free. Full run ~11 min prove; be patient; if turn
limit looms leave the run going (nohup) + write what's measured.
