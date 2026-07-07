# Task: Stage B of the transport rebuild — kill the per-tensor fold latency, convert zkob_rescale, two-driver chain + forgery battery

Stage A is DONE (vrf_toy_batchopen + zkob_batchopen + zkob_fc claim-mode, all passing,
committed; see /root/zkllm/STAGE_A_REPORT.md and zkllm-src/). It confirmed inline IPAs are
99.3% of per-driver verify, single-driver 2.24x. BUT Stage A flagged two costs that now
dominate the batched verify and BLOCK the 10-60s target:
  (1) per-tensor fold_chain is launch-latency bound (~34-51 ms × ~1242 tensors ≈ 60 s);
  (2) me_weights host loop at G=32768 ≈ 15 s (the §6.2 GPU-lift item).
Stage B (TRANSPORT_REBUILD_DESIGN.md §6 Stage B) addresses these and extends to a chain.

Read first: TRANSPORT_REBUILD_DESIGN.md (esp. §2.2 batch reduction, §2.7 verify budget,
§6 Stage B), TRANSPORT_REVIEW.md (the pins F3-F6 + battery BO-1..BO-12), STAGE_A_REPORT.md
(what exists + the two flagged costs + the claims_match/canonicalization + one-gen-file-
per-domain notes), and vrf_common.cuh + zkob_lookup.cuh + zkob_batchopen.cu + zkob_fc.cu.

## Work items
1. **Batched fold kernel.** Replace the per-tensor fold_chain launches in zkob_batchopen
   verify (and prove where applicable) with a FLATTENED batched kernel that folds all
   tensors of a domain in one (or few) launches — eliminate the ~60 s launch-latency.
   PROBE the new kernel shape at runtime (-dlto miscompile rule); cross-check the batched
   fold against the per-tensor result element-exact (a convention selftest, both must
   match before trusting). This is Fr-only (safe kernel family) — confirm no G1 kernel.
2. **me_weights GPU lift for G=32768.** Use the fast-helper pattern already in zkob_rowmax
   (k_pp_expand / fast_me_weights / fast_s_vector — §2.8 of STAGE3_FAITHFUL_DESIGN) inside
   zkob_batchopen's IPA so the gen-32768 opening isn't host-loop bound. Cross-check
   fast-vs-slow element-exact. (If this requires a header edit, FLAG IT LOUDLY — it
   triggers the all-driver re-validation rule; prefer keeping it driver-local to batchopen.)
3. **Convert zkob_rescale to claim mode** (behind the same --claims flag as fc; old inline
   tail stays default/compilable). Update its selftest to run both modes (old honest+
   forgeries pass; claim-mode honest ACCEPT + same forgeries reject via the batch).
4. **Two-driver accumulator through the orchestrator** on the validated fc→rescale pair
   INCLUDING the lm_head gen32768 instances: the global accumulator collects both drivers'
   claims, one zkob_batchopen discharges them, the verifier checks the batch + the chain
   edge between them. Canonicalize comref paths (the Stage-A flag) and assert one-gen-file-
   per-domain-size.
5. **BO forgery battery at pair scale** — BO-1..BO-12 from TRANSPORT_REVIEW §5.2 incl. the
   F3 trailing-rows test and F8 locus correction: each must reject by exactly the named
   check. Add: a cross-driver claim-drop (omit rescale's claim, fc's still there) rejects;
   a chain-edge tamper rejects.
6. **Measure** the verify split after (1)+(2): the fold pass and the gen-32768 IPA times
   specifically, vs Stage A's numbers — does the §2.7 budget (10-60 s) now look reachable
   on the fc+rescale+lm_head subset? Report the prover batch-sumcheck overhead too (the
   §2.2 unmeasured flag).

## Gates
- T3: two-driver chain through the orchestrator ACCEPTs honestly; BO battery + chain
  forgeries all reject on the named checks; batched-fold and fast-me_weights convention
  cross-checks element-exact; fc and rescale selftests pass BOTH modes.
- A go/no-go on §2.5's PCS contingency: with (1)+(2) done, is the single batched IPA the
  bottleneck or not? State it with numbers.

## Deliverable
/root/zkllm/STAGE_B_REPORT.md: what was built, T3 results, the verify split before/after
the two kernel fixes (numbers), the PCS go/no-go, any header edits (flag loudly), deviations,
concerns. Copy changed .cu to /workspace/projects/zk-hillclimb/zkllm-src/ when passing.
NO git commits. Build with pinned -dlto commands. No GitHub; int-model-approximation
untouched. GPU is free (a light python capacity job runs concurrently, no GPU). Iterate
until T3 passes; STOP and report any genuine blocker.
