# Task: independent soundness audit of the orchestrator VERIFIER (verify_walk.py)

Context: OUR OWN defensive verifiable-inference codebase. The orchestrator
(/workspace/projects/zk-hillclimb/orchestrator/) just passed its end-to-end selftest
(stage 1: 30 obligations on real llama-68m weights, ACCEPT; tampers rejected). The
VERIFIER side — verify_walk.py plus whatever it imports from common.py — is the system's
trust boundary: a flaw there voids every driver-level guarantee. You are the independent
reviewer required before acceptance. The driver binaries themselves are already audited;
your scope is the PYTHON layer.

## Read
1. orchestrator/ORCHESTRATOR_DESIGN.md — esp. §4 (chain-edge map), §5 (the independence
   argument), §6 (transcript semantics). The design is the claim; audit the CODE against it.
2. orchestrator/verify_walk.py, common.py (verifier-relevant parts), make_stage1_manifest.py,
   selftest.sh. Read register.py and prove_walk.py only to understand what the prover could
   manipulate — the adversary model is: PROVER-CONTROLLED proofs/ dir and data/ dir,
   honest registration/ + public.json (hash-pinned), and an honest verifier host.
3. PHASE0_NOTES.md §3 (transcript/harness-hack semantics), §14-15 pinned orchestrator
   obligations; harness/check_transcript.py (the downstream consumer).

## Hunt list (walk ALL)
1. **Path/argument injection from prover-controlled content**: does verify_walk derive any
   subprocess argument, file path, dimension, or constant from anything under proofs/ or
   data/? Every driver CLI arg must come from public.json/constants/the manifest. List
   every subprocess invocation and the provenance of each argument.
2. **Registration hash checking**: every file under registration/ that any driver verify
   reads — gens, q, registered coms, input, tables — re-hashed against public.json BEFORE
   use? Any file a driver reads that is NOT hash-pinned? run_seed re-derived from
   public.json bytes (not trusted from disk)?
3. **Chain-edge completeness and correctness**: compare the §4 edge map against the code —
   every edge actually checked? byte-equality done on full file contents? the skip (⊕)
   edges using zkob_skip verify correctly (right files, right order)? Could a prover pass
   with com files that differ where no edge looks? Cross-check against the manifest:
   any covered obligation whose input commitment is NOT transitively chained to
   registration or to a declared-open boundary?
4. **Verdict logic**: any path where a driver nonzero-exit, a missing file, a malformed
   transcript entry, or an exception leaves an id in `checked` or fails to flip the
   verdict? Fail-closed everywhere? `checked` strictly = genuinely verified ids (harness
   hack #1)? SKIPPED ids never counted as checked?
5. **Subprocess result interpretation**: is ACCEPT parsed robustly (exit code vs stdout
   string — what if a driver prints ACCEPT then crashes? what is the actual contract)?
   timeouts? retry logic that could mask a REJECT?
6. **TOCTOU / state leakage**: does verify_walk re-read anything prove_walk cached in a
   way a prover could swap between hash-check and use? Does it ever import or execute
   prover-written content (pickles, json with surprises, eval)?
7. **Selftest honesty**: do the three selftest phases actually exercise the claims
   (esp. that the tamper REJECTs fire in the verifier layer being audited, not as a
   side-effect of the prover failing)?

## Rules
READ-ONLY except the report: /workspace/projects/zk-hillclimb/orchestrator/VERIFIER_REVIEW.md
— VERDICT (SOUND / ISSUES-FOUND / BROKEN), CRITICAL/MAJOR/MINOR, per finding file:line +
what incorrect prover data gets wrongly accepted + fix. For clean categories, state what
you checked. You may run selftest.sh or verify_walk.py against the existing
/root/zkorch/selftest-* run dirs; experiments only under /tmp/vaudit-scratch/.
GPU shared; serialize, retry once on OOM.
