# Task: update selftest byte-tamper EXPECTATIONS for the new canonical-commitment graceful-reject (selftest-only)

S1 (canonical-affine G1 commitment encoding) + the verify-wrap are SOUND — all forgeries
reject, honest accepts. But 2+ drivers' selftests FAIL on a test-EXPECTATION mismatch, NOT a
soundness bug: a byte-tamper of a com_*.bin now makes the canonical G1 decode FAIL, so verify
rejects EARLY as `REJECT: malformed proof artifact: zkg1: invalid canonical G1 encoding ...`
(graceful, fail-closed, rc=1) instead of the OLD downstream locus the selftest asserts (e.g.
"IPA opening of eQ01 vs com_Q", "transcript divergence"). The forgery IS caught; only the
expected reject-STRING differs. Confirmed failing: zkob_fc, zkob_headslice (their com-tamper
cases). Other drivers may have the same latent issue depending on which file/offset their
tamper hits.

## Do (SELFTEST CODE ONLY — do NOT touch verify(), prove(), the protocol, or the 3 protected
## headers vrf_common.cuh/zkob_lookup.cuh):
1. In EVERY driver's selftest byte-tamper loop, when the tampered file is a COMMITMENT
   (com_*.bin) and the rejection reason is "malformed proof artifact" / the canonical-decode
   throw, ACCEPT that as a valid rejection (it is a fail-closed REJECT of a corrupted
   commitment — the canonical encoding catches the corruption at decode, earlier than the old
   downstream check). Keep the strict expected-locus check for NON-commitment tampers (proof
   files, ipa, claims) and for SEMANTIC evil modes — those must still reject at their named
   logic locus. The ONLY relaxation: a com_*.bin byte-tamper may reject either at its old
   downstream locus OR as "malformed proof artifact". Do this uniformly across all 12 drivers
   (fc, rescale, skip, glu, rmsnorm, softmax, softmax8, rope, headslice, headmerge, rowmax,
   batchopen) so it's consistent, not just the 2 that happened to trip.
2. Rebuild all affected drivers (pinned -dlto: compile zkob_X.cu -> .o, link with
   "bls12-381.o ioutils.o commitment.o fr-tensor.o g1-tensor.o proof.o zkrelu.o zkfc.o
   tlookup.o polynomial.o zksoftmax.o rescaling.o timer.o"). Re-run ALL 12 selftests ->
   every one ALL PASS. Honest ACCEPT, every semantic forgery rejects at its NAMED locus,
   every byte-tamper rejects (downstream-locus or malformed-artifact, both valid).

## Deliverable
Append to /root/zkllm/S1_F2_REPORT.md a short "test-expectation fix" note (what changed, why
it's sound: tampers still reject fail-closed, only the com-tamper reject-string broadened).
Print the final 12/12 selftest verdict. Copy changed .cu to /workspace/projects/zk-hillclimb/
zkllm-src/ when ALL 12 pass. NO git commits. No GitHub; int-model-approximation untouched.
GPU free. Be thorough — I (coordinator) will re-run all 12 selftests + the full batched walk
myself before committing, so the fix must genuinely hold.
