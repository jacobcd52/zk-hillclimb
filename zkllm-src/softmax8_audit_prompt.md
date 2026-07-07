# Task: independent soundness audit of zkob_softmax8.cu + the zkob_headmerge perm-flag diff

OUR OWN defensive codebase; second-engineer review before trusted-base entry. Both
selftests ALL PASS (coordinator re-ran: 167 / 166 checks). Bar/format: the prior
*_REVIEW.md registers (read SOFTMAX_REVIEW.md and ROPE_REVIEW.md minimum).

Read: STAGE3_FAITHFUL_DESIGN.md §4.2 + §4.3 (normative), §1.2-1.3 (the quirks being
fixed); SOFTMAX_DESIGN.md (the temp-128 ancestor — softmax8 inherits its §4 obligations
with §4.3's redone arithmetic); PHASE0_NOTES.md; shared headers (trusted; audit usage);
under review: /root/zkllm/zkob_softmax8.cu and /root/zkllm/zkob_headmerge.cu (focus the
headmerge audit on the DIFF vs the previously-audited version — git history in
/workspace/projects/zk-hillclimb/zkllm-src/ has the pre-flag version at commit f792978;
use `git -C ... show f792978:zkllm-src/zkob_headmerge.cu` to diff).

## softmax8 checklist
1. FS ordering vs §4.3's schedule (incl. com_mx absorb position and the mx chain).
2. Verifier independence table (every disk read anchored).
3. The §4.3 arithmetic: E8 table domain/sentinel handling for masked positions (could a
   prover exploit the sentinel row? is the sentinel value excluded from honest range?);
   the rounding bracket with S ≤ 2^26 and 2×14-bit limbs (redo the §4.3 bounds yourself);
   Dm = z − mx_bcast ≤ 0 binding (how is the table indexed and is the index forced
   in-domain?); the mx broadcast binding.
4. Inheritance check: every load-bearing verifier requirement of zkob_softmax (U_f2
   rules, constant-claim discipline, broadcast row-bit openings, round/row-count guards)
   present in softmax8 — list any that differ and why that's per-design.
5. Selftest honesty incl. the agent-reported guard-test fix (z-envelope vs mx guard
   ordering — is the fixed test actually testing the z guard?).
6. New kernels: should be ZERO.

## headmerge-diff checklist
7. The diff is minimal per §4.2 (gather formula + PERM absorb + CLI + dims + selftest)?
   Anything else changed? (Any change outside that scope = MAJOR finding.)
8. The PERM absorb position and dims.bin cross-check make cross-mode splices diverge —
   verify both directions; could a prover omit/forge the absorb?
9. The concat-mode gather formula correctness (it must be the §1.3 M, i.e. plain
   head-concat: O[t, 64h+d] = out_h[t,d]) — and pi157 mode unchanged vs the audited
   original (byte-compare the formula paths).

READ-ONLY except /root/zkllm/SOFTMAX8_REVIEW.md (VERDICT per file + overall;
CRITICAL/MAJOR/MINOR; file:line; what wrong prover data passes; fix; clean categories
documented). May run both selftests; experiments under /tmp/s8-audit/ only. GPU may be
shared with another audit.
