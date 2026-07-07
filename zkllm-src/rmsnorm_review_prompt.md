# Task: independent soundness audit of `/root/zkllm/zkob_rmsnorm.cu`

Context: this is OUR OWN research codebase (a defensive verifiable-inference project — we are
building the verifier and need it to be sound). A first engineer wrote
`/root/zkllm/zkob_rmsnorm.cu` (917 lines) and its selftest reports ALL PASS. You are the
second engineer doing the standard independent code review that our process requires before
the file is accepted into the trusted base. Your job is to audit the implementation for
soundness gaps — places where the verifier fails to enforce the relation the spec claims it
enforces — that the selftest would not catch. Do not take the selftest output on trust;
check the logic itself.

## Context to read first
1. `/workspace/projects/zk-hillclimb/HANDOFF.md` — section "IMMEDIATE NEXT STEP" is the spec
   the driver must implement EXACTLY.
2. `/workspace/projects/zk-hillclimb/submissions/baseline-native/PHASE0_NOTES.md` — pinned
   conventions (FS rules, IPA layout, Montgomery conventions).
3. `/root/zkllm/zkob_lookup.cuh` and `/root/zkllm/vrf_common.cuh` — shared machinery (these
   are trusted/validated; the question is whether the new driver USES them correctly).
4. `/root/zkllm/zkob_rmsnorm.cu` — the file under review.
5. `/root/zkllm/RMSNORM_REPORT.md` — the author's claims.

## What to hunt for (checklist — go through ALL of these explicitly)
1. **Fiat-Shamir ordering**: is every challenge squeezed only AFTER absorbing every message it
   binds? Compare the prove and verify FS schedules line by line — any absorb present in prove
   but missing in verify (or vice versa), any value absorbed in a different order, any proof
   value that the verifier uses but never absorbs?
2. **Verifier independence**: does verify() recompute everything it should, or does it trust a
   prover-supplied value it shouldn't? Every claim must be anchored either in a commitment
   opening (IPA), a homomorphic relation on commitments, or a public constant. List every
   value read from disk in verify() and say what anchors it.
3. **Openings**: are all ~17 IPA openings actually verified (open_verify called, result
   required), at the right points, against the right commitments, with the right expected
   values? Any opening point reused incorrectly? Any com used with wrong generator count?
4. **The bracket logic**: does the verifier actually enforce P1 = 2^64·C − (R−1)²M and
   P2 = (R+1)²M − 2^64·C through the quartic sumchecks + affine limb links + range lookup,
   such that a prover with R off by ≥2 cannot pass? Walk the algebra. Check the limb-link
   weights, the 2^64·C constant construction, the eq-factor handling in the quartic verifier,
   and the A_f == B_f requirement.
5. **SS sumcheck**: eq factor applied only on row-bit rounds in the verifier? S_f2 == U_f2
   enforced? M opening at u_b against com_M enforced? Could a prover use an M inconsistent
   with X?
6. **Range lookup**: are the limbs proven to be in [0, 65536) via the logUp argument, with the
   multiplicity commitment absorbed before beta/alpha? Is the affine link binding com_P1/com_P2
   to the SAME com_L rows that the lookup constrains?
7. **Hadamard + outer + internal rescale**: is W_ = rescale(W) actually bound, or just
   computed prover-side? (Per spec the binding happens in a SEPARATE zkob_rescale obligation
   via byte-equal commitments — check the driver emits/commits exactly what that requires:
   com_W and com_W_ both absorbed and saved.) Is Y bound to W_ and X at a fresh point?
8. **Padding soundness**: B == B_pad enforced? C padding zero-extended where the algebra needs
   it? Limb matrix zero rows covered by the table (0 in table)?
9. **Selftest honesty**: do the evil modes actually exercise the checks claimed? Would any
   evil mode also be caught earlier by a prover-side sanity check (making the verifier check
   untested)? Are the byte tampers covering ALL files the verifier reads? Is there any file
   the verifier reads that is never tampered?\n10. **Numeric/representation bugs**: Montgomery vs plain mix-ups in any new kernel or host
   math; fr_from_u128 usage; __int128 overflow (check bounds actually guarantee no overflow);
   signed wraparound for negative values mod p.
11. **NEW CUDA kernels in the driver**: list them. Any new G1 kernel = automatic CRITICAL
    (known miscompilation). For Fr kernels, check the Montgomery convention (mont-ify all
    factors except one).

## Rules
- READ ONLY review: do not modify ANY file except writing your report.
- Write your report to `/root/zkllm/RMSNORM_REVIEW.md` with sections:
  VERDICT (one of: SOUND / ISSUES-FOUND / BROKEN), CRITICAL findings, MAJOR, MINOR/notes,
  and for each finding: file:line, the soundness gap or failure it creates (what incorrect prover data the verifier would wrongly accept), and a suggested fix.
  If you find nothing in a category, say explicitly what you checked and why it's fine.
- Be specific and concrete. No hand-waving. If you run anything, only read-only commands
  (you may run `./zkob_rmsnorm selftest` if useful, and you may create/modify files ONLY
  under /tmp/review-scratch/ for experiments — never touch /root/zkllm itself except the
  report file).
