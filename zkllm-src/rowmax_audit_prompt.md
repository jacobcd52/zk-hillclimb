# Task: independent soundness audit of /root/zkllm/zkob_rowmax.cu

OUR OWN defensive codebase; standard second-engineer review before the file enters the
trusted base. Selftest reports ALL PASS (160 checks, coordinator re-ran). Audit for
soundness gaps the selftest would not catch. Bar/format: RMSNORM_REVIEW.md,
SOFTMAX_REVIEW.md, ROPE_REVIEW.md.

Read: STAGE3_FAITHFUL_DESIGN.md §2 (the normative spec — statement §2.1, obligations
§2.3, capacity §2.4, bounds §2.5, FS §2.6, files §2.7, kernel §2.8, selftest §2.9);
PHASE0_NOTES.md; the shared headers (trusted; audit the USAGE); zkob_rowmax.cu (under
review, 1633 lines); ROWMAX_REPORT.md (claims; includes a memory-fix round with a
byte-identity assertion — verify the assertion's basis).

Walk ALL of:
1. FS ordering prove-vs-verify absorb-for-absorb vs §2.6 (incl. TSTAR preamble in vpad).
2. Verifier independence — every disk read anchored (table format per precedent).
3. The constant-claim discipline: BIN/SUM/MASK/T-BIND claims imposed by the verifier at
   round 0 AND required equal to serialized claim_H, never absorbed (this exact gap was
   found-and-fixed during implementation as evil=3 — verify the fix is complete across
   ALL FOUR constant-claim instances, not just BIN).
4. The X1–X5 algebra: BIN's U_f2 == S_f2 − 1 (load-bearing); SUM/MASK weight
   construction (AL and 1−AL — pads and masked positions); ATT's three openings; DOM's
   c2 − c1 == v0 + 2^20·v1 and the mx broadcast binding at pt_c2's row-bit suffix;
   plane-bit openings of com_L; the field-wrap reliance note (§2.1) — is anything
   beyond the documented chain-composition assumption silently relied on?
5. The selector tie channel: confirm the implementation matches §2.4 (canonical lowest-
   index witness; no extra freedom beyond ties).
6. k_pp_expand + fast_me_weights + fast_s_vector: Montgomery convention; the element-
   exact cross-checks vs slow header versions (present? at both scales? in every honest
   prove or only selftest?); could fast/slow divergence slip through in prove-only paths?
7. Both modes' layout math (§2.5): n1 values, com row counts, the NPL=1 vs 2 plane
   handling, vpad zero-padding of z columns V→NCOL.
8. Selftest honesty: every evil mode hits exactly the named check; byte tampers cover
   every file verify() reads; the memory-fix byte-identity claim (re-run the toy diff
   yourself if cheap).
9. New kernels: exactly ONE Fr-only kernel (k_pp_expand) — any G1 kernel = CRITICAL.

READ-ONLY except /root/zkllm/ROWMAX_REVIEW.md: VERDICT (SOUND/ISSUES-FOUND/BROKEN),
CRITICAL/MAJOR/MINOR per finding with file:line + what incorrect prover data gets
accepted + fix; clean categories say what was checked. May run the selftest; experiments
under /tmp/rowmax-audit/ only. GPU may be shared with another implementation job.
