# Task: independent soundness audit of `/root/zkllm/zkob_softmax.cu`

Context: OUR OWN research codebase (defensive verifiable-inference project — we build the
verifier and need it sound). A first engineer implemented `/root/zkllm/zkob_softmax.cu`
(selftest reports ALL PASS, 119 checks). You are the second engineer doing the required
independent review before the file enters the trusted base. Audit for soundness gaps —
places where the verifier fails to enforce what the spec claims — that the selftest would
not catch. Do not take the selftest on trust; check the logic.

## Read
1. /workspace/projects/zk-hillclimb/SOFTMAX_DESIGN.md — the normative spec (§2 statement,
   §4 obligations, §5 FS schedule, §6 bounds, §7 CLI/files, §8 selftest).
2. /workspace/projects/zk-hillclimb/submissions/baseline-native/PHASE0_NOTES.md — pinned
   conventions.
3. /root/zkllm/zkob_lookup.cuh + /root/zkllm/vrf_common.cuh — trusted shared machinery
   (the question is whether the driver USES it correctly).
4. /root/zkllm/zkob_softmax.cu — under review. 5. /root/zkllm/SOFTMAX_REPORT.md — claims.
6. /root/zkllm/RMSNORM_REVIEW.md — the audit register/bar from the previous component.

## Checklist (walk ALL of these explicitly)
1. FS ordering: prove vs verify absorb-for-absorb identical; §5 schedule honored label by
   label; every challenge derived after what it binds; nothing the verifier uses is
   unabsorbed.
2. Verifier independence: enumerate every disk value verify() reads and its anchor
   (opening / homomorphic relation / public constant / recomputation). The RMSNORM_REVIEW
   table format is the bar.
3. The 16 IPA openings: all called, results required, right points, right commitments,
   right generator counts; the constant-0/1 plane bits handled correctly in fold/IPA.
4. The rounding bracket: does the verifier enforce P = round_half_up(2^16·MK·E/S) exactly —
   I1/I2 identities (plain-field arithmetic, right constants 2^17 and 2^20), limb-plane MLE
   reconstruction (v00 + 2^20·v10) consistent with the committed L layout, the limb lookup
   actually constraining the SAME com_L, masked positions forced to P=0?
5. Row-sum: W_rs = bcast(eq(u_b))⊙MK recomputed and folded by the VERIFIER itself; U_f2==1
   required (load-bearing); S opened at u_b vs com_S; could a prover use S inconsistent
   with E?
6. V2 broadcast binding: U_f2 opened against com_S at the row-bit suffix of pt2 — is the
   bit-slice correct for the pinned layout (LSB-first columns)? Could a prover smuggle a
   non-broadcast U?
7. Exp lookup: com_comb = com_z + r·com_E homomorphic (host helpers); B_f/T_f recomputed
   from the public table; the table loaded from the CLI path; negative z_ handled as field
   elements consistently between witness, table, and combiner; mapped(0)==0 check correctly
   ABSENT (design §4.1).
8. Layout/padding: B==NCOL==power-of-2 enforced; no padding anywhere claimed — verify no
   hidden pad path; lookup layout constraints (n1 values) per §6.
9. Selftest honesty: the five evil modes hit exactly the named checks and nothing earlier
   (prover sanity disabled only for the targeted path); byte tampers cover every file
   verify() reads (cross-check the §7.2 list against the fopen calls); the evil==0
   fold-vs-multi_dim_me convention checks present.
10. Numeric: int64 host math safe per §6; X_E table values vs int32; Fr plain-form
    throughout; any new kernel (should be NONE — automatic CRITICAL if a new G1 kernel
    exists; for any new Fr kernel check the Montgomery convention).

## Rules
READ-ONLY except the report. Write /root/zkllm/SOFTMAX_REVIEW.md: VERDICT
(SOUND / ISSUES-FOUND / BROKEN), CRITICAL, MAJOR, MINOR/notes; per finding file:line, the
soundness gap (what incorrect prover data would be wrongly accepted), suggested fix; for
clean categories state what you checked. You may run ./zkob_softmax selftest; experiments
only under /tmp/audit-scratch/. The GPU is shared with other jobs — retry once on OOM.
