# Task: independent soundness audit of zkob_rope.cu, zkob_headslice.cu, zkob_headmerge.cu

Context: OUR OWN defensive verifiable-inference codebase. A first engineer implemented the
three attention-binding drivers (all selftests ALL PASS — coordinator re-ran them). You are
the required independent reviewer before they enter the trusted base. Audit for soundness
gaps the selftests would not catch. The bar and format: RMSNORM_REVIEW.md and
SOFTMAX_REVIEW.md (read both registers first).

## Read
1. /workspace/projects/zk-hillclimb/ROPE_ATTENTION_DESIGN.md — the normative spec.
2. PHASE0_NOTES.md (conventions), /root/zkllm/zkob_lookup.cuh + vrf_common.cuh (trusted
   machinery — audit the USAGE).
3. The three files under review: /root/zkllm/zkob_rope.cu, zkob_headslice.cu,
   zkob_headmerge.cu; plus /root/zkllm/ROPE_IMPL_REPORT.md (coordinator-written — note
   its open question about the rope/merge proof-size mapping and resolve it).

## Checklist (ALL, per driver)
1. FS ordering: prove vs verify absorb-for-absorb; §5 schedules label-for-label; every
   challenge after what it binds.
2. Verifier independence: enumerate every disk read in verify() + its anchor (the
   RMSNORM_REVIEW table is the format).
3. Openings: rope 3 IPAs (esp. pt2' = (pt2[0..5), 1−pt2[5], pt2[6..)) computed by the
   VERIFIER, opened against com_T with eval S_f2); headslice 6 per head ×12 (paired
   openings MUST both verify against the SAME absorbed eval; bits(h) at positions 6–9;
   the KhT coordinate swap); headmerge 13 (12 × com_O{hh} + com_O2).
4. The algebra: rope c1+c2==ev with W1/W2 REBUILT by the verifier (signs! σ(e) = (e&32)?
   +1:−1 — check sign handling end to end incl. mod-p negatives); headslice Schwartz-Zippel
   binding (would a slice differing from the head block actually fail?); headmerge
   Σ c_h == ev with Wm_h built from the π⁻¹ gather (check the π⁻¹ formula against the
   design §1.3 EXACTLY — m = e·1024+t, i = m div 768, j = m mod 768 — an index swap here
   is the classic bug and the selftest's B==C transpose case may NOT catch all of it:
   analyze which π errors the B≠C toy cases would miss).
5. Public-weight folds: verifier-side rebuild of W1/W2/Wm_h and the U_f2==1 requirements
   present and load-bearing where the design says.
6. Padding: rope's [e<768] weight rule on both sides; headmerge forcing O2 padding to
   exact zero via the identity; headslice's no-pad slice shapes; flip staying inside
   padding blocks.
7. Selftest honesty: evil modes hit exactly the named checks (esp. rope evil=2 — wrong
   permutation — and any headslice swapped-head / transposed-wrong evil); byte tampers
   cover every file verify() reads; evil==0 convention checks present.
8. Numeric: int64 safety per §6; negative table values mod-p consistent between
   witness-side and verifier-side weight construction; the |Y64| ≥ 2^47 throw.
9. New kernels: should be ZERO (automatic CRITICAL for any new G1 kernel; flag any new
   Fr kernel and check the Montgomery rule). Also confirm whether me_weights memoization
   was applied in headslice (host-only is sanctioned) and that it does not change results.
10. Resolve the ROPE_IMPL_REPORT proof-size question (which driver produced 1.97 MB vs
    309 KB) by inspecting the obdir file lists in the selftest logs or re-deriving from
    §7.2.

## Rules
READ-ONLY except the report → /root/zkllm/ROPE_REVIEW.md: VERDICT (SOUND / ISSUES-FOUND /
BROKEN) per driver + overall, CRITICAL/MAJOR/MINOR with file:line, what incorrect prover
data would be wrongly accepted, fix. State what you checked for clean categories. You may
run the selftests; experiments only under /tmp/rope-audit/. GPU otherwise free.
