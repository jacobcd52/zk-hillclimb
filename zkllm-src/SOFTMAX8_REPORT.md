# SOFTMAX8_REPORT — zkob_softmax8 + zkob_headmerge perm flag

Implemented by a Fable agent from STAGE3_FAITHFUL_DESIGN.md §4.2/§4.3 (agent session ended
before writing this report; coordinator completed it from the acceptance logs, 2026-06-11).

- zkob_softmax8: temp-8 softmax, mx chained from zkob_rowmax (RM2 com_mx byte-identity
  confirmed in selftest), E8 table + masked sentinel per §4.3, 2x14-bit limbs.
  Coordinator acceptance: ALL PASS, 167 checks (log: softmax8_selftest_accept.log).
  8 evil modes all rejected by exactly the named checks across 3 toy shapes.
- zkob_headmerge: pinned <perm> flag (pi157|concat), PERM absorbed after HD, minimal diff;
  cross-mode splice test REJECTS both directions. Coordinator acceptance: ALL PASS,
  166 checks both modes (log: headmerge_selftest_accept2.log).
- Shared headers untouched (agent sha256s: zkob_lookup.cuh d7fcd101…, vrf_common.cuh
  09965932… identical before/after).
- gen_softmax8_table.py written per §4.3 pinned generator.
- One implementation-round note: a selftest harness bug (z-envelope guard test recomputed
  mx from oversized z so the mx guard fired first) was found and fixed by the agent.
Audit pending (next gate).

## Hardening round (2026-06-11, post-audit)

The independent audit (SOFTMAX8_REVIEW.md, VERDICT: **SOUND** for both zkob_softmax8.cu
and the zkob_headmerge perm-flag diff; 0 critical, 0 major, 7 MINOR) was applied
finding-by-finding under the standard scope rules. **verify() is byte-for-byte unchanged
in both drivers; prove()'s honest path is byte-for-byte unchanged in both drivers;
zkob_headmerge.cu is entirely unchanged** (its two findings are an orchestrator pin and
a no-action note). The only code edits are in zkob_softmax8.cu: a selftest-side negative
test and two comment blocks (the file-header interlock note and the new test's comment).
Shared headers untouched (`sha256sum`: zkob_lookup.cuh `d7fcd101…`, vrf_common.cuh
`09965932…` — identical before/after, sha256-prove untouched).

Per finding:

- **MINOR-1 (z_/mx not range-proven; chain edges load-bearing) → documented, pinned.**
  The audit's fix text is an orchestrator obligation, not a driver change: enforce
  **RM1/RM2/SX8a/SX8b byte-identically and run the chained rowmax instance** — a
  standalone softmax8 ACCEPT binds (Dm, E, S, P) internally consistent with the
  committed z_/mx but does not pin z_/mx upstream, prove allowed z−mx ≤ 0, or exclude a
  field wrap. Pinned in PHASE0_NOTES.md §20. The optional design-Q5 z_ range belt is
  explicitly "NOT included in v1 — acceptable" per the audit; verify() was therefore
  left untouched (adding it would also break the audited proof format/FS schedule).
- **MINOR-2 (missing/short proof file ⇒ throw, not REJECT) → documented, pinned.**
  Audit text: fail-closed as-is; **the orchestrator must treat ANY nonzero exit as
  reject**, never parse for the REJECT line. Pinned in PHASE0_NOTES.md §20 (same
  posture as softmax MINOR-4 / rope MINOR-7). No code change, by the audit's text.
- **MINOR-3 (MK-free row-sum/V1/bracket weights sound only via the masked-E=0
  interlock) → selftest pin + source note.** The interlock (Dm identity + mapping
  lookup + sentinel check) is now (a) spelled out in a SOUNDNESS INTERLOCK block in the
  file header so a future edit to the Dm block / table loading / registered table must
  re-establish masked-E=0 before touching the MK-free weights, and (b) **negatively
  tested**: a new selftest check verifies an HONEST proof dir against a public table
  whose last entry ≠ 0 and requires rejection at exactly **"table sentinel (last
  entry) != 0"** — the verify()-side sentinel check (verify line ~660), which was
  previously only positively exercised (the table is a CLI/registered input, not a
  proof file, so the 40-file byte-tamper loop never touched it; only the prove-side
  throw had a guard test). Rejected on the exact named check in all three toy cases on
  the first run.
- **MINOR-4 (no standalone semantic forgery test for cD1's z_-side terminal) → no
  action, by the audit's text.** cD1 folds the committed z_ directly (no
  never-committed intermediate analogous to cD2's broadcast), so there is nothing to
  forge that the ipa_z_cD1.bin byte tamper and every honest pass don't already cover;
  the audit's summary says a semantic cD1 note "is not needed". Documented only.
- **MINOR-5 (real-scale tamper is single-file) → no action, by the audit's text
  ("Acceptable").** The three toy shapes tamper all 40 verifier-read files each with
  restore-and-reverify; real scale tampers lookup_E8.bin. Same posture as
  softmax/rope.
- **MINOR-6 (reused FS labels, duplicated static helpers — cosmetic) → no action, as
  the audit prescribes ("noted so nobody 'fixes' it").** Positional disambiguation is
  verified harmless; if fold_public/tamper_byte/file_size are ever hoisted into a
  shared header, the edit-requires-rerunning-EVERY-selftest rule applies.
- **MINOR-7 (headmerge pi157 transcripts not binary-compatible across the flag — by
  design) → documented, pinned.** The added "PERM" absorb is a domain-separation
  strengthening; the baseline-native submission keeps its own f792978 binary and
  remains re-verifiable. **The orchestrator must pass `headmerge_perm` from
  public.json to BOTH prove and verify** for each submission (mode/argv mismatch is
  caught by the dims cross-check + transcript divergence, both directions
  splice-tested). Pinned in PHASE0_NOTES.md §21. No code change.

Design-doc errata for the coordinator (STAGE3_FAITHFUL_DESIGN.md NOT edited, per the
standing rule): §4.3 names the table generator `gen_softmax8_exp_table.py`; the actual
pinned file is **`gen_softmax8_table.py`** (contents verbatim the §4.3 script). No
other errata — the audit found none.

Totals after rebuild (pinned sm_89 `-dc -dlto` compile + standard link list, both
drivers rebuilt):

- **zkob_softmax8 selftest: ALL PASS, 170 PASS / 0 FAIL, exit 0** (was 167; +1
  sentinel-pin check × 3 toy cases). Per toy case: honest ACCEPT + 8 semantic evils +
  the new sentinel-tampered-table rejection + 40 byte tampers + restored ACCEPT (×3
  shapes), 11 guard throws, real-scale chained case (rowmax ACCEPT, mx == host max,
  honest ACCEPT, RM2 + SX8a/RM1 com byte-identities, tamper reject). Real scale:
  **prove 12.03 s, verify 12.84 s, proof+commitments 2,141,552 B** — matches the
  audit's independent re-run (12.15 s / 12.94 s / identical bytes).
  Log: /tmp/softmax8_selftest_hardened.log.
- **zkob_headmerge selftest: ALL PASS, 166 PASS / 0 FAIL, exit 0** (unchanged — no
  code edits; rebuild + re-run as insurance). Both modes ALL PASS; cross-mode splice
  rejects in both directions at both layers (dims.bin mismatch; forged-PERM transcript
  divergence at "merge hadamard 00 round 1"). Real scale per mode: **prove 3.48 s,
  verify 2.18–2.20 s, proof+commitments 1,966,888 B** — matches the audit (3.50 s /
  2.19 s). Log: /tmp/headmerge_selftest_hardened.log.

No rejection expectation was loosened; every new/existing negative test names its exact
check. GPU was shared with a concurrent hardening job during part of the run; no
transient CUDA failures occurred (no retries needed).
