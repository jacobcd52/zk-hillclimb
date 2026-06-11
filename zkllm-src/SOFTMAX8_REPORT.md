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
