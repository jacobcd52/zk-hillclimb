# Task: bring zkob_rowmax's vpad prover memory under the gate + finish validation

State: /root/zkllm/zkob_rowmax.cu (built; binary present) implements STAGE3_FAITHFUL_DESIGN
§2 and its selftest is fully passing EXCEPT the real-scale vpad case fails ONLY on the
memory gate: honest ACCEPT works (prove 43.2 s, verify 4.0 s, tamper rejected) but the
prover's GPU peak is 18.21 GiB vs the ~18 GiB §6.3 gate (design predicted ≈8 GiB — §2.5).
Log: /root/zkllm/rowmax_selftest_check.log. A previous agent session was cut off before
writing the report; the constant-claim fix for evil=3 is already in and passing.

## Your job
1. Profile where the prover's vpad peak comes from (allocation timeline around the limb
   lookup: L 2 GiB + A_L 2 GiB + fold halves + z/S resident + eq/weight buffers — find
   what is ALIVE simultaneously that §2.5's freeing discipline says should not be).
2. Reduce the peak to ≤ 14 GiB by memory discipline ONLY: free each obligation block's
   tensors before the next (§2.5 pinned requirement), reuse buffers, drop prover-side
   copies that can be recomputed or built in halves. PROTOCOL CHANGES ARE FORBIDDEN:
   the FS transcript, all proof files, commitments, and every verifier byte must be
   unchanged. Confirm by re-running a TOY-case prove before/after your change with the
   same seed and `diff -r` the two obdirs — byte-identical is required (do this; report it).
3. Re-run the FULL selftest: everything must pass including the vpad gate line
   (print the new peak). If ≤ 14 GiB is genuinely unreachable without protocol changes,
   get as low as possible, and if still > 18 GiB report honestly that the §6.3
   column-block fallback is needed (do NOT design or implement it — STOP there).
4. Write /root/zkllm/ROWMAX_REPORT.md (the missing deliverable): everything the original
   task's report section asked for (see /root/zkllm/rowmax_impl_prompt.md "Deliverables"
   item 2) PLUS the memory fix (before/after peak, what was changed, the byte-identity
   confirmation) and the evil=3 constant-claim fix history (it is in the code; describe
   it from the source).

Rules: only edit zkob_rowmax.cu; pinned build commands; no edits to headers or other
drivers; no GitHub. GPU free. Honest reporting.
