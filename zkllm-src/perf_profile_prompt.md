# Task: profile the proving + verification pipeline — where do the minutes actually go?

The faithful-arch-v1 run proves in 1062 s and verifies in 999 s (llama-68m, seq 1024,
RTX 4090). Produce a precise cost breakdown so optimization targets the real hot spots.

1. Read ORCHESTRATOR_REPORT.md (all stages) and the per-obligation timings in
   /root/zkorch/stage3v2-fa/prove_manifest.json + transcript.json timing block —
   aggregate: total seconds by DRIVER TYPE (fc/rescale/glu/rmsnorm/rope/headslice/
   softmax8/rowmax/headmerge/skip) and by COST CLASS.
2. Instrument INSIDE the two most expensive driver types (no protocol changes — wall-
   clock printf timers in a COPY of the driver source compiled to a separate binary,
   e.g. zkob_softmax8_prof; never overwrite the trusted binaries/sources): split time
   into (a) commitment MSMs, (b) lookup machinery (inv grids, multiplicities, phase1/2
   recursion), (c) sumcheck round kernels, (d) IPA openings (incl. me_weights host
   loops), (e) host<->device transfers + serialization, (f) field inversions. Run at
   real scale, report the split.
3. Same for verification: instrument verify paths of the two most expensive drivers +
   measure the orchestrator-level distribution from transcript timing.
4. Identify the top 5 concrete optimization targets with estimated savings each (e.g.
   "me_weights host loop: X s across run", "com_* recommit duplication: Y s",
   "serialized GPU lock idle: Z s", "lookup inv_grid inversions: W s"). Check GPU
   utilization during a sample prove (nvidia-smi dmon in background) — is the GPU even
   busy, or are we host-bound?
5. Deliverable: /workspace/projects/zk-hillclimb/PERF_PROFILE.md — the breakdown tables,
   utilization findings, top-5 targets with numbers, and honest method notes.
Rules: do NOT modify any trusted source/binary (profiling copies only, *_prof names);
no git commits; GPU is free-ish (a web-research agent runs concurrently but uses no GPU).
