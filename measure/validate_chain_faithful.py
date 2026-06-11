"""End-to-end validation of int_chain.FaithfulChain against the stage3v2-fa run
(the validated faithful-arch-v1 submission, orchestrator 65/65 ACCEPT):

  registered input -> EVERY chain tensor -> final logits, 100% byte-equality
  against the driver-emitted witness files in <run>/data/, plus
  argmax(logits[:, :32000]) == registration/tstar.i32.bin.

On the first mismatching tensor it reports the divergence precisely and stops.
Also asserts COVERAGE: every witness file in data/ was compared (no silent
skips), so "every chain tensor" is checked, not a subset.

Run: /root/int-model-env/bin/python validate_chain_faithful.py [run_dir]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from int_chain import FaithfulChain, SEQ, EMBED, VOCAB

RUN = sys.argv[1] if len(sys.argv) > 1 else "/root/zkorch/stage3v2-fa"
D = os.path.join(RUN, "data")

chain = FaithfulChain(os.path.join(RUN, "registration"))
chain.trace = {}
x0 = np.fromfile(os.path.join(RUN, "registration", "input.i32.bin"),
                 dtype=np.int32).reshape(SEQ, EMBED)
t0 = time.time()
final = chain.forward(x0)
logits_i = chain.logits_i32(final)
print(f"faithful forward+head in {time.time()-t0:.1f}s; stats: {chain.stats}")

# --- compare every traced tensor against its witness file --------------------
n_ok = 0
for name in sorted(chain.trace):
    path = os.path.join(D, name)
    mine = chain.trace[name]
    if not os.path.exists(path):
        print(f"MISSING witness file for traced tensor: {name}")
        sys.exit(1)
    ref = np.fromfile(path, dtype=mine.dtype)
    if mine.reshape(-1).shape != ref.shape or not np.array_equal(mine.reshape(-1), ref):
        if mine.reshape(-1).shape != ref.shape:
            print(f"FIRST DIVERGENCE {name}: shape {mine.reshape(-1).shape} vs file {ref.shape}")
        else:
            m, r = mine.reshape(-1), ref
            i = int(np.flatnonzero(m != r)[0])
            print(f"FIRST DIVERGENCE {name}: flat index {i}: chain={m[i]} witness={r[i]} "
                  f"({int((m != r).sum())} differing elements)")
        sys.exit(1)
    n_ok += 1
print(f"byte-exact: {n_ok}/{len(chain.trace)} chain tensors match the witness files")

# --- coverage: every data/ witness file must have been compared --------------
witness = []
for root, _, files in os.walk(D):
    for f in files:
        witness.append(os.path.relpath(os.path.join(root, f), D))
uncovered = sorted(set(witness) - set(chain.trace))
if uncovered:
    print(f"UNCOVERED witness files (not compared): {uncovered}")
    sys.exit(1)
print(f"coverage: all {len(witness)} files in data/ compared")

# --- served-token binding: argmax over the real vocab == registered t* -------
tstar = np.fromfile(os.path.join(RUN, "registration", "tstar.i32.bin"), dtype=np.int32)
tstar_chain = np.argmax(logits_i[:, :VOCAB], axis=1).astype(np.int32)
ok = np.array_equal(tstar, tstar_chain)
print("argmax(chain logits) == registered t*:", ok)
sys.exit(0 if ok else 1)
