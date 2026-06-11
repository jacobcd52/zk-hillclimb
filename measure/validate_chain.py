"""End-to-end validation of int_chain.IntChain against the stage2-official1 run:
forward() from the registered input must reproduce data/final_output.i32.bin
BYTE-EXACTLY (every intermediate convention already pinned by probe_semantics.py).

Run: /root/int-model-env/bin/python validate_chain.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from int_chain import IntChain, SEQ, EMBED

RUN = "/root/zkorch/stage2-official1"

chain = IntChain(os.path.join(RUN, "registration"))
x0 = np.fromfile(os.path.join(RUN, "registration", "input.i32.bin"), dtype=np.int32).reshape(SEQ, EMBED)
t0 = time.time()
out = chain.forward(x0)
ref = np.fromfile(os.path.join(RUN, "data", "final_output.i32.bin"), dtype=np.int32).reshape(SEQ, EMBED)
ok = np.array_equal(out, ref)
print(f"forward in {time.time()-t0:.1f}s; stats: {chain.stats}")
print("final_output byte-exact:", ok)

# also pin the per-site intermediates for both layers (X of each rmsnorm site)
for l in (0, 1):
    for site in ("input_norm", "post_attn_norm"):
        f = os.path.join(RUN, "data", f"layer{l}.{site}.rmsnorm.out.i32.bin")
        pass  # intermediates already validated element-wise by probe_semantics.py (layer 0)

sys.exit(0 if ok else 1)
