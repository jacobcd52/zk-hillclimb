# Pinned exp-table generator for zkob_softmax (SOFTMAX_DESIGN.md section 7.4).
# Run ONCE; register the output by sha256 in public.json next to the swiglu
# table. The sha256 registration, not regeneration, is the source of truth;
# the C++ driver never generates the real table (the selftest's in-driver
# fallback is flagged non-authoritative).
import numpy as np
LOW_E, LEN_E = -(1 << 19), 1 << 20
v = np.arange(LOW_E, LOW_E + LEN_E, dtype=np.float64)
np.rint(65536.0 * np.exp(v / 65536.0)).astype(np.int32).tofile("softmax-exp-table.bin")
