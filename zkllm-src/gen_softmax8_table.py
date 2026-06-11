# Pinned exp-table generator for zkob_softmax8 (STAGE3_FAITHFUL_DESIGN.md
# section 4.3, the temperature-8 table). Run ONCE; register the output by
# sha256 in public.json next to the temp-128 softmax table. The sha256
# registration, not regeneration, is the source of truth; the C++ driver never
# generates the real table (the selftest's in-driver fallback is flagged
# non-authoritative). Domain v in [-1048574, +1]; v = z_ - mx <= 0 for allowed
# entries (exponent = v/2^12: scale 2^9 times temperature 8); the top slot
# v = +1 is the pinned masked-position sentinel row, mapped to 0.
import numpy as np
LOW8, LEN8 = -(1 << 20) + 2, 1 << 20          # domain v in [-1048574, +1]
v = np.arange(LOW8, LOW8 + LEN8, dtype=np.float64)
tab = np.rint(65536.0 * np.exp(v / 4096.0))    # exponent = v/4096  (scale 2^9 * temp 8)
tab[v > 0] = 0.0                               # sentinel row(s): v = +1 maps to 0
np.rint(tab).astype(np.int32).tofile("softmax8-exp-table.bin")
