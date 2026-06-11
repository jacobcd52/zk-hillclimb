# Pinned cos/sin table generator for zkob_rope (ROPE_ATTENTION_DESIGN.md section 2).
# Run ONCE; register both outputs by sha256 in public.json. The sha256 registration,
# not regeneration, is the source of truth; the C++ driver never generates the real
# tables (the selftest's in-driver fallback is flagged non-authoritative).
import numpy as np
SEQ, HEAD_DIM, THETA, SCALE = 1024, 64, 10000.0, 1 << 16
half = HEAD_DIM // 2
inv_freq = THETA ** (-np.arange(half, dtype=np.float64) / half)        # 10000^(-k/32)
ang = np.arange(SEQ, dtype=np.float64)[:, None] * inv_freq[None, :]    # (1024, 32)
ang = np.concatenate([ang, ang], axis=1)                               # (1024, 64) = cat(freqs, freqs)
np.rint(SCALE * np.cos(ang)).astype(np.int32).tofile("rope-cos-table.bin")
np.rint(SCALE * np.sin(ang)).astype(np.int32).tofile("rope-sin-table.bin")
