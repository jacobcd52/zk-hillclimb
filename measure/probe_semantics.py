"""Probe: pin every integer-op convention of the witness chain against the
stage-2 run's DRIVER-EMITTED data files (byte-exact checks on real tensors).

This validates that the numpy re-implementation in int_chain.py reproduces the
driver semantics exactly before it is trusted on new (harness-prompt) inputs.

Run: /root/int-model-env/bin/python probe_semantics.py /root/zkorch/stage2-official1
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/projects/zk-hillclimb/orchestrator")
from prove_walk import compute_R  # the pinned integer-exact rmsnorm advice

RUN = sys.argv[1] if len(sys.argv) > 1 else "/root/zkorch/stage2-official1"
D = os.path.join(RUN, "data")
REG = os.path.join(RUN, "registration")
SEQ, EMBED, INTER, HD, NH = 1024, 768, 3072, 64, 12
C_EPS = 3298535
SWIGLU_LOW = -(1 << 21)
SOFTMAX_LOW_E = -(1 << 19)

fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def load(path, dtype, shape=None):
    a = np.fromfile(path, dtype=dtype)
    return a.reshape(shape) if shape is not None else a


def rescale(v, log_sf):
    """zkob_rescale semantics: rem in [-sf/2, sf/2), y = floor((v + sf/2) / sf)."""
    return (v + (1 << (log_sf - 1))) >> log_sf


def imatmul(X, W):
    """Exact integer matmul via float64 BLAS; asserts the no-rounding bound."""
    Y = X.astype(np.float64) @ W.astype(np.float64)
    assert np.abs(Y).max() < 2**52, "float64 matmul exactness bound violated"
    return Y.astype(np.int64)


# --- weights / tables -------------------------------------------------------
W = {}
for f in os.listdir(os.path.join(REG, "weights")):
    if f.endswith("-int.bin"):
        W[f[:-8]] = np.fromfile(os.path.join(REG, "weights", f), dtype=np.int32)
swiglu_tab = load(os.path.join(REG, "swiglu-table.bin"), np.int32)
exp_tab = load(os.path.join(REG, "softmax-exp-table.bin"), np.int32)
cos_tab = load(os.path.join(REG, "rope-cos-table.bin"), np.int32, (SEQ, HD))
sin_tab = load(os.path.join(REG, "rope-sin-table.bin"), np.int32, (SEQ, HD))

x0 = load(os.path.join(REG, "input.i32.bin"), np.int32, (SEQ, EMBED))

l = 0  # probe layer 0; the full validator covers both layers

# --- rmsnorm site (input_norm) ----------------------------------------------
X = load(os.path.join(D, f"layer{l}.input_norm.rmsnorm.X.i32.bin"), np.int32, (SEQ, EMBED))
check("input_norm.X == registered input", np.array_equal(X, x0))
R = load(os.path.join(D, f"layer{l}.input_norm.rmsnorm.R.i32.bin"), np.int32)
check("R == compute_R(X) [exact integer bracket]", np.array_equal(R, compute_R(X, C_EPS)))
g = W[f"layer{l}.input_norm.g"].astype(np.int64)
Wt = load(os.path.join(D, f"layer{l}.input_norm.rmsnorm.W.i64.bin"), np.int64, (SEQ, EMBED))
check("rmsnorm W.i64 == R (x) g (outer product)",
      np.array_equal(Wt, R.astype(np.int64)[:, None] * g[None, :]))
Wr = rescale(Wt, 16)
Y = load(os.path.join(D, f"layer{l}.input_norm.rmsnorm.Y.i64.bin"), np.int64, (SEQ, EMBED))
check("rmsnorm Y.i64 == rescale16(W) * X", np.array_equal(Y, Wr * X.astype(np.int64)))
out = load(os.path.join(D, f"layer{l}.input_norm.rmsnorm.out.i32.bin"), np.int32, (SEQ, EMBED))
check("rmsnorm out == rescale16(Y)", np.array_equal(out.astype(np.int64), rescale(Y, 16)))
attn_in = out

# --- q/k/v proj fc + rescale --------------------------------------------------
proj = {}
for pj in ("q_proj", "k_proj", "v_proj"):
    Wp = W[f"layer{l}.attn.{pj}"].reshape(EMBED, EMBED)
    Yp = load(os.path.join(D, f"layer{l}.attn.{pj}.matmul.Y.i64.bin"), np.int64, (SEQ, EMBED))
    check(f"{pj} fc Y.i64 == X @ W_int", np.array_equal(Yp, imatmul(attn_in, Wp)))
    op = load(os.path.join(D, f"layer{l}.attn.{pj}.rescaling.out.i32.bin"), np.int32, (SEQ, EMBED))
    check(f"{pj} rescale out == rescale16(Y)", np.array_equal(op.astype(np.int64), rescale(Yp, 16)))
    proj[pj] = op

# --- rope ---------------------------------------------------------------------
e = np.arange(EMBED)
sigma = np.where((e & 32) != 0, 1, -1).astype(np.int64)
W1 = cos_tab[:, e % HD].astype(np.int64)              # (SEQ, EMBED)
W2 = sigma[None, :] * sin_tab[:, e % HD].astype(np.int64)
flip = e ^ 32
roped = {}
for t, pj in (("q", "q_proj"), ("k", "k_proj")):
    T = proj[pj].astype(np.int64)
    Y64 = T * W1 + T[:, flip] * W2
    Y64f = load(os.path.join(D, f"layer{l}.attn.rope.{t}.Y64.i64.bin"), np.int64, (SEQ, EMBED))
    check(f"rope.{t} Y64 == T*W1 + T[flip]*W2", np.array_equal(Y64, Y64f))
    of = load(os.path.join(D, f"layer{l}.attn.rope.{t}.out.i32.bin"), np.int32, (SEQ, EMBED))
    check(f"rope.{t} out == rescale16(Y64)", np.array_equal(of.astype(np.int64), rescale(Y64f, 16)))
    roped[t] = of

# --- headslice ----------------------------------------------------------------
sd = os.path.join(D, f"layer{l}.attn.slice")
hh = 3
Qh = load(os.path.join(sd, f"Qh{hh:02d}.i32.bin"), np.int32, (SEQ, HD))
KhT = load(os.path.join(sd, f"KhT{hh:02d}.i32.bin"), np.int32, (HD, SEQ))
Vh = load(os.path.join(sd, f"Vh{hh:02d}.i32.bin"), np.int32, (SEQ, HD))
check("slice Qh == Qr[:, 64h:64h+64]", np.array_equal(Qh, roped["q"][:, 64 * hh:64 * hh + 64]))
check("slice KhT == Kr slice transposed", np.array_equal(KhT, roped["k"][:, 64 * hh:64 * hh + 64].T))
check("slice Vh == V slice", np.array_equal(Vh, proj["v_proj"][:, 64 * hh:64 * hh + 64]))

# --- scores -> softmax -> values for one head ---------------------------------
z = load(os.path.join(D, f"layer{l}.attn.scores.h{hh:02d}.z.i64.bin"), np.int64, (SEQ, SEQ))
check("scores z == Qh @ KhT", np.array_equal(z, imatmul(Qh, KhT)))
z13 = load(os.path.join(D, f"layer{l}.attn.scores.h{hh:02d}.z13.i32.bin"), np.int32, (SEQ, SEQ))
check("z13 == rescale13(z)", np.array_equal(z13.astype(np.int64), rescale(z, 13)))
z_ = load(os.path.join(D, f"layer{l}.attn.scores.h{hh:02d}.z_.i32.bin"), np.int32, (SEQ, SEQ))
check("z_ == rescale10(widen(z13))", np.array_equal(z_.astype(np.int64), rescale(z13.astype(np.int64), 10)))

E = exp_tab[z_ - SOFTMAX_LOW_E].astype(np.int64)
MK = np.tril(np.ones((SEQ, SEQ), dtype=np.int64))
S = (E * MK).sum(axis=1)
# P = round_half_up(2^16 * MK * E / S): unique int with r1 = 2^17*MK*E + S - 2*P*S in [0, 2S)
P = (((MK * E) << 17) + S[:, None]) // (2 * S[:, None])
Pf = load(os.path.join(D, f"layer{l}.attn.softmax.h{hh:02d}.P.i32.bin"), np.int32, (SEQ, SEQ))
check("softmax P == round_half_up(2^16*MK*E/S)", np.array_equal(P, Pf.astype(np.int64)))

out64 = load(os.path.join(D, f"layer{l}.attn.values.h{hh:02d}.out64.i64.bin"), np.int64, (SEQ, HD))
check("values out64 == P @ Vh", np.array_equal(out64, imatmul(Pf, Vh)))
oh = load(os.path.join(D, f"layer{l}.attn.values", f"out{hh:02d}.i32.bin"), np.int32, (SEQ, HD))
check("values out == rescale16(out64)", np.array_equal(oh.astype(np.int64), rescale(out64, 16)))

# --- headmerge (line-157 pi permutation) ---------------------------------------
M = np.empty((SEQ, EMBED), dtype=np.int32)
for h in range(NH):
    M[:, 64 * h:64 * h + 64] = load(os.path.join(D, f"layer{l}.attn.values", f"out{h:02d}.i32.bin"),
                                    np.int32, (SEQ, HD))
O2 = M.T.reshape(SEQ, EMBED)  # transpose(0,1).reshape(seq, embed) — the line-157 permutation
attn_out = load(os.path.join(D, f"layer{l}.attn_out.i32.bin"), np.int32, (SEQ, EMBED))
check("attn_out == pi(concat(out_h)) [line-157]", np.array_equal(O2, attn_out))

# --- skips + post_attn_norm + MLP ----------------------------------------------
z1 = x0.astype(np.int64) + attn_out.astype(np.int64)
Xp = load(os.path.join(D, f"layer{l}.post_attn_norm.rmsnorm.X.i32.bin"), np.int32, (SEQ, EMBED))
check("attn_skip: post_attn_norm.X == input + attn_out", np.array_equal(z1, Xp.astype(np.int64)))

ffn_in = load(os.path.join(D, f"layer{l}.post_attn_norm.rmsnorm.out.i32.bin"), np.int32, (SEQ, EMBED))
Wg = W[f"layer{l}.mlp.gate_proj"].reshape(EMBED, INTER)
Wu = W[f"layer{l}.mlp.up_proj"].reshape(EMBED, INTER)
Wd = W[f"layer{l}.mlp.down_proj"].reshape(INTER, EMBED)
Yg = load(os.path.join(D, f"layer{l}.mlp.gate_proj.matmul.Y.i64.bin"), np.int64, (SEQ, INTER))
check("gate fc Y == X @ W", np.array_equal(Yg, imatmul(ffn_in, Wg)))
G = load(os.path.join(D, f"layer{l}.mlp.gate_proj.rescaling.out.i32.bin"), np.int32, (SEQ, INTER))
check("gate rescale20", np.array_equal(G.astype(np.int64), rescale(Yg, 20)))
U = load(os.path.join(D, f"layer{l}.mlp.up_proj.rescaling.out.i32.bin"), np.int32, (SEQ, INTER))
H = load(os.path.join(D, f"layer{l}.mlp.swiglu.H.i64.bin"), np.int64, (SEQ, INTER))
check("glu H == swiglu_tab[G - LOW] * U",
      np.array_equal(H, swiglu_tab[G - SWIGLU_LOW].astype(np.int64) * U.astype(np.int64)))
Hr = load(os.path.join(D, f"layer{l}.mlp.swiglu.Hr.i32.bin"), np.int32, (SEQ, INTER))
check("hrescale16", np.array_equal(Hr.astype(np.int64), rescale(H, 16)))
Yd = load(os.path.join(D, f"layer{l}.mlp.down_proj.matmul.Y.i64.bin"), np.int64, (SEQ, EMBED))
check("down fc Y == Hr @ W", np.array_equal(Yd, imatmul(Hr, Wd)))
dout = load(os.path.join(D, f"layer{l}.mlp.down_proj.rescaling.out.i32.bin"), np.int32, (SEQ, EMBED))
check("down rescale16", np.array_equal(dout.astype(np.int64), rescale(Yd, 16)))

z2 = z1 + dout.astype(np.int64)
Xn = load(os.path.join(D, "layer1.input_norm.rmsnorm.X.i32.bin"), np.int32, (SEQ, EMBED))
check("mlp_skip: layer1.input_norm.X == z1 + ffn_out", np.array_equal(z2, Xn.astype(np.int64)))

print(f"\n{'ALL SEMANTICS PINNED' if not fails else f'{len(fails)} FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
