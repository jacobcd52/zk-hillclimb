#!/usr/bin/env python3
"""int_layer_ref.py -- NORMATIVE integer reference for the composed INT
transformer layer (p3_int_layer.cuh / p3_int_gadgets.cuh).

Fixed-point llama block, residual scale 2^16 (zkob-style), pure int64
arithmetic.  The exp / silu / inverse-sqrt tables are loaded from
int_tables.bin (dumped by the C++ side) so both implementations share
BIT-IDENTICAL nonlinearities; everything else here is independent code.

Semantics (all divisions are round-half-up by a power of two:
rshu(x, s) = (x + 2^(s-1)) >> s):

  rmsnorm(x, g):  per row:  M = sum x^2 + 2^14;
                  find e >= 0 with m = M >> 2e in [2^14, 2^16)   (unique);
                  Rp = ISQ[m]  (= round(2^32 sqrt(d)/sqrt(m)));
                  R  = (Rp + h) >> e,  h = 2^(e-1) (0 at e=0);
                  W(i,j) = rshu(R * g[j], 16);  y = rshu(W * x, 16)
  proj:           acc = x @ w  (ints, scale 2^32);  y = rshu(acc, 16)
  rope:           y[t,e] = rshu(q[t,e]*C[t,e] + sgn(e)*q[t,e^dh/2]*S[t,e], 14)
                  C/S int cos/sin at scale 2^14, position s = t mod seq
  scores:         z = rshu(qk_acc, 24 + tsh),  tsh = (log2(dh)+1)//2
  softmax:        mx = causal row max; t = mx - z (allowed) else 2^16;
                  E = EXPT[t]  (= round(2^16 exp(-t/2^8)), 0 for t >= 2^16);
                  S = sum E;  P = (2^17 E + S) // (2 S)   (= round_half_up)
  swiglu:         sil = SILU[G + 2^19]; mo = rshu(sil * U, 16)
  residual:       plain add

Usage:  python3 int_layer_ref.py [trace.bin] [tables.bin]
Recomputes the whole chain from (x0, weights, tables) and compares every
dumped intermediate BITWISE.  Prints REF_OK or the first mismatch.
"""
import struct, sys
import numpy as np

def rshu(x, s):
    return (x + (1 << (s - 1))) >> s

def load_tables(path):
    with open(path, 'rb') as f:
        hdr = struct.unpack('<5q', f.read(40))
        assert hdr[0] == 0x494E5454
        d, ne, ns, ni = hdr[1], hdr[2], hdr[3], hdr[4]
        expt = np.frombuffer(f.read(8 * ne), dtype=np.int64)
        silu = np.frombuffer(f.read(8 * ns), dtype=np.int64)
        isq = np.frombuffer(f.read(8 * ni), dtype=np.int64)
    return d, expt, silu, isq

def load_trace(path):
    arrs = {}
    with open(path, 'rb') as f:
        hdr = struct.unpack('<8q', f.read(64))
        assert hdr[0] == 0x49545246
        cfg = dict(seq=hdr[1], d=hdr[2], nh=hdr[3], dh=hdr[4], dff=hdr[5],
                   batch=hdr[6])
        while True:
            b = f.read(8)
            if len(b) < 8:
                break
            nl = struct.unpack('<q', b)[0]
            nm = f.read(nl).decode()
            n = struct.unpack('<q', f.read(8))[0]
            arrs[nm] = np.frombuffer(f.read(8 * n), dtype=np.int64).copy()
    return cfg, arrs

def rmsnorm(x, g, isq, T, d):
    """x: (T,d) int64; returns (M(T), R(T), y(T,d))."""
    M = (x.astype(object) ** 2).sum(axis=1) + (1 << 14)   # exact (fits i64 anyway)
    M = np.array([int(v) for v in M], dtype=np.int64)
    R = np.zeros(T, dtype=np.int64)
    for i in range(T):
        Mi = int(M[i]); e = 0
        while (Mi >> (2 * e)) >= (1 << 16):
            e += 1
        m = Mi >> (2 * e)
        assert (1 << 14) <= m < (1 << 16)
        Rp = int(isq[m])
        h = (1 << (e - 1)) if e else 0
        R[i] = (Rp + h) >> e
    W = rshu(R[:, None] * g[None, :], 16)
    y = rshu(W * x, 16)
    return M, R, y

def rope(q, ct, st, seq, dh, T, d):
    dh2 = dh // 2
    y = np.zeros_like(q)
    for t in range(T):
        s = t % seq
        for e in range(d):
            jj = e % dh
            hb = (jj >> int(np.log2(dh) - 1)) & 1
            j2 = jj % dh2
            C = int(ct[s * dh2 + j2]); S = int(st[s * dh2 + j2])
            S2 = S if hb else -S
            qf = int(q[t, e ^ (dh // 2)])
            y[t, e] = rshu(int(q[t, e]) * C + qf * S2, 14)
    return y

def softmax_int(z, expt, A, seq):
    """z: (A*seq, seq); returns mx, S, P."""
    NR = A * seq
    mx = np.zeros(NR, dtype=np.int64)
    S = np.zeros(NR, dtype=np.int64)
    P = np.zeros_like(z)
    for r in range(NR):
        i = r % seq
        mx[r] = z[r, :i + 1].max()
        Ssum = 0
        E = np.zeros(seq, dtype=np.int64)
        for j in range(seq):
            t = int(mx[r]) - int(z[r, j]) if j <= i else (1 << 16)
            assert 0 <= t < (1 << 17)
            E[j] = expt[t]
            Ssum += int(E[j])
        S[r] = Ssum
        for j in range(seq):
            P[r, j] = ((1 << 17) * int(E[j]) + Ssum) // (2 * Ssum)
    return mx, S, P

def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else 'int_layer_trace.bin'
    tables = sys.argv[2] if len(sys.argv) > 2 else 'int_tables.bin'
    d_tab, expt, silu, isq = load_tables(tables)
    cfg, a = load_trace(trace)
    seq, d, nh, dh, dff, batch = (cfg['seq'], cfg['d'], cfg['nh'], cfg['dh'],
                                  cfg['dff'], cfg['batch'])
    assert d == d_tab, "table d mismatch"
    T = seq * batch; A = batch * nh
    tsh = (int(np.log2(dh)) + 1) // 2

    fails = []
    def chk(nm, got):
        got = np.asarray(got, dtype=np.int64).reshape(-1)
        want = a[nm]
        if got.shape != want.shape or not np.array_equal(got, want):
            i = int(np.argmax(got != want)) if got.shape == want.shape else -1
            fails.append((nm, i, int(want[i]) if i >= 0 else None,
                          int(got[i]) if i >= 0 else None))
            print(f"  MISMATCH {nm}: first at {i}: dumped={want[i] if i>=0 else '?'} "
                  f"ref={got[i] if i>=0 else '?'}")
        else:
            print(f"  ok {nm}")

    x0 = a['x0'].reshape(T, d)
    g1, g2 = a['g1'], a['g2']
    w = {nm: a[nm] for nm in ('wq', 'wk', 'wv', 'wo', 'wg', 'wu', 'wd')}
    ct, st = a['ct'], a['st']

    # rms1
    M, R, h1 = rmsnorm(x0, g1, isq, T, d)
    chk('rmsM', M); chk('rmsR', R); chk('h1', h1)
    # QKV projections (+ Wo/Wg/Wu/Wd shapes below)
    def proj(x, wname, K, N):
        wm = w[wname].reshape(K, N)
        return x.astype(object) @ wm.astype(object)   # exact
    def toi64(acc):
        return np.array([[int(v) for v in row] for row in acc], dtype=np.int64)
    accq = toi64(proj(h1, 'wq', d, d)); chk('accq', accq)
    acck = toi64(proj(h1, 'wk', d, d)); chk('acck', acck)
    accv = toi64(proj(h1, 'wv', d, d)); chk('accv', accv)
    yq = rshu(accq, 16); yk = rshu(acck, 16); yv = rshu(accv, 16)
    chk('yq', yq); chk('yk', yk); chk('yv', yv)
    # rope
    rq = rope(yq, ct, st, seq, dh, T, d); chk('rq', rq)
    rk = rope(yk, ct, st, seq, dh, T, d); chk('rk', rk)
    # attention scores per (b,h)
    sc = np.zeros((A * seq, seq), dtype=np.int64)
    for b in range(batch):
        for h in range(nh):
            ai = b * nh + h
            qs = rq[b * seq:(b + 1) * seq, h * dh:(h + 1) * dh]
            ks = rk[b * seq:(b + 1) * seq, h * dh:(h + 1) * dh]
            sc[ai * seq:(ai + 1) * seq, :] = toi64(qs.astype(object) @ ks.astype(object).T)
    chk('sc', sc)
    z = rshu(sc, 24 + tsh); chk('z', z)
    mx, S, P = softmax_int(z, expt, A, seq)
    chk('mx', mx); chk('S', S); chk('p', P)
    # P.V per (b,h) -> concat
    pva = np.zeros((T, d), dtype=np.int64)
    for b in range(batch):
        for h in range(nh):
            ai = b * nh + h
            ps = P[ai * seq:(ai + 1) * seq, :]
            vs = yv[b * seq:(b + 1) * seq, h * dh:(h + 1) * dh]
            pva[b * seq:(b + 1) * seq, h * dh:(h + 1) * dh] = \
                toi64(ps.astype(object) @ vs.astype(object))
    chk('pva', pva)
    at = rshu(pva, 16); chk('at', at)
    acco = toi64(proj(at, 'wo', d, d)); chk('acco', acco)
    yo = rshu(acco, 16); chk('yo', yo)
    res1 = x0 + yo; chk('res1', res1)
    # rms2 + MLP
    _, _, h2 = rmsnorm(res1, g2, isq, T, d); chk('h2', h2)
    accg = toi64(proj(h2, 'wg', d, dff)); chk('accg', accg)
    accu = toi64(proj(h2, 'wu', d, dff)); chk('accu', accu)
    gg = rshu(accg, 16); chk('gg', gg)
    uu = rshu(accu, 16); chk('uu', uu)
    sil = silu[gg + (1 << 19)]
    mo = rshu(sil * uu, 16); chk('mo', mo)
    accd = toi64(proj(mo, 'wd', dff, d)); chk('accd', accd)
    yd = rshu(accd, 16); chk('yd', yd)
    out = res1 + yd; chk('out', out)

    if fails:
        print(f"REF_FAIL: {len(fails)} mismatching arrays")
        sys.exit(1)
    print("REF_OK: every intermediate matches the C++ witness replay bitwise")

if __name__ == '__main__':
    main()
