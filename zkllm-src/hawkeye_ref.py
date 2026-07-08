#!/usr/bin/env python3
"""Bit-exact CPU (numpy) replica of the Hawkeye fp8 accumulator replay
(/workspace/projects/int-model-approximation/src/int_model_approximation/hawkeye.py).

This is the golden generator for the Hawkeye ZKP: `hawkeye_ref` reproduces the
Triton kernel's integer transition logic bit-for-bit (validated below against
the Triton kernel on the GPU), and `dump_product_witness` emits the per-product
witness vectors (decode-multiply lookup row + truncating-alignment q/r) that
drive the p3_hawkeye_prod selftest.

Run:  python3 hawkeye_ref.py            # full Triton-vs-numpy bitwise battery
      python3 hawkeye_ref.py --dump F   # also write witness vectors to F
"""
import sys
import numpy as np

I64 = np.int64


def decode_fp8_e4m3(raw):
    """raw uint8 array -> (exp_eff, signed_sig, nonzero), all int64/bool.
    NOTE: matches the kernel, NOT IEEE: NaN codes 0x7F/0xFF decode as ordinary
    values (exp_eff=15, |sig|=15)."""
    raw = raw.astype(I64)
    sign = (raw >> 7) & 1
    exp_bits = (raw >> 3) & 15
    mant = raw & 7
    sig_abs = np.where(exp_bits != 0, mant | 8, mant)
    exp_eff = np.where(exp_bits != 0, exp_bits, 1)
    signed_sig = np.where(sign != 0, -sig_abs, sig_abs)
    nonzero = sig_abs != 0
    return exp_eff, signed_sig, nonzero


def scale_to_internal(signed_product, iw):
    if iw >= 7:
        return signed_product << (iw - 7)
    mag = np.abs(signed_product) >> (7 - iw)
    return np.where(signed_product < 0, -mag, mag)


def sshift_right_tz(x, shift):
    """signed shift right toward zero, shift clamped to [0,62] elementwise."""
    shift = np.clip(shift, 0, 62)
    mag = np.abs(x) >> shift
    return np.where(x < 0, -mag, mag)


def bit_width(mag):
    w = np.zeros_like(mag)
    for bit in range(0, 31):
        w = np.where(mag >= (I64(1) << bit), bit + 1, w)
    return w


def normalize_total(total, max_exp, iw, zero_exp):
    out_sign = total < 0
    mag = np.abs(total)
    nonzero = mag != 0
    width = bit_width(mag)
    out_exp = max_exp + width - iw
    right = np.maximum(width - iw, 0)
    left = np.maximum(iw - width, 0)
    out_sig = np.where(width > iw, mag >> right, mag << left)
    sub = np.maximum(-126 - out_exp, 0)
    out_sig = np.where(out_exp < -126, out_sig >> sub, out_sig)
    out_exp = np.maximum(out_exp, -126)
    if iw < 24:
        out_sig = out_sig << (24 - iw)
    else:
        out_sig = out_sig >> (iw - 24)
    keep = nonzero & (out_sig != 0)
    out_sig = np.where(keep, out_sig, 0)
    out_exp = np.where(keep, out_exp, zero_exp)
    out_sign = np.where(keep & out_sign, 1, 0)
    return out_sign, out_exp, out_sig


def gfloat_to_f32(sign, exp, sig):
    eb = exp + 127
    leading_missing = (sig & 0x800000) == 0
    eb = np.where(leading_missing & (sig != 0), eb - 1, eb)
    eb = np.clip(eb, 0, 254)
    mant = sig & 0x7FFFFF
    bits = (sign.astype(np.uint64) << 31) | (eb.astype(np.uint64) << 23) | mant.astype(np.uint64)
    bits = np.where(sig != 0, bits, 0).astype(np.uint32)
    return bits.view(np.float32)


def hawkeye_ref(x_codes, x_scale, w_codes, w_scale, *,
                products_per_group=32, internal_width=14, zero_exponent=-139,
                record=None):
    """x_codes: (M,K) uint8, w_codes: (N,K) uint8, scales float32.
    Returns (M,N) bf16-bit-pattern uint16 array (matches the Triton kernel's
    bf16 output viewed as uint16).  If record is a list, appends per-product
    witness dicts in (m-major, n, group, kk) order -- see dump_product_witness."""
    M, K = x_codes.shape
    N, K2 = w_codes.shape
    assert K == K2
    G, IW, ZE = products_per_group, internal_width, zero_exponent

    acc_sign = np.zeros((M, N), I64)
    acc_exp = np.full((M, N), ZE, I64)
    acc_sig = np.zeros((M, N), I64)

    for k0 in range(0, K, G):
        acc_exp_eff = np.where(acc_sig != 0, acc_exp, ZE)
        max_exp = acc_exp_eff.copy()
        # pass 1: max_exp over present products (masked tail loads -> raw 0)
        prods = []
        for kk in range(G):
            k = k0 + kk
            valid = k < K
            a_raw = x_codes[:, k] if valid else np.zeros(M, np.uint8)
            b_raw = w_codes[:, k] if valid else np.zeros(N, np.uint8)
            a_exp, a_sig, a_nz = decode_fp8_e4m3(a_raw)
            b_exp, b_sig, b_nz = decode_fp8_e4m3(b_raw)
            present = a_nz[:, None] & b_nz[None, :]
            prod_exp = a_exp[:, None] + b_exp[None, :] - 14
            max_exp = np.maximum(max_exp, np.where(present, prod_exp, ZE))
            prods.append((a_raw, b_raw, a_sig, b_sig, prod_exp, present))
        # pass 2: aligned contributions
        contribution = np.zeros((M, N), I64)
        for kk in range(G):
            a_raw, b_raw, a_sig, b_sig, prod_exp, present = prods[kk]
            signed_product = a_sig[:, None] * b_sig[None, :]
            scaled = scale_to_internal(signed_product, IW)
            aligned = sshift_right_tz(scaled, max_exp - prod_exp)
            contribution += np.where(present, aligned, 0)
            if record is not None:
                record.append(dict(a=a_raw, b=b_raw, prod_exp=prod_exp,
                                   scaled=scaled, present=present,
                                   shift=np.clip(max_exp - prod_exp, 0, 62),
                                   aligned_masked=np.where(present, aligned, 0)))
        if internal_width < 24:
            acc_base = acc_sig >> (24 - IW)
        else:
            acc_base = acc_sig << (IW - 24)
        aligned_acc = acc_base >> np.clip(max_exp - acc_exp_eff, 0, 62)
        aligned_acc = np.where(acc_sign != 0, -aligned_acc, aligned_acc)
        acc_sign, acc_exp, acc_sig = normalize_total(aligned_acc + contribution,
                                                     max_exp, IW, ZE)
        if record is not None:
            record[-1]['_contribution'] = contribution  # group-level sanity hook

    y = gfloat_to_f32(acc_sign, acc_exp, acc_sig)
    y = (y * x_scale.reshape(-1, 1).astype(np.float32)) * w_scale.reshape(1, -1).astype(np.float32)
    # fp32 -> bf16 RNE, as torch/the GPU store does
    b = y.view(np.uint32)
    rnd = ((b >> 16) & 1) + 0x7FFF
    bf = ((b + rnd) >> 16).astype(np.uint16)
    return bf


# ---------------- per-product witness vectors for the ZKP gadget ----------------

def product_witness_rows(x_codes, w_codes, *, products_per_group=32,
                         internal_width=14, zero_exponent=-139):
    """Flatten the recorded per-product data into gadget witness columns.
    Columns (all int64): a, b, eb (prod_exp+12), mag (=|scaled|), sg, pr(present),
    sh (clamped shift), q, r, al (signed masked aligned value).
    Layout: for each group g, for each kk, all (m,n) row-major."""
    rec = []
    hawkeye_ref(x_codes, np.ones(x_codes.shape[0], np.float32),
                w_codes, np.ones(w_codes.shape[0], np.float32),
                products_per_group=products_per_group,
                internal_width=internal_width, zero_exponent=zero_exponent,
                record=rec)
    cols = {k: [] for k in 'a b eb mag sg pr sh q r al'.split()}
    M = x_codes.shape[0]
    N = w_codes.shape[0]
    for e in rec:
        a2 = np.broadcast_to(e['a'].astype(I64)[:, None], (M, N))
        b2 = np.broadcast_to(e['b'].astype(I64)[None, :], (M, N))
        mag = np.abs(e['scaled'])
        sg = (e['scaled'] < 0).astype(I64)
        sh = e['shift']
        shc = np.minimum(sh, 15)
        q = mag >> shc
        r = mag - (q << shc)
        al = np.where(e['present'], (1 - 2 * sg) * q, 0)
        # cross-check vs the replay's own aligned value
        assert (al == e['aligned_masked']).all()
        cols['a'].append(a2.ravel()); cols['b'].append(b2.ravel())
        cols['eb'].append((e['prod_exp'] + 12).ravel())
        cols['mag'].append(mag.ravel()); cols['sg'].append(sg.ravel())
        cols['pr'].append(e['present'].astype(I64).ravel())
        cols['sh'].append(sh.ravel()); cols['q'].append(q.ravel())
        cols['r'].append(r.ravel()); cols['al'].append(al.ravel())
    return {k: np.concatenate(v).astype(I64) for k, v in cols.items()}


def dump_product_witness(path, layers):
    """Binary format: int64 n_rows, then 10 int64 arrays of length n_rows in
    column order a,b,eb,mag,sg,pr,sh,q,r,al."""
    allc = None
    for (x, w) in layers:
        c = product_witness_rows(x, w)
        allc = c if allc is None else {k: np.concatenate([allc[k], c[k]]) for k in c}
    n = len(allc['a'])
    with open(path, 'wb') as f:
        np.array([n], I64).tofile(f)
        for k in 'a b eb mag sg pr sh q r al'.split():
            allc[k].tofile(f)
    return n, allc


def group_witness_rows(x_codes, w_codes, **kw):
    """Like product_witness_rows but GROUP-CONTIGUOUS: rows reordered to
    (group, kk) with kk INNERMOST, group enumerating (g, m, n) -- the order the
    Binius accumulation gadget requires (the adder-tree pairing coordinate is
    row-index bit 0).  Also returns the golden per-group sums:
    P = sum of positive al, N = sum of |negative al|, S = P - N = sum(al)."""
    G = kw.get('products_per_group', 32)
    cols = product_witness_rows(x_codes, w_codes, **kw)
    M, N = x_codes.shape[0], w_codes.shape[0]
    nk = len(cols['a']) // (G * M * N)
    out = {}
    for k, v in cols.items():
        out[k] = v.reshape(nk, G, M * N).transpose(0, 2, 1).reshape(-1).copy()
    al = out['al'].reshape(-1, G)
    P = np.where(al > 0, al, 0).sum(axis=1)
    Nn = np.where(al < 0, -al, 0).sum(axis=1)
    return out, P.astype(I64), Nn.astype(I64), al.sum(axis=1).astype(I64)


def dump_acc_witness(path, layers):
    """Group-ordered witness + golden group sums for the accumulation gadget.
    Format: int64 n_rows; 10 int64 col arrays (a,b,eb,mag,sg,pr,sh,q,r,al) in
    group-contiguous order; int64 magic 0x41434332 ('ACC2'); int64 n_groups;
    then P, N, S int64 arrays of length n_groups."""
    allc, P, Nn, S = None, [], [], []
    for (x, w) in layers:
        c, p, nn, s = group_witness_rows(x, w)
        allc = c if allc is None else {k: np.concatenate([allc[k], c[k]]) for k in c}
        P.append(p); Nn.append(nn); S.append(s)
    P, Nn, S = np.concatenate(P), np.concatenate(Nn), np.concatenate(S)
    n = len(allc['a'])
    assert n == 32 * len(P)
    # cross-check the reorder: per-group sums of al match P - N
    assert (allc['al'].reshape(-1, 32).sum(axis=1) == S).all()
    with open(path, 'wb') as f:
        np.array([n], I64).tofile(f)
        for k in 'a b eb mag sg pr sh q r al'.split():
            allc[k].tofile(f)
        np.array([0x41434332, len(P)], I64).tofile(f)
        P.tofile(f); Nn.tofile(f); S.tofile(f)
    return n, len(P)


# ---------------- golden full-layer battery for the composed ZKP ----------------

def layer_battery():
    """(name, x_codes, xs, w_codes, ws) tuples covering the composed-proof edge
    cases: shift0, shift>=width, negatives, absent/masked lanes, NaN codes,
    K%32!=0, multi-group acc chains, zero/negative scales, all-zero rows."""
    rng = np.random.default_rng(20260702)
    out = []
    def rnd(B, K, N, name, xs=None, ws=None):
        x = rng.integers(0, 256, (B, K)).astype(np.uint8)
        w = rng.integers(0, 256, (N, K)).astype(np.uint8)
        if xs is None: xs = np.exp(rng.normal(0, 2, B)).astype(np.float32)
        if ws is None: ws = np.exp(rng.normal(0, 2, N)).astype(np.float32)
        out.append((name, x, xs, w, ws))
    rnd(4, 64, 8,  "rand 4x64x8 (2 groups)")
    rnd(2, 40, 4,  "rand 2x40x4 (K%32=8 masked tail)")
    rnd(3, 96, 4,  "rand 3x96x4 (3-group chain)")
    rnd(1, 1, 1,   "minimal 1x1x1")
    rnd(1, 31, 2,  "single partial group K=31")
    rnd(2, 33, 3,  "K%32=1")
    rnd(2, 256, 4, "8-group acc chain")
    # zero and negative scales (bf16 sign-of-zero paths)
    xs = np.array([0.0, 2.5], np.float32); ws = np.array([-3.5, 1.0, 0.0], np.float32)
    rnd(2, 64, 3, "zero & negative scales", xs=xs, ws=ws)
    # directed: huge-exponent first product forces shift>=15 -> al=0 lanes
    x = np.full((2, 32), 0x08, np.uint8); x[0, 0] = 0x78
    w = np.full((3, 32), 0x08, np.uint8); w[0, 0] = 0x78
    out.append(("directed big-shift", x, np.ones(2, np.float32), w, np.ones(3, np.float32)))
    # directed: equal exponents (shift 0), negatives, zeros interleaved
    x = np.tile(np.array([0x3F, 0xBF, 0x00, 0x40], np.uint8), (2, 8))
    w = np.tile(np.array([0xBF, 0x3F, 0x40, 0x00], np.uint8), (3, 8))
    out.append(("directed shift0/neg/zero", x, np.ones(2, np.float32), w, np.ones(3, np.float32)))
    # NaN codes decoded as values + an all-zero X row (Y row = +-0)
    x = rng.integers(0, 256, (3, 40)).astype(np.uint8)
    x[0, ::3] = 0x7F; x[1, ::5] = 0xFF; x[2, :] = 0
    w = rng.integers(0, 256, (4, 40)).astype(np.uint8); w[0, ::4] = 0x7F
    out.append(("nan codes + zero row", x,
                np.array([1.0, -2.0, 3.0], np.float32), w,
                np.exp(rng.normal(0, 2, 4)).astype(np.float32)))
    return out


def dump_layers(path):
    layers = layer_battery()
    with open(path, 'wb') as f:
        np.array([0x484B4C59, len(layers)], I64).tofile(f)   # 'HKLY', count
        for (name, x, xs, w, ws) in layers:
            B, K = x.shape; N = w.shape[0]
            ref = hawkeye_ref(x, xs, w, ws)
            tri = run_triton(x, xs, w, ws)
            assert (ref == tri).all(), f"layer '{name}': ref != triton"
            np.array([B, K, N], I64).tofile(f)
            x.tofile(f); w.tofile(f)
            xs.view(np.uint32).tofile(f); ws.view(np.uint32).tofile(f)
            ref.tofile(f)
            print(f"  dumped '{name}' B={B} K={K} N={N} (triton-checked)")
    print(f"wrote {len(layers)} golden layers to {path}")


# ---------------- Triton bitwise comparison battery ----------------

def run_triton(x_codes, x_scale, w_codes, w_scale, **kw):
    import torch
    sys.path.insert(0, '/workspace/projects/int-model-approximation/src')
    from int_model_approximation.hawkeye import hawkeye_fp8_sum
    xt = torch.from_numpy(x_codes).cuda().view(torch.float8_e4m3fn)
    wt = torch.from_numpy(w_codes).cuda().view(torch.float8_e4m3fn)
    xs = torch.from_numpy(x_scale).cuda()
    ws = torch.from_numpy(w_scale).cuda()
    out, _ = hawkeye_fp8_sum(xt, xs, wt, ws, **kw)
    return out.view(torch.uint16).cpu().numpy()


def battery():
    rng = np.random.default_rng(20260702)
    cfgs = [
        dict(M=4, K=64, N=32),
        dict(M=7, K=40, N=17),                      # masked M/N/K tails
        dict(M=1, K=1, N=1),
        dict(M=8, K=33, N=8),                       # K % G == 1
        dict(M=4, K=256, N=32),                     # 8 groups (acc chain)
        dict(M=3, K=31, N=5),                       # single partial group
        dict(M=4, K=64, N=32, block_m=2, block_n=8),  # tile invariance
        dict(M=4, K=96, N=16, products_per_group=8),  # non-default G
        dict(M=4, K=64, N=16, internal_width=12),     # non-default IW
    ]
    nfail = 0
    for i, cfg in enumerate(cfgs):
        M, K, N = cfg.pop('M'), cfg.pop('K'), cfg.pop('N')
        x = rng.integers(0, 256, (M, K)).astype(np.uint8)
        w = rng.integers(0, 256, (N, K)).astype(np.uint8)
        xs = (np.exp(rng.normal(0, 2, M))).astype(np.float32)
        ws = (np.exp(rng.normal(0, 2, N))).astype(np.float32)
        if i == 1:      # exercise zero / negative scales too
            xs[0] = 0.0; ws[0] = -3.5
        kw_ref = {k: v for k, v in cfg.items() if k not in ('block_m', 'block_n')}
        ref = hawkeye_ref(x, xs, w, ws, **kw_ref)
        tri = run_triton(x, xs, w, ws, **cfg)
        ok = (ref == tri).all()
        nmis = int((ref != tri).sum())
        print(f"  [{'PASS' if ok else 'FAIL'}] M={M} K={K} N={N} {cfg} "
              f"mismatches={nmis}/{M*N}")
        if not ok:
            nfail += 1
            bad = np.argwhere(ref != tri)[:5]
            for m, n in bad:
                print(f"      ({m},{n}): ref=0x{ref[m,n]:04x} triton=0x{tri[m,n]:04x}")
    # directed edge battery: zero codes, +-0, NaN codes, extreme exponent mixes
    edge = np.array([[0x00, 0x80, 0x7F, 0xFF, 0x01, 0x81, 0x78, 0xF8,
                      0x08, 0x88, 0x77, 0xF7, 0x0F, 0x8F, 0x38, 0xB8,
                      0x7E, 0xFE, 0x07, 0x87, 0x40, 0xC0, 0x3F, 0xBF,
                      0x00, 0x7F, 0x01, 0x78, 0x80, 0xFF, 0x81, 0xF8]],
                    dtype=np.uint8)
    for wrow in [edge, np.flip(edge, axis=1).copy()]:
        ref = hawkeye_ref(edge, np.ones(1, np.float32), wrow, np.ones(1, np.float32))
        tri = run_triton(edge, np.ones(1, np.float32), wrow, np.ones(1, np.float32))
        ok = (ref == tri).all()
        print(f"  [{'PASS' if ok else 'FAIL'}] directed edge-code row (K=32)")
        nfail += 0 if ok else 1
    # exhaustive single-product: all 256x256 code pairs, K=1 (shift always 0)
    a_all = np.repeat(np.arange(256, dtype=np.uint8), 1).reshape(256, 1)
    ref = hawkeye_ref(a_all, np.ones(256, np.float32), a_all, np.ones(256, np.float32))
    tri = run_triton(a_all, np.ones(256, np.float32), a_all, np.ones(256, np.float32))
    ok = (ref == tri).all()
    print(f"  [{'PASS' if ok else 'FAIL'}] exhaustive 256x256 single-product grid "
          f"mismatches={int((ref != tri).sum())}/65536")
    nfail += 0 if ok else 1
    return nfail


if __name__ == '__main__':
    print("=== hawkeye_ref vs Triton hawkeye_fp8_sum: bitwise bf16 battery ===")
    nfail = battery()
    print(f"battery: {'ALL PASS' if nfail == 0 else f'{nfail} FAILURES'}")
    if '--dump' in sys.argv:
        path = sys.argv[sys.argv.index('--dump') + 1]
        rng = np.random.default_rng(7)
        layers = []
        # random tiny layers (cover all codes incl. NaN/zero, several groups)
        for (M, K, N) in [(4, 64, 8), (2, 40, 4), (3, 96, 4)]:
            layers.append((rng.integers(0, 256, (M, K)).astype(np.uint8),
                           rng.integers(0, 256, (N, K)).astype(np.uint8)))
        # directed: huge-exponent product first -> forces big shifts (>=15 -> al=0)
        x = np.full((1, 32), 0x08, np.uint8); x[0, 0] = 0x78   # 2^8*1.0 vs 2^-6*1.0
        w = np.full((1, 32), 0x08, np.uint8); w[0, 0] = 0x78
        layers.append((x, w))
        # directed: equal exponents -> shift 0; negatives; zeros interleaved
        x = np.tile(np.array([0x3F, 0xBF, 0x00, 0x40], np.uint8), (1, 8))
        w = np.tile(np.array([0xBF, 0x3F, 0x40, 0x00], np.uint8), (1, 8))
        layers.append((x, w))
        n, cols = dump_product_witness(path, layers)
        mx = {k: int(v.max()) for k, v in cols.items()}
        print(f"wrote {n} product-witness rows to {path}; col maxima: {mx}")
        assert nfail == 0, "REFUSING to bless vectors: Triton mismatch above"
    if '--dumpacc' in sys.argv:
        path = sys.argv[sys.argv.index('--dumpacc') + 1]
        rng = np.random.default_rng(7)
        layers = []
        for (M, K, N) in [(4, 64, 8), (2, 40, 4), (3, 96, 4)]:
            layers.append((rng.integers(0, 256, (M, K)).astype(np.uint8),
                           rng.integers(0, 256, (N, K)).astype(np.uint8)))
        x = np.full((1, 32), 0x08, np.uint8); x[0, 0] = 0x78
        w = np.full((1, 32), 0x08, np.uint8); w[0, 0] = 0x78
        layers.append((x, w))
        x = np.tile(np.array([0x3F, 0xBF, 0x00, 0x40], np.uint8), (1, 8))
        w = np.tile(np.array([0xBF, 0x3F, 0x40, 0x00], np.uint8), (1, 8))
        layers.append((x, w))
        n, ng = dump_acc_witness(path, layers)
        print(f"wrote {n} group-ordered rows ({ng} groups) to {path}")
        assert nfail == 0, "REFUSING to bless vectors: Triton mismatch above"
    if '--dumplayers' in sys.argv:
        assert nfail == 0, "REFUSING to bless layers: Triton mismatch above"
        dump_layers(sys.argv[sys.argv.index('--dumplayers') + 1])
