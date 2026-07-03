#!/usr/bin/env python3
"""transformer_ref.py -- CANONICAL bit-exact reference of one full llama-style
transformer-layer fp8 forward pass, on a tiny config:

    d_model=64, n_heads=2, head_dim=32, d_ff=128, seq=4     (rope theta 10000,
    rms_norm_eps = 1e-6, causal attention)

All 7 matmuls are the PROVEN Hawkeye fp8 atom (hawkeye_ref.hawkeye_ref, which is
bitwise-identical to the Triton hawkeye_fp8_sum kernel).  Everything else --
RMSNorm, activation quantization, RoPE, softmax, SwiGLU, residual adds -- is
defined HERE, exactly, as integer arithmetic on bf16 bit patterns:

  * scalar bf16 semantics: round-to-nearest-even on the EXACT value, subnormal
    inputs/outputs flushed to (signed) zero, overflow saturates to the max
    finite bf16, exponent-field 255 treated as an ORDINARY binade (no inf/nan
    special cases -- same policy as the Hawkeye kernel's fp8 NaN handling).
  * row reductions (sum of squares, softmax denominator): "block-float" exact
    sums -- per-element truncating alignment to the row's max binade (with a
    phantom floor binade EMIN), 16-bit lanes, exact integer sum.  This is the
    same max/shift/truncate vocabulary the Hawkeye matmul proof already uses.
  * 1-in/1-out nonlinearities (exp, silu, rsqrt, recip, e4m3 quantization,
    bf16 mantissa product): finite lookup tables generated here and PINNED as
    artifacts (the ZKP absorbs their hashes; canonical = these exact bytes).
    rsqrt/recip/mul tables are computed with EXACT integer arithmetic; the
    exp/silu tables use float64 libm then exact RNE (documented caveat: the
    float64 kernel is the canonical definition via the pinned artifact).

HONESTY: "bit-exact" for the nonlinear ops means bitwise-equal to THIS
canonical spec, exactly as the matmul is bitwise-equal to the Hawkeye kernel.
Matching a specific vendor kernel (torch's bf16 RMSNorm/softmax) is a separate
FAITHFULNESS question: torch accumulates in fp32 with libm rsqrt/exp, so
outputs differ from this spec by ~1 ulp on a small fraction of elements.  The
battery below MEASURES that gap instead of hiding it.

Run:  python3 transformer_ref.py                 # full validation battery
      python3 transformer_ref.py --dump-tables p3_rmsnorm_tables.bin
      python3 transformer_ref.py --dump-goldens p3_rmsnorm_golden.bin
      python3 transformer_ref.py --dump-layer   transformer_layer.bin
"""
import sys
import math
import numpy as np
from hawkeye_ref import hawkeye_ref

# ============================ scalar bf16 toolbox ============================
# pattern <-> (sign s, exact magnitude m * 2^t); m == 0 encodes (signed) zero.
# Normal bf16: m in [128, 256), t = e - 134  (value = (128+mb)/128 * 2^(e-127)).

BF_MAXFIN = 0x7F7F      # 254<<7 | 127: canonical saturation value


def bf_dec(p):
    """pattern -> (s, e, mb) bit fields."""
    return (p >> 15) & 1, (p >> 7) & 255, p & 127


def bf_val(p):
    """pattern -> (s, m, t) exact value (-1)^s * m * 2^t; m=0 for zero AND
    subnormal patterns (canonical FTZ).  e=255 decodes as an ordinary binade."""
    s, e, mb = bf_dec(p)
    if e == 0:
        return s, 0, 0
    return s, 128 + mb, e - 134


def rne_bf16(s, m, t):
    """exact (-1)^s * m * 2^t (m > 0 int) -> canonical bf16 pattern.
    RNE; flush to signed zero if the rounded exponent field would be <= 0;
    saturate to +-BF_MAXFIN if it would be >= 255."""
    assert m > 0
    w = m.bit_length()
    eb = t + w - 1 + 127                    # biased exponent of the leading bit
    sh = w - 8
    if sh > 0:
        q = m >> sh
        r = m & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if r > half or (r == half and (q & 1)):
            q += 1
        if q == 256:
            q = 128
            eb += 1
    else:
        q = m << (-sh)
    if eb >= 255:
        return (s << 15) | BF_MAXFIN
    if eb <= 0:
        return s << 15                      # flush (signed zero)
    return (s << 15) | (eb << 7) | (q - 128)


def bf_canon(p):
    """FTZ-canonicalize a pattern (subnormals -> signed zero)."""
    s, m, t = bf_val(p)
    return (s << 15) if m == 0 else p


def bf_neg(p):
    return p ^ 0x8000


def bf_mul(p1, p2):
    s1, m1, t1 = bf_val(p1)
    s2, m2, t2 = bf_val(p2)
    s = s1 ^ s2
    if m1 == 0 or m2 == 0:
        return s << 15
    return rne_bf16(s, m1 * m2, t1 + t2)


def bf_add(p1, p2):
    s1, m1, t1 = bf_val(p1)
    s2, m2, t2 = bf_val(p2)
    if m1 == 0 and m2 == 0:
        return (s1 & s2) << 15              # IEEE: +0 unless both are -0
    if m1 == 0:
        return bf_canon(p2)
    if m2 == 0:
        return bf_canon(p1)
    t = min(t1, t2)
    M = (-m1 if s1 else m1) * (1 << (t1 - t)) + (-m2 if s2 else m2) * (1 << (t2 - t))
    if M == 0:
        return 0                            # exact cancellation -> +0
    return rne_bf16(1 if M < 0 else 0, abs(M), t)


def bf_from_f64(x):
    """float64 -> canonical bf16 pattern (exact RNE via integer mantissa).
    Only used to build pinned tables and for faithfulness comparisons."""
    if x == 0.0:
        return 0x8000 if math.copysign(1.0, x) < 0 else 0
    if math.isinf(x) or math.isnan(x):
        return BF_MAXFIN if (math.isnan(x) or x > 0) else (0x8000 | BF_MAXFIN)
    s = 1 if x < 0 else 0
    fm, fe = math.frexp(abs(x))             # abs(x) = fm * 2^fe, fm in [0.5,1)
    m = int(fm * (1 << 53))                 # exact: float64 mantissa
    return rne_bf16(s, m, fe - 53)


def bf_to_f64(p):
    s, m, t = bf_val(p)
    return (-1.0 if s else 1.0) * m * (2.0 ** t)


# =========================== canonical fixed tables ==========================
# All tables are canonical ARTIFACTS: the ZKP pins their hashes; these
# generators are the normative construction.

RMS_T = 16          # top-bits window for the rsqrt/recip normalizations
EPS_BITS = np.float32(1e-6).view(np.uint32).item()   # rms_norm_eps as fp32


def eps_decompose():
    """fp32 eps -> exact (m, t) with eps = m * 2^t, m 24-bit."""
    e = (EPS_BITS >> 23) & 255
    mb = EPS_BITS & 0x7FFFFF
    assert 0 < e < 255
    return (1 << 23) | mb, e - 127 - 23


def epsa_table(ld):
    """EPSA[E] = floor(eps * 2^(268 + ld - E)) for E in [EMIN, 512); 0 below.
    EMIN = smallest E with EPSA(E) < 2^18 (so S' = S + EPSA stays < 2^23 for
    d <= 2^6 and eps keeps >= 17 bits of relative precision at the floor)."""
    em, et = eps_decompose()
    def epsa(E):
        sh = et + 268 + ld - E
        return em << sh if sh >= 0 else em >> (-sh)
    EMIN = 0
    while epsa(EMIN) >= (1 << 18):
        EMIN += 1
    assert (1 << 17) <= epsa(EMIN) < (1 << 18)
    tab = np.zeros(512, np.int64)
    for E in range(EMIN, 512):
        tab[E] = epsa(E)
    return EMIN, tab


def rsq_table():
    """RSQ[j], j = (pp<<15) | (u16 - 2^15): canonical bf16 of 1/sqrt(u16*2^pp),
    u16 in [2^15, 2^16), pp in {0,1}.  Output stored as (mr, hb): mantissa bits
    mr in [0,128) and binade bit hb (biased exp of the bf16 = 118 + hb).
    EXACT integer RNE: k <= 2^C/sqrt(v)  <=>  k^2 * v <= 2^2C."""
    mr = np.zeros(65536, np.uint8)
    hb = np.zeros(65536, np.uint8)
    for j in range(65536):
        u = 32768 + (j & 32767)
        pp = j >> 15
        v = u << pp
        # binade: value 1/sqrt(v) >= 2^-8  <=>  v <= 2^16
        if v <= (1 << 16):
            C = 15  # value*2^15 in [128,256): mantissa scale for binade -8
            h = 1
        else:
            C = 16  # binade -9
            h = 0
        q = math.isqrt((1 << (2 * C)) // v)
        while (q + 1) * (q + 1) * v <= (1 << (2 * C)):
            q += 1
        # q = floor(2^C/sqrt(v)); RNE: round up iff (q+.5)^2 < 2^2C / v
        r2 = (2 * q + 1) * (2 * q + 1) * v
        if r2 < (1 << (2 * C + 2)) or (r2 == (1 << (2 * C + 2)) and (q & 1)):
            q += 1
        if q == 256:
            q, h = 128, h + 1
        assert 128 <= q < 256 and h in (0, 1, 2)
        if h == 2:      # can only happen from the 256 bump at the top binade
            raise AssertionError("rsq binade overflow")
        mr[j] = q - 128
        hb[j] = h
    return mr, hb


def rcp_table():
    """RCP[u16 - 2^15]: canonical bf16 of 1/u16, u16 in [2^15, 2^16).
    Stored as (mr, hb): biased exp of the bf16 = 111 + hb (binade -16 or -15).
    Exact integer RNE at scale 2^23 (value*2^23 in (2^7, 2^8])."""
    mr = np.zeros(32768, np.uint8)
    hb = np.zeros(32768, np.uint8)
    for i in range(32768):
        u = 32768 + i
        q, rem = divmod(1 << 23, u)
        if 2 * rem > u or (2 * rem == u and (q & 1)):
            q += 1
        h = 0
        if q == 256:
            q, h = 128, 1
        if u == 32768:                       # exact 2^-15
            q, h = 128, 1
        assert 128 <= q < 256
        mr[i] = q - 128
        hb[i] = h
    return mr, hb


def mul7_table():
    """MUL7[j], j = (ma-128)<<7 | (mb-128): RNE bf16 mantissa product.
    ma, mb in [128,256); product value (ma/128)*(mb/128) in [1,4).
    Stored as (mo in [128,256) as mo-128, einc in {0,1,2})."""
    mo = np.zeros(16384, np.uint8)
    einc = np.zeros(16384, np.uint8)
    for j in range(16384):
        ma = 128 + (j >> 7)
        mb = 128 + (j & 127)
        mm = ma * mb                          # in [2^14, 2^16)
        c = 1 if mm >= (1 << 15) else 0
        sh = 7 + c
        q = mm >> sh
        r = mm & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if r > half or (r == half and (q & 1)):
            q += 1
        e = c
        if q == 256:
            q, e = 128, c + 1
        mo[j] = q - 128
        einc[j] = e
    return mo, einc


def exp_table():
    """EXP[p] for all 2^16 bf16 patterns p: canonical bf16 of e^value(p).
    float64 libm exp + exact RNE; pinned artifact (see module docstring)."""
    t = np.zeros(65536, np.uint16)
    for p in range(65536):
        s, m, te = bf_val(p)
        if m == 0:
            t[p] = 0x3F80                    # e^0 = 1
            continue
        lg = (math.log2(m) + te)             # log2|x|
        if lg > 7.1:                         # |x| > ~137: e^x over/underflows
            t[p] = 0 if s else BF_MAXFIN
            continue
        v = math.exp((-1.0 if s else 1.0) * m * (2.0 ** te))
        t[p] = bf_from_f64(v)
    return t


def silu_table():
    """SILU[p]: canonical bf16 of value/(1+e^-value).  Same artifact policy."""
    t = np.zeros(65536, np.uint16)
    for p in range(65536):
        s, m, te = bf_val(p)
        if m == 0:
            t[p] = s << 15                   # silu(+-0) = +-0
            continue
        x = (-1.0 if s else 1.0) * m * (2.0 ** te)
        if x > 30:
            v = x                            # sigmoid == 1 to way below 1 ulp
        elif x < -85:
            v = x * math.exp(x)              # == x*e^x to 1e-37 rel; no overflow
        else:
            v = x / (1.0 + math.exp(-x))
        t[p] = bf_from_f64(v)
    return t


QE4M3_DEXP = 32     # dexp rows in the quantization table (>= 20 rounds to 0)


def q_e4m3_mag_table():
    """QE4M3[j], j = dexp<<7 | mb (dexp in [0, QE4M3_DEXP)): magnitude e4m3fn code of
    (128+mb)/128 * 2^(8-dexp)  -- i.e. a bf16 magnitude whose biased exponent
    sits dexp below the row max, divided by the row scale 2^(emax_unb-8).
    RNE to e4m3fn with saturation at 448 and subnormal codes supported."""
    tab = np.zeros(QE4M3_DEXP * 128, np.uint8)
    for j in range(QE4M3_DEXP * 128):
        dexp = j >> 7
        mb = j & 127
        m = 128 + mb                          # value = m * 2^(1-dexp) (m/128*2^(8-dexp))
        t = 1 - dexp
        # e4m3fn magnitude grid: normal m4*2^(e4-7-3), m4 in [8,16), e4 in [1,15]
        #                        subnormal m4*2^(-9), m4 in [0,8)
        # find RNE among representable magnitudes of value m*2^t
        best = None
        for code in range(0x80):
            e4 = code >> 3
            m4 = code & 7
            if e4 == 0:
                vm, vt = m4, -9
            else:
                vm, vt = m4 + 8, e4 - 10
            # compare |m*2^t - vm*2^vt| exactly at common scale 2^min(t,vt)
            sc = min(t, vt)
            d = abs(m * (1 << (t - sc)) - vm * (1 << (vt - sc)))
            key = (d, code & 1)                    # tie -> even mantissa field
            if best is None or key < best[0]:
                best = (key, code)
        tab[j] = best[1]
    return tab


# ============================ canonical row reductions =======================

def blockfloat_sum(lanes, EMIN):
    """lanes: list of (sq, esq) with sq an int in [0, 2^16), esq the binade
    (int >= 0); zero lanes have sq == 0 (esq ignored).  Returns (S, E):
    E = max(EMIN, max esq over sq != 0);  S = sum of sq >> min(E - esq, 16).
    This is the canonical exact block-float sum used by RMSNorm and softmax."""
    E = EMIN
    for sq, esq in lanes:
        if sq != 0 and esq > E:
            E = esq
    S = 0
    for sq, esq in lanes:
        if sq != 0:
            S += sq >> min(E - esq, 16)
    return S, E


def normalize_top(Sp):
    """Sp >= 1 -> (u16, wd): wd = bitlength(Sp), u16 = top RMS_T bits of Sp
    (truncated), u16 in [2^15, 2^16)."""
    wd = Sp.bit_length()
    if wd >= RMS_T:
        u16 = Sp >> (wd - RMS_T)
    else:
        u16 = Sp << (RMS_T - wd)
    assert (1 << 15) <= u16 < (1 << 16)
    return u16, wd


# ================================ RMSNorm ====================================

class RMSNormSpec:
    """Canonical RMSNorm for row length d = 2^ld.  Owns the EPSA table and the
    shared RSQ/MUL7 tables.  y_i = bf16( bf16(x_i * r) * g_i ), r = canonical
    bf16 rsqrt of (blockfloat mean of x_i^2) + eps  (HF Llama op order)."""

    def __init__(self, ld):
        self.ld = ld
        self.EMIN, self.EPSA = epsa_table(ld)
        self.RSQ_MR, self.RSQ_HB = RSQ_MR, RSQ_HB
        self.MUL_MO, self.MUL_EINC = MUL_MO, MUL_EINC

    def row_r(self, xrow):
        """canonical rsqrt bf16 pattern for one row of bf16 patterns."""
        lanes = []
        for p in xrow:
            s, m, t = bf_val(int(p))
            if m == 0:
                lanes.append((0, 0))
            else:
                e = (int(p) >> 7) & 255
                lanes.append((m * m, 2 * e))          # sq in [2^14, 2^16)
        S, E = blockfloat_sum(lanes, self.EMIN)
        Sp = S + int(self.EPSA[E])
        assert 1 <= Sp < (1 << 23), "S' out of the canonical 23-bit window"
        u16, wd = normalize_top(Sp)
        Xexp = E + wd - (284 + self.ld)               # value ~ u16 * 2^Xexp
        qp = Xexp >> 1                                # floor division
        pp = Xexp - 2 * qp
        j = (pp << 15) | (u16 - 32768)
        mr, hb = int(self.RSQ_MR[j]), int(self.RSQ_HB[j])
        teb = 118 + hb - qp                            # final biased exponent
        assert 1 <= teb <= 254, "rsqrt exponent outside the supported domain"
        return (teb << 7) | mr

    def bfmul(self, p1, p2):
        """table-driven canonical bf16 mul (identical to bf_mul on the
        supported domain; asserts the exponent stays normal)."""
        s1, e1, mb1 = bf_dec(p1)
        s2, e2, mb2 = bf_dec(p2)
        s = s1 ^ s2
        if e1 == 0 or e2 == 0:
            return s << 15
        j = (mb1 << 7) | mb2
        mo = 128 + int(self.MUL_MO[j])
        einc = int(self.MUL_EINC[j])
        eo = e1 + e2 - 127 + einc
        assert 1 <= eo <= 254, "bf16 mul exponent outside the supported domain"
        return (s << 15) | (eo << 7) | (mo - 128)

    def forward(self, X, G):
        """X: (B, d) uint16 patterns; G: (d,) uint16 patterns -> (B, d)."""
        B, d = X.shape
        assert d == (1 << self.ld)
        Y = np.zeros_like(X)
        for b in range(B):
            r = self.row_r(X[b])
            for i in range(d):
                tmid = self.bfmul(int(X[b, i]), r)
                Y[b, i] = self.bfmul(tmid, int(G[i]))
        return Y


# ======================== activation fp8 quantization =======================

def quant_rows_e4m3(X):
    """X: (B, d) bf16 patterns -> (codes (B,d) uint8, scales (B,) fp32 bits).
    Canonical: per-row power-of-two scale 2^(emax_unb - 8) with the fp32 scale
    exponent clamped to >= 1 (normal fp32, Hawkeye's supported domain); each
    element quantized by the QE4M3 table keyed on (eb_max - eb, mantissa)."""
    B, d = X.shape
    codes = np.zeros((B, d), np.uint8)
    scales = np.zeros(B, np.uint32)
    for b in range(B):
        ebs = [(int(p) >> 7) & 255 for p in X[b]]
        nz = [e for e in ebs if e != 0]
        emax = max(nz) if nz else 1
        se = max(emax - 8, 1)                # fp32 biased exponent of the scale
        scales[b] = np.uint32(se << 23)      # scale = 2^(se-127), sign/mant 0
        for i in range(d):
            p = int(X[b, i])
            s, e, mb = bf_dec(p)
            if e == 0:
                codes[b, i] = s << 7
                continue
            dexp = 8 - e + se                # element binade below the window top
            if dexp >= QE4M3_DEXP:
                codes[b, i] = s << 7          # underflows the e4m3 grid to 0
            else:
                codes[b, i] = (s << 7) | int(QE4M3[(dexp << 7) | mb])
    return codes, scales


# ================================ softmax ====================================

SM_EMIN = 100   # phantom floor binade for the denominator block-float sum
                # (exps are in (0, 1]: biased exponents <= 127; 100 => the sum
                # keeps >= 2^-27-relative lanes; a pure-canonical constant)


def softmax_rows(Srow_patterns, mask):
    """causal softmax over one row of bf16 score patterns.  mask[j] True =
    position participates.  Canonical: rowmax over the participating patterns
    (total order: sign/exponent/mantissa), x - max via exact bf_add, e^x via
    the pinned EXP table, denominator via blockfloat_sum on 16-bit lanes
    (mantissa << 8), reciprocal via the RCP table, p_j = bf16(e_j * rcp).
    Masked outputs are +0."""
    n = len(Srow_patterns)
    key = []
    for j in range(n):
        p = int(Srow_patterns[j])
        s, e, mb = bf_dec(p)
        key.append((32767 - (e << 7) - mb) if s else (32768 + (e << 7) + mb))
    mx = None
    for j in range(n):
        if mask[j] and (mx is None or key[j] > key[mx]):
            mx = j
    out = np.zeros(n, np.uint16)
    if mx is None:
        return out
    mxp = int(Srow_patterns[mx])
    exps = []
    for j in range(n):
        if not mask[j]:
            exps.append(0)
            continue
        dp = bf_add(int(Srow_patterns[j]), bf_neg(mxp))
        exps.append(int(EXP_TAB[dp]))
    lanes = []
    for ep in exps:
        s, m, t = bf_val(ep)
        if m == 0:
            lanes.append((0, 0))
        else:
            e = (ep >> 7) & 255
            lanes.append((m << 8, e + 8))     # 16-bit lane, binade e (+8 scale)
    S, E = blockfloat_sum(lanes, SM_EMIN)
    assert S >= 1
    u16, wd = normalize_top(S)
    # lane (m<<8, e+8): value m*2^(e-134) = (m<<8)*2^((e+8)-150), so the summed
    # denominator = S * 2^(E-150) ~ u16 * 2^(E-150+wd-16)
    dval_exp = E - 150 + (wd - 16)
    i = u16 - 32768
    mrc, hbc = int(RCP_MR[i]), int(RCP_HB[i])
    # RCP gives 1/u16 as bf16 with biased exp 111 + hb (1/u16 in (2^-16, 2^-15])
    reb = 111 + hbc - dval_exp
    assert 1 <= reb <= 254, "softmax recip exponent outside the supported domain"
    rcp = (reb << 7) | mrc
    for j in range(n):
        if mask[j]:
            out[j] = bf_mul(exps[j], rcp)     # canonical bf16 mul
    return out


# ================================= RoPE ======================================

def rope_tables(seq, dh, theta=10000.0):
    """PINNED cos/sin bf16 tables, (seq, dh/2)."""
    half = dh // 2
    cos = np.zeros((seq, half), np.uint16)
    sin = np.zeros((seq, half), np.uint16)
    for pos in range(seq):
        for j in range(half):
            ang = pos * (theta ** (-2.0 * j / dh))
            cos[pos, j] = bf_from_f64(math.cos(ang))
            sin[pos, j] = bf_from_f64(math.sin(ang))
    return cos, sin


def rope_apply(Q, cos, sin):
    """Q: (seq, dh) bf16 patterns -> rotated, llama rotate_half convention:
    q'_j     = q_j*c_j - q_{j+h}*s_j ;  q'_{j+h} = q_{j+h}*c_j + q_j*s_j
    each product bf16-RNE, each combine ONE bf16 add (canonical op order)."""
    seq, dh = Q.shape
    half = dh // 2
    out = np.zeros_like(Q)
    for p in range(seq):
        for j in range(half):
            a, b = int(Q[p, j]), int(Q[p, j + half])
            c, s = int(cos[p, j]), int(sin[p, j])
            out[p, j] = bf_add(bf_mul(a, c), bf_neg(bf_mul(b, s)))
            out[p, j + half] = bf_add(bf_mul(b, c), bf_mul(a, s))
    return out


# ================================ full layer =================================

class TinyLayer:
    """One llama-style layer at d_model=64, n_heads=2, head_dim=32, d_ff=128,
    seq=4.  Weights are fp8 codes + fp32 per-row scales (as the Hawkeye atom
    consumes them); 1/sqrt(head_dim) is folded into the Wq scales OFFLINE."""

    def __init__(self, rng):
        self.d, self.nh, self.dh, self.dff, self.seq = 64, 2, 32, 128, 4
        d, dff = self.d, self.dff

        def wmat(n, k, scale_sigma=0.3, extra=1.0):
            codes = rng.integers(0, 256, (n, k)).astype(np.uint8)
            sc = (np.exp(rng.normal(0, scale_sigma, n)) * 0.02 * extra).astype(np.float32)
            return codes, sc

        self.Wq = wmat(d, d, extra=1.0 / math.sqrt(self.dh))
        self.Wk = wmat(d, d)
        self.Wv = wmat(d, d)
        self.Wo = wmat(d, d)
        self.Wg = wmat(dff, d)
        self.Wu = wmat(dff, d)
        self.Wd = wmat(d, dff)
        gs = 1.0 + rng.normal(0, 0.1, d)
        gs2 = 1.0 + rng.normal(0, 0.1, d)
        self.g1 = np.array([bf_from_f64(v) for v in gs], np.uint16)
        self.g2 = np.array([bf_from_f64(v) for v in gs2], np.uint16)
        self.rms = RMSNormSpec(6)
        self.cos, self.sin = rope_tables(self.seq, self.dh)

    def matmul(self, Xp, W):
        """bf16 activations -> quantize -> Hawkeye fp8 matmul -> bf16."""
        codes, xsc = quant_rows_e4m3(Xp)
        wcodes, wsc = W
        y = hawkeye_ref(codes, xsc.view(np.float32), wcodes, wsc)
        return y, (codes, xsc)

    def forward(self, Xp, trace=None):
        """Xp: (seq, d) bf16 patterns -> (seq, d) bf16 patterns."""
        T = trace if trace is not None else {}
        seq, d, nh, dh = self.seq, self.d, self.nh, self.dh
        h1 = self.rms.forward(Xp, self.g1);              T['rms1'] = h1
        Q, qq = self.matmul(h1, self.Wq);                T['q'] = Q
        K, qk = self.matmul(h1, self.Wk);                T['k'] = K
        V, qv = self.matmul(h1, self.Wv);                T['v'] = V
        attn = np.zeros((seq, d), np.uint16)
        for h in range(nh):
            sl = slice(h * dh, (h + 1) * dh)
            Qh = rope_apply(Q[:, sl], self.cos, self.sin)
            Kh = rope_apply(K[:, sl], self.cos, self.sin)
            T[f'ropeq{h}'] = Qh; T[f'ropek{h}'] = Kh
            qc, qs = quant_rows_e4m3(Qh)
            kc, ks = quant_rows_e4m3(Kh)
            S = hawkeye_ref(qc, qs.view(np.float32), kc, ks.view(np.float32))
            T[f'scores{h}'] = S
            P = np.zeros((seq, seq), np.uint16)
            for i in range(seq):
                P[i] = softmax_rows(S[i], [j <= i for j in range(seq)])
            T[f'probs{h}'] = P
            pc, ps = quant_rows_e4m3(P)
            VhT = V[:, sl].T.copy()                       # (dh, seq)
            vc, vs = quant_rows_e4m3(VhT)
            Oh = hawkeye_ref(pc, ps.view(np.float32), vc, vs.view(np.float32))
            T[f'attnout{h}'] = Oh
            attn[:, sl] = Oh
        O, qo = self.matmul(attn, self.Wo);              T['oproj'] = O
        x2 = np.zeros_like(Xp)
        for i in range(seq):
            for j in range(d):
                x2[i, j] = bf_add(int(Xp[i, j]), int(O[i, j]))
        T['resid1'] = x2
        h2 = self.rms.forward(x2, self.g2);              T['rms2'] = h2
        Gt, _ = self.matmul(h2, self.Wg);                T['gate'] = Gt
        Up, _ = self.matmul(h2, self.Wu);                T['up'] = Up
        M = np.zeros_like(Gt)
        for i in range(seq):
            for j in range(self.dff):
                M[i, j] = bf_mul(int(SILU_TAB[int(Gt[i, j])]), int(Up[i, j]))
        T['swiglu'] = M
        D, _ = self.matmul(M, self.Wd);                  T['down'] = D
        out = np.zeros_like(Xp)
        for i in range(seq):
            for j in range(d):
                out[i, j] = bf_add(int(x2[i, j]), int(D[i, j]))
        T['out'] = out
        return out


# ========================== validation battery ==============================

def torch_bf16_pat(t):
    import torch
    return t.view(torch.uint16).numpy()


def battery():
    import torch
    nfail = 0

    def ck(name, cond, extra=""):
        nonlocal nfail
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{extra}")
        if not cond:
            nfail += 1

    rng = np.random.default_rng(20260703)

    # -- scalar ops vs torch bf16 (normal range; canonical == IEEE there) --
    pats = rng.integers(0, 65536, 20000).astype(np.uint16)
    def normal(p):
        e = (p >> 7) & 255
        return 64 < e < 189          # keep products/sums inside the normal
                                     # range (canonical saturates where IEEE
                                     # makes inf; that difference is by design)
    a = np.array([p for p in pats[:10000] if normal(p)], np.uint16)
    b = np.array([p for p in pats[10000:] if normal(p)], np.uint16)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    ta = torch.from_numpy(a.copy()).view(torch.bfloat16)
    tb = torch.from_numpy(b.copy()).view(torch.bfloat16)
    tm = torch_bf16_pat(ta * tb)
    ts = torch_bf16_pat(ta + tb)
    mm = np.array([bf_mul(int(x), int(y)) for x, y in zip(a, b)], np.uint16)
    ss = np.array([bf_add(int(x), int(y)) for x, y in zip(a, b)], np.uint16)
    ck(f"bf_mul == torch bf16 mul on {n} normal-range pairs", (mm == tm).all(),
       f"  mismatches={int((mm != tm).sum())}")
    ck(f"bf_add == torch bf16 add on {n} normal-range pairs", (ss == ts).all(),
       f"  mismatches={int((ss != ts).sum())}")

    # -- table-driven mul == scalar mul on the supported domain --
    spec = RMSNormSpec(6)
    okmul = True
    for x, y in zip(a[:2000], b[:2000]):
        try:
            got = spec.bfmul(int(x), int(y))
        except AssertionError:
            continue
        if got != bf_mul(int(x), int(y)):
            okmul = False
    ck("MUL7-table mul == exact scalar mul (2000 pairs)", okmul)

    # -- rsqrt table: exact-integer construction vs float64, <= 1 ulp --
    idx = rng.integers(0, 65536, 4000)
    worst = 0
    for j in idx:
        u = 32768 + (int(j) & 32767); pp = int(j) >> 15
        want = bf_from_f64(1.0 / math.sqrt(u << pp))
        got = ((118 + int(RSQ_HB[j])) << 7) | int(RSQ_MR[j])
        worst = max(worst, abs(got - want))
    ck("RSQ table matches float64 rsqrt bitwise on 4000 samples", worst == 0,
       f"  (max pattern delta {worst})")

    # -- RCP table --
    idx = rng.integers(0, 32768, 4000)
    worst = 0
    for i in idx:
        u = 32768 + int(i)
        want = bf_from_f64(1.0 / u)
        got = ((111 + int(RCP_HB[i])) << 7) | int(RCP_MR[i])
        worst = max(worst, abs(got - want))
    ck("RCP table matches float64 recip bitwise on 4000 samples", worst == 0,
       f"  (max pattern delta {worst})")

    # -- RMSNorm: determinism + faithfulness gap vs torch fp32-accum RMSNorm --
    X = np.array([[bf_from_f64(v) for v in row]
                  for row in rng.normal(0, 1.5, (8, 64))], np.uint16)
    G = np.array([bf_from_f64(v) for v in 1 + rng.normal(0, 0.1, 64)], np.uint16)
    Y1 = spec.forward(X, G); Y2 = spec.forward(X.copy(), G.copy())
    ck("canonical RMSNorm deterministic (two runs byte-equal)", (Y1 == Y2).all())
    xt = torch.from_numpy(X.copy()).view(torch.bfloat16).float()
    gt = torch.from_numpy(G.copy()).view(torch.bfloat16).float()
    var = xt.pow(2).mean(-1, keepdim=True)
    yt = (xt * torch.rsqrt(var + 1e-6) * gt).bfloat16()
    ytp = torch_bf16_pat(yt)
    du = np.abs(Y1.astype(np.int64) - ytp.astype(np.int64))
    print(f"      faithfulness vs torch fp32-accum RMSNorm: "
          f"{int((du != 0).sum())}/{du.size} elements differ, max {int(du.max())} ulp "
          f"(EXPECTED: canonical block-float spec != torch kernel; documented)")
    ck("canonical RMSNorm within 1 ulp of torch RMSNorm", int(du.max()) <= 1)

    # -- RMSNorm edge rows: zero row, one-hot, huge/small exponents, -0s --
    edge = np.zeros((5, 64), np.uint16)
    edge[1, 7] = bf_from_f64(3.0)
    edge[2, :] = bf_from_f64(1e30)
    edge[3, :] = bf_from_f64(1e-30)
    edge[4, ::2] = 0x8000
    edge[4, 1::2] = bf_from_f64(-2.5)
    Ye = spec.forward(edge, G)
    ck("RMSNorm zero row -> all (signed) zeros", all((int(p) & 0x7FFF) == 0 for p in Ye[0]))
    ck("RMSNorm edge rows deterministic", (Ye == spec.forward(edge.copy(), G)).all())

    # -- quantization: dequantized codes vs torch fp8 cast --
    Xq = np.array([[bf_from_f64(v) for v in row]
                   for row in rng.normal(0, 2.0, (6, 64))], np.uint16)
    codes, scales = quant_rows_e4m3(Xq)
    ok = True
    for bidx in range(6):
        sc = scales[bidx:bidx+1].view(np.float32)[0]
        xr = torch.from_numpy(Xq[bidx:bidx+1].copy()).view(torch.bfloat16).float()
        tc = (xr / sc).to(torch.float8_e4m3fn).view(torch.uint8).numpy()[0]
        if not (tc == codes[bidx]).all():
            ok = False
            bad = np.argwhere(tc != codes[bidx])[:3]
            print(f"      row {bidx} quant mismatch at {bad.ravel()}")
    ck("pow2-scale e4m3 quantization == torch fp8 cast (6 rows)", ok)

    # -- softmax: determinism + faithfulness vs torch --
    Srow = np.array([bf_from_f64(v) for v in rng.normal(0, 3, 16)], np.uint16)
    m = [True] * 16
    p1 = softmax_rows(Srow, m); p2 = softmax_rows(Srow.copy(), m)
    ck("canonical softmax deterministic", (p1 == p2).all())
    st = torch.from_numpy(Srow.copy()).view(torch.bfloat16).float()
    pt = torch_bf16_pat(torch.softmax(st, -1).bfloat16())
    dq = np.abs(p1.astype(np.int64) - pt.astype(np.int64))
    print(f"      faithfulness vs torch softmax: {int((dq != 0).sum())}/16 differ, "
          f"max {int(dq.max())} ulp (documented canonical gap)")
    sums = sum(bf_to_f64(int(q)) for q in p1)
    ck("canonical softmax row-sum ~ 1", abs(sums - 1) < 0.02, f"  (sum={sums:.5f})")

    # -- full tiny layer: determinism + trace self-consistency --
    layer = TinyLayer(np.random.default_rng(20260703))
    Xin = np.array([[bf_from_f64(v) for v in row]
                    for row in rng.normal(0, 1.0, (4, 64))], np.uint16)
    tr1, tr2 = {}, {}
    o1 = layer.forward(Xin, tr1)
    o2 = layer.forward(Xin.copy(), tr2)
    ck("FULL TINY LAYER deterministic (outputs byte-equal)", (o1 == o2).all())
    ck("full-layer trace byte-equal across runs",
       all((tr1[k] == tr2[k]).all() for k in tr1))
    ck("layer output finite patterns (no saturation hit)",
       all(int(p) & 0x7FFF != BF_MAXFIN for p in o1.ravel()))
    nsteps = len(tr1)
    print(f"      layer trace: {nsteps} recorded op outputs "
          f"({', '.join(sorted(tr1.keys()))})")

    # -- matmul inputs stayed in the Hawkeye supported scale domain (scales are
    #    normal fp32 by construction: pow2 with exponent >= 1) --
    ck("all activation quant scales normal fp32",
       all((int(s) >> 23) & 255 >= 1 for s in
           np.concatenate([quant_rows_e4m3(tr1['rms1'])[1],
                           quant_rows_e4m3(tr1['rms2'])[1]])))
    return nfail


# ================================ dumps ======================================

def dump_tables(path, ld=6):
    EMIN, EPSA = epsa_table(ld)
    with open(path, 'wb') as f:
        np.array([0x524D5354, 1, ld, EMIN, RMS_T, EPS_BITS, SM_EMIN],
                 np.int64).tofile(f)                       # 'RMST', version
        MUL_MO.astype(np.uint8).tofile(f)
        MUL_EINC.astype(np.uint8).tofile(f)
        RSQ_MR.astype(np.uint8).tofile(f)
        RSQ_HB.astype(np.uint8).tofile(f)
        RCP_MR.astype(np.uint8).tofile(f)
        RCP_HB.astype(np.uint8).tofile(f)
        EPSA.astype(np.int64).tofile(f)
        EXP_TAB.astype(np.uint16).tofile(f)
        SILU_TAB.astype(np.uint16).tofile(f)
        QE4M3.astype(np.uint8).tofile(f)
    print(f"wrote canonical tables (ld={ld}, EMIN={EMIN}) to {path}")


def dump_goldens(path, ld=6):
    """RMSNorm golden cases for the C++ gadget test (in-domain rows only)."""
    rng = np.random.default_rng(20260703)
    spec = RMSNormSpec(ld)
    d = 1 << ld
    cases = []
    def add(name, X, G):
        Y = spec.forward(X, G)
        cases.append((name, X, G, Y))
    G = np.array([bf_from_f64(v) for v in 1 + rng.normal(0, 0.1, d)], np.uint16)
    X = np.array([[bf_from_f64(v) for v in row]
                  for row in rng.normal(0, 1.5, (4, d))], np.uint16)
    add("random 4x64", X, G)
    X = np.array([[bf_from_f64(v) for v in row]
                  for row in rng.normal(0, 1.0, (3, d)) *
                  np.exp(rng.normal(0, 4, (3, 1)))], np.uint16)
    add("3 rows, wildly different row norms", X, G)
    E = np.zeros((4, d), np.uint16)
    E[1, 5] = bf_from_f64(7.0)
    E[2, :] = bf_from_f64(1e-15)
    E[3, ::3] = 0x8000                       # -0 patterns
    E[3, 1::3] = bf_from_f64(1.25)
    add("edge: zero row / one-hot / tiny / signed zeros", E, G)
    Gz = G.copy(); Gz[::5] = 0; Gz[3] = 0x8000
    X = np.array([[bf_from_f64(v) for v in row]
                  for row in rng.normal(0, 1.0, (2, d))], np.uint16)
    add("gains with +-0 entries", X, Gz)
    Xs = np.array([[bf_from_f64(v) for v in row]
                   for row in rng.normal(0, 1.0, (2, d)) * 1e-33], np.uint16)
    add("eps-dominated rows (E at the EMIN floor)", Xs, G)
    with open(path, 'wb') as f:
        np.array([0x524D5347, len(cases), ld], np.int64).tofile(f)  # 'RMSG'
        for (name, X, G_, Y) in cases:
            np.array([X.shape[0]], np.int64).tofile(f)
            X.tofile(f); G_.tofile(f); Y.tofile(f)
            print(f"  golden '{name}': B={X.shape[0]} d={d}")
    print(f"wrote {len(cases)} RMSNorm goldens to {path}")


def dump_goldens_swiglu(path):
    """SwiGLU golden cases: flat pow2-length vectors (gate, up, out) with
    out_j = bf16( SILU[gate_j] * up_j ), the canonical layer op."""
    rng = np.random.default_rng(20260703)
    cases = []
    def add(name, Gt, Up):
        M = np.array([bf_mul(int(SILU_TAB[int(g)]), int(u))
                      for g, u in zip(Gt, Up)], np.uint16)
        cases.append((name, Gt, Up, M))
    def rnd(n, sg=2.0, su=2.0):
        Gt = np.array([bf_from_f64(v) for v in rng.normal(0, sg, n)], np.uint16)
        Up = np.array([bf_from_f64(v) for v in rng.normal(0, su, n)], np.uint16)
        return Gt, Up
    add("random 512", *rnd(512))
    Gt, Up = rnd(256, sg=8.0)                 # deep negative gates -> tiny silu
    add("wide gates 256", Gt, Up)
    Gt, Up = rnd(128)
    Gt[::7] = 0; Gt[3] = 0x8000               # +-0 gates
    Up[::5] = 0; Up[8] = 0x8000               # +-0 ups
    Gt[1] = 0x0040; Up[2] = 0x8020            # subnormal inputs (canonical FTZ)
    add("zeros + subnormals 128", Gt, Up)
    Gt, Up = rnd(128, sg=1.0, su=1.0)
    Gt[:32] = np.array([bf_from_f64(v) for v in rng.normal(0, 1, 32) * 1e-20],
                       np.uint16)             # silu(x) ~ x/2 for tiny x
    add("tiny gates 128", Gt, Up)
    with open(path, 'wb') as f:
        np.array([0x53574747, len(cases)], np.int64).tofile(f)      # 'SWGG'
        for (name, Gt, Up, M) in cases:
            np.array([len(Gt)], np.int64).tofile(f)
            Gt.tofile(f); Up.tofile(f); M.tofile(f)
            print(f"  golden '{name}': n={len(Gt)}")
    print(f"wrote {len(cases)} SwiGLU goldens to {path}")


def dump_layer(path):
    rng = np.random.default_rng(20260703)
    layer = TinyLayer(np.random.default_rng(20260703))
    Xin = np.array([[bf_from_f64(v) for v in row]
                    for row in rng.normal(0, 1.0, (4, 64))], np.uint16)
    tr = {}
    out = layer.forward(Xin, tr)
    with open(path, 'wb') as f:
        np.array([0x54524C59, len(tr) + 1], np.int64).tofile(f)     # 'TRLY'
        def put(name, arr):
            nb = name.encode()
            np.array([len(nb)], np.int64).tofile(f)
            f.write(nb)
            np.array(list(arr.shape) + [0] * (2 - arr.ndim), np.int64).tofile(f)
            arr.astype(np.uint16).tofile(f)
        put('input', Xin)
        for k in sorted(tr):
            put(k, tr[k])
    print(f"wrote full tiny-layer trace ({len(tr)} ops) to {path}")


# ============================ module-level tables ============================
print("building canonical tables (exact-integer rsqrt/recip/mul + pinned exp/silu)...",
      flush=True)
MUL_MO, MUL_EINC = mul7_table()
RSQ_MR, RSQ_HB = rsq_table()
RCP_MR, RCP_HB = rcp_table()
EXP_TAB = exp_table()
SILU_TAB = silu_table()
QE4M3 = q_e4m3_mag_table()

if __name__ == '__main__':
    if '--dump-tables' in sys.argv:
        dump_tables(sys.argv[sys.argv.index('--dump-tables') + 1])
    elif '--dump-goldens-swiglu' in sys.argv:
        dump_goldens_swiglu(sys.argv[sys.argv.index('--dump-goldens-swiglu') + 1])
    elif '--dump-goldens' in sys.argv:
        dump_goldens(sys.argv[sys.argv.index('--dump-goldens') + 1])
    elif '--dump-layer' in sys.argv:
        dump_layer(sys.argv[sys.argv.index('--dump-layer') + 1])
    else:
        print("=== canonical transformer-layer reference: validation battery ===")
        nf = battery()
        print(f"battery: {'ALL PASS' if nf == 0 else f'{nf} FAILURES'}")
        sys.exit(1 if nf else 0)
