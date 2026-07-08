// bf16 ADD gadget: sound (non-ZK) prover+verifier that a committed bf16 output
// OUT equals the CANONICAL bf16 addition (transformer_ref.py bf_add) of two
// committed bf16 inputs X1, X2, elementwise and bitwise.  This is the shared
// primitive behind the residual adds (this file's standalone gadget IS the
// residual op), the RoPE combines and the softmax subtract.
//
// Per element, the canonical semantics (FTZ, RNE, exact integer arithmetic):
//   decompose both operands; z = (eb==0)
//   both zero -> (s1 AND s2)<<15;  one zero -> the other pattern
//   hi/lo by magnitude (swap bit SG; ordering enforced by D = EH-EL and, at
//     D=0, DM = MH-ML being range-bound nonnegative)
//   FAR split at D >= 10: the aligned low addend is then strictly below the
//     RNE half-ulp of the high one for BOTH signs, so OUT = hi bitwise
//     (exhaustively re-derived; the boundary cases D=9..11 are in the goldens)
//   near (D <= 9): EXACT A = MH*2^D +- ML < 2^17  [move 3: (KD,PWD) table]
//     exact cancellation A=0 -> +0 (CZ selector, AV*AI = 1-CZ)
//     pow2 sandwich PLO <= AN < PHI pins w = bitlen(A)   [move 4: WDA table]
//     mantissa (QM+128)*PDN + RR = AN*PUP                [move 3: RM17]
//     RNE: RR = RB*PDH + RT (RMH table also forces RB=0 when PDH=0),
//          RUP = RB & (RT!=0 | Q odd)                    [section-2-P6 bits]
//     carry Q+RUP=256 -> mantissa 128, exponent +1;  EO = EL + W - 8 + C
//
// Supported domain (proof REJECTS otherwise -- sound, not complete): the
// result exponent stays in [1,254] -- on near rows EO = EL+W-8+C, on far rows
// EO := EH (the canonical rne SATURATES eb=255 results to 0x7F7F even when the
// value is exactly representable, so a binade-255 hi operand in a far add is
// NOT "out = hi"; v1 excludes it).  The canonical reference is total (flushes
// underflow to signed zero / saturates overflow); rows that flush or saturate
// are outside this gadget's v1 domain (same policy as the RMSNorm/SwiGLU
// product-exponent rule).  Zeros, signed zeros, subnormal inputs, exact
// cancellation and binade-crossing subtracts are all in-domain.
//
// The block core (ba_fill / ba_constraints / ba_lu_defs, all column-base
// relative) is designed for REUSE: RoPE and the softmax subtract instantiate
// the same 55 columns + 16 lookups inside their own domains.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <deque>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_batchopen.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "fs_transcript.hpp"

namespace p3bfa {

using p3lu::Col; using p3lu::Table; using p3lu::make_table; using p3lu::commit_col_nc;
using p3lu::chal; using p3lu::chal_vec; using p3lu::ilog2;
using p3lu::LuColSpec; using p3lu::LC;
using p3hwl::Msg5; using p3hwl::CFn; using p3hwl::sc5_prove; using p3hwl::sc5_verify;
using p3hwl::claimc; using p3hwl::claimv; using p3hwl::beq; using p3hwl::enc;

// ---------------- block columns (base-relative) ----------------
enum {
    BA_S1 = 0, BA_E1, BA_M1, BA_Z1, BA_I1,          // operand 1 decomposition
    BA_S2, BA_E2, BA_M2, BA_Z2, BA_I2,              // operand 2 decomposition
    BA_SG, BA_EH, BA_EL, BA_MH, BA_ML, BA_SHS, BA_SLS, BA_OP,   // hi/lo mux
    BA_KD, BA_PWD, BA_DR, BA_DZ, BA_DI, BA_DM, BA_FAR,          // exponent diff
    BA_ZB, BA_NZ, BA_NNF, BA_NN,                    // selector products
    BA_AV, BA_CZ, BA_AI, BA_AN,                     // exact magnitude sum
    BA_W, BA_PLO, BA_PHI, BA_PDN, BA_PDH, BA_PUP,   // pow2 sandwich (WDA row)
    BA_U1H, BA_U1L, BA_U2H, BA_U2L,                 // sandwich range splits
    BA_QM, BA_RR, BA_RB, BA_RT, BA_RTI, BA_ZRT, BA_L0, BA_QH, BA_RUP, BA_C, BA_CI,
    BA_EO, NBA };

// ---------------- block lookups (base-relative) ----------------
enum { BT_R128 = 0, BT_R256, BT_R512, BT_D16, BT_WDA, BT_RM17, BT_RMH, BT_REXP, NBT };
enum { BLU_E1 = 0, BLU_E2, BLU_M1, BLU_M2, BLU_D16, BLU_DR, BLU_DM,
       BLU_WDA, BLU_U1H, BLU_U1L, BLU_U2H, BLU_U2L, BLU_RM, BLU_QM, BLU_RMH, BLU_EO,
       NBLU };

struct Tables { Table t[NBT]; };
static inline Tables build_tables() {
    Tables T;
    auto range = [](uint32_t n) { std::vector<gl_t> v(n);
        for (uint32_t j = 0; j < n; j++) v[j] = j; return make_table({v}); };
    T.t[BT_R128] = range(128); T.t[BT_R256] = range(256); T.t[BT_R512] = range(512);
    { std::vector<gl_t> k(16), p(16);                       // near shift 2^D, D<=9
      for (uint32_t j = 0; j < 16; j++) {
          uint32_t kd = j < 10 ? j : 0;
          k[j] = kd; p[j] = 1ULL << kd;
      }
      T.t[BT_D16] = make_table({k, p}); }
    { std::vector<gl_t> w(32), plo(32), phi(32), pdn(32), pdh(32), pup(32);
      for (uint32_t j = 0; j < 32; j++) {                    // w in [1,18]
          uint32_t wv = (j >= 1 && j <= 18) ? j : 1;
          w[j] = wv; plo[j] = 1ULL << (wv - 1); phi[j] = 1ULL << wv;
          pdn[j] = 1ULL << (wv > 8 ? wv - 8 : 0);
          pdh[j] = (1ULL << (wv > 8 ? wv - 8 : 0)) >> 1;
          pup[j] = 1ULL << (wv < 8 ? 8 - wv : 0);
      }
      T.t[BT_WDA] = make_table({w, plo, phi, pdn, pdh, pup}); }
    { std::vector<gl_t> p(131072, 1), r(131072, 0); uint32_t row = 0;  // r < pw <= 2^16
      for (uint32_t t = 0; t <= 16; t++)
          for (uint32_t x = 0; x < (1u << t); x++) { p[row] = 1ULL << t; r[row] = x; row++; }
      T.t[BT_RM17] = make_table({p, r}); }
    { std::vector<gl_t> h(2048), b(2048), x(2048);           // (pdh, rb, rt<pdh) + (0,0,0)
      h[0] = 0; b[0] = 0; x[0] = 0;
      uint32_t row = 1;
      for (uint32_t t = 0; t <= 9; t++)
          for (uint32_t rb = 0; rb <= 1; rb++)
              for (uint32_t rt = 0; rt < (1u << t); rt++) {
                  h[row] = 1ULL << t; b[row] = rb; x[row] = rt; row++;
              }
      for (; row < 2048; row++) { h[row] = 0; b[row] = 0; x[row] = 0; }
      T.t[BT_RMH] = make_table({h, b, x}); }
    { std::vector<gl_t> v(256);
      for (uint32_t j = 0; j < 256; j++) v[j] = (j >= 1 && j <= 254) ? j : 1;
      T.t[BT_REXP] = make_table({v}); }
    return T;
}
static inline uint32_t rm17_idx(uint64_t pw, int64_t r) { return (uint32_t)((pw - 1) + (uint64_t)r); }
static inline uint32_t rmh_idx(int64_t pdh, int64_t rb, int64_t rt) {
    if (pdh == 0) return 0;
    return (uint32_t)(2 * pdh - 1 + rb * pdh + rt);
}

// lookup descriptors: which table, which block columns
struct BaLuDef { int tab; std::vector<int> cols; };
static inline std::vector<BaLuDef> ba_lu_defs() {
    std::vector<BaLuDef> L(NBLU);
    L[BLU_E1]  = {BT_R256, {BA_E1}};
    L[BLU_E2]  = {BT_R256, {BA_E2}};
    L[BLU_M1]  = {BT_R128, {BA_M1}};
    L[BLU_M2]  = {BT_R128, {BA_M2}};
    L[BLU_D16] = {BT_D16, {BA_KD, BA_PWD}};
    L[BLU_DR]  = {BT_R256, {BA_DR}};
    L[BLU_DM]  = {BT_R128, {BA_DM}};
    L[BLU_WDA] = {BT_WDA, {BA_W, BA_PLO, BA_PHI, BA_PDN, BA_PDH, BA_PUP}};
    L[BLU_U1H] = {BT_R512, {BA_U1H}};
    L[BLU_U1L] = {BT_R512, {BA_U1L}};
    L[BLU_U2H] = {BT_R512, {BA_U2H}};
    L[BLU_U2L] = {BT_R512, {BA_U2L}};
    L[BLU_RM]  = {BT_RM17, {BA_PDN, BA_RR}};
    L[BLU_QM]  = {BT_R128, {BA_QM}};
    L[BLU_RMH] = {BT_RMH, {BA_PDH, BA_RB, BA_RT}};
    L[BLU_EO]  = {BT_REXP, {BA_EO}};
    return L;
}

// ---------------- witness fill (canonical replay of bf_add) ----------------
// Selftest-only semantic forgeries; ONE forgery, honest downstream replay.
enum { BAT_NONE = 0,
       BAT_RUP,     // RNE round bit flipped (out changes by 1 ulp)
       BAT_CZ,      // false cancellation claim (out forced to +0)
       BAT_FAR,     // near row claimed FAR (DR = D-10 goes negative)
       BAT_SWAP,    // hi/lo swapped on a D=0, MH!=ML row (DM goes negative)
       BAT_EO };    // result exponent forged +1 (out one binade up)
struct BaTamper { int mode = BAT_NONE; };

static inline gl_t inv_or0g(gl_t x) { return x ? gl_inv(x) : 0; }
static inline int bitlen64(uint64_t x) { int w = 0; while (x) { w++; x >>= 1; } return w; }

struct BaVals {
    gl_t v[NBA];
    uint32_t lu[NBLU];
    uint16_t out;
};
// throws std::runtime_error on gadget-v1 domain violations (flush/saturate)
static inline BaVals ba_fill(uint16_t p1, uint16_t p2, const BaTamper* tm = nullptr) {
    BaVals o{};
    int64_t s1 = (p1 >> 15) & 1, e1 = (p1 >> 7) & 255, m1 = p1 & 127;
    int64_t s2 = (p2 >> 15) & 1, e2 = (p2 >> 7) & 255, m2 = p2 & 127;
    int64_t z1 = e1 == 0, z2 = e2 == 0;
    int64_t sg = (e2 > e1 || (e2 == e1 && m2 > m1)) ? 1 : 0;
    if (tm && tm->mode == BAT_SWAP) sg ^= 1;
    int64_t eh = sg ? e2 : e1, el = sg ? e1 : e2;
    int64_t mh = 128 + (sg ? m2 : m1), ml = 128 + (sg ? m1 : m2);
    int64_t shs = sg ? s2 : s1, sls = sg ? s1 : s2;
    int64_t op = shs ^ sls;
    int64_t d = eh - el;
    int64_t far = (tm && tm->mode == BAT_FAR) ? 1 : (d >= 10 ? 1 : 0);
    int64_t kd = far ? 0 : d;
    int64_t pwd = kd >= 0 && kd <= 9 ? (int64_t)1 << kd : 1;
    int64_t dr = far ? d - 10 : 0;
    int64_t dz = d == 0, dm = 0;
    int64_t nz = (!z1 && !z2) ? 1 : 0;
    if (nz && dz) dm = mh - ml;
    int64_t zb = (z1 && z2) ? 1 : 0;
    int64_t nnf = (nz && !far) ? 1 : 0;
    int64_t av = 128, cz = 0;
    if (nnf) {
        av = mh * pwd + (op ? -ml : ml);
        if (av == 0) cz = 1;
    }
    if (tm && tm->mode == BAT_CZ && nnf && av != 0) cz = 1;
    int64_t nn = (nnf && !cz) ? 1 : 0;
    int64_t an = (av > 0 && !cz) ? av : 128;
    int64_t w = bitlen64((uint64_t)an);
    if (w < 1 || w > 18) throw std::runtime_error("bfa: bitlen window");
    int64_t plo = (int64_t)1 << (w - 1), phi = (int64_t)1 << w;
    int64_t pdn = w > 8 ? (int64_t)1 << (w - 8) : 1;
    int64_t pdh = pdn >> 1;
    int64_t pup = w < 8 ? (int64_t)1 << (8 - w) : 1;
    int64_t u1 = an - plo, u2 = phi - 1 - an;
    int64_t q = w > 8 ? an >> (w - 8) : an << (8 - w);
    int64_t rr = an * pup - q * pdn;
    int64_t rb = pdh > 0 && rr >= pdh ? 1 : 0;
    int64_t rt = rr - rb * pdh;
    int64_t zrt = rt == 0, l0 = q & 1, qh = q >> 1;
    int64_t rup = (rb && (!zrt || l0)) ? 1 : 0;
    if (tm && tm->mode == BAT_RUP && nn) rup ^= 1;
    int64_t qr = q + rup;
    int64_t c = qr == 256;
    int64_t eo = 1;
    if (nn) {
        eo = el + w - 8 + c;
        if (eo < 1 || eo > 254) throw std::runtime_error("bfa: add exp domain");
        if (tm && tm->mode == BAT_EO) eo += 1;
    } else if (nz && far) {
        // far result = hi bitwise, but the canonical rne saturates eb=255:
        // bind EO := EH so the REXP range keeps far outputs in [1,254] too
        eo = eh;
        if (eo > 254) throw std::runtime_error("bfa: far exp domain");
    }
    // canonical output (honest downstream of any single forgery above)
    uint16_t out;
    if (zb) out = (uint16_t)((s1 & s2) << 15);
    else if (z1) out = p2;
    else if (z2) out = p1;
    else if (far) out = (uint16_t)((shs << 15) | (eh << 7) | (mh - 128));
    else if (cz) out = 0;
    else out = (uint16_t)((shs << 15) | (eo << 7) | (qr - 128 * c - 128));

    gl_t* v = o.v;
    v[BA_S1] = (gl_t)s1; v[BA_E1] = (gl_t)e1; v[BA_M1] = (gl_t)m1;
    v[BA_Z1] = (gl_t)z1; v[BA_I1] = inv_or0g((gl_t)e1);
    v[BA_S2] = (gl_t)s2; v[BA_E2] = (gl_t)e2; v[BA_M2] = (gl_t)m2;
    v[BA_Z2] = (gl_t)z2; v[BA_I2] = inv_or0g((gl_t)e2);
    v[BA_SG] = (gl_t)sg; v[BA_EH] = (gl_t)eh; v[BA_EL] = (gl_t)el;
    v[BA_MH] = (gl_t)mh; v[BA_ML] = (gl_t)ml;
    v[BA_SHS] = (gl_t)shs; v[BA_SLS] = (gl_t)sls; v[BA_OP] = (gl_t)op;
    v[BA_KD] = enc(kd); v[BA_PWD] = (gl_t)pwd; v[BA_DR] = enc(dr);
    v[BA_DZ] = (gl_t)dz; v[BA_DI] = dz ? 0 : gl_inv(enc(d));
    v[BA_DM] = enc(dm); v[BA_FAR] = (gl_t)far;
    v[BA_ZB] = (gl_t)zb; v[BA_NZ] = (gl_t)nz;
    v[BA_NNF] = (gl_t)nnf; v[BA_NN] = (gl_t)nn;
    v[BA_AV] = enc(av); v[BA_CZ] = (gl_t)cz;
    v[BA_AI] = av ? gl_inv(enc(av)) : 0;
    v[BA_AN] = (gl_t)an;
    v[BA_W] = (gl_t)w; v[BA_PLO] = (gl_t)plo; v[BA_PHI] = (gl_t)phi;
    v[BA_PDN] = (gl_t)pdn; v[BA_PDH] = (gl_t)pdh; v[BA_PUP] = (gl_t)pup;
    v[BA_U1H] = (gl_t)(u1 >> 9); v[BA_U1L] = (gl_t)(u1 & 511);
    v[BA_U2H] = (gl_t)(u2 >> 9); v[BA_U2L] = (gl_t)(u2 & 511);
    v[BA_QM] = (gl_t)(q - 128); v[BA_RR] = (gl_t)rr;
    v[BA_RB] = (gl_t)rb; v[BA_RT] = (gl_t)rt;
    v[BA_RTI] = rt ? gl_inv((gl_t)rt) : 0; v[BA_ZRT] = (gl_t)zrt;
    v[BA_L0] = (gl_t)l0; v[BA_QH] = (gl_t)qh; v[BA_RUP] = (gl_t)rup;
    v[BA_C] = (gl_t)c; v[BA_CI] = c ? 0 : gl_inv(enc(qr - 256));
    v[BA_EO] = (gl_t)eo;
    o.lu[BLU_E1] = (uint32_t)e1; o.lu[BLU_E2] = (uint32_t)e2;
    o.lu[BLU_M1] = (uint32_t)m1; o.lu[BLU_M2] = (uint32_t)m2;
    o.lu[BLU_D16] = kd >= 0 && kd <= 9 ? (uint32_t)kd : 0;
    o.lu[BLU_DR] = dr >= 0 ? (uint32_t)dr : 0;
    o.lu[BLU_DM] = dm >= 0 ? (uint32_t)dm : 0;
    o.lu[BLU_WDA] = (uint32_t)w;
    o.lu[BLU_U1H] = (uint32_t)(u1 >> 9); o.lu[BLU_U1L] = (uint32_t)(u1 & 511);
    o.lu[BLU_U2H] = (uint32_t)(u2 >> 9); o.lu[BLU_U2L] = (uint32_t)(u2 & 511);
    o.lu[BLU_RM] = rm17_idx((uint64_t)pdn, rr);
    o.lu[BLU_QM] = (uint32_t)(q - 128);
    o.lu[BLU_RMH] = rmh_idx(pdh, rb, rt);
    o.lu[BLU_EO] = (uint32_t)eo;
    o.out = out;
    return o;
}

// ---------------- constraint emitter ----------------
// c = the NBA block columns at this row; x1/x2/out = the operand and output
// pattern values.  Emits N_BA_C residuals, each of TOTAL degree <= 3 in the
// committed columns (so eq * sum fits the quartic sumcheck).
static const int N_BA_C = 52;
static inline void ba_constraints(const gl_t* c, gl_t x1, gl_t x2, gl_t out, gl_t* r) {
    gl_t one = 1ULL;
    auto boolc = [](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    gl_t nz1 = gl_sub(one, c[BA_Z1]), nz2 = gl_sub(one, c[BA_Z2]);
    gl_t nfar = gl_sub(one, c[BA_FAR]);
    gl_t D = gl_sub(c[BA_EH], c[BA_EL]);
    r[0]  = gl_sub(x1, gl_add(gl_add(gl_mul(c[BA_S1], 32768ULL),
                                     gl_mul(c[BA_E1], 128ULL)), c[BA_M1]));
    r[1]  = boolc(c[BA_S1]);
    r[2]  = boolc(c[BA_Z1]);
    r[3]  = gl_mul(c[BA_Z1], c[BA_E1]);
    r[4]  = gl_sub(gl_mul(c[BA_E1], c[BA_I1]), nz1);
    r[5]  = gl_sub(x2, gl_add(gl_add(gl_mul(c[BA_S2], 32768ULL),
                                     gl_mul(c[BA_E2], 128ULL)), c[BA_M2]));
    r[6]  = boolc(c[BA_S2]);
    r[7]  = boolc(c[BA_Z2]);
    r[8]  = gl_mul(c[BA_Z2], c[BA_E2]);
    r[9]  = gl_sub(gl_mul(c[BA_E2], c[BA_I2]), nz2);
    r[10] = boolc(c[BA_SG]);
    r[11] = gl_sub(c[BA_EH], gl_add(c[BA_E1], gl_mul(c[BA_SG], gl_sub(c[BA_E2], c[BA_E1]))));
    r[12] = gl_sub(c[BA_EL], gl_add(c[BA_E2], gl_mul(c[BA_SG], gl_sub(c[BA_E1], c[BA_E2]))));
    r[13] = gl_sub(c[BA_MH], gl_add(gl_add(128ULL, c[BA_M1]),
                                    gl_mul(c[BA_SG], gl_sub(c[BA_M2], c[BA_M1]))));
    r[14] = gl_sub(c[BA_ML], gl_add(gl_add(128ULL, c[BA_M2]),
                                    gl_mul(c[BA_SG], gl_sub(c[BA_M1], c[BA_M2]))));
    r[15] = gl_sub(c[BA_SHS], gl_add(c[BA_S1], gl_mul(c[BA_SG], gl_sub(c[BA_S2], c[BA_S1]))));
    r[16] = gl_sub(c[BA_SLS], gl_add(c[BA_S2], gl_mul(c[BA_SG], gl_sub(c[BA_S1], c[BA_S2]))));
    r[17] = gl_sub(gl_add(c[BA_OP], gl_mul(gl_add(c[BA_SHS], c[BA_SHS]), c[BA_SLS])),
                   gl_add(c[BA_SHS], c[BA_SLS]));
    r[18] = boolc(c[BA_FAR]);
    r[19] = gl_mul(nfar, gl_sub(c[BA_KD], D));
    r[20] = gl_mul(c[BA_FAR], c[BA_KD]);
    r[21] = gl_mul(c[BA_FAR], gl_sub(D, gl_add(c[BA_DR], 10ULL)));
    r[22] = boolc(c[BA_DZ]);
    r[23] = gl_mul(c[BA_DZ], D);
    r[24] = gl_sub(gl_mul(D, c[BA_DI]), gl_sub(one, c[BA_DZ]));
    r[25] = gl_mul(gl_mul(c[BA_NZ], c[BA_DZ]),
                   gl_sub(gl_sub(c[BA_MH], c[BA_ML]), c[BA_DM]));
    r[26] = gl_sub(c[BA_ZB], gl_mul(c[BA_Z1], c[BA_Z2]));
    r[27] = gl_sub(c[BA_NZ], gl_mul(nz1, nz2));
    r[28] = gl_sub(c[BA_NNF], gl_mul(c[BA_NZ], nfar));
    r[29] = gl_mul(c[BA_FAR], c[BA_CZ]);
    r[30] = gl_add(gl_sub(c[BA_NN], c[BA_NNF]), gl_mul(c[BA_NZ], c[BA_CZ]));
    r[31] = gl_mul(c[BA_NNF],
                   gl_add(gl_sub(gl_sub(c[BA_AV], gl_mul(c[BA_MH], c[BA_PWD])), c[BA_ML]),
                          gl_mul(gl_add(c[BA_OP], c[BA_OP]), c[BA_ML])));
    r[32] = boolc(c[BA_CZ]);
    r[33] = gl_mul(c[BA_CZ], c[BA_AV]);
    r[34] = gl_sub(gl_mul(c[BA_AV], c[BA_AI]), gl_sub(one, c[BA_CZ]));
    r[35] = gl_sub(gl_add(c[BA_AN], gl_mul(c[BA_AV], c[BA_CZ])),
                   gl_add(c[BA_AV], gl_mul(c[BA_CZ], 128ULL)));
    r[36] = gl_sub(gl_sub(c[BA_AN], c[BA_PLO]),
                   gl_add(gl_mul(c[BA_U1H], 512ULL), c[BA_U1L]));
    r[37] = gl_sub(gl_sub(gl_sub(c[BA_PHI], one), c[BA_AN]),
                   gl_add(gl_mul(c[BA_U2H], 512ULL), c[BA_U2L]));
    r[38] = gl_sub(gl_add(gl_mul(gl_add(c[BA_QM], 128ULL), c[BA_PDN]), c[BA_RR]),
                   gl_mul(c[BA_AN], c[BA_PUP]));
    r[39] = gl_sub(c[BA_RR], gl_add(gl_mul(c[BA_RB], c[BA_PDH]), c[BA_RT]));
    r[40] = gl_sub(gl_mul(c[BA_RT], c[BA_RTI]), gl_sub(one, c[BA_ZRT]));
    r[41] = gl_mul(c[BA_ZRT], c[BA_RT]);
    r[42] = boolc(c[BA_ZRT]);
    r[43] = boolc(c[BA_L0]);
    r[44] = gl_sub(gl_add(c[BA_QM], 128ULL), gl_add(gl_add(c[BA_QH], c[BA_QH]), c[BA_L0]));
    r[45] = gl_sub(gl_add(c[BA_RUP], gl_mul(gl_mul(c[BA_RB], c[BA_ZRT]),
                                            gl_sub(one, c[BA_L0]))), c[BA_RB]);
    r[46] = boolc(c[BA_C]);
    gl_t qr256 = gl_sub(gl_add(c[BA_QM], c[BA_RUP]), 128ULL);     // Q+RUP-256
    r[47] = gl_mul(c[BA_C], qr256);
    r[48] = gl_sub(gl_mul(qr256, c[BA_CI]), gl_sub(one, c[BA_C]));
    r[49] = gl_mul(c[BA_NN],
                   gl_sub(gl_add(c[BA_EO], 8ULL),
                          gl_add(gl_add(c[BA_EL], c[BA_W]), c[BA_C])));
    gl_t hipat = gl_sub(gl_add(gl_add(gl_mul(c[BA_SHS], 32768ULL),
                                      gl_mul(c[BA_EH], 128ULL)), c[BA_MH]), 128ULL);
    gl_t nrpat = gl_sub(gl_add(gl_add(gl_add(gl_mul(c[BA_SHS], 32768ULL),
                                             gl_mul(c[BA_EO], 128ULL)),
                                      gl_add(c[BA_QM], c[BA_RUP])),
                               0ULL), gl_mul(c[BA_C], 128ULL));
    r[50] = gl_sub(out,
             gl_add(gl_add(gl_mul(gl_mul(c[BA_ZB], c[BA_S1]), gl_mul(c[BA_S2], 32768ULL)),
                           gl_add(gl_mul(gl_mul(c[BA_Z1], nz2), x2),
                                  gl_mul(gl_mul(nz1, c[BA_Z2]), x1))),
                    gl_add(gl_mul(gl_mul(c[BA_NZ], c[BA_FAR]), hipat),
                           gl_mul(c[BA_NN], nrpat))));
    r[51] = gl_mul(gl_mul(c[BA_NZ], c[BA_FAR]), gl_sub(c[BA_EO], c[BA_EH]));
}

// =====================================================================
// Standalone elementwise gadget (this IS the residual op): committed X1,
// X2, OUT over a flat pow2 domain n = 2^ln.
// =====================================================================

struct Golden {
    uint32_t n = 0; int64_t flags = 0;
    std::vector<uint16_t> a, b, o;
};
static inline bool load_goldens(const char* path, std::vector<Golden>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[2];
    if (fread(hdr, 8, 2, f) != 2 || hdr[0] != 0x42464147) { fclose(f); return false; }
    out.resize(hdr[1]);
    for (auto& G : out) {
        int64_t nf[2];
        if (fread(nf, 8, 2, f) != 2) { fclose(f); return false; }
        G.n = (uint32_t)nf[0]; G.flags = nf[1];
        G.a.resize(G.n); G.b.resize(G.n); G.o.resize(G.n);
        if (fread(G.a.data(), 2, G.n, f) != G.n ||
            fread(G.b.data(), 2, G.n, f) != G.n ||
            fread(G.o.data(), 2, G.n, f) != G.n) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

struct Wit {
    uint32_t n = 0, ln = 0;
    std::vector<gl_t> x1, x2, opat;
    std::vector<uint16_t> O;                  // computed output patterns
    std::vector<gl_t> ws[NBA];
    std::vector<uint32_t> lidx[NBLU];
};
static inline Wit gen_witness(const Golden& L, const BaTamper* tm = nullptr,
                              uint32_t tj = 0) {
    Wit wt;
    wt.n = L.n; wt.ln = ilog2(L.n);
    if ((1u << wt.ln) != L.n) throw std::runtime_error("bfa: n must be pow2");
    wt.x1.assign(L.n, 0); wt.x2.assign(L.n, 0); wt.opat.assign(L.n, 0);
    wt.O.assign(L.n, 0);
    for (int c = 0; c < NBA; c++) wt.ws[c].assign(L.n, 0);
    for (int i = 0; i < NBLU; i++) wt.lidx[i].assign(L.n, 0);
    for (uint32_t j = 0; j < L.n; j++) {
        BaVals bv = ba_fill(L.a[j], L.b[j], (tm && j == tj) ? tm : nullptr);
        wt.x1[j] = L.a[j]; wt.x2[j] = L.b[j];
        for (int c = 0; c < NBA; c++) wt.ws[c][j] = bv.v[c];
        for (int i = 0; i < NBLU; i++) wt.lidx[i][j] = bv.lu[i];
        wt.opat[j] = bv.out; wt.O[j] = bv.out;
    }
    return wt;
}

struct Operands { Col X1, X2, OUT; };
static inline Operands commit_operands(const Wit& wt, uint32_t R) {
    Operands ops;
    ops.X1 = commit_col_nc(wt.x1, R);
    ops.X2 = commit_col_nc(wt.x2, R);
    ops.OUT = commit_col_nc(wt.opat, R);
    return ops;
}

// zero-check: v = [E, X1, X2, OUT, ws cols]; lam[N_BA_C]
static inline gl_t F_ba(const gl_t* v, const gl_t* lam) {
    gl_t r[N_BA_C];
    ba_constraints(v + 4, v[1], v[2], v[3], r);
    gl_t s = 0;
    for (int j = 0; j < N_BA_C; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}

struct BfaProof {
    uint32_t n = 0;
    p3fri::Hash rws[NBA];
    std::vector<p3lu::GroupProof> lug;   // standalone merged lookup groups
    std::vector<Msg5> mE; gl_t yE[NBA] = {}; gl_t yX1 = 0, yX2 = 0, yOUT = 0;
    p3zkc::Blind zbl[1];                       // zk: element zero-check blind
    std::vector<p3bo::BatchProof> batches;
};

// ==================== prover ====================
static inline BfaProof prove(fs::Transcript& tr, const Wit& wt, const Tables& T,
                             const Operands& ops, uint32_t R, uint32_t Q,
                             bool strict = true, p3lu::XCtx* xc = nullptr) {
    BfaProof pf; pf.n = wt.n;
    uint32_t hdr[1] = {wt.n};
    tr.absorb("bfa-dims", hdr, sizeof hdr);
    for (int i = 0; i < NBT; i++) tr.absorb("bfa-tab", T.t[i].id.data(), 32);
    tr.absorb("bfa-X1", ops.X1.root.data(), 32);
    tr.absorb("bfa-X2", ops.X2.root.data(), 32);
    tr.absorb("bfa-O", ops.OUT.root.data(), 32);

    p3lu::XCtx xc_loc;
    p3lu::XCtx& XC = xc ? *xc : xc_loc;
    p3bo::PLedger& lg = XC.lg;
    std::deque<Col>& lucols = XC.keep;

    std::vector<Col>& C = XC.vec(NBA);
    for (int c = 0; c < NBA; c++) { C[c] = commit_col_nc(wt.ws[c], R); pf.rws[c] = C[c].root; }
    for (int c = 0; c < NBA; c++) tr.absorb("bfa-cw", pf.rws[c].data(), 32);

    auto LD = ba_lu_defs();
    for (int i = 0; i < NBLU; i++) {
        std::vector<LuColSpec> spec;
        for (int cid : LD[i].cols) spec.push_back(LC(&C[cid]));
        p3lu::defer_v(XC, std::move(spec), wt.lidx[i], T.t[LD[i].tab],
                      "bfaLU" + std::to_string(i));
    }

    std::vector<gl_t> zE = chal_vec(tr, wt.ln);
    gl_t lamE = chal(tr), lamEv[N_BA_C]; lamEv[0] = 1;
    for (int j = 1; j < N_BA_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(4 + NBA);
        cols.push_back(beq(p3zkc::zpt(zE)));
        cols.push_back(ops.X1.v); cols.push_back(ops.X2.v); cols.push_back(ops.OUT.v);
        for (int c = 0; c < NBA; c++) cols.push_back(C[c].v);
        CFn F = [&](const gl_t* v) { return F_ba(v, lamEv); };
        std::vector<gl_t> rE = p3hwl::sc5z(tr, "bfa-scE", wt.ln, std::move(cols), F, pf.mE,
                                           0, R, lg, lucols, pf.zbl[0]);
        pf.yX1 = claimc(tr, lg, ops.X1, rE);
        pf.yX2 = claimc(tr, lg, ops.X2, rE);
        pf.yOUT = claimc(tr, lg, ops.OUT, rE);
        for (int c = 0; c < NBA; c++) pf.yE[c] = claimc(tr, lg, C[c], rE);
    }

    if (!xc) {
        pf.lug = p3lu::lu_flush(tr, XC, R, Q, strict);
        for (size_t i = 0; i < lg.cls.size(); i++)
            pf.batches.push_back(p3bo::prove_class(tr, lg.cls[i], R, Q,
                                                   "bfa-bo" + std::to_string(i)));
    }
    return pf;
}

// ==================== verifier ====================
static inline bool verify(fs::Transcript& tr, const Tables& T, const BfaProof& pf,
                          const p3fri::Hash& rX1, const p3fri::Hash& rX2,
                          const p3fri::Hash& rOUT, uint32_t n,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr,
                          p3lu::VCtx* xv = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.n != n) return fail("dims mismatch");
    uint32_t ln = ilog2(n);
    if ((1u << ln) != n) return fail("n must be pow2");
    p3lu::VCtx vc_loc;
    p3lu::VCtx& VC = xv ? *xv : vc_loc;
    p3bo::VLedger& vlg = VC.vlg;

    uint32_t hdr[1] = {n};
    tr.absorb("bfa-dims", hdr, sizeof hdr);
    for (int i = 0; i < NBT; i++) tr.absorb("bfa-tab", T.t[i].id.data(), 32);
    tr.absorb("bfa-X1", rX1.data(), 32);
    tr.absorb("bfa-X2", rX2.data(), 32);
    tr.absorb("bfa-O", rOUT.data(), 32);
    for (int c = 0; c < NBA; c++) tr.absorb("bfa-cw", pf.rws[c].data(), 32);

    auto LD = ba_lu_defs();
    for (int i = 0; i < NBLU; i++) {
        std::vector<const p3fri::Hash*> roots;
        for (int cid : LD[i].cols) roots.push_back(&pf.rws[cid]);
        p3lu::vdefer_v(VC, std::move(roots), T.t[LD[i].tab], ln,
                       "bfaLU" + std::to_string(i));
    }

    std::vector<gl_t> zE = chal_vec(tr, ln);
    gl_t lamE = chal(tr), lamEv[N_BA_C]; lamEv[0] = 1;
    for (int j = 1; j < N_BA_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.zbl[0]);
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.mE, p3zkc::vfull(ln), gl_mul(rho, pf.zbl[0].H),
                        tr, "bfa-scE", rE, claim)) return fail("De sumcheck");
        if (!p3hwl::sc5vz_claims(tr, vlg, pf.zbl[0], rE)) return fail("sc5 blind ip");
        std::vector<gl_t> v(4 + NBA);
        v[0] = p3bf::eq_point(rE, p3zkc::zpt(zE));
        v[1] = claimv(tr, vlg, rX1, rE, pf.yX1);
        v[2] = claimv(tr, vlg, rX2, rE, pf.yX2);
        v[3] = claimv(tr, vlg, rOUT, rE, pf.yOUT);
        for (int c = 0; c < NBA; c++) v[4 + c] = claimv(tr, vlg, pf.rws[c], rE, pf.yE[c]);
        gl_t end = gl_add(F_ba(v.data(), lamEv),
                          p3hwl::sc5_blindterm(pf.zbl[0], rho, v[0]));
        if (end != claim) return fail("De terminal");
    }

    if (!xv) {
        if (!p3lu::lu_verify_flush(tr, VC, pf.lug, Q_pub, R_pub, why)) return false;
        if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
        for (size_t i = 0; i < vlg.cls.size(); i++)
            if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                    "bfa-bo" + std::to_string(i), why)) return false;
    } else if (!pf.batches.empty() || !pf.lug.empty()) return fail("unexpected batches");

    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const BfaProof& pf) {
    size_t s = 4 + NBA * 32;
    for (auto& g : pf.lug) s += p3lu::sz_group(g);
    s += pf.mE.size() * 40;
    s += 8 * (NBA + 3);
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3bfa
