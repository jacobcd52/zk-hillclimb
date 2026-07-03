// RMSNorm gadget: sound (non-ZK) prover+verifier that a committed bf16 output
// Y equals the CANONICAL RMSNorm (transformer_ref.py RMSNormSpec) of a
// committed bf16 input X with committed bf16 gains G, bitwise.
//
//   per row b (length d = 2^ld):
//     decompose x = s*2^15 + eb*2^7 + mb; z = (eb==0)  [canonical FTZ]
//     sq = (128+mb)^2, binade esq = 2*eb                       (present: !z)
//     E  = max(EMIN, max_present esq)     [dominance + attainment selectors,
//                                          phantom floor at the public EMIN]
//     per element: sh = E - esq (present), pw = 2^min(sh,16), q*pw + r = sq
//     S  = sum_present q                  [row-binding sumcheck]
//     S' = S + EPSA(E)                    [public 512-row eps table]
//     wd = bitlen(S')  [pow2 sandwich];  u16 = top-16 bits  [u16*PDN+NR=S'*PUP]
//     Xexp = E + wd - (284+ld);  Xexp = 2*(QB-200) + PP
//     (mr, hb) = RSQ[u16, PP];  r = bf16(mantissa mr, exp TEB = 318+HB-QB)
//     y_i = MUL7(x_i, r) then MUL7(t, g_i)   [RNE mantissa-product table,
//                                             exponents linear + REXP range]
//
// Supported domain (proof REJECTS otherwise -- sound, not complete): the
// rsqrt exponent TEB and both product exponents stay in the normal bf16 range
// [1,254].  The canonical REFERENCE is total (it flushes/saturates); rows that
// clamp are outside this gadget's v1 domain.  Everything else (zero rows,
// subnormal inputs, +-0 gains, eps-dominated rows, binade-255 inputs whose
// products stay in range) is in-domain.
//
// Reuses the p3 stack end to end: p3lu logUp lookups (deferred openings),
// p3hwl quartic zero-check sumchecks, p3bo batched per-size-class openings.
// Non-ZK: the mask-slice + Libra-blind pass (design doc section 10) applies to
// these columns unchanged once the layer composes.
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

namespace p3rms {

using p3lu::Col; using p3lu::Table; using p3lu::make_table; using p3lu::commit_col_nc;
using p3lu::chal; using p3lu::chal_vec; using p3lu::ilog2;
using p3lu::LuColSpec; using p3lu::LC; using p3lu::LV;
using p3hwl::Msg5; using p3hwl::CFn; using p3hwl::sc5_prove; using p3hwl::sc5_verify;
using p3hwl::claimc; using p3hwl::claimv; using p3hwl::beq; using p3hwl::enc;
using p3hwl::pow2_at_least; using p3hwl::now_ms;

// ---------------- canonical-table artifact (transformer_ref.py --dump-tables) --
struct Art {
    int64_t ld = 0, EMIN = 0, T = 0, eps_bits = 0, sm_emin = 0;
    std::vector<uint8_t> mul_mo, mul_einc;     // 16384: RNE mantissa product
    std::vector<uint8_t> rsq_mr, rsq_hb;       // 65536: rsqrt(u16 * 2^pp)
    std::vector<uint8_t> rcp_mr, rcp_hb;       // 32768: 1/u16 (softmax; unused here)
    std::vector<int64_t> epsa;                 // 512
    std::vector<uint16_t> exp_tab, silu_tab;   // 65536 each (future gadgets)
    std::vector<uint8_t> qe4m3;                // 4096 (future quantize gadget)
};
static inline bool load_art(const char* path, Art& a) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[7];
    if (fread(hdr, 8, 7, f) != 7 || hdr[0] != 0x524D5354 || hdr[1] != 1) { fclose(f); return false; }
    a.ld = hdr[2]; a.EMIN = hdr[3]; a.T = hdr[4]; a.eps_bits = hdr[5]; a.sm_emin = hdr[6];
    auto rd8 = [&](std::vector<uint8_t>& v, size_t n) {
        v.resize(n); return fread(v.data(), 1, n, f) == n; };
    auto rd16 = [&](std::vector<uint16_t>& v, size_t n) {
        v.resize(n); return fread(v.data(), 2, n, f) == n; };
    bool ok = rd8(a.mul_mo, 16384) && rd8(a.mul_einc, 16384)
           && rd8(a.rsq_mr, 65536) && rd8(a.rsq_hb, 65536)
           && rd8(a.rcp_mr, 32768) && rd8(a.rcp_hb, 32768);
    a.epsa.resize(512);
    ok = ok && fread(a.epsa.data(), 8, 512, f) == 512;
    ok = ok && rd16(a.exp_tab, 65536) && rd16(a.silu_tab, 65536) && rd8(a.qe4m3, 4096);
    fclose(f);
    return ok && a.T == 16;
}

// ---------------- fixed public tables ----------------
struct Tables {
    Table R128, R256, R512, R11, R12, R16;     // plain ranges
    Table SH512;                                // (sh in [0,512), pw=2^min(sh,16))
    Table RM17, RM8;                            // (pw, r) with r < pw <= 2^16 / 2^7
    Table WD24;                                 // wd in [0,23]: plo,phi,pdn,pup (T=16)
    Table EPSAT;                                // (E, EPSA(E)) 512 rows
    Table RSQT;                                 // (u16, pp, 128+mr, hb) 65536 rows
    Table MUL7;                                 // (ma, mb, 128+mo, einc) 16384 rows
    Table REXP;                                 // normal exponent [1,254]
};
static inline Tables build_tables(const Art& a) {
    Tables T;
    auto range = [](uint32_t n) { std::vector<gl_t> v(n);
        for (uint32_t j = 0; j < n; j++) v[j] = j; return make_table({v}); };
    T.R128 = range(128); T.R256 = range(256); T.R512 = range(512);
    T.R11 = range(2048); T.R12 = range(4096); T.R16 = range(65536);
    { std::vector<gl_t> s(512), p(512);
      for (uint32_t j = 0; j < 512; j++) { s[j] = j; p[j] = 1ULL << (j < 16 ? j : 16); }
      T.SH512 = make_table({s, p}); }
    { std::vector<gl_t> p(131072, 1), r(131072, 0); uint32_t row = 0;   // r < pw <= 2^16
      for (uint32_t t = 0; t <= 16; t++)
          for (uint32_t x = 0; x < (1u << t); x++) { p[row] = 1ULL << t; r[row] = x; row++; }
      T.RM17 = make_table({p, r}); }
    { std::vector<gl_t> p(256, 1), r(256, 0); uint32_t row = 0;         // r < pw <= 2^7
      for (uint32_t t = 0; t <= 7; t++)
          for (uint32_t x = 0; x < (1u << t); x++) { p[row] = 1ULL << t; r[row] = x; row++; }
      T.RM8 = make_table({p, r}); }
    { std::vector<gl_t> w(32), plo(32), phi(32), pdn(32), pup(32);
      for (uint32_t j = 0; j < 32; j++) {
          uint32_t wd = j <= 23 ? j : 0;
          w[j] = wd; plo[j] = wd ? (1ULL << (wd - 1)) : 0; phi[j] = 1ULL << wd;
          pdn[j] = 1ULL << (wd > 16 ? wd - 16 : 0);
          pup[j] = 1ULL << (wd < 16 ? 16 - wd : 0);
      }
      T.WD24 = make_table({w, plo, phi, pdn, pup}); }
    { std::vector<gl_t> e(512), v(512);
      for (uint32_t j = 0; j < 512; j++) { e[j] = j; v[j] = (gl_t)a.epsa[j]; }
      T.EPSAT = make_table({e, v}); }
    { std::vector<gl_t> u(65536), p(65536), m(65536), h(65536);
      for (uint32_t j = 0; j < 65536; j++) {
          u[j] = 32768 + (j & 32767); p[j] = j >> 15;
          m[j] = 128 + a.rsq_mr[j]; h[j] = a.rsq_hb[j];
      }
      T.RSQT = make_table({u, p, m, h}); }
    { std::vector<gl_t> ma(16384), mb(16384), mo(16384), ei(16384);
      for (uint32_t j = 0; j < 16384; j++) {
          ma[j] = 128 + (j >> 7); mb[j] = 128 + (j & 127);
          mo[j] = 128 + a.mul_mo[j]; ei[j] = a.mul_einc[j];
      }
      T.MUL7 = make_table({ma, mb, mo, ei}); }
    { std::vector<gl_t> v(256);
      for (uint32_t j = 0; j < 256; j++) v[j] = (j >= 1 && j <= 254) ? j : 1;
      T.REXP = make_table({v}); }
    return T;
}

// ---------------- column enums ----------------
enum {  // De: per-element (index e = b*d + i, bits [i | b])
    D_XS = 0, D_XEB, D_XMB, D_XZ, D_XEI, D_SH, D_PW, D_Q, D_RR, D_SEL,
    D_MO1, D_EI1, D_EO1, D_MO2, D_EI2, D_EO2, NDE };
enum {  // Db: per-row
    R_E = 0, R_FSEL, R_EDIF, R_S, R_EPSA, R_WD, R_PLO, R_PHI, R_PDN, R_PUP,
    R_U1H, R_U1L, R_U2H, R_U2L, R_NR, R_U16, R_QB, R_PP, R_MR, R_HB, R_TEB, NDB };
enum {  // Dw: gain decomposition (length d)
    W_GS = 0, W_GEB, W_GMB, W_GZ, W_GEI, NDW };
enum {  // lookup instances, fixed order
    LUR_XEB = 0, LUR_XMB, LUR_SH, LUR_RM, LUR_Q, LUR_M1, LUR_M2, LUR_EO1, LUR_EO2,
    LUR_GEB, LUR_GMB,
    LUR_EPSA, LUR_WD, LUR_U1H, LUR_U1L, LUR_U2H, LUR_U2L, LUR_NR, LUR_RSQ,
    LUR_EDIF, LUR_QB, LUR_TEB, NLUR };

// ---------------- dims / golden ----------------
struct Dims {
    uint32_t B, d, ld;
    uint32_t Bpad, lb, le;                    // le = lb + ld
    size_t Ne;                                // padded element count
};
static inline Dims make_dims(uint32_t B, uint32_t ld) {
    Dims dm; dm.B = B; dm.ld = ld; dm.d = 1u << ld;
    dm.Bpad = pow2_at_least(B < 2 ? 2 : B);
    dm.lb = ilog2(dm.Bpad); dm.le = dm.lb + ld;
    dm.Ne = (size_t)dm.Bpad << ld;
    return dm;
}
struct Golden {
    uint32_t B = 0, d = 0;
    std::vector<uint16_t> x, g, y;            // B*d, d, B*d bf16 patterns
};
static inline bool load_goldens(const char* path, std::vector<Golden>& out, uint32_t& ld) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[3];
    if (fread(hdr, 8, 3, f) != 3 || hdr[0] != 0x524D5347) { fclose(f); return false; }
    ld = (uint32_t)hdr[2];
    uint32_t d = 1u << ld;
    out.resize(hdr[1]);
    for (auto& G : out) {
        int64_t b;
        if (fread(&b, 8, 1, f) != 1) { fclose(f); return false; }
        G.B = (uint32_t)b; G.d = d;
        G.x.resize((size_t)G.B * d); G.g.resize(d); G.y.resize((size_t)G.B * d);
        if (fread(G.x.data(), 2, G.x.size(), f) != G.x.size() ||
            fread(G.g.data(), 2, d, f) != d ||
            fread(G.y.data(), 2, G.y.size(), f) != G.y.size()) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

// ---------------- witness ----------------
// Selftest-only semantic forgeries: apply ONE forgery, replay HONESTLY
// downstream, so exactly one sub-argument must catch it.
enum { RT_NONE = 0,
       RT_SUM,      // row sum-of-squares S+1 (downstream honest from forged S)
       RT_RSQ,      // rsqrt mantissa MR+1 (not a table row)
       RT_EPSA,     // EPSA+1 (not a table row)
       RT_MAXEXP,   // row max E overstated by 1 (no attainer -> fsel forced)
       RT_ROUND,    // per-element align rounds up (q+1, r-=pw)
       RT_MULUP };  // output mul rounds up (MO2+1: not a table row)
struct RTamper { int mode = RT_NONE; uint32_t b = 0, i = 0; };

struct Wit {
    Dims dm;
    std::vector<gl_t> xpat, gpat, ypat;       // padded pattern columns
    std::vector<uint16_t> Y;                  // real B*d output patterns
    std::vector<gl_t> de[NDE], db[NDB], dw[NDW];
    std::vector<gl_t> xmf, mrf, gmf;          // virtual MUL7 key columns (De)
    std::vector<uint32_t> lidx[NLUR];
};

static inline int bitlen(uint64_t x) { int w = 0; while (x) { w++; x >>= 1; } return w; }
static inline gl_t inv_or0(uint64_t x) { return x ? gl_inv((gl_t)x) : 0; }

// canonical witness replay (mirrors transformer_ref.RMSNormSpec bit for bit)
static inline Wit gen_witness(const Golden& L, const Art& a, const RTamper* tm = nullptr) {
    Wit wt; Dims& dm = wt.dm; dm = make_dims(L.B, (uint32_t)a.ld);
    if (L.d != dm.d) throw std::runtime_error("rms: golden d mismatch");
    const uint32_t d = dm.d, ld = dm.ld;
    const int64_t EMIN = a.EMIN;

    wt.xpat.assign(dm.Ne, 0); wt.gpat.assign(d, 0); wt.ypat.assign(dm.Ne, 0);
    for (uint32_t b = 0; b < L.B; b++)
        for (uint32_t i = 0; i < d; i++)
            wt.xpat[((size_t)b << ld) | i] = L.x[(size_t)b * d + i];
    for (uint32_t i = 0; i < d; i++) wt.gpat[i] = L.g[i];
    wt.Y.assign((size_t)L.B * d, 0);

    for (int c = 0; c < NDE; c++) wt.de[c].assign(dm.Ne, 0);
    for (int c = 0; c < NDB; c++) wt.db[c].assign(dm.Bpad, 0);
    for (int c = 0; c < NDW; c++) wt.dw[c].assign(d, 0);
    wt.xmf.assign(dm.Ne, 0); wt.mrf.assign(dm.Ne, 0); wt.gmf.assign(dm.Ne, 0);
    for (int i = 0; i < NLUR; i++) {
        size_t sz = dm.Ne;
        if (i == LUR_GEB || i == LUR_GMB) sz = d;
        else if (i >= LUR_EPSA) sz = dm.Bpad;
        wt.lidx[i].assign(sz, 0);
    }
    auto rm_idx = [](uint64_t pw, int64_t r) { return (uint32_t)((pw - 1) + (uint64_t)r); };

    // gain decomposition
    std::vector<int64_t> gs(d), geb(d), gmb(d), gz(d);
    for (uint32_t i = 0; i < d; i++) {
        uint32_t p = (uint32_t)wt.gpat[i];
        gs[i] = (p >> 15) & 1; geb[i] = (p >> 7) & 255; gmb[i] = p & 127;
        gz[i] = geb[i] == 0 ? 1 : 0;
        wt.dw[W_GS][i] = (gl_t)gs[i]; wt.dw[W_GEB][i] = (gl_t)geb[i];
        wt.dw[W_GMB][i] = (gl_t)gmb[i]; wt.dw[W_GZ][i] = (gl_t)gz[i];
        wt.dw[W_GEI][i] = inv_or0((uint64_t)geb[i]);
        wt.lidx[LUR_GEB][i] = (uint32_t)geb[i];
        wt.lidx[LUR_GMB][i] = (uint32_t)gmb[i];
    }

    for (uint32_t b = 0; b < dm.Bpad; b++) {
        // decode row
        std::vector<int64_t> xs(d), xeb(d), xmb(d), xz(d);
        for (uint32_t i = 0; i < d; i++) {
            uint32_t p = (uint32_t)wt.xpat[((size_t)b << ld) | i];
            xs[i] = (p >> 15) & 1; xeb[i] = (p >> 7) & 255; xmb[i] = p & 127;
            xz[i] = xeb[i] == 0 ? 1 : 0;
        }
        // row max binade E with phantom floor EMIN
        int64_t E = EMIN;
        for (uint32_t i = 0; i < d; i++)
            if (!xz[i] && 2 * xeb[i] > E) E = 2 * xeb[i];
        bool forged_max = tm && tm->mode == RT_MAXEXP && b == tm->b;
        if (forged_max) E += 1;
        // attainment: first present attainer, else the floor
        int64_t fsel = 1; int sel_i = -1;
        if (!forged_max)
            for (uint32_t i = 0; i < d; i++)
                if (!xz[i] && 2 * xeb[i] == E) { sel_i = (int)i; fsel = 0; break; }
        int64_t edif = E - EMIN;
        // per-element alignment + sum
        int64_t S = 0;
        std::vector<int64_t> sh(d, 0), pw(d, 1), q(d, 0), rr(d, 0);
        for (uint32_t i = 0; i < d; i++) {
            size_t e = ((size_t)b << ld) | i;
            if (!xz[i]) {
                sh[i] = E - 2 * xeb[i];
                int64_t shc = sh[i] < 16 ? sh[i] : 16;
                pw[i] = (int64_t)1 << shc;
                int64_t sq = (128 + xmb[i]) * (128 + xmb[i]);
                q[i] = sq >> shc; rr[i] = sq - (q[i] << shc);
                if (tm && tm->mode == RT_ROUND && b == tm->b && i == tm->i && rr[i] > 0) {
                    q[i] += 1; rr[i] -= pw[i];
                }
                S += q[i];
            }
            wt.de[D_XS][e] = (gl_t)xs[i]; wt.de[D_XEB][e] = (gl_t)xeb[i];
            wt.de[D_XMB][e] = (gl_t)xmb[i]; wt.de[D_XZ][e] = (gl_t)xz[i];
            wt.de[D_XEI][e] = inv_or0((uint64_t)xeb[i]);
            wt.de[D_SH][e] = (gl_t)sh[i]; wt.de[D_PW][e] = (gl_t)pw[i];
            wt.de[D_Q][e] = (gl_t)q[i]; wt.de[D_RR][e] = enc(rr[i]);
            wt.de[D_SEL][e] = (sel_i == (int)i) ? 1 : 0;
            wt.lidx[LUR_XEB][e] = (uint32_t)xeb[i];
            wt.lidx[LUR_XMB][e] = (uint32_t)xmb[i];
            wt.lidx[LUR_SH][e] = (uint32_t)sh[i];
            wt.lidx[LUR_RM][e] = rr[i] >= 0 ? rm_idx((uint64_t)pw[i], rr[i]) : 0;
            wt.lidx[LUR_Q][e] = (uint32_t)q[i];
        }
        if (tm && tm->mode == RT_SUM && b == tm->b) S += 1;
        // eps + normalize + rsqrt
        int64_t epsa = a.epsa[E < 512 ? E : 511];
        if (tm && tm->mode == RT_EPSA && b == tm->b) epsa += 1;
        int64_t Sp = S + epsa;
        if (Sp < 1 || Sp >= (1 << 23)) throw std::runtime_error("rms: S' window");
        int64_t wd = bitlen((uint64_t)Sp);
        int64_t plo = (int64_t)1 << (wd - 1), phi = (int64_t)1 << wd;
        int64_t pdn = (int64_t)1 << (wd > 16 ? wd - 16 : 0);
        int64_t pup = (int64_t)1 << (wd < 16 ? 16 - wd : 0);
        int64_t u1 = Sp - plo, u2 = phi - 1 - Sp;
        int64_t nr = wd > 16 ? Sp & (pdn - 1) : 0;
        int64_t u16 = wd > 16 ? Sp >> (wd - 16) : Sp << (16 - wd);
        int64_t Xexp = E + wd - (284 + (int64_t)ld);
        int64_t qb = ((Xexp + 400) >> 1);            // = floor(Xexp/2) + 200
        int64_t pp = Xexp + 400 - 2 * qb;
        uint32_t rsj = (uint32_t)((pp << 15) | (u16 - 32768));
        int64_t mrfull = 128 + a.rsq_mr[rsj], hb = a.rsq_hb[rsj];
        if (tm && tm->mode == RT_RSQ && b == tm->b) mrfull += (mrfull < 255 ? 1 : -1);
        int64_t teb = 318 + hb - qb;
        if (teb < 1 || teb > 254) throw std::runtime_error("rms: rsqrt exp domain");
        wt.db[R_E][b] = (gl_t)E; wt.db[R_FSEL][b] = (gl_t)fsel;
        wt.db[R_EDIF][b] = (gl_t)edif; wt.db[R_S][b] = (gl_t)S;
        wt.db[R_EPSA][b] = (gl_t)epsa; wt.db[R_WD][b] = (gl_t)wd;
        wt.db[R_PLO][b] = (gl_t)plo; wt.db[R_PHI][b] = (gl_t)phi;
        wt.db[R_PDN][b] = (gl_t)pdn; wt.db[R_PUP][b] = (gl_t)pup;
        wt.db[R_U1H][b] = (gl_t)(u1 >> 12); wt.db[R_U1L][b] = (gl_t)(u1 & 4095);
        wt.db[R_U2H][b] = (gl_t)(u2 >> 12); wt.db[R_U2L][b] = (gl_t)(u2 & 4095);
        wt.db[R_NR][b] = (gl_t)nr; wt.db[R_U16][b] = (gl_t)u16;
        wt.db[R_QB][b] = (gl_t)qb; wt.db[R_PP][b] = (gl_t)pp;
        wt.db[R_MR][b] = (gl_t)mrfull; wt.db[R_HB][b] = (gl_t)hb;
        wt.db[R_TEB][b] = (gl_t)teb;
        wt.lidx[LUR_EPSA][b] = (uint32_t)(E < 512 ? E : 511);
        wt.lidx[LUR_WD][b] = (uint32_t)wd;
        wt.lidx[LUR_U1H][b] = (uint32_t)(u1 >> 12); wt.lidx[LUR_U1L][b] = (uint32_t)(u1 & 4095);
        wt.lidx[LUR_U2H][b] = (uint32_t)(u2 >> 12); wt.lidx[LUR_U2L][b] = (uint32_t)(u2 & 4095);
        wt.lidx[LUR_NR][b] = rm_idx((uint64_t)pdn, nr);
        wt.lidx[LUR_RSQ][b] = rsj;
        wt.lidx[LUR_EDIF][b] = (uint32_t)edif;
        wt.lidx[LUR_QB][b] = (uint32_t)qb;
        wt.lidx[LUR_TEB][b] = (uint32_t)teb;
        // output multiplies
        for (uint32_t i = 0; i < d; i++) {
            size_t e = ((size_t)b << ld) | i;
            uint32_t j1 = (uint32_t)((xmb[i] << 7) | (mrfull - 128));
            int64_t mo1 = 128 + a.mul_mo[j1], ei1 = a.mul_einc[j1];
            int64_t eo1 = xz[i] ? 1 : xeb[i] + teb - 127 + ei1;
            if (eo1 < 1 || eo1 > 254) throw std::runtime_error("rms: mul1 exp domain");
            uint32_t j2 = (uint32_t)(((mo1 - 128) << 7) | gmb[i]);
            int64_t mo2 = 128 + a.mul_mo[j2], ei2 = a.mul_einc[j2];
            if (tm && tm->mode == RT_MULUP && b == tm->b && i == tm->i)
                mo2 += (mo2 < 255 ? 1 : -1);
            int64_t z2 = (xz[i] || gz[i]) ? 1 : 0;
            int64_t eo2 = z2 ? 1 : eo1 + geb[i] - 127 + ei2;
            if (eo2 < 1 || eo2 > 254) throw std::runtime_error("rms: mul2 exp domain");
            int64_t sy = xs[i] ^ gs[i];
            int64_t y = (sy << 15) | (z2 ? 0 : ((eo2 << 7) | (mo2 - 128)));
            wt.de[D_MO1][e] = (gl_t)mo1; wt.de[D_EI1][e] = (gl_t)ei1;
            wt.de[D_EO1][e] = (gl_t)eo1;
            wt.de[D_MO2][e] = (gl_t)mo2; wt.de[D_EI2][e] = (gl_t)ei2;
            wt.de[D_EO2][e] = (gl_t)eo2;
            wt.xmf[e] = (gl_t)(128 + xmb[i]); wt.mrf[e] = (gl_t)mrfull;
            wt.gmf[e] = (gl_t)(128 + gmb[i]);
            wt.lidx[LUR_M1][e] = j1; wt.lidx[LUR_M2][e] = j2;
            wt.lidx[LUR_EO1][e] = (uint32_t)eo1; wt.lidx[LUR_EO2][e] = (uint32_t)eo2;
            wt.ypat[e] = (gl_t)y;
            if (b < L.B) wt.Y[(size_t)b * d + i] = (uint16_t)y;
        }
    }
    return wt;
}

// ---------------- operand commitments ----------------
struct Operands { Col X, G, Y; };
static inline Operands commit_operands(const Wit& wt, uint32_t R) {
    Operands ops;
    ops.X = commit_col_nc(wt.xpat, R);
    ops.G = commit_col_nc(wt.gpat, R);
    ops.Y = commit_col_nc(wt.ypat, R);
    return ops;
}

// ---------------- constraint functions ----------------
// De zero-check: v = [E, X, Y, de cols, Ebc, TEBbc, GSbc, GEBbc, GZbc]; lam[16]
static const int N_DE_C = 16;
static inline gl_t F_de(const gl_t* v, const gl_t* lam) {
    const gl_t* c = v + 3;
    gl_t one = 1ULL;
    gl_t Ebc = v[3 + NDE], TEB = v[4 + NDE], GS = v[5 + NDE], GEB = v[6 + NDE], GZ = v[7 + NDE];
    gl_t nz = gl_sub(one, c[D_XZ]);
    gl_t z2 = gl_sub(gl_add(c[D_XZ], GZ), gl_mul(c[D_XZ], GZ));
    gl_t nz2 = gl_sub(one, z2);
    gl_t xmf = gl_add(128ULL, c[D_XMB]);
    gl_t sy = gl_sub(gl_add(c[D_XS], GS), gl_mul(gl_add(c[D_XS], c[D_XS]), GS));
    auto boolc = [](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    gl_t r[N_DE_C];
    r[0]  = gl_sub(v[1], gl_add(gl_add(gl_mul(c[D_XS], 32768ULL),
                                       gl_mul(c[D_XEB], 128ULL)), c[D_XMB]));
    r[1]  = boolc(c[D_XS]);
    r[2]  = boolc(c[D_XZ]);
    r[3]  = gl_mul(c[D_XZ], c[D_XEB]);
    r[4]  = gl_sub(gl_mul(c[D_XEB], c[D_XEI]), nz);
    r[5]  = gl_sub(gl_add(gl_mul(c[D_Q], c[D_PW]), c[D_RR]),
                   gl_mul(nz, gl_mul(xmf, xmf)));
    r[6]  = gl_mul(nz, gl_sub(gl_add(c[D_SH], gl_add(c[D_XEB], c[D_XEB])), Ebc));
    r[7]  = gl_mul(c[D_XZ], c[D_SH]);
    r[8]  = boolc(c[D_SEL]);
    r[9]  = gl_mul(c[D_SEL], c[D_SH]);
    r[10] = gl_mul(c[D_SEL], c[D_XZ]);
    r[11] = gl_mul(nz, gl_sub(c[D_EO1],
                    gl_sub(gl_add(gl_add(c[D_XEB], TEB), c[D_EI1]), 127ULL)));
    r[12] = gl_mul(c[D_XZ], gl_sub(c[D_EO1], 1ULL));
    r[13] = gl_mul(z2, gl_sub(c[D_EO2], 1ULL));
    r[14] = gl_mul(nz2, gl_sub(c[D_EO2],
                    gl_sub(gl_add(gl_add(c[D_EO1], GEB), c[D_EI2]), 127ULL)));
    r[15] = gl_sub(v[2], gl_add(gl_mul(sy, 32768ULL),
                    gl_mul(nz2, gl_sub(gl_add(gl_mul(c[D_EO2], 128ULL), c[D_MO2]),
                                       128ULL))));
    gl_t s = 0;
    for (int j = 0; j < N_DE_C; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}
// Db zero-check: v = [E, db cols]; lam[10]; needs public EMIN and (284+ld)
static const int N_DB_C = 10;
static inline gl_t F_db(const gl_t* v, const gl_t* lam, gl_t EMIN, gl_t C284) {
    const gl_t* c = v + 1;
    auto boolc = [](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    gl_t Sp = gl_add(c[R_S], c[R_EPSA]);
    gl_t r[N_DB_C];
    r[0] = boolc(c[R_FSEL]);
    r[1] = gl_mul(c[R_FSEL], c[R_EDIF]);
    r[2] = gl_sub(c[R_E], gl_add(EMIN, c[R_EDIF]));
    r[3] = boolc(c[R_PP]);
    r[4] = boolc(c[R_HB]);
    r[5] = gl_sub(gl_add(gl_add(c[R_E], c[R_WD]), 400ULL),
                  gl_add(gl_add(gl_add(c[R_QB], c[R_QB]), c[R_PP]), C284));
    r[6] = gl_sub(gl_add(c[R_TEB], c[R_QB]), gl_add(318ULL, c[R_HB]));
    r[7] = gl_sub(gl_sub(Sp, c[R_PLO]),
                  gl_add(gl_mul(c[R_U1H], 4096ULL), c[R_U1L]));
    r[8] = gl_sub(gl_sub(gl_sub(c[R_PHI], 1ULL), Sp),
                  gl_add(gl_mul(c[R_U2H], 4096ULL), c[R_U2L]));
    r[9] = gl_sub(gl_add(gl_mul(c[R_U16], c[R_PDN]), c[R_NR]), gl_mul(Sp, c[R_PUP]));
    gl_t s = 0;
    for (int j = 0; j < N_DB_C; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}
// Dw zero-check: v = [E, G, gs, geb, gmb, gz, gei]; lam[5]
static const int N_DW_C = 5;
static inline gl_t F_dw(const gl_t* v, const gl_t* lam) {
    gl_t one = 1ULL;
    auto boolc = [](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    gl_t r[N_DW_C];
    r[0] = gl_sub(v[1], gl_add(gl_add(gl_mul(v[2], 32768ULL), gl_mul(v[3], 128ULL)), v[4]));
    r[1] = boolc(v[2]);
    r[2] = boolc(v[5]);
    r[3] = gl_mul(v[5], v[3]);
    r[4] = gl_sub(gl_mul(v[3], v[6]), gl_sub(one, v[5]));
    gl_t s = 0;
    for (int j = 0; j < N_DW_C; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}
// row binding: v = [EQb, XZ, Q, SEL]; lam[2]
static inline gl_t F_bind(const gl_t* v, const gl_t* lam) {
    return gl_mul(v[0], gl_add(gl_mul(lam[0], gl_mul(gl_sub(1ULL, v[1]), v[2])),
                               gl_mul(lam[1], v[3])));
}

// ---------------- lookup descriptors ----------------
// dom: 0=De 1=Dw 2=Db; col >= 0 committed; -1 xmf, -2 mrf, -3 gmf (virtual)
struct LuDef { const Table* tab; int dom; std::vector<int> cols; const char* label; };
static inline std::vector<LuDef> lu_defs(const Tables& T) {
    std::vector<LuDef> L(NLUR);
    L[LUR_XEB] = {&T.R256, 0, {D_XEB}, "rmsXEB"};
    L[LUR_XMB] = {&T.R128, 0, {D_XMB}, "rmsXMB"};
    L[LUR_SH]  = {&T.SH512, 0, {D_SH, D_PW}, "rmsSH"};
    L[LUR_RM]  = {&T.RM17, 0, {D_PW, D_RR}, "rmsRM"};
    L[LUR_Q]   = {&T.R16, 0, {D_Q}, "rmsQ"};
    L[LUR_M1]  = {&T.MUL7, 0, {-1, -2, D_MO1, D_EI1}, "rmsM1"};
    L[LUR_M2]  = {&T.MUL7, 0, {D_MO1, -3, D_MO2, D_EI2}, "rmsM2"};
    L[LUR_EO1] = {&T.REXP, 0, {D_EO1}, "rmsEO1"};
    L[LUR_EO2] = {&T.REXP, 0, {D_EO2}, "rmsEO2"};
    L[LUR_GEB] = {&T.R256, 1, {W_GEB}, "rmsGEB"};
    L[LUR_GMB] = {&T.R128, 1, {W_GMB}, "rmsGMB"};
    L[LUR_EPSA]= {&T.EPSAT, 2, {R_E, R_EPSA}, "rmsEPSA"};
    L[LUR_WD]  = {&T.WD24, 2, {R_WD, R_PLO, R_PHI, R_PDN, R_PUP}, "rmsWD"};
    L[LUR_U1H] = {&T.R11, 2, {R_U1H}, "rmsU1H"};
    L[LUR_U1L] = {&T.R12, 2, {R_U1L}, "rmsU1L"};
    L[LUR_U2H] = {&T.R11, 2, {R_U2H}, "rmsU2H"};
    L[LUR_U2L] = {&T.R12, 2, {R_U2L}, "rmsU2L"};
    L[LUR_NR]  = {&T.RM8, 2, {R_PDN, R_NR}, "rmsNR"};
    L[LUR_RSQ] = {&T.RSQT, 2, {R_U16, R_PP, R_MR, R_HB}, "rmsRSQ"};
    L[LUR_EDIF]= {&T.R512, 2, {R_EDIF}, "rmsEDIF"};
    L[LUR_QB]  = {&T.R512, 2, {R_QB}, "rmsQB"};
    L[LUR_TEB] = {&T.REXP, 2, {R_TEB}, "rmsTEB"};
    return L;
}

// ---------------- proof object ----------------
struct RmsProof {
    uint32_t B = 0, ld = 0;
    p3fri::Hash rde[NDE], rdb[NDB], rdw[NDW];
    p3lu::LookupProof lu[NLUR];
    gl_t yM1xmb = 0, yM1mr = 0, yM2gmb = 0;      // virtual-key bindings
    std::vector<Msg5> mDe; gl_t yDe[NDE] = {}; gl_t yDeX = 0, yDeY = 0;
    gl_t yDeE = 0, yDeTEB = 0, yDeGS = 0, yDeGEB = 0, yDeGZ = 0;
    std::vector<Msg5> mDb; gl_t yDb[NDB] = {};
    std::vector<Msg5> mDw; gl_t yDw[NDW] = {}; gl_t yDwG = 0;
    gl_t yBS = 0, yBF = 0;
    std::vector<Msg5> mBind; gl_t yBXZ = 0, yBQ = 0, yBSEL = 0;
    std::vector<p3bo::BatchProof> batches;
};

// ==================== prover ====================
static inline RmsProof prove(fs::Transcript& tr, const Wit& wt, const Tables& T,
                             const Art& a, const Operands& ops,
                             uint32_t R, uint32_t Q, bool strict = true) {
    const Dims& dm = wt.dm;
    RmsProof pf; pf.B = dm.B; pf.ld = dm.ld;

    uint32_t hdr[4] = {dm.B, dm.ld, (uint32_t)a.EMIN, (uint32_t)a.eps_bits};
    tr.absorb("rms-dims", hdr, sizeof hdr);
    const Table* tabs[15] = {&T.R128, &T.R256, &T.R512, &T.R11, &T.R12, &T.R16,
                             &T.SH512, &T.RM17, &T.RM8, &T.WD24, &T.EPSAT, &T.RSQT,
                             &T.MUL7, &T.REXP, nullptr};
    for (int i = 0; i < 14; i++) tr.absorb("rms-tab", tabs[i]->id.data(), 32);
    tr.absorb("rms-X", ops.X.root.data(), 32);
    tr.absorb("rms-G", ops.G.root.data(), 32);
    tr.absorb("rms-Y", ops.Y.root.data(), 32);

    p3bo::PLedger lg;
    std::deque<Col> lucols;

    std::vector<Col> CDe(NDE), CDb(NDB), CDw(NDW);
    for (int c = 0; c < NDE; c++) { CDe[c] = commit_col_nc(wt.de[c], R); pf.rde[c] = CDe[c].root; }
    for (int c = 0; c < NDB; c++) { CDb[c] = commit_col_nc(wt.db[c], R); pf.rdb[c] = CDb[c].root; }
    for (int c = 0; c < NDW; c++) { CDw[c] = commit_col_nc(wt.dw[c], R); pf.rdw[c] = CDw[c].root; }
    for (int c = 0; c < NDE; c++) tr.absorb("rms-ce", pf.rde[c].data(), 32);
    for (int c = 0; c < NDB; c++) tr.absorb("rms-cb", pf.rdb[c].data(), 32);
    for (int c = 0; c < NDW; c++) tr.absorb("rms-cw", pf.rdw[c].data(), 32);

    // -- lookups (fixed order); virtual MUL7 keys bound right after each --
    auto LD = lu_defs(T);
    for (int i = 0; i < NLUR; i++) {
        std::vector<LuColSpec> spec;
        for (int cid : LD[i].cols) {
            if (cid == -1) spec.push_back(LV(&wt.xmf));
            else if (cid == -2) spec.push_back(LV(&wt.mrf));
            else if (cid == -3) spec.push_back(LV(&wt.gmf));
            else spec.push_back(LC(LD[i].dom == 0 ? &CDe[cid] :
                                   LD[i].dom == 1 ? &CDw[cid] : &CDb[cid]));
        }
        std::vector<gl_t> rA;
        bool need_rA = (i == LUR_M1 || i == LUR_M2);
        pf.lu[i] = p3lu::prove_v(tr, spec, wt.lidx[i], *LD[i].tab, R, Q, LD[i].label,
                                 true, strict, need_rA ? &rA : nullptr, &lg, &lucols);
        if (i == LUR_M1) {
            pf.yM1xmb = claimc(tr, lg, CDe[D_XMB], rA);
            std::vector<gl_t> rb(rA.begin() + dm.ld, rA.end());
            pf.yM1mr = claimc(tr, lg, CDb[R_MR], rb);
        } else if (i == LUR_M2) {
            std::vector<gl_t> ri(rA.begin(), rA.begin() + dm.ld);
            pf.yM2gmb = claimc(tr, lg, CDw[W_GMB], ri);
        }
    }

    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, dm.le);
    gl_t lamE = chal(tr), lamEv[N_DE_C]; lamEv[0] = 1;
    for (int j = 1; j < N_DE_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(3 + NDE + 5);
        cols.push_back(beq(zE));
        cols.push_back(wt.xpat); cols.push_back(wt.ypat);
        for (int c = 0; c < NDE; c++) cols.push_back(wt.de[c]);
        std::vector<gl_t> Ebc(dm.Ne), TEBbc(dm.Ne), GSbc(dm.Ne), GEBbc(dm.Ne), GZbc(dm.Ne);
        for (size_t e = 0; e < dm.Ne; e++) {
            uint32_t b = (uint32_t)(e >> dm.ld), i = (uint32_t)(e & (dm.d - 1));
            Ebc[e] = wt.db[R_E][b]; TEBbc[e] = wt.db[R_TEB][b];
            GSbc[e] = wt.dw[W_GS][i]; GEBbc[e] = wt.dw[W_GEB][i]; GZbc[e] = wt.dw[W_GZ][i];
        }
        cols.push_back(std::move(Ebc)); cols.push_back(std::move(TEBbc));
        cols.push_back(std::move(GSbc)); cols.push_back(std::move(GEBbc));
        cols.push_back(std::move(GZbc));
        CFn F = [&](const gl_t* v) { return F_de(v, lamEv); };
        std::vector<gl_t> rE = sc5_prove(tr, "rms-scE", std::move(cols), F, pf.mDe);
        pf.yDeX = claimc(tr, lg, ops.X, rE);
        pf.yDeY = claimc(tr, lg, ops.Y, rE);
        for (int c = 0; c < NDE; c++) pf.yDe[c] = claimc(tr, lg, CDe[c], rE);
        std::vector<gl_t> rb(rE.begin() + dm.ld, rE.end());
        std::vector<gl_t> ri(rE.begin(), rE.begin() + dm.ld);
        pf.yDeE = claimc(tr, lg, CDb[R_E], rb);
        pf.yDeTEB = claimc(tr, lg, CDb[R_TEB], rb);
        pf.yDeGS = claimc(tr, lg, CDw[W_GS], ri);
        pf.yDeGEB = claimc(tr, lg, CDw[W_GEB], ri);
        pf.yDeGZ = claimc(tr, lg, CDw[W_GZ], ri);
    }

    // -- Db zero-check --
    std::vector<gl_t> zB = chal_vec(tr, dm.lb);
    gl_t lamB = chal(tr), lamBv[N_DB_C]; lamBv[0] = 1;
    for (int j = 1; j < N_DB_C; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(1 + NDB);
        cols.push_back(beq(zB));
        for (int c = 0; c < NDB; c++) cols.push_back(wt.db[c]);
        gl_t EMINc = (gl_t)a.EMIN, C284 = (gl_t)(284 + dm.ld);
        CFn F = [&](const gl_t* v) { return F_db(v, lamBv, EMINc, C284); };
        std::vector<gl_t> rB = sc5_prove(tr, "rms-scB", std::move(cols), F, pf.mDb);
        for (int c = 0; c < NDB; c++) pf.yDb[c] = claimc(tr, lg, CDb[c], rB);
    }

    // -- Dw zero-check (binds the G commitment to the decomposition) --
    std::vector<gl_t> zW = chal_vec(tr, dm.ld);
    gl_t lamW = chal(tr), lamWv[N_DW_C]; lamWv[0] = 1;
    for (int j = 1; j < N_DW_C; j++) lamWv[j] = gl_mul(lamWv[j-1], lamW);
    {
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(beq(zW)); cols.push_back(wt.gpat);
        for (int c = 0; c < NDW; c++) cols.push_back(wt.dw[c]);
        CFn F = [&](const gl_t* v) { return F_dw(v, lamWv); };
        std::vector<gl_t> rW = sc5_prove(tr, "rms-scW", std::move(cols), F, pf.mDw);
        pf.yDwG = claimc(tr, lg, ops.G, rW);
        for (int c = 0; c < NDW; c++) pf.yDw[c] = claimc(tr, lg, CDw[c], rW);
    }

    // -- row binding: sum_i (1-XZ)*Q = S and sum_i SEL = 1 - FSEL --
    std::vector<gl_t> z2 = chal_vec(tr, dm.lb);
    pf.yBS = claimc(tr, lg, CDb[R_S], z2);
    pf.yBF = claimc(tr, lg, CDb[R_FSEL], z2);
    gl_t lam1 = chal(tr), lam2 = chal(tr);
    {
        std::vector<gl_t> eqb = beq(z2);
        std::vector<gl_t> EQb(dm.Ne);
        for (size_t e = 0; e < dm.Ne; e++) EQb[e] = eqb[e >> dm.ld];
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(std::move(EQb));
        cols.push_back(wt.de[D_XZ]); cols.push_back(wt.de[D_Q]); cols.push_back(wt.de[D_SEL]);
        gl_t lam[2] = {lam1, lam2};
        CFn F = [&](const gl_t* v) { return F_bind(v, lam); };
        std::vector<gl_t> rS = sc5_prove(tr, "rms-scS", std::move(cols), F, pf.mBind);
        pf.yBXZ = claimc(tr, lg, CDe[D_XZ], rS);
        pf.yBQ = claimc(tr, lg, CDe[D_Q], rS);
        pf.yBSEL = claimc(tr, lg, CDe[D_SEL], rS);
    }

    // -- batched openings --
    for (size_t i = 0; i < lg.cls.size(); i++)
        pf.batches.push_back(p3bo::prove_class(tr, lg.cls[i], R, Q,
                                               "rms-bo" + std::to_string(i)));
    return pf;
}

// ==================== verifier ====================
// rX/rG/rY, B, ld, EMIN, eps_bits and Q/R are PUBLIC caller inputs.
static inline bool verify(fs::Transcript& tr, const Tables& T, const Art& a,
                          const RmsProof& pf, const p3fri::Hash& rX,
                          const p3fri::Hash& rG, const p3fri::Hash& rY,
                          uint32_t B, uint32_t ld,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.B != B || pf.ld != ld) return fail("dims mismatch");
    if ((uint32_t)a.ld != ld) return fail("artifact ld mismatch");
    Dims dm = make_dims(B, ld);
    p3bo::VLedger vlg;

    uint32_t hdr[4] = {B, ld, (uint32_t)a.EMIN, (uint32_t)a.eps_bits};
    tr.absorb("rms-dims", hdr, sizeof hdr);
    const Table* tabs[14] = {&T.R128, &T.R256, &T.R512, &T.R11, &T.R12, &T.R16,
                             &T.SH512, &T.RM17, &T.RM8, &T.WD24, &T.EPSAT, &T.RSQT,
                             &T.MUL7, &T.REXP};
    for (auto* t : tabs) tr.absorb("rms-tab", t->id.data(), 32);
    tr.absorb("rms-X", rX.data(), 32);
    tr.absorb("rms-G", rG.data(), 32);
    tr.absorb("rms-Y", rY.data(), 32);
    for (int c = 0; c < NDE; c++) tr.absorb("rms-ce", pf.rde[c].data(), 32);
    for (int c = 0; c < NDB; c++) tr.absorb("rms-cb", pf.rdb[c].data(), 32);
    for (int c = 0; c < NDW; c++) tr.absorb("rms-cw", pf.rdw[c].data(), 32);

    // -- lookups --
    auto LD = lu_defs(T);
    for (int i = 0; i < NLUR; i++) {
        uint32_t explog = LD[i].dom == 0 ? dm.le : LD[i].dom == 1 ? dm.ld : dm.lb;
        if (pf.lu[i].n != explog) return fail("lookup domain");
        std::vector<const p3fri::Hash*> roots;
        for (int cid : LD[i].cols) {
            if (cid < 0) roots.push_back(nullptr);
            else roots.push_back(LD[i].dom == 0 ? &pf.rde[cid] :
                                 LD[i].dom == 1 ? &pf.rdw[cid] : &pf.rdb[cid]);
        }
        std::vector<gl_t> rA, yv;
        if (!p3lu::verify_v(tr, roots, *LD[i].tab, pf.lu[i], Q_pub, R_pub, LD[i].label,
                            why, &rA, &yv, nullptr, &vlg)) return false;
        if (i == LUR_M1) {
            if (yv.size() != 2) return fail("M1 y_virt count");
            gl_t yxmb = claimv(tr, vlg, pf.rde[D_XMB], rA, pf.yM1xmb);
            std::vector<gl_t> rb(rA.begin() + ld, rA.end());
            gl_t ymr = claimv(tr, vlg, pf.rdb[R_MR], rb, pf.yM1mr);
            if (yv[0] != gl_add(128ULL, yxmb)) return fail("M1 xmf binding");
            if (yv[1] != ymr) return fail("M1 mrf binding");
        } else if (i == LUR_M2) {
            if (yv.size() != 1) return fail("M2 y_virt count");
            std::vector<gl_t> ri(rA.begin(), rA.begin() + ld);
            gl_t ygmb = claimv(tr, vlg, pf.rdw[W_GMB], ri, pf.yM2gmb);
            if (yv[0] != gl_add(128ULL, ygmb)) return fail("M2 gmf binding");
        }
    }

    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, dm.le);
    gl_t lamE = chal(tr), lamEv[N_DE_C]; lamEv[0] = 1;
    for (int j = 1; j < N_DE_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.mDe, dm.le, 0, tr, "rms-scE", rE, claim)) return fail("De sumcheck");
        gl_t v[3 + NDE + 5]; v[0] = p3bf::eq_point(rE, zE);
        v[1] = claimv(tr, vlg, rX, rE, pf.yDeX);
        v[2] = claimv(tr, vlg, rY, rE, pf.yDeY);
        for (int c = 0; c < NDE; c++) v[3 + c] = claimv(tr, vlg, pf.rde[c], rE, pf.yDe[c]);
        std::vector<gl_t> rb(rE.begin() + ld, rE.end());
        std::vector<gl_t> ri(rE.begin(), rE.begin() + ld);
        v[3 + NDE] = claimv(tr, vlg, pf.rdb[R_E], rb, pf.yDeE);
        v[4 + NDE] = claimv(tr, vlg, pf.rdb[R_TEB], rb, pf.yDeTEB);
        v[5 + NDE] = claimv(tr, vlg, pf.rdw[W_GS], ri, pf.yDeGS);
        v[6 + NDE] = claimv(tr, vlg, pf.rdw[W_GEB], ri, pf.yDeGEB);
        v[7 + NDE] = claimv(tr, vlg, pf.rdw[W_GZ], ri, pf.yDeGZ);
        if (F_de(v, lamEv) != claim) return fail("De terminal");
    }

    // -- Db zero-check --
    std::vector<gl_t> zB = chal_vec(tr, dm.lb);
    gl_t lamB = chal(tr), lamBv[N_DB_C]; lamBv[0] = 1;
    for (int j = 1; j < N_DB_C; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        std::vector<gl_t> rB; gl_t claim;
        if (!sc5_verify(pf.mDb, dm.lb, 0, tr, "rms-scB", rB, claim)) return fail("Db sumcheck");
        gl_t v[1 + NDB]; v[0] = p3bf::eq_point(rB, zB);
        for (int c = 0; c < NDB; c++) v[1 + c] = claimv(tr, vlg, pf.rdb[c], rB, pf.yDb[c]);
        if (F_db(v, lamBv, (gl_t)a.EMIN, (gl_t)(284 + ld)) != claim)
            return fail("Db terminal");
    }

    // -- Dw zero-check --
    std::vector<gl_t> zW = chal_vec(tr, dm.ld);
    gl_t lamW = chal(tr), lamWv[N_DW_C]; lamWv[0] = 1;
    for (int j = 1; j < N_DW_C; j++) lamWv[j] = gl_mul(lamWv[j-1], lamW);
    {
        std::vector<gl_t> rW; gl_t claim;
        if (!sc5_verify(pf.mDw, dm.ld, 0, tr, "rms-scW", rW, claim)) return fail("Dw sumcheck");
        gl_t v[2 + NDW]; v[0] = p3bf::eq_point(rW, zW);
        v[1] = claimv(tr, vlg, rG, rW, pf.yDwG);
        for (int c = 0; c < NDW; c++) v[2 + c] = claimv(tr, vlg, pf.rdw[c], rW, pf.yDw[c]);
        if (F_dw(v, lamWv) != claim) return fail("Dw terminal");
    }

    // -- row binding --
    std::vector<gl_t> z2 = chal_vec(tr, dm.lb);
    {
        gl_t yS = claimv(tr, vlg, pf.rdb[R_S], z2, pf.yBS);
        gl_t yF = claimv(tr, vlg, pf.rdb[R_FSEL], z2, pf.yBF);
        gl_t lam1 = chal(tr), lam2 = chal(tr);
        gl_t claim0 = gl_add(gl_mul(lam1, yS), gl_mul(lam2, gl_sub(1ULL, yF)));
        std::vector<gl_t> rS; gl_t claim;
        if (!sc5_verify(pf.mBind, dm.le, claim0, tr, "rms-scS", rS, claim))
            return fail("bind sumcheck");
        gl_t yXZ = claimv(tr, vlg, pf.rde[D_XZ], rS, pf.yBXZ);
        gl_t yQ = claimv(tr, vlg, pf.rde[D_Q], rS, pf.yBQ);
        gl_t ySEL = claimv(tr, vlg, pf.rde[D_SEL], rS, pf.yBSEL);
        std::vector<gl_t> rSb(rS.begin() + ld, rS.end());
        gl_t end = gl_mul(p3bf::eq_point(rSb, z2),
                          gl_add(gl_mul(lam1, gl_mul(gl_sub(1ULL, yXZ), yQ)),
                                 gl_mul(lam2, ySEL)));
        if (end != claim) return fail("bind terminal");
    }

    // -- batched openings --
    if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
    for (size_t i = 0; i < vlg.cls.size(); i++)
        if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                "rms-bo" + std::to_string(i), why)) return false;

    if (why) *why = "ok";
    return true;
}

// ---------------- proof size (serialized-payload bytes) ----------------
static inline size_t proof_size(const RmsProof& pf) {
    size_t s = 8 + (NDE + NDB + NDW) * 32;
    for (int i = 0; i < NLUR; i++) s += p3hwl::sz_lu(pf.lu[i]);
    auto msgs = [&](const std::vector<Msg5>& m) { return m.size() * 40; };
    s += msgs(pf.mDe) + msgs(pf.mDb) + msgs(pf.mDw) + msgs(pf.mBind);
    s += 8 * (3 + NDE + 7 + NDB + NDW + 1 + 5);
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3rms
