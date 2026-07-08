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
    // S' window is d-dependent: S < 2^(ld+16) and EPSA < 2^18 give
    // S' < 2^WMAX, WMAX = 17 + ld (== the historical 23 at ld = 6, so the
    // ld=6 tables and transcripts are bit-identical to the fixed-window build).
    const uint32_t WMAX = 17 + (uint32_t)a.ld;
    if (WMAX > 30) throw std::runtime_error("rms: ld too large (WMAX > 30)");
    auto range = [](uint32_t n) { std::vector<gl_t> v(n);
        for (uint32_t j = 0; j < n; j++) v[j] = j; return make_table({v}); };
    T.R128 = range(128); T.R256 = range(256); T.R512 = range(512);
    T.R11 = range(1u << (WMAX - 12));          // U1H/U2H high limb (low limb 12b)
    T.R12 = range(4096); T.R16 = range(65536);
    { std::vector<gl_t> s(512), p(512);
      for (uint32_t j = 0; j < 512; j++) { s[j] = j; p[j] = 1ULL << (j < 16 ? j : 16); }
      T.SH512 = make_table({s, p}); }
    { std::vector<gl_t> p(131072, 1), r(131072, 0); uint32_t row = 0;   // r < pw <= 2^16
      for (uint32_t t = 0; t <= 16; t++)
          for (uint32_t x = 0; x < (1u << t); x++) { p[row] = 1ULL << t; r[row] = x; row++; }
      T.RM17 = make_table({p, r}); }
    { uint32_t tmax = WMAX - 16;                                // r < pw <= 2^(WMAX-16)
      std::vector<gl_t> p(1u << (tmax + 1), 1), r(1u << (tmax + 1), 0); uint32_t row = 0;
      for (uint32_t t = 0; t <= tmax; t++)
          for (uint32_t x = 0; x < (1u << t); x++) { p[row] = 1ULL << t; r[row] = x; row++; }
      T.RM8 = make_table({p, r}); }
    { std::vector<gl_t> w(32), plo(32), phi(32), pdn(32), pup(32);
      for (uint32_t j = 0; j < 32; j++) {
          uint32_t wd = j <= WMAX ? j : 0;
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
        if (Sp < 1 || Sp >= ((int64_t)1 << (17 + a.ld)))
            throw std::runtime_error("rms: S' window");
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
    std::vector<p3lu::GroupProof> lug;           // standalone merged lookup groups
    // virtual-key binding claims ride the group members' `extra` slots
    std::vector<Msg5> mDe; gl_t yDe[NDE] = {}; gl_t yDeX = 0, yDeY = 0;
    gl_t yDeE = 0, yDeTEB = 0, yDeGS = 0, yDeGEB = 0, yDeGZ = 0;
    std::vector<Msg5> mDb; gl_t yDb[NDB] = {};
    std::vector<Msg5> mDw; gl_t yDw[NDW] = {}; gl_t yDwG = 0;
    gl_t yBS = 0, yBF = 0;
    std::vector<Msg5> mBind; gl_t yBXZ = 0, yBQ = 0, yBSEL = 0;
    p3zkc::Blind zbl[4];                         // zk: De, Db, Dw, bind blinds
    std::vector<p3bo::BatchProof> batches;
};

// ==================== prover ====================
static inline RmsProof prove(fs::Transcript& tr, const Wit& wt, const Tables& T,
                             const Art& a, const Operands& ops,
                             uint32_t R, uint32_t Q, bool strict = true,
                             p3lu::XCtx* xc = nullptr) {
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

    p3lu::XCtx xc_loc;
    p3lu::XCtx& XC = xc ? *xc : xc_loc;
    p3bo::PLedger& lg = XC.lg;
    std::deque<Col>& lucols = XC.keep;

    std::vector<Col>& CDe = XC.vec(NDE);
    std::vector<Col>& CDb = XC.vec(NDB);
    std::vector<Col>& CDw = XC.vec(NDW);
    // zk: the row binding sum_i (1-XZ)*Q = S and sum_i SEL = 1-FSEL are CLAIM
    // algebra -> derive the row columns' slice-1 masks by the same row formulas
    const bool zk = p3zkc::G.on;
    std::vector<gl_t> mXZ, mQz, mSEL, mS, mFSEL;
    if (zk) {
        mXZ = p3zkc::fresh_mask(dm.le); mQz = p3zkc::fresh_mask(dm.le);
        mSEL = p3zkc::fresh_mask(dm.le);
        mS = p3zkc::fresh_mask(dm.lb); mFSEL = p3zkc::fresh_mask(dm.lb);
        for (uint32_t b = 0; b < dm.Bpad; b++) {
            gl_t sq = 0, sl = 0;
            for (uint32_t i = 0; i < dm.d; i++) {
                size_t e = ((size_t)b << dm.ld) | i;
                sq = gl_add(sq, gl_mul(gl_sub(1ULL, mXZ[e]), mQz[e]));
                sl = gl_add(sl, mSEL[e]);
            }
            mS[b] = sq;
            mFSEL[b] = gl_sub(1ULL, sl);
        }
    }
    for (int c = 0; c < NDE; c++) {
        const std::vector<gl_t>* m = nullptr;
        if (zk) { if (c == D_XZ) m = &mXZ; else if (c == D_Q) m = &mQz;
                  else if (c == D_SEL) m = &mSEL; }
        CDe[c] = commit_col_nc(wt.de[c], R, m); pf.rde[c] = CDe[c].root;
    }
    for (int c = 0; c < NDB; c++) {
        const std::vector<gl_t>* m = nullptr;
        if (zk) { if (c == R_S) m = &mS; else if (c == R_FSEL) m = &mFSEL; }
        CDb[c] = commit_col_nc(wt.db[c], R, m); pf.rdb[c] = CDb[c].root;
    }
    for (int c = 0; c < NDW; c++) { CDw[c] = commit_col_nc(wt.dw[c], R); pf.rdw[c] = CDw[c].root; }
    for (int c = 0; c < NDE; c++) tr.absorb("rms-ce", pf.rde[c].data(), 32);
    for (int c = 0; c < NDB; c++) tr.absorb("rms-cb", pf.rdb[c].data(), 32);
    for (int c = 0; c < NDW; c++) tr.absorb("rms-cw", pf.rdw[c].data(), 32);

    // -- lookups (fixed order); virtual MUL7 keys bound right after each --
    // zk: virtual keys rebuilt over the augmented domain (affine/broadcast of
    // committed columns -- exact, so the bindings hold at augmented points)
    const std::vector<gl_t> *pxm = &wt.xmf, *pmr = &wt.mrf, *pgm = &wt.gmf;
    if (zk) {
        std::vector<gl_t> xmf_z = CDe[D_XMB].v; for (auto& x : xmf_z) x = gl_add(x, 128ULL);
        std::vector<gl_t> gmf_z = p3zkc::bc_aug(CDw[W_GMB].v, dm.ld, dm.le, dm.Ne,
                              [&](size_t e) { return e & (dm.d - 1); });
        for (auto& x : gmf_z) x = gl_add(x, 128ULL);
        pxm = &XC.varr(std::move(xmf_z));
        pmr = &XC.varr(p3zkc::bc_aug(CDb[R_MR].v, dm.lb, dm.le, dm.Ne,
                              [&](size_t e) { return e >> dm.ld; }));
        pgm = &XC.varr(std::move(gmf_z));
    }
    auto LD = lu_defs(T);
    for (int i = 0; i < NLUR; i++) {
        std::vector<LuColSpec> spec;
        for (int cid : LD[i].cols) {
            if (cid == -1) spec.push_back(LV(pxm));
            else if (cid == -2) spec.push_back(LV(pmr));
            else if (cid == -3) spec.push_back(LV(pgm));
            else spec.push_back(LC(LD[i].dom == 0 ? &CDe[cid] :
                                   LD[i].dom == 1 ? &CDw[cid] : &CDb[cid]));
        }
        p3lu::PBind bind;
        if (i == LUR_M1) {
            const Col *xmb = &CDe[D_XMB], *mr = &CDb[R_MR];
            Dims dmc = dm;
            bind = [xmb, mr, dmc](fs::Transcript& trb, p3lu::XCtx& xcb,
                                  const std::vector<gl_t>& pm, const std::vector<gl_t>&) {
                gl_t y1 = claimc(trb, xcb.lg, *xmb, pm);
                std::vector<gl_t> rb(pm.begin() + dmc.ld, pm.begin() + dmc.le);
                gl_t y2 = claimc(trb, xcb.lg, *mr, p3zkc::expt(rb, pm, dmc.le));
                return std::vector<gl_t>{y1, y2};
            };
        } else if (i == LUR_M2) {
            const Col* gmb = &CDw[W_GMB];
            Dims dmc = dm;
            bind = [gmb, dmc](fs::Transcript& trb, p3lu::XCtx& xcb,
                              const std::vector<gl_t>& pm, const std::vector<gl_t>&) {
                std::vector<gl_t> ri(pm.begin(), pm.begin() + dmc.ld);
                gl_t y = claimc(trb, xcb.lg, *gmb, p3zkc::expt(ri, pm, dmc.le));
                return std::vector<gl_t>{y};
            };
        }
        p3lu::defer_v(XC, std::move(spec), wt.lidx[i], *LD[i].tab, LD[i].label, std::move(bind));
    }

    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, dm.le);
    gl_t lamE = chal(tr), lamEv[N_DE_C]; lamEv[0] = 1;
    for (int j = 1; j < N_DE_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(3 + NDE + 5);
        cols.push_back(beq(p3zkc::zpt(zE)));
        cols.push_back(ops.X.v); cols.push_back(ops.Y.v);
        for (int c = 0; c < NDE; c++) cols.push_back(CDe[c].v);
        auto bmap = [&](size_t e) { return e >> dm.ld; };
        auto imap = [&](size_t e) { return e & (dm.d - 1); };
        cols.push_back(p3zkc::bc_aug(CDb[R_E].v, dm.lb, dm.le, dm.Ne, bmap));
        cols.push_back(p3zkc::bc_aug(CDb[R_TEB].v, dm.lb, dm.le, dm.Ne, bmap));
        cols.push_back(p3zkc::bc_aug(CDw[W_GS].v, dm.ld, dm.le, dm.Ne, imap));
        cols.push_back(p3zkc::bc_aug(CDw[W_GEB].v, dm.ld, dm.le, dm.Ne, imap));
        cols.push_back(p3zkc::bc_aug(CDw[W_GZ].v, dm.ld, dm.le, dm.Ne, imap));
        CFn F = [&](const gl_t* v) { return F_de(v, lamEv); };
        std::vector<gl_t> rE = p3hwl::sc5z(tr, "rms-scE", dm.le, std::move(cols), F, pf.mDe,
                                           0, R, lg, lucols, pf.zbl[0]);
        pf.yDeX = claimc(tr, lg, ops.X, rE);
        pf.yDeY = claimc(tr, lg, ops.Y, rE);
        for (int c = 0; c < NDE; c++) pf.yDe[c] = claimc(tr, lg, CDe[c], rE);
        std::vector<gl_t> rb(rE.begin() + dm.ld, rE.begin() + dm.le);
        std::vector<gl_t> ri(rE.begin(), rE.begin() + dm.ld);
        pf.yDeE = claimc(tr, lg, CDb[R_E], p3zkc::expt(rb, rE, dm.le));
        pf.yDeTEB = claimc(tr, lg, CDb[R_TEB], p3zkc::expt(rb, rE, dm.le));
        pf.yDeGS = claimc(tr, lg, CDw[W_GS], p3zkc::expt(ri, rE, dm.le));
        pf.yDeGEB = claimc(tr, lg, CDw[W_GEB], p3zkc::expt(ri, rE, dm.le));
        pf.yDeGZ = claimc(tr, lg, CDw[W_GZ], p3zkc::expt(ri, rE, dm.le));
    }

    // -- Db zero-check --
    std::vector<gl_t> zB = chal_vec(tr, dm.lb);
    gl_t lamB = chal(tr), lamBv[N_DB_C]; lamBv[0] = 1;
    for (int j = 1; j < N_DB_C; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(1 + NDB);
        cols.push_back(beq(p3zkc::zpt(zB)));
        for (int c = 0; c < NDB; c++) cols.push_back(CDb[c].v);
        gl_t EMINc = (gl_t)a.EMIN, C284 = (gl_t)(284 + dm.ld);
        CFn F = [&](const gl_t* v) { return F_db(v, lamBv, EMINc, C284); };
        std::vector<gl_t> rB = p3hwl::sc5z(tr, "rms-scB", dm.lb, std::move(cols), F, pf.mDb,
                                           0, R, lg, lucols, pf.zbl[1]);
        for (int c = 0; c < NDB; c++) pf.yDb[c] = claimc(tr, lg, CDb[c], rB);
    }

    // -- Dw zero-check (binds the G commitment to the decomposition) --
    std::vector<gl_t> zW = chal_vec(tr, dm.ld);
    gl_t lamW = chal(tr), lamWv[N_DW_C]; lamWv[0] = 1;
    for (int j = 1; j < N_DW_C; j++) lamWv[j] = gl_mul(lamWv[j-1], lamW);
    {
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(beq(p3zkc::zpt(zW))); cols.push_back(ops.G.v);
        for (int c = 0; c < NDW; c++) cols.push_back(CDw[c].v);
        CFn F = [&](const gl_t* v) { return F_dw(v, lamWv); };
        std::vector<gl_t> rW = p3hwl::sc5z(tr, "rms-scW", dm.ld, std::move(cols), F, pf.mDw,
                                           0, R, lg, lucols, pf.zbl[2]);
        pf.yDwG = claimc(tr, lg, ops.G, rW);
        for (int c = 0; c < NDW; c++) pf.yDw[c] = claimc(tr, lg, CDw[c], rW);
    }

    // -- row binding: sum_i (1-XZ)*Q = S and sum_i SEL = 1 - FSEL --
    std::vector<gl_t> z2 = chal_vec(tr, dm.lb);
    gl_t zexr = zk ? chal(tr) : 0;
    pf.yBS = claimc(tr, lg, CDb[R_S], p3zkc::xpt(z2, zexr));
    pf.yBF = claimc(tr, lg, CDb[R_FSEL], p3zkc::xpt(z2, zexr));
    gl_t lam1 = chal(tr), lam2 = chal(tr);
    {
        uint32_t e_e = p3zkc::e_of(dm.le);
        std::vector<gl_t> ptb = z2;
        if (zk) { ptb.push_back(zexr); ptb.resize(dm.lb + e_e, 0); }
        std::vector<gl_t> eqb = beq(ptb);
        size_t NeA = CDe[D_XZ].v.size();
        std::vector<gl_t> EQb(NeA);
        for (size_t q = 0; q < NeA; q++) {
            size_t ex = q >> dm.le, e = q & (dm.Ne - 1);
            EQb[q] = eqb[(ex << dm.lb) | (e >> dm.ld)];
        }
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(std::move(EQb));
        cols.push_back(CDe[D_XZ].v); cols.push_back(CDe[D_Q].v); cols.push_back(CDe[D_SEL].v);
        gl_t lam[2] = {lam1, lam2};
        CFn F = [&](const gl_t* v) { return F_bind(v, lam); };
        gl_t base0 = gl_add(gl_mul(lam1, pf.yBS), gl_mul(lam2, gl_sub(1ULL, pf.yBF)));
        std::vector<gl_t> rS = p3hwl::sc5z(tr, "rms-scS", dm.le, std::move(cols), F, pf.mBind,
                                           base0, R, lg, lucols, pf.zbl[3]);
        pf.yBXZ = claimc(tr, lg, CDe[D_XZ], rS);
        pf.yBQ = claimc(tr, lg, CDe[D_Q], rS);
        pf.yBSEL = claimc(tr, lg, CDe[D_SEL], rS);
    }

    // -- merged lookup flush + batched openings (standalone only) --
    if (!xc) {
        pf.lug = p3lu::lu_flush(tr, XC, R, Q, strict);
        for (size_t i = 0; i < lg.cls.size(); i++)
            pf.batches.push_back(p3bo::prove_class(tr, lg.cls[i], R, Q,
                                                   "rms-bo" + std::to_string(i)));
    }
    return pf;
}

// ==================== verifier ====================
// rX/rG/rY, B, ld, EMIN, eps_bits and Q/R are PUBLIC caller inputs.
static inline bool verify(fs::Transcript& tr, const Tables& T, const Art& a,
                          const RmsProof& pf, const p3fri::Hash& rX,
                          const p3fri::Hash& rG, const p3fri::Hash& rY,
                          uint32_t B, uint32_t ld,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr,
                          p3lu::VCtx* xv = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.B != B || pf.ld != ld) return fail("dims mismatch");
    if ((uint32_t)a.ld != ld) return fail("artifact ld mismatch");
    Dims dm = make_dims(B, ld);
    p3lu::VCtx vc_loc;
    p3lu::VCtx& VC = xv ? *xv : vc_loc;
    p3bo::VLedger& vlg = VC.vlg;

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

    // -- lookups: DEFERRED to the ledger owner's merged-group flush --
    auto LD = lu_defs(T);
    for (int i = 0; i < NLUR; i++) {
        uint32_t explog = LD[i].dom == 0 ? dm.le : LD[i].dom == 1 ? dm.ld : dm.lb;
        std::vector<const p3fri::Hash*> roots;
        for (int cid : LD[i].cols) {
            if (cid < 0) roots.push_back(nullptr);
            else roots.push_back(LD[i].dom == 0 ? &pf.rde[cid] :
                                 LD[i].dom == 1 ? &pf.rdw[cid] : &pf.rdb[cid]);
        }
        p3lu::VBind bind;
        if (i == LUR_M1) {
            p3fri::Hash hx = pf.rde[D_XMB], hm = pf.rdb[R_MR]; Dims dmc = dm;
            bind = [hx, hm, dmc](fs::Transcript& trb, p3lu::VCtx& vc,
                                 const std::vector<gl_t>& pm, const std::vector<gl_t>& yv,
                                 const std::vector<gl_t>& ex, const char** wy) {
                auto f = [&](const char* m) { if (wy) *wy = m; return false; };
                if (yv.size() != 2) return f("M1 y_virt count");
                if (ex.size() != 2) return f("M1 extra count");
                gl_t yxmb = claimv(trb, vc.vlg, hx, pm, ex[0]);
                std::vector<gl_t> rb(pm.begin() + dmc.ld, pm.begin() + dmc.le);
                gl_t ymr = claimv(trb, vc.vlg, hm, p3zkc::expt(rb, pm, dmc.le), ex[1]);
                if (yv[0] != gl_add(128ULL, yxmb)) return f("M1 xmf binding");
                if (yv[1] != ymr) return f("M1 mrf binding");
                return true;
            };
        } else if (i == LUR_M2) {
            p3fri::Hash hg = pf.rdw[W_GMB]; Dims dmc = dm;
            bind = [hg, dmc](fs::Transcript& trb, p3lu::VCtx& vc,
                             const std::vector<gl_t>& pm, const std::vector<gl_t>& yv,
                             const std::vector<gl_t>& ex, const char** wy) {
                auto f = [&](const char* m) { if (wy) *wy = m; return false; };
                if (yv.size() != 1) return f("M2 y_virt count");
                if (ex.size() != 1) return f("M2 extra count");
                std::vector<gl_t> ri(pm.begin(), pm.begin() + dmc.ld);
                gl_t ygmb = claimv(trb, vc.vlg, hg, p3zkc::expt(ri, pm, dmc.le), ex[0]);
                if (yv[0] != gl_add(128ULL, ygmb)) return f("M2 gmf binding");
                return true;
            };
        }
        p3lu::vdefer_v(VC, std::move(roots), *LD[i].tab, explog, LD[i].label, std::move(bind));
    }

    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, dm.le);
    gl_t lamE = chal(tr), lamEv[N_DE_C]; lamEv[0] = 1;
    for (int j = 1; j < N_DE_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.zbl[0]);
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.mDe, p3zkc::vfull(dm.le), gl_mul(rho, pf.zbl[0].H),
                        tr, "rms-scE", rE, claim)) return fail("De sumcheck");
        if (!p3hwl::sc5vz_claims(tr, vlg, pf.zbl[0], rE)) return fail("sc5 blind ip");
        gl_t v[3 + NDE + 5]; v[0] = p3bf::eq_point(rE, p3zkc::zpt(zE));
        v[1] = claimv(tr, vlg, rX, rE, pf.yDeX);
        v[2] = claimv(tr, vlg, rY, rE, pf.yDeY);
        for (int c = 0; c < NDE; c++) v[3 + c] = claimv(tr, vlg, pf.rde[c], rE, pf.yDe[c]);
        std::vector<gl_t> rb(rE.begin() + ld, rE.begin() + dm.le);
        std::vector<gl_t> ri(rE.begin(), rE.begin() + ld);
        v[3 + NDE] = claimv(tr, vlg, pf.rdb[R_E], p3zkc::expt(rb, rE, dm.le), pf.yDeE);
        v[4 + NDE] = claimv(tr, vlg, pf.rdb[R_TEB], p3zkc::expt(rb, rE, dm.le), pf.yDeTEB);
        v[5 + NDE] = claimv(tr, vlg, pf.rdw[W_GS], p3zkc::expt(ri, rE, dm.le), pf.yDeGS);
        v[6 + NDE] = claimv(tr, vlg, pf.rdw[W_GEB], p3zkc::expt(ri, rE, dm.le), pf.yDeGEB);
        v[7 + NDE] = claimv(tr, vlg, pf.rdw[W_GZ], p3zkc::expt(ri, rE, dm.le), pf.yDeGZ);
        gl_t end = gl_add(F_de(v, lamEv), p3hwl::sc5_blindterm(pf.zbl[0], rho, v[0]));
        if (end != claim) return fail("De terminal");
    }

    // -- Db zero-check --
    std::vector<gl_t> zB = chal_vec(tr, dm.lb);
    gl_t lamB = chal(tr), lamBv[N_DB_C]; lamBv[0] = 1;
    for (int j = 1; j < N_DB_C; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.zbl[1]);
        std::vector<gl_t> rB; gl_t claim;
        if (!sc5_verify(pf.mDb, p3zkc::vfull(dm.lb), gl_mul(rho, pf.zbl[1].H),
                        tr, "rms-scB", rB, claim)) return fail("Db sumcheck");
        if (!p3hwl::sc5vz_claims(tr, vlg, pf.zbl[1], rB)) return fail("sc5 blind ip");
        gl_t v[1 + NDB]; v[0] = p3bf::eq_point(rB, p3zkc::zpt(zB));
        for (int c = 0; c < NDB; c++) v[1 + c] = claimv(tr, vlg, pf.rdb[c], rB, pf.yDb[c]);
        gl_t end = gl_add(F_db(v, lamBv, (gl_t)a.EMIN, (gl_t)(284 + ld)),
                          p3hwl::sc5_blindterm(pf.zbl[1], rho, v[0]));
        if (end != claim)
            return fail("Db terminal");
    }

    // -- Dw zero-check --
    std::vector<gl_t> zW = chal_vec(tr, dm.ld);
    gl_t lamW = chal(tr), lamWv[N_DW_C]; lamWv[0] = 1;
    for (int j = 1; j < N_DW_C; j++) lamWv[j] = gl_mul(lamWv[j-1], lamW);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.zbl[2]);
        std::vector<gl_t> rW; gl_t claim;
        if (!sc5_verify(pf.mDw, p3zkc::vfull(dm.ld), gl_mul(rho, pf.zbl[2].H),
                        tr, "rms-scW", rW, claim)) return fail("Dw sumcheck");
        if (!p3hwl::sc5vz_claims(tr, vlg, pf.zbl[2], rW)) return fail("sc5 blind ip");
        gl_t v[2 + NDW]; v[0] = p3bf::eq_point(rW, p3zkc::zpt(zW));
        v[1] = claimv(tr, vlg, rG, rW, pf.yDwG);
        for (int c = 0; c < NDW; c++) v[2 + c] = claimv(tr, vlg, pf.rdw[c], rW, pf.yDw[c]);
        gl_t end = gl_add(F_dw(v, lamWv), p3hwl::sc5_blindterm(pf.zbl[2], rho, v[0]));
        if (end != claim) return fail("Dw terminal");
    }

    // -- row binding --
    std::vector<gl_t> z2 = chal_vec(tr, dm.lb);
    {
        const bool zk = p3zkc::G.on;
        gl_t zexr = zk ? chal(tr) : 0;
        gl_t yS = claimv(tr, vlg, pf.rdb[R_S], p3zkc::xpt(z2, zexr), pf.yBS);
        gl_t yF = claimv(tr, vlg, pf.rdb[R_FSEL], p3zkc::xpt(z2, zexr), pf.yBF);
        gl_t lam1 = chal(tr), lam2 = chal(tr);
        gl_t claim0 = gl_add(gl_mul(lam1, yS), gl_mul(lam2, gl_sub(1ULL, yF)));
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.zbl[3]);
        claim0 = gl_add(claim0, gl_mul(rho, pf.zbl[3].H));
        std::vector<gl_t> rS; gl_t claim;
        if (!sc5_verify(pf.mBind, p3zkc::vfull(dm.le), claim0, tr, "rms-scS", rS, claim))
            return fail("bind sumcheck");
        if (!p3hwl::sc5vz_claims(tr, vlg, pf.zbl[3], rS)) return fail("sc5 blind ip");
        gl_t yXZ = claimv(tr, vlg, pf.rde[D_XZ], rS, pf.yBXZ);
        gl_t yQ = claimv(tr, vlg, pf.rde[D_Q], rS, pf.yBQ);
        gl_t ySEL = claimv(tr, vlg, pf.rde[D_SEL], rS, pf.yBSEL);
        std::vector<gl_t> rSb(rS.begin() + ld, rS.begin() + dm.le);
        rSb.insert(rSb.end(), rS.begin() + dm.le, rS.end());
        std::vector<gl_t> ptb = z2;
        if (zk) { ptb.push_back(zexr); ptb.resize(dm.lb + p3zkc::e_of(dm.le), 0); }
        gl_t w = p3bf::eq_point(rSb, ptb);
        gl_t end = gl_mul(w, gl_add(gl_mul(lam1, gl_mul(gl_sub(1ULL, yXZ), yQ)),
                                    gl_mul(lam2, ySEL)));
        end = gl_add(end, p3hwl::sc5_blindterm(pf.zbl[3], rho, w));
        if (end != claim) return fail("bind terminal");
    }

    // -- merged lookup flush + batched openings (standalone only) --
    if (!xv) {
        if (!p3lu::lu_verify_flush(tr, VC, pf.lug, Q_pub, R_pub, why)) return false;
        if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
        for (size_t i = 0; i < vlg.cls.size(); i++)
            if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                    "rms-bo" + std::to_string(i), why)) return false;
    } else if (!pf.batches.empty() || !pf.lug.empty()) return fail("unexpected batches");

    if (why) *why = "ok";
    return true;
}

// ---------------- proof size (serialized-payload bytes) ----------------
static inline size_t proof_size(const RmsProof& pf) {
    size_t s = 8 + (NDE + NDB + NDW) * 32;
    for (auto& g : pf.lug) s += p3lu::sz_group(g);
    auto msgs = [&](const std::vector<Msg5>& m) { return m.size() * 40; };
    s += msgs(pf.mDe) + msgs(pf.mDb) + msgs(pf.mDw) + msgs(pf.mBind);
    s += 8 * (3 + NDE + 7 + NDB + NDW + 1 + 5);
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3rms
