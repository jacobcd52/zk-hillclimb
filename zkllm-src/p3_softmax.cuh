// Softmax gadget: sound (non-ZK) prover+verifier that a committed bf16 output
// P equals the CANONICAL masked softmax (transformer_ref.py softmax_rows) of a
// committed bf16 score matrix S, bitwise, for a PUBLIC structural mask (the
// causal mask; masked lanes output +0 and are EXCLUDED from max and sum).
//
//   per row b (n = 2^ln lanes, mask MSK public, EMP = row-all-masked public):
//     key_j = 32768 - s + (1-2s)(128*eb + mb)    [monotone in the bf16 total
//                                                 order, LINEAR in the fields]
//     KMAX  = max over participating keys        [dominance DK = KMAX-key >= 0
//                                                 via R16 + SEL1 attainment,
//                                                 SEL1 binds MXP = the max
//                                                 score pattern]
//     dp_j  = bf_add(MSK*S_j, MSK*neg(MXP))      [ONE p3_bfadd block; the
//                                                 negation is bound through
//                                                 the block's own operand-2
//                                                 decomposition]
//     EXV_j = EXP[dp_j]                          [pinned 65536-entry table,
//                                                 keyed directly on dp]
//     denom: block-float sum of lanes ((128+emb)<<8, eeb+8) over present =
//            MSK*(EXV nonzero), floor SM_EMIN=100 -- the RMSNorm S machinery
//            verbatim (dominance/attainment E, shift q*pw+r, row-bound sum,
//            pow2 sandwich wd/u16)
//     rcp   = RCP[u16] with biased exponent REB = 277 + hb - E - wd
//     P_j   = MSK*(1-EZ) * MUL7(EXV_j, rcp)      [masked and underflowed
//                                                 lanes -> +0]
//
// Supported domain (proof REJECTS otherwise -- sound, not complete): the
// subtract obeys the p3_bfadd v1 rule; the reciprocal exponent REB and the
// output exponents EO3 stay in [1,254].  Empty-mask rows (incl. batch padding)
// are in-domain and output all +0 (the public EMP bit gates the denominator
// normalization).
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
#include "p3_rmsnorm.cuh"
#include "p3_bfadd.cuh"
#include "fs_transcript.hpp"

namespace p3smx {

using p3lu::Col; using p3lu::Table; using p3lu::make_table; using p3lu::commit_col_nc;
using p3lu::chal; using p3lu::chal_vec; using p3lu::ilog2; using p3lu::bind_lsb;
using p3lu::LuColSpec; using p3lu::LC; using p3lu::LV;
using p3hwl::Msg5; using p3hwl::CFn; using p3hwl::sc5_prove; using p3hwl::sc5_verify;
using p3hwl::claimc; using p3hwl::claimv; using p3hwl::beq; using p3hwl::pow2_at_least;
using p3hwl::enc;
using p3rms::Art;

static inline gl_t mle_eval(std::vector<gl_t> f, const std::vector<gl_t>& r) {
    for (gl_t a : r) bind_lsb(f, a);
    return f[0];
}

// ---------------- columns ----------------
enum {  // De: per-element (e = (b << ln) | j)
    SM_XM = 0,   // MSK * score pattern (the subtract's x1)
    SM_X2M,      // MSK * neg(max) pattern (the subtract's x2)
    SM_DP,       // subtract output pattern
    SM_DK,       // KMAX - key (participating lanes)
    SM_SEL1,     // key-max attainment selector
    SM_EXV, SM_EEB, SM_EMB, SM_EZ, SM_EZI,      // EXP output + decomposition
    SM_SH2, SM_PW2, SM_Q2, SM_RR2, SM_SEL2,     // denominator lanes
    SM_MO3, SM_EI3, SM_EO3,                     // output multiply
    SM_FIX };
static const int SM_A = SM_FIX;                  // bfadd block base
static const int NDE2 = SM_FIX + p3bfa::NBA;
enum {  // Db: per-row
    B_KMAX = 0, B_MXP, B_E, B_EDIF, B_FSEL, B_S, B_WD, B_PLO, B_PHI, B_PDN,
    B_PUP, B_U1H, B_U1L, B_U2H, B_U2L, B_NR, B_U16, B_MRC, B_HBC, B_REB, NDB2 };
enum {  // lookups
    LSM_DK = 0, LSM_EXP, LSM_EEB, LSM_EMB, LSM_SH2, LSM_RM2, LSM_Q2,
    LSM_M3, LSM_EO3,
    LSM_EDIF, LSM_WD, LSM_U1H, LSM_U1L, LSM_U2H, LSM_U2L, LSM_NR, LSM_RCP, LSM_REB,
    LSM_FIX };
static const int LSM_A = LSM_FIX;                // 16 bfadd-block lookups
static const int NLSM = LSM_FIX + p3bfa::NBLU;

struct Tables {
    p3bfa::Tables BT;                            // R128/R256/R512/RM17/REXP/...
    Table R11, R12, R16, SH512, WD24, RM8;       // rmsnorm-style constructions
    Table MUL7, EXPT, RCPT;
    uint32_t wmax = 23;                          // max supported bitlen(S): rows
                                                 // of length n need wmax >= 16+ln
};
static inline Tables build_tables(const Art& a, uint32_t wmax = 23) {
    Tables T;
    // Denominator window is row-length dependent: S < 2^(16+ln) arithmetically
    // (16-bit lanes).  wmax = 23 keeps the historical tables (n <= 128)
    // bit-identical; pass wmax = 16 + ln for longer rows.
    if (wmax < 23 || wmax > 30) throw std::runtime_error("smx: wmax out of [23,30]");
    T.wmax = wmax;
    T.BT = p3bfa::build_tables();
    auto range = [](uint32_t n) { std::vector<gl_t> v(n);
        for (uint32_t j = 0; j < n; j++) v[j] = j; return make_table({v}); };
    T.R11 = range(1u << (wmax - 12));            // U1H/U2H high limb (low limb 12b)
    T.R12 = range(4096); T.R16 = range(65536);
    { std::vector<gl_t> s(512), p(512);
      for (uint32_t j = 0; j < 512; j++) { s[j] = j; p[j] = 1ULL << (j < 16 ? j : 16); }
      T.SH512 = make_table({s, p}); }
    { std::vector<gl_t> w(32), plo(32), phi(32), pdn(32), pup(32);
      for (uint32_t j = 0; j < 32; j++) {
          uint32_t wd = j <= wmax ? j : 0;
          w[j] = wd; plo[j] = wd ? (1ULL << (wd - 1)) : 0; phi[j] = 1ULL << wd;
          pdn[j] = 1ULL << (wd > 16 ? wd - 16 : 0);
          pup[j] = 1ULL << (wd < 16 ? 16 - wd : 0);
      }
      T.WD24 = make_table({w, plo, phi, pdn, pup}); }
    { uint32_t tmax = wmax - 16;                                // r < pw <= 2^(wmax-16)
      std::vector<gl_t> p(1u << (tmax + 1), 1), r(1u << (tmax + 1), 0); uint32_t row = 0;
      for (uint32_t t = 0; t <= tmax; t++)
          for (uint32_t x = 0; x < (1u << t); x++) { p[row] = 1ULL << t; r[row] = x; row++; }
      T.RM8 = make_table({p, r}); }
    { std::vector<gl_t> ma(16384), mb(16384), mo(16384), ei(16384);
      for (uint32_t j = 0; j < 16384; j++) {
          ma[j] = 128 + (j >> 7); mb[j] = 128 + (j & 127);
          mo[j] = 128 + a.mul_mo[j]; ei[j] = a.mul_einc[j];
      }
      T.MUL7 = make_table({ma, mb, mo, ei}); }
    { std::vector<gl_t> in(65536), out(65536);
      for (uint32_t j = 0; j < 65536; j++) { in[j] = j; out[j] = a.exp_tab[j]; }
      T.EXPT = make_table({in, out}); }
    { std::vector<gl_t> u(32768), m(32768), h(32768);
      for (uint32_t j = 0; j < 32768; j++) {
          u[j] = 32768 + j; m[j] = 128 + a.rcp_mr[j]; h[j] = a.rcp_hb[j];
      }
      T.RCPT = make_table({u, m, h}); }
    return T;
}

// ---------------- dims / golden ----------------
struct Dims {
    uint32_t B, n, ln;
    uint32_t Bpad, lb, le;
    size_t Ne;
};
static inline Dims make_dims(uint32_t B, uint32_t ln) {
    Dims dm; dm.B = B; dm.ln = ln; dm.n = 1u << ln;
    dm.Bpad = pow2_at_least(B < 2 ? 2 : B);
    dm.lb = ilog2(dm.Bpad); dm.le = dm.lb + ln;
    dm.Ne = (size_t)dm.Bpad << ln;
    return dm;
}
struct Golden {
    uint32_t B = 0, n = 0;
    std::vector<uint16_t> s, p;                  // B*n score / prob patterns
    std::vector<uint8_t> msk;                    // B*n mask bytes
};
static inline bool load_goldens(const char* path, std::vector<Golden>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[2];
    if (fread(hdr, 8, 2, f) != 2 || hdr[0] != 0x534D5847) { fclose(f); return false; }
    out.resize(hdr[1]);
    for (auto& G : out) {
        int64_t bn[2];
        if (fread(bn, 8, 2, f) != 2) { fclose(f); return false; }
        G.B = (uint32_t)bn[0]; G.n = (uint32_t)bn[1];
        size_t N = (size_t)G.B * G.n;
        G.s.resize(N); G.msk.resize(N); G.p.resize(N);
        if (fread(G.s.data(), 2, N, f) != N ||
            fread(G.msk.data(), 1, N, f) != N ||
            fread(G.p.data(), 2, N, f) != N) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

// ---------------- witness ----------------
enum { SMT_NONE = 0,
       SMT_MAX,       // rowmax attained at a WRONG participating lane
       SMT_EXP,       // exp table output EXV+1 (not an EXPT row)
       SMT_DENOM,     // denominator row sum S+1 (downstream honest)
       SMT_RCP,       // reciprocal mantissa MRC+1 (not an RCPT row)
       SMT_MASKLEAK };// a MASKED lane's weight leaks into the denominator
struct SmTamper { int mode = SMT_NONE; uint32_t b = 0, j = 0; };

struct Wit {
    Dims dm;
    std::vector<gl_t> spat, ppat;                // committed operand values
    std::vector<gl_t> mskc;                      // PUBLIC mask column (0/1)
    std::vector<gl_t> empc;                      // PUBLIC empty-row column
    std::vector<uint16_t> P;                     // computed outputs (B*n)
    std::vector<gl_t> de[NDE2], db[NDB2];
    std::vector<uint32_t> lidx[NLSM];
    std::vector<gl_t> emf, mrcf;                 // MUL7 virtual key columns
};
static inline gl_t inv_or0(uint64_t x) { return x ? gl_inv((gl_t)x) : 0; }
static inline int bitlen(uint64_t x) { int w = 0; while (x) { w++; x >>= 1; } return w; }
static inline int64_t key_of(uint32_t pat) {
    int64_t s = (pat >> 15) & 1, eb = (pat >> 7) & 255, mb = pat & 127;
    return 32768 - s + (1 - 2 * s) * (128 * eb + mb);
}

static inline Wit gen_witness(const Golden& L, const Art& a, const SmTamper* tm = nullptr) {
    Wit wt; Dims& dm = wt.dm;
    uint32_t ln = ilog2(L.n);
    if ((1u << ln) != L.n) throw std::runtime_error("smx: n must be pow2");
    dm = make_dims(L.B, ln);
    const uint32_t n = dm.n;
    const int64_t EMIN = a.sm_emin;

    wt.spat.assign(dm.Ne, 0); wt.ppat.assign(dm.Ne, 0);
    wt.mskc.assign(dm.Ne, 0); wt.empc.assign(dm.Bpad, 1);
    for (uint32_t b = 0; b < L.B; b++)
        for (uint32_t j = 0; j < n; j++) {
            wt.spat[((size_t)b << ln) | j] = L.s[(size_t)b * n + j];
            wt.mskc[((size_t)b << ln) | j] = L.msk[(size_t)b * n + j] ? 1 : 0;
        }
    wt.P.assign((size_t)L.B * n, 0);
    for (int c = 0; c < NDE2; c++) wt.de[c].assign(dm.Ne, 0);
    for (int c = 0; c < NDB2; c++) wt.db[c].assign(dm.Bpad, 0);
    for (int i = 0; i < NLSM; i++)
        wt.lidx[i].assign(i >= LSM_EDIF && i <= LSM_REB ? dm.Bpad : dm.Ne, 0);
    wt.emf.assign(dm.Ne, 0); wt.mrcf.assign(dm.Ne, 0);

    for (uint32_t b = 0; b < dm.Bpad; b++) {
        std::vector<int64_t> msk(n, 0);
        std::vector<uint32_t> sp(n, 0);
        for (uint32_t j = 0; j < n; j++) {
            size_t e = ((size_t)b << ln) | j;
            msk[j] = (int64_t)wt.mskc[e];
            sp[j] = (uint32_t)wt.spat[e];
        }
        // row max over participating keys
        int mx = -1;
        for (uint32_t j = 0; j < n; j++)
            if (msk[j] && (mx < 0 || key_of(sp[j]) > key_of(sp[mx]))) mx = (int)j;
        if (tm && tm->mode == SMT_MAX && b == tm->b) {
            int alt = -1;
            for (uint32_t j = 0; j < n; j++)
                if (msk[j] && (int)j != mx && key_of(sp[j]) < key_of(sp[mx]))
                    { alt = (int)j; break; }
            if (alt >= 0) mx = alt;              // forged max, honest downstream
        }
        int64_t emp = mx < 0 ? 1 : 0;
        wt.empc[b] = (gl_t)emp;
        uint32_t mxp = mx >= 0 ? sp[mx] : 0;
        int64_t kmax = mx >= 0 ? key_of(mxp) : 0;
        wt.db[B_KMAX][b] = (gl_t)kmax; wt.db[B_MXP][b] = (gl_t)mxp;
        uint32_t nmx = mxp ^ 0x8000;
        // subtract + exp + denominator lanes
        int64_t E = EMIN;
        std::vector<int64_t> exv(n), eeb(n), emb(n), ez(n), present(n);
        for (uint32_t j = 0; j < n; j++) {
            size_t e = ((size_t)b << ln) | j;
            uint32_t xm = msk[j] ? sp[j] : 0;
            uint32_t x2m = msk[j] ? nmx : 0;
            p3bfa::BaVals bv = p3bfa::ba_fill((uint16_t)xm, (uint16_t)x2m);
            for (int c = 0; c < p3bfa::NBA; c++) wt.de[SM_A + c][e] = bv.v[c];
            for (int i = 0; i < p3bfa::NBLU; i++) wt.lidx[LSM_A + i][e] = bv.lu[i];
            wt.de[SM_XM][e] = xm; wt.de[SM_X2M][e] = x2m;
            wt.de[SM_DP][e] = bv.out;
            int64_t dk = msk[j] ? kmax - key_of(sp[j]) : 0;
            if (dk < 0) { wt.de[SM_DK][e] = enc(dk); wt.lidx[LSM_DK][e] = 0; }
            else { wt.de[SM_DK][e] = (gl_t)dk; wt.lidx[LSM_DK][e] = (uint32_t)dk; }
            wt.de[SM_SEL1][e] = (mx == (int)j) ? 1 : 0;
            int64_t xv = a.exp_tab[bv.out];
            if (tm && tm->mode == SMT_EXP && b == tm->b && j == tm->j) xv += 1;
            exv[j] = xv;
            eeb[j] = (xv >> 7) & 255; emb[j] = xv & 127;
            if ((xv >> 15) & 1) throw std::runtime_error("smx: negative exp");
            ez[j] = eeb[j] == 0 ? 1 : 0;
            present[j] = (msk[j] && !ez[j]) ? 1 : 0;
            if (tm && tm->mode == SMT_MASKLEAK && b == tm->b && j == tm->j)
                present[j] = 1;                  // leak a masked lane's weight
            wt.de[SM_EXV][e] = (gl_t)xv;
            wt.de[SM_EEB][e] = (gl_t)eeb[j]; wt.de[SM_EMB][e] = (gl_t)emb[j];
            wt.de[SM_EZ][e] = (gl_t)ez[j]; wt.de[SM_EZI][e] = inv_or0((uint64_t)eeb[j]);
            wt.lidx[LSM_EXP][e] = (uint32_t)bv.out;
            wt.lidx[LSM_EEB][e] = (uint32_t)eeb[j];
            wt.lidx[LSM_EMB][e] = (uint32_t)emb[j];
            if (present[j] && eeb[j] + 8 > E) E = eeb[j] + 8;
        }
        int64_t fsel = 1; int sel2 = -1;
        for (uint32_t j = 0; j < n; j++)
            if (present[j] && eeb[j] + 8 == E) { sel2 = (int)j; fsel = 0; break; }
        int64_t edif = E - EMIN;
        int64_t S = 0;
        auto rm_idx = [](uint64_t pw, int64_t r) { return (uint32_t)((pw - 1) + (uint64_t)r); };
        for (uint32_t j = 0; j < n; j++) {
            size_t e = ((size_t)b << ln) | j;
            int64_t sh = 0, pw = 1, q = 0, rr = 0;
            if (present[j]) {
                sh = E - (eeb[j] + 8);
                int64_t shc = sh < 16 ? sh : 16;
                pw = (int64_t)1 << shc;
                int64_t sq = (128 + emb[j]) << 8;
                q = sq >> shc; rr = sq - (q << shc);
                S += q;
            }
            wt.de[SM_SH2][e] = (gl_t)sh; wt.de[SM_PW2][e] = (gl_t)pw;
            wt.de[SM_Q2][e] = (gl_t)q; wt.de[SM_RR2][e] = (gl_t)rr;
            wt.de[SM_SEL2][e] = (sel2 == (int)j) ? 1 : 0;
            wt.lidx[LSM_SH2][e] = (uint32_t)sh;
            wt.lidx[LSM_RM2][e] = rm_idx((uint64_t)pw, rr);
            wt.lidx[LSM_Q2][e] = (uint32_t)q;
        }
        if (tm && tm->mode == SMT_DENOM && b == tm->b) S += 1;
        // normalize + reciprocal (gated by the public EMP bit)
        int64_t wd = 1, plo = 1, phi = 2, pdn = 1, pup = 1 << 15;
        int64_t u1 = 0, u2 = 0, nr = 0, u16 = 32768, mrc = 128, hbc = 1, reb = 1;
        if (!emp) {
            int64_t wcap = 16 + (int64_t)ln < 30 ? 16 + (int64_t)ln : 30;
            if (wcap < 23) wcap = 23;
            if (S < 1 || S >= ((int64_t)1 << wcap)) throw std::runtime_error("smx: S window");
            wd = bitlen((uint64_t)S);
            plo = (int64_t)1 << (wd - 1); phi = (int64_t)1 << wd;
            pdn = (int64_t)1 << (wd > 16 ? wd - 16 : 0);
            pup = (int64_t)1 << (wd < 16 ? 16 - wd : 0);
            u1 = S - plo; u2 = phi - 1 - S;
            nr = wd > 16 ? S & (pdn - 1) : 0;
            u16 = wd > 16 ? S >> (wd - 16) : S << (16 - wd);
            uint32_t ri = (uint32_t)(u16 - 32768);
            mrc = 128 + a.rcp_mr[ri]; hbc = a.rcp_hb[ri];
            if (tm && tm->mode == SMT_RCP && b == tm->b) mrc += (mrc < 255 ? 1 : -1);
            reb = 277 + hbc - E - wd;
            if (reb < 1 || reb > 254) throw std::runtime_error("smx: recip exp domain");
        } else {
            hbc = 1;                             // benign RCPT row 0 (u16=32768)
        }
        wt.db[B_E][b] = (gl_t)E; wt.db[B_EDIF][b] = (gl_t)edif;
        wt.db[B_FSEL][b] = (gl_t)fsel; wt.db[B_S][b] = (gl_t)S;
        wt.db[B_WD][b] = (gl_t)wd; wt.db[B_PLO][b] = (gl_t)plo;
        wt.db[B_PHI][b] = (gl_t)phi; wt.db[B_PDN][b] = (gl_t)pdn;
        wt.db[B_PUP][b] = (gl_t)pup;
        wt.db[B_U1H][b] = (gl_t)(u1 >> 12); wt.db[B_U1L][b] = (gl_t)(u1 & 4095);
        wt.db[B_U2H][b] = (gl_t)(u2 >> 12); wt.db[B_U2L][b] = (gl_t)(u2 & 4095);
        wt.db[B_NR][b] = (gl_t)nr; wt.db[B_U16][b] = (gl_t)u16;
        wt.db[B_MRC][b] = (gl_t)mrc; wt.db[B_HBC][b] = (gl_t)hbc;
        wt.db[B_REB][b] = (gl_t)reb;
        wt.lidx[LSM_EDIF][b] = (uint32_t)edif;
        wt.lidx[LSM_WD][b] = (uint32_t)wd;
        wt.lidx[LSM_U1H][b] = (uint32_t)(u1 >> 12); wt.lidx[LSM_U1L][b] = (uint32_t)(u1 & 4095);
        wt.lidx[LSM_U2H][b] = (uint32_t)(u2 >> 12); wt.lidx[LSM_U2L][b] = (uint32_t)(u2 & 4095);
        wt.lidx[LSM_NR][b] = rm_idx((uint64_t)pdn, nr);
        wt.lidx[LSM_RCP][b] = (uint32_t)(u16 - 32768);
        wt.lidx[LSM_REB][b] = (uint32_t)reb;
        // output multiplies
        for (uint32_t j = 0; j < n; j++) {
            size_t e = ((size_t)b << ln) | j;
            uint32_t mj = (uint32_t)((emb[j] << 7) | (mrc - 128));
            int64_t mo = 128 + a.mul_mo[mj], ei = a.mul_einc[mj];
            int64_t outp = msk[j] && !ez[j];
            int64_t eo = 1;
            if (outp) {
                eo = eeb[j] + reb - 127 + ei;
                if (eo < 1 || eo > 254) throw std::runtime_error("smx: out exp domain");
            }
            int64_t p = outp ? ((eo << 7) | (mo - 128)) : 0;
            wt.de[SM_MO3][e] = (gl_t)mo; wt.de[SM_EI3][e] = (gl_t)ei;
            wt.de[SM_EO3][e] = (gl_t)eo;
            wt.emf[e] = (gl_t)(128 + emb[j]); wt.mrcf[e] = (gl_t)mrc;
            wt.lidx[LSM_M3][e] = mj; wt.lidx[LSM_EO3][e] = (uint32_t)eo;
            wt.ppat[e] = (gl_t)p;
            if (b < L.B) wt.P[(size_t)b * n + j] = (uint16_t)p;
        }
    }
    return wt;
}

struct Operands { Col S, P; };
static inline Operands commit_operands(const Wit& wt, uint32_t R) {
    Operands ops;
    ops.S = commit_col_nc(wt.spat, R);
    ops.P = commit_col_nc(wt.ppat, R);
    return ops;
}

// ---------------- constraints ----------------
// De: v = [E, Sv, Pv, MSK, de(NDE2), KMAXbc, MXPbc, Ebc, REBbc]
static const int N_SM_DE = 22 + p3bfa::N_BA_C;
static inline gl_t F_de(const gl_t* v, const gl_t* lam) {
    const gl_t* c = v + 4;
    const gl_t* cA = c + SM_A;
    gl_t one = 1ULL;
    gl_t M = v[3];
    gl_t KM = v[4 + NDE2], MX = v[5 + NDE2], Eb = v[6 + NDE2], RB = v[7 + NDE2];
    auto boolc = [](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    gl_t nM = gl_sub(one, M);
    // absent = 1 - M*(1-EZ) = 1 - M + M*EZ
    gl_t MEZ = gl_mul(M, c[SM_EZ]);
    gl_t absent = gl_add(gl_sub(one, M), MEZ);
    gl_t r[N_SM_DE];
    r[0] = gl_sub(c[SM_XM], gl_mul(M, v[1]));
    r[1] = gl_mul(nM, c[SM_X2M]);
    r[2] = gl_mul(M, gl_sub(MX, gl_add(gl_sub(32768ULL, gl_mul(cA[p3bfa::BA_S2], 32768ULL)),
                    gl_add(gl_mul(cA[p3bfa::BA_E2], 128ULL), cA[p3bfa::BA_M2]))));
    {   // key dominance: key = 32768 - S1 + (1-2*S1)*(128*E1 + M1)
        gl_t S1 = cA[p3bfa::BA_S1];
        gl_t mag = gl_add(gl_mul(cA[p3bfa::BA_E1], 128ULL), cA[p3bfa::BA_M1]);
        gl_t key = gl_add(gl_sub(32768ULL, S1),
                          gl_sub(mag, gl_mul(gl_add(S1, S1), mag)));
        r[3] = gl_mul(M, gl_sub(KM, gl_add(key, c[SM_DK])));
    }
    r[4] = gl_mul(nM, c[SM_DK]);
    r[5] = boolc(c[SM_SEL1]);
    r[6] = gl_mul(c[SM_SEL1], nM);
    r[7] = gl_mul(c[SM_SEL1], c[SM_DK]);
    r[8] = gl_mul(c[SM_SEL1], gl_sub(c[SM_XM], MX));
    r[9] = gl_sub(c[SM_EXV], gl_add(gl_mul(c[SM_EEB], 128ULL), c[SM_EMB]));
    r[10] = boolc(c[SM_EZ]);
    r[11] = gl_mul(c[SM_EZ], c[SM_EEB]);
    r[12] = gl_sub(gl_mul(c[SM_EEB], c[SM_EZI]), gl_sub(one, c[SM_EZ]));
    r[13] = gl_mul(c[SM_SH2], absent);
    r[14] = gl_sub(gl_mul(M, gl_sub(gl_add(c[SM_SH2], gl_add(c[SM_EEB], 8ULL)), Eb)),
                   gl_mul(MEZ, gl_sub(gl_add(c[SM_SH2], gl_add(c[SM_EEB], 8ULL)), Eb)));
    {   // Q2*PW2 + RR2 = present * (128+EMB)*256
        gl_t sq = gl_mul(gl_add(128ULL, c[SM_EMB]), 256ULL);
        gl_t psq = gl_sub(gl_mul(M, sq), gl_mul(MEZ, sq));
        r[15] = gl_sub(gl_add(gl_mul(c[SM_Q2], c[SM_PW2]), c[SM_RR2]), psq);
    }
    r[16] = boolc(c[SM_SEL2]);
    r[17] = gl_mul(c[SM_SEL2], c[SM_SH2]);
    r[18] = gl_mul(c[SM_SEL2], absent);
    {   // output exponent
        gl_t eo3rel = gl_sub(gl_add(c[SM_EO3], 127ULL),
                             gl_add(gl_add(c[SM_EEB], RB), c[SM_EI3]));
        r[19] = gl_sub(gl_mul(M, eo3rel), gl_mul(MEZ, eo3rel));
        r[20] = gl_mul(absent, gl_sub(c[SM_EO3], one));
    }
    {   // output pattern (sign always 0)
        gl_t pat = gl_sub(gl_add(gl_mul(c[SM_EO3], 128ULL), c[SM_MO3]), 128ULL);
        r[21] = gl_sub(v[2], gl_sub(gl_mul(M, pat), gl_mul(MEZ, pat)));
    }
    p3bfa::ba_constraints(cA, c[SM_XM], c[SM_X2M], c[SM_DP], r + 22);
    gl_t s = 0;
    for (int j = 0; j < N_SM_DE; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}
// Db: v = [E, EMP, db(NDB2)]; needs public SM_EMIN
static const int N_SM_DB = 7;
static inline gl_t F_db(const gl_t* v, const gl_t* lam, gl_t EMIN) {
    const gl_t* c = v + 2;
    gl_t one = 1ULL;
    gl_t nE = gl_sub(one, v[1]);
    auto boolc = [](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    gl_t r[N_SM_DB];
    r[0] = boolc(c[B_FSEL]);
    r[1] = gl_mul(c[B_FSEL], c[B_EDIF]);
    r[2] = gl_sub(c[B_E], gl_add(EMIN, c[B_EDIF]));
    r[3] = gl_mul(nE, gl_sub(gl_sub(c[B_S], c[B_PLO]),
                             gl_add(gl_mul(c[B_U1H], 4096ULL), c[B_U1L])));
    r[4] = gl_mul(nE, gl_sub(gl_sub(gl_sub(c[B_PHI], one), c[B_S]),
                             gl_add(gl_mul(c[B_U2H], 4096ULL), c[B_U2L])));
    r[5] = gl_mul(nE, gl_sub(gl_add(gl_mul(c[B_U16], c[B_PDN]), c[B_NR]),
                             gl_mul(c[B_S], c[B_PUP])));
    r[6] = gl_mul(nE, gl_sub(gl_add(gl_add(c[B_REB], c[B_E]), c[B_WD]),
                             gl_add(277ULL, c[B_HBC])));
    gl_t s = 0;
    for (int j = 0; j < N_SM_DB; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}
// row binding: v = [EQb, SEL1, SEL2, Q2]; lam[3]
static inline gl_t F_bind(const gl_t* v, const gl_t* lam) {
    return gl_mul(v[0], gl_add(gl_add(gl_mul(lam[0], v[1]), gl_mul(lam[1], v[2])),
                               gl_mul(lam[2], v[3])));
}

// ---------------- lookup descriptors ----------------
// dom 0 = De, 1 = Db; cols >= 0 committed (De or Db per dom); -1 emf, -2 mrcf
struct LuDef { int tab; int dom; std::vector<int> cols; };
// tab ids: 0..NBT-1 in BT; 100+ = the extra tables
enum { XT_R11 = 100, XT_R12, XT_R16, XT_SH512, XT_WD24, XT_RM8, XT_MUL7, XT_EXPT, XT_RCPT };
static inline const Table& tab_of(const Tables& T, int id) {
    if (id < 100) return T.BT.t[id];
    switch (id) {
        case XT_R11: return T.R11;   case XT_R12: return T.R12;
        case XT_R16: return T.R16;   case XT_SH512: return T.SH512;
        case XT_WD24: return T.WD24; case XT_RM8: return T.RM8;
        case XT_MUL7: return T.MUL7; case XT_EXPT: return T.EXPT;
        default: return T.RCPT;
    }
}
static inline std::vector<LuDef> lu_defs() {
    std::vector<LuDef> L(NLSM);
    L[LSM_DK]  = {XT_R16, 0, {SM_DK}};
    L[LSM_EXP] = {XT_EXPT, 0, {SM_DP, SM_EXV}};
    L[LSM_EEB] = {p3bfa::BT_R256, 0, {SM_EEB}};
    L[LSM_EMB] = {p3bfa::BT_R128, 0, {SM_EMB}};
    L[LSM_SH2] = {XT_SH512, 0, {SM_SH2, SM_PW2}};
    L[LSM_RM2] = {p3bfa::BT_RM17, 0, {SM_PW2, SM_RR2}};
    L[LSM_Q2]  = {XT_R16, 0, {SM_Q2}};
    L[LSM_M3]  = {XT_MUL7, 0, {-1, -2, SM_MO3, SM_EI3}};
    L[LSM_EO3] = {p3bfa::BT_REXP, 0, {SM_EO3}};
    L[LSM_EDIF]= {p3bfa::BT_R512, 1, {B_EDIF}};
    L[LSM_WD]  = {XT_WD24, 1, {B_WD, B_PLO, B_PHI, B_PDN, B_PUP}};
    L[LSM_U1H] = {XT_R11, 1, {B_U1H}};
    L[LSM_U1L] = {XT_R12, 1, {B_U1L}};
    L[LSM_U2H] = {XT_R11, 1, {B_U2H}};
    L[LSM_U2L] = {XT_R12, 1, {B_U2L}};
    L[LSM_NR]  = {XT_RM8, 1, {B_PDN, B_NR}};
    L[LSM_RCP] = {XT_RCPT, 1, {B_U16, B_MRC, B_HBC}};
    L[LSM_REB] = {p3bfa::BT_REXP, 1, {B_REB}};
    auto BL = p3bfa::ba_lu_defs();
    for (int i = 0; i < p3bfa::NBLU; i++) {
        std::vector<int> cc;
        for (int cid : BL[i].cols) cc.push_back(SM_A + cid);
        L[LSM_A + i] = {BL[i].tab, 0, cc};
    }
    return L;
}

// ---------------- proof object ----------------
struct SmxProof {
    uint32_t B = 0, ln = 0;
    p3fri::Hash rde[NDE2], rdb[NDB2];
    std::vector<p3lu::GroupProof> lug;           // standalone merged lookup groups
    // MUL7 virtual-key binding claims ride the M3 member's `extra` slots
    std::vector<Msg5> mDe; std::vector<gl_t> yDe;
    gl_t yDeS = 0, yDeP = 0, yDeKM = 0, yDeMX = 0, yDeE = 0, yDeRB = 0;
    std::vector<Msg5> mDb; gl_t yDb[NDB2] = {};
    gl_t yBF = 0, yBS = 0;
    std::vector<Msg5> mBind; gl_t yBSEL1 = 0, yBSEL2 = 0, yBQ2 = 0;
    p3zkc::Blind zbl[3];                         // zk: De, Db, bind blinds
    std::vector<p3bo::BatchProof> batches;
};

// ==================== prover ====================
static inline SmxProof prove(fs::Transcript& tr, const Wit& wt, const Tables& T,
                             const Art& a, const Operands& ops, uint32_t R, uint32_t Q,
                             bool strict = true, p3lu::XCtx* xc = nullptr) {
    const Dims& dm = wt.dm;
    SmxProof pf; pf.B = dm.B; pf.ln = dm.ln;
    if (16 + dm.ln > T.wmax)
        throw std::runtime_error("smx: tables too small (need wmax >= 16+ln)");

    uint32_t hdr[3] = {dm.B, dm.ln, (uint32_t)a.sm_emin};
    tr.absorb("smx-dims", hdr, sizeof hdr);
    for (int i = 0; i < p3bfa::NBT; i++) tr.absorb("smx-tab", T.BT.t[i].id.data(), 32);
    const Table* xt[9] = {&T.R11, &T.R12, &T.R16, &T.SH512, &T.WD24, &T.RM8,
                          &T.MUL7, &T.EXPT, &T.RCPT};
    for (auto* t : xt) tr.absorb("smx-tab", t->id.data(), 32);
    tr.absorb("smx-msk", wt.mskc.data(), wt.mskc.size() * sizeof(gl_t));
    tr.absorb("smx-S", ops.S.root.data(), 32);
    tr.absorb("smx-P", ops.P.root.data(), 32);

    p3lu::XCtx xc_loc;
    p3lu::XCtx& XC = xc ? *xc : xc_loc;
    p3bo::PLedger& lg = XC.lg;
    std::deque<Col>& lucols = XC.keep;

    std::vector<Col>& CDe = XC.vec(NDE2);
    std::vector<Col>& CDb = XC.vec(NDB2);
    // zk: the row bindings sum SEL1 = 1-EMP (EMP public, zero-extended -> the
    // slice-1 rows must sum to 1), sum SEL2 = 1-FSEL, sum Q2 = S are CLAIM
    // algebra -> slice-1 linked masks (p3_zkc mechanism 1)
    const bool zk = p3zkc::G.on;
    std::vector<gl_t> mSEL1, mSEL2, mQ2, mFSEL, mS;
    if (zk) {
        mSEL1 = p3zkc::fresh_mask(dm.le); mSEL2 = p3zkc::fresh_mask(dm.le);
        mQ2 = p3zkc::fresh_mask(dm.le);
        mFSEL = p3zkc::fresh_mask(dm.lb); mS = p3zkc::fresh_mask(dm.lb);
        for (uint32_t b = 0; b < dm.Bpad; b++) {
            gl_t s1 = 0, s2 = 0, sq = 0;
            for (uint32_t j = 0; j + 1 < dm.n; j++) s1 = gl_add(s1, mSEL1[((size_t)b << dm.ln) | j]);
            mSEL1[((size_t)b << dm.ln) | (dm.n - 1)] = gl_sub(1ULL, s1);   // rowsum = 1
            for (uint32_t j = 0; j < dm.n; j++) {
                s2 = gl_add(s2, mSEL2[((size_t)b << dm.ln) | j]);
                sq = gl_add(sq, mQ2[((size_t)b << dm.ln) | j]);
            }
            mFSEL[b] = gl_sub(1ULL, s2);
            mS[b] = sq;
        }
    }
    for (int c = 0; c < NDE2; c++) {
        const std::vector<gl_t>* m = nullptr;
        if (zk) { if (c == SM_SEL1) m = &mSEL1; else if (c == SM_SEL2) m = &mSEL2;
                  else if (c == SM_Q2) m = &mQ2; }
        CDe[c] = commit_col_nc(wt.de[c], R, m); pf.rde[c] = CDe[c].root;
    }
    for (int c = 0; c < NDB2; c++) {
        const std::vector<gl_t>* m = nullptr;
        if (zk) { if (c == B_FSEL) m = &mFSEL; else if (c == B_S) m = &mS; }
        CDb[c] = commit_col_nc(wt.db[c], R, m); pf.rdb[c] = CDb[c].root;
    }
    for (int c = 0; c < NDE2; c++) tr.absorb("smx-ce", pf.rde[c].data(), 32);
    for (int c = 0; c < NDB2; c++) tr.absorb("smx-cb", pf.rdb[c].data(), 32);

    // zk: virtual MUL7 keys rebuilt over the augmented domain (affine/broadcast)
    const std::vector<gl_t> *pem = &wt.emf, *pmr = &wt.mrcf;
    if (zk) {
        std::vector<gl_t> emf_z = CDe[SM_EMB].v; for (auto& x : emf_z) x = gl_add(x, 128ULL);
        pem = &XC.varr(std::move(emf_z));
        pmr = &XC.varr(p3zkc::bc_aug(CDb[B_MRC].v, dm.lb, dm.le, dm.Ne,
                               [&](size_t e) { return e >> dm.ln; }));
    }
    auto LD = lu_defs();
    for (int i = 0; i < NLSM; i++) {
        std::vector<LuColSpec> spec;
        for (int cid : LD[i].cols) {
            if (cid == -1) spec.push_back(LV(pem));
            else if (cid == -2) spec.push_back(LV(pmr));
            else spec.push_back(LC(LD[i].dom == 0 ? &CDe[cid] : &CDb[cid]));
        }
        p3lu::PBind bind;
        if (i == LSM_M3) {
            const Col *emb = &CDe[SM_EMB], *mrc = &CDb[B_MRC];
            Dims dmc = dm;
            bind = [emb, mrc, dmc](fs::Transcript& trb, p3lu::XCtx& xcb,
                                   const std::vector<gl_t>& pm, const std::vector<gl_t>&) {
                gl_t y1 = claimc(trb, xcb.lg, *emb, pm);
                std::vector<gl_t> rb(pm.begin() + dmc.ln, pm.begin() + dmc.le);
                gl_t y2 = claimc(trb, xcb.lg, *mrc, p3zkc::expt(rb, pm, dmc.le));
                return std::vector<gl_t>{y1, y2};
            };
        }
        p3lu::defer_v(XC, std::move(spec), wt.lidx[i], tab_of(T, LD[i].tab),
                      "smxLU" + std::to_string(i), std::move(bind));
    }

    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, dm.le);
    gl_t lamE = chal(tr);
    std::vector<gl_t> lamEv(N_SM_DE); lamEv[0] = 1;
    for (int j = 1; j < N_SM_DE; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(8 + NDE2);
        cols.push_back(beq(p3zkc::zpt(zE)));
        cols.push_back(ops.S.v); cols.push_back(ops.P.v);
        if (zk) { auto mz = wt.mskc; mz.resize(CDe[0].v.size(), 0); cols.push_back(std::move(mz)); }
        else cols.push_back(wt.mskc);
        for (int c = 0; c < NDE2; c++) cols.push_back(CDe[c].v);
        auto bmap = [&](size_t e) { return e >> dm.ln; };
        cols.push_back(p3zkc::bc_aug(CDb[B_KMAX].v, dm.lb, dm.le, dm.Ne, bmap));
        cols.push_back(p3zkc::bc_aug(CDb[B_MXP].v, dm.lb, dm.le, dm.Ne, bmap));
        cols.push_back(p3zkc::bc_aug(CDb[B_E].v, dm.lb, dm.le, dm.Ne, bmap));
        cols.push_back(p3zkc::bc_aug(CDb[B_REB].v, dm.lb, dm.le, dm.Ne, bmap));
        CFn F = [&](const gl_t* v) { return F_de(v, lamEv.data()); };
        std::vector<gl_t> rE = p3hwl::sc5z(tr, "smx-scE", dm.le, std::move(cols), F, pf.mDe,
                                           0, R, lg, lucols, pf.zbl[0]);
        pf.yDeS = claimc(tr, lg, ops.S, rE);
        pf.yDeP = claimc(tr, lg, ops.P, rE);
        pf.yDe.resize(NDE2);
        for (int c = 0; c < NDE2; c++) pf.yDe[c] = claimc(tr, lg, CDe[c], rE);
        std::vector<gl_t> rb(rE.begin() + dm.ln, rE.begin() + dm.le);
        pf.yDeKM = claimc(tr, lg, CDb[B_KMAX], p3zkc::expt(rb, rE, dm.le));
        pf.yDeMX = claimc(tr, lg, CDb[B_MXP], p3zkc::expt(rb, rE, dm.le));
        pf.yDeE = claimc(tr, lg, CDb[B_E], p3zkc::expt(rb, rE, dm.le));
        pf.yDeRB = claimc(tr, lg, CDb[B_REB], p3zkc::expt(rb, rE, dm.le));
    }

    // -- Db zero-check --
    std::vector<gl_t> zB = chal_vec(tr, dm.lb);
    gl_t lamB = chal(tr), lamBv[N_SM_DB]; lamBv[0] = 1;
    for (int j = 1; j < N_SM_DB; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(2 + NDB2);
        cols.push_back(beq(p3zkc::zpt(zB)));
        if (zk) { auto ez = wt.empc; ez.resize(CDb[0].v.size(), 0); cols.push_back(std::move(ez)); }
        else cols.push_back(wt.empc);
        for (int c = 0; c < NDB2; c++) cols.push_back(CDb[c].v);
        gl_t EMINc = (gl_t)a.sm_emin;
        CFn F = [&](const gl_t* v) { return F_db(v, lamBv, EMINc); };
        std::vector<gl_t> rB = p3hwl::sc5z(tr, "smx-scB", dm.lb, std::move(cols), F, pf.mDb,
                                           0, R, lg, lucols, pf.zbl[1]);
        for (int c = 0; c < NDB2; c++) pf.yDb[c] = claimc(tr, lg, CDb[c], rB);
    }

    // -- row binding: sum SEL1 = 1-EMP, sum SEL2 = 1-FSEL, sum Q2 = S --
    std::vector<gl_t> z2 = chal_vec(tr, dm.lb);
    gl_t zexs = zk ? chal(tr) : 0;
    pf.yBF = claimc(tr, lg, CDb[B_FSEL], p3zkc::xpt(z2, zexs));
    pf.yBS = claimc(tr, lg, CDb[B_S], p3zkc::xpt(z2, zexs));
    gl_t lam1 = chal(tr), lam2 = chal(tr), lam3 = chal(tr);
    {
        uint32_t e_e = p3zkc::e_of(dm.le);
        std::vector<gl_t> ptb = z2;
        if (zk) { ptb.push_back(zexs); ptb.resize(dm.lb + e_e, 0); }
        std::vector<gl_t> eqb = beq(ptb);
        size_t NeA = CDe[SM_SEL1].v.size();
        std::vector<gl_t> EQb(NeA);
        for (size_t q = 0; q < NeA; q++) {
            size_t ex = q >> dm.le, e = q & (dm.Ne - 1);
            EQb[q] = eqb[(ex << dm.lb) | (e >> dm.ln)];
        }
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(std::move(EQb));
        cols.push_back(CDe[SM_SEL1].v); cols.push_back(CDe[SM_SEL2].v);
        cols.push_back(CDe[SM_Q2].v);
        gl_t lam[3] = {lam1, lam2, lam3};
        CFn F = [&](const gl_t* v) { return F_bind(v, lam); };
        gl_t yEMPa = mle_eval(wt.empc, z2);
        if (zk) yEMPa = gl_mul(gl_sub(1ULL, zexs), yEMPa);
        gl_t base0 = gl_add(gl_add(gl_mul(lam1, gl_sub(1ULL, yEMPa)),
                                   gl_mul(lam2, gl_sub(1ULL, pf.yBF))),
                            gl_mul(lam3, pf.yBS));
        std::vector<gl_t> rS = p3hwl::sc5z(tr, "smx-scS", dm.le, std::move(cols), F, pf.mBind,
                                           base0, R, lg, lucols, pf.zbl[2]);
        pf.yBSEL1 = claimc(tr, lg, CDe[SM_SEL1], rS);
        pf.yBSEL2 = claimc(tr, lg, CDe[SM_SEL2], rS);
        pf.yBQ2 = claimc(tr, lg, CDe[SM_Q2], rS);
    }

    if (!xc) {
        pf.lug = p3lu::lu_flush(tr, XC, R, Q, strict);
        for (size_t i = 0; i < lg.cls.size(); i++)
            pf.batches.push_back(p3bo::prove_class(tr, lg.cls[i], R, Q,
                                                   "smx-bo" + std::to_string(i)));
    }
    return pf;
}

// ==================== verifier ====================
// PUBLIC: mask bytes (B*n, causal or any structural mask), roots, dims, params.
static inline bool verify(fs::Transcript& tr, const Tables& T, const Art& a,
                          const SmxProof& pf, const std::vector<uint8_t>& mask,
                          const p3fri::Hash& rS, const p3fri::Hash& rP,
                          uint32_t B, uint32_t n,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr,
                          p3lu::VCtx* xv = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    uint32_t ln = ilog2(n);
    if ((1u << ln) != n) return fail("n must be pow2");
    if (pf.B != B || pf.ln != ln) return fail("dims mismatch");
    if (16 + ln > T.wmax) return fail("smx tables too small (wmax)");
    if (mask.size() != (size_t)B * n) return fail("mask size");
    if (pf.yDe.size() != NDE2) return fail("yDe count");
    Dims dm = make_dims(B, ln);
    p3lu::VCtx vc_loc;
    p3lu::VCtx& VC = xv ? *xv : vc_loc;
    p3bo::VLedger& vlg = VC.vlg;

    // rebuild the public mask / empty-row columns (padding rows all-masked)
    std::vector<gl_t> mskc(dm.Ne, 0), empc(dm.Bpad, 1);
    for (uint32_t b = 0; b < B; b++) {
        bool any = false;
        for (uint32_t j = 0; j < n; j++) {
            bool m = mask[(size_t)b * n + j] != 0;
            mskc[((size_t)b << ln) | j] = m ? 1 : 0;
            any = any || m;
        }
        empc[b] = any ? 0 : 1;
    }

    uint32_t hdr[3] = {B, ln, (uint32_t)a.sm_emin};
    tr.absorb("smx-dims", hdr, sizeof hdr);
    for (int i = 0; i < p3bfa::NBT; i++) tr.absorb("smx-tab", T.BT.t[i].id.data(), 32);
    const Table* xt[9] = {&T.R11, &T.R12, &T.R16, &T.SH512, &T.WD24, &T.RM8,
                          &T.MUL7, &T.EXPT, &T.RCPT};
    for (auto* t : xt) tr.absorb("smx-tab", t->id.data(), 32);
    tr.absorb("smx-msk", mskc.data(), mskc.size() * sizeof(gl_t));
    tr.absorb("smx-S", rS.data(), 32);
    tr.absorb("smx-P", rP.data(), 32);
    for (int c = 0; c < NDE2; c++) tr.absorb("smx-ce", pf.rde[c].data(), 32);
    for (int c = 0; c < NDB2; c++) tr.absorb("smx-cb", pf.rdb[c].data(), 32);

    auto LD = lu_defs();
    for (int i = 0; i < NLSM; i++) {
        uint32_t explog = LD[i].dom == 0 ? dm.le : dm.lb;
        std::vector<const p3fri::Hash*> roots;
        for (int cid : LD[i].cols) {
            if (cid < 0) roots.push_back(nullptr);
            else roots.push_back(LD[i].dom == 0 ? &pf.rde[cid] : &pf.rdb[cid]);
        }
        p3lu::VBind bind;
        if (i == LSM_M3) {
            p3fri::Hash he = pf.rde[SM_EMB], hm = pf.rdb[B_MRC]; Dims dmc = dm;
            bind = [he, hm, dmc](fs::Transcript& trb, p3lu::VCtx& vc,
                                 const std::vector<gl_t>& pm, const std::vector<gl_t>& yv,
                                 const std::vector<gl_t>& ex, const char** wy) {
                auto f = [&](const char* m) { if (wy) *wy = m; return false; };
                if (yv.size() != 2) return f("M3 y_virt count");
                if (ex.size() != 2) return f("M3 extra count");
                gl_t yemb = claimv(trb, vc.vlg, he, pm, ex[0]);
                std::vector<gl_t> rb(pm.begin() + dmc.ln, pm.begin() + dmc.le);
                gl_t ymrc = claimv(trb, vc.vlg, hm, p3zkc::expt(rb, pm, dmc.le), ex[1]);
                if (yv[0] != gl_add(128ULL, yemb)) return f("M3 emf binding");
                if (yv[1] != ymrc) return f("M3 mrcf binding");
                return true;
            };
        }
        p3lu::vdefer_v(VC, std::move(roots), tab_of(T, LD[i].tab), explog,
                       "smxLU" + std::to_string(i), std::move(bind));
    }

    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, dm.le);
    gl_t lamE = chal(tr);
    std::vector<gl_t> lamEv(N_SM_DE); lamEv[0] = 1;
    for (int j = 1; j < N_SM_DE; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.zbl[0]);
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.mDe, p3zkc::vfull(dm.le), gl_mul(rho, pf.zbl[0].H),
                        tr, "smx-scE", rE, claim)) return fail("De sumcheck");
        p3hwl::sc5vz_claims(tr, vlg, pf.zbl[0], rE);
        std::vector<gl_t> v(8 + NDE2);
        v[0] = p3bf::eq_point(rE, p3zkc::zpt(zE));
        v[1] = claimv(tr, vlg, rS, rE, pf.yDeS);
        v[2] = claimv(tr, vlg, rP, rE, pf.yDeP);
        std::vector<gl_t> rEl(rE.begin(), rE.begin() + dm.le);
        gl_t exf = 1ULL;                          // zk: public mask zero-extended
        for (size_t ii = dm.le; ii < rE.size(); ii++) exf = gl_mul(exf, gl_sub(1ULL, rE[ii]));
        v[3] = gl_mul(mle_eval(mskc, rEl), exf);
        for (int c = 0; c < NDE2; c++) v[4 + c] = claimv(tr, vlg, pf.rde[c], rE, pf.yDe[c]);
        std::vector<gl_t> rb(rE.begin() + ln, rE.begin() + dm.le);
        v[4 + NDE2] = claimv(tr, vlg, pf.rdb[B_KMAX], p3zkc::expt(rb, rE, dm.le), pf.yDeKM);
        v[5 + NDE2] = claimv(tr, vlg, pf.rdb[B_MXP], p3zkc::expt(rb, rE, dm.le), pf.yDeMX);
        v[6 + NDE2] = claimv(tr, vlg, pf.rdb[B_E], p3zkc::expt(rb, rE, dm.le), pf.yDeE);
        v[7 + NDE2] = claimv(tr, vlg, pf.rdb[B_REB], p3zkc::expt(rb, rE, dm.le), pf.yDeRB);
        gl_t end = gl_add(F_de(v.data(), lamEv.data()),
                          p3hwl::sc5_blindterm(pf.zbl[0], rho, v[0]));
        if (end != claim) return fail("De terminal");
    }

    // -- Db zero-check --
    std::vector<gl_t> zB = chal_vec(tr, dm.lb);
    gl_t lamB = chal(tr), lamBv[N_SM_DB]; lamBv[0] = 1;
    for (int j = 1; j < N_SM_DB; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.zbl[1]);
        std::vector<gl_t> rB; gl_t claim;
        if (!sc5_verify(pf.mDb, p3zkc::vfull(dm.lb), gl_mul(rho, pf.zbl[1].H),
                        tr, "smx-scB", rB, claim)) return fail("Db sumcheck");
        p3hwl::sc5vz_claims(tr, vlg, pf.zbl[1], rB);
        std::vector<gl_t> v(2 + NDB2);
        v[0] = p3bf::eq_point(rB, p3zkc::zpt(zB));
        std::vector<gl_t> rBl(rB.begin(), rB.begin() + dm.lb);
        gl_t exf = 1ULL;
        for (size_t ii = dm.lb; ii < rB.size(); ii++) exf = gl_mul(exf, gl_sub(1ULL, rB[ii]));
        v[1] = gl_mul(mle_eval(empc, rBl), exf);
        for (int c = 0; c < NDB2; c++) v[2 + c] = claimv(tr, vlg, pf.rdb[c], rB, pf.yDb[c]);
        gl_t end = gl_add(F_db(v.data(), lamBv, (gl_t)a.sm_emin),
                          p3hwl::sc5_blindterm(pf.zbl[1], rho, v[0]));
        if (end != claim) return fail("Db terminal");
    }

    // -- row binding --
    std::vector<gl_t> z2 = chal_vec(tr, dm.lb);
    {
        const bool zk = p3zkc::G.on;
        gl_t zexs = zk ? chal(tr) : 0;
        gl_t yF = claimv(tr, vlg, pf.rdb[B_FSEL], p3zkc::xpt(z2, zexs), pf.yBF);
        gl_t yS = claimv(tr, vlg, pf.rdb[B_S], p3zkc::xpt(z2, zexs), pf.yBS);
        gl_t lam1 = chal(tr), lam2 = chal(tr), lam3 = chal(tr);
        gl_t yEMP = mle_eval(empc, z2);
        if (zk) yEMP = gl_mul(gl_sub(1ULL, zexs), yEMP);
        gl_t claim0 = gl_add(gl_add(gl_mul(lam1, gl_sub(1ULL, yEMP)),
                                    gl_mul(lam2, gl_sub(1ULL, yF))),
                             gl_mul(lam3, yS));
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.zbl[2]);
        claim0 = gl_add(claim0, gl_mul(rho, pf.zbl[2].H));
        std::vector<gl_t> rSc; gl_t claim;
        if (!sc5_verify(pf.mBind, p3zkc::vfull(dm.le), claim0, tr, "smx-scS", rSc, claim))
            return fail("bind sumcheck");
        p3hwl::sc5vz_claims(tr, vlg, pf.zbl[2], rSc);
        gl_t y1 = claimv(tr, vlg, pf.rde[SM_SEL1], rSc, pf.yBSEL1);
        gl_t y2 = claimv(tr, vlg, pf.rde[SM_SEL2], rSc, pf.yBSEL2);
        gl_t y3 = claimv(tr, vlg, pf.rde[SM_Q2], rSc, pf.yBQ2);
        std::vector<gl_t> rSb(rSc.begin() + ln, rSc.begin() + dm.le);
        rSb.insert(rSb.end(), rSc.begin() + dm.le, rSc.end());
        std::vector<gl_t> ptb = z2;
        if (zk) { ptb.push_back(zexs); ptb.resize(dm.lb + p3zkc::e_of(dm.le), 0); }
        gl_t w = p3bf::eq_point(rSb, ptb);
        gl_t end = gl_mul(w, gl_add(gl_add(gl_mul(lam1, y1), gl_mul(lam2, y2)),
                                    gl_mul(lam3, y3)));
        end = gl_add(end, p3hwl::sc5_blindterm(pf.zbl[2], rho, w));
        if (end != claim) return fail("bind terminal");
    }

    if (!xv) {
        if (!p3lu::lu_verify_flush(tr, VC, pf.lug, Q_pub, R_pub, why)) return false;
        if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
        for (size_t i = 0; i < vlg.cls.size(); i++)
            if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                    "smx-bo" + std::to_string(i), why)) return false;
    } else if (!pf.batches.empty() || !pf.lug.empty()) return fail("unexpected batches");

    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const SmxProof& pf) {
    size_t s = 8 + (NDE2 + NDB2) * 32;
    for (auto& g : pf.lug) s += p3lu::sz_group(g);
    auto msgs = [&](const std::vector<Msg5>& m) { return m.size() * 40; };
    s += msgs(pf.mDe) + msgs(pf.mDb) + msgs(pf.mBind);
    s += 8 * (NDE2 + 8 + NDB2 + 2 + 3 + 2);
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3smx
