// Full single-FC-layer prover+verifier for the EXACT Hawkeye fp8 forward pass
// (hawkeye.py semantics, products_per_group G=32, internal_width IW=14,
// zero_exponent -139), over the p3 stack (Goldilocks + Basefold + logUp).
//
// Statement: committed fp8 operands X (B x K codes), W (N x K codes) and
// committed per-row fp32 scales xs, ws produce the PUBLIC bf16 output Y
// (B x N, uint16 bit patterns) bit-identical to hawkeye_ref / the Triton
// hawkeye_fp8_sum kernel.
//
// Composition (design doc HAWKEYE_ZKP_DESIGN.md):
//   P1  fused decode-multiply lookup (a,b VIRTUAL: bound to X/W by openings)
//   P2  truncating per-product alignment (q*pw + r = mag, SHIFT/REM/RANGE15)
//   P3  group max_exp (dominance via sh linear binding + attainment selectors)
//   P4  accumulator realign (acc_sig split >>10, then P2-style shift, signed)
//   P5  normalize (pow2-sandwich bit width, truncate-to-14-bit q/r, out exp)
//   P6  output binding (gfloat->fp32, two fp32 RNE multiplies by the scales,
//       bf16 RNE downcast = integer RNE on the fp32 bits), Y bound as a
//       public-matrix MLE evaluation restricted to the real BxN grid.
//   chain: per-output accumulator recurrence across K/32 groups via ONE
//       lambda-weighted shift sumcheck over the group domain (A(g+1)=out(g)).
//
// Layouts (LSB-first bit order of flattened indices):
//   output   o  = b*Npad + n                bits [n | b]           (Do)
//   group    gi = g*Opad + o                bits [n | b | g]       (Dg)
//   product  p  = gi*32 + kk                bits [kk | n | b | g]  (Dp)
//   X idx = b*Kpad + g*32 + kk              bits [kk | g | b]
//   W idx = n*Kpad + g*32 + kk              bits [kk | g | n]
// so every broadcast (a(p)=X(b,k), group->product, scale->output) is a
// variable-subset/permutation of an already committed MLE: its evaluation is
// ONE opening of the base column at a rearranged point ("virtual broadcast").
//
// Exponents are biased by +139 so all committed exponents are nonnegative
// (biased zero_exponent = 0; biased prod_exp = eb + 127 with eb = prod_exp+12).
//
// Supported scale domain (proof REJECTS otherwise -- sound, not complete):
// each scale is +-0 or a normal finite fp32, and neither RNE multiply
// overflows/underflows the normal range (result exponent in [1,254]).
// bf16 rounding to +-inf at the top of the range IS handled.
//
// Non-ZK (mask-slice hardening is the follow-up), base-field challenges
// (GL2 upgrade is the stack-wide production change).
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <deque>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_batchopen.cuh"
#include "p3_logup.cuh"
#include "fs_transcript.hpp"

namespace p3hwl {

using p3lu::Col; using p3lu::Table; using p3lu::commit_col; using p3lu::make_table;
using p3lu::chal; using p3lu::chal_vec; using p3lu::bind_lsb; using p3lu::ilog2;
using p3lu::LuColSpec; using p3lu::LC; using p3lu::LV;

static inline gl_t enc(int64_t x) { return x >= 0 ? (gl_t)x : gl_sub(0ULL, (gl_t)(-x)); }
static inline uint32_t pow2_at_least(uint32_t x) { uint32_t p = 1; while (p < x) p <<= 1; return p; }

// Free each per-product witness column (wt.dp[c]) right after its commitment
// copies it: the prover only reads the COMMITTED copy afterwards.  Halves the
// dominant host-memory term at llama-68m dims.  Off by default because it
// mutates the caller's witness (a second prove() on the same LayerWit would
// see empty dp columns); the scale bench opts in.
static bool g_free_dp = false;

// ---------------- column enums ----------------
enum {  // Dp: per-product (a,b are VIRTUAL -- not committed)
    P_EB = 0, P_MAG, P_SG, P_PR, P_SH, P_PW, P_Q, P_R, P_AL, P_SEL, NDP };
enum {  // Dg: per-group (A-state asig/bexp/asgn is the IN-state of group g)
    G_ASIG = 0, G_BEXP, G_ASGN, G_AZ, G_AINV, G_ABASE, G_ALO, G_ASH, G_APW,
    G_AQ, G_AR, G_MAX, G_ASEL, G_CSUM, G_TSGN, G_TMAG, G_TZ, G_TINV, G_WD,
    G_PLO, G_PHI, G_PDN, G_PUP, G_NR, G_S14, G_U1H, G_U1L, G_U2H, G_U2L, NDG };
enum {  // Do: per-output (P6)
    O_S14F = 0, O_BEXP, O_SGN, O_FZ, O_FINV,
    O_C1, O_REM1, O_RB1, O_RT1, O_RTI1, O_ZRT1, O_FH1, O_FL1, O_FLH1, O_L01,
    O_RUP1, O_C21, O_M1, O_E1, O_S1, O_Z1,
    O_C2, O_RB2, O_TH2, O_TL2, O_RTI2, O_ZRT2, O_FH2, O_FL2, O_FLH2, O_L02,
    O_RUP2, O_C22, O_M2, O_E2, O_S2, O_Z2,
    O_Q16, O_Q16H, O_L03, O_RB3, O_RT3, O_RTI3, O_ZRT3, O_RUP3, O_YB, NDO };
enum {  // Db / Dn: per-row scale decomposition (xs/ws bit columns are operands)
    S_SS = 0, S_SE, S_SEI, S_ZS, S_SMH, S_SML, NDS };

// lookup instances, fixed order
enum {
    LU_DM = 0, LU_SH, LU_RM, LU_Q15,                                       // Dp
    LU_ASH, LU_ARM, LU_AQ, LU_ABASE, LU_ALO, LU_WD,
    LU_U1H, LU_U1L, LU_U2H, LU_U2L, LU_NR, LU_S14,                          // Dg
    LU_M1REM, LU_M1RT, LU_M1FH, LU_M1FL, LU_M1FLH, LU_M1E,                  // Do mul1
    LU_M2TH, LU_M2TL, LU_M2FH, LU_M2FL, LU_M2FLH, LU_M2E,                   // Do mul2
    LU_Q16, LU_Q16H, LU_RT3, LU_S14F,                                       // Do dcast
    LU_XSE, LU_XSMH, LU_XSML, LU_WSE, LU_WSMH, LU_WSML,                     // scales
    NLU };

// ---------------- fixed public tables ----------------
struct Tables {
    Table DM, SH, RM, R15;                       // increment-1 tables
    Table ASH, WIDTH, R10, R11, R12, R16, FH, REXP, SE, CREM13, CREM12, CTH;
};
static inline void decode_e4m3(uint32_t raw, int& exp_eff, int& sig_abs, int& sign) {
    sign = (raw >> 7) & 1;
    int eb = (raw >> 3) & 15, mant = raw & 7;
    sig_abs = eb != 0 ? (mant | 8) : mant;
    exp_eff = eb != 0 ? eb : 1;
}
static inline Tables build_tables() {
    Tables T;
    { // DM: fused decode+multiply, keyed by (a,b)
        std::vector<gl_t> a(65536), b(65536), eb(65536), mag(65536), sg(65536), pr(65536);
        for (uint32_t j = 0; j < 65536; j++) {
            uint32_t ca = j >> 8, cb = j & 255;
            int ea, siga, sna, ebx, sigb, snb;
            decode_e4m3(ca, ea, siga, sna); decode_e4m3(cb, ebx, sigb, snb);
            a[j] = ca; b[j] = cb;
            eb[j] = (gl_t)(ea + ebx - 2);
            mag[j] = (gl_t)((uint64_t)siga * sigb << 7);
            pr[j] = (siga != 0 && sigb != 0) ? 1 : 0;
            sg[j] = (pr[j] && (sna ^ snb)) ? 1 : 0;
        }
        T.DM = make_table({a, b, eb, mag, sg, pr});
    }
    { std::vector<gl_t> s(64), p(64);                       // SHIFT (products)
      for (uint32_t j = 0; j < 64; j++) { s[j] = j; p[j] = 1ULL << (j < 15 ? j : 15); }
      T.SH = make_table({s, p}); }
    { std::vector<gl_t> s(256), p(256);                     // ASHIFT (acc realign)
      for (uint32_t j = 0; j < 256; j++) { s[j] = j; p[j] = 1ULL << (j < 15 ? j : 15); }
      T.ASH = make_table({s, p}); }
    { std::vector<gl_t> p(65536), r(65536); uint32_t row = 0;   // REM
      for (uint32_t t = 0; t <= 15; t++)
          for (uint32_t x = 0; x < (1u << t); x++) { p[row] = 1ULL << t; r[row] = x; row++; }
      p[65535] = 1; r[65535] = 0;
      T.RM = make_table({p, r}); }
    { // WIDTH: wd in [0,20] -> plo=2^(wd-1) (0 for wd=0), phi=2^wd,
      // pdn=2^max(wd-14,0), pup=2^max(14-wd,0); rows 21..31 pad = row 0.
      std::vector<gl_t> w(32), plo(32), phi(32), pdn(32), pup(32);
      for (uint32_t j = 0; j < 32; j++) {
          uint32_t wd = j <= 20 ? j : 0;
          w[j] = wd; plo[j] = wd ? (1ULL << (wd - 1)) : 0; phi[j] = 1ULL << wd;
          pdn[j] = 1ULL << (wd > 14 ? wd - 14 : 0);
          pup[j] = 1ULL << (wd < 14 ? 14 - wd : 0);
      }
      T.WIDTH = make_table({w, plo, phi, pdn, pup}); }
    auto range = [](uint32_t n) { std::vector<gl_t> v(n); for (uint32_t j = 0; j < n; j++) v[j] = j;
                                  return make_table({v}); };
    T.R10 = range(1024); T.R11 = range(2048); T.R12 = range(4096);
    T.R15 = range(32768); T.R16 = range(65536);
    { std::vector<gl_t> v(2048);                            // FH: mfl high limb in [2^11, 2^12)
      for (uint32_t j = 0; j < 2048; j++) v[j] = 2048 + j;
      T.FH = make_table({v}); }
    { std::vector<gl_t> v(256);                             // REXP: normal fp32 exp [1,254]
      for (uint32_t j = 0; j < 256; j++) v[j] = (j >= 1 && j <= 254) ? j : 1;
      T.REXP = make_table({v}); }
    { std::vector<gl_t> v(256);                             // SE: scale exp [0,254] (no inf/nan)
      for (uint32_t j = 0; j < 256; j++) v[j] = j <= 254 ? j : 0;
      T.SE = make_table({v}); }
    { // CREM13: (c, rem) with rem < 2^(13+c). c=0 rows [0,8192), c=1 rows [8192,24576+8192)
      std::vector<gl_t> c(32768, 0), r(32768, 0);
      for (uint32_t x = 0; x < 8192; x++) { c[x] = 0; r[x] = x; }
      for (uint32_t x = 0; x < 16384; x++) { c[8192 + x] = 1; r[8192 + x] = x; }
      // rows 24576..32767 pad = (0,0)
      T.CREM13 = make_table({c, r}); }
    { // CREM12: (c, rt) with rt < 2^(12+c)
      std::vector<gl_t> c(16384, 0), r(16384, 0);
      for (uint32_t x = 0; x < 4096; x++) { c[x] = 0; r[x] = x; }
      for (uint32_t x = 0; x < 8192; x++) { c[4096 + x] = 1; r[4096 + x] = x; }
      T.CREM12 = make_table({c, r}); }
    { // CTH: (c, th) with th < 2^(10+c)
      std::vector<gl_t> c(4096, 0), t(4096, 0);
      for (uint32_t x = 0; x < 1024; x++) { c[x] = 0; t[x] = x; }
      for (uint32_t x = 0; x < 2048; x++) { c[1024 + x] = 1; t[1024 + x] = x; }
      T.CTH = make_table({c, t}); }
    return T;
}

// ---------------- dimensions ----------------
struct Dims {
    uint32_t B, K, N;                 // real
    uint32_t NG;                      // real groups = ceil(K/32)
    uint32_t Bpad, Npad, Gpad, Kpad, Opad;
    uint32_t lb, ln, lg, lo;          // log2 of pads; lo = ln + lb
    size_t G, P;                      // padded group / product counts
    uint32_t lgi, lp;                 // log2(G), log2(P)
};
static inline Dims make_dims(uint32_t B, uint32_t K, uint32_t N) {
    Dims d; d.B = B; d.K = K; d.N = N;
    d.NG = (K + 31) / 32;
    d.Bpad = pow2_at_least(B < 2 ? 2 : B);
    d.Npad = pow2_at_least(N < 2 ? 2 : N);
    d.Gpad = pow2_at_least(d.NG + 1);         // >= one phantom group after the last real one
    d.Kpad = 32 * d.Gpad;
    d.Opad = d.Bpad * d.Npad;
    d.lb = ilog2(d.Bpad); d.ln = ilog2(d.Npad); d.lg = ilog2(d.Gpad);
    d.lo = d.lb + d.ln;
    d.G = (size_t)d.Gpad * d.Opad; d.P = 32 * d.G;
    d.lgi = d.lo + d.lg; d.lp = d.lgi + 5;
    return d;
}

// ---------------- golden layer input ----------------
struct Golden {
    uint32_t B, K, N;
    std::vector<uint8_t> x, w;        // B*K, N*K codes
    std::vector<uint32_t> xs, ws;     // fp32 bits
    std::vector<uint16_t> y;          // golden bf16 bits (B*N)
};
static inline bool load_layers(const char* path, std::vector<Golden>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[2];
    if (fread(hdr, 8, 2, f) != 2 || hdr[0] != 0x484B4C59) { fclose(f); return false; }
    out.resize(hdr[1]);
    for (auto& L : out) {
        int64_t dims[3];
        if (fread(dims, 8, 3, f) != 3) { fclose(f); return false; }
        L.B = dims[0]; L.K = dims[1]; L.N = dims[2];
        L.x.resize((size_t)L.B * L.K); L.w.resize((size_t)L.N * L.K);
        L.xs.resize(L.B); L.ws.resize(L.N); L.y.resize((size_t)L.B * L.N);
        if (fread(L.x.data(), 1, L.x.size(), f) != L.x.size()) { fclose(f); return false; }
        if (fread(L.w.data(), 1, L.w.size(), f) != L.w.size()) { fclose(f); return false; }
        if (fread(L.xs.data(), 4, L.B, f) != L.B) { fclose(f); return false; }
        if (fread(L.ws.data(), 4, L.N, f) != L.N) { fclose(f); return false; }
        if (fread(L.y.data(), 2, L.y.size(), f) != L.y.size()) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

// ---------------- witness ----------------
struct LayerWit {
    Dims d;
    std::vector<uint8_t> xcodes, wcodes;      // padded operand code arrays (X/W layout)
    std::vector<gl_t> xsb, wsb;               // padded fp32-bit scale columns
    std::vector<uint16_t> Y;                  // computed bf16 output, REAL B*N
    std::vector<gl_t> dp[NDP], dg[NDG], dob[NDO], db[NDS], dn[NDS];
    std::vector<gl_t> va, vb;                 // virtual a,b product code arrays (Dp)
    std::vector<uint32_t> lidx[NLU];          // per-lookup table-row indices
};

static inline int bitwidth64(uint64_t x) { int w = 0; while (x) { w++; x >>= 1; } return w; }
static inline gl_t inv_or0(uint64_t x) { return x ? gl_inv((gl_t)x) : 0; }

// fp32 helpers for the independent float cross-check of the witness
static inline float bits_to_f32(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static inline uint32_t f32_to_bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static inline uint16_t bf16_of_f32bits(uint32_t b) {
    uint32_t rnd = ((b >> 16) & 1) + 0x7FFF;
    return (uint16_t)((b + rnd) >> 16);
}

// Selftest-only witness tampering: each mode applies ONE semantic forgery at a
// chosen location and then continues the replay HONESTLY from the forged value,
// so the resulting witness is consistent everywhere EXCEPT the one sub-argument
// that must catch it (the prover then claims the forged layer's own Y).
enum { TM_NONE = 0,
       TM_ROUND_UP,   // per-product align rounds up instead of truncating (q+1, r-=pw)
       TM_MAXEXP,     // group max_exp overstated by 1
       TM_WD_DOWN,    // normalize bit-width understated by 1
       TM_AQ_UP,      // acc realign rounds up instead of truncating (aq+1, ar-=apw)
       TM_STATE };    // accumulator state teleported (+1024 in sig entering group g)
struct Tamper { int mode = TM_NONE; uint32_t o = 0, g = 0, kk = 0; };

// Generate the FULL layer witness by replaying the exact Hawkeye recurrence
// over the padded grids.  Throws if the layer is outside the supported scale
// domain.
static inline LayerWit gen_witness(const Golden& L, bool check_float = true,
                                   const Tamper* tm = nullptr,
                                   const std::vector<uint8_t>* xpad_ovr = nullptr) {
    LayerWit wt; Dims& d = wt.d; d = make_dims(L.B, L.K, L.N);
    // padded operands (zero codes / zero scales beyond the real grid)
    wt.xcodes.assign((size_t)d.Bpad * d.Kpad, 0);
    wt.wcodes.assign((size_t)d.Npad * d.Kpad, 0);
    for (uint32_t b = 0; b < L.B; b++)
        for (uint32_t k = 0; k < L.K; k++) wt.xcodes[(size_t)b * d.Kpad + k] = L.x[(size_t)b * L.K + k];
    for (uint32_t n = 0; n < L.N; n++)
        for (uint32_t k = 0; k < L.K; k++) wt.wcodes[(size_t)n * d.Kpad + k] = L.w[(size_t)n * L.K + k];
    // selftest-only: replace the PADDED X code array (padding-smuggle forgeries:
    // the replay below stays self-consistent with the smuggled codes)
    if (xpad_ovr) {
        if (xpad_ovr->size() != wt.xcodes.size())
            throw std::runtime_error("hwl: xpad override size");
        wt.xcodes = *xpad_ovr;
    }
    wt.xsb.assign(d.Bpad, 0); wt.wsb.assign(d.Npad, 0);
    for (uint32_t b = 0; b < L.B; b++) wt.xsb[b] = L.xs[b];
    for (uint32_t n = 0; n < L.N; n++) wt.wsb[n] = L.ws[n];
    wt.Y.assign((size_t)L.B * L.N, 0);

    for (int c = 0; c < NDP; c++) wt.dp[c].assign(d.P, 0);
    for (int c = 0; c < NDG; c++) wt.dg[c].assign(d.G, 0);
    for (int c = 0; c < NDO; c++) wt.dob[c].assign(d.Opad, 0);
    for (int c = 0; c < NDS; c++) { wt.db[c].assign(d.Bpad, 0); wt.dn[c].assign(d.Npad, 0); }
    wt.va.assign(d.P, 0); wt.vb.assign(d.P, 0);
    for (int i = 0; i < NLU; i++) {
        size_t sz = d.P;
        if (i >= LU_ASH && i <= LU_S14) sz = d.G;
        else if (i >= LU_M1REM && i <= LU_RT3) sz = d.Opad;
        else if (i == LU_S14F) sz = d.Opad;
        else if (i >= LU_XSE && i <= LU_XSML) sz = d.Bpad;
        else if (i >= LU_WSE && i <= LU_WSML) sz = d.Npad;
        wt.lidx[i].assign(sz, 0);
    }
    auto rm_idx = [](uint64_t pw, uint64_t r) { return (uint32_t)((pw - 1) + r); };

    // ---- scale decomposition (Db / Dn) ----
    auto fill_scale = [&](std::vector<gl_t>* S, const std::vector<gl_t>& bits,
                          int lu_se, int lu_smh, int lu_sml, const char* which) {
        for (size_t i = 0; i < bits.size(); i++) {
            uint32_t v = (uint32_t)bits[i];
            uint32_t ss = v >> 31, se = (v >> 23) & 255, sm = v & 0x7FFFFF;
            if (se == 255) throw std::runtime_error(std::string("hwl: inf/nan ") + which + "-scale unsupported");
            if (se == 0 && sm != 0) throw std::runtime_error(std::string("hwl: subnormal ") + which + "-scale unsupported");
            S[S_SS][i] = ss; S[S_SE][i] = se; S[S_SEI][i] = inv_or0(se);
            S[S_ZS][i] = se == 0 ? 1 : 0;
            S[S_SMH][i] = sm >> 12; S[S_SML][i] = sm & 4095;
            wt.lidx[lu_se][i] = se; wt.lidx[lu_smh][i] = sm >> 12; wt.lidx[lu_sml][i] = sm & 4095;
        }
    };
    fill_scale(wt.db, wt.xsb, LU_XSE, LU_XSMH, LU_XSML, "x");
    fill_scale(wt.dn, wt.wsb, LU_WSE, LU_WSMH, LU_WSML, "w");

    // ---- per-output replay ----
    for (uint32_t o = 0; o < d.Opad; o++) {
        uint32_t b = o >> d.ln, n = o & (d.Npad - 1);
        // accumulator state (in biased-exponent gfloat form)
        int64_t sgn = 0, bexp = 0, sig = 0;
        for (uint32_t g = 0; g < d.Gpad; g++) {
            size_t gi = (size_t)g * d.Opad + o;
            if (tm && tm->mode == TM_STATE && o == tm->o && g == tm->g) sig += 1024;
            int64_t az = sig == 0 ? 1 : 0;
            int64_t baee = az ? 0 : bexp;
            wt.dg[G_ASIG][gi] = (gl_t)sig; wt.dg[G_BEXP][gi] = (gl_t)bexp;
            wt.dg[G_ASGN][gi] = (gl_t)sgn; wt.dg[G_AZ][gi] = (gl_t)az;
            wt.dg[G_AINV][gi] = inv_or0((uint64_t)sig);
            // decode the 32 products
            int64_t bpe[32], mag[32], sg_[32], pr_[32]; uint32_t ca_[32], cb_[32];
            int64_t bgmax = baee;
            for (uint32_t kk = 0; kk < 32; kk++) {
                uint32_t k = g * 32 + kk;
                uint32_t ca = wt.xcodes[(size_t)b * d.Kpad + k];
                uint32_t cb = wt.wcodes[(size_t)n * d.Kpad + k];
                int ea, siga, sna, ebx, sigb, snb;
                decode_e4m3(ca, ea, siga, sna); decode_e4m3(cb, ebx, sigb, snb);
                ca_[kk] = ca; cb_[kk] = cb;
                pr_[kk] = (siga && sigb) ? 1 : 0;
                int64_t eb = ea + ebx - 2;
                bpe[kk] = eb + 127;
                mag[kk] = (int64_t)siga * sigb << 7;
                sg_[kk] = (pr_[kk] && (sna ^ snb)) ? 1 : 0;
                if (pr_[kk] && bpe[kk] > bgmax) bgmax = bpe[kk];
            }
            if (tm && tm->mode == TM_MAXEXP && o == tm->o && g == tm->g) bgmax += 1;
            // attainment selector: acc first, else first present product at max
            int64_t asel = (baee == bgmax) ? 1 : 0;
            int sel_kk = -1;
            if (!asel) for (uint32_t kk = 0; kk < 32; kk++)
                if (pr_[kk] && bpe[kk] == bgmax) { sel_kk = (int)kk; break; }
            if (!asel && sel_kk < 0) {
                if (tm) asel = 1;                 // forged max has no attainer
                else throw std::runtime_error("hwl: max unattained");
            }
            // product rows
            int64_t csum = 0;
            for (uint32_t kk = 0; kk < 32; kk++) {
                size_t p = gi * 32 + kk;
                int64_t eb = bpe[kk] - 127;
                int64_t sh = pr_[kk] ? bgmax - bpe[kk] : 0;
                if (sh > 63) throw std::runtime_error("hwl: product shift > 63");
                int64_t shc = sh < 15 ? sh : 15, pw = (int64_t)1 << shc;
                int64_t q = mag[kk] >> shc, r = mag[kk] - (q << shc);
                if (tm && tm->mode == TM_ROUND_UP && o == tm->o && g == tm->g && kk == tm->kk
                    && r > 0) { q += 1; r -= pw; }
                int64_t al = pr_[kk] ? (sg_[kk] ? -q : q) : 0;
                csum += al;
                wt.va[p] = ca_[kk]; wt.vb[p] = cb_[kk];
                wt.dp[P_EB][p] = (gl_t)eb; wt.dp[P_MAG][p] = (gl_t)mag[kk];
                wt.dp[P_SG][p] = (gl_t)sg_[kk]; wt.dp[P_PR][p] = (gl_t)pr_[kk];
                wt.dp[P_SH][p] = (gl_t)sh; wt.dp[P_PW][p] = (gl_t)pw;
                wt.dp[P_Q][p] = (gl_t)q; wt.dp[P_R][p] = enc(r);
                wt.dp[P_AL][p] = enc(al);
                wt.dp[P_SEL][p] = (sel_kk == (int)kk) ? 1 : 0;
                wt.lidx[LU_DM][p] = (uint32_t)(ca_[kk] * 256 + cb_[kk]);
                wt.lidx[LU_SH][p] = (uint32_t)sh;
                wt.lidx[LU_RM][p] = r >= 0 ? rm_idx(pw, r) : 0;
                wt.lidx[LU_Q15][p] = (uint32_t)q;
            }
            // acc realign
            int64_t abase = sig >> 10, alo = sig & 1023;
            int64_t ash = bgmax - baee;
            if (ash < 0 || ash > 255) throw std::runtime_error("hwl: acc shift out of [0,255]");
            int64_t ashc = ash < 15 ? ash : 15, apw = (int64_t)1 << ashc;
            int64_t aq = abase >> ashc, ar = abase - (aq << ashc);
            if (tm && tm->mode == TM_AQ_UP && o == tm->o && g == tm->g) { aq += 1; ar -= apw; }
            int64_t aal = sgn ? -aq : aq;
            int64_t total = csum + aal;
            int64_t tsgn = total < 0 ? 1 : 0, tmag = total < 0 ? -total : total;
            int64_t tz = tmag == 0 ? 1 : 0;
            int wd = bitwidth64((uint64_t)tmag);
            if (wd > 20) throw std::runtime_error("hwl: total width > 20");
            if (tm && tm->mode == TM_WD_DOWN && o == tm->o && g == tm->g && wd >= 2) wd -= 1;
            int64_t plo = wd ? (int64_t)1 << (wd - 1) : 0, phi = (int64_t)1 << wd;
            int64_t pdn = (int64_t)1 << (wd > 14 ? wd - 14 : 0);
            int64_t pup = (int64_t)1 << (wd < 14 ? 14 - wd : 0);
            int64_t u1 = tmag - plo, u2 = phi - 1 - tmag;
            int64_t u1h = u1 >= 0 ? u1 >> 10 : 0, u1l = u1 >= 0 ? u1 & 1023 : 0;
            int64_t u2h = u2 >= 0 ? u2 >> 10 : 0, u2l = u2 >= 0 ? u2 & 1023 : 0;
            int64_t nr = wd > 14 ? tmag & (pdn - 1) : 0;
            int64_t s14 = wd > 14 ? tmag >> (wd - 14) : tmag << (14 - wd);
            if (s14 * pdn + nr != tmag * pup) throw std::runtime_error("hwl: s14 identity");
            wt.dg[G_ABASE][gi] = (gl_t)abase; wt.dg[G_ALO][gi] = (gl_t)alo;
            wt.dg[G_ASH][gi] = (gl_t)ash; wt.dg[G_APW][gi] = (gl_t)apw;
            wt.dg[G_AQ][gi] = (gl_t)aq; wt.dg[G_AR][gi] = enc(ar);
            wt.dg[G_MAX][gi] = (gl_t)bgmax; wt.dg[G_ASEL][gi] = (gl_t)asel;
            wt.dg[G_CSUM][gi] = enc(csum);
            wt.dg[G_TSGN][gi] = (gl_t)tsgn; wt.dg[G_TMAG][gi] = (gl_t)tmag;
            wt.dg[G_TZ][gi] = (gl_t)tz; wt.dg[G_TINV][gi] = inv_or0((uint64_t)tmag);
            wt.dg[G_WD][gi] = (gl_t)wd; wt.dg[G_PLO][gi] = (gl_t)plo; wt.dg[G_PHI][gi] = (gl_t)phi;
            wt.dg[G_PDN][gi] = (gl_t)pdn; wt.dg[G_PUP][gi] = (gl_t)pup;
            wt.dg[G_NR][gi] = (gl_t)nr; wt.dg[G_S14][gi] = (gl_t)s14;
            wt.dg[G_U1H][gi] = (gl_t)u1h; wt.dg[G_U1L][gi] = (gl_t)u1l;
            wt.dg[G_U2H][gi] = (gl_t)u2h; wt.dg[G_U2L][gi] = (gl_t)u2l;
            wt.lidx[LU_ASH][gi] = (uint32_t)ash;
            wt.lidx[LU_ARM][gi] = ar >= 0 ? rm_idx(apw, ar) : 0;
            wt.lidx[LU_AQ][gi] = (uint32_t)aq;
            wt.lidx[LU_ABASE][gi] = (uint32_t)abase;
            wt.lidx[LU_ALO][gi] = (uint32_t)alo;
            wt.lidx[LU_WD][gi] = (uint32_t)wd;
            wt.lidx[LU_U1H][gi] = (uint32_t)u1h; wt.lidx[LU_U1L][gi] = (uint32_t)u1l;
            wt.lidx[LU_U2H][gi] = (uint32_t)u2h; wt.lidx[LU_U2L][gi] = (uint32_t)u2l;
            wt.lidx[LU_NR][gi] = rm_idx(pdn, nr);
            wt.lidx[LU_S14][gi] = (uint32_t)s14;
            // state update
            if (!tz) { sgn = tsgn; bexp = bgmax + wd - 14; sig = s14 << 10; }
            else { sgn = 0; bexp = 0; sig = 0; }
        }
        // ---- P6 per-output witness ----
        int64_t fsgn = sgn, fbexp = bexp, fsig = sig;
        int64_t s14f = fsig >> 10;
        if ((s14f << 10) != fsig) throw std::runtime_error("hwl: fsig low bits");
        int64_t fz = fsig == 0 ? 1 : 0;
        wt.dob[O_S14F][o] = (gl_t)s14f; wt.dob[O_BEXP][o] = (gl_t)fbexp;
        wt.dob[O_SGN][o] = (gl_t)fsgn; wt.dob[O_FZ][o] = (gl_t)fz;
        wt.dob[O_FINV][o] = inv_or0((uint64_t)s14f);
        wt.lidx[LU_S14F][o] = (uint32_t)s14f;
        uint32_t xb = (uint32_t)wt.xsb[b], wb2 = (uint32_t)wt.wsb[n];
        uint32_t ssx = xb >> 31, sex = (xb >> 23) & 255, smx = xb & 0x7FFFFF;
        uint32_t ssw = wb2 >> 31, sew = (wb2 >> 23) & 255, smw = wb2 & 0x7FFFFF;
        int64_t zsx = sex == 0 ? 1 : 0, zsw = sew == 0 ? 1 : 0;
        int64_t z1 = (fz || zsx) ? 1 : 0, z2 = (z1 || zsw) ? 1 : 0;
        int64_t s1o = fsgn ^ (int64_t)ssx, s2o = s1o ^ (int64_t)ssw;
        // mul1: gfloat (s14f * 2^10 mantissa, exp fbexp-12) * xs
        int64_t c1 = 0, rem1 = 0, rb1 = 0, rt1 = 0, zrt1 = 1, rup1 = 0, c21 = 0;
        int64_t fh1 = 2048, fl1 = 0, flh1 = 0, l01 = 0, m1o = 1 << 23, e1o = 1;
        if (!z1) {
            uint64_t m2 = (1u << 23) | smx;
            uint64_t mm1 = (uint64_t)s14f * m2;               // < 2^38
            c1 = mm1 >= (1ULL << 37) ? 1 : 0;
            uint64_t mfl1 = mm1 >> (13 + c1);
            rem1 = (int64_t)(mm1 & (((uint64_t)1 << (13 + c1)) - 1));
            fh1 = (int64_t)(mfl1 >> 12); fl1 = (int64_t)(mfl1 & 4095);
            flh1 = fl1 >> 1; l01 = fl1 & 1;
            rb1 = rem1 >> (12 + c1); rt1 = rem1 & (((int64_t)1 << (12 + c1)) - 1);
            zrt1 = rt1 == 0 ? 1 : 0;
            rup1 = (rb1 && (!zrt1 || l01)) ? 1 : 0;
            int64_t mr1 = (int64_t)mfl1 + rup1;
            c21 = mr1 == (1 << 24) ? 1 : 0;
            m1o = mr1 - (c21 << 23);
            e1o = (fbexp - 12) + (int64_t)sex - 127 + c1 + c21;
            if (e1o < 1 || e1o > 254)
                throw std::runtime_error("hwl: mul1 out of normal fp32 range");
        }
        wt.dob[O_C1][o] = (gl_t)c1; wt.dob[O_REM1][o] = (gl_t)rem1;
        wt.dob[O_RB1][o] = (gl_t)rb1; wt.dob[O_RT1][o] = (gl_t)rt1;
        wt.dob[O_RTI1][o] = inv_or0((uint64_t)rt1); wt.dob[O_ZRT1][o] = (gl_t)zrt1;
        wt.dob[O_FH1][o] = (gl_t)fh1; wt.dob[O_FL1][o] = (gl_t)fl1;
        wt.dob[O_FLH1][o] = (gl_t)flh1; wt.dob[O_L01][o] = (gl_t)l01;
        wt.dob[O_RUP1][o] = (gl_t)rup1; wt.dob[O_C21][o] = (gl_t)c21;
        wt.dob[O_M1][o] = (gl_t)m1o; wt.dob[O_E1][o] = (gl_t)e1o;
        wt.dob[O_S1][o] = (gl_t)s1o; wt.dob[O_Z1][o] = (gl_t)z1;
        wt.lidx[LU_M1REM][o] = (uint32_t)(c1 ? 8192 + rem1 : rem1);
        wt.lidx[LU_M1RT][o] = (uint32_t)(c1 ? 4096 + rt1 : rt1);
        wt.lidx[LU_M1FH][o] = (uint32_t)(fh1 - 2048);
        wt.lidx[LU_M1FL][o] = (uint32_t)fl1;
        wt.lidx[LU_M1FLH][o] = (uint32_t)flh1;
        wt.lidx[LU_M1E][o] = (uint32_t)e1o;
        // mul2: y1 * ws (full 24x24 mantissa product)
        int64_t c2 = 0, rb2 = 0, th2 = 0, tl2 = 0, zrt2 = 1, rup2 = 0, c22 = 0;
        int64_t fh2 = 2048, fl2 = 0, flh2 = 0, l02 = 0, m2o = 1 << 23, e2o = 1;
        if (!z2) {
            uint64_t m3 = (1u << 23) | smw;
            uint64_t mm2 = (uint64_t)m1o * m3;                // < 2^48
            c2 = mm2 >= (1ULL << 47) ? 1 : 0;
            uint64_t mfl2 = mm2 >> (23 + c2);
            uint64_t rem2 = mm2 & (((uint64_t)1 << (23 + c2)) - 1);
            fh2 = (int64_t)(mfl2 >> 12); fl2 = (int64_t)(mfl2 & 4095);
            flh2 = fl2 >> 1; l02 = fl2 & 1;
            rb2 = (int64_t)(rem2 >> (22 + c2));
            uint64_t rt2 = rem2 & (((uint64_t)1 << (22 + c2)) - 1);
            th2 = (int64_t)(rt2 >> 12); tl2 = (int64_t)(rt2 & 4095);
            zrt2 = rt2 == 0 ? 1 : 0;
            rup2 = (rb2 && (!zrt2 || l02)) ? 1 : 0;
            int64_t mr2 = (int64_t)mfl2 + rup2;
            c22 = mr2 == (1 << 24) ? 1 : 0;
            m2o = mr2 - (c22 << 23);
            e2o = e1o + (int64_t)sew - 127 + c2 + c22;
            if (e2o < 1 || e2o > 254)
                throw std::runtime_error("hwl: mul2 out of normal fp32 range");
        }
        wt.dob[O_C2][o] = (gl_t)c2; wt.dob[O_RB2][o] = (gl_t)rb2;
        wt.dob[O_TH2][o] = (gl_t)th2; wt.dob[O_TL2][o] = (gl_t)tl2;
        wt.dob[O_RTI2][o] = inv_or0((uint64_t)(th2 * 4096 + tl2)); wt.dob[O_ZRT2][o] = (gl_t)zrt2;
        wt.dob[O_FH2][o] = (gl_t)fh2; wt.dob[O_FL2][o] = (gl_t)fl2;
        wt.dob[O_FLH2][o] = (gl_t)flh2; wt.dob[O_L02][o] = (gl_t)l02;
        wt.dob[O_RUP2][o] = (gl_t)rup2; wt.dob[O_C22][o] = (gl_t)c22;
        wt.dob[O_M2][o] = (gl_t)m2o; wt.dob[O_E2][o] = (gl_t)e2o;
        wt.dob[O_S2][o] = (gl_t)s2o; wt.dob[O_Z2][o] = (gl_t)z2;
        wt.lidx[LU_M2TH][o] = (uint32_t)(c2 ? 1024 + th2 : th2);
        wt.lidx[LU_M2TL][o] = (uint32_t)tl2;
        wt.lidx[LU_M2FH][o] = (uint32_t)(fh2 - 2048);
        wt.lidx[LU_M2FL][o] = (uint32_t)fl2;
        wt.lidx[LU_M2FLH][o] = (uint32_t)flh2;
        wt.lidx[LU_M2E][o] = (uint32_t)e2o;
        // bf16 downcast = integer RNE of the fp32 bits >> 16
        uint64_t b32 = ((uint64_t)s2o << 31)
                     | (z2 ? 0 : (((uint64_t)e2o << 23) | (uint64_t)(m2o - (1 << 23))));
        int64_t q16 = (int64_t)(b32 >> 16), rem3 = (int64_t)(b32 & 0xFFFF);
        int64_t rb3 = rem3 >> 15, rt3 = rem3 & 0x7FFF;
        int64_t zrt3 = rt3 == 0 ? 1 : 0, l03 = q16 & 1, q16h = q16 >> 1;
        int64_t rup3 = (rb3 && (!zrt3 || l03)) ? 1 : 0;
        int64_t yb = q16 + rup3;
        wt.dob[O_Q16][o] = (gl_t)q16; wt.dob[O_Q16H][o] = (gl_t)q16h;
        wt.dob[O_L03][o] = (gl_t)l03; wt.dob[O_RB3][o] = (gl_t)rb3;
        wt.dob[O_RT3][o] = (gl_t)rt3; wt.dob[O_RTI3][o] = inv_or0((uint64_t)rt3);
        wt.dob[O_ZRT3][o] = (gl_t)zrt3; wt.dob[O_RUP3][o] = (gl_t)rup3;
        wt.dob[O_YB][o] = (gl_t)yb;
        wt.lidx[LU_Q16][o] = (uint32_t)q16;
        wt.lidx[LU_Q16H][o] = (uint32_t)q16h;
        wt.lidx[LU_RT3][o] = (uint32_t)rt3;
        // independent float cross-check of the whole P6 path
        if (check_float) {
            uint32_t gb = 0;
            if (fsig != 0) {
                int64_t exp32 = fbexp - 139 + 127;           // = fbexp - 12
                gb = ((uint32_t)fsgn << 31) | ((uint32_t)exp32 << 23)
                   | (uint32_t)(fsig - (1 << 23));
            }
            float yf = bits_to_f32(gb) * bits_to_f32(xb) * bits_to_f32(wb2);
            uint16_t ybf = bf16_of_f32bits(f32_to_bits(yf));
            if ((uint16_t)yb != ybf) throw std::runtime_error("hwl: witness != float replay");
        }
        if (b < L.B && n < L.N) wt.Y[(size_t)b * L.N + n] = (uint16_t)yb;
    }
    return wt;
}

// ==================== proof machinery ====================

struct Msg5 { gl_t s0, s1, s2, s3, s4; };
static inline gl_t quartic_eval(const Msg5& m, gl_t t) {
    gl_t i2 = gl_inv(2ULL), i6 = gl_inv(6ULL), i24 = gl_inv(24ULL), i4 = gl_inv(4ULL);
    gl_t t1 = gl_sub(t,1ULL), t2 = gl_sub(t,2ULL), t3 = gl_sub(t,3ULL), t4 = gl_sub(t,4ULL);
    auto neg = [](gl_t x){ return gl_sub(0ULL, x); };
    gl_t L0 = gl_mul(gl_mul(gl_mul(t1,t2),gl_mul(t3,t4)), i24);
    gl_t L1 = neg(gl_mul(gl_mul(gl_mul(t,t2),gl_mul(t3,t4)), i6));
    gl_t L2 = gl_mul(gl_mul(gl_mul(t,t1),gl_mul(t3,t4)), i4);
    gl_t L3 = neg(gl_mul(gl_mul(gl_mul(t,t1),gl_mul(t2,t4)), i6));
    gl_t L4 = gl_mul(gl_mul(gl_mul(t,t1),gl_mul(t2,t3)), i24);
    gl_t acc = gl_mul(m.s0, L0);
    acc = gl_add(acc, gl_mul(m.s1, L1)); acc = gl_add(acc, gl_mul(m.s2, L2));
    acc = gl_add(acc, gl_mul(m.s3, L3)); acc = gl_add(acc, gl_mul(m.s4, L4));
    return acc;
}

// generic quartic sumcheck of  sum_b F(cols(b))  (deg(F) <= 4 per variable pair)
typedef std::function<gl_t(const gl_t*)> CFn;
static inline std::vector<gl_t> sc5_prove(fs::Transcript& tr, const char* tag,
        std::vector<std::vector<gl_t>> cols, const CFn& F, std::vector<Msg5>& msgs) {
    uint32_t v = ilog2(cols[0].size());
    size_t nc = cols.size();
    std::vector<gl_t> r(v);
    for (uint32_t rd = 0; rd < v; rd++) {
        size_t half = cols[0].size() / 2;
        gl_t s[5] = {0,0,0,0,0};
        // parallel partial sums combine to the SAME field elements (exact
        // arithmetic, order-independent); serial for small rounds
        const int P = half >= 65536 ? 128 : 1;
        std::vector<std::array<gl_t, 5>> part(P, {0, 0, 0, 0, 0});
        #pragma omp parallel for schedule(static) if (P > 1)
        for (int p = 0; p < P; p++) {
            size_t lo = half * p / P, hi = half * (p + 1) / P;
            std::vector<gl_t> cur(nc), dd(nc);
            std::array<gl_t, 5>& sp = part[p];
            for (size_t i = lo; i < hi; i++) {
                for (size_t k = 0; k < nc; k++) { cur[k] = cols[k][2*i]; dd[k] = gl_sub(cols[k][2*i+1], cur[k]); }
                for (int t = 0; t < 5; t++) {
                    sp[t] = gl_add(sp[t], F(cur.data()));
                    if (t < 4) for (size_t k = 0; k < nc; k++) cur[k] = gl_add(cur[k], dd[k]);
                }
            }
        }
        for (int p = 0; p < P; p++)
            for (int t = 0; t < 5; t++) s[t] = gl_add(s[t], part[p][t]);
        Msg5 m{s[0], s[1], s[2], s[3], s[4]};
        msgs.push_back(m); tr.absorb(tag, &m, sizeof m);
        gl_t a = chal(tr); r[rd] = a;
        for (auto& c : cols) bind_lsb(c, a);
    }
    return r;
}
// -------- borrow-through-round-0 column sources --------
// A zero-check over the P-domain copies ~10 committed columns (5+ GB augmented
// at llama-68m dims) just to consume them; ColSrc lets those columns be
// BORROWED: round 0 reads the committed copy in place and the first bind
// writes the (halved) owned vector.  Messages are identical.
struct ColSrc {
    std::vector<gl_t> own; const std::vector<gl_t>* bor = nullptr;
    ColSrc() {}
    ColSrc(std::vector<gl_t>&& v) : own(std::move(v)) {}
    explicit ColSrc(const std::vector<gl_t>* p) : bor(p) {}
    const std::vector<gl_t>& get() const { return bor ? *bor : own; }
};
static inline std::vector<gl_t> sc5_prove_srcs(fs::Transcript& tr, const char* tag,
        std::vector<ColSrc> cols, const CFn& F, std::vector<Msg5>& msgs) {
    uint32_t v = ilog2(cols[0].get().size());
    size_t nc = cols.size();
    std::vector<gl_t> r(v);
    for (uint32_t rd = 0; rd < v; rd++) {
        size_t half = cols[0].get().size() / 2;
        gl_t s[5] = {0,0,0,0,0};
        const int P = half >= 65536 ? 128 : 1;
        std::vector<std::array<gl_t, 5>> part(P, {0, 0, 0, 0, 0});
        #pragma omp parallel for schedule(static) if (P > 1)
        for (int p = 0; p < P; p++) {
            size_t lo = half * p / P, hi = half * (p + 1) / P;
            std::vector<gl_t> cur(nc), dd(nc);
            std::vector<const gl_t*> cp(nc);
            for (size_t k = 0; k < nc; k++) cp[k] = cols[k].get().data();
            std::array<gl_t, 5>& sp = part[p];
            for (size_t i = lo; i < hi; i++) {
                for (size_t k = 0; k < nc; k++) { cur[k] = cp[k][2*i]; dd[k] = gl_sub(cp[k][2*i+1], cur[k]); }
                for (int t = 0; t < 5; t++) {
                    sp[t] = gl_add(sp[t], F(cur.data()));
                    if (t < 4) for (size_t k = 0; k < nc; k++) cur[k] = gl_add(cur[k], dd[k]);
                }
            }
        }
        for (int p = 0; p < P; p++)
            for (int t = 0; t < 5; t++) s[t] = gl_add(s[t], part[p][t]);
        Msg5 m{s[0], s[1], s[2], s[3], s[4]};
        msgs.push_back(m); tr.absorb(tag, &m, sizeof m);
        gl_t a = chal(tr); r[rd] = a;
        for (auto& c : cols) {
            const gl_t* src = c.get().data();
            std::vector<gl_t> nf(half);
            #pragma omp parallel for schedule(static) if (half >= 65536)
            for (size_t i = 0; i < half; i++)
                nf[i] = gl_add(src[2*i], gl_mul(a, gl_sub(src[2*i+1], src[2*i])));
            c.own = std::move(nf); c.bor = nullptr;
        }
    }
    return r;
}
// GPU dispatch of ColSrc columns (identical messages/transcript to sc5_run)
template <typename FF>
static inline std::vector<gl_t> sc5_run_srcs(fs::Transcript& tr, const char* tag,
        std::vector<ColSrc>&& cols, const gl_t* lam, uint32_t nlam,
        const CFn& Fhost, std::vector<Msg5>& msgs) {
    size_t N = cols[0].get().size();
    if (p3fri::g_gpu_merkle && N >= (1u << 16)) {
        std::vector<gl_t*> dc(cols.size());
        for (size_t i = 0; i < cols.size(); i++) {
            dc[i] = p3bf::dmalloc(N, "sc5:col");
            cudaMemcpy(dc[i], cols[i].get().data(), N * 8, cudaMemcpyHostToDevice);
            std::vector<gl_t>().swap(cols[i].own); cols[i].bor = nullptr;
        }
        std::vector<gl_t> r = p3sg::sc_prove_gpu<FF, Msg5, 5, 32>(
            tr, tag, dc, (uint32_t)N, lam, nlam, msgs);
        for (auto p : dc) cudaFreeAsync(p, 0);
        return r;
    }
    return sc5_prove_srcs(tr, tag, std::move(cols), Fhost, msgs);
}

static inline bool sc5_verify(const std::vector<Msg5>& msgs, uint32_t v, gl_t claim0,
        fs::Transcript& tr, const char* tag, std::vector<gl_t>& r_out, gl_t& claim_out) {
    if (msgs.size() != v) return false;
    gl_t claim = claim0;
    r_out.clear();
    for (uint32_t rd = 0; rd < v; rd++) {
        const Msg5& m = msgs[rd];
        if (gl_add(m.s0, m.s1) != claim) return false;
        tr.absorb(tag, &m, sizeof m);
        gl_t a = chal(tr); r_out.push_back(a);
        claim = quartic_eval(m, a);
    }
    claim_out = claim;
    return true;
}

static inline std::vector<gl_t> beq(const std::vector<gl_t>& z) {
    return (z.size() >= 16 && p3fri::g_gpu_merkle) ? p3bf::build_eq_gpu(z) : p3bf::build_eq(z);
}
// prover-side opening CLAIM: absorb the evaluation and defer the proof to the
// end-of-protocol batched opening of the column's size class (p3_batchopen).
static inline gl_t claimc(fs::Transcript& tr, p3bo::PLedger& lg, const Col& c,
                          const std::vector<gl_t>& z) {
    gl_t y = (z.size() >= 16 && p3fri::g_gpu_merkle)
           ? p3bf::eval_h_gpu(c.v, z) : p3bf::eval_h(c.v, p3bf::build_eq(z));
    tr.absorb("hwl-y", &y, sizeof y);
    lg.add(&c.v, c.root, z, y, c.sseed);
    return y;
}
// verifier-side mirror: absorb the claimed evaluation and register the
// obligation; the per-class batch proofs at the end back every claim.
static inline gl_t claimv(fs::Transcript& tr, p3bo::VLedger& lg, const p3fri::Hash& root,
                          const std::vector<gl_t>& z, gl_t y) {
    tr.absorb("hwl-y", &y, sizeof y);
    lg.add(root, z, y);
    return y;
}

// ---------------- constraint functions (shared prover terminal / verifier) ----------------
// Dp zero-check: v = [E, dp cols 0..NDP-1, bgmaxP]; lam[6]
static inline __host__ __device__ gl_t F_dp(const gl_t* v, const gl_t* lam) {
    const gl_t* c = v + 1; gl_t gmax = v[1 + NDP];
    gl_t C1 = gl_sub(gl_add(gl_mul(c[P_Q], c[P_PW]), c[P_R]), c[P_MAG]);
    gl_t C2 = gl_sub(c[P_AL], gl_mul(gl_mul(c[P_PR], c[P_Q]),
                                     gl_sub(1ULL, gl_add(c[P_SG], c[P_SG]))));
    gl_t DOM = gl_mul(c[P_PR], gl_sub(gl_add(c[P_SH], gl_add(c[P_EB], 127ULL)), gmax));
    gl_t SS = gl_mul(c[P_SEL], c[P_SH]);
    gl_t SB = gl_sub(gl_mul(c[P_SEL], c[P_SEL]), c[P_SEL]);
    gl_t SP = gl_mul(c[P_SEL], gl_sub(1ULL, c[P_PR]));
    gl_t s = gl_mul(lam[0], C1);
    s = gl_add(s, gl_mul(lam[1], C2)); s = gl_add(s, gl_mul(lam[2], DOM));
    s = gl_add(s, gl_mul(lam[3], SS)); s = gl_add(s, gl_mul(lam[4], SB));
    s = gl_add(s, gl_mul(lam[5], SP));
    return gl_mul(v[0], s);
}
// Dg zero-check: v = [E, dg cols 0..NDG-1]; lam[18]
static inline __host__ __device__ gl_t F_dg(const gl_t* v, const gl_t* lam) {
    const gl_t* c = v + 1;
    gl_t one = 1ULL;
    gl_t r[18];
    r[0]  = gl_mul(c[G_AZ], c[G_ASIG]);
    r[1]  = gl_sub(gl_mul(c[G_ASIG], c[G_AINV]), gl_sub(one, c[G_AZ]));
    r[2]  = gl_sub(c[G_ASIG], gl_add(gl_mul(c[G_ABASE], 1024ULL), c[G_ALO]));
    r[3]  = gl_add(gl_sub(c[G_ASH], c[G_MAX]), gl_mul(gl_sub(one, c[G_AZ]), c[G_BEXP]));
    r[4]  = gl_sub(gl_add(gl_mul(c[G_AQ], c[G_APW]), c[G_AR]), c[G_ABASE]);
    r[5]  = gl_mul(c[G_ASEL], c[G_ASH]);
    r[6]  = gl_sub(gl_sub(gl_mul(gl_sub(one, gl_add(c[G_TSGN], c[G_TSGN])), c[G_TMAG]), c[G_CSUM]),
                   gl_mul(gl_sub(one, gl_add(c[G_ASGN], c[G_ASGN])), c[G_AQ]));
    r[7]  = gl_mul(c[G_TZ], c[G_TMAG]);
    r[8]  = gl_sub(gl_mul(c[G_TMAG], c[G_TINV]), gl_sub(one, c[G_TZ]));
    r[9]  = gl_mul(c[G_TZ], c[G_TSGN]);
    r[10] = gl_sub(gl_sub(c[G_TMAG], c[G_PLO]), gl_add(gl_mul(c[G_U1H], 1024ULL), c[G_U1L]));
    r[11] = gl_sub(gl_sub(gl_sub(c[G_PHI], one), c[G_TMAG]),
                   gl_add(gl_mul(c[G_U2H], 1024ULL), c[G_U2L]));
    r[12] = gl_sub(gl_add(gl_mul(c[G_S14], c[G_PDN]), c[G_NR]), gl_mul(c[G_TMAG], c[G_PUP]));
    r[13] = gl_sub(gl_mul(c[G_AZ], c[G_AZ]), c[G_AZ]);
    r[14] = gl_sub(gl_mul(c[G_ASEL], c[G_ASEL]), c[G_ASEL]);
    r[15] = gl_sub(gl_mul(c[G_TSGN], c[G_TSGN]), c[G_TSGN]);
    r[16] = gl_sub(gl_mul(c[G_TZ], c[G_TZ]), c[G_TZ]);
    r[17] = gl_sub(gl_mul(c[G_ASGN], c[G_ASGN]), c[G_ASGN]);
    gl_t s = 0;
    for (int j = 0; j < 18; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}
// chain: v = [U, V, U0, asig, bexp, asgn, tz, s14, gmax, wd, tsgn]; lam[6]
static inline __host__ __device__ gl_t F_ch(const gl_t* v, const gl_t* lam) {
    gl_t one = 1ULL, ntz = gl_sub(one, v[6]);
    gl_t inA  = gl_add(gl_add(gl_mul(lam[0], v[3]), gl_mul(lam[1], v[4])), gl_mul(lam[2], v[5]));
    gl_t outS = gl_mul(gl_mul(ntz, v[7]), 1024ULL);
    gl_t outE = gl_mul(ntz, gl_sub(gl_add(v[8], v[9]), 14ULL));
    gl_t outv = gl_add(gl_add(gl_mul(lam[0], outS), gl_mul(lam[1], outE)), gl_mul(lam[2], v[10]));
    gl_t bnd  = gl_add(gl_add(gl_mul(lam[3], v[3]), gl_mul(lam[4], v[4])), gl_mul(lam[5], v[5]));
    return gl_add(gl_sub(gl_mul(v[0], inA), gl_mul(v[1], outv)), gl_mul(v[2], bnd));
}
// group-sum + attainment: v = [EQg, al, sel]; lam[2]
static inline __host__ __device__ gl_t F_gs(const gl_t* v, const gl_t* lam) {
    return gl_mul(v[0], gl_add(gl_mul(lam[0], v[1]), gl_mul(lam[1], v[2])));
}
// Y binding: v = [EQy, R, yb]
static inline __host__ __device__ gl_t F_y(const gl_t* v, const gl_t*) {
    return gl_mul(gl_mul(v[0], v[1]), v[2]);
}
// Db/Dn zero-check: v = [E, sbits, ss, se, sei, zs, smh, sml]; lam[6]
static inline __host__ __device__ gl_t F_ds(const gl_t* v, const gl_t* lam) {
    gl_t one = 1ULL;
    gl_t sm = gl_add(gl_mul(v[6], 4096ULL), v[7]);
    gl_t r0 = gl_sub(v[1], gl_add(gl_add(gl_mul(v[2], 2147483648ULL), gl_mul(v[3], 8388608ULL)), sm));
    gl_t r1 = gl_sub(gl_mul(v[2], v[2]), v[2]);
    gl_t r2 = gl_sub(gl_mul(v[5], v[5]), v[5]);
    gl_t r3 = gl_mul(v[5], v[3]);
    gl_t r4 = gl_sub(gl_mul(v[3], v[4]), gl_sub(one, v[5]));
    gl_t r5 = gl_mul(v[5], sm);
    gl_t s = gl_mul(lam[0], r0);
    s = gl_add(s, gl_mul(lam[1], r1)); s = gl_add(s, gl_mul(lam[2], r2));
    s = gl_add(s, gl_mul(lam[3], r3)); s = gl_add(s, gl_mul(lam[4], r4));
    s = gl_add(s, gl_mul(lam[5], r5));
    return gl_mul(v[0], s);
}
// Do zero-check: v = [E, do cols 0..NDO-1, ssx,sex,zsx,smhx,smlx, ssw,sew,zsw,smhw,smlw]
enum { VX_SS = 0, VX_SE, VX_ZS, VX_SMH, VX_SML, VW_SS, VW_SE, VW_ZS, VW_SMH, VW_SML, NVO };
static const int N_DO_CONSTR = 49;
static inline __host__ __device__ gl_t F_do(const gl_t* v, const gl_t* lam) {
    const gl_t* c = v + 1; const gl_t* vx = v + 1 + NDO;
    gl_t one = 1ULL;
    gl_t nz1 = gl_sub(one, c[O_Z1]), nz2 = gl_sub(one, c[O_Z2]);
    gl_t smx = gl_add(gl_mul(vx[VX_SMH], 4096ULL), vx[VX_SML]);
    gl_t smw = gl_add(gl_mul(vx[VW_SMH], 4096ULL), vx[VW_SML]);
    gl_t m2x = gl_add(8388608ULL, smx), m3w = gl_add(8388608ULL, smw);
    gl_t p1c = gl_add(one, c[O_C1]), p2c = gl_add(one, c[O_C2]);
    gl_t mfl1 = gl_add(gl_mul(c[O_FH1], 4096ULL), c[O_FL1]);
    gl_t mfl2 = gl_add(gl_mul(c[O_FH2], 4096ULL), c[O_FL2]);
    gl_t r[N_DO_CONSTR];
    auto boolc = [&](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    r[0] = gl_mul(c[O_FZ], c[O_S14F]);
    r[1] = gl_sub(gl_mul(c[O_S14F], c[O_FINV]), gl_sub(one, c[O_FZ]));
    r[2] = boolc(c[O_FZ]);
    r[3] = boolc(c[O_SGN]);
    r[4] = gl_sub(c[O_Z1], gl_sub(gl_add(c[O_FZ], vx[VX_ZS]), gl_mul(c[O_FZ], vx[VX_ZS])));
    r[5] = gl_sub(c[O_Z2], gl_sub(gl_add(c[O_Z1], vx[VW_ZS]), gl_mul(c[O_Z1], vx[VW_ZS])));
    r[6] = gl_mul(nz1, gl_sub(gl_mul(c[O_S14F], m2x),
                   gl_add(gl_mul(gl_mul(mfl1, 8192ULL), p1c), c[O_REM1])));
    r[7] = gl_sub(c[O_REM1], gl_add(gl_mul(gl_mul(c[O_RB1], 4096ULL), p1c), c[O_RT1]));
    r[8] = boolc(c[O_C1]); r[9] = boolc(c[O_RB1]); r[10] = boolc(c[O_C21]);
    r[11] = boolc(c[O_L01]); r[12] = boolc(c[O_ZRT1]);
    r[13] = gl_sub(c[O_FL1], gl_add(gl_add(c[O_FLH1], c[O_FLH1]), c[O_L01]));
    r[14] = gl_mul(c[O_ZRT1], c[O_RT1]);
    r[15] = gl_sub(gl_mul(c[O_RT1], c[O_RTI1]), gl_sub(one, c[O_ZRT1]));
    r[16] = gl_sub(c[O_RUP1], gl_mul(c[O_RB1],
                    gl_add(gl_sub(one, c[O_ZRT1]), gl_mul(c[O_ZRT1], c[O_L01]))));
    r[17] = gl_mul(c[O_C21], gl_sub(gl_add(mfl1, c[O_RUP1]), 16777216ULL));
    r[18] = gl_mul(nz1, gl_add(gl_sub(gl_sub(c[O_M1], mfl1), c[O_RUP1]),
                               gl_mul(c[O_C21], 8388608ULL)));
    r[19] = gl_mul(c[O_Z1], gl_sub(c[O_M1], 8388608ULL));
    r[20] = gl_mul(nz1, gl_sub(c[O_E1],
                    gl_add(gl_sub(gl_add(c[O_BEXP], vx[VX_SE]), 139ULL),
                           gl_add(c[O_C1], c[O_C21]))));
    r[21] = gl_mul(c[O_Z1], gl_sub(c[O_E1], one));
    r[22] = gl_sub(c[O_S1], gl_sub(gl_add(c[O_SGN], vx[VX_SS]),
                    gl_mul(gl_add(c[O_SGN], c[O_SGN]), vx[VX_SS])));
    r[23] = gl_mul(nz2, gl_sub(gl_mul(c[O_M1], m3w),
                    gl_add(gl_add(gl_mul(gl_mul(mfl2, 8388608ULL), p2c),
                                  gl_mul(gl_mul(c[O_RB2], 4194304ULL), p2c)),
                           gl_add(gl_mul(c[O_TH2], 4096ULL), c[O_TL2]))));
    r[24] = boolc(c[O_C2]); r[25] = boolc(c[O_RB2]); r[26] = boolc(c[O_C22]);
    r[27] = boolc(c[O_L02]); r[28] = boolc(c[O_ZRT2]);
    r[29] = gl_sub(c[O_FL2], gl_add(gl_add(c[O_FLH2], c[O_FLH2]), c[O_L02]));
    r[30] = gl_mul(c[O_ZRT2], c[O_TH2]);
    r[31] = gl_mul(c[O_ZRT2], c[O_TL2]);
    r[32] = gl_sub(gl_mul(gl_add(gl_mul(c[O_TH2], 4096ULL), c[O_TL2]), c[O_RTI2]),
                   gl_sub(one, c[O_ZRT2]));
    r[33] = gl_sub(c[O_RUP2], gl_mul(c[O_RB2],
                    gl_add(gl_sub(one, c[O_ZRT2]), gl_mul(c[O_ZRT2], c[O_L02]))));
    r[34] = gl_mul(c[O_C22], gl_sub(gl_add(mfl2, c[O_RUP2]), 16777216ULL));
    r[35] = gl_mul(nz2, gl_add(gl_sub(gl_sub(c[O_M2], mfl2), c[O_RUP2]),
                               gl_mul(c[O_C22], 8388608ULL)));
    r[36] = gl_mul(c[O_Z2], gl_sub(c[O_M2], 8388608ULL));
    r[37] = gl_mul(nz2, gl_sub(c[O_E2],
                    gl_add(gl_sub(gl_add(c[O_E1], vx[VW_SE]), 127ULL),
                           gl_add(c[O_C2], c[O_C22]))));
    r[38] = gl_mul(c[O_Z2], gl_sub(c[O_E2], one));
    r[39] = gl_sub(c[O_S2], gl_sub(gl_add(c[O_S1], vx[VW_SS]),
                    gl_mul(gl_add(c[O_S1], c[O_S1]), vx[VW_SS])));
    r[40] = gl_sub(gl_add(gl_add(gl_mul(c[O_Q16], 65536ULL), gl_mul(c[O_RB3], 32768ULL)), c[O_RT3]),
                   gl_add(gl_mul(c[O_S2], 2147483648ULL),
                          gl_mul(nz2, gl_sub(gl_add(gl_mul(c[O_E2], 8388608ULL), c[O_M2]),
                                             8388608ULL))));
    r[41] = gl_sub(c[O_Q16], gl_add(gl_add(c[O_Q16H], c[O_Q16H]), c[O_L03]));
    r[42] = boolc(c[O_RB3]); r[43] = boolc(c[O_L03]); r[44] = boolc(c[O_ZRT3]);
    r[45] = gl_mul(c[O_ZRT3], c[O_RT3]);
    r[46] = gl_sub(gl_mul(c[O_RT3], c[O_RTI3]), gl_sub(one, c[O_ZRT3]));
    r[47] = gl_sub(c[O_RUP3], gl_mul(c[O_RB3],
                    gl_add(gl_sub(one, c[O_ZRT3]), gl_mul(c[O_ZRT3], c[O_L03]))));
    r[48] = gl_sub(c[O_YB], gl_add(c[O_Q16], c[O_RUP3]));
    gl_t s = 0;
    for (int j = 0; j < N_DO_CONSTR; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}

// device functors for the GPU sumcheck port (p3_scgpu)
struct FFdp { static __device__ gl_t eval(const gl_t* v, const gl_t* p) { return F_dp(v, p); } };
struct FFdg { static __device__ gl_t eval(const gl_t* v, const gl_t* p) { return F_dg(v, p); } };
struct FFch { static __device__ gl_t eval(const gl_t* v, const gl_t* p) { return F_ch(v, p); } };
struct FFgs { static __device__ gl_t eval(const gl_t* v, const gl_t* p) { return F_gs(v, p); } };

// dispatch a quartic zero-check sumcheck: device-resident when the domain is
// large (identical messages/transcript to the host loop), host otherwise.
template <typename FF>
static inline std::vector<gl_t> sc5_run(fs::Transcript& tr, const char* tag,
        std::vector<std::vector<gl_t>>&& cols, const gl_t* lam, uint32_t nlam,
        const CFn& Fhost, std::vector<Msg5>& msgs) {
    size_t N = cols[0].size();
    if (p3fri::g_gpu_merkle && N >= (1u << 16)) {
        std::vector<gl_t*> dc(cols.size());
        for (size_t i = 0; i < cols.size(); i++) {
            cudaMallocAsync(&dc[i], N * 8, 0);
            cudaMemcpy(dc[i], cols[i].data(), N * 8, cudaMemcpyHostToDevice);
        }
        std::vector<gl_t> r = p3sg::sc_prove_gpu<FF, Msg5, 5, 32>(
            tr, tag, dc, (uint32_t)N, lam, nlam, msgs);
        for (auto p : dc) cudaFreeAsync(p, 0);
        return r;
    }
    return sc5_prove(tr, tag, std::move(cols), Fhost, msgs);
}

// ================= zk quartic sumcheck (p3_zkc mechanism 2) =================
// Commits 4 blind columns over the sumcheck's (augmented) domain, publishes
// H = sum_b [B1 + E*B2 + E^2*B3 + E^3*B4] (E = cols[0], the public weight
// column) AFTER the weight's point but BEFORE rho, then proves
//     sum_b [ F(cols(b)) + rho*(B1 + E*B2 + E^2*B3 + E^3*B4) ]
//       = base_claim + rho*H .
// The four terms have round degree 1..4, spanning every coefficient of the
// quartic message (a plain multilinear blind leaves the t^2..t^4 finite
// differences as pure witness functionals).  Blind evals at the terminal are
// ordinary ledger claims.  In simulator mode (tape replay), H is shifted by
// (S_actual - base_claim)/rho with the tape-known rho, so the honest chain of
// a FAKE witness verifies -- the HVZK simulator IS this prover on garbage.
static inline std::vector<gl_t> sc5z(fs::Transcript& tr, const char* tag, uint32_t vreal,
        std::vector<std::vector<gl_t>>&& cols, const CFn& F, std::vector<Msg5>& msgs,
        gl_t base_claim, uint32_t R,
        p3bo::PLedger& lg, std::deque<Col>& keep, p3zkc::Blind& bl) {
    if (!p3zkc::G.on) return sc5_prove(tr, tag, std::move(cols), F, msgs);
    size_t N = cols[0].size();
    if (p3bf::memlog() && N >= ((size_t)1 << 24)) {
        char b[96]; snprintf(b, sizeof b, "sc5z %s N=2^%u x %zu cols", tag,
                             ilog2(N), cols.size());
        p3bf::rsslog(b);
    }
    std::vector<Col> B(4);
    uint64_t bseed[4];
    for (int j = 0; j < 4; j++) {
        std::vector<gl_t> m;
        bseed[j] = p3zkc::G.blind_on ? p3zkc::next_seed() : 0;
        B[j] = p3lu::commit_col_nc(p3zkc::blind_col_seeded(vreal, bseed[j], m), R, &m);
        bl.rt[j] = B[j].root; tr.absorb("sc5-bl", B[j].root.data(), 32);
    }
    bl.nb = 4;
    gl_t H = 0;
    {
        const std::vector<gl_t>& E = cols[0];
        const int P = N >= 65536 ? 128 : 1;
        std::vector<gl_t> part(P, 0);
        #pragma omp parallel for schedule(static) if (P > 1)
        for (int p = 0; p < P; p++) {
            size_t lo = N * p / P, hi = N * (p + 1) / P;
            gl_t acc = 0;
            for (size_t b = lo; b < hi; b++) {
                gl_t e = E[b];
                acc = gl_add(acc, gl_add(B[0].v[b], gl_mul(e, gl_add(B[1].v[b],
                        gl_mul(e, gl_add(B[2].v[b], gl_mul(e, B[3].v[b])))))));
            }
            part[p] = acc;
        }
        for (int p = 0; p < P; p++) H = gl_add(H, part[p]);
    }
    if (p3zkc::G.sim && fs::g_tape && !fs::g_tape->record) {
        gl_t Sact = 0;
        { size_t nc = cols.size(); std::vector<gl_t> vv(nc);
          for (size_t b = 0; b < N; b++) {
              for (size_t k2 = 0; k2 < nc; k2++) vv[k2] = cols[k2][b];
              Sact = gl_add(Sact, F(vv.data()));
          } }
        if (Sact != base_claim && fs::g_tape->pos < fs::g_tape->ch.size()) {
            const auto& cb = fs::g_tape->ch[fs::g_tape->pos];      // rho, tape-known
            uint64_t rv = 0; for (int i = 0; i < 8; i++) rv |= (uint64_t)cb[i] << (8 * i);
            gl_t rho_pk = rv % GL_P;
            if (rho_pk != 0)
                H = gl_add(H, gl_mul(gl_sub(Sact, base_claim), gl_inv(rho_pk)));
        }
    }
    bl.H = H;
    tr.absorb("sc5-H", &H, 8);
    gl_t rho = chal(tr);
    size_t i0 = cols.size();
    for (int j = 0; j < 4; j++) cols.push_back(B[j].v);
    CFn F2 = [&F, rho, i0](const gl_t* v) {
        gl_t e = v[0];
        gl_t b_ = gl_add(v[i0], gl_mul(e, gl_add(v[i0 + 1],
                    gl_mul(e, gl_add(v[i0 + 2], gl_mul(e, v[i0 + 3]))))));
        return gl_add(F(v), gl_mul(rho, b_));
    };
    std::vector<gl_t> r = sc5_prove(tr, tag, std::move(cols), F2, msgs);
    std::vector<gl_t> eqr = p3bf::build_eq(r);
    for (int j = 0; j < 4; j++) {
        bl.yB[j] = p3bf::eval_h(B[j].v, eqr);
        tr.absorb("sc5-yB", &bl.yB[j], 8);
        uint64_t ss = B[j].sseed;
        keep.push_back(std::move(B[j]));
        // blinds are pure PRNG streams: register a regenerator and DROP the
        // values -- the batched opening rebuilds them transiently per use
        uint32_t vr = vreal; uint64_t bs = bseed[j];
        lg.add(&keep.back().v, bl.rt[j], r, bl.yB[j], ss,
               [vr, bs] { return p3zkc::blind_col_aug(vr, bs); });
        keep.back().v.clear(); keep.back().v.shrink_to_fit();
    }
    return r;
}
// verifier mirror, phase 1: absorb blind roots + H, draw rho (0 when zk off)
static inline gl_t sc5vz_pre(fs::Transcript& tr, const p3zkc::Blind& bl) {
    if (!p3zkc::G.on) return 0;
    for (int j = 0; j < 4; j++) tr.absorb("sc5-bl", bl.rt[j].data(), 32);
    gl_t H = bl.H; tr.absorb("sc5-H", &H, 8);
    return chal(tr);
}
// verifier mirror, phase 2 (call IMMEDIATELY after sc5_verify, BEFORE the
// gadget's terminal claimv calls -- the prover absorbs these blind evals and
// registers their ledger claims inside sc5z right after the sumcheck, so the
// transcript+ledger order must match here).
static inline void sc5vz_claims(fs::Transcript& tr, p3bo::VLedger& vlg,
                                const p3zkc::Blind& bl, const std::vector<gl_t>& r) {
    if (!p3zkc::G.on) return;
    for (int j = 0; j < 4; j++) {
        gl_t yb = bl.yB[j];
        tr.absorb("sc5-yB", &yb, 8);
        vlg.add(bl.rt[j], r, yb);
    }
}
// terminal add-on rho*(yB0 + w*yB1 + w^2*yB2 + w^3*yB3), w = the terminal weight
// value (verifier-computed value of cols[0] at r) -- pure arithmetic, no absorb
static inline gl_t sc5_blindterm(const p3zkc::Blind& bl, gl_t rho, gl_t w) {
    if (!p3zkc::G.on) return 0;
    return gl_mul(rho, gl_add(bl.yB[0], gl_mul(w, gl_add(bl.yB[1],
                   gl_mul(w, gl_add(bl.yB[2], gl_mul(w, bl.yB[3])))))));
}
// zk-aware dispatch: legacy path (GPU when large) with zk off, blinded host
// chain with zk on
template <typename FF>
static inline std::vector<gl_t> sc5rz(fs::Transcript& tr, const char* tag, uint32_t vreal,
        std::vector<std::vector<gl_t>>&& cols, const gl_t* lam, uint32_t nlam,
        const CFn& F, std::vector<Msg5>& msgs, gl_t base_claim, uint32_t R,
        p3bo::PLedger& lg, std::deque<Col>& keep, p3zkc::Blind& bl) {
    if (!p3zkc::G.on) return sc5_run<FF>(tr, tag, std::move(cols), lam, nlam, F, msgs);
    return sc5z(tr, tag, vreal, std::move(cols), F, msgs, base_claim, R, lg, keep, bl);
}

// zk variant over ColSrc columns (borrowed committed columns are not copied;
// bytes on the transcript identical to sc5z)
static inline std::vector<gl_t> sc5z_srcs(fs::Transcript& tr, const char* tag, uint32_t vreal,
        std::vector<ColSrc>&& cols, const CFn& F, std::vector<Msg5>& msgs,
        gl_t base_claim, uint32_t R,
        p3bo::PLedger& lg, std::deque<Col>& keep, p3zkc::Blind& bl) {
    if (!p3zkc::G.on) return sc5_prove_srcs(tr, tag, std::move(cols), F, msgs);
    size_t N = cols[0].get().size();
    if (p3bf::memlog() && N >= ((size_t)1 << 24)) {
        char b[96]; snprintf(b, sizeof b, "sc5z_srcs %s N=2^%u x %zu cols", tag,
                             ilog2(N), cols.size());
        p3bf::rsslog(b);
    }
    std::vector<Col> B(4);
    uint64_t bseed[4];
    for (int j = 0; j < 4; j++) {
        std::vector<gl_t> m;
        bseed[j] = p3zkc::G.blind_on ? p3zkc::next_seed() : 0;
        B[j] = p3lu::commit_col_nc(p3zkc::blind_col_seeded(vreal, bseed[j], m), R, &m);
        bl.rt[j] = B[j].root; tr.absorb("sc5-bl", B[j].root.data(), 32);
    }
    bl.nb = 4;
    gl_t H = 0;
    {
        const std::vector<gl_t>& E = cols[0].get();
        const int P = N >= 65536 ? 128 : 1;
        std::vector<gl_t> part(P, 0);
        #pragma omp parallel for schedule(static) if (P > 1)
        for (int p = 0; p < P; p++) {
            size_t lo = N * p / P, hi = N * (p + 1) / P;
            gl_t acc = 0;
            for (size_t b = lo; b < hi; b++) {
                gl_t e = E[b];
                acc = gl_add(acc, gl_add(B[0].v[b], gl_mul(e, gl_add(B[1].v[b],
                        gl_mul(e, gl_add(B[2].v[b], gl_mul(e, B[3].v[b])))))));
            }
            part[p] = acc;
        }
        for (int p = 0; p < P; p++) H = gl_add(H, part[p]);
    }
    if (p3zkc::G.sim && fs::g_tape && !fs::g_tape->record) {
        gl_t Sact = 0;
        { size_t nc = cols.size(); std::vector<gl_t> vv(nc);
          for (size_t b = 0; b < N; b++) {
              for (size_t k2 = 0; k2 < nc; k2++) vv[k2] = cols[k2].get()[b];
              Sact = gl_add(Sact, F(vv.data()));
          } }
        if (Sact != base_claim && fs::g_tape->pos < fs::g_tape->ch.size()) {
            const auto& cb = fs::g_tape->ch[fs::g_tape->pos];
            uint64_t rv = 0; for (int i = 0; i < 8; i++) rv |= (uint64_t)cb[i] << (8 * i);
            gl_t rho_pk = rv % GL_P;
            if (rho_pk != 0)
                H = gl_add(H, gl_mul(gl_sub(Sact, base_claim), gl_inv(rho_pk)));
        }
    }
    bl.H = H;
    tr.absorb("sc5-H", &H, 8);
    gl_t rho = chal(tr);
    size_t i0 = cols.size();
    for (int j = 0; j < 4; j++) cols.push_back(ColSrc(&B[j].v));   // borrowed blinds
    CFn F2 = [&F, rho, i0](const gl_t* v) {
        gl_t e = v[0];
        gl_t b_ = gl_add(v[i0], gl_mul(e, gl_add(v[i0 + 1],
                    gl_mul(e, gl_add(v[i0 + 2], gl_mul(e, v[i0 + 3]))))));
        return gl_add(F(v), gl_mul(rho, b_));
    };
    std::vector<gl_t> r = sc5_prove_srcs(tr, tag, std::move(cols), F2, msgs);
    std::vector<gl_t> eqr = p3bf::build_eq(r);
    for (int j = 0; j < 4; j++) {
        bl.yB[j] = p3bf::eval_h(B[j].v, eqr);
        tr.absorb("sc5-yB", &bl.yB[j], 8);
        uint64_t ss = B[j].sseed;
        keep.push_back(std::move(B[j]));
        uint32_t vr = vreal; uint64_t bs = bseed[j];
        lg.add(&keep.back().v, bl.rt[j], r, bl.yB[j], ss,
               [vr, bs] { return p3zkc::blind_col_aug(vr, bs); });
        keep.back().v.clear(); keep.back().v.shrink_to_fit();
    }
    if (N >= ((size_t)1 << 24)) p3bf::trim_heap();
    return r;
}
// GPU wrapper functor: quartic base F plus the Libra blind term
// rho*(B1 + E*B2 + E^2*B3 + E^3*B4); par = [rho, ncb, base pars...], blinds
// are the 4 columns starting at index ncb, E = cols[0].
template <typename FF>
struct FF5Zk {
    static __device__ gl_t eval(const gl_t* c, const gl_t* p) {
        int ncb = (int)p[1];
        gl_t e = c[0];
        gl_t b = gl_add(c[ncb], gl_mul(e, gl_add(c[ncb + 1],
                  gl_mul(e, gl_add(c[ncb + 2], gl_mul(e, c[ncb + 3]))))));
        return gl_add(FF::eval(c, p + 2), gl_mul(p[0], b));
    }
};
// zk quartic zero-check with the SUMCHECK ON THE DEVICE: the host never holds
// the bound working set (at llama-68m dims the host copy of one P-domain
// zero-check plus its first-bind halves alone breaches the 41 GB container
// cap, while the card has >10 GB free at that point).  Blind terminal values
// are read from the fully-bound device columns (the v-round fold IS the
// multilinear restriction, exact).  Transcript bytes identical to sc5z.
template <typename FF>
static inline std::vector<gl_t> sc5z_gpu(fs::Transcript& tr, const char* tag, uint32_t vreal,
        std::vector<ColSrc>&& cols, const gl_t* lam, uint32_t nlam,
        std::vector<Msg5>& msgs, gl_t base_claim, uint32_t R,
        p3bo::PLedger& lg, std::deque<Col>& keep, p3zkc::Blind& bl) {
    size_t N = cols[0].get().size();
    if (p3bf::memlog() && N >= ((size_t)1 << 24)) {
        char b[96]; snprintf(b, sizeof b, "sc5z_gpu %s N=2^%u x %zu cols", tag,
                             ilog2(N), cols.size());
        p3bf::rsslog(b);
    }
    std::vector<Col> B(4);
    uint64_t bseed[4];
    for (int j = 0; j < 4; j++) {
        std::vector<gl_t> m;
        bseed[j] = p3zkc::G.blind_on ? p3zkc::next_seed() : 0;
        B[j] = p3lu::commit_col_nc(p3zkc::blind_col_seeded(vreal, bseed[j], m), R, &m);
        bl.rt[j] = B[j].root; tr.absorb("sc5-bl", B[j].root.data(), 32);
    }
    bl.nb = 4;
    gl_t H = 0;
    {
        const std::vector<gl_t>& E = cols[0].get();
        const int P = N >= 65536 ? 128 : 1;
        std::vector<gl_t> part(P, 0);
        #pragma omp parallel for schedule(static) if (P > 1)
        for (int p = 0; p < P; p++) {
            size_t lo = N * p / P, hi = N * (p + 1) / P;
            gl_t acc = 0;
            for (size_t b = lo; b < hi; b++) {
                gl_t e = E[b];
                acc = gl_add(acc, gl_add(B[0].v[b], gl_mul(e, gl_add(B[1].v[b],
                        gl_mul(e, gl_add(B[2].v[b], gl_mul(e, B[3].v[b])))))));
            }
            part[p] = acc;
        }
        for (int p = 0; p < P; p++) H = gl_add(H, part[p]);
    }
    bl.H = H;
    tr.absorb("sc5-H", &H, 8);
    gl_t rho = chal(tr);
    const uint32_t ncb = (uint32_t)cols.size();
    std::vector<gl_t*> dc(ncb + 4);
    for (uint32_t k = 0; k < ncb; k++) {
        dc[k] = p3bf::dmalloc(N, "sc5zg:col");
        cudaMemcpy(dc[k], cols[k].get().data(), N * 8, cudaMemcpyHostToDevice);
        std::vector<gl_t>().swap(cols[k].own); cols[k].bor = nullptr;
    }
    for (int j = 0; j < 4; j++) {
        dc[ncb + j] = p3bf::dmalloc(N, "sc5zg:blind");
        cudaMemcpy(dc[ncb + j], B[j].v.data(), N * 8, cudaMemcpyHostToDevice);
        std::vector<gl_t>().swap(B[j].v);        // regenerable from bseed
    }
    std::vector<gl_t> par(2 + nlam);
    par[0] = rho; par[1] = (gl_t)ncb;
    for (uint32_t i = 0; i < nlam; i++) par[2 + i] = lam[i];
    std::vector<gl_t> r = p3sg::sc_prove_gpu<FF5Zk<FF>, Msg5, 5, 32>(
        tr, tag, dc, (uint32_t)N, par.data(), 2 + nlam, msgs);
    for (int j = 0; j < 4; j++) {
        cudaMemcpy(&bl.yB[j], dc[ncb + j], 8, cudaMemcpyDeviceToHost);
        tr.absorb("sc5-yB", &bl.yB[j], 8);
        uint64_t ss = B[j].sseed;
        keep.push_back(std::move(B[j]));
        uint32_t vr = vreal; uint64_t bs = bseed[j];
        lg.add(&keep.back().v, bl.rt[j], r, bl.yB[j], ss,
               [vr, bs] { return p3zkc::blind_col_aug(vr, bs); });
    }
    for (auto p : dc) cudaFreeAsync(p, 0);
    p3bf::ckcuda("sc5z_gpu");
    if (N >= ((size_t)1 << 24)) p3bf::trim_heap();
    return r;
}
template <typename FF>
static inline std::vector<gl_t> sc5rz_srcs(fs::Transcript& tr, const char* tag, uint32_t vreal,
        std::vector<ColSrc>&& cols, const gl_t* lam, uint32_t nlam,
        const CFn& F, std::vector<Msg5>& msgs, gl_t base_claim, uint32_t R,
        p3bo::PLedger& lg, std::deque<Col>& keep, p3zkc::Blind& bl) {
    if (!p3zkc::G.on) return sc5_run_srcs<FF>(tr, tag, std::move(cols), lam, nlam, F, msgs);
    if (p3fri::g_gpu_merkle && cols[0].get().size() >= (1u << 20) && !p3zkc::G.sim)
        return sc5z_gpu<FF>(tr, tag, vreal, std::move(cols), lam, nlam, msgs,
                            base_claim, R, lg, keep, bl);
    return sc5z_srcs(tr, tag, vreal, std::move(cols), F, msgs, base_claim, R, lg, keep, bl);
}

// ---------------- lookup instance descriptors ----------------
struct LuDef { const Table* tab; int dom; std::vector<int> cols; const char* label; };
// dom: 0=Dp 1=Dg 2=Do 3=Db 4=Dn; col -1 = virtual a, -2 = virtual b
static inline std::vector<LuDef> lu_defs(const Tables& T) {
    std::vector<LuDef> L(NLU);
    L[LU_DM]   = {&T.DM, 0, {-1, -2, P_EB, P_MAG, P_SG, P_PR}, "hwlDM"};
    L[LU_SH]   = {&T.SH, 0, {P_SH, P_PW}, "hwlSH"};
    L[LU_RM]   = {&T.RM, 0, {P_PW, P_R}, "hwlRM"};
    L[LU_Q15]  = {&T.R15, 0, {P_Q}, "hwlQ15"};
    L[LU_ASH]  = {&T.ASH, 1, {G_ASH, G_APW}, "hwlASH"};
    L[LU_ARM]  = {&T.RM, 1, {G_APW, G_AR}, "hwlARM"};
    L[LU_AQ]   = {&T.R15, 1, {G_AQ}, "hwlAQ"};
    L[LU_ABASE]= {&T.R15, 1, {G_ABASE}, "hwlABASE"};
    L[LU_ALO]  = {&T.R10, 1, {G_ALO}, "hwlALO"};
    L[LU_WD]   = {&T.WIDTH, 1, {G_WD, G_PLO, G_PHI, G_PDN, G_PUP}, "hwlWD"};
    L[LU_U1H]  = {&T.R10, 1, {G_U1H}, "hwlU1H"};
    L[LU_U1L]  = {&T.R10, 1, {G_U1L}, "hwlU1L"};
    L[LU_U2H]  = {&T.R10, 1, {G_U2H}, "hwlU2H"};
    L[LU_U2L]  = {&T.R10, 1, {G_U2L}, "hwlU2L"};
    L[LU_NR]   = {&T.RM, 1, {G_PDN, G_NR}, "hwlNR"};
    L[LU_S14]  = {&T.R15, 1, {G_S14}, "hwlS14"};
    L[LU_M1REM]= {&T.CREM13, 2, {O_C1, O_REM1}, "hwlM1REM"};
    L[LU_M1RT] = {&T.CREM12, 2, {O_C1, O_RT1}, "hwlM1RT"};
    L[LU_M1FH] = {&T.FH, 2, {O_FH1}, "hwlM1FH"};
    L[LU_M1FL] = {&T.R12, 2, {O_FL1}, "hwlM1FL"};
    L[LU_M1FLH]= {&T.R11, 2, {O_FLH1}, "hwlM1FLH"};
    L[LU_M1E]  = {&T.REXP, 2, {O_E1}, "hwlM1E"};
    L[LU_M2TH] = {&T.CTH, 2, {O_C2, O_TH2}, "hwlM2TH"};
    L[LU_M2TL] = {&T.R12, 2, {O_TL2}, "hwlM2TL"};
    L[LU_M2FH] = {&T.FH, 2, {O_FH2}, "hwlM2FH"};
    L[LU_M2FL] = {&T.R12, 2, {O_FL2}, "hwlM2FL"};
    L[LU_M2FLH]= {&T.R11, 2, {O_FLH2}, "hwlM2FLH"};
    L[LU_M2E]  = {&T.REXP, 2, {O_E2}, "hwlM2E"};
    L[LU_Q16]  = {&T.R16, 2, {O_Q16}, "hwlQ16"};
    L[LU_Q16H] = {&T.R15, 2, {O_Q16H}, "hwlQ16H"};
    L[LU_RT3]  = {&T.R15, 2, {O_RT3}, "hwlRT3"};
    L[LU_S14F] = {&T.R15, 2, {O_S14F}, "hwlS14F"};
    L[LU_XSE]  = {&T.SE, 3, {S_SE}, "hwlXSE"};
    L[LU_XSMH] = {&T.R11, 3, {S_SMH}, "hwlXSMH"};
    L[LU_XSML] = {&T.R12, 3, {S_SML}, "hwlXSML"};
    L[LU_WSE]  = {&T.SE, 4, {S_SE}, "hwlWSE"};
    L[LU_WSMH] = {&T.R11, 4, {S_SMH}, "hwlWSMH"};
    L[LU_WSML] = {&T.R12, 4, {S_SML}, "hwlWSML"};
    return L;
}

// point rearrangements for the virtual (a,b) -> (X,W) bindings
static inline std::vector<gl_t> zx_point(const Dims& d, const std::vector<gl_t>& rA) {
    std::vector<gl_t> z;
    for (int i = 0; i < 5; i++) z.push_back(rA[i]);
    for (uint32_t i = 0; i < d.lg; i++) z.push_back(rA[5 + d.ln + d.lb + i]);
    for (uint32_t i = 0; i < d.lb; i++) z.push_back(rA[5 + d.ln + i]);
    return z;
}
static inline std::vector<gl_t> zw_point(const Dims& d, const std::vector<gl_t>& rA) {
    std::vector<gl_t> z;
    for (int i = 0; i < 5; i++) z.push_back(rA[i]);
    for (uint32_t i = 0; i < d.lg; i++) z.push_back(rA[5 + d.ln + d.lb + i]);
    for (uint32_t i = 0; i < d.ln; i++) z.push_back(rA[5 + i]);
    return z;
}

// ---------------- operand commitments ----------------
struct Operands { Col X, W, XS, WS; };
static inline Operands commit_operands(const LayerWit& wt, uint32_t R, bool gpu = true) {
    (void)gpu;                      // openings are batched: root-only commits
    Operands ops;
    std::vector<gl_t> xv(wt.xcodes.size()), wv(wt.wcodes.size());
    for (size_t i = 0; i < xv.size(); i++) xv[i] = wt.xcodes[i];
    for (size_t i = 0; i < wv.size(); i++) wv[i] = wt.wcodes[i];
    ops.X = p3lu::commit_col_nc(xv, R); ops.W = p3lu::commit_col_nc(wv, R);
    ops.XS = p3lu::commit_col_nc(wt.xsb, R); ops.WS = p3lu::commit_col_nc(wt.wsb, R);
    return ops;
}

// ---------------- proof object ----------------
struct LayerProof {
    uint32_t B = 0, K = 0, N = 0;
    p3fri::Hash rdp[NDP], rdg[NDG], rdo[NDO], rdb[NDS], rdn[NDS];
    std::vector<p3lu::GroupProof> lug;     // standalone-mode merged lookup groups
    // all column openings are CLAIMED evaluations backed by the per-size-class
    // batch proofs at the end (DM virtual a,b bindings ride on the DM group's y_virt)
    std::vector<Msg5> mDp; gl_t yDp[NDP] = {}; gl_t yDpG = 0;
    std::vector<Msg5> mDg; gl_t yDg[NDG] = {};
    std::vector<Msg5> mCh; gl_t yCh[8] = {};
    gl_t yGSc = 0, yGSa = 0;
    std::vector<Msg5> mGS; gl_t yGSal = 0, yGSsel = 0;
    std::vector<Msg5> mDo; gl_t yDo[NDO] = {};
    gl_t yDbBC[5] = {}, yDnBC[5] = {};                    // ss,se,zs,smh,sml broadcasts
    std::vector<Msg5> mDb; gl_t yDb[6] = {}; gl_t yXsb = 0;
    std::vector<Msg5> mDn; gl_t yDn[6] = {}; gl_t yWsb = 0;
    gl_t ySlA = 0, ySlE = 0, ySlS = 0, ySlF = 0, ySlBE = 0, ySlSG = 0;
    std::vector<Msg5> mY; gl_t yY = 0;
    // zk: Libra blinds, fixed order Dp, Dg, chain, gsum, Do, Db, Dn
    p3zkc::Blind zbl[7];
    std::vector<p3bo::BatchProof> batches;                // one per size class
};

// per-phase prover timing (ms)
struct Prof {
    double commit_ops = 0, commit_wit = 0;
    double lu_dp = 0, lu_dg = 0, lu_do = 0, lu_sc = 0;
    double zc_dp = 0, zc_dg = 0, chain = 0, gsum = 0, zc_do = 0, zc_ds = 0, slice = 0, ybind = 0;
    double open_dp = 0, open_dg = 0, open_do = 0;
    double batch = 0;
    double total = 0;
};
static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ==================== prover ====================
// Ypub lets the selftest force a WRONG public output claim with an otherwise
// honest transcript (default: the witness's own bf16 output).
static inline LayerProof prove(fs::Transcript& tr, const LayerWit& wt, const Tables& T,
                               const Operands& ops, uint32_t R, uint32_t Q,
                               bool gpu = true, bool strict = true, Prof* prof = nullptr,
                               const std::vector<uint16_t>* Ypub = nullptr,
                               p3lu::XCtx* xc = nullptr,
                               const Col* yb_shared = nullptr) {
    const Dims& d = wt.d;
    if (p3bf::memlog()) {
        char b[96]; snprintf(b, sizeof b, "hwl prove enter B=%u K=%u N=%u (P=2^%u)",
                             d.B, d.K, d.N, d.lp);
        p3bf::rsslog(b);
    }
    const std::vector<uint16_t>& Y = Ypub ? *Ypub : wt.Y;
    LayerProof pf; pf.B = d.B; pf.K = d.K; pf.N = d.N;
    Prof pr_loc; Prof& P = prof ? *prof : pr_loc;
    double tall = now_ms(), tp;

    const bool zk = p3zkc::G.on;

    // -- public preamble --
    uint32_t dims3[3] = {d.B, d.K, d.N};
    tr.absorb("hwl-dims", dims3, sizeof dims3);
    const Table* tabs[16] = {&T.DM, &T.SH, &T.RM, &T.R15, &T.ASH, &T.WIDTH, &T.R10, &T.R11,
                             &T.R12, &T.R16, &T.FH, &T.REXP, &T.SE, &T.CREM13, &T.CREM12, &T.CTH};
    for (auto* t : tabs) tr.absorb("hwl-tab", t->id.data(), 32);
    tr.absorb("hwl-X", ops.X.root.data(), 32);
    tr.absorb("hwl-W", ops.W.root.data(), 32);
    tr.absorb("hwl-xs", ops.XS.root.data(), 32);
    tr.absorb("hwl-ws", ops.WS.root.data(), 32);
    if (!zk) tr.absorb("hwl-Y", Y.data(), Y.size() * 2);   // zk: no public output

    // opening obligations: every column evaluation is claimed inline and proven
    // once per size class by the batched-opening phase at the end; lookups are
    // DEFERRED into the context queue and flushed as merged groups by the
    // ledger owner (here when standalone, the composed layer otherwise)
    p3lu::XCtx xc_loc;
    p3lu::XCtx& XC = xc ? *xc : xc_loc;
    p3bo::PLedger& lg = XC.lg;
    std::deque<Col>& lucols = XC.keep;

    // -- witness commitments (root-only: openings are batched, no host codeword) --
    tp = now_ms();
    std::vector<Col>& CDp = XC.vec(NDP);
    std::vector<Col>& CDg = XC.vec(NDG);
    std::vector<Col>& CDo = XC.vec(NDO);
    std::vector<Col>& CDb = XC.vec(NDS);
    std::vector<Col>& CDn = XC.vec(NDS);
    // zk LINKED slice-1 masks: the gsum row-sum binding and the final-state
    // slice binding are CLAIM algebra, so their identities must hold on the
    // mask slice the (z||zex||0..) claims touch -- the dependent columns'
    // slice-1 masks are DERIVED by the same row formulas.
    std::vector<gl_t> mAL, mSEL, mCSUM, mASEL, mASIG, mBEXP, mASGN, mS14F, mOBEX, mOSGN;
    if (zk) {
        uint32_t lgP = ilog2(d.P), lgG = ilog2(d.G), lgO = ilog2(d.Opad);
        mAL = p3zkc::fresh_mask(lgP);  mSEL = p3zkc::fresh_mask(lgP);
        mCSUM = p3zkc::fresh_mask(lgG); mASEL = p3zkc::fresh_mask(lgG);
        mASIG = p3zkc::fresh_mask(lgG); mBEXP = p3zkc::fresh_mask(lgG);
        mASGN = p3zkc::fresh_mask(lgG);
        mS14F = p3zkc::fresh_mask(lgO); mOBEX = p3zkc::fresh_mask(lgO);
        mOSGN = p3zkc::fresh_mask(lgO);
        for (size_t gi = 0; gi < d.G; gi++) {          // slice 1: group row sums
            gl_t s = 0, sl = 0;
            for (uint32_t kk = 0; kk < 32; kk++) {
                s = gl_add(s, mAL[gi * 32 + kk]);
                sl = gl_add(sl, mSEL[gi * 32 + kk]);
            }
            mCSUM[gi] = s;
            mASEL[gi] = gl_sub(1ULL, sl);
        }
        gl_t i1024 = gl_inv(1024ULL);
        for (uint32_t o = 0; o < d.Opad; o++) {        // slice 1: g=NG slice
            size_t gi = (size_t)d.NG * d.Opad + o;
            mS14F[o] = gl_mul(mASIG[gi], i1024);
            mOBEX[o] = mBEXP[gi];
            mOSGN[o] = mASGN[gi];
        }
    }
    auto mof = [&](int c, int tgt1, std::vector<gl_t>* m1, int tgt2 = -1,
                   std::vector<gl_t>* m2 = nullptr, int tgt3 = -1,
                   std::vector<gl_t>* m3 = nullptr) -> const std::vector<gl_t>* {
        if (!zk) return nullptr;
        if (c == tgt1) return m1;
        if (c == tgt2) return m2;
        if (c == tgt3) return m3;
        return nullptr;
    };
    for (int c = 0; c < NDP; c++) {
        CDp[c] = p3lu::commit_col_nc(wt.dp[c], R, mof(c, P_AL, &mAL, P_SEL, &mSEL));
        pf.rdp[c] = CDp[c].root;
        if (g_free_dp) std::vector<gl_t>().swap(const_cast<LayerWit&>(wt).dp[c]);
    }
    for (int c = 0; c < NDG; c++) {
        const std::vector<gl_t>* m = nullptr;
        if (zk) {
            if (c == G_CSUM) m = &mCSUM; else if (c == G_ASEL) m = &mASEL;
            else if (c == G_ASIG) m = &mASIG; else if (c == G_BEXP) m = &mBEXP;
            else if (c == G_ASGN) m = &mASGN;
        }
        CDg[c] = p3lu::commit_col_nc(wt.dg[c], R, m);
        pf.rdg[c] = CDg[c].root;
    }
    for (int c = 0; c < NDO; c++) {
        // zk composed: the O_YB output column is the SHARED chained operand --
        // reuse the caller's single commitment (same mask+salt) so producer and
        // consumer bind to ONE root (double-committing would give divergent
        // random masks and break the shared-root chain)
        if (c == O_YB && yb_shared) { CDo[c] = *yb_shared; pf.rdo[c] = yb_shared->root; continue; }
        CDo[c] = p3lu::commit_col_nc(wt.dob[c], R,
                     mof(c, O_S14F, &mS14F, O_BEXP, &mOBEX, O_SGN, &mOSGN));
        pf.rdo[c] = CDo[c].root;
    }
    for (int c = 0; c < NDS; c++) { CDb[c] = p3lu::commit_col_nc(wt.db[c], R); pf.rdb[c] = CDb[c].root; }
    for (int c = 0; c < NDS; c++) { CDn[c] = p3lu::commit_col_nc(wt.dn[c], R); pf.rdn[c] = CDn[c].root; }
    for (int c = 0; c < NDP; c++) tr.absorb("hwl-cp", pf.rdp[c].data(), 32);
    for (int c = 0; c < NDG; c++) tr.absorb("hwl-cg", pf.rdg[c].data(), 32);
    for (int c = 0; c < NDO; c++) tr.absorb("hwl-co", pf.rdo[c].data(), 32);
    for (int c = 0; c < NDS; c++) tr.absorb("hwl-cb", pf.rdb[c].data(), 32);
    for (int c = 0; c < NDS; c++) tr.absorb("hwl-cn", pf.rdn[c].data(), 32);
    P.commit_wit += now_ms() - tp;

    // -- lookups: DEFERRED to the ledger owner's merged-group flush --
    auto LD = lu_defs(T);
    // zk: the virtual a,b broadcasts are built over the AUGMENTED product
    // domain from the augmented X/W commitments (exact by construction, so the
    // rearranged-point binding claim needs no linkage)
    const std::vector<gl_t> *pva = &wt.va, *pvb = &wt.vb;
    uint32_t lgP2 = ilog2(d.P);
    if (zk) {
        uint32_t lx = ilog2((size_t)d.Bpad * d.Kpad), lw = ilog2((size_t)d.Npad * d.Kpad);
        auto xmap = [&](size_t p) {
            uint32_t kk = (uint32_t)(p & 31); size_t rest = p >> 5;
            uint32_t b = (uint32_t)((rest >> d.ln) & (d.Bpad - 1));
            uint32_t g = (uint32_t)(rest >> d.lo);
            return (size_t)b * d.Kpad + (size_t)g * 32 + kk;
        };
        auto wmap = [&](size_t p) {
            uint32_t kk = (uint32_t)(p & 31); size_t rest = p >> 5;
            uint32_t n = (uint32_t)(rest & (d.Npad - 1));
            uint32_t g = (uint32_t)(rest >> d.lo);
            return (size_t)n * d.Kpad + (size_t)g * 32 + kk;
        };
        pva = &XC.varr(p3zkc::bc_aug(ops.X.v, lx, lgP2, d.P, xmap));
        pvb = &XC.varr(p3zkc::bc_aug(ops.W.v, lw, lgP2, d.P, wmap));
    }
    for (int i = 0; i < NLU; i++) {
        std::vector<LuColSpec> spec;
        for (int cid : LD[i].cols) {
            if (cid == -1) spec.push_back(LV(pva));
            else if (cid == -2) spec.push_back(LV(pvb));
            else spec.push_back(LC(LD[i].dom == 0 ? &CDp[cid] : LD[i].dom == 1 ? &CDg[cid] :
                                   LD[i].dom == 2 ? &CDo[cid] : LD[i].dom == 3 ? &CDb[cid] : &CDn[cid]));
        }
        p3lu::PBind bind;
        if (i == LU_DM) {
            // bind the virtual (a,b) claims to the X/W commitments in the batch
            const Col *pX = &ops.X, *pW = &ops.W;
            Dims dd = d;
            bind = [pX, pW, dd, lgP2](fs::Transcript&, p3lu::XCtx& xcb,
                                      const std::vector<gl_t>& pm,
                                      const std::vector<gl_t>& yv) {
                xcb.lg.add(&pX->v, pX->root, p3zkc::expt(zx_point(dd, pm), pm, lgP2),
                           yv[0], pX->sseed);
                xcb.lg.add(&pW->v, pW->root, p3zkc::expt(zw_point(dd, pm), pm, lgP2),
                           yv[1], pW->sseed);
                return std::vector<gl_t>{};
            };
        }
        p3lu::defer_v(XC, std::move(spec), wt.lidx[i], *LD[i].tab, LD[i].label, std::move(bind));
    }

    // -- Dp zero-check (C1, C2, dominance, attainment-local) --
    tp = now_ms();
    std::vector<gl_t> zC = chal_vec(tr, d.lp);
    gl_t lamP = chal(tr), lamPv[6]; lamPv[0] = 1;
    for (int j = 1; j < 6; j++) lamPv[j] = gl_mul(lamPv[j-1], lamP);
    {
        std::vector<ColSrc> cols; cols.reserve(NDP + 2);
        cols.push_back(ColSrc(beq(p3zkc::zpt(zC))));
        for (int c = 0; c < NDP; c++) cols.push_back(ColSrc(&CDp[c].v));  // borrowed
        cols.push_back(ColSrc(p3zkc::bc_aug(CDg[G_MAX].v, ilog2(d.G), lgP2, d.P,
                                            [](size_t p) { return p >> 5; })));
        CFn F = [&](const gl_t* v) { return F_dp(v, lamPv); };
        std::vector<gl_t> rC = sc5rz_srcs<FFdp>(tr, "hwl-scP", d.lp, std::move(cols), lamPv, 6,
                                                F, pf.mDp, 0, R, lg, lucols, pf.zbl[0]);
        P.zc_dp += now_ms() - tp; tp = now_ms();
        for (int c = 0; c < NDP; c++)
            pf.yDp[c] = claimc(tr, lg, CDp[c], rC);
        std::vector<gl_t> rCg(rC.begin() + 5, rC.begin() + d.lp);
        pf.yDpG = claimc(tr, lg, CDg[G_MAX], p3zkc::expt(rCg, rC, d.lp));
        P.open_dp += now_ms() - tp;
    }

    // -- Dg zero-check (realign, normalize, flags) --
    tp = now_ms();
    std::vector<gl_t> zG = chal_vec(tr, d.lgi);
    gl_t lamG = chal(tr), lamGv[18]; lamGv[0] = 1;
    for (int j = 1; j < 18; j++) lamGv[j] = gl_mul(lamGv[j-1], lamG);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(NDG + 1);
        cols.push_back(beq(p3zkc::zpt(zG)));
        for (int c = 0; c < NDG; c++) cols.push_back(CDg[c].v);
        CFn F = [&](const gl_t* v) { return F_dg(v, lamGv); };
        std::vector<gl_t> rG = sc5rz<FFdg>(tr, "hwl-scG", d.lgi, std::move(cols), lamGv, 18,
                                           F, pf.mDg, 0, R, lg, lucols, pf.zbl[1]);
        P.zc_dg += now_ms() - tp; tp = now_ms();
        for (int c = 0; c < NDG; c++)
            pf.yDg[c] = claimc(tr, lg, CDg[c], rG);
        P.open_dg += now_ms() - tp;
    }

    // -- accumulator chain: A(g+1) = out(g), A(0) = 0 --
    tp = now_ms();
    std::vector<gl_t> zc = chal_vec(tr, d.lo);
    gl_t lch = chal(tr), lam6[6];
    for (int j = 0; j < 6; j++) lam6[j] = chal(tr);
    {
        std::vector<gl_t> lpow(d.Gpad, 1);
        for (uint32_t g = 1; g < d.Gpad; g++) lpow[g] = gl_mul(lpow[g-1], lch);
        std::vector<gl_t> ug(d.Gpad, 0), vg(d.Gpad, 0);
        for (uint32_t g = 1; g <= d.Gpad - 1; g++) ug[g] = lpow[g-1];
        for (uint32_t g = 0; g + 1 <= d.Gpad - 1; g++) vg[g] = lpow[g];
        std::vector<gl_t> eqO = p3bf::build_eq(zc);
        size_t Gaug = CDg[G_ASIG].v.size();       // zk: zero-extended weights (masks free)
        std::vector<gl_t> U(Gaug, 0), V(Gaug, 0), U0(Gaug, 0);
        for (size_t gi = 0; gi < d.G; gi++) {
            uint32_t g = (uint32_t)(gi >> d.lo), o = (uint32_t)(gi & (d.Opad - 1));
            U[gi] = gl_mul(eqO[o], ug[g]); V[gi] = gl_mul(eqO[o], vg[g]);
            if (g == 0) U0[gi] = eqO[o];
        }
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(std::move(U)); cols.push_back(std::move(V)); cols.push_back(std::move(U0));
        cols.push_back(CDg[G_ASIG].v); cols.push_back(CDg[G_BEXP].v); cols.push_back(CDg[G_ASGN].v);
        cols.push_back(CDg[G_TZ].v); cols.push_back(CDg[G_S14].v); cols.push_back(CDg[G_MAX].v);
        cols.push_back(CDg[G_WD].v); cols.push_back(CDg[G_TSGN].v);
        CFn F = [&](const gl_t* v) { return F_ch(v, lam6); };
        std::vector<gl_t> rH = sc5rz<FFch>(tr, "hwl-scH", d.lgi, std::move(cols), lam6, 6,
                                           F, pf.mCh, 0, R, lg, lucols, pf.zbl[2]);
        static const int CH_COLS[8] = {G_ASIG, G_BEXP, G_ASGN, G_TZ, G_S14, G_MAX, G_WD, G_TSGN};
        for (int j = 0; j < 8; j++)
            pf.yCh[j] = claimc(tr, lg, CDg[CH_COLS[j]], rH);
    }
    P.chain += now_ms() - tp;

    // -- group-sum + attainment-sum binding (Dp -> Dg) --
    // zk: a SUM binding, so the claims use one fresh ex challenge and the
    // slice-1 masks are linked (mCSUM/mASEL above); the weight is built from
    // the target point zero-extended to the product domain's ex count
    tp = now_ms();
    std::vector<gl_t> z2 = chal_vec(tr, d.lgi);
    gl_t zex2 = zk ? chal(tr) : 0;
    pf.yGSc = claimc(tr, lg, CDg[G_CSUM], p3zkc::xpt(z2, zex2));
    pf.yGSa = claimc(tr, lg, CDg[G_ASEL], p3zkc::xpt(z2, zex2));
    gl_t lamS1 = chal(tr), lamS2 = chal(tr);
    {
        uint32_t e_p = p3zkc::e_of(d.lp);
        std::vector<gl_t> ptg = z2;
        if (zk) { ptg.push_back(zex2); ptg.resize(d.lgi + e_p, 0); }
        std::vector<gl_t> eqDg = beq(ptg);
        size_t Paug = CDp[P_AL].v.size();
        std::vector<gl_t> EQg(Paug);
        for (size_t q = 0; q < Paug; q++) {
            size_t ex = q >> d.lp, p = q & (((size_t)1 << d.lp) - 1);
            EQg[q] = eqDg[(ex << d.lgi) | (p >> 5)];
        }
        std::vector<ColSrc> cols;
        cols.push_back(ColSrc(std::move(EQg)));
        cols.push_back(ColSrc(&CDp[P_AL].v)); cols.push_back(ColSrc(&CDp[P_SEL].v));
        gl_t lam2[2] = {lamS1, lamS2};
        CFn F = [&](const gl_t* v) { return F_gs(v, lam2); };
        gl_t base0 = gl_add(gl_mul(lamS1, pf.yGSc), gl_mul(lamS2, gl_sub(1ULL, pf.yGSa)));
        std::vector<gl_t> rS = sc5rz_srcs<FFgs>(tr, "hwl-scS", d.lp, std::move(cols), lam2, 2,
                                                F, pf.mGS, base0, R, lg, lucols, pf.zbl[3]);
        pf.yGSal = claimc(tr, lg, CDp[P_AL], rS);
        pf.yGSsel = claimc(tr, lg, CDp[P_SEL], rS);
    }
    P.gsum += now_ms() - tp;

    // -- Do zero-check (P6: fp32 assembly, 2x RNE mul, bf16 downcast) --
    tp = now_ms();
    std::vector<gl_t> zO = chal_vec(tr, d.lo);
    gl_t lamO = chal(tr), lamOv[N_DO_CONSTR]; lamOv[0] = 1;
    for (int j = 1; j < N_DO_CONSTR; j++) lamOv[j] = gl_mul(lamOv[j-1], lamO);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(1 + NDO + NVO);
        cols.push_back(p3bf::build_eq(p3zkc::zpt(zO)));
        for (int c = 0; c < NDO; c++) cols.push_back(CDo[c].v);
        static const int BCX[5] = {S_SS, S_SE, S_ZS, S_SMH, S_SML};
        uint32_t lgB = ilog2(d.Bpad), lgN = ilog2(d.Npad);
        for (int j = 0; j < 5; j++)                         // x-scale broadcasts
            cols.push_back(p3zkc::bc_aug(CDb[BCX[j]].v, lgB, d.lo, d.Opad,
                                         [&](size_t o) { return o >> d.ln; }));
        for (int j = 0; j < 5; j++)                         // w-scale broadcasts
            cols.push_back(p3zkc::bc_aug(CDn[BCX[j]].v, lgN, d.lo, d.Opad,
                                         [&](size_t o) { return o & (d.Npad - 1); }));
        CFn F = [&](const gl_t* v) { return F_do(v, lamOv); };
        std::vector<gl_t> rO = sc5z(tr, "hwl-scO", d.lo, std::move(cols), F, pf.mDo,
                                    0, R, lg, lucols, pf.zbl[4]);
        P.zc_do += now_ms() - tp; tp = now_ms();
        for (int c = 0; c < NDO; c++)
            pf.yDo[c] = claimc(tr, lg, CDo[c], rO);
        std::vector<gl_t> pB(rO.begin() + d.ln, rO.begin() + d.lo),
                          pN(rO.begin(), rO.begin() + d.ln);
        for (int j = 0; j < 5; j++)
            pf.yDbBC[j] = claimc(tr, lg, CDb[BCX[j]], p3zkc::expt(pB, rO, d.lo));
        for (int j = 0; j < 5; j++)
            pf.yDnBC[j] = claimc(tr, lg, CDn[BCX[j]], p3zkc::expt(pN, rO, d.lo));
        P.open_do += now_ms() - tp;
    }

    // -- Db / Dn scale-decomposition zero-checks (bound to the xs/ws commitments) --
    tp = now_ms();
    {
        std::vector<gl_t> zB = chal_vec(tr, d.lb);
        gl_t lamB = chal(tr), lamBv[6]; lamBv[0] = 1;
        for (int j = 1; j < 6; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(p3bf::build_eq(p3zkc::zpt(zB))); cols.push_back(ops.XS.v);
        static const int SC[6] = {S_SS, S_SE, S_SEI, S_ZS, S_SMH, S_SML};
        for (int j = 0; j < 6; j++) cols.push_back(CDb[SC[j]].v);
        CFn F = [&](const gl_t* v) { return F_ds(v, lamBv); };
        std::vector<gl_t> rB = sc5z(tr, "hwl-scB", d.lb, std::move(cols), F, pf.mDb,
                                    0, R, lg, lucols, pf.zbl[5]);
        for (int j = 0; j < 6; j++)
            pf.yDb[j] = claimc(tr, lg, CDb[SC[j]], rB);
        pf.yXsb = claimc(tr, lg, ops.XS, rB);
    }
    {
        std::vector<gl_t> zN = chal_vec(tr, d.ln);
        gl_t lamN = chal(tr), lamNv[6]; lamNv[0] = 1;
        for (int j = 1; j < 6; j++) lamNv[j] = gl_mul(lamNv[j-1], lamN);
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(p3bf::build_eq(p3zkc::zpt(zN))); cols.push_back(ops.WS.v);
        static const int SC[6] = {S_SS, S_SE, S_SEI, S_ZS, S_SMH, S_SML};
        for (int j = 0; j < 6; j++) cols.push_back(CDn[SC[j]].v);
        CFn F = [&](const gl_t* v) { return F_ds(v, lamNv); };
        std::vector<gl_t> rN = sc5z(tr, "hwl-scN", d.ln, std::move(cols), F, pf.mDn,
                                    0, R, lg, lucols, pf.zbl[6]);
        for (int j = 0; j < 6; j++)
            pf.yDn[j] = claimc(tr, lg, CDn[SC[j]], rN);
        pf.yWsb = claimc(tr, lg, ops.WS, rN);
    }
    P.zc_ds += now_ms() - tp;

    // -- final-state slice binding: A(.,g=NG) feeds P6 --
    // zk: claim algebra across two classes -> shared fresh ex challenge and
    // slice-1 mask linkage (mS14F/mOBEX/mOSGN above)
    tp = now_ms();
    {
        std::vector<gl_t> zc2 = chal_vec(tr, d.lo), ptg = zc2;
        for (uint32_t i = 0; i < d.lg; i++) ptg.push_back((d.NG >> i) & 1);
        gl_t zex3 = zk ? chal(tr) : 0;
        pf.ySlA = claimc(tr, lg, CDg[G_ASIG], p3zkc::xpt(ptg, zex3));
        pf.ySlE = claimc(tr, lg, CDg[G_BEXP], p3zkc::xpt(ptg, zex3));
        pf.ySlS = claimc(tr, lg, CDg[G_ASGN], p3zkc::xpt(ptg, zex3));
        pf.ySlF = claimc(tr, lg, CDo[O_S14F], p3zkc::xpt(zc2, zex3));
        pf.ySlBE = claimc(tr, lg, CDo[O_BEXP], p3zkc::xpt(zc2, zex3));
        pf.ySlSG = claimc(tr, lg, CDo[O_SGN], p3zkc::xpt(zc2, zex3));
    }
    P.slice += now_ms() - tp;

    // -- public-Y binding restricted to the real BxN grid (non-zk only: the
    //    zk composed layer binds outputs through the seams + the final public
    //    output binding instead of per-matmul public vectors) --
    tp = now_ms();
    if (!zk) {
        std::vector<gl_t> zY = chal_vec(tr, d.lo);
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(p3bf::build_eq(zY));
        std::vector<gl_t> Rw(d.Opad, 0);
        for (uint32_t o = 0; o < d.Opad; o++) {
            uint32_t b = o >> d.ln, n = o & (d.Npad - 1);
            if (b < d.B && n < d.N) Rw[o] = 1;
        }
        cols.push_back(std::move(Rw)); cols.push_back(wt.dob[O_YB]);
        CFn F = [](const gl_t* v) { return F_y(v, nullptr); };
        std::vector<gl_t> rY = sc5_prove(tr, "hwl-scY", std::move(cols), F, pf.mY);
        pf.yY = claimc(tr, lg, CDo[O_YB], rY);
    }
    P.ybind += now_ms() - tp;

    // -- batched openings: one reduction + one RLC opening per size class --
    // (deferred to the caller's shared batch under an external ledger)
    tp = now_ms();
    if (!xc) {
        double tl = now_ms();
        pf.lug = p3lu::lu_flush(tr, XC, R, Q, strict, gpu);
        P.lu_dp += now_ms() - tl;              // lumped merged-lookup time
        tp = now_ms();
        for (size_t i = 0; i < lg.cls.size(); i++)
            pf.batches.push_back(p3bo::prove_class(tr, lg.cls[i], R, Q,
                                                   "hwl-bo" + std::to_string(i)));
    }
    P.batch += now_ms() - tp;
    P.total += now_ms() - tall;
    p3bf::trim_heap();
    return pf;
}

// ==================== verifier ====================
// Q_pub/R_pub, dims and the operand/output commitments are PUBLIC caller
// inputs -- never read from the proof.
static inline bool verify(fs::Transcript& tr, const Tables& T, const LayerProof& pf,
                          const p3fri::Hash& rX, const p3fri::Hash& rW,
                          const p3fri::Hash& rXS, const p3fri::Hash& rWS,
                          const std::vector<uint16_t>& Y,
                          uint32_t B, uint32_t K, uint32_t N,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr,
                          p3lu::VCtx* xv = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    const bool zk = p3zkc::G.on;
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.B != B || pf.K != K || pf.N != N) return fail("dims mismatch");
    if (!zk && Y.size() != (size_t)B * N) return fail("Y size");
    Dims d = make_dims(B, K, N);
    p3lu::VCtx vc_loc;                       // deferred opening + lookup obligations
    p3lu::VCtx& VC = xv ? *xv : vc_loc;
    p3bo::VLedger& vlg = VC.vlg;

    uint32_t dims3[3] = {B, K, N};
    tr.absorb("hwl-dims", dims3, sizeof dims3);
    const Table* tabs[16] = {&T.DM, &T.SH, &T.RM, &T.R15, &T.ASH, &T.WIDTH, &T.R10, &T.R11,
                             &T.R12, &T.R16, &T.FH, &T.REXP, &T.SE, &T.CREM13, &T.CREM12, &T.CTH};
    for (auto* t : tabs) tr.absorb("hwl-tab", t->id.data(), 32);
    tr.absorb("hwl-X", rX.data(), 32); tr.absorb("hwl-W", rW.data(), 32);
    tr.absorb("hwl-xs", rXS.data(), 32); tr.absorb("hwl-ws", rWS.data(), 32);
    if (!zk) tr.absorb("hwl-Y", Y.data(), Y.size() * 2);
    for (int c = 0; c < NDP; c++) tr.absorb("hwl-cp", pf.rdp[c].data(), 32);
    for (int c = 0; c < NDG; c++) tr.absorb("hwl-cg", pf.rdg[c].data(), 32);
    for (int c = 0; c < NDO; c++) tr.absorb("hwl-co", pf.rdo[c].data(), 32);
    for (int c = 0; c < NDS; c++) tr.absorb("hwl-cb", pf.rdb[c].data(), 32);
    for (int c = 0; c < NDS; c++) tr.absorb("hwl-cn", pf.rdn[c].data(), 32);

    // -- lookups: DEFERRED to the ledger owner's merged-group flush --
    auto LD = lu_defs(T);
    for (int i = 0; i < NLU; i++) {
        uint32_t explog = LD[i].dom == 0 ? d.lp : LD[i].dom == 1 ? d.lgi :
                          LD[i].dom == 2 ? d.lo : LD[i].dom == 3 ? d.lb : d.ln;
        std::vector<const p3fri::Hash*> roots;
        for (int cid : LD[i].cols) {
            if (cid < 0) roots.push_back(nullptr);
            else roots.push_back(LD[i].dom == 0 ? &pf.rdp[cid] : LD[i].dom == 1 ? &pf.rdg[cid] :
                                 LD[i].dom == 2 ? &pf.rdo[cid] : LD[i].dom == 3 ? &pf.rdb[cid] : &pf.rdn[cid]);
        }
        p3lu::VBind bind;
        if (i == LU_DM) {
            // bind the virtual (a,b) evaluations to the X/W commitments
            p3fri::Hash hX = rX, hW = rW; Dims dd = d;
            bind = [hX, hW, dd](fs::Transcript&, p3lu::VCtx& vc,
                                const std::vector<gl_t>& pm, const std::vector<gl_t>& yv,
                                const std::vector<gl_t>&, const char** wy) {
                if (yv.size() != 2) { if (wy) *wy = "DM y_virt count"; return false; }
                vc.vlg.add(hX, p3zkc::expt(zx_point(dd, pm), pm, dd.lp), yv[0]);
                vc.vlg.add(hW, p3zkc::expt(zw_point(dd, pm), pm, dd.lp), yv[1]);
                return true;
            };
        }
        p3lu::vdefer_v(VC, std::move(roots), *LD[i].tab, explog, LD[i].label, std::move(bind));
    }

    // -- Dp zero-check --
    std::vector<gl_t> zC = chal_vec(tr, d.lp);
    gl_t lamP = chal(tr), lamPv[6]; lamPv[0] = 1;
    for (int j = 1; j < 6; j++) lamPv[j] = gl_mul(lamPv[j-1], lamP);
    {
        gl_t rho = sc5vz_pre(tr, pf.zbl[0]);
        std::vector<gl_t> rC; gl_t claim;
        if (!sc5_verify(pf.mDp, p3zkc::vfull(d.lp), gl_mul(rho, pf.zbl[0].H),
                        tr, "hwl-scP", rC, claim)) return fail("Dp sumcheck");
        sc5vz_claims(tr, vlg, pf.zbl[0], rC);
        gl_t v[2 + NDP]; v[0] = p3bf::eq_point(rC, p3zkc::zpt(zC));
        for (int c = 0; c < NDP; c++)
            v[1 + c] = claimv(tr, vlg, pf.rdp[c], rC, pf.yDp[c]);
        std::vector<gl_t> rCg(rC.begin() + 5, rC.begin() + d.lp);
        v[1 + NDP] = claimv(tr, vlg, pf.rdg[G_MAX], p3zkc::expt(rCg, rC, d.lp), pf.yDpG);
        gl_t end = gl_add(F_dp(v, lamPv), sc5_blindterm(pf.zbl[0], rho, v[0]));
        if (end != claim) return fail("Dp terminal");
    }

    // -- Dg zero-check --
    std::vector<gl_t> zG = chal_vec(tr, d.lgi);
    gl_t lamG = chal(tr), lamGv[18]; lamGv[0] = 1;
    for (int j = 1; j < 18; j++) lamGv[j] = gl_mul(lamGv[j-1], lamG);
    {
        gl_t rho = sc5vz_pre(tr, pf.zbl[1]);
        std::vector<gl_t> rG; gl_t claim;
        if (!sc5_verify(pf.mDg, p3zkc::vfull(d.lgi), gl_mul(rho, pf.zbl[1].H),
                        tr, "hwl-scG", rG, claim)) return fail("Dg sumcheck");
        sc5vz_claims(tr, vlg, pf.zbl[1], rG);
        gl_t v[1 + NDG]; v[0] = p3bf::eq_point(rG, p3zkc::zpt(zG));
        for (int c = 0; c < NDG; c++)
            v[1 + c] = claimv(tr, vlg, pf.rdg[c], rG, pf.yDg[c]);
        gl_t end = gl_add(F_dg(v, lamGv), sc5_blindterm(pf.zbl[1], rho, v[0]));
        if (end != claim) return fail("Dg terminal");
    }

    // -- accumulator chain --
    std::vector<gl_t> zc = chal_vec(tr, d.lo);
    gl_t lch = chal(tr), lam6[6];
    for (int j = 0; j < 6; j++) lam6[j] = chal(tr);
    {
        gl_t rho = sc5vz_pre(tr, pf.zbl[2]);
        std::vector<gl_t> rH; gl_t claim;
        if (!sc5_verify(pf.mCh, p3zkc::vfull(d.lgi), gl_mul(rho, pf.zbl[2].H),
                        tr, "hwl-scH", rH, claim)) return fail("chain sumcheck");
        sc5vz_claims(tr, vlg, pf.zbl[2], rH);
        static const int CH_COLS[8] = {G_ASIG, G_BEXP, G_ASGN, G_TZ, G_S14, G_MAX, G_WD, G_TSGN};
        gl_t v[11];
        for (int j = 0; j < 8; j++)
            v[3 + j] = claimv(tr, vlg, pf.rdg[CH_COLS[j]], rH, pf.yCh[j]);
        std::vector<gl_t> lpow(d.Gpad, 1);
        for (uint32_t g = 1; g < d.Gpad; g++) lpow[g] = gl_mul(lpow[g-1], lch);
        std::vector<gl_t> ug(d.Gpad, 0), vg(d.Gpad, 0);
        for (uint32_t g = 1; g <= d.Gpad - 1; g++) ug[g] = lpow[g-1];
        for (uint32_t g = 0; g + 1 <= d.Gpad - 1; g++) vg[g] = lpow[g];
        std::vector<gl_t> rlo(rH.begin(), rH.begin() + d.lo),
                          rhi(rH.begin() + d.lo, rH.begin() + d.lgi);
        std::vector<gl_t> eqg = p3bf::build_eq(rhi);
        gl_t uT = 0, vT = 0;
        for (uint32_t g = 0; g < d.Gpad; g++) {
            uT = gl_add(uT, gl_mul(ug[g], eqg[g])); vT = gl_add(vT, gl_mul(vg[g], eqg[g]));
        }
        gl_t eqo = p3bf::eq_point(rlo, zc);
        gl_t exf = 1ULL;                          // zk: U/V/U0 are zero-extended
        for (size_t i = d.lgi; i < rH.size(); i++) exf = gl_mul(exf, gl_sub(1ULL, rH[i]));
        eqo = gl_mul(eqo, exf);
        v[0] = gl_mul(eqo, uT); v[1] = gl_mul(eqo, vT); v[2] = gl_mul(eqo, eqg[0]);
        gl_t end = gl_add(F_ch(v, lam6), sc5_blindterm(pf.zbl[2], rho, v[0]));
        if (end != claim) return fail("chain terminal");
    }

    // -- group-sum + attainment --
    std::vector<gl_t> z2 = chal_vec(tr, d.lgi);
    {
        gl_t zex2 = zk ? chal(tr) : 0;
        gl_t yc = claimv(tr, vlg, pf.rdg[G_CSUM], p3zkc::xpt(z2, zex2), pf.yGSc);
        gl_t ya = claimv(tr, vlg, pf.rdg[G_ASEL], p3zkc::xpt(z2, zex2), pf.yGSa);
        gl_t lamS1 = chal(tr), lamS2 = chal(tr);
        gl_t claim0 = gl_add(gl_mul(lamS1, yc), gl_mul(lamS2, gl_sub(1ULL, ya)));
        gl_t rho = sc5vz_pre(tr, pf.zbl[3]);
        claim0 = gl_add(claim0, gl_mul(rho, pf.zbl[3].H));
        std::vector<gl_t> rS; gl_t claim;
        if (!sc5_verify(pf.mGS, p3zkc::vfull(d.lp), claim0, tr, "hwl-scS", rS, claim))
            return fail("gsum sumcheck");
        sc5vz_claims(tr, vlg, pf.zbl[3], rS);
        gl_t yal = claimv(tr, vlg, pf.rdp[P_AL], rS, pf.yGSal);
        gl_t ysel = claimv(tr, vlg, pf.rdp[P_SEL], rS, pf.yGSsel);
        // weight terminal: the eq of (z2 || zex2 || 0..) against the group+ex
        // coordinates of rS (constant in the 5 kk coordinates)
        std::vector<gl_t> rSg(rS.begin() + 5, rS.begin() + d.lp);
        rSg.insert(rSg.end(), rS.begin() + d.lp, rS.end());
        std::vector<gl_t> ptg = z2;
        if (zk) { ptg.push_back(zex2); ptg.resize(d.lgi + p3zkc::e_of(d.lp), 0); }
        gl_t w = p3bf::eq_point(rSg, ptg);
        gl_t end = gl_mul(w, gl_add(gl_mul(lamS1, yal), gl_mul(lamS2, ysel)));
        end = gl_add(end, sc5_blindterm(pf.zbl[3], rho, w));
        if (end != claim) return fail("gsum terminal");
    }

    // -- Do zero-check --
    std::vector<gl_t> zO = chal_vec(tr, d.lo);
    gl_t lamO = chal(tr), lamOv[N_DO_CONSTR]; lamOv[0] = 1;
    for (int j = 1; j < N_DO_CONSTR; j++) lamOv[j] = gl_mul(lamOv[j-1], lamO);
    {
        gl_t rho = sc5vz_pre(tr, pf.zbl[4]);
        std::vector<gl_t> rO; gl_t claim;
        if (!sc5_verify(pf.mDo, p3zkc::vfull(d.lo), gl_mul(rho, pf.zbl[4].H),
                        tr, "hwl-scO", rO, claim)) return fail("Do sumcheck");
        sc5vz_claims(tr, vlg, pf.zbl[4], rO);
        gl_t v[1 + NDO + NVO]; v[0] = p3bf::eq_point(rO, p3zkc::zpt(zO));
        for (int c = 0; c < NDO; c++)
            v[1 + c] = claimv(tr, vlg, pf.rdo[c], rO, pf.yDo[c]);
        std::vector<gl_t> pB(rO.begin() + d.ln, rO.begin() + d.lo),
                          pN(rO.begin(), rO.begin() + d.ln);
        static const int BCX[5] = {S_SS, S_SE, S_ZS, S_SMH, S_SML};
        for (int j = 0; j < 5; j++)
            v[1 + NDO + j] = claimv(tr, vlg, pf.rdb[BCX[j]], p3zkc::expt(pB, rO, d.lo), pf.yDbBC[j]);
        for (int j = 0; j < 5; j++)
            v[1 + NDO + 5 + j] = claimv(tr, vlg, pf.rdn[BCX[j]], p3zkc::expt(pN, rO, d.lo), pf.yDnBC[j]);
        gl_t end = gl_add(F_do(v, lamOv), sc5_blindterm(pf.zbl[4], rho, v[0]));
        if (end != claim) return fail("Do terminal");
    }

    // -- Db / Dn scale decomposition --
    {
        std::vector<gl_t> zB = chal_vec(tr, d.lb);
        gl_t lamB = chal(tr), lamBv[6]; lamBv[0] = 1;
        for (int j = 1; j < 6; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
        gl_t rho = sc5vz_pre(tr, pf.zbl[5]);
        std::vector<gl_t> rB; gl_t claim;
        if (!sc5_verify(pf.mDb, p3zkc::vfull(d.lb), gl_mul(rho, pf.zbl[5].H),
                        tr, "hwl-scB", rB, claim)) return fail("Db sumcheck");
        sc5vz_claims(tr, vlg, pf.zbl[5], rB);
        static const int SC[6] = {S_SS, S_SE, S_SEI, S_ZS, S_SMH, S_SML};
        gl_t v[8]; v[0] = p3bf::eq_point(rB, p3zkc::zpt(zB));
        for (int j = 0; j < 6; j++)
            v[2 + j] = claimv(tr, vlg, pf.rdb[SC[j]], rB, pf.yDb[j]);
        v[1] = claimv(tr, vlg, rXS, rB, pf.yXsb);
        gl_t end = gl_add(F_ds(v, lamBv), sc5_blindterm(pf.zbl[5], rho, v[0]));
        if (end != claim) return fail("Db terminal");
    }
    {
        std::vector<gl_t> zN = chal_vec(tr, d.ln);
        gl_t lamN = chal(tr), lamNv[6]; lamNv[0] = 1;
        for (int j = 1; j < 6; j++) lamNv[j] = gl_mul(lamNv[j-1], lamN);
        gl_t rho = sc5vz_pre(tr, pf.zbl[6]);
        std::vector<gl_t> rN; gl_t claim;
        if (!sc5_verify(pf.mDn, p3zkc::vfull(d.ln), gl_mul(rho, pf.zbl[6].H),
                        tr, "hwl-scN", rN, claim)) return fail("Dn sumcheck");
        sc5vz_claims(tr, vlg, pf.zbl[6], rN);
        static const int SC[6] = {S_SS, S_SE, S_SEI, S_ZS, S_SMH, S_SML};
        gl_t v[8]; v[0] = p3bf::eq_point(rN, p3zkc::zpt(zN));
        for (int j = 0; j < 6; j++)
            v[2 + j] = claimv(tr, vlg, pf.rdn[SC[j]], rN, pf.yDn[j]);
        v[1] = claimv(tr, vlg, rWS, rN, pf.yWsb);
        gl_t end = gl_add(F_ds(v, lamNv), sc5_blindterm(pf.zbl[6], rho, v[0]));
        if (end != claim) return fail("Dn terminal");
    }

    // -- final-state slice binding --
    {
        std::vector<gl_t> zc2 = chal_vec(tr, d.lo), ptg = zc2;
        for (uint32_t i = 0; i < d.lg; i++) ptg.push_back((d.NG >> i) & 1);
        gl_t zex3 = zk ? chal(tr) : 0;
        gl_t yA = claimv(tr, vlg, pf.rdg[G_ASIG], p3zkc::xpt(ptg, zex3), pf.ySlA);
        gl_t yE = claimv(tr, vlg, pf.rdg[G_BEXP], p3zkc::xpt(ptg, zex3), pf.ySlE);
        gl_t yS = claimv(tr, vlg, pf.rdg[G_ASGN], p3zkc::xpt(ptg, zex3), pf.ySlS);
        gl_t yF = claimv(tr, vlg, pf.rdo[O_S14F], p3zkc::xpt(zc2, zex3), pf.ySlF);
        gl_t yBE = claimv(tr, vlg, pf.rdo[O_BEXP], p3zkc::xpt(zc2, zex3), pf.ySlBE);
        gl_t ySG = claimv(tr, vlg, pf.rdo[O_SGN], p3zkc::xpt(zc2, zex3), pf.ySlSG);
        if (yA != gl_mul(1024ULL, yF)) return fail("slice sig binding");
        if (yE != yBE) return fail("slice exp binding");
        if (yS != ySG) return fail("slice sign binding");
    }

    // -- public-Y binding (non-zk only) --
    if (!zk) {
        std::vector<gl_t> zY = chal_vec(tr, d.lo);
        std::vector<gl_t> eqY = p3bf::build_eq(zY);
        gl_t claim0 = 0;
        for (uint32_t b = 0; b < B; b++)
            for (uint32_t n = 0; n < N; n++)
                claim0 = gl_add(claim0, gl_mul((gl_t)Y[(size_t)b * N + n],
                                               eqY[(size_t)b * d.Npad + n]));
        std::vector<gl_t> rY; gl_t claim;
        if (!sc5_verify(pf.mY, d.lo, claim0, tr, "hwl-scY", rY, claim)) return fail("Y sumcheck");
        gl_t yyb = claimv(tr, vlg, pf.rdo[O_YB], rY, pf.yY);
        std::vector<gl_t> rn(rY.begin(), rY.begin() + d.ln), rb(rY.begin() + d.ln, rY.end());
        std::vector<gl_t> eqn = p3bf::build_eq(rn), eqb = p3bf::build_eq(rb);
        gl_t realN = 0, realB = 0;
        for (uint32_t n = 0; n < N; n++) realN = gl_add(realN, eqn[n]);
        for (uint32_t b = 0; b < B; b++) realB = gl_add(realB, eqb[b]);
        gl_t end = gl_mul(gl_mul(yyb, gl_mul(realB, realN)), p3bf::eq_point(rY, zY));
        if (end != claim) return fail("Y terminal");
    }

    // -- batched openings: every claimed evaluation above gets proven here --
    if (!xv) {
        if (!p3lu::lu_verify_flush(tr, VC, pf.lug, Q_pub, R_pub, why)) return false;
        if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
        for (size_t i = 0; i < vlg.cls.size(); i++)
            if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                    "hwl-bo" + std::to_string(i), why)) return false;
    } else if (!pf.batches.empty() || !pf.lug.empty()) return fail("unexpected batches");

    if (why) *why = "ok";
    return true;
}

// ==================== proof size (serialized-payload bytes) ====================
static inline size_t sz_eval(const p3bf::EvalProof& e) {
    size_t s = 12 + e.roots.size() * 32 + e.msgs.size() * 24 + e.final_word.size() * 8
             + e.z.size() * 8 + 8;
    for (auto& q : e.queries)
        for (auto& r : q.rounds) s += 16 + (r.pa.size() + r.pb.size()) * 32;
    return s;
}
static inline size_t sz_lu(const p3lu::LookupProof& l) {
    size_t s = 12 + 3 * 32 + 8 + (l.msgsA.size() + l.msgsT.size()) * 32 + l.y_virt.size() * 8;
    s += sz_eval(l.open_hA) + sz_eval(l.open_hT) + sz_eval(l.open_cnt);
    for (auto& o : l.open_W) s += sz_eval(o);
    s += l.yW.size() * 8 + 24;                         // deferred-mode claims
    return s;
}
static inline size_t proof_size(const LayerProof& pf) {
    size_t s = 12 + (NDP + NDG + NDO + 2 * NDS) * 32;
    for (auto& g : pf.lug) s += p3lu::sz_group(g);
    auto msgs = [&](const std::vector<Msg5>& m) { return m.size() * 40; };
    s += msgs(pf.mDp) + msgs(pf.mDg) + msgs(pf.mCh) + msgs(pf.mGS) + msgs(pf.mDo)
       + msgs(pf.mDb) + msgs(pf.mDn) + msgs(pf.mY);
    // claimed evaluations
    s += 8 * (NDP + 1 + NDG + 8 + 4 + NDO + 10 + 6 + 6 + 2 + 6 + 1);
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3hwl
