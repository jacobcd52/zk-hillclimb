// Quantize gadget: sound (non-ZK) prover+verifier that committed fp8 CODES and
// per-row fp32 SCALES equal the CANONICAL activation quantization
// (transformer_ref.py quant_rows_e4m3) of a committed bf16 input X, bitwise.
// This is the glue that chains every op output into the next Hawkeye matmul
// (7 call sites per layer: rms->Wq/Wk/Wv, rope->QK^T, softmax->.V, rms2->Wg/Wu,
// swiglu->Wd).
//
//   per row b (length d = 2^ld):
//     decompose x = s*2^15 + eb*2^7 + mb; z = (eb==0)  [canonical FTZ]
//     E  = max(9, max_present eb)         [dominance + attainment selectors,
//                                          phantom floor at 9: the reference's
//                                          se = max(emax-8, 1) = E - 8]
//     SCALE = (E - 8) << 23               [pow2 fp32, sign/mantissa zero]
//     per element: dexp = E - eb (present; the dominance shift itself),
//       mag = QEXT[dexp, mb]              [QE4M3 table EXTENDED to dexp<256:
//                                          rows >= 32 are 0 = the reference's
//                                          "underflows the e4m3 grid" rule]
//       CODE = s*128 + (1-z)*mag
//
// The QEXT lookup simultaneously range-binds dexp in [0,256) (no wraparound in
// the dominance identity, since eb is R256-bound) and mb in [0,128), so the
// whole gadget costs THREE logUp instances.  Total: no domain restrictions --
// the canonical reference is total here and so is the gadget.
//
// Reuses the p3 stack end to end (p3lu deferred lookups, p3hwl quartic
// zero-checks, p3bo batched openings).  Non-ZK: section-10 masking applies to
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
#include "p3_rmsnorm.cuh"
#include "fs_transcript.hpp"

namespace p3qnt {

using p3lu::Col; using p3lu::Table; using p3lu::make_table; using p3lu::commit_col_nc;
using p3lu::chal; using p3lu::chal_vec; using p3lu::ilog2;
using p3lu::LuColSpec; using p3lu::LC; using p3lu::LV;
using p3hwl::Msg5; using p3hwl::CFn; using p3hwl::sc5_prove; using p3hwl::sc5_verify;
using p3hwl::claimc; using p3hwl::claimv; using p3hwl::beq; using p3hwl::pow2_at_least;
using p3rms::Art;

static const int64_t Q_EFLOOR = 9;         // E = max(9, emax): se = E - 8 >= 1
static const int64_t Q_DEXP_ROWS = 32;     // artifact rows; extension is 0

// ---------------- tables ----------------
struct Tables {
    Table R256;                            // eb / edif ranges
    Table QEXT;                            // (dexp, mb, mag) 256*128 rows
};
static inline Tables build_tables(const Art& a) {
    Tables T;
    { std::vector<gl_t> v(256);
      for (uint32_t j = 0; j < 256; j++) v[j] = j;
      T.R256 = make_table({v}); }
    { std::vector<gl_t> dx(32768), mb(32768), mg(32768);
      for (uint32_t j = 0; j < 32768; j++) {
          uint32_t dexp = j >> 7, m = j & 127;
          dx[j] = dexp; mb[j] = m;
          mg[j] = dexp < (uint32_t)Q_DEXP_ROWS ? a.qe4m3[(dexp << 7) | m] : 0;
      }
      T.QEXT = make_table({dx, mb, mg}); }
    return T;
}

// ---------------- column enums ----------------
enum {  // De: per-element (index e = b*d + i, bits [i | b])
    D_XS = 0, D_XEB, D_XMB, D_XZ, D_XEI, D_SH, D_SEL, D_MAG, NDE };
enum {  // Db: per-row
    R_E = 0, R_EDIF, R_FSEL, NDB };
enum {  // lookup instances, fixed order
    LUQ_XEB = 0, LUQ_QE, LUQ_EDIF, NLUQ };

// ---------------- dims / golden ----------------
struct Dims {
    uint32_t B, d, ld;
    uint32_t Bpad, lb, le;
    size_t Ne;
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
    std::vector<uint16_t> x;                  // B*d bf16 patterns
    std::vector<uint8_t> codes;               // B*d e4m3 codes
    std::vector<uint32_t> scales;             // B fp32 bit patterns
};
static inline bool load_goldens(const char* path, std::vector<Golden>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[2];
    if (fread(hdr, 8, 2, f) != 2 || hdr[0] != 0x514E5447) { fclose(f); return false; }
    out.resize(hdr[1]);
    for (auto& G : out) {
        int64_t bd[2];
        if (fread(bd, 8, 2, f) != 2) { fclose(f); return false; }
        G.B = (uint32_t)bd[0]; G.d = (uint32_t)bd[1];
        size_t n = (size_t)G.B * G.d;
        G.x.resize(n); G.codes.resize(n); G.scales.resize(G.B);
        if (fread(G.x.data(), 2, n, f) != n ||
            fread(G.codes.data(), 1, n, f) != n ||
            fread(G.scales.data(), 4, G.B, f) != G.B) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

// ---------------- witness ----------------
// Selftest-only semantic forgeries: ONE forgery, honest downstream replay, so
// exactly one sub-argument must catch it.
enum { QT_NONE = 0,
       QT_MAXEXP,    // row E overstated by 1 (no attainer -> fsel forced)
       QT_MAG,       // element magnitude code MAG+1 (not a QEXT row)
       QT_SCALE };   // row scale one binade up (E honest -> Db zero-check)
struct QTamper { int mode = QT_NONE; uint32_t b = 0, i = 0; };

struct Wit {
    Dims dm;
    std::vector<gl_t> xpat, cpat, spat;       // X patterns, codes, scales (padded)
    std::vector<uint8_t> C;                   // real B*d computed codes
    std::vector<uint32_t> S;                  // real B computed scale bits
    std::vector<gl_t> de[NDE], db[NDB];
    std::vector<uint32_t> lidx[NLUQ];
};
static inline gl_t inv_or0(uint64_t x) { return x ? gl_inv((gl_t)x) : 0; }

// canonical witness replay (mirrors transformer_ref.quant_rows_e4m3 bit for bit)
static inline Wit gen_witness(const Golden& L, const Art& a, const QTamper* tm = nullptr) {
    Wit wt; Dims& dm = wt.dm;
    uint32_t ld = ilog2(L.d);
    if ((1u << ld) != L.d) throw std::runtime_error("qnt: d must be pow2");
    dm = make_dims(L.B, ld);
    const uint32_t d = dm.d;

    wt.xpat.assign(dm.Ne, 0); wt.cpat.assign(dm.Ne, 0); wt.spat.assign(dm.Bpad, 0);
    for (uint32_t b = 0; b < L.B; b++)
        for (uint32_t i = 0; i < d; i++)
            wt.xpat[((size_t)b << ld) | i] = L.x[(size_t)b * d + i];
    wt.C.assign((size_t)L.B * d, 0);
    wt.S.assign(L.B, 0);

    for (int c = 0; c < NDE; c++) wt.de[c].assign(dm.Ne, 0);
    for (int c = 0; c < NDB; c++) wt.db[c].assign(dm.Bpad, 0);
    wt.lidx[LUQ_XEB].assign(dm.Ne, 0);
    wt.lidx[LUQ_QE].assign(dm.Ne, 0);
    wt.lidx[LUQ_EDIF].assign(dm.Bpad, 0);

    for (uint32_t b = 0; b < dm.Bpad; b++) {
        std::vector<int64_t> xs(d), xeb(d), xmb(d), xz(d);
        for (uint32_t i = 0; i < d; i++) {
            uint32_t p = (uint32_t)wt.xpat[((size_t)b << ld) | i];
            xs[i] = (p >> 15) & 1; xeb[i] = (p >> 7) & 255; xmb[i] = p & 127;
            xz[i] = xeb[i] == 0 ? 1 : 0;
        }
        // row max exponent with the phantom floor at 9
        int64_t E = Q_EFLOOR;
        for (uint32_t i = 0; i < d; i++)
            if (!xz[i] && xeb[i] > E) E = xeb[i];
        bool forged_max = tm && tm->mode == QT_MAXEXP && b == tm->b;
        if (forged_max) E += 1;
        int64_t fsel = 1; int sel_i = -1;
        if (!forged_max)
            for (uint32_t i = 0; i < d; i++)
                if (!xz[i] && xeb[i] == E) { sel_i = (int)i; fsel = 0; break; }
        int64_t edif = E - Q_EFLOOR;
        int64_t se = E - 8;
        int64_t scale = se << 23;
        if (tm && tm->mode == QT_SCALE && b == tm->b) scale += (int64_t)1 << 23;
        wt.db[R_E][b] = (gl_t)E; wt.db[R_EDIF][b] = (gl_t)edif;
        wt.db[R_FSEL][b] = (gl_t)fsel;
        wt.spat[b] = (gl_t)scale;
        wt.lidx[LUQ_EDIF][b] = (uint32_t)edif;
        if (b < L.B) wt.S[b] = (uint32_t)scale;
        for (uint32_t i = 0; i < d; i++) {
            size_t e = ((size_t)b << ld) | i;
            int64_t sh = xz[i] ? 0 : E - xeb[i];
            int64_t mag = sh < 256
                ? (sh < Q_DEXP_ROWS ? a.qe4m3[(size_t)((sh << 7) | xmb[i])] : 0)
                : -1;
            if (mag < 0) throw std::runtime_error("qnt: dexp out of range");
            if (tm && tm->mode == QT_MAG && b == tm->b && i == tm->i)
                mag += (mag < 127 ? 1 : -1);
            int64_t code = xs[i] * 128 + (xz[i] ? 0 : mag);
            wt.de[D_XS][e] = (gl_t)xs[i]; wt.de[D_XEB][e] = (gl_t)xeb[i];
            wt.de[D_XMB][e] = (gl_t)xmb[i]; wt.de[D_XZ][e] = (gl_t)xz[i];
            wt.de[D_XEI][e] = inv_or0((uint64_t)xeb[i]);
            wt.de[D_SH][e] = (gl_t)sh;
            wt.de[D_SEL][e] = (sel_i == (int)i) ? 1 : 0;
            wt.de[D_MAG][e] = (gl_t)mag;
            wt.lidx[LUQ_XEB][e] = (uint32_t)xeb[i];
            wt.lidx[LUQ_QE][e] = (uint32_t)((sh << 7) | xmb[i]);
            wt.cpat[e] = (gl_t)code;
            if (b < L.B) wt.C[(size_t)b * d + i] = (uint8_t)code;
        }
    }
    return wt;
}

// ---------------- operand commitments ----------------
struct Operands { Col X, CODES, SCALES; };
static inline Operands commit_operands(const Wit& wt, uint32_t R) {
    Operands ops;
    ops.X = commit_col_nc(wt.xpat, R);
    ops.CODES = commit_col_nc(wt.cpat, R);
    ops.SCALES = commit_col_nc(wt.spat, R);
    return ops;
}

// ---------------- constraint functions ----------------
// De zero-check: v = [E, X, CODES, de cols, Ebc]; lam[11]
static const int N_DE_C = 11;
static inline gl_t F_de(const gl_t* v, const gl_t* lam) {
    const gl_t* c = v + 3;
    gl_t one = 1ULL;
    gl_t Ebc = v[3 + NDE];
    gl_t nz = gl_sub(one, c[D_XZ]);
    auto boolc = [](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    gl_t r[N_DE_C];
    r[0]  = gl_sub(v[1], gl_add(gl_add(gl_mul(c[D_XS], 32768ULL),
                                       gl_mul(c[D_XEB], 128ULL)), c[D_XMB]));
    r[1]  = boolc(c[D_XS]);
    r[2]  = boolc(c[D_XZ]);
    r[3]  = gl_mul(c[D_XZ], c[D_XEB]);
    r[4]  = gl_sub(gl_mul(c[D_XEB], c[D_XEI]), nz);
    r[5]  = gl_mul(nz, gl_sub(gl_add(c[D_SH], c[D_XEB]), Ebc));
    r[6]  = gl_mul(c[D_XZ], c[D_SH]);
    r[7]  = boolc(c[D_SEL]);
    r[8]  = gl_mul(c[D_SEL], c[D_SH]);
    r[9]  = gl_mul(c[D_SEL], c[D_XZ]);
    r[10] = gl_sub(v[2], gl_add(gl_mul(c[D_XS], 128ULL), gl_mul(nz, c[D_MAG])));
    gl_t s = 0;
    for (int j = 0; j < N_DE_C; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}
// Db zero-check: v = [E, SCALES, db cols]; lam[4]
static const int N_DB_C = 4;
static inline gl_t F_db(const gl_t* v, const gl_t* lam) {
    const gl_t* c = v + 2;
    auto boolc = [](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    gl_t r[N_DB_C];
    r[0] = boolc(c[R_FSEL]);
    r[1] = gl_mul(c[R_FSEL], c[R_EDIF]);
    r[2] = gl_sub(c[R_E], gl_add((gl_t)Q_EFLOOR, c[R_EDIF]));
    r[3] = gl_sub(v[1], gl_mul(gl_sub(c[R_E], 8ULL), (gl_t)(1ULL << 23)));
    gl_t s = 0;
    for (int j = 0; j < N_DB_C; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}
// row binding: v = [EQb, SEL]; sum_i SEL = 1 - FSEL
static inline gl_t F_bind(const gl_t* v) { return gl_mul(v[0], v[1]); }

// ---------------- lookup descriptors ----------------
struct LuDef { const Table* tab; int dom; std::vector<int> cols; const char* label; };
static inline std::vector<LuDef> lu_defs(const Tables& T) {
    std::vector<LuDef> L(NLUQ);
    L[LUQ_XEB]  = {&T.R256, 0, {D_XEB}, "qntXEB"};
    L[LUQ_QE]   = {&T.QEXT, 0, {D_SH, D_XMB, D_MAG}, "qntQE"};
    L[LUQ_EDIF] = {&T.R256, 1, {R_EDIF}, "qntEDIF"};
    return L;
}

// ---------------- proof object ----------------
struct QntProof {
    uint32_t B = 0, ld = 0;
    p3fri::Hash rde[NDE], rdb[NDB];
    std::vector<p3lu::GroupProof> lug;   // standalone merged lookup groups
    std::vector<Msg5> mDe; gl_t yDe[NDE] = {}; gl_t yDeX = 0, yDeC = 0, yDeE = 0;
    std::vector<Msg5> mDb; gl_t yDb[NDB] = {}; gl_t yDbS = 0;
    gl_t yBF = 0;
    std::vector<Msg5> mBind; gl_t yBSEL = 0;
    p3zkc::Blind zbl[3];                       // zk: De, Db, bind blinds
    std::vector<p3bo::BatchProof> batches;
};

// ==================== prover ====================
static inline QntProof prove(fs::Transcript& tr, const Wit& wt, const Tables& T,
                             const Operands& ops, uint32_t R, uint32_t Q,
                             bool strict = true, p3lu::XCtx* xc = nullptr) {
    const Dims& dm = wt.dm;
    QntProof pf; pf.B = dm.B; pf.ld = dm.ld;

    uint32_t hdr[2] = {dm.B, dm.ld};
    tr.absorb("qnt-dims", hdr, sizeof hdr);
    tr.absorb("qnt-tab", T.R256.id.data(), 32);
    tr.absorb("qnt-tab", T.QEXT.id.data(), 32);
    tr.absorb("qnt-X", ops.X.root.data(), 32);
    tr.absorb("qnt-C", ops.CODES.root.data(), 32);
    tr.absorb("qnt-S", ops.SCALES.root.data(), 32);

    p3lu::XCtx xc_loc;
    p3lu::XCtx& XC = xc ? *xc : xc_loc;
    p3bo::PLedger& lg = XC.lg;
    std::deque<Col>& lucols = XC.keep;

    std::vector<Col>& CDe = XC.vec(NDE);
    std::vector<Col>& CDb = XC.vec(NDB);
    const bool zk = p3zkc::G.on;
    // zk: the row binding sum_i SEL = 1 - FSEL is CLAIM algebra -> the identity
    // must hold on the mask slice the (z||zex) claims touch (p3_zkc mechanism 1)
    std::vector<gl_t> mSEL, mFSEL;
    if (zk) {
        mSEL = p3zkc::fresh_mask(dm.le);
        mFSEL = p3zkc::fresh_mask(dm.lb);
        for (uint32_t b = 0; b < dm.Bpad; b++) {
            gl_t s = 0;
            for (uint32_t i = 0; i < dm.d; i++) s = gl_add(s, mSEL[((size_t)b << dm.ld) | i]);
            mFSEL[b] = gl_sub(1ULL, s);
        }
    }
    for (int c = 0; c < NDE; c++) {
        CDe[c] = commit_col_nc(wt.de[c], R, (zk && c == D_SEL) ? &mSEL : nullptr);
        pf.rde[c] = CDe[c].root;
    }
    for (int c = 0; c < NDB; c++) {
        CDb[c] = commit_col_nc(wt.db[c], R, (zk && c == R_FSEL) ? &mFSEL : nullptr);
        pf.rdb[c] = CDb[c].root;
    }
    for (int c = 0; c < NDE; c++) tr.absorb("qnt-ce", pf.rde[c].data(), 32);
    for (int c = 0; c < NDB; c++) tr.absorb("qnt-cb", pf.rdb[c].data(), 32);

    auto LD = lu_defs(T);
    for (int i = 0; i < NLUQ; i++) {
        std::vector<LuColSpec> spec;
        for (int cid : LD[i].cols)
            spec.push_back(LC(LD[i].dom == 0 ? &CDe[cid] : &CDb[cid]));
        p3lu::defer_v(XC, std::move(spec), wt.lidx[i], *LD[i].tab, LD[i].label);
    }

    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, dm.le);
    gl_t lamE = chal(tr), lamEv[N_DE_C]; lamEv[0] = 1;
    for (int j = 1; j < N_DE_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(3 + NDE + 1);
        cols.push_back(beq(p3zkc::zpt(zE)));
        cols.push_back(ops.X.v); cols.push_back(ops.CODES.v);
        for (int c = 0; c < NDE; c++) cols.push_back(CDe[c].v);
        cols.push_back(p3zkc::bc_aug(CDb[R_E].v, dm.lb, dm.le, dm.Ne,
                                     [&](size_t e) { return e >> dm.ld; }));
        CFn F = [&](const gl_t* v) { return F_de(v, lamEv); };
        std::vector<gl_t> rE = p3hwl::sc5z(tr, "qnt-scE", dm.le, std::move(cols), F, pf.mDe,
                                           0, R, lg, lucols, pf.zbl[0]);
        pf.yDeX = claimc(tr, lg, ops.X, rE);
        pf.yDeC = claimc(tr, lg, ops.CODES, rE);
        for (int c = 0; c < NDE; c++) pf.yDe[c] = claimc(tr, lg, CDe[c], rE);
        std::vector<gl_t> rb(rE.begin() + dm.ld, rE.begin() + dm.le);
        pf.yDeE = claimc(tr, lg, CDb[R_E], p3zkc::expt(rb, rE, dm.le));
    }

    // -- Db zero-check --
    std::vector<gl_t> zB = chal_vec(tr, dm.lb);
    gl_t lamB = chal(tr), lamBv[N_DB_C]; lamBv[0] = 1;
    for (int j = 1; j < N_DB_C; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(2 + NDB);
        cols.push_back(beq(p3zkc::zpt(zB)));
        cols.push_back(ops.SCALES.v);
        for (int c = 0; c < NDB; c++) cols.push_back(CDb[c].v);
        CFn F = [&](const gl_t* v) { return F_db(v, lamBv); };
        std::vector<gl_t> rB = p3hwl::sc5z(tr, "qnt-scB", dm.lb, std::move(cols), F, pf.mDb,
                                           0, R, lg, lucols, pf.zbl[1]);
        pf.yDbS = claimc(tr, lg, ops.SCALES, rB);
        for (int c = 0; c < NDB; c++) pf.yDb[c] = claimc(tr, lg, CDb[c], rB);
    }

    // -- row binding: sum_i SEL = 1 - FSEL (zk: fresh ex challenge + slice-1
    //    linked masks; weight built from the target point zero-extended) --
    std::vector<gl_t> z2 = chal_vec(tr, dm.lb);
    gl_t zexq = zk ? chal(tr) : 0;
    pf.yBF = claimc(tr, lg, CDb[R_FSEL], p3zkc::xpt(z2, zexq));
    {
        uint32_t e_e = p3zkc::e_of(dm.le);
        std::vector<gl_t> ptb = z2;
        if (zk) { ptb.push_back(zexq); ptb.resize(dm.lb + e_e, 0); }
        std::vector<gl_t> eqb = beq(ptb);
        size_t NeA = CDe[D_SEL].v.size();
        std::vector<gl_t> EQb(NeA);
        for (size_t q = 0; q < NeA; q++) {
            size_t ex = q >> dm.le, e = q & (dm.Ne - 1);
            EQb[q] = eqb[(ex << dm.lb) | (e >> dm.ld)];
        }
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(std::move(EQb));
        cols.push_back(CDe[D_SEL].v);
        CFn F = [&](const gl_t* v) { return F_bind(v); };
        gl_t base0 = gl_sub(1ULL, pf.yBF);
        std::vector<gl_t> rS = p3hwl::sc5z(tr, "qnt-scS", dm.le, std::move(cols), F, pf.mBind,
                                           base0, R, lg, lucols, pf.zbl[2]);
        pf.yBSEL = claimc(tr, lg, CDe[D_SEL], rS);
    }

    // -- batched openings (deferred to the caller under an external ledger) --
    if (!xc) {
        pf.lug = p3lu::lu_flush(tr, XC, R, Q, strict);
        for (size_t i = 0; i < lg.cls.size(); i++)
            pf.batches.push_back(p3bo::prove_class(tr, lg.cls[i], R, Q,
                                                   "qnt-bo" + std::to_string(i)));
    }
    return pf;
}

// ==================== verifier ====================
// rX/rCODES/rSCALES, B, ld and Q/R are PUBLIC caller inputs.
static inline bool verify(fs::Transcript& tr, const Tables& T, const QntProof& pf,
                          const p3fri::Hash& rX, const p3fri::Hash& rCODES,
                          const p3fri::Hash& rSCALES, uint32_t B, uint32_t ld,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr,
                          p3lu::VCtx* xv = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.B != B || pf.ld != ld) return fail("dims mismatch");
    Dims dm = make_dims(B, ld);
    p3lu::VCtx vc_loc;
    p3lu::VCtx& VC = xv ? *xv : vc_loc;
    p3bo::VLedger& vlg = VC.vlg;

    uint32_t hdr[2] = {B, ld};
    tr.absorb("qnt-dims", hdr, sizeof hdr);
    tr.absorb("qnt-tab", T.R256.id.data(), 32);
    tr.absorb("qnt-tab", T.QEXT.id.data(), 32);
    tr.absorb("qnt-X", rX.data(), 32);
    tr.absorb("qnt-C", rCODES.data(), 32);
    tr.absorb("qnt-S", rSCALES.data(), 32);
    for (int c = 0; c < NDE; c++) tr.absorb("qnt-ce", pf.rde[c].data(), 32);
    for (int c = 0; c < NDB; c++) tr.absorb("qnt-cb", pf.rdb[c].data(), 32);

    auto LD = lu_defs(T);
    for (int i = 0; i < NLUQ; i++) {
        uint32_t explog = LD[i].dom == 0 ? dm.le : dm.lb;
        std::vector<const p3fri::Hash*> roots;
        for (int cid : LD[i].cols)
            roots.push_back(LD[i].dom == 0 ? &pf.rde[cid] : &pf.rdb[cid]);
        p3lu::vdefer_v(VC, std::move(roots), *LD[i].tab, explog, LD[i].label);
    }

    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, dm.le);
    gl_t lamE = chal(tr), lamEv[N_DE_C]; lamEv[0] = 1;
    for (int j = 1; j < N_DE_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.zbl[0]);
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.mDe, p3zkc::vfull(dm.le), gl_mul(rho, pf.zbl[0].H),
                        tr, "qnt-scE", rE, claim)) return fail("De sumcheck");
        p3hwl::sc5vz_claims(tr, vlg, pf.zbl[0], rE);
        gl_t v[3 + NDE + 1]; v[0] = p3bf::eq_point(rE, p3zkc::zpt(zE));
        v[1] = claimv(tr, vlg, rX, rE, pf.yDeX);
        v[2] = claimv(tr, vlg, rCODES, rE, pf.yDeC);
        for (int c = 0; c < NDE; c++) v[3 + c] = claimv(tr, vlg, pf.rde[c], rE, pf.yDe[c]);
        std::vector<gl_t> rb(rE.begin() + ld, rE.begin() + dm.le);
        v[3 + NDE] = claimv(tr, vlg, pf.rdb[R_E], p3zkc::expt(rb, rE, dm.le), pf.yDeE);
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
                        tr, "qnt-scB", rB, claim)) return fail("Db sumcheck");
        p3hwl::sc5vz_claims(tr, vlg, pf.zbl[1], rB);
        gl_t v[2 + NDB]; v[0] = p3bf::eq_point(rB, p3zkc::zpt(zB));
        v[1] = claimv(tr, vlg, rSCALES, rB, pf.yDbS);
        for (int c = 0; c < NDB; c++) v[2 + c] = claimv(tr, vlg, pf.rdb[c], rB, pf.yDb[c]);
        gl_t end = gl_add(F_db(v, lamBv), p3hwl::sc5_blindterm(pf.zbl[1], rho, v[0]));
        if (end != claim) return fail("Db terminal");
    }

    // -- row binding --
    std::vector<gl_t> z2 = chal_vec(tr, dm.lb);
    {
        const bool zk = p3zkc::G.on;
        gl_t zexq = zk ? p3lu::chal(tr) : 0;
        gl_t yF = claimv(tr, vlg, pf.rdb[R_FSEL], p3zkc::xpt(z2, zexq), pf.yBF);
        gl_t claim0 = gl_sub(1ULL, yF);
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.zbl[2]);
        claim0 = gl_add(claim0, gl_mul(rho, pf.zbl[2].H));
        std::vector<gl_t> rS; gl_t claim;
        if (!sc5_verify(pf.mBind, p3zkc::vfull(dm.le), claim0, tr, "qnt-scS", rS, claim))
            return fail("bind sumcheck");
        p3hwl::sc5vz_claims(tr, vlg, pf.zbl[2], rS);
        gl_t ySEL = claimv(tr, vlg, pf.rde[D_SEL], rS, pf.yBSEL);
        std::vector<gl_t> rSb(rS.begin() + ld, rS.begin() + dm.le);
        rSb.insert(rSb.end(), rS.begin() + dm.le, rS.end());
        std::vector<gl_t> ptb = z2;
        if (zk) { ptb.push_back(zexq); ptb.resize(dm.lb + p3zkc::e_of(dm.le), 0); }
        gl_t w = p3bf::eq_point(rSb, ptb);
        gl_t end = gl_add(gl_mul(w, ySEL), p3hwl::sc5_blindterm(pf.zbl[2], rho, w));
        if (end != claim) return fail("bind terminal");
    }

    // -- batched openings (caller-run under an external ledger) --
    if (!xv) {
        if (!p3lu::lu_verify_flush(tr, VC, pf.lug, Q_pub, R_pub, why)) return false;
        if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
        for (size_t i = 0; i < vlg.cls.size(); i++)
            if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                    "qnt-bo" + std::to_string(i), why)) return false;
    } else if (!pf.batches.empty() || !pf.lug.empty()) return fail("unexpected batches");

    if (why) *why = "ok";
    return true;
}

// ---------------- proof size (serialized-payload bytes) ----------------
static inline size_t proof_size(const QntProof& pf) {
    size_t s = 8 + (NDE + NDB) * 32;
    for (auto& g : pf.lug) s += p3lu::sz_group(g);
    auto msgs = [&](const std::vector<Msg5>& m) { return m.size() * 40; };
    s += msgs(pf.mDe) + msgs(pf.mDb) + msgs(pf.mBind);
    s += 8 * (NDE + 3 + NDB + 1 + 2);
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3qnt
