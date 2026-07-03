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
    p3lu::LookupProof lu[NLUQ];
    std::vector<Msg5> mDe; gl_t yDe[NDE] = {}; gl_t yDeX = 0, yDeC = 0, yDeE = 0;
    std::vector<Msg5> mDb; gl_t yDb[NDB] = {}; gl_t yDbS = 0;
    gl_t yBF = 0;
    std::vector<Msg5> mBind; gl_t yBSEL = 0;
    std::vector<p3bo::BatchProof> batches;
};

// ==================== prover ====================
static inline QntProof prove(fs::Transcript& tr, const Wit& wt, const Tables& T,
                             const Operands& ops, uint32_t R, uint32_t Q,
                             bool strict = true) {
    const Dims& dm = wt.dm;
    QntProof pf; pf.B = dm.B; pf.ld = dm.ld;

    uint32_t hdr[2] = {dm.B, dm.ld};
    tr.absorb("qnt-dims", hdr, sizeof hdr);
    tr.absorb("qnt-tab", T.R256.id.data(), 32);
    tr.absorb("qnt-tab", T.QEXT.id.data(), 32);
    tr.absorb("qnt-X", ops.X.root.data(), 32);
    tr.absorb("qnt-C", ops.CODES.root.data(), 32);
    tr.absorb("qnt-S", ops.SCALES.root.data(), 32);

    p3bo::PLedger lg;
    std::deque<Col> lucols;

    std::vector<Col> CDe(NDE), CDb(NDB);
    for (int c = 0; c < NDE; c++) { CDe[c] = commit_col_nc(wt.de[c], R); pf.rde[c] = CDe[c].root; }
    for (int c = 0; c < NDB; c++) { CDb[c] = commit_col_nc(wt.db[c], R); pf.rdb[c] = CDb[c].root; }
    for (int c = 0; c < NDE; c++) tr.absorb("qnt-ce", pf.rde[c].data(), 32);
    for (int c = 0; c < NDB; c++) tr.absorb("qnt-cb", pf.rdb[c].data(), 32);

    auto LD = lu_defs(T);
    for (int i = 0; i < NLUQ; i++) {
        std::vector<LuColSpec> spec;
        for (int cid : LD[i].cols)
            spec.push_back(LC(LD[i].dom == 0 ? &CDe[cid] : &CDb[cid]));
        pf.lu[i] = p3lu::prove_v(tr, spec, wt.lidx[i], *LD[i].tab, R, Q, LD[i].label,
                                 true, strict, nullptr, &lg, &lucols);
    }

    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, dm.le);
    gl_t lamE = chal(tr), lamEv[N_DE_C]; lamEv[0] = 1;
    for (int j = 1; j < N_DE_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(3 + NDE + 1);
        cols.push_back(beq(zE));
        cols.push_back(wt.xpat); cols.push_back(wt.cpat);
        for (int c = 0; c < NDE; c++) cols.push_back(wt.de[c]);
        std::vector<gl_t> Ebc(dm.Ne);
        for (size_t e = 0; e < dm.Ne; e++) Ebc[e] = wt.db[R_E][e >> dm.ld];
        cols.push_back(std::move(Ebc));
        CFn F = [&](const gl_t* v) { return F_de(v, lamEv); };
        std::vector<gl_t> rE = sc5_prove(tr, "qnt-scE", std::move(cols), F, pf.mDe);
        pf.yDeX = claimc(tr, lg, ops.X, rE);
        pf.yDeC = claimc(tr, lg, ops.CODES, rE);
        for (int c = 0; c < NDE; c++) pf.yDe[c] = claimc(tr, lg, CDe[c], rE);
        std::vector<gl_t> rb(rE.begin() + dm.ld, rE.end());
        pf.yDeE = claimc(tr, lg, CDb[R_E], rb);
    }

    // -- Db zero-check --
    std::vector<gl_t> zB = chal_vec(tr, dm.lb);
    gl_t lamB = chal(tr), lamBv[N_DB_C]; lamBv[0] = 1;
    for (int j = 1; j < N_DB_C; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(2 + NDB);
        cols.push_back(beq(zB));
        cols.push_back(wt.spat);
        for (int c = 0; c < NDB; c++) cols.push_back(wt.db[c]);
        CFn F = [&](const gl_t* v) { return F_db(v, lamBv); };
        std::vector<gl_t> rB = sc5_prove(tr, "qnt-scB", std::move(cols), F, pf.mDb);
        pf.yDbS = claimc(tr, lg, ops.SCALES, rB);
        for (int c = 0; c < NDB; c++) pf.yDb[c] = claimc(tr, lg, CDb[c], rB);
    }

    // -- row binding: sum_i SEL = 1 - FSEL --
    std::vector<gl_t> z2 = chal_vec(tr, dm.lb);
    pf.yBF = claimc(tr, lg, CDb[R_FSEL], z2);
    {
        std::vector<gl_t> eqb = beq(z2);
        std::vector<gl_t> EQb(dm.Ne);
        for (size_t e = 0; e < dm.Ne; e++) EQb[e] = eqb[e >> dm.ld];
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(std::move(EQb));
        cols.push_back(wt.de[D_SEL]);
        CFn F = [&](const gl_t* v) { return F_bind(v); };
        std::vector<gl_t> rS = sc5_prove(tr, "qnt-scS", std::move(cols), F, pf.mBind);
        pf.yBSEL = claimc(tr, lg, CDe[D_SEL], rS);
    }

    // -- batched openings --
    for (size_t i = 0; i < lg.cls.size(); i++)
        pf.batches.push_back(p3bo::prove_class(tr, lg.cls[i], R, Q,
                                               "qnt-bo" + std::to_string(i)));
    return pf;
}

// ==================== verifier ====================
// rX/rCODES/rSCALES, B, ld and Q/R are PUBLIC caller inputs.
static inline bool verify(fs::Transcript& tr, const Tables& T, const QntProof& pf,
                          const p3fri::Hash& rX, const p3fri::Hash& rCODES,
                          const p3fri::Hash& rSCALES, uint32_t B, uint32_t ld,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.B != B || pf.ld != ld) return fail("dims mismatch");
    Dims dm = make_dims(B, ld);
    p3bo::VLedger vlg;

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
        if (pf.lu[i].n != explog) return fail("lookup domain");
        std::vector<const p3fri::Hash*> roots;
        for (int cid : LD[i].cols)
            roots.push_back(LD[i].dom == 0 ? &pf.rde[cid] : &pf.rdb[cid]);
        if (!p3lu::verify_v(tr, roots, *LD[i].tab, pf.lu[i], Q_pub, R_pub, LD[i].label,
                            why, nullptr, nullptr, nullptr, &vlg)) return false;
    }

    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, dm.le);
    gl_t lamE = chal(tr), lamEv[N_DE_C]; lamEv[0] = 1;
    for (int j = 1; j < N_DE_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.mDe, dm.le, 0, tr, "qnt-scE", rE, claim)) return fail("De sumcheck");
        gl_t v[3 + NDE + 1]; v[0] = p3bf::eq_point(rE, zE);
        v[1] = claimv(tr, vlg, rX, rE, pf.yDeX);
        v[2] = claimv(tr, vlg, rCODES, rE, pf.yDeC);
        for (int c = 0; c < NDE; c++) v[3 + c] = claimv(tr, vlg, pf.rde[c], rE, pf.yDe[c]);
        std::vector<gl_t> rb(rE.begin() + ld, rE.end());
        v[3 + NDE] = claimv(tr, vlg, pf.rdb[R_E], rb, pf.yDeE);
        if (F_de(v, lamEv) != claim) return fail("De terminal");
    }

    // -- Db zero-check --
    std::vector<gl_t> zB = chal_vec(tr, dm.lb);
    gl_t lamB = chal(tr), lamBv[N_DB_C]; lamBv[0] = 1;
    for (int j = 1; j < N_DB_C; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        std::vector<gl_t> rB; gl_t claim;
        if (!sc5_verify(pf.mDb, dm.lb, 0, tr, "qnt-scB", rB, claim)) return fail("Db sumcheck");
        gl_t v[2 + NDB]; v[0] = p3bf::eq_point(rB, zB);
        v[1] = claimv(tr, vlg, rSCALES, rB, pf.yDbS);
        for (int c = 0; c < NDB; c++) v[2 + c] = claimv(tr, vlg, pf.rdb[c], rB, pf.yDb[c]);
        if (F_db(v, lamBv) != claim) return fail("Db terminal");
    }

    // -- row binding --
    std::vector<gl_t> z2 = chal_vec(tr, dm.lb);
    {
        gl_t yF = claimv(tr, vlg, pf.rdb[R_FSEL], z2, pf.yBF);
        gl_t claim0 = gl_sub(1ULL, yF);
        std::vector<gl_t> rS; gl_t claim;
        if (!sc5_verify(pf.mBind, dm.le, claim0, tr, "qnt-scS", rS, claim))
            return fail("bind sumcheck");
        gl_t ySEL = claimv(tr, vlg, pf.rde[D_SEL], rS, pf.yBSEL);
        std::vector<gl_t> rSb(rS.begin() + ld, rS.end());
        if (gl_mul(p3bf::eq_point(rSb, z2), ySEL) != claim) return fail("bind terminal");
    }

    // -- batched openings --
    if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
    for (size_t i = 0; i < vlg.cls.size(); i++)
        if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                "qnt-bo" + std::to_string(i), why)) return false;

    if (why) *why = "ok";
    return true;
}

// ---------------- proof size (serialized-payload bytes) ----------------
static inline size_t proof_size(const QntProof& pf) {
    size_t s = 8 + (NDE + NDB) * 32;
    for (int i = 0; i < NLUQ; i++) s += p3hwl::sz_lu(pf.lu[i]);
    auto msgs = [&](const std::vector<Msg5>& m) { return m.size() * 40; };
    s += msgs(pf.mDe) + msgs(pf.mDb) + msgs(pf.mBind);
    s += 8 * (NDE + 3 + NDB + 1 + 2);
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3qnt
