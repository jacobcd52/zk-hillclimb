// SwiGLU gadget: sound (non-ZK) prover+verifier that a committed bf16 output
// M equals the CANONICAL SwiGLU combine (transformer_ref.py):
//      m_j = bf16_mul( SILU[gate_j], up_j )        elementwise, n = 2^ln
// bitwise, where SILU is the pinned 65536-entry bf16->bf16 table and the
// multiply is the canonical RNE mantissa-product (MUL7 table + linear
// exponent + REXP range).
//
// This is the "1-in/1-out nonlinearity + bf16 multiply" template: the SILU
// lookup keys DIRECTLY on the committed operand column (gate patterns), so a
// unary nonlinearity costs exactly one logUp instance; the multiply reuses the
// same decomposition/MUL7/REXP vocabulary as the RMSNorm output stage.
//
// Supported domain (proof REJECTS otherwise -- sound, not complete): the
// product exponent stays in [1,254].  Zeros, signed zeros, subnormal inputs
// (canonical FTZ) and saturated silu outputs are all in-domain.
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

namespace p3swg {

using p3lu::Col; using p3lu::Table; using p3lu::make_table; using p3lu::commit_col_nc;
using p3lu::chal; using p3lu::chal_vec; using p3lu::ilog2;
using p3lu::LuColSpec; using p3lu::LC; using p3lu::LV;
using p3hwl::Msg5; using p3hwl::CFn; using p3hwl::sc5_prove; using p3hwl::sc5_verify;
using p3hwl::claimc; using p3hwl::claimv; using p3hwl::beq;
using p3rms::Art;

// ---------------- tables ----------------
struct Tables {
    Table R128, R256, MUL7, REXP;              // same construction as p3rms
    Table SILU;                                // (pattern, silu-pattern) 65536 rows
};
static inline Tables build_tables(const Art& a) {
    Tables T;
    auto range = [](uint32_t n) { std::vector<gl_t> v(n);
        for (uint32_t j = 0; j < n; j++) v[j] = j; return make_table({v}); };
    T.R128 = range(128); T.R256 = range(256);
    { std::vector<gl_t> ma(16384), mb(16384), mo(16384), ei(16384);
      for (uint32_t j = 0; j < 16384; j++) {
          ma[j] = 128 + (j >> 7); mb[j] = 128 + (j & 127);
          mo[j] = 128 + a.mul_mo[j]; ei[j] = a.mul_einc[j];
      }
      T.MUL7 = make_table({ma, mb, mo, ei}); }
    { std::vector<gl_t> v(256);
      for (uint32_t j = 0; j < 256; j++) v[j] = (j >= 1 && j <= 254) ? j : 1;
      T.REXP = make_table({v}); }
    { std::vector<gl_t> in(65536), out(65536);
      for (uint32_t j = 0; j < 65536; j++) { in[j] = j; out[j] = a.silu_tab[j]; }
      T.SILU = make_table({in, out}); }
    return T;
}

// ---------------- columns / lookups ----------------
enum {  // element-domain witness columns
    S_SG = 0, S_SGS, S_SGEB, S_SGMB, S_SGZ, S_SGEI,
    S_US, S_UEB, S_UMB, S_UZ, S_UEI,
    S_MO, S_EI, S_EO, NDS };
enum {  // lookup instances, fixed order; SILU keys on the GATE operand itself
    LUS_SILU = 0, LUS_SGEB, LUS_SGMB, LUS_UEB, LUS_UMB, LUS_MUL, LUS_EO, NLUS };

struct Golden {
    uint32_t n = 0;
    std::vector<uint16_t> gate, up, m;
};
static inline bool load_goldens(const char* path, std::vector<Golden>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[2];
    if (fread(hdr, 8, 2, f) != 2 || hdr[0] != 0x53574747) { fclose(f); return false; }
    out.resize(hdr[1]);
    for (auto& G : out) {
        int64_t n;
        if (fread(&n, 8, 1, f) != 1) { fclose(f); return false; }
        G.n = (uint32_t)n;
        G.gate.resize(n); G.up.resize(n); G.m.resize(n);
        if (fread(G.gate.data(), 2, n, f) != (size_t)n ||
            fread(G.up.data(), 2, n, f) != (size_t)n ||
            fread(G.m.data(), 2, n, f) != (size_t)n) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

// ---------------- witness ----------------
enum { ST_NONE = 0,
       ST_SILU,      // silu output pattern +1 (not a table row), honest downstream
       ST_MULUP };   // product mantissa MO+-1 (not a table row), honest downstream
struct STamper { int mode = ST_NONE; uint32_t j = 0; };

struct Wit {
    uint32_t n = 0, ln = 0;
    std::vector<gl_t> gpat, upat, mpat;       // operand pattern columns
    std::vector<uint16_t> M;                  // computed output patterns
    std::vector<gl_t> ws[NDS];
    std::vector<gl_t> sgmf, umf;              // virtual MUL7 key columns
    std::vector<uint32_t> lidx[NLUS];
};
static inline gl_t inv_or0(uint64_t x) { return x ? gl_inv((gl_t)x) : 0; }

static inline Wit gen_witness(const Golden& L, const Art& a, const STamper* tm = nullptr) {
    Wit wt;
    wt.n = L.n; wt.ln = ilog2(L.n);
    if ((1u << wt.ln) != L.n) throw std::runtime_error("swg: n must be pow2");
    wt.gpat.assign(L.n, 0); wt.upat.assign(L.n, 0); wt.mpat.assign(L.n, 0);
    wt.M.assign(L.n, 0);
    for (int c = 0; c < NDS; c++) wt.ws[c].assign(L.n, 0);
    wt.sgmf.assign(L.n, 0); wt.umf.assign(L.n, 0);
    for (int i = 0; i < NLUS; i++) wt.lidx[i].assign(L.n, 0);

    for (uint32_t j = 0; j < L.n; j++) {
        uint32_t gp = L.gate[j], up = L.up[j];
        wt.gpat[j] = gp; wt.upat[j] = up;
        int64_t sg = a.silu_tab[gp];
        if (tm && tm->mode == ST_SILU && j == tm->j) sg += 1;
        int64_t sgs = (sg >> 15) & 1, sgeb = (sg >> 7) & 255, sgmb = sg & 127;
        int64_t sgz = sgeb == 0 ? 1 : 0;
        int64_t us = (up >> 15) & 1, ueb = (up >> 7) & 255, umb = up & 127;
        int64_t uz = ueb == 0 ? 1 : 0;
        uint32_t mj = (uint32_t)((sgmb << 7) | umb);
        int64_t mo = 128 + a.mul_mo[mj], ei = a.mul_einc[mj];
        if (tm && tm->mode == ST_MULUP && j == tm->j) mo += (mo < 255 ? 1 : -1);
        int64_t z2 = (sgz || uz) ? 1 : 0;
        int64_t eo = z2 ? 1 : sgeb + ueb - 127 + ei;
        if (eo < 1 || eo > 254) throw std::runtime_error("swg: mul exp domain");
        int64_t sm = sgs ^ us;
        int64_t m = (sm << 15) | (z2 ? 0 : ((eo << 7) | (mo - 128)));
        wt.ws[S_SG][j] = (gl_t)sg;
        wt.ws[S_SGS][j] = (gl_t)sgs; wt.ws[S_SGEB][j] = (gl_t)sgeb;
        wt.ws[S_SGMB][j] = (gl_t)sgmb; wt.ws[S_SGZ][j] = (gl_t)sgz;
        wt.ws[S_SGEI][j] = inv_or0((uint64_t)sgeb);
        wt.ws[S_US][j] = (gl_t)us; wt.ws[S_UEB][j] = (gl_t)ueb;
        wt.ws[S_UMB][j] = (gl_t)umb; wt.ws[S_UZ][j] = (gl_t)uz;
        wt.ws[S_UEI][j] = inv_or0((uint64_t)ueb);
        wt.ws[S_MO][j] = (gl_t)mo; wt.ws[S_EI][j] = (gl_t)ei;
        wt.ws[S_EO][j] = (gl_t)eo;
        wt.sgmf[j] = (gl_t)(128 + sgmb); wt.umf[j] = (gl_t)(128 + umb);
        wt.lidx[LUS_SILU][j] = gp;
        wt.lidx[LUS_SGEB][j] = (uint32_t)sgeb; wt.lidx[LUS_SGMB][j] = (uint32_t)sgmb;
        wt.lidx[LUS_UEB][j] = (uint32_t)ueb; wt.lidx[LUS_UMB][j] = (uint32_t)umb;
        wt.lidx[LUS_MUL][j] = mj;
        wt.lidx[LUS_EO][j] = (uint32_t)eo;
        wt.mpat[j] = (gl_t)m;
        wt.M[j] = (uint16_t)m;
    }
    return wt;
}

struct Operands { Col GATE, UP, M; };
static inline Operands commit_operands(const Wit& wt, uint32_t R) {
    Operands ops;
    ops.GATE = commit_col_nc(wt.gpat, R);
    ops.UP = commit_col_nc(wt.upat, R);
    ops.M = commit_col_nc(wt.mpat, R);
    return ops;
}

// ---------------- constraints ----------------
// v = [E, UP, M, ws cols]; lam[13]
static const int N_SW_C = 13;
static inline gl_t F_sw(const gl_t* v, const gl_t* lam) {
    const gl_t* c = v + 3;
    gl_t one = 1ULL;
    auto boolc = [](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    gl_t z2 = gl_sub(gl_add(c[S_SGZ], c[S_UZ]), gl_mul(c[S_SGZ], c[S_UZ]));
    gl_t nz2 = gl_sub(one, z2);
    gl_t sm = gl_sub(gl_add(c[S_SGS], c[S_US]),
                     gl_mul(gl_add(c[S_SGS], c[S_SGS]), c[S_US]));
    gl_t r[N_SW_C];
    r[0]  = gl_sub(c[S_SG], gl_add(gl_add(gl_mul(c[S_SGS], 32768ULL),
                                          gl_mul(c[S_SGEB], 128ULL)), c[S_SGMB]));
    r[1]  = boolc(c[S_SGS]);
    r[2]  = boolc(c[S_SGZ]);
    r[3]  = gl_mul(c[S_SGZ], c[S_SGEB]);
    r[4]  = gl_sub(gl_mul(c[S_SGEB], c[S_SGEI]), gl_sub(one, c[S_SGZ]));
    r[5]  = gl_sub(v[1], gl_add(gl_add(gl_mul(c[S_US], 32768ULL),
                                       gl_mul(c[S_UEB], 128ULL)), c[S_UMB]));
    r[6]  = boolc(c[S_US]);
    r[7]  = boolc(c[S_UZ]);
    r[8]  = gl_mul(c[S_UZ], c[S_UEB]);
    r[9]  = gl_sub(gl_mul(c[S_UEB], c[S_UEI]), gl_sub(one, c[S_UZ]));
    r[10] = gl_mul(z2, gl_sub(c[S_EO], one));
    r[11] = gl_mul(nz2, gl_sub(c[S_EO],
                    gl_sub(gl_add(gl_add(c[S_SGEB], c[S_UEB]), c[S_EI]), 127ULL)));
    r[12] = gl_sub(v[2], gl_add(gl_mul(sm, 32768ULL),
                    gl_mul(nz2, gl_sub(gl_add(gl_mul(c[S_EO], 128ULL), c[S_MO]),
                                       128ULL))));
    gl_t s = 0;
    for (int j = 0; j < N_SW_C; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}

// ---------------- proof object ----------------
struct SwgProof {
    uint32_t n = 0;
    p3fri::Hash rws[NDS];
    p3lu::LookupProof lu[NLUS];
    gl_t yMsgmb = 0, yMumb = 0;                // virtual-key bindings
    std::vector<Msg5> mE; gl_t yE[NDS] = {}; gl_t yUP = 0, yM = 0;
    std::vector<p3bo::BatchProof> batches;
};

// lookup descriptors: dom col >= 0 witness; -1 sgmf, -2 umf (virtual);
// -3 = the GATE operand column
struct LuDef { const Table* tab; std::vector<int> cols; const char* label; };
static inline std::vector<LuDef> lu_defs(const Tables& T) {
    std::vector<LuDef> L(NLUS);
    L[LUS_SILU] = {&T.SILU, {-3, S_SG}, "swgSILU"};
    L[LUS_SGEB] = {&T.R256, {S_SGEB}, "swgSGEB"};
    L[LUS_SGMB] = {&T.R128, {S_SGMB}, "swgSGMB"};
    L[LUS_UEB]  = {&T.R256, {S_UEB}, "swgUEB"};
    L[LUS_UMB]  = {&T.R128, {S_UMB}, "swgUMB"};
    L[LUS_MUL]  = {&T.MUL7, {-1, -2, S_MO, S_EI}, "swgMUL"};
    L[LUS_EO]   = {&T.REXP, {S_EO}, "swgEO"};
    return L;
}

// ==================== prover ====================
static inline SwgProof prove(fs::Transcript& tr, const Wit& wt, const Tables& T,
                             const Operands& ops, uint32_t R, uint32_t Q,
                             bool strict = true) {
    SwgProof pf; pf.n = wt.n;
    uint32_t hdr[1] = {wt.n};
    tr.absorb("swg-dims", hdr, sizeof hdr);
    const Table* tabs[5] = {&T.R128, &T.R256, &T.MUL7, &T.REXP, &T.SILU};
    for (auto* t : tabs) tr.absorb("swg-tab", t->id.data(), 32);
    tr.absorb("swg-G", ops.GATE.root.data(), 32);
    tr.absorb("swg-U", ops.UP.root.data(), 32);
    tr.absorb("swg-M", ops.M.root.data(), 32);

    p3bo::PLedger lg;
    std::deque<Col> lucols;

    std::vector<Col> C(NDS);
    for (int c = 0; c < NDS; c++) { C[c] = commit_col_nc(wt.ws[c], R); pf.rws[c] = C[c].root; }
    for (int c = 0; c < NDS; c++) tr.absorb("swg-cw", pf.rws[c].data(), 32);

    auto LD = lu_defs(T);
    for (int i = 0; i < NLUS; i++) {
        std::vector<LuColSpec> spec;
        for (int cid : LD[i].cols) {
            if (cid == -1) spec.push_back(LV(&wt.sgmf));
            else if (cid == -2) spec.push_back(LV(&wt.umf));
            else if (cid == -3) spec.push_back(LC(&ops.GATE));
            else spec.push_back(LC(&C[cid]));
        }
        std::vector<gl_t> rA;
        pf.lu[i] = p3lu::prove_v(tr, spec, wt.lidx[i], *LD[i].tab, R, Q, LD[i].label,
                                 true, strict, i == LUS_MUL ? &rA : nullptr, &lg, &lucols);
        if (i == LUS_MUL) {
            pf.yMsgmb = claimc(tr, lg, C[S_SGMB], rA);
            pf.yMumb = claimc(tr, lg, C[S_UMB], rA);
        }
    }

    std::vector<gl_t> zE = chal_vec(tr, wt.ln);
    gl_t lamE = chal(tr), lamEv[N_SW_C]; lamEv[0] = 1;
    for (int j = 1; j < N_SW_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(3 + NDS);
        cols.push_back(beq(zE));
        cols.push_back(wt.upat); cols.push_back(wt.mpat);
        for (int c = 0; c < NDS; c++) cols.push_back(wt.ws[c]);
        CFn F = [&](const gl_t* v) { return F_sw(v, lamEv); };
        std::vector<gl_t> rE = sc5_prove(tr, "swg-scE", std::move(cols), F, pf.mE);
        pf.yUP = claimc(tr, lg, ops.UP, rE);
        pf.yM = claimc(tr, lg, ops.M, rE);
        for (int c = 0; c < NDS; c++) pf.yE[c] = claimc(tr, lg, C[c], rE);
    }

    for (size_t i = 0; i < lg.cls.size(); i++)
        pf.batches.push_back(p3bo::prove_class(tr, lg.cls[i], R, Q,
                                               "swg-bo" + std::to_string(i)));
    return pf;
}

// ==================== verifier ====================
static inline bool verify(fs::Transcript& tr, const Tables& T, const SwgProof& pf,
                          const p3fri::Hash& rGATE, const p3fri::Hash& rUP,
                          const p3fri::Hash& rM, uint32_t n,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.n != n) return fail("dims mismatch");
    uint32_t ln = ilog2(n);
    if ((1u << ln) != n) return fail("n must be pow2");
    p3bo::VLedger vlg;

    uint32_t hdr[1] = {n};
    tr.absorb("swg-dims", hdr, sizeof hdr);
    const Table* tabs[5] = {&T.R128, &T.R256, &T.MUL7, &T.REXP, &T.SILU};
    for (auto* t : tabs) tr.absorb("swg-tab", t->id.data(), 32);
    tr.absorb("swg-G", rGATE.data(), 32);
    tr.absorb("swg-U", rUP.data(), 32);
    tr.absorb("swg-M", rM.data(), 32);
    for (int c = 0; c < NDS; c++) tr.absorb("swg-cw", pf.rws[c].data(), 32);

    auto LD = lu_defs(T);
    for (int i = 0; i < NLUS; i++) {
        if (pf.lu[i].n != ln) return fail("lookup domain");
        std::vector<const p3fri::Hash*> roots;
        for (int cid : LD[i].cols) {
            if (cid == -1 || cid == -2) roots.push_back(nullptr);
            else if (cid == -3) roots.push_back(&rGATE);
            else roots.push_back(&pf.rws[cid]);
        }
        std::vector<gl_t> rA, yv;
        if (!p3lu::verify_v(tr, roots, *LD[i].tab, pf.lu[i], Q_pub, R_pub, LD[i].label,
                            why, &rA, &yv, nullptr, &vlg)) return false;
        if (i == LUS_MUL) {
            if (yv.size() != 2) return fail("MUL y_virt count");
            gl_t ysgmb = claimv(tr, vlg, pf.rws[S_SGMB], rA, pf.yMsgmb);
            gl_t yumb = claimv(tr, vlg, pf.rws[S_UMB], rA, pf.yMumb);
            if (yv[0] != gl_add(128ULL, ysgmb)) return fail("MUL sgmf binding");
            if (yv[1] != gl_add(128ULL, yumb)) return fail("MUL umf binding");
        }
    }

    std::vector<gl_t> zE = chal_vec(tr, ln);
    gl_t lamE = chal(tr), lamEv[N_SW_C]; lamEv[0] = 1;
    for (int j = 1; j < N_SW_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.mE, ln, 0, tr, "swg-scE", rE, claim)) return fail("De sumcheck");
        gl_t v[3 + NDS]; v[0] = p3bf::eq_point(rE, zE);
        v[1] = claimv(tr, vlg, rUP, rE, pf.yUP);
        v[2] = claimv(tr, vlg, rM, rE, pf.yM);
        for (int c = 0; c < NDS; c++) v[3 + c] = claimv(tr, vlg, pf.rws[c], rE, pf.yE[c]);
        if (F_sw(v, lamEv) != claim) return fail("De terminal");
    }

    if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
    for (size_t i = 0; i < vlg.cls.size(); i++)
        if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                "swg-bo" + std::to_string(i), why)) return false;

    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const SwgProof& pf) {
    size_t s = 4 + NDS * 32;
    for (int i = 0; i < NLUS; i++) s += p3hwl::sz_lu(pf.lu[i]);
    s += pf.mE.size() * 40;
    s += 8 * (NDS + 4);
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3swg
