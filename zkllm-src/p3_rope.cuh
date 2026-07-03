// RoPE gadget: sound (non-ZK) prover+verifier that a committed bf16 output
// OUT equals the CANONICAL RoPE rotation (transformer_ref.py rope_apply) of a
// committed bf16 input Q, bitwise, for PUBLIC pinned bf16 cos/sin tables:
//   per (pos p, pair j < dh/2), with a = Q[p,j], b = Q[p,j+h]:
//     OUT[p,j]   = bf_add( bf_mul(a, cos[p,j]), -bf_mul(b, sin[p,j]) )
//     OUT[p,j+h] = bf_add( bf_mul(b, cos[p,j]),  bf_mul(a, sin[p,j]) )
//   (one RNE per product, one canonical bf_add per combine, llama rotate_half)
//
// Composition demonstrator for the section-11.5 vocabulary:
//   * Q and OUT are committed ONCE over (pos, dh); the a/b halves and the two
//     output halves are NOT re-committed -- the gadget opens the SAME parent
//     column at points with the half-select index bit fixed to 0/1 (the "head
//     slice = fix an index bit of the opening point" move).
//   * cos/sin are PUBLIC: their field columns enter the zero-check as plain
//     arrays and the verifier evaluates their MLEs itself (no commitment, no
//     opening); the raw tables are absorbed into the transcript.
//   * the two combines INSTANTIATE the p3_bfadd block (55 columns + 16
//     lookups each) -- the RNE-add primitive is proven once, reused here.
//   * the four products reuse the MUL7/REXP mul vocabulary (RMSNorm/SwiGLU).
//
// Supported domain (proof REJECTS otherwise): product and combine exponents
// in [1,254] (the p3_bfadd v1 rule).  Zero/subnormal inputs and sin(0)=+0
// rows are in-domain.
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

namespace p3rope {

using p3lu::Col; using p3lu::Table; using p3lu::make_table; using p3lu::commit_col_nc;
using p3lu::chal; using p3lu::chal_vec; using p3lu::ilog2; using p3lu::bind_lsb;
using p3lu::LuColSpec; using p3lu::LC; using p3lu::LV;
using p3hwl::Msg5; using p3hwl::CFn; using p3hwl::sc5_prove; using p3hwl::sc5_verify;
using p3hwl::claimc; using p3hwl::claimv; using p3hwl::beq;
using p3rms::Art;

// evaluate the MLE of a PUBLIC value vector at a point (LSB-first fold)
static inline gl_t mle_eval(std::vector<gl_t> f, const std::vector<gl_t>& r) {
    for (gl_t a : r) bind_lsb(f, a);
    return f[0];
}
// opening point with one index bit inserted at position pos
static inline std::vector<gl_t> ins_pt(const std::vector<gl_t>& r, uint32_t pos, gl_t bit) {
    std::vector<gl_t> p(r.begin(), r.begin() + pos);
    p.push_back(bit);
    p.insert(p.end(), r.begin() + pos, r.end());
    return p;
}

// ---------------- columns ----------------
enum {
    RP_AS = 0, RP_AEB, RP_AMB, RP_AZ, RP_AEI,       // a = Q[p, j]
    RP_BS, RP_BEB, RP_BMB, RP_BZ, RP_BEI,           // b = Q[p, j+h]
    RP_MOAC, RP_EIAC, RP_EOAC, RP_PAC,              // a*cos
    RP_MOSB, RP_EISB, RP_EOSB, RP_PSB,              // b*sin
    RP_MOCB, RP_EICB, RP_EOCB, RP_PCB,              // b*cos
    RP_MOSA, RP_EISA, RP_EOSA, RP_PSA,              // a*sin
    RP_NSB,                                          // -(b*sin) pattern
    RP_FIX };
static const int RP_A1 = RP_FIX;                     // combine 1 add block
static const int RP_A2 = RP_FIX + p3bfa::NBA;        // combine 2 add block
static const int NDR = RP_FIX + 2 * p3bfa::NBA;

enum { LRP_AEB = 0, LRP_AMB, LRP_BEB, LRP_BMB,
       LRP_MAC, LRP_MSB, LRP_MCB, LRP_MSA,
       LRP_EAC, LRP_ESB, LRP_ECB, LRP_ESA, LRP_FIX };
static const int LRP_A1 = LRP_FIX;                   // 16 add-block lookups
static const int LRP_A2 = LRP_FIX + p3bfa::NBLU;
static const int NLRP = LRP_FIX + 2 * p3bfa::NBLU;

struct Tables {
    p3bfa::Tables BT;                                // R128/R256/R512/.../REXP
    Table MUL7;
};
static inline Tables build_tables(const Art& a) {
    Tables T;
    T.BT = p3bfa::build_tables();
    std::vector<gl_t> ma(16384), mb(16384), mo(16384), ei(16384);
    for (uint32_t j = 0; j < 16384; j++) {
        ma[j] = 128 + (j >> 7); mb[j] = 128 + (j & 127);
        mo[j] = 128 + a.mul_mo[j]; ei[j] = a.mul_einc[j];
    }
    T.MUL7 = make_table({ma, mb, mo, ei});
    return T;
}

// ---------------- golden ----------------
struct GoldenSet {
    uint32_t seq = 0, dh = 0;
    std::vector<uint16_t> cos, sin;                  // seq * dh/2 each
    struct Case { int64_t flags; std::vector<uint16_t> q, out; };
    std::vector<Case> cases;
};
static inline bool load_goldens(const char* path, GoldenSet& gs) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[4];
    if (fread(hdr, 8, 4, f) != 4 || hdr[0] != 0x524F5047) { fclose(f); return false; }
    gs.seq = (uint32_t)hdr[2]; gs.dh = (uint32_t)hdr[3];
    size_t nh = (size_t)gs.seq * (gs.dh / 2), nq = (size_t)gs.seq * gs.dh;
    gs.cos.resize(nh); gs.sin.resize(nh);
    if (fread(gs.cos.data(), 2, nh, f) != nh ||
        fread(gs.sin.data(), 2, nh, f) != nh) { fclose(f); return false; }
    gs.cases.resize(hdr[1]);
    for (auto& C : gs.cases) {
        if (fread(&C.flags, 8, 1, f) != 1) { fclose(f); return false; }
        C.q.resize(nq); C.out.resize(nq);
        if (fread(C.q.data(), 2, nq, f) != nq ||
            fread(C.out.data(), 2, nq, f) != nq) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

// ---------------- witness ----------------
enum { RPT_NONE = 0,
       RPT_MUL,       // a*cos mantissa MO+1 (not a MUL7 row), honest downstream
       RPT_ADDRUP };  // combine-1 RNE round bit flipped, honest downstream
struct RpTamper { int mode = RPT_NONE; uint32_t e = 0; };

struct Wit {
    uint32_t seq = 0, half = 0, lp = 0, lh = 0, ln = 0;
    size_t Ne = 0, Nq = 0;
    std::vector<gl_t> qpat, opat;                    // committed (Nq)
    std::vector<gl_t> qa, qb, oa, ob;                // parent-column halves (Ne)
    std::vector<gl_t> cf[4], sf[4];                  // public fields s,eb,mb,z
    std::vector<uint16_t> OUT;                       // computed outputs (Nq)
    std::vector<gl_t> ws[NDR];
    std::vector<uint32_t> lidx[NLRP];
    std::vector<gl_t> amf, bmf, cmf, smf;            // +128 MUL7 key columns
};
static inline gl_t inv_or0(uint64_t x) { return x ? gl_inv((gl_t)x) : 0; }

static inline Wit gen_witness(const GoldenSet& gs, const GoldenSet::Case& L,
                              const Art& a, const RpTamper* tm = nullptr) {
    Wit wt;
    wt.seq = gs.seq; wt.half = gs.dh / 2;
    wt.lp = ilog2(wt.seq); wt.lh = ilog2(wt.half);
    if ((1u << wt.lp) != wt.seq || (1u << wt.lh) != wt.half)
        throw std::runtime_error("rope: dims must be pow2");
    wt.ln = wt.lp + wt.lh;
    wt.Ne = (size_t)wt.seq * wt.half; wt.Nq = 2 * wt.Ne;
    wt.qpat.assign(wt.Nq, 0); wt.opat.assign(wt.Nq, 0);
    wt.qa.assign(wt.Ne, 0); wt.qb.assign(wt.Ne, 0);
    wt.oa.assign(wt.Ne, 0); wt.ob.assign(wt.Ne, 0);
    for (int k = 0; k < 4; k++) { wt.cf[k].assign(wt.Ne, 0); wt.sf[k].assign(wt.Ne, 0); }
    wt.OUT.assign(wt.Nq, 0);
    for (int c = 0; c < NDR; c++) wt.ws[c].assign(wt.Ne, 0);
    for (int i = 0; i < NLRP; i++) wt.lidx[i].assign(wt.Ne, 0);
    wt.amf.assign(wt.Ne, 0); wt.bmf.assign(wt.Ne, 0);
    wt.cmf.assign(wt.Ne, 0); wt.smf.assign(wt.Ne, 0);

    const uint32_t dh = gs.dh, half = wt.half;
    for (uint32_t p = 0; p < wt.seq; p++)
    for (uint32_t j = 0; j < half; j++) {
        size_t e = ((size_t)p << wt.lh) | j;
        uint32_t ap = L.q[(size_t)p * dh + j], bp = L.q[(size_t)p * dh + half + j];
        uint32_t cp = gs.cos[e], sp = gs.sin[e];
        auto dec = [](uint32_t x, int64_t* d) {
            d[0] = (x >> 15) & 1; d[1] = (x >> 7) & 255; d[2] = x & 127;
            d[3] = d[1] == 0 ? 1 : 0;
        };
        int64_t da[4], db[4], dc[4], ds[4];
        dec(ap, da); dec(bp, db); dec(cp, dc); dec(sp, ds);
        wt.ws[RP_AS][e] = (gl_t)da[0]; wt.ws[RP_AEB][e] = (gl_t)da[1];
        wt.ws[RP_AMB][e] = (gl_t)da[2]; wt.ws[RP_AZ][e] = (gl_t)da[3];
        wt.ws[RP_AEI][e] = inv_or0((uint64_t)da[1]);
        wt.ws[RP_BS][e] = (gl_t)db[0]; wt.ws[RP_BEB][e] = (gl_t)db[1];
        wt.ws[RP_BMB][e] = (gl_t)db[2]; wt.ws[RP_BZ][e] = (gl_t)db[3];
        wt.ws[RP_BEI][e] = inv_or0((uint64_t)db[1]);
        for (int k = 0; k < 4; k++) { wt.cf[k][e] = (gl_t)dc[k]; wt.sf[k][e] = (gl_t)ds[k]; }
        wt.qa[e] = ap; wt.qb[e] = bp;
        wt.qpat[((size_t)p << (wt.lh + 1)) | j] = ap;
        wt.qpat[((size_t)p << (wt.lh + 1)) | half | j] = bp;
        wt.amf[e] = (gl_t)(128 + da[2]); wt.bmf[e] = (gl_t)(128 + db[2]);
        wt.cmf[e] = (gl_t)(128 + dc[2]); wt.smf[e] = (gl_t)(128 + ds[2]);
        wt.lidx[LRP_AEB][e] = (uint32_t)da[1]; wt.lidx[LRP_AMB][e] = (uint32_t)da[2];
        wt.lidx[LRP_BEB][e] = (uint32_t)db[1]; wt.lidx[LRP_BMB][e] = (uint32_t)db[2];
        // the four products
        auto mul = [&](const int64_t* x, const int64_t* t, int mo_c, int lu_m, int lu_e,
                       bool tamper) -> uint16_t {
            uint32_t mj = (uint32_t)((x[2] << 7) | t[2]);
            int64_t mo = 128 + a.mul_mo[mj], ei = a.mul_einc[mj];
            if (tamper) mo += (mo < 255 ? 1 : -1);
            int64_t z2 = (x[3] || t[3]) ? 1 : 0;
            int64_t eo = 1;
            if (!z2) {
                eo = x[1] + t[1] - 127 + ei;
                if (eo < 1 || eo > 254) throw std::runtime_error("rope: mul exp domain");
            }
            int64_t sg = x[0] ^ t[0];
            uint16_t pat = (uint16_t)((sg << 15) | (z2 ? 0 : ((eo << 7) | (mo - 128))));
            wt.ws[mo_c][e] = (gl_t)mo; wt.ws[mo_c + 1][e] = (gl_t)ei;
            wt.ws[mo_c + 2][e] = (gl_t)eo; wt.ws[mo_c + 3][e] = (gl_t)pat;
            wt.lidx[lu_m][e] = mj; wt.lidx[lu_e][e] = (uint32_t)eo;
            return pat;
        };
        bool tmul = tm && tm->mode == RPT_MUL && e == tm->e;
        uint16_t pac = mul(da, dc, RP_MOAC, LRP_MAC, LRP_EAC, tmul);
        uint16_t psb = mul(db, ds, RP_MOSB, LRP_MSB, LRP_ESB, false);
        uint16_t pcb = mul(db, dc, RP_MOCB, LRP_MCB, LRP_ECB, false);
        uint16_t psa = mul(da, ds, RP_MOSA, LRP_MSA, LRP_ESA, false);
        uint16_t nsb = psb ^ 0x8000;
        wt.ws[RP_NSB][e] = (gl_t)nsb;
        // the two combines: instantiated p3_bfadd blocks
        p3bfa::BaTamper bt; bt.mode = p3bfa::BAT_RUP;
        bool tadd = tm && tm->mode == RPT_ADDRUP && e == tm->e;
        p3bfa::BaVals a1 = p3bfa::ba_fill(pac, nsb, tadd ? &bt : nullptr);
        p3bfa::BaVals a2 = p3bfa::ba_fill(pcb, psa);
        for (int c = 0; c < p3bfa::NBA; c++) {
            wt.ws[RP_A1 + c][e] = a1.v[c];
            wt.ws[RP_A2 + c][e] = a2.v[c];
        }
        for (int i = 0; i < p3bfa::NBLU; i++) {
            wt.lidx[LRP_A1 + i][e] = a1.lu[i];
            wt.lidx[LRP_A2 + i][e] = a2.lu[i];
        }
        wt.oa[e] = a1.out; wt.ob[e] = a2.out;
        wt.opat[((size_t)p << (wt.lh + 1)) | j] = a1.out;
        wt.opat[((size_t)p << (wt.lh + 1)) | half | j] = a2.out;
        wt.OUT[(size_t)p * dh + j] = a1.out;
        wt.OUT[(size_t)p * dh + half + j] = a2.out;
    }
    return wt;
}

struct Operands { Col Q, OUT; };
static inline Operands commit_operands(const Wit& wt, uint32_t R) {
    Operands ops;
    ops.Q = commit_col_nc(wt.qpat, R);
    ops.OUT = commit_col_nc(wt.opat, R);
    return ops;
}

// ---------------- constraints ----------------
// v = [Eq, QA, QB, OA, OB, CS,CEB,CMB,CZ, SS,SEB,SMB,SZ, ws...]; lam[N_RP_C]
static const int N_RP_C = 23 + 2 * p3bfa::N_BA_C;
static inline gl_t F_rp(const gl_t* v, const gl_t* lam) {
    const gl_t* c = v + 13;
    gl_t one = 1ULL;
    auto boolc = [](gl_t x) { return gl_sub(gl_mul(x, x), x); };
    gl_t r[N_RP_C];
    // operand decompositions against the Q parent-column halves
    r[0] = gl_sub(v[1], gl_add(gl_add(gl_mul(c[RP_AS], 32768ULL),
                                      gl_mul(c[RP_AEB], 128ULL)), c[RP_AMB]));
    r[1] = boolc(c[RP_AS]);
    r[2] = boolc(c[RP_AZ]);
    r[3] = gl_mul(c[RP_AZ], c[RP_AEB]);
    r[4] = gl_sub(gl_mul(c[RP_AEB], c[RP_AEI]), gl_sub(one, c[RP_AZ]));
    r[5] = gl_sub(v[2], gl_add(gl_add(gl_mul(c[RP_BS], 32768ULL),
                                      gl_mul(c[RP_BEB], 128ULL)), c[RP_BMB]));
    r[6] = boolc(c[RP_BS]);
    r[7] = boolc(c[RP_BZ]);
    r[8] = gl_mul(c[RP_BZ], c[RP_BEB]);
    r[9] = gl_sub(gl_mul(c[RP_BEB], c[RP_BEI]), gl_sub(one, c[RP_BZ]));
    // the four products (public t operand fields from v)
    auto mulc = [&](int xs, int xeb, int xz, int tv, int mo_c, gl_t* rr) {
        gl_t TS = v[tv], TEB = v[tv + 1], TZ = v[tv + 3];
        gl_t z2 = gl_sub(gl_add(c[xz], TZ), gl_mul(c[xz], TZ));
        gl_t nz2 = gl_sub(one, z2);
        gl_t sg = gl_sub(gl_add(c[xs], TS), gl_mul(gl_add(c[xs], c[xs]), TS));
        rr[0] = gl_mul(z2, gl_sub(c[mo_c + 2], one));
        rr[1] = gl_mul(nz2, gl_sub(c[mo_c + 2],
                        gl_sub(gl_add(gl_add(c[xeb], TEB), c[mo_c + 1]), 127ULL)));
        rr[2] = gl_sub(c[mo_c + 3], gl_add(gl_mul(sg, 32768ULL),
                        gl_mul(nz2, gl_sub(gl_add(gl_mul(c[mo_c + 2], 128ULL),
                                                  c[mo_c]), 128ULL))));
    };
    mulc(RP_AS, RP_AEB, RP_AZ, 5, RP_MOAC, r + 10);   // a*cos
    mulc(RP_BS, RP_BEB, RP_BZ, 9, RP_MOSB, r + 13);   // b*sin
    mulc(RP_BS, RP_BEB, RP_BZ, 5, RP_MOCB, r + 16);   // b*cos
    mulc(RP_AS, RP_AEB, RP_AZ, 9, RP_MOSA, r + 19);   // a*sin
    // negation of b*sin: sign bit flip as +-32768
    {
        gl_t sgsb = gl_sub(gl_add(c[RP_BS], v[9]),
                           gl_mul(gl_add(c[RP_BS], c[RP_BS]), v[9]));
        r[22] = gl_sub(gl_add(c[RP_NSB], gl_mul(sgsb, 65536ULL)),
                       gl_add(c[RP_PSB], 32768ULL));
    }
    // the two combines (bf_add blocks against the OUT parent-column halves)
    p3bfa::ba_constraints(c + RP_A1, c[RP_PAC], c[RP_NSB], v[3], r + 23);
    p3bfa::ba_constraints(c + RP_A2, c[RP_PCB], c[RP_PSA], v[4],
                          r + 23 + p3bfa::N_BA_C);
    gl_t s = 0;
    for (int j = 0; j < N_RP_C; j++) s = gl_add(s, gl_mul(lam[j], r[j]));
    return gl_mul(v[0], s);
}

// ---------------- lookup descriptors ----------------
// cols: >=0 committed ws; -1 amf, -2 bmf, -3 cmf(public), -4 smf(public)
struct LuDef { int tab; std::vector<int> cols; };    // tab: 0..NBT-1 in BT, -1 MUL7
static inline std::vector<LuDef> lu_defs() {
    std::vector<LuDef> L(NLRP);
    L[LRP_AEB] = {p3bfa::BT_R256, {RP_AEB}};
    L[LRP_AMB] = {p3bfa::BT_R128, {RP_AMB}};
    L[LRP_BEB] = {p3bfa::BT_R256, {RP_BEB}};
    L[LRP_BMB] = {p3bfa::BT_R128, {RP_BMB}};
    L[LRP_MAC] = {-1, {-1, -3, RP_MOAC, RP_EIAC}};
    L[LRP_MSB] = {-1, {-2, -4, RP_MOSB, RP_EISB}};
    L[LRP_MCB] = {-1, {-2, -3, RP_MOCB, RP_EICB}};
    L[LRP_MSA] = {-1, {-1, -4, RP_MOSA, RP_EISA}};
    L[LRP_EAC] = {p3bfa::BT_REXP, {RP_EOAC}};
    L[LRP_ESB] = {p3bfa::BT_REXP, {RP_EOSB}};
    L[LRP_ECB] = {p3bfa::BT_REXP, {RP_EOCB}};
    L[LRP_ESA] = {p3bfa::BT_REXP, {RP_EOSA}};
    auto BL = p3bfa::ba_lu_defs();
    for (int i = 0; i < p3bfa::NBLU; i++) {
        std::vector<int> c1, c2;
        for (int cid : BL[i].cols) { c1.push_back(RP_A1 + cid); c2.push_back(RP_A2 + cid); }
        L[LRP_A1 + i] = {BL[i].tab, c1};
        L[LRP_A2 + i] = {BL[i].tab, c2};
    }
    return L;
}

// ---------------- proof object ----------------
struct RopeProof {
    uint32_t lp = 0, lh = 0;
    p3fri::Hash rws[NDR];
    p3lu::LookupProof lu[NLRP];
    gl_t yMK[4] = {};                                // AMB/BMB bindings at rA
    std::vector<Msg5> mE; std::vector<gl_t> yE;
    gl_t yQA = 0, yQB = 0, yOA = 0, yOB = 0;
    std::vector<p3bo::BatchProof> batches;
};

// ==================== prover ====================
static inline RopeProof prove(fs::Transcript& tr, const Wit& wt, const Tables& T,
                              const Operands& ops, uint32_t R, uint32_t Q,
                              bool strict = true) {
    RopeProof pf; pf.lp = wt.lp; pf.lh = wt.lh;
    uint32_t hdr[2] = {wt.lp, wt.lh};
    tr.absorb("rop-dims", hdr, sizeof hdr);
    for (int i = 0; i < p3bfa::NBT; i++) tr.absorb("rop-tab", T.BT.t[i].id.data(), 32);
    tr.absorb("rop-tab", T.MUL7.id.data(), 32);
    for (int k = 0; k < 4; k++) {                    // pinned cos/sin fields
        tr.absorb("rop-cos", wt.cf[k].data(), wt.cf[k].size() * sizeof(gl_t));
        tr.absorb("rop-sin", wt.sf[k].data(), wt.sf[k].size() * sizeof(gl_t));
    }
    tr.absorb("rop-Q", ops.Q.root.data(), 32);
    tr.absorb("rop-O", ops.OUT.root.data(), 32);

    p3bo::PLedger lg;
    std::deque<Col> lucols;

    std::vector<Col> C(NDR);
    for (int c = 0; c < NDR; c++) { C[c] = commit_col_nc(wt.ws[c], R); pf.rws[c] = C[c].root; }
    for (int c = 0; c < NDR; c++) tr.absorb("rop-cw", pf.rws[c].data(), 32);

    auto LD = lu_defs();
    for (int i = 0; i < NLRP; i++) {
        std::vector<LuColSpec> spec;
        for (int cid : LD[i].cols) {
            if (cid == -1) spec.push_back(LV(&wt.amf));
            else if (cid == -2) spec.push_back(LV(&wt.bmf));
            else if (cid == -3) spec.push_back(LV(&wt.cmf));
            else if (cid == -4) spec.push_back(LV(&wt.smf));
            else spec.push_back(LC(&C[cid]));
        }
        const Table& tab = LD[i].tab < 0 ? T.MUL7 : T.BT.t[LD[i].tab];
        bool ismul = i >= LRP_MAC && i <= LRP_MSA;
        std::vector<gl_t> rA;
        pf.lu[i] = p3lu::prove_v(tr, spec, wt.lidx[i], tab, R, Q,
                                 "ropLU" + std::to_string(i), true, strict,
                                 ismul ? &rA : nullptr, &lg, &lucols);
        if (ismul) {
            int wcol = (i == LRP_MAC || i == LRP_MSA) ? RP_AMB : RP_BMB;
            pf.yMK[i - LRP_MAC] = claimc(tr, lg, C[wcol], rA);
        }
    }

    std::vector<gl_t> zE = chal_vec(tr, wt.ln);
    gl_t lamE = chal(tr);
    std::vector<gl_t> lamEv(N_RP_C); lamEv[0] = 1;
    for (int j = 1; j < N_RP_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<std::vector<gl_t>> cols; cols.reserve(13 + NDR);
        cols.push_back(beq(zE));
        cols.push_back(wt.qa); cols.push_back(wt.qb);
        cols.push_back(wt.oa); cols.push_back(wt.ob);
        for (int k = 0; k < 4; k++) cols.push_back(wt.cf[k]);
        for (int k = 0; k < 4; k++) cols.push_back(wt.sf[k]);
        for (int c = 0; c < NDR; c++) cols.push_back(wt.ws[c]);
        CFn F = [&](const gl_t* v) { return F_rp(v, lamEv.data()); };
        std::vector<gl_t> rE = sc5_prove(tr, "rop-scE", std::move(cols), F, pf.mE);
        pf.yQA = claimc(tr, lg, ops.Q, ins_pt(rE, wt.lh, 0));
        pf.yQB = claimc(tr, lg, ops.Q, ins_pt(rE, wt.lh, 1));
        pf.yOA = claimc(tr, lg, ops.OUT, ins_pt(rE, wt.lh, 0));
        pf.yOB = claimc(tr, lg, ops.OUT, ins_pt(rE, wt.lh, 1));
        pf.yE.resize(NDR);
        for (int c = 0; c < NDR; c++) pf.yE[c] = claimc(tr, lg, C[c], rE);
    }

    for (size_t i = 0; i < lg.cls.size(); i++)
        pf.batches.push_back(p3bo::prove_class(tr, lg.cls[i], R, Q,
                                               "rop-bo" + std::to_string(i)));
    return pf;
}

// ==================== verifier ====================
// PUBLIC inputs: cos/sin pattern tables, Q/OUT roots, dims, Q/R params.
static inline bool verify(fs::Transcript& tr, const Tables& T, const RopeProof& pf,
                          const std::vector<uint16_t>& cosp,
                          const std::vector<uint16_t>& sinp,
                          const p3fri::Hash& rQ, const p3fri::Hash& rOUT,
                          uint32_t seq, uint32_t dh,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    uint32_t half = dh / 2, lp = ilog2(seq), lh = ilog2(half);
    if ((1u << lp) != seq || (1u << lh) != half) return fail("dims must be pow2");
    if (pf.lp != lp || pf.lh != lh) return fail("dims mismatch");
    uint32_t ln = lp + lh;
    size_t Ne = (size_t)seq * half;
    if (cosp.size() != Ne || sinp.size() != Ne) return fail("cos/sin size");
    if (pf.yE.size() != NDR) return fail("yE count");
    p3bo::VLedger vlg;

    // rebuild the public field columns
    std::vector<gl_t> cf[4], sf[4], cmf(Ne), smf(Ne);
    for (int k = 0; k < 4; k++) { cf[k].assign(Ne, 0); sf[k].assign(Ne, 0); }
    for (size_t e = 0; e < Ne; e++) {
        uint32_t cp = cosp[e], sp = sinp[e];
        cf[0][e] = (cp >> 15) & 1; cf[1][e] = (cp >> 7) & 255;
        cf[2][e] = cp & 127; cf[3][e] = ((cp >> 7) & 255) == 0 ? 1 : 0;
        sf[0][e] = (sp >> 15) & 1; sf[1][e] = (sp >> 7) & 255;
        sf[2][e] = sp & 127; sf[3][e] = ((sp >> 7) & 255) == 0 ? 1 : 0;
        cmf[e] = gl_add(128ULL, cf[2][e]); smf[e] = gl_add(128ULL, sf[2][e]);
    }

    uint32_t hdr[2] = {lp, lh};
    tr.absorb("rop-dims", hdr, sizeof hdr);
    for (int i = 0; i < p3bfa::NBT; i++) tr.absorb("rop-tab", T.BT.t[i].id.data(), 32);
    tr.absorb("rop-tab", T.MUL7.id.data(), 32);
    for (int k = 0; k < 4; k++) {
        tr.absorb("rop-cos", cf[k].data(), cf[k].size() * sizeof(gl_t));
        tr.absorb("rop-sin", sf[k].data(), sf[k].size() * sizeof(gl_t));
    }
    tr.absorb("rop-Q", rQ.data(), 32);
    tr.absorb("rop-O", rOUT.data(), 32);
    for (int c = 0; c < NDR; c++) tr.absorb("rop-cw", pf.rws[c].data(), 32);

    auto LD = lu_defs();
    for (int i = 0; i < NLRP; i++) {
        if (pf.lu[i].n != ln) return fail("lookup domain");
        std::vector<const p3fri::Hash*> roots;
        for (int cid : LD[i].cols)
            roots.push_back(cid < 0 ? nullptr : &pf.rws[cid]);
        const Table& tab = LD[i].tab < 0 ? T.MUL7 : T.BT.t[LD[i].tab];
        bool ismul = i >= LRP_MAC && i <= LRP_MSA;
        std::vector<gl_t> rA, yv;
        if (!p3lu::verify_v(tr, roots, tab, pf.lu[i], Q_pub, R_pub,
                            "ropLU" + std::to_string(i), why,
                            ismul ? &rA : nullptr, ismul ? &yv : nullptr,
                            nullptr, &vlg)) return false;
        if (ismul) {
            if (yv.size() != 2) return fail("MUL y_virt count");
            int wcol = (i == LRP_MAC || i == LRP_MSA) ? RP_AMB : RP_BMB;
            gl_t ymb = claimv(tr, vlg, pf.rws[wcol], rA, pf.yMK[i - LRP_MAC]);
            if (yv[0] != gl_add(128ULL, ymb)) return fail("MUL witness-key binding");
            // the public cos/sin key: the verifier evaluates the MLE itself
            const std::vector<gl_t>& pub =
                (i == LRP_MAC || i == LRP_MCB) ? cmf : smf;
            if (yv[1] != mle_eval(pub, rA)) return fail("MUL public-key binding");
        }
    }

    std::vector<gl_t> zE = chal_vec(tr, ln);
    gl_t lamE = chal(tr);
    std::vector<gl_t> lamEv(N_RP_C); lamEv[0] = 1;
    for (int j = 1; j < N_RP_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.mE, ln, 0, tr, "rop-scE", rE, claim)) return fail("De sumcheck");
        std::vector<gl_t> v(13 + NDR);
        v[0] = p3bf::eq_point(rE, zE);
        v[1] = claimv(tr, vlg, rQ, ins_pt(rE, lh, 0), pf.yQA);
        v[2] = claimv(tr, vlg, rQ, ins_pt(rE, lh, 1), pf.yQB);
        v[3] = claimv(tr, vlg, rOUT, ins_pt(rE, lh, 0), pf.yOA);
        v[4] = claimv(tr, vlg, rOUT, ins_pt(rE, lh, 1), pf.yOB);
        for (int k = 0; k < 4; k++) v[5 + k] = mle_eval(cf[k], rE);
        for (int k = 0; k < 4; k++) v[9 + k] = mle_eval(sf[k], rE);
        for (int c = 0; c < NDR; c++) v[13 + c] = claimv(tr, vlg, pf.rws[c], rE, pf.yE[c]);
        if (F_rp(v.data(), lamEv.data()) != claim) return fail("De terminal");
    }

    if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
    for (size_t i = 0; i < vlg.cls.size(); i++)
        if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                "rop-bo" + std::to_string(i), why)) return false;

    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const RopeProof& pf) {
    size_t s = 8 + NDR * 32;
    for (int i = 0; i < NLRP; i++) s += p3hwl::sz_lu(pf.lu[i]);
    s += pf.mE.size() * 40;
    s += 8 * (NDR + 8);
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3rope
