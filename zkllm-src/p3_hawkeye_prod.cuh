// Hawkeye per-product gadget over the p3 stack (Goldilocks + Basefold + logUp).
//
// Proves, for every product slot i of the Hawkeye fp8 accumulator replay
// (hawkeye.py, products_per_group=32, internal_width=14), that the committed
// per-product columns satisfy EXACTLY the kernel's decode-multiply-scale-align
// semantics, GIVEN the per-product shift sh_i (sh_i = clamp(max_exp - prod_exp,
// 0, 62) is bound by the increment-2 max_exp gadget; here it is a committed
// input column):
//
//   (a_i, b_i)            fp8 codes (uint8, incl. NaN codes decoded as values)
//   DM lookup:            (a, b, eb, mag, sg, pr) is a row of the 65536-row
//                         decode-multiply table:  eb = a_exp+b_exp-2 (biased
//                         prod_exp), mag = |a_sig*b_sig|<<7 (=|scaled|, IW=14),
//                         sg = sign(product), pr = a_nonzero & b_nonzero.
//                         This FUSES decode + multiply + scale into ONE lookup.
//   SHIFT lookup:         (sh, pw) with pw = 2^min(sh,15)   [64 rows]
//                         (mag < 2^15, so mag>>sh == mag>>min(sh,15))
//   alignment (the irreducible per-product cost, TRUNCATING, no G/R/S bits):
//     C1:  q*pw + r = mag          (field equation; no wrap: < 2^30 << p)
//     REM lookup:  (pw, r) with 0 <= r < pw                [65536 rows]
//     RANGE lookup: q in [0, 2^15)                          [32768 rows]
//     => q = floor(mag / pw) = mag >> min(sh,15), the exact truncated shift.
//   C2:  al = pr * (1-2*sg) * q    (masked signed aligned value; al is the
//                                   term Hawkeye adds into the group sum)
//
// C1, C2 are enforced by ONE eq-weighted batched zero sumcheck (degree 4).
// All witness columns are Basefold-committed; lookups via p3_logup.cuh.
// Non-ZK (masking = later hardening, as in p3pfc); challenges base-field
// (GL2 upgrade is the stack-wide production change).
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "fs_transcript.hpp"

namespace p3hw {

using p3lu::Col; using p3lu::Table; using p3lu::commit_col; using p3lu::make_table;
using p3lu::chal; using p3lu::chal_vec; using p3lu::bind_lsb; using p3lu::ilog2;

// column indices (order fixed; roots absorbed in this order)
enum { CA = 0, CB, CEB, CMAG, CSG, CPR, CSH, CPW, CQ, CR, CAL, NCOL };
static const char* COLNAME[NCOL] = {"a","b","eb","mag","sg","pr","sh","pw","q","r","al"};

struct Tables { Table DM, SH, RM, R15; };

static inline void decode_e4m3(uint32_t raw, int& exp_eff, int& sig_abs, int& sign) {
    sign = (raw >> 7) & 1;
    int eb = (raw >> 3) & 15, mant = raw & 7;
    sig_abs = eb != 0 ? (mant | 8) : mant;
    exp_eff = eb != 0 ? eb : 1;
}

static inline Tables build_tables() {
    Tables T;
    { // DM: fused decode+multiply+scale, keyed by (a,b) -- 2^16 rows, 6 cols
        std::vector<gl_t> a(65536), b(65536), eb(65536), mag(65536), sg(65536), pr(65536);
        for (uint32_t j = 0; j < 65536; j++) {
            uint32_t ca = j >> 8, cb = j & 255;
            int ea, siga, sna, ebx, sigb, snb;
            decode_e4m3(ca, ea, siga, sna); decode_e4m3(cb, ebx, sigb, snb);
            a[j] = ca; b[j] = cb;
            eb[j] = (gl_t)(ea + ebx - 2);              // prod_exp + 12, in [0,28]
            mag[j] = (gl_t)((uint64_t)siga * sigb << 7);  // |scaled|, internal_width=14
            pr[j] = (siga != 0 && sigb != 0) ? 1 : 0;     // present (both nonzero)
            sg[j] = (pr[j] && (sna ^ snb)) ? 1 : 0;       // sign of the product
        }
        T.DM = make_table({a, b, eb, mag, sg, pr});
    }
    { // SHIFT: sh -> 2^min(sh,15) -- 64 rows, 2 cols
        std::vector<gl_t> s(64), p(64);
        for (uint32_t j = 0; j < 64; j++) { s[j] = j; p[j] = 1ULL << (j < 15 ? j : 15); }
        T.SH = make_table({s, p});
    }
    { // REM: (pw, r) with 0 <= r < pw, pw = 2^t, t in [0,15] -- 65535 rows + 1 pad
        std::vector<gl_t> p(65536), r(65536);
        uint32_t row = 0;
        for (uint32_t t = 0; t <= 15; t++)
            for (uint32_t x = 0; x < (1u << t); x++) { p[row] = 1ULL << t; r[row] = x; row++; }
        p[65535] = 1; r[65535] = 0;                    // pad = duplicate of row 0
        T.RM = make_table({p, r});
    }
    { // RANGE15: q in [0, 2^15) -- 32768 rows, 1 col
        std::vector<gl_t> q(32768);
        for (uint32_t j = 0; j < 32768; j++) q[j] = j;
        T.R15 = make_table({q});
    }
    return T;
}

// signed int64 -> field
static inline gl_t enc(int64_t x) { return x >= 0 ? (gl_t)x : gl_sub(0ULL, (gl_t)(-x)); }

// prover-side witness build from raw per-product integer vectors (numpy ref
// layout: a,b,eb,mag,sg,pr,sh,q,r,al).  Pads to a power of two with the
// all-zero-codes product row, which is a valid row of every table.
struct ProdWitness {
    std::vector<gl_t> col[NCOL];
    std::vector<uint32_t> idxDM, idxSH, idxRM, idxQ15;
    size_t n_real = 0;
};
static inline ProdWitness build_witness(const std::vector<int64_t>* raw /*10 arrays*/) {
    ProdWitness W;
    size_t n = raw[0].size(), N = 1; while (N < n) N <<= 1;
    W.n_real = n;
    for (int c = 0; c < NCOL; c++) W.col[c].assign(N, 0);
    W.idxDM.assign(N, 0); W.idxSH.assign(N, 0); W.idxRM.assign(N, 0); W.idxQ15.assign(N, 0);
    for (size_t i = 0; i < N; i++) {
        int64_t a = 0, b = 0, eb = 0, mag = 0, sg = 0, pr = 0, sh = 0, q = 0, r = 0, al = 0;
        if (i < n) { a = raw[0][i]; b = raw[1][i]; eb = raw[2][i]; mag = raw[3][i];
                     sg = raw[4][i]; pr = raw[5][i]; sh = raw[6][i]; q = raw[7][i];
                     r = raw[8][i]; al = raw[9][i]; }
        int64_t shc = sh < 15 ? sh : 15, pw = 1ll << shc;
        W.col[CA][i] = (gl_t)a; W.col[CB][i] = (gl_t)b; W.col[CEB][i] = (gl_t)eb;
        W.col[CMAG][i] = (gl_t)mag; W.col[CSG][i] = (gl_t)sg; W.col[CPR][i] = (gl_t)pr;
        W.col[CSH][i] = (gl_t)sh; W.col[CPW][i] = (gl_t)pw; W.col[CQ][i] = (gl_t)q;
        W.col[CR][i] = (gl_t)r; W.col[CAL][i] = enc(al);
        W.idxDM[i] = (uint32_t)(a * 256 + b);
        W.idxSH[i] = (uint32_t)sh;
        W.idxRM[i] = (uint32_t)((pw - 1) + r);         // block for 2^t starts at 2^t-1
        W.idxQ15[i] = (uint32_t)q;
    }
    return W;
}

struct Msg5 { gl_t s0, s1, s2, s3, s4; };
static inline gl_t quartic_eval(gl_t s0, gl_t s1, gl_t s2, gl_t s3, gl_t s4, gl_t t) {
    gl_t i2 = gl_inv(2ULL), i6 = gl_inv(6ULL), i24 = gl_inv(24ULL), i4 = gl_inv(4ULL);
    gl_t t1 = gl_sub(t,1ULL), t2 = gl_sub(t,2ULL), t3 = gl_sub(t,3ULL), t4 = gl_sub(t,4ULL);
    auto neg = [](gl_t x){ return gl_sub(0ULL, x); };
    gl_t L0 = gl_mul(gl_mul(gl_mul(t1,t2),gl_mul(t3,t4)), i24);
    gl_t L1 = neg(gl_mul(gl_mul(gl_mul(t,t2),gl_mul(t3,t4)), i6));
    gl_t L2 = gl_mul(gl_mul(gl_mul(t,t1),gl_mul(t3,t4)), i4);
    gl_t L3 = neg(gl_mul(gl_mul(gl_mul(t,t1),gl_mul(t2,t4)), i6));
    gl_t L4 = gl_mul(gl_mul(gl_mul(t,t1),gl_mul(t2,t3)), i24);
    gl_t acc = gl_mul(s0, L0);
    acc = gl_add(acc, gl_mul(s1, L1)); acc = gl_add(acc, gl_mul(s2, L2));
    acc = gl_add(acc, gl_mul(s3, L3)); acc = gl_add(acc, gl_mul(s4, L4));
    return acc;
}

struct ProdProof {
    uint32_t n = 0;
    p3fri::Hash root[NCOL];
    p3lu::LookupProof L_DM, L_SH, L_RM, L_Q15;
    std::vector<Msg5> msgs;                              // n rounds, quartic
    p3bf::EvalProof open_q, open_pw, open_r, open_mag, open_al, open_pr, open_sg;
};

// ---------------- prover ----------------
static inline ProdProof prove(fs::Transcript& tr, const ProdWitness& W,
                              const std::vector<Col>& C /*NCOL committed cols*/,
                              const Tables& T, uint32_t R, uint32_t Q,
                              bool gpu = true, bool strict = true) {
    ProdProof pf;
    size_t N = C[0].v.size();
    pf.n = ilog2(N);
    for (int c = 0; c < NCOL; c++) { pf.root[c] = C[c].root; tr.absorb("hw-col", C[c].root.data(), 32); }

    pf.L_DM  = p3lu::prove(tr, {&C[CA], &C[CB], &C[CEB], &C[CMAG], &C[CSG], &C[CPR]},
                           W.idxDM, T.DM, R, Q, "hwDM", gpu, strict);
    pf.L_SH  = p3lu::prove(tr, {&C[CSH], &C[CPW]}, W.idxSH, T.SH, R, Q, "hwSH", gpu, strict);
    pf.L_RM  = p3lu::prove(tr, {&C[CPW], &C[CR]}, W.idxRM, T.RM, R, Q, "hwRM", gpu, strict);
    pf.L_Q15 = p3lu::prove(tr, {&C[CQ]}, W.idxQ15, T.R15, R, Q, "hwQ15", gpu, strict);

    // batched eq-weighted zero sumcheck of C1, C2 (public claim 0)
    std::vector<gl_t> zC = chal_vec(tr, pf.n);
    gl_t lamA = chal(tr), lamB = chal(tr);
    std::vector<gl_t> E = p3bf::build_eq(zC);
    std::vector<gl_t> vq = C[CQ].v, vpw = C[CPW].v, vr = C[CR].v, vmag = C[CMAG].v,
                      val = C[CAL].v, vpr = C[CPR].v, vsg = C[CSG].v;
    std::vector<gl_t> rC;
    for (uint32_t rd = 0; rd < pf.n; rd++) {
        uint32_t half = (uint32_t)vq.size() / 2;
        gl_t s[5] = {0,0,0,0,0};
        for (uint32_t i = 0; i < half; i++) {
            gl_t e  = E[2*i],    dE  = gl_sub(E[2*i+1], E[2*i]);
            gl_t x1 = vq[2*i],   d1  = gl_sub(vq[2*i+1], vq[2*i]);
            gl_t x2 = vpw[2*i],  d2  = gl_sub(vpw[2*i+1], vpw[2*i]);
            gl_t x3 = vr[2*i],   d3  = gl_sub(vr[2*i+1], vr[2*i]);
            gl_t x4 = vmag[2*i], d4  = gl_sub(vmag[2*i+1], vmag[2*i]);
            gl_t x5 = val[2*i],  d5  = gl_sub(val[2*i+1], val[2*i]);
            gl_t x6 = vpr[2*i],  d6  = gl_sub(vpr[2*i+1], vpr[2*i]);
            gl_t x7 = vsg[2*i],  d7  = gl_sub(vsg[2*i+1], vsg[2*i]);
            for (int t = 0; t < 5; t++) {
                gl_t C1 = gl_sub(gl_add(gl_mul(x1, x2), x3), x4);
                gl_t one_m2sg = gl_sub(1ULL, gl_add(x7, x7));
                gl_t C2 = gl_sub(x5, gl_mul(gl_mul(x6, x1), one_m2sg));
                s[t] = gl_add(s[t], gl_mul(e, gl_add(gl_mul(lamA, C1), gl_mul(lamB, C2))));
                e = gl_add(e, dE); x1 = gl_add(x1, d1); x2 = gl_add(x2, d2);
                x3 = gl_add(x3, d3); x4 = gl_add(x4, d4); x5 = gl_add(x5, d5);
                x6 = gl_add(x6, d6); x7 = gl_add(x7, d7);
            }
        }
        Msg5 m{s[0], s[1], s[2], s[3], s[4]};
        pf.msgs.push_back(m); tr.absorb("hw-sc", &m, sizeof m);
        gl_t a = chal(tr); rC.push_back(a);
        bind_lsb(E, a); bind_lsb(vq, a); bind_lsb(vpw, a); bind_lsb(vr, a);
        bind_lsb(vmag, a); bind_lsb(val, a); bind_lsb(vpr, a); bind_lsb(vsg, a);
    }
    // open the 7 constraint columns at rC
    auto open1 = [&](const Col& c, const char* lbl) {
        gl_t y = p3bf::eval_h(c.v, p3bf::build_eq(rC));
        return p3bf::prove_eval(c.v, rC, y, R, Q, c.cw, std::string("hwC-") + lbl);
    };
    pf.open_q  = open1(C[CQ], "q");   pf.open_pw = open1(C[CPW], "pw");
    pf.open_r  = open1(C[CR], "r");   pf.open_mag = open1(C[CMAG], "mag");
    pf.open_al = open1(C[CAL], "al"); pf.open_pr = open1(C[CPR], "pr");
    pf.open_sg = open1(C[CSG], "sg");
    return pf;
}

// ---------------- verifier ----------------
static inline bool verify(fs::Transcript& tr, const Tables& T, const ProdProof& pf,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    for (int c = 0; c < NCOL; c++) tr.absorb("hw-col", pf.root[c].data(), 32);

    auto rt = [&](int c) { return pf.root[c]; };
    if (!p3lu::verify(tr, {rt(CA), rt(CB), rt(CEB), rt(CMAG), rt(CSG), rt(CPR)}, T.DM,
                      pf.L_DM, Q_pub, R_pub, "hwDM", why)) return false;
    if (!p3lu::verify(tr, {rt(CSH), rt(CPW)}, T.SH, pf.L_SH, Q_pub, R_pub, "hwSH", why)) return false;
    if (!p3lu::verify(tr, {rt(CPW), rt(CR)}, T.RM, pf.L_RM, Q_pub, R_pub, "hwRM", why)) return false;
    if (!p3lu::verify(tr, {rt(CQ)}, T.R15, pf.L_Q15, Q_pub, R_pub, "hwQ15", why)) return false;
    // all lookups must be over the same witness length
    if (pf.L_DM.n != pf.n || pf.L_SH.n != pf.n || pf.L_RM.n != pf.n || pf.L_Q15.n != pf.n)
        return fail("lookup length");

    std::vector<gl_t> zC = chal_vec(tr, pf.n);
    gl_t lamA = chal(tr), lamB = chal(tr);
    if (pf.msgs.size() != pf.n) return fail("msg count");
    gl_t claim = 0;                                     // PUBLIC zero claim
    std::vector<gl_t> rC;
    for (uint32_t rd = 0; rd < pf.n; rd++) {
        const Msg5& m = pf.msgs[rd];
        if (gl_add(m.s0, m.s1) != claim) return fail("constraint sumcheck claim");
        tr.absorb("hw-sc", &m, sizeof m);
        gl_t a = chal(tr); rC.push_back(a);
        claim = quartic_eval(m.s0, m.s1, m.s2, m.s3, m.s4, a);
    }
    // openings: bind root + point + public params, then verify
    struct OB { const p3bf::EvalProof* e; int c; const char* lbl; };
    OB obs[7] = {{&pf.open_q, CQ, "q"}, {&pf.open_pw, CPW, "pw"}, {&pf.open_r, CR, "r"},
                 {&pf.open_mag, CMAG, "mag"}, {&pf.open_al, CAL, "al"},
                 {&pf.open_pr, CPR, "pr"}, {&pf.open_sg, CSG, "sg"}};
    for (auto& o : obs) {
        if (o.e->roots.empty() || !(o.e->roots[0] == pf.root[o.c]) || o.e->z != rC)
            return fail("constraint opening bind");
        if (o.e->Q != Q_pub || o.e->R != R_pub || o.e->logN != pf.n)
            return fail("constraint opening params");
        if (!p3bf::verify_eval(*o.e, std::string("hwC-") + o.lbl, why)) return false;
    }
    gl_t yq = pf.open_q.y, ypw = pf.open_pw.y, yr = pf.open_r.y, ymag = pf.open_mag.y,
         yal = pf.open_al.y, ypr = pf.open_pr.y, ysg = pf.open_sg.y;
    gl_t C1 = gl_sub(gl_add(gl_mul(yq, ypw), yr), ymag);
    gl_t C2 = gl_sub(yal, gl_mul(gl_mul(ypr, yq), gl_sub(1ULL, gl_add(ysg, ysg))));
    gl_t end = gl_mul(p3bf::eq_point(rC, zC), gl_add(gl_mul(lamA, C1), gl_mul(lamB, C2)));
    if (claim != end) return fail("constraint terminal");
    if (why) *why = "ok";
    return true;
}

} // namespace p3hw
