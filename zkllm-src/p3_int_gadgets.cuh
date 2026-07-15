// p3_int_gadgets.cuh -- INTEGER transformer-layer gadget set on the Goldilocks
// hash/Basefold substrate (the integer counterpart of the fp8 gadget files).
//
// Semantics: fixed-point llama block at residual scale 2^16 (zkob-style); the
// normative reference is int_layer_ref.py.  Every gadget runs on a SHARED
// Fiat-Shamir transcript + p3lu::XCtx ledger (composition per p3_transformer),
// with a standalone fallback (own lookup flush + batched openings) for unit
// tests, exactly like p3_quant.cuh.
//
// Design highlights (INT_LAYER_LOG.md session 1):
//  * p3irs  -- rescale y = floor((x + 2^(sf-1)) / 2^sf): one eq-weighted
//    zero-check + 16-bit limb range lookups on the remainder + a signed range
//    lookup on y.  This is the integer analogue of the fp8 `qnt` seam op.
//  * p3imm  -- integer matmul WITHOUT product-domain commitments: one hiding
//    claim on the committed accumulator column Y at (z||zex), then ONE cubic
//    sumcheck sum_u EQ(u)*Xb(u)*Wb(u) over the (k,j,i,ex) product domain where
//    Xb/Wb are VIRTUAL broadcasts of the committed operands (index maps only;
//    terminal claims land on the operand commitments at points with random
//    ex-coordinates -> hiding, per the bc_aug/expt contract).  zk: Y's mask
//    slice 1 is LINKED = matmul of the operands' mask slice-1s so the claim
//    algebra holds slice-by-slice (p3_zkc mechanism-1 seam rule).  Operand
//    index maps make head slices / transposes / concat FREE (claims at
//    partially fixed points on the producer commitments).
//  * p3irms / p3ismx / p3irope / p3iswg / p3iadd -- see each section.
//
// Range/no-wrap ledger: activations/weights signed <2^19 (RS20 lookups),
// accumulators <2^48, every advice column lookup-bounded before it enters a
// product; all magnitudes << p ~ 2^64, so in-field identities are integer
// identities.
#pragma once
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
#include "p3_hawkeye.cuh"
#include "fs_transcript.hpp"

namespace p3ig {

using p3lu::Col; using p3lu::Table; using p3lu::make_table; using p3lu::commit_col_nc;
using p3lu::chal; using p3lu::chal_vec; using p3lu::ilog2;
using p3lu::LuColSpec; using p3lu::LC;
using p3hwl::Msg5; using p3hwl::CFn; using p3hwl::sc5_verify;
using p3hwl::claimc; using p3hwl::claimv; using p3hwl::beq;
using Hash = p3fri::Hash;

// ---------------- signed helpers ----------------
static inline gl_t gsig(int64_t x) { return x >= 0 ? (gl_t)x : gl_sub(0ULL, (gl_t)(-x)); }
static inline int64_t sig64(gl_t v) {
    return v > (GL_P >> 1) ? (int64_t)v - (int64_t)GL_P : (int64_t)v;
}

// ---------------- fixed-point layer constants ----------------
static const uint32_t SX = 16;                  // residual / activation scale bits
static const int64_t  ABIT = 19;                // |activation|, |weight| < 2^ABIT
static const int64_t  ABND = 1LL << ABIT;       // RS20 window [-2^19, 2^19)
static const int64_t  CEPS = 1LL << 14;         // rmsnorm eps term (M >= 2^14)
static const uint32_t ZBIT = 15;                // |score| < 2^15 (RS16 window)
static const uint32_t EXPB = 17;                // EXPT rows = 2^17
static const uint32_t SILB = 20;                // SILU rows = 2^20 (x in +-2^19)
static const uint32_t E4ROWS = 32, E4MAX = 17;  // EXP4 rows; e capped at 17

// ---------------- tables ----------------
struct Tables {
    Table R16;                  // [0, 2^16) unsigned (limbs / rescale rems / m)
    Table R14;                  // [0, 2^14) (rope rescale rem)
    Table RTOP;                 // [0, 2^btop) score-rescale top limb
    Table RS16;                 // signed [-2^15, 2^15) (scores z, rowmax mx)
    Table RS20;                 // signed [-2^19, 2^19) (activations, weights)
    Table EXPT;                 // (t, E=round(2^16 exp(-t/2^8))), 2^17 rows
    Table SILU;                 // (x signed, round(2^16 silu(x/2^16))), 2^20 rows
    Table ISQ;                  // (m, round(2^32 sqrt(d)/sqrt(m))), 2^16 rows
    Table EXP4;                 // (4^e, 2^e, floor(2^e/2)), 32 rows, e capped 17
    uint32_t btop = 0;          // RTOP width = sf_scores - 16
};
static inline Table range_table(uint32_t b, int64_t off) {
    std::vector<gl_t> v((size_t)1 << b);
    for (size_t j = 0; j < v.size(); j++) v[j] = gsig((int64_t)j + off);
    return make_table({v});
}
// exp/silu/isq entries are computed in double once here; int_layer_ref.py
// consumes the dumped tables (dump_tables) so both sides are bit-identical.
static inline Tables build_tables(uint32_t d, uint32_t btop) {
    Tables T; T.btop = btop;
    T.R16 = range_table(16, 0);
    T.R14 = range_table(14, 0);
    T.RTOP = range_table(btop, 0);
    T.RS16 = range_table(16, -(1LL << 15));
    T.RS20 = range_table(20, -(1LL << 19));
    {   std::vector<gl_t> t((size_t)1 << EXPB), e((size_t)1 << EXPB);
        for (size_t j = 0; j < t.size(); j++) {
            t[j] = j;
            e[j] = j < 65536 ? (gl_t)llround(65536.0 * exp(-(double)j / 256.0)) : 0;
        }
        T.EXPT = make_table({t, e}); }
    {   std::vector<gl_t> x((size_t)1 << SILB), s((size_t)1 << SILB);
        for (size_t j = 0; j < x.size(); j++) {
            int64_t xi = (int64_t)j - (1LL << 19);
            double xr = (double)xi / 65536.0;
            x[j] = gsig(xi);
            s[j] = gsig(llround(65536.0 * xr / (1.0 + exp(-xr))));
        }
        T.SILU = make_table({x, s}); }
    {   std::vector<gl_t> m(65536), r(65536);
        for (size_t j = 0; j < 65536; j++) {
            m[j] = j;
            r[j] = j ? (gl_t)llround(4294967296.0 * sqrt((double)d / (double)j)) : 0;
        }
        T.ISQ = make_table({m, r}); }
    {   std::vector<gl_t> p4(E4ROWS), p2(E4ROWS), h2(E4ROWS);
        for (uint32_t e = 0; e < E4ROWS; e++) {
            uint32_t er = e < E4MAX ? e : E4MAX;
            p4[e] = 1ULL << (2 * er); p2[e] = 1ULL << er;
            h2[e] = er ? (1ULL << (er - 1)) : 0;
        }
        T.EXP4 = make_table({p4, p2, h2}); }
    return T;
}
// dump exp/silu/isq (+ dims) for int_layer_ref.py (bitwise-shared semantics)
static inline bool dump_tables(const char* path, const Tables& T, uint32_t d) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    int64_t hdr[5] = {0x494E5454, d, (int64_t)T.EXPT.cols[0].size(),
                      (int64_t)T.SILU.cols[0].size(), (int64_t)T.ISQ.cols[0].size()};
    fwrite(hdr, 8, 5, f);
    fwrite(T.EXPT.cols[1].data(), 8, T.EXPT.cols[1].size(), f);
    for (auto v : T.SILU.cols[1]) { int64_t s = sig64(v); fwrite(&s, 8, 1, f); }
    fwrite(T.ISQ.cols[1].data(), 8, T.ISQ.cols[1].size(), f);
    fclose(f);
    return true;
}

// ---------------- row-sum sumcheck helpers (the qnt SEL/FSEL pattern) ----------------
// point of the row-eq weight: (zb || [zex] || 0...) over lb + e_of(le) vars;
// hide=true (xpt-style) weights mask slice 1 by zex -- the caller must have
// LINKED the slice-1 masks so the claim algebra holds slice-by-slice.
static inline std::vector<gl_t> ptb_pt(const std::vector<gl_t>& zb, gl_t zex,
                                       bool hide, uint32_t le) {
    std::vector<gl_t> p = zb;
    if (p3zkc::G.on && hide) p.push_back(zex);
    p.resize(zb.size() + p3zkc::e_of(le), 0);
    return p;
}
// EQb over the element AUG domain: EQb[e | ex<<le] = eq(ptb)[(ex<<lb) | row(e)]
static inline std::vector<gl_t> eqb_col(const std::vector<gl_t>& ptb, uint32_t lb,
                                        uint32_t lcols, size_t NeA) {
    std::vector<gl_t> eqb = beq(ptb);
    const uint32_t le = lcols + lb;
    const size_t Ne = (size_t)1 << le;
    std::vector<gl_t> EQb(NeA);
    for (size_t q = 0; q < NeA; q++) {
        size_t ex = q >> le, e = q & (Ne - 1);
        EQb[q] = eqb[(ex << lb) | (e >> lcols)];
    }
    return EQb;
}
// verifier terminal weight of the row-eq column at the sumcheck point rS
static inline gl_t rowsum_w(const std::vector<gl_t>& rS, uint32_t lcols, uint32_t le,
                            const std::vector<gl_t>& ptb) {
    std::vector<gl_t> rSb(rS.begin() + lcols, rS.begin() + le);
    rSb.insert(rSb.end(), rS.begin() + le, rS.end());
    return p3bf::eq_point(rSb, ptb);
}
// row sums of a mask-slice-1 array (mk_linked cons1 builders)
static inline std::vector<gl_t> m1_rowsum(const std::vector<gl_t>& el1, uint32_t lcols,
                                          uint32_t lb, bool square, gl_t add0 = 0) {
    std::vector<gl_t> out((size_t)1 << lb, add0);
    for (size_t e = 0; e < el1.size(); e++) {
        gl_t v = square ? gl_mul(el1[e], el1[e]) : el1[e];
        out[e >> lcols] = gl_add(out[e >> lcols], v);
    }
    return out;
}

// standalone-mode tail (unit tests): one lookup flush + per-class batches
struct Tail { std::vector<p3lu::GroupProof> lug; std::vector<p3bo::BatchProof> batches; };
static inline void tail_prove(fs::Transcript& tr, p3lu::XCtx& xc, uint32_t R, uint32_t Q,
                              bool strict, Tail& t, const char* tag) {
    t.lug = p3lu::lu_flush(tr, xc, R, Q, strict);
    for (size_t i = 0; i < xc.lg.cls.size(); i++)
        t.batches.push_back(p3bo::prove_class(tr, xc.lg.cls[i], R, Q,
                                              std::string(tag) + std::to_string(i),
                                              &xc.lg.resolve, &xc.lg.dresolve));
}
static inline bool tail_verify(fs::Transcript& tr, p3lu::VCtx& vc, const Tail& t,
                               uint32_t Q, uint32_t R, const char* tag, const char** why) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (!p3lu::lu_verify_flush(tr, vc, t.lug, Q, R, why)) return false;
    if (t.batches.size() != vc.vlg.cls.size()) return fail("batch count");
    for (size_t i = 0; i < vc.vlg.cls.size(); i++)
        if (!p3bo::verify_class(tr, vc.vlg.cls[i], t.batches[i], Q, R,
                                std::string(tag) + std::to_string(i), why)) return false;
    return true;
}

} // namespace p3ig

// ==========================================================================
// p3irs -- RESCALE gadget: committed ACC (scale 2^(16+sf-16)... generic) and
// committed Y with  ACC + 2^(sf-1) == Y*2^sf + rem,  rem in [0,2^sf) via
// 16-bit limbs (top limb against an exact-width table), plus a signed range
// lookup on Y.  This is the after-matmul requantize (the integer `qnt`).
// ==========================================================================
namespace p3irs {

using namespace p3ig;

enum { RNG_NONE = 0, RNG_S20, RNG_S16 };        // y range window
enum { IT_NONE = 0,
       IT_SHIFT,      // y+1 / rem-2^sf: zero-check holds, rem lookup must reject
       IT_LIMB,       // limb flipped: zero-check must reject
       IT_RANGE };    // y out of window, acc adjusted: range lookup must reject
struct Tamper { int mode = IT_NONE; size_t i = 0; };

struct Wit {
    uint32_t le = 0, sf = 0, nl = 0, rng = RNG_S20;
    std::vector<gl_t> acc, y;                    // element grids (2^le)
    std::vector<gl_t> L[2];                      // limb columns
    std::vector<uint32_t> idxL[2], idxY;         // lookup indices
};

// canonical replay: acc holds signed ints (as field elements), |acc| < 2^47
static inline Wit gen_witness(uint32_t le, uint32_t sf, int rng,
                              const std::vector<gl_t>& acc, const Tamper* tm = nullptr) {
    Wit w; w.le = le; w.sf = sf; w.rng = rng;
    w.nl = sf <= 16 ? 1 : 2;
    size_t N = (size_t)1 << le;
    if (acc.size() != N) throw std::runtime_error("irs: acc size");
    w.acc = acc;
    w.y.assign(N, 0);
    for (uint32_t l = 0; l < w.nl; l++) { w.L[l].assign(N, 0); w.idxL[l].assign(N, 0); }
    w.idxY.assign(N, 0);
    const int64_t H = 1LL << (sf - 1);
    const int64_t win = rng == RNG_S20 ? (1LL << 19) : (1LL << 15);
    for (size_t i = 0; i < N; i++) {
        int64_t a = sig64(acc[i]);
        int64_t yv = (a + H) >> sf;              // floor((a+H)/2^sf), a > -2^62
        int64_t rem = (a + H) - (yv << sf);
        if (tm && tm->mode == IT_SHIFT && i == tm->i) { yv += 1; rem -= 1LL << sf; }
        if (tm && tm->mode == IT_RANGE && i == tm->i) {
            int64_t bad = win + 3;               // out of window, acc made consistent
            w.acc[i] = gsig((bad << sf) + 5 - H);
            yv = bad; rem = 5;
        }
        if (rng != RNG_NONE && (yv < -win || yv >= win) && !(tm && tm->mode == IT_RANGE && i == tm->i))
            throw std::runtime_error("irs: y out of range (data too large)");
        w.y[i] = gsig(yv);
        if (rem >= 0) {
            int64_t r0 = rem & 0xFFFF, r1 = rem >> 16;
            w.L[0][i] = (gl_t)r0; w.idxL[0][i] = (uint32_t)r0;
            if (w.nl == 2) { w.L[1][i] = (gl_t)r1; w.idxL[1][i] = (uint32_t)r1; }
        } else {
            // IT_SHIFT forge: keep the zero-check satisfied (L0 = rem in-field),
            // the limb LOOKUP must reject (value is not a table row)
            w.L[0][i] = gsig(rem);
            w.idxL[0][i] = (uint32_t)((uint64_t)rem & 0xFFFF);
            if (w.nl == 2) { w.L[1][i] = 0; w.idxL[1][i] = 0; }
        }
        if (tm && tm->mode == IT_LIMB && i == tm->i)
            w.L[0][i] = gl_add(w.L[0][i], 1ULL);
        w.idxY[i] = (uint32_t)(yv + win);
    }
    return w;
}

struct Operands { const Col* X; const Col* Y; };  // persistent, caller-owned

struct Proof {
    uint32_t le = 0, sf = 0, nl = 0, rng = RNG_S20;
    Hash rL[2];
    std::vector<Msg5> m;
    gl_t yX = 0, yY = 0, yL[2] = {0, 0};
    p3zkc::Blind bl;
    Tail tail;                                    // standalone only
};

static inline const Table& limb_tab(const Tables& T, uint32_t sf, uint32_t l) {
    if (l == 0) return sf == 14 ? T.R14 : T.R16;
    return sf == 32 ? T.R16 : T.RTOP;            // top limb: full 16 bits at sf=32,
}                                                // exact width (RTOP) for scores

static inline Proof prove(fs::Transcript& tr, const Wit& w, const Tables& T,
                          const Operands& ops, uint32_t R, uint32_t Q,
                          bool strict = true, p3lu::XCtx* xc = nullptr,
                          const char* tag = "irs") {
    Proof pf; pf.le = w.le; pf.sf = w.sf; pf.nl = w.nl; pf.rng = w.rng;
    uint32_t hdr[4] = {w.le, w.sf, w.nl, (uint32_t)w.rng};
    tr.absorb("irs-dims", hdr, sizeof hdr);
    tr.absorb("irs-X", ops.X->root.data(), 32);
    tr.absorb("irs-Y", ops.Y->root.data(), 32);

    p3lu::XCtx xloc;
    p3lu::XCtx& XC = xc ? *xc : xloc;
    std::vector<Col>& CL = XC.vec(w.nl);
    for (uint32_t l = 0; l < w.nl; l++) {
        CL[l] = commit_col_nc(w.L[l], R);
        pf.rL[l] = CL[l].root;
        tr.absorb("irs-L", pf.rL[l].data(), 32);
    }
    for (uint32_t l = 0; l < w.nl; l++)
        p3lu::defer_v(XC, {LC(&CL[l])}, w.idxL[l], limb_tab(T, w.sf, l),
                      std::string(tag) + "L" + std::to_string(l));
    if (w.rng != RNG_NONE)
        p3lu::defer_v(XC, {LC(ops.Y)}, w.idxY,
                      w.rng == RNG_S20 ? T.RS20 : T.RS16, std::string(tag) + "Y");

    // zero-check: eq(z,.)*(acc + H - y*2^sf - L0 - 2^16 L1) == 0
    std::vector<gl_t> zE = chal_vec(tr, w.le);
    const gl_t H = 1ULL << (w.sf - 1), P2 = 1ULL << w.sf;
    const uint32_t nl = w.nl;
    {
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(beq(p3zkc::zpt(zE)));
        cols.push_back(ops.X->v);
        cols.push_back(ops.Y->v);
        for (uint32_t l = 0; l < nl; l++) cols.push_back(CL[l].v);
        CFn F = [H, P2, nl](const gl_t* v) {
            gl_t t = gl_sub(gl_add(v[1], H), gl_mul(v[2], P2));
            t = gl_sub(t, v[3]);
            if (nl == 2) t = gl_sub(t, gl_mul(v[4], 65536ULL));
            return gl_mul(v[0], t);
        };
        std::vector<gl_t> rE = p3hwl::sc5z(tr, "irs-sc", w.le, std::move(cols), F, pf.m,
                                           0, R, XC.lg, XC.keep, pf.bl);
        pf.yX = claimc(tr, XC.lg, *ops.X, rE);
        pf.yY = claimc(tr, XC.lg, *ops.Y, rE);
        for (uint32_t l = 0; l < nl; l++) pf.yL[l] = claimc(tr, XC.lg, CL[l], rE);
    }
    if (!xc) tail_prove(tr, XC, R, Q, strict, pf.tail, "irs-bo");
    return pf;
}

static inline bool verify(fs::Transcript& tr, const Tables& T, const Proof& pf,
                          const Hash& rX, const Hash& rY, uint32_t le, uint32_t sf,
                          int rng, uint32_t Q_pub, uint32_t R_pub,
                          const char** why = nullptr, p3lu::VCtx* xv = nullptr,
                          const char* tag = "irs") {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.le != le || pf.sf != sf || pf.rng != (uint32_t)rng) return fail("irs dims");
    uint32_t nl = sf <= 16 ? 1u : 2u;
    if (pf.nl != nl) return fail("irs nl");
    uint32_t hdr[4] = {le, sf, nl, (uint32_t)rng};
    tr.absorb("irs-dims", hdr, sizeof hdr);
    tr.absorb("irs-X", rX.data(), 32);
    tr.absorb("irs-Y", rY.data(), 32);
    p3lu::VCtx vloc;
    p3lu::VCtx& VC = xv ? *xv : vloc;
    for (uint32_t l = 0; l < nl; l++) tr.absorb("irs-L", pf.rL[l].data(), 32);
    for (uint32_t l = 0; l < nl; l++)
        p3lu::vdefer_v(VC, {&pf.rL[l]}, limb_tab(T, sf, l), le,
                       std::string(tag) + "L" + std::to_string(l));
    if (rng != RNG_NONE)
        p3lu::vdefer_v(VC, {&rY}, rng == RNG_S20 ? T.RS20 : T.RS16, le,
                       std::string(tag) + "Y");

    std::vector<gl_t> zE = chal_vec(tr, le);
    const gl_t H = 1ULL << (sf - 1), P2 = 1ULL << sf;
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl);
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.m, p3zkc::vfull(le), gl_mul(rho, pf.bl.H),
                        tr, "irs-sc", rE, claim)) return fail("irs sumcheck");
        if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl, rE)) return fail("irs blind ip");
        gl_t w0 = p3bf::eq_point(rE, p3zkc::zpt(zE));
        gl_t yX = claimv(tr, VC.vlg, rX, rE, pf.yX);
        gl_t yY = claimv(tr, VC.vlg, rY, rE, pf.yY);
        gl_t yL0 = claimv(tr, VC.vlg, pf.rL[0], rE, pf.yL[0]);
        gl_t yL1 = nl == 2 ? claimv(tr, VC.vlg, pf.rL[1], rE, pf.yL[1]) : 0;
        gl_t t = gl_sub(gl_add(yX, H), gl_mul(yY, P2));
        t = gl_sub(t, yL0);
        if (nl == 2) t = gl_sub(t, gl_mul(yL1, 65536ULL));
        gl_t end = gl_add(gl_mul(w0, t), p3hwl::sc5_blindterm(pf.bl, rho, w0));
        if (end != claim) return fail("irs terminal");
    }
    if (!xv) { if (!tail_verify(tr, VC, pf.tail, Q_pub, R_pub, "irs-bo", why)) return false; }
    else if (!pf.tail.lug.empty() || !pf.tail.batches.empty()) return fail("unexpected tail");
    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const Proof& pf) {
    size_t s = 16 + pf.nl * 32 + pf.m.size() * 40 + 8 * (2 + pf.nl);
    for (auto& g : pf.tail.lug) s += p3lu::sz_group(g);
    for (auto& b : pf.tail.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3irs

// ==========================================================================
// p3imm -- INTEGER MATMUL: Y(i,k) = sum_j X(i,j) W(j,k) on committed columns,
// with operand INDEX MAPS (bit templates) so X/W/Y may live inside larger
// shared commitments (head slices, transposes, per-instance blocks).
// ==========================================================================
namespace p3imm {

using namespace p3ig;

// bit template: for each real variable of the committed column, where its
// value comes from: the contraction index j, the free index a (= i for X /
// k for W / one of (k,i) for Y), or a fixed constant bit.
enum { S_J = 0, S_A = 1, S_I = 2, S_C = 3 };
struct Sel { uint8_t src; uint8_t bit; };
// operand view of a committed column
struct OpView {
    const Col* c = nullptr;      // prover side
    Hash root;                   // verifier side
    uint32_t v = 0;              // real var count of the commitment
    std::vector<Sel> sel;        // size v
};
// direct layouts: X grid [j low | i high], W grid [k low | j high] (row-major
// (j,k)), Y grid [k low | i high]
static inline OpView direct_x(const Col* c, uint32_t lj, uint32_t li) {
    OpView o; o.c = c; if (c) o.root = c->root; o.v = lj + li;
    for (uint32_t b = 0; b < lj; b++) o.sel.push_back({S_J, (uint8_t)b});
    for (uint32_t b = 0; b < li; b++) o.sel.push_back({S_A, (uint8_t)b});
    return o;
}
static inline OpView direct_w(const Col* c, uint32_t lj, uint32_t lk) {
    OpView o; o.c = c; if (c) o.root = c->root; o.v = lj + lk;
    for (uint32_t b = 0; b < lk; b++) o.sel.push_back({S_A, (uint8_t)b});
    for (uint32_t b = 0; b < lj; b++) o.sel.push_back({S_J, (uint8_t)b});
    return o;
}
// Y view: sel over sources S_A (= k bits) and S_I (= i bits) and S_C consts
static inline OpView direct_y(const Col* c, uint32_t lk, uint32_t li) {
    OpView o; o.c = c; if (c) o.root = c->root; o.v = lk + li;
    for (uint32_t b = 0; b < lk; b++) o.sel.push_back({S_A, (uint8_t)b});
    for (uint32_t b = 0; b < li; b++) o.sel.push_back({S_I, (uint8_t)b});
    return o;
}

// operand offset: (j, a); Y offset: (k, i)
static inline size_t op_off(const OpView& o, uint32_t j, uint32_t a) {
    size_t idx = 0;
    for (uint32_t t = 0; t < o.v; t++) {
        const Sel& s = o.sel[t];
        uint32_t bit = s.src == S_J ? ((j >> s.bit) & 1)
                     : s.src == S_A ? ((a >> s.bit) & 1)
                     : s.src == S_C ? s.bit : 0;
        idx |= (size_t)bit << t;
    }
    return idx;
}
static inline size_t y_off(const OpView& o, uint32_t k, uint32_t i) {
    size_t idx = 0;
    for (uint32_t t = 0; t < o.v; t++) {
        const Sel& s = o.sel[t];
        uint32_t bit = s.src == S_A ? ((k >> s.bit) & 1)
                     : s.src == S_I ? ((i >> s.bit) & 1)
                     : s.src == S_C ? s.bit : 0;
        idx |= (size_t)bit << t;
    }
    return idx;
}
// claim point of an operand view: real coords from (rj, ra), then e_of(v)
// ex coords copied from rfull[src_v..] (truncate / zero-pad; expt contract)
static inline std::vector<gl_t> op_pt(const OpView& o, const std::vector<gl_t>& rj,
                                      const std::vector<gl_t>& ra,
                                      const std::vector<gl_t>& rfull, uint32_t src_v) {
    std::vector<gl_t> p(o.v);
    for (uint32_t t = 0; t < o.v; t++) {
        const Sel& s = o.sel[t];
        p[t] = s.src == S_J ? rj[s.bit]
             : s.src == S_A ? ra[s.bit]
             : (gl_t)(s.src == S_C ? s.bit : 0);
    }
    return p3zkc::expt(p, rfull, src_v);
}
static inline std::vector<gl_t> y_pt(const OpView& o, const std::vector<gl_t>& rk,
                                     const std::vector<gl_t>& ri) {
    std::vector<gl_t> p(o.v);
    for (uint32_t t = 0; t < o.v; t++) {
        const Sel& s = o.sel[t];
        p[t] = s.src == S_A ? rk[s.bit]
             : s.src == S_I ? ri[s.bit]
             : (gl_t)(s.src == S_C ? s.bit : 0);
    }
    return p;
}

// honest Y grid values (also used for witness chaining): out[k | i<<lk]
static inline std::vector<gl_t> compute_y(const std::vector<gl_t>& X, const OpView& xv,
                                          const std::vector<gl_t>& W, const OpView& wv,
                                          uint32_t lj, uint32_t lk, uint32_t li) {
    const uint32_t J = 1u << lj, K = 1u << lk, I = 1u << li;
    std::vector<gl_t> Y((size_t)K * I, 0);
    #pragma omp parallel for schedule(static) if ((size_t)K * I >= 4096)
    for (int64_t i = 0; i < (int64_t)I; i++)
        for (uint32_t k = 0; k < K; k++) {
            gl_t acc = 0;
            for (uint32_t j = 0; j < J; j++)
                acc = gl_add(acc, gl_mul(X[op_off(xv, j, (uint32_t)i)],
                                         W[op_off(wv, j, k)]));
            Y[k | ((size_t)i << lk)] = acc;
        }
    return Y;
}

// zk mask linkage: accumulate the matmul of the operands' mask slice-1s into
// the Y view's region of m1 (the caller commits Y with mk_linked(vY, m1)).
static inline void accum_mask1(std::vector<gl_t>& m1, const OpView& xv, const OpView& wv,
                               const OpView& yv, uint32_t lj, uint32_t lk, uint32_t li) {
    if (!p3zkc::G.on) return;
    std::vector<gl_t> xm = p3zkc::slice1(xv.c->v, xv.v);
    std::vector<gl_t> wm = p3zkc::slice1(wv.c->v, wv.v);
    const uint32_t J = 1u << lj, K = 1u << lk, I = 1u << li;
    #pragma omp parallel for schedule(static) if ((size_t)K * I >= 4096)
    for (int64_t i = 0; i < (int64_t)I; i++)
        for (uint32_t k = 0; k < K; k++) {
            gl_t acc = 0;
            for (uint32_t j = 0; j < J; j++)
                acc = gl_add(acc, gl_mul(xm[op_off(xv, j, (uint32_t)i)],
                                         wm[op_off(wv, j, k)]));
            m1[y_off(yv, k, (uint32_t)i)] = acc;
        }
}

struct Proof {
    uint32_t lj = 0, lk = 0, li = 0;
    std::vector<Msg5> m;
    gl_t yY = 0, yX = 0, yW = 0;
    p3zkc::Blind bl;
    Tail tail;
};

// device functor of the cubic matmul summand EQ * Xb * Wb (GPU dispatch via
// p3hwl::sc5rz; identical messages/transcript to the host chain)
struct FImmGpu {
    static __device__ gl_t eval(const gl_t* c, const gl_t* p) {
        (void)p;
        return gl_mul(c[0], gl_mul(c[1], c[2]));
    }
};

static inline Proof prove(fs::Transcript& tr, const OpView& xv, const OpView& wv,
                          const OpView& yv, uint32_t lj, uint32_t lk, uint32_t li,
                          uint32_t R, uint32_t Q, bool strict = true,
                          p3lu::XCtx* xc = nullptr) {
    Proof pf; pf.lj = lj; pf.lk = lk; pf.li = li;
    uint32_t hdr[3] = {lj, lk, li};
    tr.absorb("imm-dims", hdr, sizeof hdr);
    tr.absorb("imm-X", xv.c->root.data(), 32);
    tr.absorb("imm-W", wv.c->root.data(), 32);
    tr.absorb("imm-Y", yv.c->root.data(), 32);

    p3lu::XCtx xloc;
    p3lu::XCtx& XC = xc ? *xc : xloc;
    const bool zk = p3zkc::G.on;

    // hiding claim on the committed accumulator at (z || zex)
    std::vector<gl_t> z = chal_vec(tr, lk + li);
    std::vector<gl_t> zk_(z.begin(), z.begin() + lk), zi_(z.begin() + lk, z.end());
    gl_t zex = zk ? chal(tr) : 0;
    pf.yY = claimc(tr, XC.lg, *yv.c, p3zkc::xpt(y_pt(yv, zk_, zi_), zex));

    // cubic sumcheck over the (j,k,i,ex) product domain of EQ * Xb * Wb
    const uint32_t lt = lj + lk + li, e_p = p3zkc::e_of(lt);
    const size_t N = (size_t)1 << (lt + e_p);
    const uint32_t eX = p3zkc::e_of(xv.v), eW = p3zkc::e_of(wv.v);
    std::vector<gl_t> ptE = z;
    if (zk) { ptE.push_back(zex); ptE.resize(lk + li + e_p, 0); }
    std::vector<gl_t> eqv = beq(ptE);
    std::vector<std::vector<gl_t>> cols(3);
    cols[0].resize(N); cols[1].resize(N); cols[2].resize(N);
    {
        const gl_t* xa = xv.c->v.data();
        const gl_t* wa = wv.c->v.data();
        const uint32_t J = 1u << lj, K = 1u << lk;
        const size_t NX = (size_t)1 << xv.v, NW = (size_t)1 << wv.v;
        #pragma omp parallel for schedule(static) if (N >= 65536)
        for (int64_t u = 0; u < (int64_t)N; u++) {
            uint32_t j = (uint32_t)(u & (J - 1));
            uint32_t k = (uint32_t)((u >> lj) & (K - 1));
            uint32_t i = (uint32_t)((u >> (lj + lk)) & ((1u << li) - 1));
            uint32_t ex = (uint32_t)(u >> lt);
            uint32_t exx = eX >= e_p ? ex : (ex & ((1u << eX) - 1));
            uint32_t exw = eW >= e_p ? ex : (ex & ((1u << eW) - 1));
            cols[0][u] = eqv[k | ((size_t)i << lk) | ((size_t)ex << (lk + li))];
            cols[1][u] = xa[op_off(xv, j, i) | ((size_t)exx << xv.v)];
            cols[2][u] = wa[op_off(wv, j, k) | ((size_t)exw << wv.v)];
        }
    }
    CFn F = [](const gl_t* v) { return gl_mul(v[0], gl_mul(v[1], v[2])); };
    std::vector<gl_t> r = p3hwl::sc5rz<FImmGpu>(tr, "imm-sc", lt, std::move(cols),
                                                nullptr, 0, F, pf.m, pf.yY, R,
                                                XC.lg, XC.keep, pf.bl);
    std::vector<gl_t> rj(r.begin(), r.begin() + lj),
                      rk(r.begin() + lj, r.begin() + lj + lk),
                      ri(r.begin() + lj + lk, r.begin() + lt);
    pf.yX = claimc(tr, XC.lg, *xv.c, op_pt(xv, rj, ri, r, lt));
    pf.yW = claimc(tr, XC.lg, *wv.c, op_pt(wv, rj, rk, r, lt));
    if (!xc) tail_prove(tr, XC, R, Q, strict, pf.tail, "imm-bo");
    return pf;
}

// verifier: the OpViews carry roots + templates only (c may be null)
static inline bool verify(fs::Transcript& tr, const Proof& pf, const OpView& xv,
                          const OpView& wv, const OpView& yv,
                          uint32_t lj, uint32_t lk, uint32_t li,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr,
                          p3lu::VCtx* xv_ctx = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.lj != lj || pf.lk != lk || pf.li != li) return fail("imm dims");
    uint32_t hdr[3] = {lj, lk, li};
    tr.absorb("imm-dims", hdr, sizeof hdr);
    tr.absorb("imm-X", xv.root.data(), 32);
    tr.absorb("imm-W", wv.root.data(), 32);
    tr.absorb("imm-Y", yv.root.data(), 32);
    p3lu::VCtx vloc;
    p3lu::VCtx& VC = xv_ctx ? *xv_ctx : vloc;
    const bool zk = p3zkc::G.on;

    std::vector<gl_t> z = chal_vec(tr, lk + li);
    std::vector<gl_t> zk_(z.begin(), z.begin() + lk), zi_(z.begin() + lk, z.end());
    gl_t zex = zk ? chal(tr) : 0;
    gl_t yY = claimv(tr, VC.vlg, yv.root, p3zkc::xpt(y_pt(yv, zk_, zi_), zex), pf.yY);

    const uint32_t lt = lj + lk + li, e_p = p3zkc::e_of(lt);
    std::vector<gl_t> ptE = z;
    if (zk) { ptE.push_back(zex); ptE.resize(lk + li + e_p, 0); }
    gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl);
    std::vector<gl_t> r; gl_t claim;
    if (!sc5_verify(pf.m, p3zkc::vfull(lt), gl_add(yY, gl_mul(rho, pf.bl.H)),
                    tr, "imm-sc", r, claim)) return fail("imm sumcheck");
    if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl, r)) return fail("imm blind ip");
    std::vector<gl_t> rj(r.begin(), r.begin() + lj),
                      rk(r.begin() + lj, r.begin() + lj + lk),
                      ri(r.begin() + lj + lk, r.begin() + lt);
    gl_t yX = claimv(tr, VC.vlg, xv.root, op_pt(xv, rj, ri, r, lt), pf.yX);
    gl_t yW = claimv(tr, VC.vlg, wv.root, op_pt(wv, rj, rk, r, lt), pf.yW);
    std::vector<gl_t> rsel(rk);
    rsel.insert(rsel.end(), ri.begin(), ri.end());
    rsel.insert(rsel.end(), r.begin() + lt, r.end());
    gl_t w0 = p3bf::eq_point(rsel, ptE);
    gl_t end = gl_add(gl_mul(w0, gl_mul(yX, yW)), p3hwl::sc5_blindterm(pf.bl, rho, w0));
    if (end != claim) return fail("imm terminal");
    if (!xv_ctx) { if (!tail_verify(tr, VC, pf.tail, Q_pub, R_pub, "imm-bo", why)) return false; }
    else if (!pf.tail.lug.empty() || !pf.tail.batches.empty()) return fail("unexpected tail");
    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const Proof& pf) {
    size_t s = 12 + pf.m.size() * 40 + 8 * 3;
    for (auto& g : pf.tail.lug) s += p3lu::sz_group(g);
    for (auto& b : pf.tail.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3imm

// ==========================================================================
// p3iadd -- residual add: OUT = A + B elementwise (one zero-check) + signed
// range lookup on OUT.
// ==========================================================================
namespace p3iadd {

using namespace p3ig;

enum { AT_NONE = 0, AT_OUT, AT_RANGE };
struct Tamper { int mode = AT_NONE; size_t i = 0; };

struct Wit {
    uint32_t le = 0;
    std::vector<gl_t> out;
    std::vector<uint32_t> idxO;
};
static inline Wit gen_witness(uint32_t le, const std::vector<gl_t>& a,
                              const std::vector<gl_t>& b, const Tamper* tm = nullptr) {
    Wit w; w.le = le;
    size_t N = (size_t)1 << le;
    w.out.assign(N, 0); w.idxO.assign(N, 0);
    for (size_t i = 0; i < N; i++) {
        int64_t o = sig64(a[i]) + sig64(b[i]);
        if (tm && tm->mode == AT_OUT && i == tm->i) o += 1;
        if (o < -ABND || o >= ABND) {
            if (!(tm && tm->mode == AT_RANGE && i == tm->i))
                throw std::runtime_error("iadd: out of range");
        }
        w.out[i] = gsig(o);
        w.idxO[i] = (uint32_t)(o + ABND);
    }
    return w;
}
// AT_RANGE: force one honest-in-field but out-of-window pair by inflating a/b
// upstream -- unit test crafts the inputs directly.

struct Operands { const Col* A; const Col* B; const Col* OUT; };

struct Proof {
    uint32_t le = 0;
    std::vector<Msg5> m;
    gl_t yA = 0, yB = 0, yO = 0;
    p3zkc::Blind bl;
    Tail tail;
};

static inline Proof prove(fs::Transcript& tr, const Wit& w, const Tables& T,
                          const Operands& ops, uint32_t R, uint32_t Q,
                          bool strict = true, p3lu::XCtx* xc = nullptr) {
    Proof pf; pf.le = w.le;
    tr.absorb("iad-dims", &w.le, 4);
    tr.absorb("iad-A", ops.A->root.data(), 32);
    tr.absorb("iad-B", ops.B->root.data(), 32);
    tr.absorb("iad-O", ops.OUT->root.data(), 32);
    p3lu::XCtx xloc;
    p3lu::XCtx& XC = xc ? *xc : xloc;
    p3lu::defer_v(XC, {LC(ops.OUT)}, w.idxO, T.RS20, "iadO");
    std::vector<gl_t> zE = chal_vec(tr, w.le);
    {
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(beq(p3zkc::zpt(zE)));
        cols.push_back(ops.A->v); cols.push_back(ops.B->v); cols.push_back(ops.OUT->v);
        CFn F = [](const gl_t* v) {
            return gl_mul(v[0], gl_sub(gl_add(v[1], v[2]), v[3]));
        };
        std::vector<gl_t> rE = p3hwl::sc5z(tr, "iad-sc", w.le, std::move(cols), F, pf.m,
                                           0, R, XC.lg, XC.keep, pf.bl);
        pf.yA = claimc(tr, XC.lg, *ops.A, rE);
        pf.yB = claimc(tr, XC.lg, *ops.B, rE);
        pf.yO = claimc(tr, XC.lg, *ops.OUT, rE);
    }
    if (!xc) tail_prove(tr, XC, R, Q, strict, pf.tail, "iad-bo");
    return pf;
}

static inline bool verify(fs::Transcript& tr, const Tables& T, const Proof& pf,
                          const Hash& rA, const Hash& rB, const Hash& rO, uint32_t le,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr,
                          p3lu::VCtx* xv = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.le != le) return fail("iadd dims");
    tr.absorb("iad-dims", &le, 4);
    tr.absorb("iad-A", rA.data(), 32);
    tr.absorb("iad-B", rB.data(), 32);
    tr.absorb("iad-O", rO.data(), 32);
    p3lu::VCtx vloc;
    p3lu::VCtx& VC = xv ? *xv : vloc;
    p3lu::vdefer_v(VC, {&rO}, T.RS20, le, "iadO");
    std::vector<gl_t> zE = chal_vec(tr, le);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl);
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.m, p3zkc::vfull(le), gl_mul(rho, pf.bl.H),
                        tr, "iad-sc", rE, claim)) return fail("iadd sumcheck");
        if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl, rE)) return fail("iadd blind ip");
        gl_t w0 = p3bf::eq_point(rE, p3zkc::zpt(zE));
        gl_t yA = claimv(tr, VC.vlg, rA, rE, pf.yA);
        gl_t yB = claimv(tr, VC.vlg, rB, rE, pf.yB);
        gl_t yO = claimv(tr, VC.vlg, rO, rE, pf.yO);
        gl_t end = gl_add(gl_mul(w0, gl_sub(gl_add(yA, yB), yO)),
                          p3hwl::sc5_blindterm(pf.bl, rho, w0));
        if (end != claim) return fail("iadd terminal");
    }
    if (!xv) { if (!tail_verify(tr, VC, pf.tail, Q_pub, R_pub, "iad-bo", why)) return false; }
    else if (!pf.tail.lug.empty() || !pf.tail.batches.empty()) return fail("unexpected tail");
    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const Proof& pf) {
    size_t s = 4 + pf.m.size() * 40 + 24;
    for (auto& g : pf.tail.lug) s += p3lu::sz_group(g);
    for (auto& b : pf.tail.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3iadd

// ==========================================================================
// p3irms -- integer RMSNorm.  Per row i of X (T x d grid, scale 2^16):
//   M = sum_j X^2 + CEPS;  M = m*4^e + r4, m in [2^14, 2^16) (unique e);
//   R' = ISQ[m] (= round(2^32 sqrt(d)/sqrt(m)));  R = floor((R'+h2)/2^e);
//   W(i,j) = rescale(R*g(j), 16);  Y = rescale(W*X, 16).
// DEVIATION from zkob_rmsnorm (documented): table-based inverse sqrt on the
// normalized mantissa instead of the (R+-1)^2*M bracket (which needs 2^80
// integers and does not fit Goldilocks); same +-1-2 ulp tolerance class, and
// the proof is BITWISE for the reference that computes exactly this.
// ==========================================================================
namespace p3irms {

using namespace p3ig;

enum { RT_NONE = 0,
       RT_M,          // M inflated: the X^2 row-sum must reject
       RT_R,          // R+1 (r2 compensated): limb lookup must reject
       RT_W,          // W value flipped: De zero-check must reject
       RT_Y };        // Y value flipped: De zero-check must reject
struct Tamper { int mode = RT_NONE; uint32_t b = 0; size_t i = 0; };

// per-row column ids
enum { B_M = 0, B_MM, B_MS, B_P4, B_P2, B_H2, B_RP, B_L40, B_L41, B_L42,
       B_D40, B_D41, B_D42, B_R, B_L20, B_D20, B_LR0, B_LR1, NDB };
// per-element column ids (X, Y are operands; W/rem are witness)
enum { E_W = 0, E_RW, E_RY, NDE };

struct Wit {
    uint32_t lT = 0, ld = 0, le = 0;
    std::vector<gl_t> db[NDB];                    // row domain (2^lT)
    std::vector<gl_t> de[NDE];                    // element domain (2^le)
    std::vector<gl_t> y;                          // output (chain)
    std::vector<uint32_t> idxE4, idxM16, idxMS, idxISQ;         // per row
    std::vector<uint32_t> idxL[9], idxLR1;        // L40..D42,L20,D20,LR0; LR1
    std::vector<uint32_t> idxRW, idxW, idxRY, idxY;             // per element
};

static inline int64_t rshu(int64_t x, uint32_t sf) {           // round-half-up
    return (x + (1LL << (sf - 1))) >> sf;
}

static inline Wit gen_witness(uint32_t lT, uint32_t ld, const std::vector<gl_t>& x,
                              const std::vector<gl_t>& g, const Tables& T,
                              const Tamper* tm = nullptr) {
    Wit w; w.lT = lT; w.ld = ld; w.le = lT + ld;
    const uint32_t B = 1u << lT, d = 1u << ld;
    const size_t Ne = (size_t)1 << w.le;
    for (int c = 0; c < NDB; c++) w.db[c].assign(B, 0);
    for (int c = 0; c < NDE; c++) w.de[c].assign(Ne, 0);
    w.y.assign(Ne, 0);
    w.idxE4.assign(B, 0); w.idxM16.assign(B, 0); w.idxMS.assign(B, 0); w.idxISQ.assign(B, 0);
    for (int l = 0; l < 9; l++) w.idxL[l].assign(B, 0);
    w.idxLR1.assign(B, 0);
    w.idxRW.assign(Ne, 0); w.idxW.assign(Ne, 0); w.idxRY.assign(Ne, 0); w.idxY.assign(Ne, 0);
    for (uint32_t b = 0; b < B; b++) {
        int64_t M = CEPS;
        for (uint32_t j = 0; j < d; j++) {
            int64_t xv = sig64(x[((size_t)b << ld) | j]);
            M += xv * xv;
        }
        if (tm && tm->mode == RT_M && b == tm->b) M += 1;
        uint32_t e = 0;
        while ((M >> (2 * e)) >= (1LL << 16)) e++;
        int64_t m = M >> (2 * e);
        int64_t p4 = 1LL << (2 * e), p2 = 1LL << e, h2 = e ? (1LL << (e - 1)) : 0;
        int64_t r4 = M - m * p4;
        int64_t d4 = p4 - 1 - r4;
        int64_t RP = (int64_t)T.ISQ.cols[1][m];
        int64_t R = (RP + h2) >> e;
        int64_t r2 = RP + h2 - (R << e);
        int64_t d2 = p2 - 1 - r2;
        if (tm && tm->mode == RT_R && b == tm->b) { R += 1; r2 -= p2; d2 += p2; }
        w.db[B_M][b] = (gl_t)M; w.db[B_MM][b] = (gl_t)m;
        w.db[B_MS][b] = (gl_t)(m - (1LL << 14));
        w.db[B_P4][b] = (gl_t)p4; w.db[B_P2][b] = (gl_t)p2; w.db[B_H2][b] = (gl_t)h2;
        w.db[B_RP][b] = (gl_t)RP;
        w.db[B_L40][b] = (gl_t)(r4 & 0xFFFF); w.db[B_L41][b] = (gl_t)((r4 >> 16) & 0xFFFF);
        w.db[B_L42][b] = (gl_t)(r4 >> 32);
        w.db[B_D40][b] = (gl_t)(d4 & 0xFFFF); w.db[B_D41][b] = (gl_t)((d4 >> 16) & 0xFFFF);
        w.db[B_D42][b] = (gl_t)(d4 >> 32);
        w.db[B_R][b] = (gl_t)R;
        w.db[B_L20][b] = gsig(r2); w.db[B_D20][b] = gsig(d2);
        w.db[B_LR0][b] = (gl_t)(R & 0xFFFF); w.db[B_LR1][b] = (gl_t)(R >> 16);
        w.idxE4[b] = e; w.idxM16[b] = (uint32_t)m;
        w.idxMS[b] = (uint32_t)(m - (1LL << 14)); w.idxISQ[b] = (uint32_t)m;
        w.idxL[0][b] = (uint32_t)(r4 & 0xFFFF); w.idxL[1][b] = (uint32_t)((r4 >> 16) & 0xFFFF);
        w.idxL[2][b] = (uint32_t)(r4 >> 32);
        w.idxL[3][b] = (uint32_t)(d4 & 0xFFFF); w.idxL[4][b] = (uint32_t)((d4 >> 16) & 0xFFFF);
        w.idxL[5][b] = (uint32_t)(d4 >> 32);
        w.idxL[6][b] = (uint32_t)((uint64_t)r2 & 0xFFFF);
        w.idxL[7][b] = (uint32_t)((uint64_t)d2 & 0xFFFF);
        w.idxL[8][b] = (uint32_t)(R & 0xFFFF);
        w.idxLR1[b] = (uint32_t)(R >> 16);
        for (uint32_t j = 0; j < d; j++) {
            size_t ei = ((size_t)b << ld) | j;
            int64_t gv = sig64(g[j]);
            int64_t w64 = R * gv;
            int64_t Wv = rshu(w64, 16);
            int64_t remW = (w64 + (1LL << 15)) - (Wv << 16);
            if (tm && tm->mode == RT_W && b == tm->b && (size_t)j == tm->i) Wv += 1;
            if (Wv < -ABND || Wv >= ABND) throw std::runtime_error("irms: W out of range");
            int64_t xv = sig64(x[ei]);
            int64_t y64 = Wv * xv;
            int64_t Yv = rshu(y64, 16);
            int64_t remY = (y64 + (1LL << 15)) - (Yv << 16);
            if (tm && tm->mode == RT_Y && b == tm->b && (size_t)j == tm->i) Yv += 1;
            if (Yv < -ABND || Yv >= ABND) throw std::runtime_error("irms: Y out of range");
            w.de[E_W][ei] = gsig(Wv); w.de[E_RW][ei] = (gl_t)remW; w.de[E_RY][ei] = (gl_t)remY;
            w.y[ei] = gsig(Yv);
            w.idxRW[ei] = (uint32_t)remW; w.idxW[ei] = (uint32_t)(Wv + ABND);
            w.idxRY[ei] = (uint32_t)remY; w.idxY[ei] = (uint32_t)(Yv + ABND);
        }
    }
    return w;
}

struct Operands { const Col* X; const Col* G; const Col* Y; };

struct Proof {
    uint32_t lT = 0, ld = 0;
    Hash rdb[NDB], rde[NDE];
    // M row-sum binding
    gl_t yM = 0, yMX = 0;
    std::vector<Msg5> mS;
    // Db per-row zero-check
    std::vector<Msg5> mB; gl_t yB[NDB] = {};
    // De per-element zero-check
    std::vector<Msg5> mE; gl_t yE[NDE] = {}, yEX = 0, yEY = 0, yER = 0, yEG = 0;
    p3zkc::Blind bl[3];
    Tail tail;
};

static const int N_B_C = 6;

static inline Proof prove(fs::Transcript& tr, const Wit& w, const Tables& T,
                          const Operands& ops, uint32_t R, uint32_t Q,
                          bool strict = true, p3lu::XCtx* xc = nullptr,
                          const char* tag = "irms") {
    Proof pf; pf.lT = w.lT; pf.ld = w.ld;
    uint32_t hdr[2] = {w.lT, w.ld};
    tr.absorb("irm-dims", hdr, sizeof hdr);
    tr.absorb("irm-X", ops.X->root.data(), 32);
    tr.absorb("irm-G", ops.G->root.data(), 32);
    tr.absorb("irm-Y", ops.Y->root.data(), 32);
    p3lu::XCtx xloc;
    p3lu::XCtx& XC = xc ? *xc : xloc;
    const bool zk = p3zkc::G.on;
    const uint32_t lT = w.lT, ld = w.ld, le = w.le;

    std::vector<Col>& CB = XC.vec(NDB);
    std::vector<Col>& CE = XC.vec(NDE);
    // M's mask slice 1 is LINKED: CEPS + row sums of squares of X's slice 1
    std::vector<gl_t> mMv;
    if (zk) {
        std::vector<gl_t> x1 = p3zkc::slice1(ops.X->v, le);
        mMv = p3zkc::mk_linked(lT, m1_rowsum(x1, ld, lT, true, (gl_t)CEPS));
    }
    for (int c = 0; c < NDB; c++) {
        CB[c] = commit_col_nc(w.db[c], R, (zk && c == B_M) ? &mMv : nullptr);
        pf.rdb[c] = CB[c].root;
        tr.absorb("irm-cb", pf.rdb[c].data(), 32);
    }
    for (int c = 0; c < NDE; c++) {
        CE[c] = commit_col_nc(w.de[c], R);
        pf.rde[c] = CE[c].root;
        tr.absorb("irm-ce", pf.rde[c].data(), 32);
    }
    // lookups
    p3lu::defer_v(XC, {LC(&CB[B_P4]), LC(&CB[B_P2]), LC(&CB[B_H2])}, w.idxE4, T.EXP4,
                  std::string(tag) + "E4");
    p3lu::defer_v(XC, {LC(&CB[B_MM])}, w.idxM16, T.R16, std::string(tag) + "m");
    p3lu::defer_v(XC, {LC(&CB[B_MS])}, w.idxMS, T.R16, std::string(tag) + "ms");
    p3lu::defer_v(XC, {LC(&CB[B_MM]), LC(&CB[B_RP])}, w.idxISQ, T.ISQ,
                  std::string(tag) + "isq");
    static const int LCOL[9] = {B_L40, B_L41, B_L42, B_D40, B_D41, B_D42,
                                B_L20, B_D20, B_LR0};
    for (int l = 0; l < 9; l++)
        p3lu::defer_v(XC, {LC(&CB[LCOL[l]])}, w.idxL[l], T.R16,
                      std::string(tag) + "L" + std::to_string(l));
    p3lu::defer_v(XC, {LC(&CB[B_LR1])}, w.idxLR1, T.R16, std::string(tag) + "LR1");
    p3lu::defer_v(XC, {LC(&CE[E_RW])}, w.idxRW, T.R16, std::string(tag) + "rw");
    p3lu::defer_v(XC, {LC(&CE[E_W])}, w.idxW, T.RS20, std::string(tag) + "W");
    p3lu::defer_v(XC, {LC(&CE[E_RY])}, w.idxRY, T.R16, std::string(tag) + "ry");
    p3lu::defer_v(XC, {LC(ops.Y)}, w.idxY, T.RS20, std::string(tag) + "Y");

    // -- M row-sum binding: sum_j X^2 = M - CEPS --
    {
        std::vector<gl_t> zb = chal_vec(tr, lT);
        gl_t zex = zk ? chal(tr) : 0;
        pf.yM = claimc(tr, XC.lg, CB[B_M], p3zkc::xpt(zb, zex));
        std::vector<gl_t> ptb = ptb_pt(zb, zex, true, le);
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(eqb_col(ptb, lT, ld, ops.X->v.size()));
        cols.push_back(ops.X->v);
        CFn F = [](const gl_t* v) { return gl_mul(v[0], gl_mul(v[1], v[1])); };
        gl_t base0 = gl_sub(pf.yM, (gl_t)CEPS);
        std::vector<gl_t> rS = p3hwl::sc5z(tr, "irm-scS", le, std::move(cols), F, pf.mS,
                                           base0, R, XC.lg, XC.keep, pf.bl[0]);
        pf.yMX = claimc(tr, XC.lg, *ops.X, rS);
    }

    // -- Db per-row zero-check --
    std::vector<gl_t> zB = chal_vec(tr, lT);
    gl_t lamB = chal(tr), lamBv[N_B_C]; lamBv[0] = 1;
    for (int j = 1; j < N_B_C; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(beq(p3zkc::zpt(zB)));
        for (int c = 0; c < NDB; c++) cols.push_back(CB[c].v);
        CFn F = [&lamBv](const gl_t* v) {
            const gl_t* c = v + 1;
            gl_t r4 = gl_add(c[B_L40], gl_add(gl_mul(c[B_L41], 65536ULL),
                                              gl_mul(c[B_L42], 4294967296ULL)));
            gl_t d4 = gl_add(c[B_D40], gl_add(gl_mul(c[B_D41], 65536ULL),
                                              gl_mul(c[B_D42], 4294967296ULL)));
            gl_t r[N_B_C];
            r[0] = gl_sub(c[B_M], gl_add(gl_mul(c[B_MM], c[B_P4]), r4));
            r[1] = gl_sub(gl_sub(gl_sub(c[B_P4], 1ULL), r4), d4);
            r[2] = gl_sub(gl_add(gl_mul(c[B_R], c[B_P2]), c[B_L20]),
                          gl_add(c[B_RP], c[B_H2]));
            r[3] = gl_sub(gl_sub(gl_sub(c[B_P2], 1ULL), c[B_L20]), c[B_D20]);
            r[4] = gl_sub(c[B_R], gl_add(c[B_LR0], gl_mul(c[B_LR1], 65536ULL)));
            r[5] = gl_sub(gl_add(c[B_MS], 16384ULL), c[B_MM]);
            gl_t s = 0;
            for (int j = 0; j < N_B_C; j++) s = gl_add(s, gl_mul(lamBv[j], r[j]));
            return gl_mul(v[0], s);
        };
        std::vector<gl_t> rB = p3hwl::sc5z(tr, "irm-scB", lT, std::move(cols), F, pf.mB,
                                           0, R, XC.lg, XC.keep, pf.bl[1]);
        for (int c = 0; c < NDB; c++) pf.yB[c] = claimc(tr, XC.lg, CB[c], rB);
    }

    // -- De per-element zero-check: W-fold + Y-fold --
    std::vector<gl_t> zE = chal_vec(tr, le);
    gl_t lamE = chal(tr);
    {
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(beq(p3zkc::zpt(zE)));
        cols.push_back(ops.X->v);
        cols.push_back(ops.Y->v);
        for (int c = 0; c < NDE; c++) cols.push_back(CE[c].v);
        cols.push_back(p3zkc::bc_aug(CB[B_R].v, lT, le, (size_t)1 << le,
                                     [ld](size_t e) { return e >> ld; }));
        cols.push_back(p3zkc::bc_aug(ops.G->v, ld, le, (size_t)1 << le,
                                     [ld](size_t e) { return e & (((size_t)1 << ld) - 1); }));
        CFn F = [lamE](const gl_t* v) {
            gl_t X = v[1], Y = v[2], W = v[3], RW = v[4], RY = v[5], Rb = v[6], Gb = v[7];
            gl_t r0 = gl_sub(gl_add(gl_mul(Rb, Gb), 32768ULL),
                             gl_add(gl_mul(W, 65536ULL), RW));
            gl_t r1 = gl_sub(gl_add(gl_mul(W, X), 32768ULL),
                             gl_add(gl_mul(Y, 65536ULL), RY));
            return gl_mul(v[0], gl_add(r0, gl_mul(lamE, r1)));
        };
        std::vector<gl_t> rE = p3hwl::sc5z(tr, "irm-scE", le, std::move(cols), F, pf.mE,
                                           0, R, XC.lg, XC.keep, pf.bl[2]);
        pf.yEX = claimc(tr, XC.lg, *ops.X, rE);
        pf.yEY = claimc(tr, XC.lg, *ops.Y, rE);
        for (int c = 0; c < NDE; c++) pf.yE[c] = claimc(tr, XC.lg, CE[c], rE);
        std::vector<gl_t> rb(rE.begin() + ld, rE.begin() + le);
        pf.yER = claimc(tr, XC.lg, CB[B_R], p3zkc::expt(rb, rE, le));
        std::vector<gl_t> rg(rE.begin(), rE.begin() + ld);
        pf.yEG = claimc(tr, XC.lg, *ops.G, p3zkc::expt(rg, rE, le));
    }
    if (!xc) tail_prove(tr, XC, R, Q, strict, pf.tail, "irm-bo");
    return pf;
}

static inline bool verify(fs::Transcript& tr, const Tables& T, const Proof& pf,
                          const Hash& rX, const Hash& rG, const Hash& rY,
                          uint32_t lT, uint32_t ld, uint32_t Q_pub, uint32_t R_pub,
                          const char** why = nullptr, p3lu::VCtx* xv = nullptr,
                          const char* tag = "irms") {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.lT != lT || pf.ld != ld) return fail("irms dims");
    const uint32_t le = lT + ld;
    uint32_t hdr[2] = {lT, ld};
    tr.absorb("irm-dims", hdr, sizeof hdr);
    tr.absorb("irm-X", rX.data(), 32);
    tr.absorb("irm-G", rG.data(), 32);
    tr.absorb("irm-Y", rY.data(), 32);
    p3lu::VCtx vloc;
    p3lu::VCtx& VC = xv ? *xv : vloc;
    const bool zk = p3zkc::G.on;
    for (int c = 0; c < NDB; c++) tr.absorb("irm-cb", pf.rdb[c].data(), 32);
    for (int c = 0; c < NDE; c++) tr.absorb("irm-ce", pf.rde[c].data(), 32);
    p3lu::vdefer_v(VC, {&pf.rdb[B_P4], &pf.rdb[B_P2], &pf.rdb[B_H2]}, T.EXP4, lT,
                   std::string(tag) + "E4");
    p3lu::vdefer_v(VC, {&pf.rdb[B_MM]}, T.R16, lT, std::string(tag) + "m");
    p3lu::vdefer_v(VC, {&pf.rdb[B_MS]}, T.R16, lT, std::string(tag) + "ms");
    p3lu::vdefer_v(VC, {&pf.rdb[B_MM], &pf.rdb[B_RP]}, T.ISQ, lT, std::string(tag) + "isq");
    static const int LCOL[9] = {B_L40, B_L41, B_L42, B_D40, B_D41, B_D42,
                                B_L20, B_D20, B_LR0};
    for (int l = 0; l < 9; l++)
        p3lu::vdefer_v(VC, {&pf.rdb[LCOL[l]]}, T.R16, lT,
                       std::string(tag) + "L" + std::to_string(l));
    p3lu::vdefer_v(VC, {&pf.rdb[B_LR1]}, T.R16, lT, std::string(tag) + "LR1");
    p3lu::vdefer_v(VC, {&pf.rde[E_RW]}, T.R16, le, std::string(tag) + "rw");
    p3lu::vdefer_v(VC, {&pf.rde[E_W]}, T.RS20, le, std::string(tag) + "W");
    p3lu::vdefer_v(VC, {&pf.rde[E_RY]}, T.R16, le, std::string(tag) + "ry");
    p3lu::vdefer_v(VC, {&rY}, T.RS20, le, std::string(tag) + "Y");

    // -- M row-sum --
    {
        std::vector<gl_t> zb = chal_vec(tr, lT);
        gl_t zex = zk ? chal(tr) : 0;
        gl_t yM = claimv(tr, VC.vlg, pf.rdb[B_M], p3zkc::xpt(zb, zex), pf.yM);
        std::vector<gl_t> ptb = ptb_pt(zb, zex, true, le);
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl[0]);
        gl_t base0 = gl_add(gl_sub(yM, (gl_t)CEPS), gl_mul(rho, pf.bl[0].H));
        std::vector<gl_t> rS; gl_t claim;
        if (!sc5_verify(pf.mS, p3zkc::vfull(le), base0, tr, "irm-scS", rS, claim))
            return fail("irms M sumcheck");
        if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl[0], rS)) return fail("irms blind ip S");
        gl_t yX = claimv(tr, VC.vlg, rX, rS, pf.yMX);
        gl_t w0 = rowsum_w(rS, ld, le, ptb);
        gl_t end = gl_add(gl_mul(w0, gl_mul(yX, yX)),
                          p3hwl::sc5_blindterm(pf.bl[0], rho, w0));
        if (end != claim) return fail("irms M terminal");
    }
    // -- Db --
    std::vector<gl_t> zB = chal_vec(tr, lT);
    gl_t lamB = chal(tr), lamBv[N_B_C]; lamBv[0] = 1;
    for (int j = 1; j < N_B_C; j++) lamBv[j] = gl_mul(lamBv[j-1], lamB);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl[1]);
        std::vector<gl_t> rB; gl_t claim;
        if (!sc5_verify(pf.mB, p3zkc::vfull(lT), gl_mul(rho, pf.bl[1].H),
                        tr, "irm-scB", rB, claim)) return fail("irms Db sumcheck");
        if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl[1], rB)) return fail("irms blind ip B");
        gl_t v[1 + NDB]; v[0] = p3bf::eq_point(rB, p3zkc::zpt(zB));
        for (int c = 0; c < NDB; c++)
            v[1 + c] = claimv(tr, VC.vlg, pf.rdb[c], rB, pf.yB[c]);
        const gl_t* c = v + 1;
        gl_t r4 = gl_add(c[B_L40], gl_add(gl_mul(c[B_L41], 65536ULL),
                                          gl_mul(c[B_L42], 4294967296ULL)));
        gl_t d4 = gl_add(c[B_D40], gl_add(gl_mul(c[B_D41], 65536ULL),
                                          gl_mul(c[B_D42], 4294967296ULL)));
        gl_t r[N_B_C];
        r[0] = gl_sub(c[B_M], gl_add(gl_mul(c[B_MM], c[B_P4]), r4));
        r[1] = gl_sub(gl_sub(gl_sub(c[B_P4], 1ULL), r4), d4);
        r[2] = gl_sub(gl_add(gl_mul(c[B_R], c[B_P2]), c[B_L20]),
                      gl_add(c[B_RP], c[B_H2]));
        r[3] = gl_sub(gl_sub(gl_sub(c[B_P2], 1ULL), c[B_L20]), c[B_D20]);
        r[4] = gl_sub(c[B_R], gl_add(c[B_LR0], gl_mul(c[B_LR1], 65536ULL)));
        r[5] = gl_sub(gl_add(c[B_MS], 16384ULL), c[B_MM]);
        gl_t s = 0;
        for (int j = 0; j < N_B_C; j++) s = gl_add(s, gl_mul(lamBv[j], r[j]));
        gl_t end = gl_add(gl_mul(v[0], s), p3hwl::sc5_blindterm(pf.bl[1], rho, v[0]));
        if (end != claim) return fail("irms Db terminal");
    }
    // -- De --
    std::vector<gl_t> zE = chal_vec(tr, le);
    gl_t lamE = chal(tr);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl[2]);
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.mE, p3zkc::vfull(le), gl_mul(rho, pf.bl[2].H),
                        tr, "irm-scE", rE, claim)) return fail("irms De sumcheck");
        if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl[2], rE)) return fail("irms blind ip E");
        gl_t w0 = p3bf::eq_point(rE, p3zkc::zpt(zE));
        gl_t yX = claimv(tr, VC.vlg, rX, rE, pf.yEX);
        gl_t yY = claimv(tr, VC.vlg, rY, rE, pf.yEY);
        gl_t yW = claimv(tr, VC.vlg, pf.rde[E_W], rE, pf.yE[E_W]);
        gl_t yRW = claimv(tr, VC.vlg, pf.rde[E_RW], rE, pf.yE[E_RW]);
        gl_t yRY = claimv(tr, VC.vlg, pf.rde[E_RY], rE, pf.yE[E_RY]);
        std::vector<gl_t> rb(rE.begin() + ld, rE.begin() + le);
        gl_t yR = claimv(tr, VC.vlg, pf.rdb[B_R], p3zkc::expt(rb, rE, le), pf.yER);
        std::vector<gl_t> rg(rE.begin(), rE.begin() + ld);
        gl_t yG = claimv(tr, VC.vlg, rG, p3zkc::expt(rg, rE, le), pf.yEG);
        gl_t r0 = gl_sub(gl_add(gl_mul(yR, yG), 32768ULL),
                         gl_add(gl_mul(yW, 65536ULL), yRW));
        gl_t r1 = gl_sub(gl_add(gl_mul(yW, yX), 32768ULL),
                         gl_add(gl_mul(yY, 65536ULL), yRY));
        gl_t end = gl_add(gl_mul(w0, gl_add(r0, gl_mul(lamE, r1))),
                          p3hwl::sc5_blindterm(pf.bl[2], rho, w0));
        if (end != claim) return fail("irms De terminal");
    }
    if (!xv) { if (!tail_verify(tr, VC, pf.tail, Q_pub, R_pub, "irm-bo", why)) return false; }
    else if (!pf.tail.lug.empty() || !pf.tail.batches.empty()) return fail("unexpected tail");
    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const Proof& pf) {
    size_t s = 8 + (NDB + NDE) * 32 + (pf.mS.size() + pf.mB.size() + pf.mE.size()) * 40
             + 8 * (2 + NDB + NDE + 4);
    for (auto& g : pf.tail.lug) s += p3lu::sz_group(g);
    for (auto& b : pf.tail.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3irms

// ==========================================================================
// p3irope -- integer RoPE on the FULL (T x d) grid (all heads/batches at
// once; positions s = t mod seq, head-internal column jj = e mod dh):
//   y64[t,e] = q[t,e]*C[t,e] + q[t, flip(e)]*S2[t,e],  flip = XOR dh/2 bit,
//   C = cos table (scale 2^14), S2 = +-sin (sign by half),  y = rescale(y64,14).
// C/S2 are PUBLIC (verifier evaluates their broadcast MLEs itself); the
// rotate-half operand is bound by opening the SAME commitment at the
// bit-flipped point (rotate-half MLE identity, zkob_rope section 1.4).
// ==========================================================================
namespace p3irope {

using namespace p3ig;

enum { RPT_NONE = 0, RPT_UNROT, RPT_Y };
struct Tamper { int mode = RPT_NONE; size_t i = 0; };

struct Pub {                                     // int cos/sin, scale 2^14
    uint32_t lseq = 0, ldh = 0;
    std::vector<int32_t> ct, st;                 // seq x dh/2
};

// real-slice broadcast values of C and S2 at element u = j | (t << ld)
static inline void cs_at(const Pub& P, uint32_t ld, size_t u, int64_t& C, int64_t& S2) {
    uint32_t j = (uint32_t)(u & (((size_t)1 << ld) - 1));
    uint32_t t = (uint32_t)(u >> ld);
    uint32_t s = t & ((1u << P.lseq) - 1);
    uint32_t jj = j & ((1u << P.ldh) - 1);
    uint32_t hb = (jj >> (P.ldh - 1)) & 1;
    uint32_t j2 = jj & ((1u << (P.ldh - 1)) - 1);
    size_t ti = (size_t)s * ((size_t)1 << (P.ldh - 1)) + j2;
    C = P.ct[ti];
    S2 = hb ? (int64_t)P.st[ti] : -(int64_t)P.st[ti];
}
// public broadcast arrays over the aug domain (mask slices = 0)
static inline std::vector<gl_t> pub_col(const Pub& P, uint32_t ld, uint32_t le, bool sin) {
    std::vector<gl_t> v((size_t)1 << p3zkc::vfull(le), 0);
    const size_t Ne = (size_t)1 << le;
    for (size_t u = 0; u < Ne; u++) {
        int64_t C, S2; cs_at(P, ld, u, C, S2);
        v[u] = gsig(sin ? S2 : C);
    }
    return v;
}
// verifier MLE of a public broadcast at the full point rE
static inline gl_t pub_eval(const Pub& P, uint32_t ld, uint32_t le, bool sin,
                            const std::vector<gl_t>& rE) {
    std::vector<gl_t> arr((size_t)1 << le);
    for (size_t u = 0; u < arr.size(); u++) {
        int64_t C, S2; cs_at(P, ld, u, C, S2);
        arr[u] = gsig(sin ? S2 : C);
    }
    std::vector<gl_t> rr(rE.begin(), rE.begin() + le);
    gl_t y = p3bf::eval_h(arr, p3bf::build_eq(rr));
    for (size_t i = le; i < rE.size(); i++) y = gl_mul(y, gl_sub(1ULL, rE[i]));
    return y;
}

struct Wit {
    uint32_t lT = 0, ld = 0, le = 0, fb = 0;     // fb = flipped index bit
    std::vector<gl_t> y, rem;
    std::vector<uint32_t> idxR, idxY;
};
static inline Wit gen_witness(uint32_t lT, uint32_t ld, const Pub& P,
                              const std::vector<gl_t>& q, const Tamper* tm = nullptr) {
    Wit w; w.lT = lT; w.ld = ld; w.le = lT + ld; w.fb = P.ldh - 1;
    const size_t Ne = (size_t)1 << w.le;
    w.y.assign(Ne, 0); w.rem.assign(Ne, 0);
    w.idxR.assign(Ne, 0); w.idxY.assign(Ne, 0);
    for (size_t u = 0; u < Ne; u++) {
        int64_t C, S2; cs_at(P, ld, u, C, S2);
        size_t uf = u ^ ((size_t)1 << w.fb);
        int64_t qv = sig64(q[u]);
        int64_t qf = sig64(q[uf]);
        if (tm && tm->mode == RPT_UNROT && u == tm->i) qf = qv;   // forgot the rotation
        int64_t y64 = qv * C + qf * S2;
        int64_t Y = (y64 + (1LL << 13)) >> 14;
        int64_t rem = (y64 + (1LL << 13)) - (Y << 14);
        if (tm && tm->mode == RPT_Y && u == tm->i) { Y += 1; rem -= 1LL << 14; }
        if (Y < -ABND || Y >= ABND) throw std::runtime_error("irope: y out of range");
        w.y[u] = gsig(Y);
        w.rem[u] = rem >= 0 ? (gl_t)rem : gsig(rem);
        w.idxR[u] = (uint32_t)((uint64_t)rem & 0x3FFF);
        w.idxY[u] = (uint32_t)(Y + ABND);
    }
    return w;
}

struct Operands { const Col* Q; const Col* Y; };

struct Proof {
    uint32_t lT = 0, ld = 0, fb = 0;
    Hash rREM;
    std::vector<Msg5> m;
    gl_t yQ = 0, yQx = 0, yY = 0, yR = 0;
    p3zkc::Blind bl;
    Tail tail;
};

static inline Proof prove(fs::Transcript& tr, const Wit& w, const Tables& T,
                          const Pub& P, const Operands& ops, uint32_t R, uint32_t Q,
                          bool strict = true, p3lu::XCtx* xc = nullptr,
                          const char* tag = "irp") {
    Proof pf; pf.lT = w.lT; pf.ld = w.ld; pf.fb = w.fb;
    uint32_t hdr[3] = {w.lT, w.ld, w.fb};
    tr.absorb("irp-dims", hdr, sizeof hdr);
    tr.absorb("irp-Q", ops.Q->root.data(), 32);
    tr.absorb("irp-Y", ops.Y->root.data(), 32);
    p3lu::XCtx xloc;
    p3lu::XCtx& XC = xc ? *xc : xloc;
    const uint32_t le = w.le;
    std::vector<Col>& CW = XC.vec(1);
    CW[0] = commit_col_nc(w.rem, R);
    pf.rREM = CW[0].root;
    tr.absorb("irp-r", pf.rREM.data(), 32);
    p3lu::defer_v(XC, {LC(&CW[0])}, w.idxR, T.R14, std::string(tag) + "r");
    p3lu::defer_v(XC, {LC(ops.Y)}, w.idxY, T.RS20, std::string(tag) + "Y");
    std::vector<gl_t> zE = chal_vec(tr, le);
    {
        // Qx = rotate-half of the FULL augmented Q array (bit fb flip is a
        // within-slice permutation -> the flipped-point claim is exact)
        std::vector<gl_t> Qx(ops.Q->v.size());
        for (size_t u = 0; u < Qx.size(); u++) Qx[u] = ops.Q->v[u ^ ((size_t)1 << w.fb)];
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(beq(p3zkc::zpt(zE)));
        cols.push_back(ops.Q->v);
        cols.push_back(std::move(Qx));
        cols.push_back(ops.Y->v);
        cols.push_back(CW[0].v);
        cols.push_back(pub_col(P, w.ld, le, false));
        cols.push_back(pub_col(P, w.ld, le, true));
        CFn F = [](const gl_t* v) {
            gl_t t = gl_add(gl_add(gl_mul(v[1], v[5]), gl_mul(v[2], v[6])), 8192ULL);
            t = gl_sub(t, gl_add(gl_mul(v[3], 16384ULL), v[4]));
            return gl_mul(v[0], t);
        };
        std::vector<gl_t> rE = p3hwl::sc5z(tr, "irp-sc", le, std::move(cols), F, pf.m,
                                           0, R, XC.lg, XC.keep, pf.bl);
        pf.yQ = claimc(tr, XC.lg, *ops.Q, rE);
        std::vector<gl_t> rf = rE; rf[w.fb] = gl_sub(1ULL, rf[w.fb]);
        pf.yQx = claimc(tr, XC.lg, *ops.Q, rf);
        pf.yY = claimc(tr, XC.lg, *ops.Y, rE);
        pf.yR = claimc(tr, XC.lg, CW[0], rE);
    }
    if (!xc) tail_prove(tr, XC, R, Q, strict, pf.tail, "irp-bo");
    return pf;
}

static inline bool verify(fs::Transcript& tr, const Tables& T, const Pub& P,
                          const Proof& pf, const Hash& rQ, const Hash& rY,
                          uint32_t lT, uint32_t ld, uint32_t Q_pub, uint32_t R_pub,
                          const char** why = nullptr, p3lu::VCtx* xv = nullptr,
                          const char* tag = "irp") {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.lT != lT || pf.ld != ld || pf.fb != P.ldh - 1) return fail("irope dims");
    const uint32_t le = lT + ld;
    uint32_t hdr[3] = {lT, ld, pf.fb};
    tr.absorb("irp-dims", hdr, sizeof hdr);
    tr.absorb("irp-Q", rQ.data(), 32);
    tr.absorb("irp-Y", rY.data(), 32);
    p3lu::VCtx vloc;
    p3lu::VCtx& VC = xv ? *xv : vloc;
    tr.absorb("irp-r", pf.rREM.data(), 32);
    p3lu::vdefer_v(VC, {&pf.rREM}, T.R14, le, std::string(tag) + "r");
    p3lu::vdefer_v(VC, {&rY}, T.RS20, le, std::string(tag) + "Y");
    std::vector<gl_t> zE = chal_vec(tr, le);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl);
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.m, p3zkc::vfull(le), gl_mul(rho, pf.bl.H),
                        tr, "irp-sc", rE, claim)) return fail("irope sumcheck");
        if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl, rE)) return fail("irope blind ip");
        gl_t w0 = p3bf::eq_point(rE, p3zkc::zpt(zE));
        gl_t yQ = claimv(tr, VC.vlg, rQ, rE, pf.yQ);
        std::vector<gl_t> rf = rE; rf[pf.fb] = gl_sub(1ULL, rf[pf.fb]);
        gl_t yQx = claimv(tr, VC.vlg, rQ, rf, pf.yQx);
        gl_t yY = claimv(tr, VC.vlg, rY, rE, pf.yY);
        gl_t yR = claimv(tr, VC.vlg, pf.rREM, rE, pf.yR);
        gl_t yC = pub_eval(P, ld, le, false, rE);
        gl_t yS = pub_eval(P, ld, le, true, rE);
        gl_t t = gl_add(gl_add(gl_mul(yQ, yC), gl_mul(yQx, yS)), 8192ULL);
        t = gl_sub(t, gl_add(gl_mul(yY, 16384ULL), yR));
        gl_t end = gl_add(gl_mul(w0, t), p3hwl::sc5_blindterm(pf.bl, rho, w0));
        if (end != claim) return fail("irope terminal");
    }
    if (!xv) { if (!tail_verify(tr, VC, pf.tail, Q_pub, R_pub, "irp-bo", why)) return false; }
    else if (!pf.tail.lug.empty() || !pf.tail.batches.empty()) return fail("unexpected tail");
    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const Proof& pf) {
    size_t s = 12 + 32 + pf.m.size() * 40 + 32;
    for (auto& g : pf.tail.lug) s += p3lu::sz_group(g);
    for (auto& b : pf.tail.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3irope

// ==========================================================================
// p3ismx -- integer softmax on the FULL (A*seq x seq) score grid (zkob
// softmax8 semantics).  Per row (i,a) of z (scale 2^8, RS16-bounded by the
// scores rescale):
//   mx = max_{j<=i} z   (SEL attainment + DM-in-table dominance);
//   DM = mask*(mx - z) + (1-mask)*2^16;  E = EXPT[DM]  (masked -> 0 by table);
//   S = sum_j E;  P = round_half_up(2^16 E / S)  by the bracket
//   r1 = 2^17 E + S - 2 P S in [0, 2S),  r2 = 2S-1-r1  (16-bit limbs).
// P is uniquely FORCED in-field by the bracket (no range lookup needed):
// r1+r2 = 2S-1 pins r1 < 2S exactly, and P = (2^17E+S-r1)/(2S) mod p equals
// the honest integer because all terms are bounded << p.
// ==========================================================================
namespace p3ismx {

using namespace p3ig;

enum { ST_NONE = 0, ST_MX, ST_E, ST_P, ST_S };
struct Tamper { int mode = ST_NONE; uint32_t row = 0; uint32_t j = 0; };

enum { X_SEL = 0, X_DM, X_EE, X_L0, X_L1, X_L2, X_L3, NXE };
enum { XB_MX = 0, XB_S, NXB };

struct Wit {
    uint32_t la = 0, lseq = 0, lr = 0, le = 0;
    std::vector<gl_t> xe[NXE];                    // element domain
    std::vector<gl_t> xb[NXB];                    // row domain
    std::vector<gl_t> p;                          // P output (caller commits: chain op)
    std::vector<uint32_t> idxDM, idxL[4], idxMX;
};

static inline bool causal(uint32_t i, uint32_t j) { return j <= i; }

static inline Wit gen_witness(uint32_t la, uint32_t lseq, const std::vector<gl_t>& z,
                              const Tables& T, const Tamper* tm = nullptr) {
    Wit w; w.la = la; w.lseq = lseq; w.lr = la + lseq; w.le = la + 2 * lseq;
    const uint32_t seq = 1u << lseq, NR = 1u << w.lr;
    const size_t Ne = (size_t)1 << w.le;
    for (int c = 0; c < NXE; c++) w.xe[c].assign(Ne, 0);
    for (int c = 0; c < NXB; c++) w.xb[c].assign(NR, 0);
    w.p.assign(Ne, 0);
    w.idxDM.assign(Ne, 0);
    for (int l = 0; l < 4; l++) w.idxL[l].assign(Ne, 0);
    w.idxMX.assign(NR, 0);
    for (uint32_t r = 0; r < NR; r++) {
        const uint32_t i = r & (seq - 1);
        int64_t mx = INT64_MIN;
        for (uint32_t j = 0; j <= i; j++) {
            int64_t zv = sig64(z[((size_t)r << lseq) | j]);
            if (zv > mx) mx = zv;
        }
        if (tm && tm->mode == ST_MX && r == tm->row) mx += 1;
        w.xb[XB_MX][r] = gsig(mx);
        w.idxMX[r] = (uint32_t)(mx + (1LL << 15));
        int64_t S = 0;
        int sel_j = -1;
        for (uint32_t j = 0; j < seq; j++) {
            size_t e = ((size_t)r << lseq) | j;
            int64_t zv = sig64(z[e]);
            int64_t dm = causal(i, j) ? mx - zv : (1LL << 16);
            if (dm < 0 || dm >= (1LL << EXPB)) throw std::runtime_error("ismx: DM range");
            int64_t E = (int64_t)T.EXPT.cols[1][dm];
            if (tm && tm->mode == ST_E && r == tm->row && j == tm->j)
                E += E < 65536 ? 1 : -1;
            if (sel_j < 0 && causal(i, j) && dm == 0) sel_j = (int)j;
            w.xe[X_DM][e] = (gl_t)dm; w.xe[X_EE][e] = (gl_t)E;
            w.idxDM[e] = (uint32_t)dm;
            S += E;
        }
        if (tm && tm->mode == ST_S && r == tm->row) S += 1;
        w.xb[XB_S][r] = (gl_t)S;
        if (sel_j >= 0) w.xe[X_SEL][((size_t)r << lseq) | (uint32_t)sel_j] = 1;
        for (uint32_t j = 0; j < seq; j++) {
            size_t e = ((size_t)r << lseq) | j;
            int64_t E = (int64_t)sig64(w.xe[X_EE][e]);
            int64_t P = ((1LL << 17) * E + S) / (2 * S);
            if (tm && tm->mode == ST_P && r == tm->row && j == tm->j) P += 1;
            int64_t r1 = (1LL << 17) * E + S - 2 * P * S;
            int64_t r2 = 2 * S - 1 - r1;
            w.p[e] = gsig(P);
            if (r1 >= 0 && r2 >= 0) {
                w.xe[X_L0][e] = (gl_t)(r1 & 0xFFFF); w.xe[X_L1][e] = (gl_t)(r1 >> 16);
                w.xe[X_L2][e] = (gl_t)(r2 & 0xFFFF); w.xe[X_L3][e] = (gl_t)(r2 >> 16);
            } else {
                // tampered P: keep the bracket identities satisfied in-field so
                // the LIMB LOOKUP is what rejects
                w.xe[X_L0][e] = gsig(r1); w.xe[X_L1][e] = 0;
                w.xe[X_L2][e] = gsig(r2); w.xe[X_L3][e] = 0;
            }
            w.idxL[0][e] = (uint32_t)((uint64_t)r1 & 0xFFFF);
            w.idxL[1][e] = (uint32_t)((uint64_t)(r1 < 0 ? 0 : r1) >> 16);
            w.idxL[2][e] = (uint32_t)((uint64_t)r2 & 0xFFFF);
            w.idxL[3][e] = (uint32_t)((uint64_t)(r2 < 0 ? 0 : r2) >> 16);
        }
    }
    return w;
}

struct Operands { const Col* Z; const Col* P; };

struct Proof {
    uint32_t la = 0, lseq = 0;
    Hash rxe[NXE], rxb[NXB];
    std::vector<Msg5> mSel, mS, mE;
    gl_t ySrow = 0, ySE = 0;                     // S row-sum binding
    gl_t ySel = 0;                               // SEL row-sum terminal
    gl_t yE[NXE] = {}, yEZ = 0, yEP = 0, yEMX = 0, yES = 0;
    p3zkc::Blind bl[3];
    Tail tail;
};

static const int N_E_C = 5;

// public causal-mask broadcast (0 on mask slices)
static inline std::vector<gl_t> mask_col(uint32_t la, uint32_t lseq, uint32_t le) {
    std::vector<gl_t> v((size_t)1 << p3zkc::vfull(le), 0);
    const uint32_t seq = 1u << lseq;
    const size_t Ne = (size_t)1 << le;
    for (size_t u = 0; u < Ne; u++) {
        uint32_t j = (uint32_t)(u & (seq - 1));
        uint32_t i = (uint32_t)((u >> lseq) & (seq - 1));
        v[u] = causal(i, j) ? 1 : 0;
    }
    return v;
}
static inline gl_t mask_eval(uint32_t la, uint32_t lseq, uint32_t le,
                             const std::vector<gl_t>& rE) {
    const uint32_t seq = 1u << lseq;
    std::vector<gl_t> arr((size_t)1 << le);
    for (size_t u = 0; u < arr.size(); u++) {
        uint32_t j = (uint32_t)(u & (seq - 1));
        uint32_t i = (uint32_t)((u >> lseq) & (seq - 1));
        arr[u] = causal(i, j) ? 1 : 0;
    }
    std::vector<gl_t> rr(rE.begin(), rE.begin() + le);
    gl_t y = p3bf::eval_h(arr, p3bf::build_eq(rr));
    for (size_t i = le; i < rE.size(); i++) y = gl_mul(y, gl_sub(1ULL, rE[i]));
    return y;
}

static inline Proof prove(fs::Transcript& tr, const Wit& w, const Tables& T,
                          const Operands& ops, uint32_t R, uint32_t Q,
                          bool strict = true, p3lu::XCtx* xc = nullptr,
                          const char* tag = "ismx") {
    Proof pf; pf.la = w.la; pf.lseq = w.lseq;
    uint32_t hdr[2] = {w.la, w.lseq};
    tr.absorb("ism-dims", hdr, sizeof hdr);
    tr.absorb("ism-Z", ops.Z->root.data(), 32);
    tr.absorb("ism-P", ops.P->root.data(), 32);
    p3lu::XCtx xloc;
    p3lu::XCtx& XC = xc ? *xc : xloc;
    const bool zk = p3zkc::G.on;
    const uint32_t lr = w.lr, le = w.le, lseq = w.lseq;

    std::vector<Col>& CE = XC.vec(NXE);
    std::vector<Col>& CB = XC.vec(NXB);
    for (int c = 0; c < NXE; c++) {
        CE[c] = commit_col_nc(w.xe[c], R);
        pf.rxe[c] = CE[c].root;
        tr.absorb("ism-ce", pf.rxe[c].data(), 32);
    }
    // S's mask slice 1 is LINKED = row sums of EE's mask slice 1
    std::vector<gl_t> mSv;
    if (zk) {
        std::vector<gl_t> e1 = p3zkc::slice1(CE[X_EE].v, le);
        mSv = p3zkc::mk_linked(lr, m1_rowsum(e1, lseq, lr, false));
    }
    CB[XB_MX] = commit_col_nc(w.xb[XB_MX], R);
    CB[XB_S] = commit_col_nc(w.xb[XB_S], R, zk ? &mSv : nullptr);
    for (int c = 0; c < NXB; c++) {
        pf.rxb[c] = CB[c].root;
        tr.absorb("ism-cb", pf.rxb[c].data(), 32);
    }
    p3lu::defer_v(XC, {LC(&CE[X_DM]), LC(&CE[X_EE])}, w.idxDM, T.EXPT,
                  std::string(tag) + "exp");
    for (int l = 0; l < 4; l++)
        p3lu::defer_v(XC, {LC(&CE[X_L0 + l])}, w.idxL[l], T.R16,
                      std::string(tag) + "L" + std::to_string(l));
    p3lu::defer_v(XC, {LC(&CB[XB_MX])}, w.idxMX, T.RS16, std::string(tag) + "mx");

    // -- SEL row-sum: sum_j SEL = 1 (public base; zpt weight) --
    {
        std::vector<gl_t> zb = chal_vec(tr, lr);
        std::vector<gl_t> ptb = ptb_pt(zb, 0, false, le);
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(eqb_col(ptb, lr, lseq, CE[X_SEL].v.size()));
        cols.push_back(CE[X_SEL].v);
        CFn F = [](const gl_t* v) { return gl_mul(v[0], v[1]); };
        std::vector<gl_t> rS = p3hwl::sc5z(tr, "ism-scsel", le, std::move(cols), F,
                                           pf.mSel, 1ULL, R, XC.lg, XC.keep, pf.bl[0]);
        pf.ySel = claimc(tr, XC.lg, CE[X_SEL], rS);
    }
    // -- S row-sum: sum_j EE = S (hiding claim + linked mask) --
    {
        std::vector<gl_t> zb = chal_vec(tr, lr);
        gl_t zex = zk ? chal(tr) : 0;
        pf.ySrow = claimc(tr, XC.lg, CB[XB_S], p3zkc::xpt(zb, zex));
        std::vector<gl_t> ptb = ptb_pt(zb, zex, true, le);
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(eqb_col(ptb, lr, lseq, CE[X_EE].v.size()));
        cols.push_back(CE[X_EE].v);
        CFn F = [](const gl_t* v) { return gl_mul(v[0], v[1]); };
        std::vector<gl_t> rS = p3hwl::sc5z(tr, "ism-scs", le, std::move(cols), F,
                                           pf.mS, pf.ySrow, R, XC.lg, XC.keep, pf.bl[1]);
        pf.ySE = claimc(tr, XC.lg, CE[X_EE], rS);
    }
    // -- De zero-check --
    std::vector<gl_t> zE = chal_vec(tr, le);
    gl_t lamE = chal(tr), lamEv[N_E_C]; lamEv[0] = 1;
    for (int j = 1; j < N_E_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(beq(p3zkc::zpt(zE)));
        cols.push_back(ops.Z->v);
        for (int c = 0; c < NXE; c++) cols.push_back(CE[c].v);
        cols.push_back(ops.P->v);
        cols.push_back(p3zkc::bc_aug(CB[XB_MX].v, lr, le, (size_t)1 << le,
                                     [lseq](size_t e) { return e >> lseq; }));
        cols.push_back(p3zkc::bc_aug(CB[XB_S].v, lr, le, (size_t)1 << le,
                                     [lseq](size_t e) { return e >> lseq; }));
        cols.push_back(mask_col(w.la, lseq, le));
        CFn F = [&lamEv](const gl_t* v) {
            gl_t Z = v[1];
            const gl_t* c = v + 2;
            gl_t P = v[2 + NXE], MXb = v[3 + NXE], Sb = v[4 + NXE], MK = v[5 + NXE];
            gl_t r[N_E_C];
            r[0] = gl_sub(gl_mul(c[X_SEL], c[X_SEL]), c[X_SEL]);
            r[1] = gl_mul(c[X_SEL], c[X_DM]);
            r[2] = gl_sub(gl_sub(c[X_DM], gl_mul(MK, gl_sub(MXb, Z))),
                          gl_mul(gl_sub(1ULL, MK), 65536ULL));
            r[3] = gl_sub(gl_add(gl_mul(c[X_EE], 131072ULL), Sb),
                          gl_add(gl_add(gl_mul(gl_mul(P, Sb), 2ULL), c[X_L0]),
                                 gl_mul(c[X_L1], 65536ULL)));
            r[4] = gl_sub(gl_add(gl_add(gl_add(c[X_L2], gl_mul(c[X_L3], 65536ULL)),
                                        gl_add(c[X_L0], gl_mul(c[X_L1], 65536ULL))),
                          1ULL), gl_mul(Sb, 2ULL));
            gl_t s = 0;
            for (int j = 0; j < N_E_C; j++) s = gl_add(s, gl_mul(lamEv[j], r[j]));
            return gl_mul(v[0], s);
        };
        std::vector<gl_t> rE = p3hwl::sc5z(tr, "ism-sce", le, std::move(cols), F, pf.mE,
                                           0, R, XC.lg, XC.keep, pf.bl[2]);
        pf.yEZ = claimc(tr, XC.lg, *ops.Z, rE);
        for (int c = 0; c < NXE; c++) pf.yE[c] = claimc(tr, XC.lg, CE[c], rE);
        pf.yEP = claimc(tr, XC.lg, *ops.P, rE);
        std::vector<gl_t> rb(rE.begin() + lseq, rE.begin() + le);
        pf.yEMX = claimc(tr, XC.lg, CB[XB_MX], p3zkc::expt(rb, rE, le));
        pf.yES = claimc(tr, XC.lg, CB[XB_S], p3zkc::expt(rb, rE, le));
    }
    if (!xc) tail_prove(tr, XC, R, Q, strict, pf.tail, "ism-bo");
    return pf;
}

static inline bool verify(fs::Transcript& tr, const Tables& T, const Proof& pf,
                          const Hash& rZ, const Hash& rP, uint32_t la, uint32_t lseq,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr,
                          p3lu::VCtx* xv = nullptr, const char* tag = "ismx") {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.la != la || pf.lseq != lseq) return fail("ismx dims");
    const uint32_t lr = la + lseq, le = la + 2 * lseq;
    uint32_t hdr[2] = {la, lseq};
    tr.absorb("ism-dims", hdr, sizeof hdr);
    tr.absorb("ism-Z", rZ.data(), 32);
    tr.absorb("ism-P", rP.data(), 32);
    p3lu::VCtx vloc;
    p3lu::VCtx& VC = xv ? *xv : vloc;
    const bool zk = p3zkc::G.on;
    for (int c = 0; c < NXE; c++) tr.absorb("ism-ce", pf.rxe[c].data(), 32);
    for (int c = 0; c < NXB; c++) tr.absorb("ism-cb", pf.rxb[c].data(), 32);
    p3lu::vdefer_v(VC, {&pf.rxe[X_DM], &pf.rxe[X_EE]}, T.EXPT, le,
                   std::string(tag) + "exp");
    for (int l = 0; l < 4; l++)
        p3lu::vdefer_v(VC, {&pf.rxe[X_L0 + l]}, T.R16, le,
                       std::string(tag) + "L" + std::to_string(l));
    p3lu::vdefer_v(VC, {&pf.rxb[XB_MX]}, T.RS16, lr, std::string(tag) + "mx");

    // -- SEL row-sum --
    {
        std::vector<gl_t> zb = chal_vec(tr, lr);
        std::vector<gl_t> ptb = ptb_pt(zb, 0, false, le);
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl[0]);
        std::vector<gl_t> rS; gl_t claim;
        if (!sc5_verify(pf.mSel, p3zkc::vfull(le), gl_add(1ULL, gl_mul(rho, pf.bl[0].H)),
                        tr, "ism-scsel", rS, claim)) return fail("ismx SEL sumcheck");
        if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl[0], rS)) return fail("ismx blind sel");
        gl_t ySel = claimv(tr, VC.vlg, pf.rxe[X_SEL], rS, pf.ySel);
        gl_t w0 = rowsum_w(rS, lseq, le, ptb);
        gl_t end = gl_add(gl_mul(w0, ySel), p3hwl::sc5_blindterm(pf.bl[0], rho, w0));
        if (end != claim) return fail("ismx SEL terminal");
    }
    // -- S row-sum --
    {
        std::vector<gl_t> zb = chal_vec(tr, lr);
        gl_t zex = zk ? chal(tr) : 0;
        gl_t yS = claimv(tr, VC.vlg, pf.rxb[XB_S], p3zkc::xpt(zb, zex), pf.ySrow);
        std::vector<gl_t> ptb = ptb_pt(zb, zex, true, le);
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl[1]);
        std::vector<gl_t> rS; gl_t claim;
        if (!sc5_verify(pf.mS, p3zkc::vfull(le), gl_add(yS, gl_mul(rho, pf.bl[1].H)),
                        tr, "ism-scs", rS, claim)) return fail("ismx S sumcheck");
        if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl[1], rS)) return fail("ismx blind s");
        gl_t yE = claimv(tr, VC.vlg, pf.rxe[X_EE], rS, pf.ySE);
        gl_t w0 = rowsum_w(rS, lseq, le, ptb);
        gl_t end = gl_add(gl_mul(w0, yE), p3hwl::sc5_blindterm(pf.bl[1], rho, w0));
        if (end != claim) return fail("ismx S terminal");
    }
    // -- De --
    std::vector<gl_t> zE = chal_vec(tr, le);
    gl_t lamE = chal(tr), lamEv[N_E_C]; lamEv[0] = 1;
    for (int j = 1; j < N_E_C; j++) lamEv[j] = gl_mul(lamEv[j-1], lamE);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl[2]);
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.mE, p3zkc::vfull(le), gl_mul(rho, pf.bl[2].H),
                        tr, "ism-sce", rE, claim)) return fail("ismx De sumcheck");
        if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl[2], rE)) return fail("ismx blind e");
        gl_t w0 = p3bf::eq_point(rE, p3zkc::zpt(zE));
        gl_t Z = claimv(tr, VC.vlg, rZ, rE, pf.yEZ);
        gl_t c[NXE];
        for (int cc = 0; cc < NXE; cc++)
            c[cc] = claimv(tr, VC.vlg, pf.rxe[cc], rE, pf.yE[cc]);
        gl_t P = claimv(tr, VC.vlg, rP, rE, pf.yEP);
        std::vector<gl_t> rb(rE.begin() + lseq, rE.begin() + le);
        gl_t MXb = claimv(tr, VC.vlg, pf.rxb[XB_MX], p3zkc::expt(rb, rE, le), pf.yEMX);
        gl_t Sb = claimv(tr, VC.vlg, pf.rxb[XB_S], p3zkc::expt(rb, rE, le), pf.yES);
        gl_t MK = mask_eval(la, lseq, le, rE);
        gl_t r[N_E_C];
        r[0] = gl_sub(gl_mul(c[X_SEL], c[X_SEL]), c[X_SEL]);
        r[1] = gl_mul(c[X_SEL], c[X_DM]);
        r[2] = gl_sub(gl_sub(c[X_DM], gl_mul(MK, gl_sub(MXb, Z))),
                      gl_mul(gl_sub(1ULL, MK), 65536ULL));
        r[3] = gl_sub(gl_add(gl_mul(c[X_EE], 131072ULL), Sb),
                      gl_add(gl_add(gl_mul(gl_mul(P, Sb), 2ULL), c[X_L0]),
                             gl_mul(c[X_L1], 65536ULL)));
        r[4] = gl_sub(gl_add(gl_add(gl_add(c[X_L2], gl_mul(c[X_L3], 65536ULL)),
                                    gl_add(c[X_L0], gl_mul(c[X_L1], 65536ULL))),
                      1ULL), gl_mul(Sb, 2ULL));
        gl_t s = 0;
        for (int j = 0; j < N_E_C; j++) s = gl_add(s, gl_mul(lamEv[j], r[j]));
        gl_t end = gl_add(gl_mul(w0, s), p3hwl::sc5_blindterm(pf.bl[2], rho, w0));
        if (end != claim) return fail("ismx De terminal");
    }
    if (!xv) { if (!tail_verify(tr, VC, pf.tail, Q_pub, R_pub, "ism-bo", why)) return false; }
    else if (!pf.tail.lug.empty() || !pf.tail.batches.empty()) return fail("unexpected tail");
    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const Proof& pf) {
    size_t s = 8 + (NXE + NXB) * 32
             + (pf.mSel.size() + pf.mS.size() + pf.mE.size()) * 40
             + 8 * (NXE + 6);
    for (auto& g : pf.tail.lug) s += p3lu::sz_group(g);
    for (auto& b : pf.tail.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3ismx

// ==========================================================================
// p3iswg -- SwiGLU: (G, SIL) mapping lookup vs the SILU table (this also
// range-binds G to +-2^19), then MO = rescale(SIL*U, 16) folded into one
// zero-check.  zkob_glu semantics with the pair lookup done natively by the
// multi-column logUp (no homomorphic comb needed).
// ==========================================================================
namespace p3iswg {

using namespace p3ig;

enum { GT_NONE = 0, GT_SIL, GT_M };
struct Tamper { int mode = GT_NONE; size_t i = 0; };

struct Wit {
    uint32_t le = 0;
    std::vector<gl_t> sil, rem, mo;
    std::vector<uint32_t> idxG, idxR, idxM;
};
static inline Wit gen_witness(uint32_t le, const std::vector<gl_t>& g,
                              const std::vector<gl_t>& u, const Tables& T,
                              const Tamper* tm = nullptr) {
    Wit w; w.le = le;
    const size_t Ne = (size_t)1 << le;
    w.sil.assign(Ne, 0); w.rem.assign(Ne, 0); w.mo.assign(Ne, 0);
    w.idxG.assign(Ne, 0); w.idxR.assign(Ne, 0); w.idxM.assign(Ne, 0);
    for (size_t i = 0; i < Ne; i++) {
        int64_t gv = sig64(g[i]);
        if (gv < -ABND || gv >= ABND) throw std::runtime_error("iswg: G out of range");
        int64_t sil = sig64(T.SILU.cols[1][gv + ABND]);
        if (tm && tm->mode == GT_SIL && i == tm->i) sil += 1;
        int64_t m64 = sil * sig64(u[i]);
        int64_t mo = (m64 + (1LL << 15)) >> 16;
        int64_t rem = (m64 + (1LL << 15)) - (mo << 16);
        if (tm && tm->mode == GT_M && i == tm->i) mo += 1;
        if (mo < -ABND || mo >= ABND) throw std::runtime_error("iswg: M out of range");
        w.sil[i] = gsig(sil); w.rem[i] = (gl_t)rem; w.mo[i] = gsig(mo);
        w.idxG[i] = (uint32_t)(gv + ABND);
        w.idxR[i] = (uint32_t)rem;
        w.idxM[i] = (uint32_t)(mo + ABND);
    }
    return w;
}

struct Operands { const Col* G; const Col* U; const Col* MO; };

struct Proof {
    uint32_t le = 0;
    Hash rSIL, rREM;
    std::vector<Msg5> m;
    gl_t yG = 0, yU = 0, yM = 0, ySIL = 0, yR = 0;
    p3zkc::Blind bl;
    Tail tail;
};

static inline Proof prove(fs::Transcript& tr, const Wit& w, const Tables& T,
                          const Operands& ops, uint32_t R, uint32_t Q,
                          bool strict = true, p3lu::XCtx* xc = nullptr,
                          const char* tag = "iswg") {
    Proof pf; pf.le = w.le;
    tr.absorb("isw-dims", &w.le, 4);
    tr.absorb("isw-G", ops.G->root.data(), 32);
    tr.absorb("isw-U", ops.U->root.data(), 32);
    tr.absorb("isw-M", ops.MO->root.data(), 32);
    p3lu::XCtx xloc;
    p3lu::XCtx& XC = xc ? *xc : xloc;
    std::vector<Col>& CW = XC.vec(2);
    CW[0] = commit_col_nc(w.sil, R);
    CW[1] = commit_col_nc(w.rem, R);
    pf.rSIL = CW[0].root; pf.rREM = CW[1].root;
    tr.absorb("isw-s", pf.rSIL.data(), 32);
    tr.absorb("isw-r", pf.rREM.data(), 32);
    p3lu::defer_v(XC, {LC(ops.G), LC(&CW[0])}, w.idxG, T.SILU, std::string(tag) + "sil");
    p3lu::defer_v(XC, {LC(&CW[1])}, w.idxR, T.R16, std::string(tag) + "r");
    p3lu::defer_v(XC, {LC(ops.MO)}, w.idxM, T.RS20, std::string(tag) + "M");
    std::vector<gl_t> zE = chal_vec(tr, w.le);
    {
        std::vector<std::vector<gl_t>> cols;
        cols.push_back(beq(p3zkc::zpt(zE)));
        cols.push_back(CW[0].v);
        cols.push_back(ops.U->v);
        cols.push_back(ops.MO->v);
        cols.push_back(CW[1].v);
        CFn F = [](const gl_t* v) {
            gl_t t = gl_add(gl_mul(v[1], v[2]), 32768ULL);
            t = gl_sub(t, gl_add(gl_mul(v[3], 65536ULL), v[4]));
            return gl_mul(v[0], t);
        };
        std::vector<gl_t> rE = p3hwl::sc5z(tr, "isw-sc", w.le, std::move(cols), F, pf.m,
                                           0, R, XC.lg, XC.keep, pf.bl);
        pf.ySIL = claimc(tr, XC.lg, CW[0], rE);
        pf.yU = claimc(tr, XC.lg, *ops.U, rE);
        pf.yM = claimc(tr, XC.lg, *ops.MO, rE);
        pf.yR = claimc(tr, XC.lg, CW[1], rE);
    }
    if (!xc) tail_prove(tr, XC, R, Q, strict, pf.tail, "isw-bo");
    return pf;
}

static inline bool verify(fs::Transcript& tr, const Tables& T, const Proof& pf,
                          const Hash& rG, const Hash& rU, const Hash& rM, uint32_t le,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr,
                          p3lu::VCtx* xv = nullptr, const char* tag = "iswg") {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (pf.le != le) return fail("iswg dims");
    tr.absorb("isw-dims", &le, 4);
    tr.absorb("isw-G", rG.data(), 32);
    tr.absorb("isw-U", rU.data(), 32);
    tr.absorb("isw-M", rM.data(), 32);
    p3lu::VCtx vloc;
    p3lu::VCtx& VC = xv ? *xv : vloc;
    tr.absorb("isw-s", pf.rSIL.data(), 32);
    tr.absorb("isw-r", pf.rREM.data(), 32);
    p3lu::vdefer_v(VC, {&rG, &pf.rSIL}, T.SILU, le, std::string(tag) + "sil");
    p3lu::vdefer_v(VC, {&pf.rREM}, T.R16, le, std::string(tag) + "r");
    p3lu::vdefer_v(VC, {&rM}, T.RS20, le, std::string(tag) + "M");
    std::vector<gl_t> zE = chal_vec(tr, le);
    {
        gl_t rho = p3hwl::sc5vz_pre(tr, pf.bl);
        std::vector<gl_t> rE; gl_t claim;
        if (!sc5_verify(pf.m, p3zkc::vfull(le), gl_mul(rho, pf.bl.H),
                        tr, "isw-sc", rE, claim)) return fail("iswg sumcheck");
        if (!p3hwl::sc5vz_claims(tr, VC.vlg, pf.bl, rE)) return fail("iswg blind ip");
        gl_t w0 = p3bf::eq_point(rE, p3zkc::zpt(zE));
        gl_t ySIL = claimv(tr, VC.vlg, pf.rSIL, rE, pf.ySIL);
        gl_t yU = claimv(tr, VC.vlg, rU, rE, pf.yU);
        gl_t yM = claimv(tr, VC.vlg, rM, rE, pf.yM);
        gl_t yR = claimv(tr, VC.vlg, pf.rREM, rE, pf.yR);
        gl_t t = gl_add(gl_mul(ySIL, yU), 32768ULL);
        t = gl_sub(t, gl_add(gl_mul(yM, 65536ULL), yR));
        gl_t end = gl_add(gl_mul(w0, t), p3hwl::sc5_blindterm(pf.bl, rho, w0));
        if (end != claim) return fail("iswg terminal");
    }
    if (!xv) { if (!tail_verify(tr, VC, pf.tail, Q_pub, R_pub, "isw-bo", why)) return false; }
    else if (!pf.tail.lug.empty() || !pf.tail.batches.empty()) return fail("unexpected tail");
    if (why) *why = "ok";
    return true;
}

static inline size_t proof_size(const Proof& pf) {
    size_t s = 4 + 64 + pf.m.size() * 40 + 40;
    for (auto& g : pf.tail.lug) s += p3lu::sz_group(g);
    for (auto& b : pf.tail.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3iswg
