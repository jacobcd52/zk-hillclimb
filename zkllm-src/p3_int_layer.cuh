// p3_int_layer.cuh -- composed FULL-LAYER prover+verifier for the INTEGER
// transformer block (the integer counterpart of p3_transformer.cuh's fp8
// Hawkeye layer, INT_LAYER_LOG.md).  One llama-style block at residual scale
// 2^16 (int_layer_ref.py is the normative reference):
//
//   x - rmsnorm(g1) - [Wq,Wk,Wv int matmul + rescale] - rope(Q,K) -
//   per (batch,head) QK^T - rescale(scores) - int softmax - per (b,h) P.V -
//   rescale - Wo + rescale - residual - rmsnorm(g2) - [Wg,Wu] + rescale -
//   swiglu - Wd + rescale - residual - out
//
// as ONE proof over ONE Fiat-Shamir transcript with ONE shared p3lu::XCtx
// ledger, one merged lookup flush and one batched-opening pass -- the exact
// composition pattern of p3tf.  CHAINING is entirely by SHARED COMMITMENTS
// (each op's operand Col IS the producer's output Col; root equality is free)
// plus PARTIAL-POINT CLAIMS: head slices, K/V transposes and the attention
// concat are index maps inside the p3imm operand views, so the int layer has
// NO per-head quantize instances, NO V^T commitment and NO concat seam.
// Public statement: dims, input x0 (patterns + root), weight/gain roots, the
// public int rope cos/sin tables, table ids, Q/R, and the OUTPUT patterns.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_batchopen.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_int_gadgets.cuh"
#include "fs_transcript.hpp"

namespace p3itf {

using namespace p3ig;
using p3hwl::now_ms;

// ---------------- config ----------------
struct Config {
    uint32_t seq = 4, d = 64, nh = 2, dh = 32, dff = 128, batch = 1;
    uint32_t lseq() const { return ilog2(seq); }
    uint32_t ld()   const { return ilog2(d); }
    uint32_t ldh()  const { return ilog2(dh); }
    uint32_t ldff() const { return ilog2(dff); }
    uint32_t lnh()  const { return ilog2(nh); }
    uint32_t lbb()  const { return ilog2(batch); }
    uint32_t T()    const { return seq * batch; }
    uint32_t lT()   const { return lseq() + lbb(); }
    uint32_t A()    const { return batch * nh; }
    uint32_t la()   const { return lnh() + lbb(); }
    uint32_t tsh()  const { return (ldh() + 1) / 2; }   // pow2 softmax temp
    uint32_t sfz()  const { return 24 + tsh(); }        // scores rescale shift
    uint32_t btop() const { return sfz() - 16; }        // top-limb table width
    bool pow2() const {
        return (1u << lseq()) == seq && (1u << ld()) == d && (1u << ldh()) == dh
            && (1u << ldff()) == dff && (1u << lnh()) == nh
            && (1u << lbb()) == batch && nh * dh == d;
    }
};

// ---------------- weights (all ints at scale 2^16, |.| < 2^19) ----------------
enum { W_Q = 0, W_K, W_V, W_O, W_G, W_U, W_D, NW };
struct Weights {
    Config cfg;
    std::vector<gl_t> w[NW];         // K x N row-major (j-major)
    std::vector<gl_t> g1, g2;        // d gains
    p3irope::Pub rp;                 // public int cos/sin (scale 2^14)
};
static inline void wshape(const Config& c, int i, uint32_t& K, uint32_t& N) {
    K = (i == W_D) ? c.dff : c.d;
    N = (i == W_G || i == W_U) ? c.dff : c.d;
}

// ---------------- composed-battery tamper vocabulary ----------------
enum { TFT_NONE = 0,
       TFT_RMS_Y,      // rms1 output flipped -> rms De zero-check
       TFT_MM_Y,       // Wq accumulator flipped -> matmul terminal
       TFT_SCORE,      // scores value flipped in instance A()-1 -> QK matmul
       TFT_SMX_P,      // P flipped -> softmax bracket
       TFT_ATTN,       // PV accumulator flipped -> PV matmul
       TFT_SWG_M,      // swiglu output flipped -> swiglu zero-check
       TFT_RES_OUT,    // final residual flipped -> add2 zero-check
       TFT_IO_OUT,     // public output claim vs honest chain -> output binding
       TFT_IO_IN,      // public input vs committed -> input binding
       // per-gadget witness forgeries in the composed context
       TFT_GW_RMS, TFT_GW_IRS, TFT_GW_ROPE, TFT_GW_SMX, TFT_GW_SWG, TFT_GW_ADD };

// ---------------- operand views (deterministic in cfg) ----------------
using p3imm::OpView; using p3imm::Sel;
using p3imm::S_J; using p3imm::S_A; using p3imm::S_I; using p3imm::S_C;
static inline void sel_const(OpView& o, uint32_t bits, uint32_t val) {
    for (uint32_t t = 0; t < bits; t++)
        o.sel.push_back({S_C, (uint8_t)((val >> t) & 1)});
}
// X of QK: X(i,j) = RQ[(h*dh+j) | ((b*seq+i)<<ld)]
static inline OpView xview_qk(const Config& c, uint32_t h, uint32_t b) {
    OpView o; o.v = c.ld() + c.lT();
    for (uint32_t t = 0; t < c.ldh(); t++) o.sel.push_back({S_J, (uint8_t)t});
    sel_const(o, c.lnh(), h);
    for (uint32_t t = 0; t < c.lseq(); t++) o.sel.push_back({S_A, (uint8_t)t});
    sel_const(o, c.lbb(), b);
    return o;
}
// W of QK: W(j,k) = RK[(h*dh+j) | ((b*seq+k)<<ld)]
static inline OpView wview_qk(const Config& c, uint32_t h, uint32_t b) {
    OpView o; o.v = c.ld() + c.lT();
    for (uint32_t t = 0; t < c.ldh(); t++) o.sel.push_back({S_J, (uint8_t)t});
    sel_const(o, c.lnh(), h);
    for (uint32_t t = 0; t < c.lseq(); t++) o.sel.push_back({S_A, (uint8_t)t});
    sel_const(o, c.lbb(), b);
    return o;
}
// Y of QK: SC[k | (i<<lseq) | (a<<2lseq)]
static inline OpView yview_qk(const Config& c, uint32_t a) {
    OpView o; o.v = c.la() + 2 * c.lseq();
    for (uint32_t t = 0; t < c.lseq(); t++) o.sel.push_back({S_A, (uint8_t)t});
    for (uint32_t t = 0; t < c.lseq(); t++) o.sel.push_back({S_I, (uint8_t)t});
    sel_const(o, c.la(), a);
    return o;
}
// X of PV: X(i,j) = P[j | (i<<lseq) | (a<<2lseq)]
static inline OpView xview_pv(const Config& c, uint32_t a) {
    OpView o; o.v = c.la() + 2 * c.lseq();
    for (uint32_t t = 0; t < c.lseq(); t++) o.sel.push_back({S_J, (uint8_t)t});
    for (uint32_t t = 0; t < c.lseq(); t++) o.sel.push_back({S_A, (uint8_t)t});
    sel_const(o, c.la(), a);
    return o;
}
// W of PV: W(j,k) = YV[(h*dh+k) | ((b*seq+j)<<ld)]  (V "transpose" for free)
static inline OpView wview_pv(const Config& c, uint32_t h, uint32_t b) {
    OpView o; o.v = c.ld() + c.lT();
    for (uint32_t t = 0; t < c.ldh(); t++) o.sel.push_back({S_A, (uint8_t)t});
    sel_const(o, c.lnh(), h);
    for (uint32_t t = 0; t < c.lseq(); t++) o.sel.push_back({S_J, (uint8_t)t});
    sel_const(o, c.lbb(), b);
    return o;
}
// Y of PV: PVA[(h*dh+k) | ((b*seq+i)<<ld)]  (concat for free)
static inline OpView yview_pv(const Config& c, uint32_t h, uint32_t b) {
    OpView o; o.v = c.ld() + c.lT();
    for (uint32_t t = 0; t < c.ldh(); t++) o.sel.push_back({S_A, (uint8_t)t});
    sel_const(o, c.lnh(), h);
    for (uint32_t t = 0; t < c.lseq(); t++) o.sel.push_back({S_I, (uint8_t)t});
    sel_const(o, c.lbb(), b);
    return o;
}

// ---------------- chained layer witness ----------------
struct TfWit {
    Config cfg;
    p3irms::Wit rms1, rms2;
    p3irs::Wit rsq, rsk, rsv;                // QKV accumulator rescales
    p3irs::Wit rsz;                          // scores rescale
    p3irs::Wit rsat;                         // attention-output rescale
    p3irs::Wit rso, rsg, rsu, rsd;           // Wo / Wg / Wu / Wd rescales
    p3irope::Wit rpq, rpk;
    p3ismx::Wit smx;
    p3iswg::Wit swg;
    p3iadd::Wit add1, add2;
    // chained activation grids (values as signed field elements)
    std::vector<gl_t> x0, h1, acc[NW], yq, yk, yv, rq, rk,
                      sc, z, p, pva, at, yo, res1, h2, gg, uu, mo, yd, out;
    // range-lookup indices for the statement columns
    std::vector<uint32_t> idxW[NW], idxG1, idxG2, idxX0;
};

static inline std::vector<uint32_t> rs20_idx(const std::vector<gl_t>& v) {
    std::vector<uint32_t> idx(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        int64_t s = sig64(v[i]);
        if (s < -ABND || s >= ABND) throw std::runtime_error("itf: value out of RS20");
        idx[i] = (uint32_t)(s + ABND);
    }
    return idx;
}

// full chained witness: the layer is COMPUTED by the gadget replays themselves,
// so a tamper propagates honestly downstream and the proof must reject exactly
// at the owning gadget / binding.
static inline TfWit build_witness(const Config& cfg, const std::vector<gl_t>& x0,
                                  const Weights& W, const Tables& T,
                                  int tamper = TFT_NONE) {
    if (!cfg.pow2()) throw std::runtime_error("itf: config must be pow2");
    TfWit w; w.cfg = cfg;
    const uint32_t ld = cfg.ld(), lT = cfg.lT(), ldff = cfg.ldff(),
                   lseq = cfg.lseq(), ldh = cfg.ldh(), la = cfg.la(),
                   nh = cfg.nh, B = cfg.batch;
    const uint32_t le = ld + lT, lef = ldff + lT, lez = la + 2 * lseq;
    w.x0 = x0;
    auto flip = [](std::vector<gl_t>& v, size_t i) { v[i] = gl_add(v[i], 1ULL); };
    for (int i = 0; i < NW; i++) w.idxW[i] = rs20_idx(W.w[i]);
    w.idxG1 = rs20_idx(W.g1); w.idxG2 = rs20_idx(W.g2);
    w.idxX0 = rs20_idx(w.x0);

    // -- rmsnorm 1 --
    {
        p3irms::Tamper rt{p3irms::RT_R, 1, 0};
        w.rms1 = p3irms::gen_witness(lT, ld, w.x0, W.g1, T,
                                     tamper == TFT_GW_RMS ? &rt : nullptr);
        w.h1 = w.rms1.y;
        if (tamper == TFT_RMS_Y) flip(w.h1, 3);
    }
    // -- Wq/Wk/Wv + rescale --
    {
        OpView xv = p3imm::direct_x(nullptr, ld, lT);
        for (int i : {W_Q, W_K, W_V}) {
            OpView wv = p3imm::direct_w(nullptr, ld, ld);
            w.acc[i] = p3imm::compute_y(w.h1, xv, W.w[i], wv, ld, ld, lT);
        }
        if (tamper == TFT_MM_Y) flip(w.acc[W_Q], 5);
        p3irs::Tamper it{p3irs::IT_SHIFT, 4};
        w.rsq = p3irs::gen_witness(le, 16, p3irs::RNG_S20, w.acc[W_Q],
                                   tamper == TFT_GW_IRS ? &it : nullptr);
        w.rsk = p3irs::gen_witness(le, 16, p3irs::RNG_S20, w.acc[W_K]);
        w.rsv = p3irs::gen_witness(le, 16, p3irs::RNG_S20, w.acc[W_V]);
        w.yq = w.rsq.y; w.yk = w.rsk.y; w.yv = w.rsv.y;
    }
    // -- rope on the full grids --
    {
        p3irope::Tamper rt{p3irope::RPT_Y, ((size_t)cfg.seq / 2) << ld | 3};
        w.rpq = p3irope::gen_witness(lT, ld, W.rp, w.yq,
                                     tamper == TFT_GW_ROPE ? &rt : nullptr);
        w.rpk = p3irope::gen_witness(lT, ld, W.rp, w.yk);
        w.rq = w.rpq.y; w.rk = w.rpk.y;
    }
    // -- QK^T per instance into the shared scores grid --
    w.sc.assign((size_t)1 << lez, 0);
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t h = 0; h < nh; h++) {
            const uint32_t a = b * nh + h;
            OpView xv = xview_qk(cfg, h, b), wv = wview_qk(cfg, h, b);
            std::vector<gl_t> yi = p3imm::compute_y(w.rq, xv, w.rk, wv, ldh, lseq, lseq);
            OpView yv2 = yview_qk(cfg, a);
            for (uint32_t k = 0; k < cfg.seq; k++)
                for (uint32_t i = 0; i < cfg.seq; i++)
                    w.sc[p3imm::y_off(yv2, k, i)] = yi[k | ((size_t)i << lseq)];
        }
    if (tamper == TFT_SCORE)
        flip(w.sc, p3imm::y_off(yview_qk(cfg, cfg.A() - 1), 1, 2));
    // -- scores rescale (pow2 temp folded in) + softmax --
    w.rsz = p3irs::gen_witness(lez, cfg.sfz(), p3irs::RNG_S16, w.sc);
    w.z = w.rsz.y;
    {
        p3ismx::Tamper st{p3ismx::ST_E, 2, 1};
        w.smx = p3ismx::gen_witness(la, lseq, w.z, T,
                                    tamper == TFT_GW_SMX ? &st : nullptr);
        w.p = w.smx.p;
        if (tamper == TFT_SMX_P) flip(w.p, 2);
    }
    // -- P.V per instance into the shared attention grid --
    w.pva.assign((size_t)1 << le, 0);
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t h = 0; h < nh; h++) {
            const uint32_t a = b * nh + h;
            OpView xv = xview_pv(cfg, a), wv = wview_pv(cfg, h, b);
            std::vector<gl_t> yi = p3imm::compute_y(w.p, xv, w.yv, wv, lseq, ldh, lseq);
            OpView yv2 = yview_pv(cfg, h, b);
            for (uint32_t k = 0; k < cfg.dh; k++)
                for (uint32_t i = 0; i < cfg.seq; i++)
                    w.pva[p3imm::y_off(yv2, k, i)] = yi[k | ((size_t)i << ldh)];
        }
    if (tamper == TFT_ATTN) flip(w.pva, 9);
    w.rsat = p3irs::gen_witness(le, 16, p3irs::RNG_S20, w.pva);
    w.at = w.rsat.y;
    // -- Wo + rescale + residual 1 --
    {
        OpView xv = p3imm::direct_x(nullptr, ld, lT);
        OpView wv = p3imm::direct_w(nullptr, ld, ld);
        w.acc[W_O] = p3imm::compute_y(w.at, xv, W.w[W_O], wv, ld, ld, lT);
    }
    w.rso = p3irs::gen_witness(le, 16, p3irs::RNG_S20, w.acc[W_O]);
    w.yo = w.rso.y;
    {
        p3iadd::Tamper at{p3iadd::AT_OUT, 6};
        w.add1 = p3iadd::gen_witness(le, w.x0, w.yo,
                                     tamper == TFT_GW_ADD ? &at : nullptr);
        w.res1 = w.add1.out;
    }
    // -- rmsnorm 2, Wg/Wu + rescale, swiglu, Wd + rescale, residual 2 --
    w.rms2 = p3irms::gen_witness(lT, ld, w.res1, W.g2, T);
    w.h2 = w.rms2.y;
    {
        OpView xv = p3imm::direct_x(nullptr, ld, lT);
        for (int i : {W_G, W_U}) {
            OpView wv = p3imm::direct_w(nullptr, ld, ldff);
            w.acc[i] = p3imm::compute_y(w.h2, xv, W.w[i], wv, ld, ldff, lT);
        }
        w.rsg = p3irs::gen_witness(lef, 16, p3irs::RNG_S20, w.acc[W_G]);
        w.rsu = p3irs::gen_witness(lef, 16, p3irs::RNG_S20, w.acc[W_U]);
        w.gg = w.rsg.y; w.uu = w.rsu.y;
    }
    {
        p3iswg::Tamper st{p3iswg::GT_SIL, 8};
        w.swg = p3iswg::gen_witness(lef, w.gg, w.uu, T,
                                    tamper == TFT_GW_SWG ? &st : nullptr);
        w.mo = w.swg.mo;
        if (tamper == TFT_SWG_M) flip(w.mo, 4);
    }
    {
        OpView xv = p3imm::direct_x(nullptr, ldff, lT);
        OpView wv = p3imm::direct_w(nullptr, ldff, ld);
        w.acc[W_D] = p3imm::compute_y(w.mo, xv, W.w[W_D], wv, ldff, ld, lT);
        w.rsd = p3irs::gen_witness(le, 16, p3irs::RNG_S20, w.acc[W_D]);
        w.yd = w.rsd.y;
    }
    w.add2 = p3iadd::gen_witness(le, w.res1, w.yd);
    w.out = w.add2.out;
    if (tamper == TFT_RES_OUT) {
        flip(w.out, 7);
        w.add2.out[7] = w.out[7];
    }
    return w;
}


// ---------------- tables ----------------
static inline Tables layer_tables(const Config& cfg) {
    return p3ig::build_tables(cfg.d, cfg.btop());
}

// ---------------- operand commitments (the chain, committed once) ----------------
struct TfOps {
    Col X0, G1, G2, W[NW];
    Col H1, ACC[NW], YQ, YK, YV, RQ, RK, SC, Z, P, PVA, AT,
        YO, RES1, H2, GG, UU, MO, YD, OUT;
};

struct MInst { OpView xv, wv, yv; uint32_t lj, lk, li; };
// commit a matmul-accumulator grid with the zk mask-slice-1 linkage
static inline Col commit_acc(const std::vector<gl_t>& vals, uint32_t vY,
                             const std::vector<MInst>& insts, uint32_t R) {
    if (!p3zkc::G.on) return commit_col_nc(vals, R);
    std::vector<gl_t> m1((size_t)1 << vY, 0);
    for (auto& t : insts)
        p3imm::accum_mask1(m1, t.xv, t.wv, t.yv, t.lj, t.lk, t.li);
    std::vector<gl_t> mask = p3zkc::mk_linked(vY, m1);
    return commit_col_nc(vals, R, &mask);
}

// x0ext: an already-committed input column (multi-layer chaining); otherwise
// the input is committed fresh here.
static inline TfOps commit_all(const TfWit& w, const Weights& WW, uint32_t R,
                               const Col* x0ext = nullptr) {
    TfOps o;
    const Config& c = w.cfg;
    const uint32_t ld = c.ld(), lT = c.lT(), ldff = c.ldff(), lseq = c.lseq(),
                   ldh = c.ldh(), la = c.la(), nh = c.nh, B = c.batch;
    const uint32_t le = ld + lT, lef = ldff + lT, lez = la + 2 * lseq;
    o.X0 = x0ext ? *x0ext : commit_col_nc(w.x0, R);
    o.G1 = commit_col_nc(WW.g1, R);
    o.G2 = commit_col_nc(WW.g2, R);
    for (int i = 0; i < NW; i++) o.W[i] = commit_col_nc(WW.w[i], R);
    o.H1 = commit_col_nc(w.h1, R);
    for (int i : {W_Q, W_K, W_V}) {
        MInst mi{p3imm::direct_x(&o.H1, ld, lT), p3imm::direct_w(&o.W[i], ld, ld),
                 p3imm::direct_y(nullptr, ld, lT), ld, ld, lT};
        o.ACC[i] = commit_acc(w.acc[i], le, {mi}, R);
    }
    o.YQ = commit_col_nc(w.yq, R);
    o.YK = commit_col_nc(w.yk, R);
    o.YV = commit_col_nc(w.yv, R);
    o.RQ = commit_col_nc(w.rq, R);
    o.RK = commit_col_nc(w.rk, R);
    {
        std::vector<MInst> mis;
        for (uint32_t b = 0; b < B; b++)
            for (uint32_t h = 0; h < nh; h++) {
                MInst mi{xview_qk(c, h, b), wview_qk(c, h, b),
                         yview_qk(c, b * nh + h), ldh, lseq, lseq};
                mi.xv.c = &o.RQ; mi.wv.c = &o.RK;
                mis.push_back(mi);
            }
        o.SC = commit_acc(w.sc, lez, mis, R);
    }
    o.Z = commit_col_nc(w.z, R);
    o.P = commit_col_nc(w.p, R);
    {
        std::vector<MInst> mis;
        for (uint32_t b = 0; b < B; b++)
            for (uint32_t h = 0; h < nh; h++) {
                MInst mi{xview_pv(c, b * nh + h), wview_pv(c, h, b),
                         yview_pv(c, h, b), lseq, ldh, lseq};
                mi.xv.c = &o.P; mi.wv.c = &o.YV;
                mis.push_back(mi);
            }
        o.PVA = commit_acc(w.pva, le, mis, R);
    }
    o.AT = commit_col_nc(w.at, R);
    {
        MInst mi{p3imm::direct_x(&o.AT, ld, lT), p3imm::direct_w(&o.W[W_O], ld, ld),
                 p3imm::direct_y(nullptr, ld, lT), ld, ld, lT};
        o.ACC[W_O] = commit_acc(w.acc[W_O], le, {mi}, R);
    }
    o.YO = commit_col_nc(w.yo, R);
    o.RES1 = commit_col_nc(w.res1, R);
    o.H2 = commit_col_nc(w.h2, R);
    for (int i : {W_G, W_U}) {
        MInst mi{p3imm::direct_x(&o.H2, ld, lT), p3imm::direct_w(&o.W[i], ld, ldff),
                 p3imm::direct_y(nullptr, ldff, lT), ld, ldff, lT};
        o.ACC[i] = commit_acc(w.acc[i], lef, {mi}, R);
    }
    o.GG = commit_col_nc(w.gg, R);
    o.UU = commit_col_nc(w.uu, R);
    o.MO = commit_col_nc(w.mo, R);
    {
        MInst mi{p3imm::direct_x(&o.MO, ldff, lT), p3imm::direct_w(&o.W[W_D], ldff, ld),
                 p3imm::direct_y(nullptr, ld, lT), ldff, ld, lT};
        o.ACC[W_D] = commit_acc(w.acc[W_D], le, {mi}, R);
    }
    o.YD = commit_col_nc(w.yd, R);
    o.OUT = commit_col_nc(w.out, R);
    return o;
}

// public weight/gain roots (independent of the prover in non-zk mode)
struct WeightRoots { Hash Wc[NW], G1, G2; };
static inline WeightRoots weight_roots(const Weights& W, uint32_t R) {
    WeightRoots wr;
    for (int i = 0; i < NW; i++) wr.Wc[i] = commit_col_nc(W.w[i], R).root;
    wr.G1 = commit_col_nc(W.g1, R).root;
    wr.G2 = commit_col_nc(W.g2, R).root;
    return wr;
}

// ---------------- proof object ----------------
struct TfProof {
    uint32_t seq = 0, d = 0, nh = 0, dh = 0, dff = 0, batch = 1;
    p3irms::Proof rms1, rms2;
    p3imm::Proof mm[NW];
    std::vector<p3imm::Proof> mmqk, mmpv;
    p3irs::Proof rsq, rsk, rsv, rsz, rsat, rso, rsg, rsu, rsd;
    p3irope::Proof rpq, rpk;
    p3ismx::Proof smx;
    p3iswg::Proof swg;
    p3iadd::Proof add1, add2;
    Hash rX0, rH1, rACC[NW], rYQ, rYK, rYV, rRQ, rRK, rSC, rZ, rP, rPVA, rAT,
         rYO, rRES1, rH2, rGG, rUU, rMO, rYD, rOUT;
    std::vector<p3lu::GroupProof> lug;
    gl_t ioin = 0, ioout = 0;
    std::vector<p3bo::BatchProof> batches;
};

struct TfProf {
    double commit = 0, rms = 0, irs = 0, mm = 0, rope = 0, smx = 0, add = 0,
           swg = 0, lug = 0, io = 0, batch = 0, total = 0;
};

// ==================== prover ====================
static inline TfProof prove(fs::Transcript& tr, const TfWit& w, const TfOps& o,
                            const Weights& WW, const Tables& T, uint32_t R, uint32_t Q,
                            bool strict = true, TfProf* prof = nullptr) {
    TfProof pf;
    TfProf pl; TfProf& P = prof ? *prof : pl;
    double tall = now_ms(), tp;
    const Config& c = w.cfg;
    pf.seq = c.seq; pf.d = c.d; pf.nh = c.nh; pf.dh = c.dh; pf.dff = c.dff; pf.batch = c.batch;
    const uint32_t ld = c.ld(), lT = c.lT(), ldff = c.ldff(), lseq = c.lseq(),
                   ldh = c.ldh(), la = c.la(), nh = c.nh, B = c.batch;
    const uint32_t le = ld + lT, lef = ldff + lT, lez = la + 2 * lseq;
    uint32_t hdr[6] = {c.seq, c.d, c.nh, c.dh, c.dff, c.batch};
    tr.absorb("itf-dims", hdr, sizeof hdr);

    pf.rX0 = o.X0.root; pf.rH1 = o.H1.root;
    for (int i = 0; i < NW; i++) pf.rACC[i] = o.ACC[i].root;
    pf.rYQ = o.YQ.root; pf.rYK = o.YK.root; pf.rYV = o.YV.root;
    pf.rRQ = o.RQ.root; pf.rRK = o.RK.root; pf.rSC = o.SC.root;
    pf.rZ = o.Z.root; pf.rP = o.P.root; pf.rPVA = o.PVA.root; pf.rAT = o.AT.root;
    pf.rYO = o.YO.root; pf.rRES1 = o.RES1.root; pf.rH2 = o.H2.root;
    pf.rGG = o.GG.root; pf.rUU = o.UU.root; pf.rMO = o.MO.root;
    pf.rYD = o.YD.root; pf.rOUT = o.OUT.root;

    p3lu::XCtx xc;
    // statement-column range lookups (weights, gains, input)
    for (int i = 0; i < NW; i++)
        p3lu::defer_v(xc, {LC(&o.W[i])}, w.idxW[i], T.RS20, "itfW" + std::to_string(i));
    p3lu::defer_v(xc, {LC(&o.G1)}, w.idxG1, T.RS20, "itfG1");
    p3lu::defer_v(xc, {LC(&o.G2)}, w.idxG2, T.RS20, "itfG2");
    p3lu::defer_v(xc, {LC(&o.X0)}, w.idxX0, T.RS20, "itfX0");

    // -- sub-proofs, fixed order, one transcript, one ledger --
    tp = now_ms();
    pf.rms1 = p3irms::prove(tr, w.rms1, T, {&o.X0, &o.G1, &o.H1}, R, Q, strict, &xc, "irms1");
    P.rms += now_ms() - tp; tp = now_ms();
    for (int i : {W_Q, W_K, W_V}) {
        OpView xv = p3imm::direct_x(&o.H1, ld, lT);
        OpView wv = p3imm::direct_w(&o.W[i], ld, ld);
        OpView yv = p3imm::direct_y(&o.ACC[i], ld, lT);
        pf.mm[i] = p3imm::prove(tr, xv, wv, yv, ld, ld, lT, R, Q, strict, &xc);
    }
    P.mm += now_ms() - tp; tp = now_ms();
    pf.rsq = p3irs::prove(tr, w.rsq, T, {&o.ACC[W_Q], &o.YQ}, R, Q, strict, &xc, "irsq");
    pf.rsk = p3irs::prove(tr, w.rsk, T, {&o.ACC[W_K], &o.YK}, R, Q, strict, &xc, "irsk");
    pf.rsv = p3irs::prove(tr, w.rsv, T, {&o.ACC[W_V], &o.YV}, R, Q, strict, &xc, "irsv");
    P.irs += now_ms() - tp; tp = now_ms();
    pf.rpq = p3irope::prove(tr, w.rpq, T, WW.rp, {&o.YQ, &o.RQ}, R, Q, strict, &xc, "irpq");
    pf.rpk = p3irope::prove(tr, w.rpk, T, WW.rp, {&o.YK, &o.RK}, R, Q, strict, &xc, "irpk");
    P.rope += now_ms() - tp; tp = now_ms();
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t h = 0; h < nh; h++) {
            OpView xv = xview_qk(c, h, b); xv.c = &o.RQ;
            OpView wv = wview_qk(c, h, b); wv.c = &o.RK;
            OpView yv = yview_qk(c, b * nh + h); yv.c = &o.SC;
            pf.mmqk.push_back(p3imm::prove(tr, xv, wv, yv, ldh, lseq, lseq, R, Q, strict, &xc));
        }
    P.mm += now_ms() - tp; tp = now_ms();
    pf.rsz = p3irs::prove(tr, w.rsz, T, {&o.SC, &o.Z}, R, Q, strict, &xc, "irsz");
    P.irs += now_ms() - tp; tp = now_ms();
    pf.smx = p3ismx::prove(tr, w.smx, T, {&o.Z, &o.P}, R, Q, strict, &xc);
    P.smx += now_ms() - tp; tp = now_ms();
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t h = 0; h < nh; h++) {
            OpView xv = xview_pv(c, b * nh + h); xv.c = &o.P;
            OpView wv = wview_pv(c, h, b); wv.c = &o.YV;
            OpView yv = yview_pv(c, h, b); yv.c = &o.PVA;
            pf.mmpv.push_back(p3imm::prove(tr, xv, wv, yv, lseq, ldh, lseq, R, Q, strict, &xc));
        }
    P.mm += now_ms() - tp; tp = now_ms();
    pf.rsat = p3irs::prove(tr, w.rsat, T, {&o.PVA, &o.AT}, R, Q, strict, &xc, "irsa");
    P.irs += now_ms() - tp; tp = now_ms();
    {
        OpView xv = p3imm::direct_x(&o.AT, ld, lT);
        OpView wv = p3imm::direct_w(&o.W[W_O], ld, ld);
        OpView yv = p3imm::direct_y(&o.ACC[W_O], ld, lT);
        pf.mm[W_O] = p3imm::prove(tr, xv, wv, yv, ld, ld, lT, R, Q, strict, &xc);
    }
    P.mm += now_ms() - tp; tp = now_ms();
    pf.rso = p3irs::prove(tr, w.rso, T, {&o.ACC[W_O], &o.YO}, R, Q, strict, &xc, "irso");
    P.irs += now_ms() - tp; tp = now_ms();
    pf.add1 = p3iadd::prove(tr, w.add1, T, {&o.X0, &o.YO, &o.RES1}, R, Q, strict, &xc);
    P.add += now_ms() - tp; tp = now_ms();
    pf.rms2 = p3irms::prove(tr, w.rms2, T, {&o.RES1, &o.G2, &o.H2}, R, Q, strict, &xc, "irms2");
    P.rms += now_ms() - tp; tp = now_ms();
    for (int i : {W_G, W_U}) {
        OpView xv = p3imm::direct_x(&o.H2, ld, lT);
        OpView wv = p3imm::direct_w(&o.W[i], ld, ldff);
        OpView yv = p3imm::direct_y(&o.ACC[i], ldff, lT);
        pf.mm[i] = p3imm::prove(tr, xv, wv, yv, ld, ldff, lT, R, Q, strict, &xc);
    }
    P.mm += now_ms() - tp; tp = now_ms();
    pf.rsg = p3irs::prove(tr, w.rsg, T, {&o.ACC[W_G], &o.GG}, R, Q, strict, &xc, "irsg");
    pf.rsu = p3irs::prove(tr, w.rsu, T, {&o.ACC[W_U], &o.UU}, R, Q, strict, &xc, "irsu");
    P.irs += now_ms() - tp; tp = now_ms();
    pf.swg = p3iswg::prove(tr, w.swg, T, {&o.GG, &o.UU, &o.MO}, R, Q, strict, &xc);
    P.swg += now_ms() - tp; tp = now_ms();
    {
        OpView xv = p3imm::direct_x(&o.MO, ldff, lT);
        OpView wv = p3imm::direct_w(&o.W[W_D], ldff, ld);
        OpView yv = p3imm::direct_y(&o.ACC[W_D], ld, lT);
        pf.mm[W_D] = p3imm::prove(tr, xv, wv, yv, ldff, ld, lT, R, Q, strict, &xc);
    }
    P.mm += now_ms() - tp; tp = now_ms();
    pf.rsd = p3irs::prove(tr, w.rsd, T, {&o.ACC[W_D], &o.YD}, R, Q, strict, &xc, "irsd");
    P.irs += now_ms() - tp; tp = now_ms();
    pf.add2 = p3iadd::prove(tr, w.add2, T, {&o.RES1, &o.YD, &o.OUT}, R, Q, strict, &xc);
    P.add += now_ms() - tp;

    // -- ONE merged-lookup flush --
    tp = now_ms();
    pf.lug = p3lu::lu_flush(tr, xc, R, Q, strict);
    P.lug += now_ms() - tp;

    // -- public IO bindings (real-slice claims of PUBLIC values) --
    tp = now_ms();
    {
        std::vector<gl_t> z = chal_vec(tr, le);
        pf.ioin = claimc(tr, xc.lg, o.X0, p3zkc::zpt(z));
        std::vector<gl_t> z2 = chal_vec(tr, le);
        pf.ioout = claimc(tr, xc.lg, o.OUT, p3zkc::zpt(z2));
    }
    P.io += now_ms() - tp;

    // -- ONE batched opening pass per size class --
    tp = now_ms();
    for (size_t i = 0; i < xc.lg.cls.size(); i++)
        pf.batches.push_back(p3bo::prove_class(tr, xc.lg.cls[i], R, Q,
                                               "itf-bo" + std::to_string(i),
                                               &xc.lg.resolve, &xc.lg.dresolve));
    P.batch += now_ms() - tp;
    P.total += now_ms() - tall;
    return pf;
}

// ==================== verifier ====================
static inline bool verify(fs::Transcript& tr, const TfProof& pf, const Tables& T,
                          const Config& c, const std::vector<gl_t>& x0pub,
                          const Hash& rX0, const WeightRoots& wr,
                          const p3irope::Pub& rp, const std::vector<gl_t>& outpub,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (!c.pow2()) return fail("config must be pow2");
    if (pf.seq != c.seq || pf.d != c.d || pf.nh != c.nh || pf.dh != c.dh
        || pf.dff != c.dff || pf.batch != c.batch) return fail("dims mismatch");
    const uint32_t ld = c.ld(), lT = c.lT(), ldff = c.ldff(), lseq = c.lseq(),
                   ldh = c.ldh(), la = c.la(), nh = c.nh, B = c.batch;
    const uint32_t le = ld + lT, lef = ldff + lT, lez = la + 2 * lseq;
    if (x0pub.size() != ((size_t)1 << le) || outpub.size() != ((size_t)1 << le))
        return fail("public i/o size");
    if (!(pf.rX0 == rX0)) return fail("input root mismatch");
    if (pf.mmqk.size() != c.A() || pf.mmpv.size() != c.A()) return fail("proof shape");
    if (rp.lseq != lseq || rp.ldh != ldh) return fail("rope table dims");

    uint32_t hdr[6] = {c.seq, c.d, c.nh, c.dh, c.dff, c.batch};
    tr.absorb("itf-dims", hdr, sizeof hdr);

    p3lu::VCtx vc;
    for (int i = 0; i < NW; i++)
        p3lu::vdefer_v(vc, {&wr.Wc[i]}, T.RS20,
                       ilog2((size_t)((i == W_D) ? c.dff : c.d)
                             * ((i == W_G || i == W_U) ? c.dff : c.d)),
                       "itfW" + std::to_string(i));
    p3lu::vdefer_v(vc, {&wr.G1}, T.RS20, ld, "itfG1");
    p3lu::vdefer_v(vc, {&wr.G2}, T.RS20, ld, "itfG2");
    p3lu::vdefer_v(vc, {&rX0}, T.RS20, le, "itfX0");

    if (!p3irms::verify(tr, T, pf.rms1, rX0, wr.G1, pf.rH1, lT, ld, Q_pub, R_pub,
                        why, &vc, "irms1")) return false;
    for (int i : {W_Q, W_K, W_V}) {
        OpView xv = p3imm::direct_x(nullptr, ld, lT); xv.root = pf.rH1;
        OpView wv = p3imm::direct_w(nullptr, ld, ld); wv.root = wr.Wc[i];
        OpView yv = p3imm::direct_y(nullptr, ld, lT); yv.root = pf.rACC[i];
        if (!p3imm::verify(tr, pf.mm[i], xv, wv, yv, ld, ld, lT, Q_pub, R_pub, why, &vc))
            return false;
    }
    if (!p3irs::verify(tr, T, pf.rsq, pf.rACC[W_Q], pf.rYQ, le, 16, p3irs::RNG_S20,
                       Q_pub, R_pub, why, &vc, "irsq")) return false;
    if (!p3irs::verify(tr, T, pf.rsk, pf.rACC[W_K], pf.rYK, le, 16, p3irs::RNG_S20,
                       Q_pub, R_pub, why, &vc, "irsk")) return false;
    if (!p3irs::verify(tr, T, pf.rsv, pf.rACC[W_V], pf.rYV, le, 16, p3irs::RNG_S20,
                       Q_pub, R_pub, why, &vc, "irsv")) return false;
    if (!p3irope::verify(tr, T, rp, pf.rpq, pf.rYQ, pf.rRQ, lT, ld, Q_pub, R_pub,
                         why, &vc, "irpq")) return false;
    if (!p3irope::verify(tr, T, rp, pf.rpk, pf.rYK, pf.rRK, lT, ld, Q_pub, R_pub,
                         why, &vc, "irpk")) return false;
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t h = 0; h < nh; h++) {
            OpView xv = xview_qk(c, h, b); xv.root = pf.rRQ;
            OpView wv = wview_qk(c, h, b); wv.root = pf.rRK;
            OpView yv = yview_qk(c, b * nh + h); yv.root = pf.rSC;
            if (!p3imm::verify(tr, pf.mmqk[b * nh + h], xv, wv, yv, ldh, lseq, lseq,
                               Q_pub, R_pub, why, &vc)) return false;
        }
    if (!p3irs::verify(tr, T, pf.rsz, pf.rSC, pf.rZ, lez, c.sfz(), p3irs::RNG_S16,
                       Q_pub, R_pub, why, &vc, "irsz")) return false;
    if (!p3ismx::verify(tr, T, pf.smx, pf.rZ, pf.rP, la, lseq, Q_pub, R_pub, why, &vc))
        return false;
    for (uint32_t b = 0; b < B; b++)
        for (uint32_t h = 0; h < nh; h++) {
            OpView xv = xview_pv(c, b * nh + h); xv.root = pf.rP;
            OpView wv = wview_pv(c, h, b); wv.root = pf.rYV;
            OpView yv = yview_pv(c, h, b); yv.root = pf.rPVA;
            if (!p3imm::verify(tr, pf.mmpv[b * nh + h], xv, wv, yv, lseq, ldh, lseq,
                               Q_pub, R_pub, why, &vc)) return false;
        }
    if (!p3irs::verify(tr, T, pf.rsat, pf.rPVA, pf.rAT, le, 16, p3irs::RNG_S20,
                       Q_pub, R_pub, why, &vc, "irsa")) return false;
    {
        OpView xv = p3imm::direct_x(nullptr, ld, lT); xv.root = pf.rAT;
        OpView wv = p3imm::direct_w(nullptr, ld, ld); wv.root = wr.Wc[W_O];
        OpView yv = p3imm::direct_y(nullptr, ld, lT); yv.root = pf.rACC[W_O];
        if (!p3imm::verify(tr, pf.mm[W_O], xv, wv, yv, ld, ld, lT, Q_pub, R_pub, why, &vc))
            return false;
    }
    if (!p3irs::verify(tr, T, pf.rso, pf.rACC[W_O], pf.rYO, le, 16, p3irs::RNG_S20,
                       Q_pub, R_pub, why, &vc, "irso")) return false;
    if (!p3iadd::verify(tr, T, pf.add1, rX0, pf.rYO, pf.rRES1, le, Q_pub, R_pub,
                        why, &vc)) return false;
    if (!p3irms::verify(tr, T, pf.rms2, pf.rRES1, wr.G2, pf.rH2, lT, ld, Q_pub, R_pub,
                        why, &vc, "irms2")) return false;
    for (int i : {W_G, W_U}) {
        OpView xv = p3imm::direct_x(nullptr, ld, lT); xv.root = pf.rH2;
        OpView wv = p3imm::direct_w(nullptr, ld, ldff); wv.root = wr.Wc[i];
        OpView yv = p3imm::direct_y(nullptr, ldff, lT); yv.root = pf.rACC[i];
        if (!p3imm::verify(tr, pf.mm[i], xv, wv, yv, ld, ldff, lT, Q_pub, R_pub, why, &vc))
            return false;
    }
    if (!p3irs::verify(tr, T, pf.rsg, pf.rACC[W_G], pf.rGG, lef, 16, p3irs::RNG_S20,
                       Q_pub, R_pub, why, &vc, "irsg")) return false;
    if (!p3irs::verify(tr, T, pf.rsu, pf.rACC[W_U], pf.rUU, lef, 16, p3irs::RNG_S20,
                       Q_pub, R_pub, why, &vc, "irsu")) return false;
    if (!p3iswg::verify(tr, T, pf.swg, pf.rGG, pf.rUU, pf.rMO, lef, Q_pub, R_pub,
                        why, &vc)) return false;
    {
        OpView xv = p3imm::direct_x(nullptr, ldff, lT); xv.root = pf.rMO;
        OpView wv = p3imm::direct_w(nullptr, ldff, ld); wv.root = wr.Wc[W_D];
        OpView yv = p3imm::direct_y(nullptr, ld, lT); yv.root = pf.rACC[W_D];
        if (!p3imm::verify(tr, pf.mm[W_D], xv, wv, yv, ldff, ld, lT, Q_pub, R_pub, why, &vc))
            return false;
    }
    if (!p3irs::verify(tr, T, pf.rsd, pf.rACC[W_D], pf.rYD, le, 16, p3irs::RNG_S20,
                       Q_pub, R_pub, why, &vc, "irsd")) return false;
    if (!p3iadd::verify(tr, T, pf.add2, pf.rRES1, pf.rYD, pf.rOUT, le, Q_pub, R_pub,
                        why, &vc)) return false;

    // -- merged-lookup flush --
    if (!p3lu::lu_verify_flush(tr, vc, pf.lug, Q_pub, R_pub, why)) return false;

    // -- public IO bindings --
    {
        std::vector<gl_t> z = chal_vec(tr, le);
        gl_t y = claimv(tr, vc.vlg, pf.rX0, p3zkc::zpt(z), pf.ioin);
        if (y != p3bf::eval_h(x0pub, p3bf::build_eq(z))) return fail("public input binding");
        std::vector<gl_t> z2 = chal_vec(tr, le);
        gl_t y2 = claimv(tr, vc.vlg, pf.rOUT, p3zkc::zpt(z2), pf.ioout);
        if (y2 != p3bf::eval_h(outpub, p3bf::build_eq(z2))) return fail("public output binding");
    }

    // -- the ONE shared batched opening per size class --
    if (pf.batches.size() != vc.vlg.cls.size()) return fail("batch count");
    for (size_t i = 0; i < vc.vlg.cls.size(); i++)
        if (!p3bo::verify_class(tr, vc.vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                "itf-bo" + std::to_string(i), why)) return false;
    if (why) *why = "ok";
    return true;
}

// ---------------- proof size (serialized-payload bytes) ----------------
static inline size_t proof_size(const TfProof& pf) {
    size_t s = 24 + 32 * (size_t)(21 + NW);
    s += p3irms::proof_size(pf.rms1) + p3irms::proof_size(pf.rms2);
    for (int i = 0; i < NW; i++) s += p3imm::proof_size(pf.mm[i]);
    for (auto& m : pf.mmqk) s += p3imm::proof_size(m);
    for (auto& m : pf.mmpv) s += p3imm::proof_size(m);
    for (auto* r : {&pf.rsq, &pf.rsk, &pf.rsv, &pf.rsz, &pf.rsat, &pf.rso,
                    &pf.rsg, &pf.rsu, &pf.rsd}) s += p3irs::proof_size(*r);
    s += p3irope::proof_size(pf.rpq) + p3irope::proof_size(pf.rpk);
    s += p3ismx::proof_size(pf.smx) + p3iswg::proof_size(pf.swg);
    s += p3iadd::proof_size(pf.add1) + p3iadd::proof_size(pf.add2);
    for (auto& g : pf.lug) s += p3lu::sz_group(g);
    s += 16;
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3itf
