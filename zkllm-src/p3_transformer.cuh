// Composed FULL-TRANSFORMER-LAYER prover+verifier (design doc section 11.6
// R3+R5): proves the ENTIRE layer dataflow of transformer_ref.py,
//
//   x - rmsnorm(g1) - quant -[Wq,Wk,Wv]- rope(Q,K) -
//     per (batch, head) [ quant - QK^T - softmax - quant - (quant V^T) - P.V ]
//   - concat - quant - Wo - residual - rmsnorm(g2) - quant -[Wg,Wu]- swiglu -
//   quant - Wd - residual - out
//
// as ONE proof: 2 rmsnorm + (4 + 4A) quantize + (7 + 2A) Hawkeye matmuls +
// 2A rope + A softmax + 2 residual bf16-adds + 1 swiglu instances (A =
// batch*n_heads attention instances) over ONE Fiat-Shamir transcript with ONE
// shared opening ledger (p3lu::XCtx), one layer-level merged-lookup flush
// (R3b) and a single per-size-class batched-opening pass at the end (R3).
//
// BATCH GENERALIZATION: tokens t = b*seq + s (s low); the token-parallel ops
// (rmsnorm, quantize, projection/MLP matmuls, swiglu, residuals) run on ONE
// T = batch*seq row grid; attention (rope, QK^T, softmax, P.V) instantiates
// per (b, h) with head-slice/transpose/concat seams that fix BOTH the head
// bits of the model dimension and the batch bits of the token dimension.
// At batch=1, nh=2 the instance order and seam points coincide with the
// original fixed-shape layer (the 30/30 battery covers exactly that shape).
//
// CHAINING (the section-11.5 vocabulary, all four moves used):
//  * shared roots: wherever producer and consumer commit the SAME values on the
//    SAME domain, the composed verifier passes ONE root hash to both gadget
//    verifiers (rmsnorm.Y == quant.X; quant.SCALES == matmul.XS; rope.OUT ==
//    quant.X; softmax.P == quant.X; swiglu.M == quant.X; bfadd.OUT ==
//    rmsnorm.X == bfadd.X1; matmul output column rdo[O_YB] == softmax.S ==
//    swiglu.GATE/UP == bfadd.X2 -- the matmul's committed output witness
//    column IS the downstream operand commitment).
//  * restriction seams: quant CODES (rows x d) vs matmul X (rows x Kpad,
//    canonical zero padding): CODES~(z) == X~(z with the high k-bits fixed to
//    0), plus one zero-claim per padding subcube [2^t, 2^{t+1}).
//  * slice / transpose / concat seams: head slice = fix the head bits of the
//    model index AND the batch bits of the token index in the opening point;
//    V^T = swap the variable groups; concat = open the parent at the
//    partially-fixed point.  Each seam = TWO ledger claims that must agree at
//    a fresh random point (Schwartz-Zippel over the shared transcript, drawn
//    AFTER all commitments).
//  * public statement bindings: committed layer input == public x (verifier
//    evaluates the public MLE itself), committed final residual output ==
//    public out (bitwise binding of the accepted output to the golden trace).
//
// Public statement: input patterns + root, g1/g2 roots, the 7 weight
// (codes+scales) roots, the pinned rope cos/sin + canonical table artifacts,
// and the OUTPUT patterns.  Intermediate activation roots ride in the proof,
// pinned by the chain; non-zk mode additionally ships the matmuls' bf16
// outputs as public vectors (zk drops them, section 12).
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
#include "p3_rope.cuh"
#include "p3_quant.cuh"
#include "p3_softmax.cuh"
#include "p3_swiglu.cuh"
#include "fs_transcript.hpp"

namespace p3tf {

using p3lu::Col; using p3lu::commit_col_nc; using p3lu::chal_vec; using p3lu::ilog2;
using p3hwl::claimc; using p3hwl::claimv; using p3hwl::now_ms;
using p3rms::Art;
using p3rope::ins_pt; using p3rope::mle_eval;
using Hash = p3fri::Hash;

// ---------------- instance indices ----------------
// attention instances are flattened a = b*nh + h (a in [0, A), A = batch*nh);
// for the tiny batch=1, nh=2 config the legacy enum values below coincide
// with the dynamic indices, so the original test vocabulary keeps working.
enum { MM_WQ = 0, MM_WK, MM_WV, MM_QK0, MM_QK1, MM_PV0, MM_PV1,
       MM_WO, MM_WG, MM_WU, MM_WD, NMM };
enum { QN_H1 = 0, QN_RQ0, QN_RK0, QN_RQ1, QN_RK1, QN_PB0, QN_PB1,
       QN_VT0, QN_VT1, QN_AT, QN_H2, QN_SW, NQN };
static const int MM_QK[2] = {MM_QK0, MM_QK1};
static const int MM_PV[2] = {MM_PV0, MM_PV1};
static const int QN_RQ[2] = {QN_RQ0, QN_RQ1};
static const int QN_RK[2] = {QN_RK0, QN_RK1};
static const int QN_PB[2] = {QN_PB0, QN_PB1};
static const int QN_VT[2] = {QN_VT0, QN_VT1};

struct Config {
    uint32_t seq = 4, d = 64, nh = 2, dh = 32, dff = 128, batch = 1;
    uint32_t lseq() const { return ilog2(seq); }
    uint32_t ld()   const { return ilog2(d); }
    uint32_t ldh()  const { return ilog2(dh); }
    uint32_t ldff() const { return ilog2(dff); }
    uint32_t lnh()  const { return ilog2(nh); }
    uint32_t lbb()  const { return ilog2(batch); }
    uint32_t T()    const { return seq * batch; }        // total tokens
    uint32_t lT()   const { return lseq() + lbb(); }
    uint32_t A()    const { return batch * nh; }         // attention instances
    bool pow2() const {
        return (1u << lseq()) == seq && (1u << ld()) == d && (1u << ldh()) == dh
            && (1u << ldff()) == dff && (1u << lnh()) == nh
            && (1u << lbb()) == batch && nh * dh == d;
    }
    // dynamic instance indices (== the legacy enums at batch=1, nh=2)
    int nmm() const { return 7 + 2 * (int)A(); }
    int mmQK(uint32_t a) const { return 3 + (int)a; }
    int mmPV(uint32_t a) const { return 3 + (int)(A() + a); }
    int mmWO() const { return 3 + 2 * (int)A(); }
    int mmWG() const { return mmWO() + 1; }
    int mmWU() const { return mmWO() + 2; }
    int mmWD() const { return mmWO() + 3; }
    int nqn() const { return 4 + 4 * (int)A(); }
    int qnRQ(uint32_t a) const { return 1 + 2 * (int)a; }
    int qnRK(uint32_t a) const { return 2 + 2 * (int)a; }
    int qnPB(uint32_t a) const { return 1 + 2 * (int)A() + (int)a; }
    int qnVT(uint32_t a) const { return 1 + 3 * (int)A() + (int)a; }
    int qnAT() const { return 1 + 4 * (int)A(); }
    int qnH2() const { return qnAT() + 1; }
    int qnSW() const { return qnAT() + 2; }
};

// ---------------- weights (transformer_ref.py --dump-weights) ----------------
struct WeightMat { uint32_t N = 0, K = 0; std::vector<uint8_t> codes; std::vector<uint32_t> scales; };
enum { W_Q = 0, W_K, W_V, W_O, W_G, W_U, W_D, NW };
struct Weights {
    Config cfg;
    WeightMat w[NW];
    std::vector<uint16_t> g1, g2, cos, sin;
};
static inline bool load_weights(const char* path, Weights& W) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[7];
    if (fread(hdr, 8, 7, f) != 7 || hdr[0] != 0x54465754 || hdr[1] != NW) { fclose(f); return false; }
    W.cfg.seq = (uint32_t)hdr[2]; W.cfg.d = (uint32_t)hdr[3]; W.cfg.nh = (uint32_t)hdr[4];
    W.cfg.dh = (uint32_t)hdr[5]; W.cfg.dff = (uint32_t)hdr[6]; W.cfg.batch = 1;
    for (int i = 0; i < NW; i++) {
        int64_t nk[2];
        if (fread(nk, 8, 2, f) != 2) { fclose(f); return false; }
        W.w[i].N = (uint32_t)nk[0]; W.w[i].K = (uint32_t)nk[1];
        size_t n = (size_t)W.w[i].N * W.w[i].K;
        W.w[i].codes.resize(n); W.w[i].scales.resize(W.w[i].N);
        if (fread(W.w[i].codes.data(), 1, n, f) != n ||
            fread(W.w[i].scales.data(), 4, W.w[i].N, f) != W.w[i].N) { fclose(f); return false; }
    }
    size_t nh2 = (size_t)W.cfg.seq * (W.cfg.dh / 2);
    W.g1.resize(W.cfg.d); W.g2.resize(W.cfg.d); W.cos.resize(nh2); W.sin.resize(nh2);
    bool ok = fread(W.g1.data(), 2, W.cfg.d, f) == W.cfg.d
           && fread(W.g2.data(), 2, W.cfg.d, f) == W.cfg.d
           && fread(W.cos.data(), 2, nh2, f) == nh2
           && fread(W.sin.data(), 2, nh2, f) == nh2;
    fclose(f);
    return ok;
}

// ---------------- golden trace (transformer_ref.py --dump-layer) ----------------
struct Trace { std::vector<std::string> names; std::vector<std::vector<uint16_t>> vals; };
static inline bool load_trace(const char* path, Trace& T) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[2];
    if (fread(hdr, 8, 2, f) != 2 || hdr[0] != 0x54524C59) { fclose(f); return false; }
    for (int64_t i = 0; i < hdr[1]; i++) {
        int64_t nl;
        if (fread(&nl, 8, 1, f) != 1) { fclose(f); return false; }
        std::string nm(nl, 0);
        if (fread(&nm[0], 1, nl, f) != (size_t)nl) { fclose(f); return false; }
        int64_t sh[2];
        if (fread(sh, 8, 2, f) != 2) { fclose(f); return false; }
        size_t n = (size_t)sh[0] * (sh[1] ? sh[1] : 1);
        std::vector<uint16_t> v(n);
        if (fread(v.data(), 2, n, f) != n) { fclose(f); return false; }
        T.names.push_back(nm); T.vals.push_back(std::move(v));
    }
    fclose(f);
    return true;
}
static inline const std::vector<uint16_t>* trace_get(const Trace& T, const char* nm) {
    for (size_t i = 0; i < T.names.size(); i++)
        if (T.names[i] == nm) return &T.vals[i];
    return nullptr;
}

// ---------------- composed-battery tamper vocabulary ----------------
enum { TFT_NONE = 0,
       // intermediate-value tampers: the committed value is flipped, the owning
       // gadget's internal witness stays honest, downstream replays honestly
       // from the flipped value -> the OWNING gadget must reject
       TFT_RMS1_Y, TFT_SMX_P, TFT_ROPE_OUT, TFT_RES1_OUT, TFT_SWG_M,
       TFT_MM_YPUB,      // matmul PUBLIC Y claim flipped (committed column honest)
       // seam breaks: EVERY sub-proof is valid on its own inputs; only a
       // composition seam claim can catch the mismatch
       TFT_SEAM_MMX,     // Wq matmul X code != quantizer CODES (restriction seam)
       TFT_SEAM_PVPAD,   // nonzero code smuggled into PV0's X k-padding (zero seam)
       TFT_SEAM_ROPEQ,   // rope q0 operand != head-0 slice of the Wq output
       TFT_SEAM_CONCAT,  // concat operand head-1 half != attnout1
       TFT_SEAM_VT,      // V^T quant input != transposed V slice
       // per-gadget witness forgeries, composed context (one per gadget kind)
       TFT_GW_MM, TFT_GW_RMS, TFT_GW_QNT, TFT_GW_SMX, TFT_GW_ROPE,
       TFT_GW_BFA, TFT_GW_SWG };

// ---------------- chained layer witness ----------------
struct TfWit {
    Config cfg;
    p3rms::Wit rms[2];
    std::vector<p3qnt::Wit> qn;
    std::vector<p3hwl::LayerWit> mm;
    p3rope::GoldenSet rgs;                        // cos/sin container
    std::vector<p3rope::Wit> rp;                  // q,k per attention instance
    std::vector<p3smx::Wit> sm;
    std::vector<uint8_t> smmask;                  // causal mask bytes (seq*seq)
    p3bfa::Wit res[2];
    p3swg::Wit sw;
    // chained activation patterns (REAL row-major grids)
    std::vector<uint16_t> x0, rms1y, attn, res1o, rms2y, swm, outp;
    std::vector<std::vector<uint16_t>> ropq, ropk, probs, vt;   // per instance a
    std::vector<std::vector<uint16_t>> mmY;       // honest matmul outputs
    std::vector<std::vector<uint16_t>> mmYpub;    // the PUBLIC per-matmul claims
};

// build the full chained witness from (input, weights): each gadget's canonical
// replay output feeds the next gadget's input -- the layer is COMPUTED by the
// gadget replays themselves, so a tamper anywhere propagates honestly and the
// proof must reject exactly at the owning gadget or seam.
static inline TfWit build_witness(const Config& cfg, const std::vector<uint16_t>& x0,
                                  const Weights& W, const Art& a, int tamper = TFT_NONE) {
    if (!cfg.pow2()) throw std::runtime_error("tf: config must be pow2");
    TfWit w; w.cfg = cfg;
    const uint32_t seq = cfg.seq, d = cfg.d, dh = cfg.dh, dff = cfg.dff,
                   B = cfg.batch, T = cfg.T(), A = cfg.A(), nh = cfg.nh;
    w.qn.resize(cfg.nqn()); w.mm.resize(cfg.nmm());
    w.rp.resize(2 * A); w.sm.resize(A);
    w.ropq.resize(A); w.ropk.resize(A); w.probs.resize(A); w.vt.resize(A);
    w.mmY.resize(cfg.nmm()); w.mmYpub.resize(cfg.nmm());
    w.x0 = x0;
    auto flip = [](std::vector<uint16_t>& v, size_t i) { v[i] ^= 1; };

    // -- rmsnorm 1 (all T tokens) --
    {
        p3rms::Golden g; g.B = T; g.d = d; g.x = w.x0; g.g = W.g1;
        p3rms::RTamper rt{p3rms::RT_RSQ, 0, 0};
        w.rms[0] = p3rms::gen_witness(g, a, tamper == TFT_GW_RMS ? &rt : nullptr);
    }
    w.rms1y = w.rms[0].Y;
    if (tamper == TFT_RMS1_Y) {
        flip(w.rms1y, 5);
        w.rms[0].ypat[5] = gl_add(w.rms[0].ypat[5], 1ULL); w.rms[0].Y[5] ^= 1;
    }

    // -- quantize h1 (shared by Wq/Wk/Wv) --
    {
        p3qnt::Golden g; g.B = T; g.d = d; g.x = w.rms1y;
        p3qnt::QTamper qt{p3qnt::QT_MAG, 0, 3};
        w.qn[QN_H1] = p3qnt::gen_witness(g, a, tamper == TFT_GW_QNT ? &qt : nullptr);
    }

    auto mmgold = [](uint32_t Bm, uint32_t K, uint32_t N,
                     const std::vector<uint8_t>& x, const std::vector<uint32_t>& xs,
                     const std::vector<uint8_t>& wc, const std::vector<uint32_t>& ws) {
        p3hwl::Golden L; L.B = Bm; L.K = K; L.N = N;
        L.x = x; L.w = wc; L.xs = xs; L.ws = ws; L.y.assign((size_t)Bm * N, 0);
        return L;
    };

    // -- Wq / Wk / Wv --
    {
        // TFT_SEAM_MMX: ALL THREE matmuls consume a code the quantizer never
        // produced (they share one X commitment) -- every sub-proof stays
        // valid; only the h1-codes restriction seam can catch it
        std::vector<uint8_t> cx = w.qn[QN_H1].C;
        if (tamper == TFT_SEAM_MMX) cx[7] ^= 1;
        p3hwl::Tamper hm{p3hwl::TM_STATE, 1, 1, 0};
        w.mm[MM_WQ] = p3hwl::gen_witness(
            mmgold(T, d, d, cx, w.qn[QN_H1].S, W.w[W_Q].codes, W.w[W_Q].scales), true,
            tamper == TFT_GW_MM ? &hm : nullptr);
        w.mm[MM_WK] = p3hwl::gen_witness(
            mmgold(T, d, d, cx, w.qn[QN_H1].S, W.w[W_K].codes, W.w[W_K].scales));
        w.mm[MM_WV] = p3hwl::gen_witness(
            mmgold(T, d, d, cx, w.qn[QN_H1].S, W.w[W_V].codes, W.w[W_V].scales));
        for (int i : {MM_WQ, MM_WK, MM_WV}) w.mmY[i] = w.mm[i].Y;
    }

    // -- attention, per (batch, head) instance --
    w.rgs.seq = seq; w.rgs.dh = dh; w.rgs.cos = W.cos; w.rgs.sin = W.sin;
    w.smmask.assign((size_t)seq * seq, 0);
    for (uint32_t i = 0; i < seq; i++)
        for (uint32_t j = 0; j <= i; j++) w.smmask[(size_t)i * seq + j] = 1;
    w.attn.assign((size_t)T * d, 0);
    for (uint32_t b = 0; b < B; b++)
    for (uint32_t h = 0; h < nh; h++) {
        const uint32_t ai = b * nh + h;
        auto slice_case = [&](const std::vector<uint16_t>& Y) {
            p3rope::GoldenSet::Case cc; cc.flags = 0; cc.q.resize((size_t)seq * dh);
            for (uint32_t p = 0; p < seq; p++)
                for (uint32_t j = 0; j < dh; j++)
                    cc.q[(size_t)p * dh + j] =
                        Y[((size_t)b * seq + p) * d + (size_t)h * dh + j];
            return cc;
        };
        // rope q
        {
            p3rope::GoldenSet::Case cq = slice_case(w.mmY[MM_WQ]);
            if (tamper == TFT_SEAM_ROPEQ && ai == 0) cq.q[3] ^= 1;
            w.rp[2 * ai] = p3rope::gen_witness(w.rgs, cq, a);
            w.ropq[ai].assign(w.rp[2 * ai].OUT.begin(), w.rp[2 * ai].OUT.end());
            if (tamper == TFT_ROPE_OUT && ai == 0) {
                flip(w.ropq[ai], 2);
                w.rp[2 * ai].opat[2] = gl_add(w.rp[2 * ai].opat[2], 1ULL);
            }
        }
        // rope k
        {
            p3rope::GoldenSet::Case ck = slice_case(w.mmY[MM_WK]);
            p3rope::RpTamper rt{p3rope::RPT_MUL, 0};
            w.rp[2 * ai + 1] = p3rope::gen_witness(w.rgs, ck, a,
                (tamper == TFT_GW_ROPE && ai == 1) ? &rt : nullptr);
            w.ropk[ai].assign(w.rp[2 * ai + 1].OUT.begin(), w.rp[2 * ai + 1].OUT.end());
        }
        // quantize rotated q / k
        {
            p3qnt::Golden g; g.B = seq; g.d = dh; g.x = w.ropq[ai];
            w.qn[cfg.qnRQ(ai)] = p3qnt::gen_witness(g, a);
            g.x = w.ropk[ai];
            w.qn[cfg.qnRK(ai)] = p3qnt::gen_witness(g, a);
        }
        // scores = QK^T
        w.mm[cfg.mmQK(ai)] = p3hwl::gen_witness(
            mmgold(seq, dh, seq, w.qn[cfg.qnRQ(ai)].C, w.qn[cfg.qnRQ(ai)].S,
                   w.qn[cfg.qnRK(ai)].C, w.qn[cfg.qnRK(ai)].S));
        w.mmY[cfg.mmQK(ai)] = w.mm[cfg.mmQK(ai)].Y;
        // softmax
        {
            p3smx::Golden g; g.B = seq; g.n = seq;
            g.s = w.mmY[cfg.mmQK(ai)]; g.msk = w.smmask;
            g.p.assign((size_t)seq * seq, 0);
            p3smx::SmTamper st{p3smx::SMT_MASKLEAK, 2, 3};
            w.sm[ai] = p3smx::gen_witness(g, a, (tamper == TFT_GW_SMX && ai == 0) ? &st : nullptr);
        }
        w.probs[ai].assign(w.sm[ai].P.begin(), w.sm[ai].P.end());
        if (tamper == TFT_SMX_P && ai == 0) {
            flip(w.probs[ai], 1);
            w.sm[ai].ppat[1] = gl_add(w.sm[ai].ppat[1], 1ULL);
        }
        // quantize probs
        {
            p3qnt::Golden g; g.B = seq; g.d = seq; g.x = w.probs[ai];
            w.qn[cfg.qnPB(ai)] = p3qnt::gen_witness(g, a);
        }
        // V^T head slice
        w.vt[ai].assign((size_t)dh * seq, 0);
        for (uint32_t j = 0; j < dh; j++)
            for (uint32_t p = 0; p < seq; p++)
                w.vt[ai][(size_t)j * seq + p] =
                    w.mmY[MM_WV][((size_t)b * seq + p) * d + (size_t)h * dh + j];
        if (tamper == TFT_SEAM_VT && ai == 0) flip(w.vt[ai], 5);
        {
            p3qnt::Golden g; g.B = dh; g.d = seq; g.x = w.vt[ai];
            w.qn[cfg.qnVT(ai)] = p3qnt::gen_witness(g, a);
        }
        // attnout = P.V
        {
            p3hwl::Golden L = mmgold(seq, seq, dh, w.qn[cfg.qnPB(ai)].C, w.qn[cfg.qnPB(ai)].S,
                                     w.qn[cfg.qnVT(ai)].C, w.qn[cfg.qnVT(ai)].S);
            if (tamper == TFT_SEAM_PVPAD && ai == 0) {
                // smuggle a nonzero code into group-0 k-padding: self-consistent
                // matmul of a value the quantizer never produced
                p3hwl::Dims dd = p3hwl::make_dims(seq, seq, dh);
                std::vector<uint8_t> ovr((size_t)dd.Bpad * dd.Kpad, 0);
                for (uint32_t bb = 0; bb < seq; bb++)
                    for (uint32_t k = 0; k < seq; k++)
                        ovr[(size_t)bb * dd.Kpad + k] = L.x[(size_t)bb * seq + k];
                ovr[0 * dd.Kpad + 5] = 0x30;
                w.mm[cfg.mmPV(ai)] = p3hwl::gen_witness(L, true, nullptr, &ovr);
            } else {
                w.mm[cfg.mmPV(ai)] = p3hwl::gen_witness(L);
            }
        }
        w.mmY[cfg.mmPV(ai)] = w.mm[cfg.mmPV(ai)].Y;
        for (uint32_t p = 0; p < seq; p++)
            for (uint32_t j = 0; j < dh; j++)
                w.attn[((size_t)b * seq + p) * d + (size_t)h * dh + j] =
                    w.mmY[cfg.mmPV(ai)][(size_t)p * dh + j];
    }
    if (tamper == TFT_SEAM_CONCAT) flip(w.attn, (size_t)1 * d + dh + 2);   // head-1 region

    // -- quantize attn, Wo, residual 1 --
    {
        p3qnt::Golden g; g.B = T; g.d = d; g.x = w.attn;
        w.qn[cfg.qnAT()] = p3qnt::gen_witness(g, a);
    }
    w.mm[cfg.mmWO()] = p3hwl::gen_witness(
        mmgold(T, d, d, w.qn[cfg.qnAT()].C, w.qn[cfg.qnAT()].S, W.w[W_O].codes, W.w[W_O].scales));
    w.mmY[cfg.mmWO()] = w.mm[cfg.mmWO()].Y;
    {
        p3bfa::Golden g; g.n = T * d; g.flags = 0;
        g.a = w.x0; g.b = w.mmY[cfg.mmWO()]; g.o.assign(g.n, 0);
        w.res[0] = p3bfa::gen_witness(g);
    }
    w.res1o.assign(w.res[0].O.begin(), w.res[0].O.end());
    if (tamper == TFT_RES1_OUT) {
        flip(w.res1o, 9);
        w.res[0].opat[9] = gl_add(w.res[0].opat[9], 1ULL);
    }

    // -- rmsnorm 2, quantize h2, Wg / Wu --
    {
        p3rms::Golden g; g.B = T; g.d = d; g.x = w.res1o; g.g = W.g2;
        w.rms[1] = p3rms::gen_witness(g, a);
    }
    w.rms2y = w.rms[1].Y;
    {
        p3qnt::Golden g; g.B = T; g.d = d; g.x = w.rms2y;
        w.qn[cfg.qnH2()] = p3qnt::gen_witness(g, a);
    }
    w.mm[cfg.mmWG()] = p3hwl::gen_witness(
        mmgold(T, d, dff, w.qn[cfg.qnH2()].C, w.qn[cfg.qnH2()].S, W.w[W_G].codes, W.w[W_G].scales));
    w.mm[cfg.mmWU()] = p3hwl::gen_witness(
        mmgold(T, d, dff, w.qn[cfg.qnH2()].C, w.qn[cfg.qnH2()].S, W.w[W_U].codes, W.w[W_U].scales));
    w.mmY[cfg.mmWG()] = w.mm[cfg.mmWG()].Y; w.mmY[cfg.mmWU()] = w.mm[cfg.mmWU()].Y;

    // -- swiglu, quantize, Wd, residual 2 --
    {
        p3swg::Golden g; g.n = T * dff;
        g.gate = w.mmY[cfg.mmWG()]; g.up = w.mmY[cfg.mmWU()]; g.m.assign(g.n, 0);
        p3swg::STamper st{p3swg::ST_SILU, 6};
        w.sw = p3swg::gen_witness(g, a, tamper == TFT_GW_SWG ? &st : nullptr);
    }
    w.swm.assign(w.sw.M.begin(), w.sw.M.end());
    if (tamper == TFT_SWG_M) {
        flip(w.swm, 4);
        w.sw.mpat[4] = gl_add(w.sw.mpat[4], 1ULL);
    }
    {
        p3qnt::Golden g; g.B = T; g.d = dff; g.x = w.swm;
        w.qn[cfg.qnSW()] = p3qnt::gen_witness(g, a);
    }
    w.mm[cfg.mmWD()] = p3hwl::gen_witness(
        mmgold(T, dff, d, w.qn[cfg.qnSW()].C, w.qn[cfg.qnSW()].S, W.w[W_D].codes, W.w[W_D].scales));
    w.mmY[cfg.mmWD()] = w.mm[cfg.mmWD()].Y;
    {
        p3bfa::Golden g; g.n = T * d; g.flags = 0;
        g.a = w.res1o; g.b = w.mmY[cfg.mmWD()]; g.o.assign(g.n, 0);
        p3bfa::BaTamper bt; bt.mode = p3bfa::BAT_RUP;
        uint32_t tj = 0;
        if (tamper == TFT_GW_BFA) {           // find a row where the RNE flip bites
            for (uint32_t j = 0; j < g.n; j++) {
                p3bfa::BaVals hon = p3bfa::ba_fill(g.a[j], g.b[j]);
                p3bfa::BaVals bad = p3bfa::ba_fill(g.a[j], g.b[j], &bt);
                if (bad.out != hon.out) { tj = j; break; }
            }
        }
        w.res[1] = p3bfa::gen_witness(g, tamper == TFT_GW_BFA ? &bt : nullptr, tj);
    }
    w.outp.assign(w.res[1].O.begin(), w.res[1].O.end());

    for (int i = 0; i < cfg.nmm(); i++) w.mmYpub[i] = w.mmY[i];
    if (tamper == TFT_MM_YPUB) w.mmYpub[cfg.mmWO()][3] ^= 1;   // committed column stays honest
    return w;
}

// ---------------- shared tables ----------------
struct TfTables {
    p3hwl::Tables hw;
    p3rms::Tables rms;
    p3qnt::Tables qnt;
    p3rope::Tables rope;
    p3smx::Tables smx;
    p3bfa::Tables bfa;
    p3swg::Tables swg;
};
// smx_wmax: softmax denominator window; rows of length seq need >= 16+lseq
// (the default 23 covers seq <= 128 and keeps historical transcripts identical).
static inline TfTables build_tables(const Art& a, uint32_t smx_wmax = 23) {
    TfTables T;
    T.hw = p3hwl::build_tables();
    T.rms = p3rms::build_tables(a);
    T.qnt = p3qnt::build_tables(a);
    T.rope = p3rope::build_tables(a);
    T.smx = p3smx::build_tables(a, smx_wmax);
    T.bfa = p3bfa::build_tables();
    T.swg = p3swg::build_tables(a);
    return T;
}

// ---------------- operand commitments (the chain, committed once) ----------------
struct TfOps {
    p3rms::Operands rms[2];
    std::vector<p3qnt::Operands> qn;
    std::vector<p3hwl::Operands> mm;
    std::vector<p3rope::Operands> rp;
    std::vector<p3smx::Operands> sm;
    p3bfa::Operands res[2];
    p3swg::Operands sw;
    std::vector<Col> Ymm;                         // matmul output columns [n|tok]
};
static inline Col commit_u8(const std::vector<uint8_t>& v, uint32_t R,
                            const std::vector<gl_t>* zkmask = nullptr) {
    std::vector<gl_t> g(v.begin(), v.end());
    return commit_col_nc(std::move(g), R, zkmask);
}
// zk seam-linkage helpers (design doc section 12) --------------------------
// producer slice-1 of a committed column
static inline std::vector<gl_t> psl1(const Col& c) { return p3zkc::slice1(c.v, c.vreal); }
// restriction consumer mask: X_real(inner<2^ldr, outer) = CODES(inner,outer),
// 0 on the k-padding -> slice 1 = zero-padded CODES slice 1
static inline std::vector<gl_t> restr_mask(const Col& codes, uint32_t ldr, uint32_t lk,
                                           uint32_t lb) {
    if (!p3zkc::G.on) return {};
    std::vector<gl_t> c1 = psl1(codes);
    size_t Nc = (size_t)1 << (lk + lb);
    std::vector<gl_t> cons1(Nc, 0);
    for (size_t idx = 0; idx < Nc; idx++) {
        size_t inner = idx & (((size_t)1 << lk) - 1), outer = idx >> lk;
        if (inner < ((size_t)1 << ldr))
            cons1[idx] = c1[inner | (outer << ldr)];
    }
    return p3zkc::mk_linked(lk + lb, cons1);
}
static inline TfOps commit_all(const TfWit& w, uint32_t R) {
    TfOps o;
    const bool zk = p3zkc::G.on;
    const Config& c = w.cfg;
    const uint32_t lseq = c.lseq(), ld = c.ld(), ldh = c.ldh(), ldff = c.ldff(),
                   lT = c.lT(), A = c.A(), nh = c.nh, seq = c.seq;
    o.qn.resize(c.nqn()); o.mm.resize(c.nmm()); o.rp.resize(2 * A);
    o.sm.resize(A); o.Ymm.resize(c.nmm());
    const uint32_t lkH1 = ilog2(w.mm[MM_WQ].d.Kpad), lkQK = ilog2(w.mm[c.mmQK(0)].d.Kpad),
                   lkPV = ilog2(w.mm[c.mmPV(0)].d.Kpad), lkSW = ilog2(w.mm[c.mmWD()].d.Kpad);
    // slice/transpose/concat consumer-mask builders from a producer Ymm slice 1.
    // rope-q/k: Q_real(j,s) = Ymm(h*dh+j, b*seq+s); Ymm layout [n | tok].
    auto head_slice_mask = [&](const Col& Ymm, uint32_t h, uint32_t b) -> std::vector<gl_t> {
        if (!zk) return {};
        std::vector<gl_t> y1 = psl1(Ymm);
        size_t Nc = (size_t)1 << (ldh + lseq);
        std::vector<gl_t> cons1(Nc, 0);
        for (uint32_t s = 0; s < seq; s++)
            for (uint32_t j = 0; j < c.dh; j++)
                cons1[j | ((size_t)s << ldh)] =
                    y1[((size_t)(h * c.dh + j)) | (((size_t)b * seq + s) << ld)];
        return p3zkc::mk_linked(ldh + lseq, cons1);
    };
    // V^T: VT.X(p in seq, j in dh) = Ymm_Wv(n=h*dh+j, b*seq+p); VT.X layout [p | j].
    auto vt_mask = [&](const Col& Ymm, uint32_t h, uint32_t b) -> std::vector<gl_t> {
        if (!zk) return {};
        std::vector<gl_t> y1 = psl1(Ymm);
        size_t Nc = (size_t)1 << (lseq + ldh);
        std::vector<gl_t> cons1(Nc, 0);
        for (uint32_t j = 0; j < c.dh; j++)
            for (uint32_t p = 0; p < seq; p++)
                cons1[p | ((size_t)j << lseq)] =
                    y1[((size_t)(h * c.dh + j)) | (((size_t)b * seq + p) << ld)];
        return p3zkc::mk_linked(lseq + ldh, cons1);
    };
    // concat: AT.X(n=h*dh+j, tok=b*seq+s) = Ymm_PV[a](j, s); AT.X layout [n | tok].
    auto concat_mask = [&]() -> std::vector<gl_t> {
        if (!zk) return {};
        size_t Nc = (size_t)1 << (ld + lT);
        std::vector<gl_t> cons1(Nc, 0);
        for (uint32_t b = 0; b < c.batch; b++)
            for (uint32_t h = 0; h < nh; h++) {
                std::vector<gl_t> y = psl1(o.Ymm[c.mmPV(b * nh + h)]);
                for (uint32_t s = 0; s < seq; s++)
                    for (uint32_t j = 0; j < c.dh; j++)
                        cons1[(size_t)(h * c.dh + j) | (((size_t)b * seq + s) << ld)] =
                            y[j | ((size_t)s << ldh)];
            }
        return p3zkc::mk_linked(ld + lT, cons1);
    };
    // rms1: X = committed layer input, G = g1, Y fresh
    o.rms[0] = p3rms::commit_operands(w.rms[0], R);
    // quant h1 shares rms1.Y as its X
    o.qn[QN_H1].X = o.rms[0].Y;
    o.qn[QN_H1].CODES = commit_col_nc(w.qn[QN_H1].cpat, R);
    o.qn[QN_H1].SCALES = commit_col_nc(w.qn[QN_H1].spat, R);
    // Wq/Wk/Wv: ONE padded X-code commitment (restriction of h1 codes), shared scales
    {
        std::vector<gl_t> xm = restr_mask(o.qn[QN_H1].CODES, ld, lkH1, lT);
        Col X1 = commit_u8(w.mm[MM_WQ].xcodes, R, (zk&&!p3zkc::G.nolink) ? &xm : nullptr);
        for (int i : {MM_WQ, MM_WK, MM_WV}) {
            o.mm[i].X = X1;
            o.mm[i].W = commit_u8(w.mm[i].wcodes, R);   // static weight (secret leaf, fresh mask)
            o.mm[i].XS = o.qn[QN_H1].SCALES;
            o.mm[i].WS = commit_col_nc(w.mm[i].wsb, R);
        }
    }
    for (int i : {MM_WQ, MM_WK, MM_WV})
        o.Ymm[i] = commit_col_nc(w.mm[i].dob[p3hwl::O_YB], R);
    for (uint32_t b = 0; b < c.batch; b++)
    for (uint32_t h = 0; h < nh; h++) {
        const uint32_t ai = b * nh + h;
        // rope q/k operands: Q mask = the (b,h) slice of the Wq/Wk output masks
        std::vector<gl_t> qmq = head_slice_mask(o.Ymm[MM_WQ], h, b);
        std::vector<gl_t> qmk = head_slice_mask(o.Ymm[MM_WK], h, b);
        o.rp[2 * ai] = p3rope::commit_operands(w.rp[2 * ai], R, (zk&&!p3zkc::G.nolink) ? &qmq : nullptr);
        o.rp[2 * ai + 1] = p3rope::commit_operands(w.rp[2 * ai + 1], R, (zk&&!p3zkc::G.nolink) ? &qmk : nullptr);
        o.qn[c.qnRQ(ai)].X = o.rp[2 * ai].OUT;
        o.qn[c.qnRQ(ai)].CODES = commit_col_nc(w.qn[c.qnRQ(ai)].cpat, R);
        o.qn[c.qnRQ(ai)].SCALES = commit_col_nc(w.qn[c.qnRQ(ai)].spat, R);
        o.qn[c.qnRK(ai)].X = o.rp[2 * ai + 1].OUT;
        o.qn[c.qnRK(ai)].CODES = commit_col_nc(w.qn[c.qnRK(ai)].cpat, R);
        o.qn[c.qnRK(ai)].SCALES = commit_col_nc(w.qn[c.qnRK(ai)].spat, R);
        std::vector<gl_t> qkx = restr_mask(o.qn[c.qnRQ(ai)].CODES, ldh, lkQK, lseq);
        std::vector<gl_t> qkw = restr_mask(o.qn[c.qnRK(ai)].CODES, ldh, lkQK, lseq);
        o.mm[c.mmQK(ai)].X = commit_u8(w.mm[c.mmQK(ai)].xcodes, R, (zk&&!p3zkc::G.nolink) ? &qkx : nullptr);
        o.mm[c.mmQK(ai)].W = commit_u8(w.mm[c.mmQK(ai)].wcodes, R, (zk&&!p3zkc::G.nolink) ? &qkw : nullptr);
        o.mm[c.mmQK(ai)].XS = o.qn[c.qnRQ(ai)].SCALES;
        o.mm[c.mmQK(ai)].WS = o.qn[c.qnRK(ai)].SCALES;
        o.Ymm[c.mmQK(ai)] = commit_col_nc(w.mm[c.mmQK(ai)].dob[p3hwl::O_YB], R);
        o.sm[ai].S = o.Ymm[c.mmQK(ai)];
        o.sm[ai].P = commit_col_nc(w.sm[ai].ppat, R);
        o.qn[c.qnPB(ai)].X = o.sm[ai].P;
        o.qn[c.qnPB(ai)].CODES = commit_col_nc(w.qn[c.qnPB(ai)].cpat, R);
        o.qn[c.qnPB(ai)].SCALES = commit_col_nc(w.qn[c.qnPB(ai)].spat, R);
        std::vector<gl_t> vtm = vt_mask(o.Ymm[MM_WV], h, b);
        o.qn[c.qnVT(ai)].X = commit_col_nc(w.qn[c.qnVT(ai)].xpat, R, (zk&&!p3zkc::G.nolink) ? &vtm : nullptr);
        o.qn[c.qnVT(ai)].CODES = commit_col_nc(w.qn[c.qnVT(ai)].cpat, R);
        o.qn[c.qnVT(ai)].SCALES = commit_col_nc(w.qn[c.qnVT(ai)].spat, R);
        std::vector<gl_t> pvx = restr_mask(o.qn[c.qnPB(ai)].CODES, lseq, lkPV, lseq);
        std::vector<gl_t> pvw = restr_mask(o.qn[c.qnVT(ai)].CODES, lseq, lkPV, ldh);
        o.mm[c.mmPV(ai)].X = commit_u8(w.mm[c.mmPV(ai)].xcodes, R, (zk&&!p3zkc::G.nolink) ? &pvx : nullptr);
        o.mm[c.mmPV(ai)].W = commit_u8(w.mm[c.mmPV(ai)].wcodes, R, (zk&&!p3zkc::G.nolink) ? &pvw : nullptr);
        o.mm[c.mmPV(ai)].XS = o.qn[c.qnPB(ai)].SCALES;
        o.mm[c.mmPV(ai)].WS = o.qn[c.qnVT(ai)].SCALES;
        o.Ymm[c.mmPV(ai)] = commit_col_nc(w.mm[c.mmPV(ai)].dob[p3hwl::O_YB], R);
    }
    {   // attn concat: AT.X mask = concat of the PV output head slices
        std::vector<gl_t> atm = concat_mask();
        o.qn[c.qnAT()].X = commit_col_nc(w.qn[c.qnAT()].xpat, R, (zk&&!p3zkc::G.nolink) ? &atm : nullptr);
    }
    o.qn[c.qnAT()].CODES = commit_col_nc(w.qn[c.qnAT()].cpat, R);
    o.qn[c.qnAT()].SCALES = commit_col_nc(w.qn[c.qnAT()].spat, R);
    {
        std::vector<gl_t> xm = restr_mask(o.qn[c.qnAT()].CODES, ld, lkH1, lT);
        o.mm[c.mmWO()].X = commit_u8(w.mm[c.mmWO()].xcodes, R, (zk&&!p3zkc::G.nolink) ? &xm : nullptr);
    }
    o.mm[c.mmWO()].W = commit_u8(w.mm[c.mmWO()].wcodes, R);
    o.mm[c.mmWO()].XS = o.qn[c.qnAT()].SCALES;
    o.mm[c.mmWO()].WS = commit_col_nc(w.mm[c.mmWO()].wsb, R);
    o.Ymm[c.mmWO()] = commit_col_nc(w.mm[c.mmWO()].dob[p3hwl::O_YB], R);
    o.res[0].X1 = o.rms[0].X;
    o.res[0].X2 = o.Ymm[c.mmWO()];
    o.res[0].OUT = commit_col_nc(w.res[0].opat, R);
    o.rms[1].X = o.res[0].OUT;
    o.rms[1].G = commit_col_nc(w.rms[1].gpat, R);
    o.rms[1].Y = commit_col_nc(w.rms[1].ypat, R);
    o.qn[c.qnH2()].X = o.rms[1].Y;
    o.qn[c.qnH2()].CODES = commit_col_nc(w.qn[c.qnH2()].cpat, R);
    o.qn[c.qnH2()].SCALES = commit_col_nc(w.qn[c.qnH2()].spat, R);
    {
        std::vector<gl_t> xm = restr_mask(o.qn[c.qnH2()].CODES, ld, lkH1, lT);
        Col X2 = commit_u8(w.mm[c.mmWG()].xcodes, R, (zk&&!p3zkc::G.nolink) ? &xm : nullptr);
        for (int i : {c.mmWG(), c.mmWU()}) {
            o.mm[i].X = X2;
            o.mm[i].W = commit_u8(w.mm[i].wcodes, R);
            o.mm[i].XS = o.qn[c.qnH2()].SCALES;
            o.mm[i].WS = commit_col_nc(w.mm[i].wsb, R);
            o.Ymm[i] = commit_col_nc(w.mm[i].dob[p3hwl::O_YB], R);
        }
    }
    o.sw.GATE = o.Ymm[c.mmWG()];
    o.sw.UP = o.Ymm[c.mmWU()];
    o.sw.M = commit_col_nc(w.sw.mpat, R);
    o.qn[c.qnSW()].X = o.sw.M;
    o.qn[c.qnSW()].CODES = commit_col_nc(w.qn[c.qnSW()].cpat, R);
    o.qn[c.qnSW()].SCALES = commit_col_nc(w.qn[c.qnSW()].spat, R);
    {
        std::vector<gl_t> xm = restr_mask(o.qn[c.qnSW()].CODES, ldff, lkSW, lT);
        o.mm[c.mmWD()].X = commit_u8(w.mm[c.mmWD()].xcodes, R, (zk&&!p3zkc::G.nolink) ? &xm : nullptr);
    }
    o.mm[c.mmWD()].W = commit_u8(w.mm[c.mmWD()].wcodes, R);
    o.mm[c.mmWD()].XS = o.qn[c.qnSW()].SCALES;
    o.mm[c.mmWD()].WS = commit_col_nc(w.mm[c.mmWD()].wsb, R);
    o.Ymm[c.mmWD()] = commit_col_nc(w.mm[c.mmWD()].dob[p3hwl::O_YB], R);
    o.res[1].X1 = o.res[0].OUT;
    o.res[1].X2 = o.Ymm[c.mmWD()];
    o.res[1].OUT = commit_col_nc(w.res[1].opat, R);
    return o;
}

// the caller's pinned weight-side commitments (the public statement): computed
// independently of the prover from the weights file + canonical padding
struct WeightRoots { Hash Wc[NW], Ws[NW], G1, G2; };
static inline WeightRoots weight_roots(const Weights& W, uint32_t R, uint32_t batch = 1) {
    WeightRoots wr;
    const Config& c = W.cfg;
    uint32_t T = c.seq * batch;
    for (int i = 0; i < NW; i++) {
        p3hwl::Dims dd = p3hwl::make_dims(T, W.w[i].K, W.w[i].N);
        std::vector<gl_t> wc((size_t)dd.Npad * dd.Kpad, 0), ws(dd.Npad, 0);
        for (uint32_t n = 0; n < W.w[i].N; n++)
            for (uint32_t k = 0; k < W.w[i].K; k++)
                wc[(size_t)n * dd.Kpad + k] = W.w[i].codes[(size_t)n * W.w[i].K + k];
        for (uint32_t n = 0; n < W.w[i].N; n++) ws[n] = W.w[i].scales[n];
        wr.Wc[i] = commit_col_nc(wc, R).root;
        wr.Ws[i] = commit_col_nc(ws, R).root;
    }
    std::vector<gl_t> g1(W.g1.begin(), W.g1.end()), g2(W.g2.begin(), W.g2.end());
    wr.G1 = commit_col_nc(g1, R).root;
    wr.G2 = commit_col_nc(g2, R).root;
    return wr;
}

// ---------------- proof object ----------------
struct TfProof {
    uint32_t seq = 0, d = 0, nh = 0, dh = 0, dff = 0, batch = 1;
    // sub-proofs (fixed protocol order)
    p3rms::RmsProof rms[2];
    std::vector<p3qnt::QntProof> qn;
    std::vector<p3hwl::LayerProof> mm;
    std::vector<p3rope::RopeProof> rp;
    std::vector<p3smx::SmxProof> sm;
    p3bfa::BfaProof res[2];
    p3swg::SwgProof sw;
    // chained intermediate roots (pinned by BOTH neighbours' sub-verifies)
    Hash rX0;                                     // committed layer input
    Hash rRmsY[2];
    std::vector<Hash> rCodes, rScales;
    std::vector<Hash> rMX, rMW;                   // padded matmul operand columns
    std::vector<Hash> rRopeQ, rRopeO;
    std::vector<Hash> rSmP, rVtX;
    Hash rAtX, rRes1, rSwM, rOut;
    // public per-matmul output claims (the Hawkeye atom's native binding)
    std::vector<std::vector<uint16_t>> mmY;
    // layer-level MERGED lookup groups (all gadget instances, R3b)
    std::vector<p3lu::GroupProof> lug;
    // seam claim evaluations, fixed order
    std::vector<gl_t> seam;
    // ONE batched opening per size class for the WHOLE layer (R3)
    std::vector<p3bo::BatchProof> batches;
};

// per-stage prover timings (ms)
struct TfProf {
    double commit = 0, rms = 0, qnt = 0, mm = 0, rope = 0, smx = 0, bfa = 0,
           swg = 0, lug = 0, seam = 0, batch = 0, total = 0;
};

// point helpers -----------------------------------------------------------
// insert nz constant-zero bits at position pos
static inline std::vector<gl_t> ins_zeros(const std::vector<gl_t>& z, uint32_t pos, uint32_t nz) {
    std::vector<gl_t> p(z.begin(), z.begin() + pos);
    for (uint32_t i = 0; i < nz; i++) p.push_back(0);
    p.insert(p.end(), z.begin() + pos, z.end());
    return p;
}
// padding subcube [2^t, 2^{t+1}) of a lk-bit index: bit t = 1, bits (t, lk) = 0
static inline std::vector<gl_t> pad_cube_pt(const std::vector<gl_t>& z, uint32_t t, uint32_t lk) {
    std::vector<gl_t> p(z.begin(), z.begin() + t);
    p.push_back(1);
    for (uint32_t i = t + 1; i < lk; i++) p.push_back(0);
    p.insert(p.end(), z.begin() + t, z.end());
    return p;
}
// head+batch slice point: consumer z = [z_inner (li) | z_seq]; producer point
// = [z_inner | bits(h) (lh) | z_seq | bits(b) (lB)] -- fixes the head bits of
// the model index and the batch bits of the token index.  At lB=0, lh=1 this
// is exactly the original ins_pt(z, li, h).
static inline std::vector<gl_t> slice_pt(const std::vector<gl_t>& z, uint32_t li,
                                         uint32_t lh, uint32_t h,
                                         uint32_t lB, uint32_t b) {
    std::vector<gl_t> p(z.begin(), z.begin() + li);
    for (uint32_t i = 0; i < lh; i++) p.push_back((h >> i) & 1);
    p.insert(p.end(), z.begin() + li, z.end());
    for (uint32_t i = 0; i < lB; i++) p.push_back((b >> i) & 1);
    return p;
}

// ==================== prover ====================
static inline TfProof prove(fs::Transcript& tr, const TfWit& w, const TfOps& o,
                            const TfTables& T, const Art& a, uint32_t R, uint32_t Q,
                            bool strict = true, TfProf* prof = nullptr) {
    const Config& c = w.cfg;
    const bool zk = p3zkc::G.on;
    TfProof pf;
    pf.seq = c.seq; pf.d = c.d; pf.nh = c.nh; pf.dh = c.dh; pf.dff = c.dff; pf.batch = c.batch;
    const uint32_t A = c.A(), nh = c.nh;
    pf.qn.resize(c.nqn()); pf.mm.resize(c.nmm()); pf.rp.resize(2 * A); pf.sm.resize(A);
    pf.rCodes.resize(c.nqn()); pf.rScales.resize(c.nqn());
    pf.rMX.resize(c.nmm()); pf.rMW.resize(c.nmm());
    pf.rRopeQ.resize(2 * A); pf.rRopeO.resize(2 * A);
    pf.rSmP.resize(A); pf.rVtX.resize(A);
    pf.mmY.resize(c.nmm());
    TfProf pl; TfProf& P = prof ? *prof : pl;
    double tall = now_ms(), tp;

    uint32_t hdr[6] = {c.seq, c.d, c.nh, c.dh, c.dff, c.batch};
    tr.absorb("tf-dims", hdr, sizeof hdr);

    p3lu::XCtx xc;

    // roots into the proof (the chain the verifier re-pins)
    pf.rX0 = o.rms[0].X.root;
    for (int i = 0; i < 2; i++) pf.rRmsY[i] = o.rms[i].Y.root;
    for (int i = 0; i < c.nqn(); i++) { pf.rCodes[i] = o.qn[i].CODES.root; pf.rScales[i] = o.qn[i].SCALES.root; }
    for (int i = 0; i < c.nmm(); i++) { pf.rMX[i] = o.mm[i].X.root; pf.rMW[i] = o.mm[i].W.root; }
    for (uint32_t i = 0; i < 2 * A; i++) { pf.rRopeQ[i] = o.rp[i].Q.root; pf.rRopeO[i] = o.rp[i].OUT.root; }
    for (uint32_t ai = 0; ai < A; ai++) { pf.rSmP[ai] = o.sm[ai].P.root; pf.rVtX[ai] = o.qn[c.qnVT(ai)].X.root; }
    pf.rAtX = o.qn[c.qnAT()].X.root;
    pf.rRes1 = o.res[0].OUT.root;
    pf.rSwM = o.sw.M.root;
    pf.rOut = o.res[1].OUT.root;
    // zk: the per-matmul PUBLIC output vectors ARE the cleartext intermediate
    // activations -- drop them entirely (outputs are bound through the seams and
    // the final public-output binding instead)
    for (int i = 0; i < c.nmm(); i++) pf.mmY[i] = zk ? std::vector<uint16_t>() : w.mmYpub[i];

    // -- sub-proofs, fixed order, one transcript, one ledger --
    tp = now_ms();
    pf.rms[0] = p3rms::prove(tr, w.rms[0], T.rms, a, o.rms[0], R, Q, strict, &xc);
    P.rms += now_ms() - tp; tp = now_ms();
    pf.qn[QN_H1] = p3qnt::prove(tr, w.qn[QN_H1], T.qnt, o.qn[QN_H1], R, Q, strict, &xc);
    P.qnt += now_ms() - tp; tp = now_ms();
    for (int i : {MM_WQ, MM_WK, MM_WV})
        pf.mm[i] = p3hwl::prove(tr, w.mm[i], T.hw, o.mm[i], R, Q, true, strict,
                                nullptr, &w.mmYpub[i], &xc, zk ? &o.Ymm[i] : nullptr);
    P.mm += now_ms() - tp;
    for (uint32_t ai = 0; ai < A; ai++) {
        tp = now_ms();
        pf.rp[2 * ai] = p3rope::prove(tr, w.rp[2 * ai], T.rope, o.rp[2 * ai], R, Q, strict, &xc);
        pf.rp[2 * ai + 1] = p3rope::prove(tr, w.rp[2 * ai + 1], T.rope, o.rp[2 * ai + 1], R, Q, strict, &xc);
        P.rope += now_ms() - tp; tp = now_ms();
        pf.qn[c.qnRQ(ai)] = p3qnt::prove(tr, w.qn[c.qnRQ(ai)], T.qnt, o.qn[c.qnRQ(ai)], R, Q, strict, &xc);
        pf.qn[c.qnRK(ai)] = p3qnt::prove(tr, w.qn[c.qnRK(ai)], T.qnt, o.qn[c.qnRK(ai)], R, Q, strict, &xc);
        P.qnt += now_ms() - tp; tp = now_ms();
        pf.mm[c.mmQK(ai)] = p3hwl::prove(tr, w.mm[c.mmQK(ai)], T.hw, o.mm[c.mmQK(ai)], R, Q,
                                       true, strict, nullptr, &w.mmYpub[c.mmQK(ai)], &xc,
                                       zk ? &o.Ymm[c.mmQK(ai)] : nullptr);
        P.mm += now_ms() - tp; tp = now_ms();
        pf.sm[ai] = p3smx::prove(tr, w.sm[ai], T.smx, a, o.sm[ai], R, Q, strict, &xc);
        P.smx += now_ms() - tp; tp = now_ms();
        pf.qn[c.qnPB(ai)] = p3qnt::prove(tr, w.qn[c.qnPB(ai)], T.qnt, o.qn[c.qnPB(ai)], R, Q, strict, &xc);
        pf.qn[c.qnVT(ai)] = p3qnt::prove(tr, w.qn[c.qnVT(ai)], T.qnt, o.qn[c.qnVT(ai)], R, Q, strict, &xc);
        P.qnt += now_ms() - tp; tp = now_ms();
        pf.mm[c.mmPV(ai)] = p3hwl::prove(tr, w.mm[c.mmPV(ai)], T.hw, o.mm[c.mmPV(ai)], R, Q,
                                       true, strict, nullptr, &w.mmYpub[c.mmPV(ai)], &xc,
                                       zk ? &o.Ymm[c.mmPV(ai)] : nullptr);
        P.mm += now_ms() - tp;
    }
    tp = now_ms();
    pf.qn[c.qnAT()] = p3qnt::prove(tr, w.qn[c.qnAT()], T.qnt, o.qn[c.qnAT()], R, Q, strict, &xc);
    P.qnt += now_ms() - tp; tp = now_ms();
    pf.mm[c.mmWO()] = p3hwl::prove(tr, w.mm[c.mmWO()], T.hw, o.mm[c.mmWO()], R, Q, true, strict,
                                nullptr, &w.mmYpub[c.mmWO()], &xc, zk ? &o.Ymm[c.mmWO()] : nullptr);
    P.mm += now_ms() - tp; tp = now_ms();
    pf.res[0] = p3bfa::prove(tr, w.res[0], T.bfa, o.res[0], R, Q, strict, &xc);
    P.bfa += now_ms() - tp; tp = now_ms();
    pf.rms[1] = p3rms::prove(tr, w.rms[1], T.rms, a, o.rms[1], R, Q, strict, &xc);
    P.rms += now_ms() - tp; tp = now_ms();
    pf.qn[c.qnH2()] = p3qnt::prove(tr, w.qn[c.qnH2()], T.qnt, o.qn[c.qnH2()], R, Q, strict, &xc);
    P.qnt += now_ms() - tp; tp = now_ms();
    for (int i : {c.mmWG(), c.mmWU()})
        pf.mm[i] = p3hwl::prove(tr, w.mm[i], T.hw, o.mm[i], R, Q, true, strict,
                                nullptr, &w.mmYpub[i], &xc, zk ? &o.Ymm[i] : nullptr);
    P.mm += now_ms() - tp; tp = now_ms();
    pf.sw = p3swg::prove(tr, w.sw, T.swg, o.sw, R, Q, strict, &xc);
    P.swg += now_ms() - tp; tp = now_ms();
    pf.qn[c.qnSW()] = p3qnt::prove(tr, w.qn[c.qnSW()], T.qnt, o.qn[c.qnSW()], R, Q, strict, &xc);
    P.qnt += now_ms() - tp; tp = now_ms();
    pf.mm[c.mmWD()] = p3hwl::prove(tr, w.mm[c.mmWD()], T.hw, o.mm[c.mmWD()], R, Q, true, strict,
                                nullptr, &w.mmYpub[c.mmWD()], &xc, zk ? &o.Ymm[c.mmWD()] : nullptr);
    P.mm += now_ms() - tp; tp = now_ms();
    pf.res[1] = p3bfa::prove(tr, w.res[1], T.bfa, o.res[1], R, Q, strict, &xc);
    P.bfa += now_ms() - tp;

    // -- ONE merged-lookup flush for ALL gadget instances (R3b) --
    tp = now_ms();
    pf.lug = p3lu::lu_flush(tr, xc, R, Q, strict);
    P.lug += now_ms() - tp;

    // -- seam claims (all roots are in the transcript; fresh challenges) --
    tp = now_ms();
    // clp: public/real-slice claim (input/output bindings, canonical-zero pads --
    // all reveal only PUBLIC values).  clx: hiding seam-pair claim at a SHARED ex
    // coordinate zex (touches real slice + mask slice 1; the two sides' slice-1
    // masks are seam-linked in commit_all, so the augmented evals agree).
    auto clp = [&](const Col& col, const std::vector<gl_t>& z) {
        gl_t y = claimc(tr, xc.lg, col, p3zkc::zpt(z));
        pf.seam.push_back(y);
        return y;
    };
    auto clx = [&](const Col& col, const std::vector<gl_t>& z, gl_t zex) {
        gl_t y = claimc(tr, xc.lg, col, p3zkc::xpt(z, zex));
        pf.seam.push_back(y);
        return y;
    };
    auto seam_pair = [&](const Col& Ac, const std::vector<gl_t>& zA,
                         const Col& Bc, const std::vector<gl_t>& zB, const char* what) {
        gl_t zex = zk ? p3lu::chal(tr) : 0;
        gl_t y1 = clx(Ac, zA, zex), y2 = clx(Bc, zB, zex);
        if (strict && y1 != y2)
            throw std::runtime_error(std::string("tf: seam mismatch: ") + what);
    };
    // quant CODES -> padded matmul operand: restriction + canonical-zero padding
    auto seam_codes = [&](const Col& C, const Col& X, uint32_t ldr, uint32_t lk,
                          uint32_t lb, const char* what) {
        std::vector<gl_t> z = chal_vec(tr, ldr + lb);
        seam_pair(C, z, X, ins_zeros(z, ldr, lk - ldr), what);
        for (uint32_t t = ldr; t < lk; t++) {
            std::vector<gl_t> zt = chal_vec(tr, t + lb);
            gl_t y = clp(X, pad_cube_pt(zt, t, lk));
            if (strict && y != 0)
                throw std::runtime_error(std::string("tf: padding not zero: ") + what);
        }
    };
    const uint32_t lseq = c.lseq(), ld = c.ld(), ldh = c.ldh(), ldff = c.ldff(),
                   lT = c.lT(), lnh = c.lnh(), lB = c.lbb();
    const uint32_t lkH1 = ilog2(w.mm[MM_WQ].d.Kpad), lkQK = ilog2(w.mm[c.mmQK(0)].d.Kpad),
                   lkPV = ilog2(w.mm[c.mmPV(0)].d.Kpad), lkSW = ilog2(w.mm[c.mmWD()].d.Kpad);
    // input binding
    {
        std::vector<gl_t> z = chal_vec(tr, ld + lT);
        clp(o.rms[0].X, z);
    }
    seam_codes(o.qn[QN_H1].CODES, o.mm[MM_WQ].X, ld, lkH1, lT, "h1 codes");
    for (uint32_t b = 0; b < c.batch; b++)
    for (uint32_t h = 0; h < nh; h++) {
        const uint32_t ai = b * nh + h;
        {   // rope q/k operands = (b,h) slices of the Wq/Wk outputs
            std::vector<gl_t> z = chal_vec(tr, ldh + lseq);
            seam_pair(o.rp[2 * ai].Q, z, o.Ymm[MM_WQ],
                      slice_pt(z, ldh, lnh, h, lB, b), "rope-q slice");
            std::vector<gl_t> z2 = chal_vec(tr, ldh + lseq);
            seam_pair(o.rp[2 * ai + 1].Q, z2, o.Ymm[MM_WK],
                      slice_pt(z2, ldh, lnh, h, lB, b), "rope-k slice");
        }
        seam_codes(o.qn[c.qnRQ(ai)].CODES, o.mm[c.mmQK(ai)].X, ldh, lkQK, lseq, "qk x codes");
        seam_codes(o.qn[c.qnRK(ai)].CODES, o.mm[c.mmQK(ai)].W, ldh, lkQK, lseq, "qk w codes");
        {   // V^T: swap the variable groups of the Wv output, fix head+batch bits
            std::vector<gl_t> z = chal_vec(tr, lseq + ldh);
            std::vector<gl_t> zt(z.begin() + lseq, z.end());        // dh bits -> n low
            for (uint32_t i = 0; i < lnh; i++) zt.push_back((h >> i) & 1);
            zt.insert(zt.end(), z.begin(), z.begin() + lseq);       // seq bits -> tok low
            for (uint32_t i = 0; i < lB; i++) zt.push_back((b >> i) & 1);
            seam_pair(o.qn[c.qnVT(ai)].X, z, o.Ymm[MM_WV], zt, "v^T transpose");
        }
        seam_codes(o.qn[c.qnPB(ai)].CODES, o.mm[c.mmPV(ai)].X, lseq, lkPV, lseq, "pv x codes");
        seam_codes(o.qn[c.qnVT(ai)].CODES, o.mm[c.mmPV(ai)].W, lseq, lkPV, ldh, "pv w codes");
        {   // concat: attn (b,h) block == attnout_a
            std::vector<gl_t> z = chal_vec(tr, ldh + lseq);
            seam_pair(o.qn[c.qnAT()].X, slice_pt(z, ldh, lnh, h, lB, b),
                      o.Ymm[c.mmPV(ai)], z, "concat half");
        }
    }
    seam_codes(o.qn[c.qnAT()].CODES, o.mm[c.mmWO()].X, ld, lkH1, lT, "attn codes");
    seam_codes(o.qn[c.qnH2()].CODES, o.mm[c.mmWG()].X, ld, lkH1, lT, "h2 codes");
    seam_codes(o.qn[c.qnSW()].CODES, o.mm[c.mmWD()].X, ldff, lkSW, lT, "swiglu codes");
    // output binding
    {
        std::vector<gl_t> z = chal_vec(tr, ld + lT);
        clp(o.res[1].OUT, z);
    }
    P.seam += now_ms() - tp;

    // -- ONE batched opening pass per size class for the whole layer --
    tp = now_ms();
    for (size_t i = 0; i < xc.lg.cls.size(); i++)
        pf.batches.push_back(p3bo::prove_class(tr, xc.lg.cls[i], R, Q,
                                               "tf-bo" + std::to_string(i)));
    P.batch += now_ms() - tp;
    P.total += now_ms() - tall;
    return pf;
}

// ==================== verifier ====================
// PUBLIC statement, all pinned by the CALLER: dims, input (patterns + root),
// g1/g2 roots, the 7 weight code+scale roots, the pinned rope cos/sin tables,
// the canonical table artifact, Q/R -- and the claimed OUTPUT patterns, which
// the accepted proof binds bitwise to the committed final residual column.
static inline bool verify(fs::Transcript& tr, const TfProof& pf, const TfTables& T,
                          const Art& a, const Config& c,
                          const std::vector<uint16_t>& x0pub, const Hash& rX0,
                          const WeightRoots& wr,
                          const std::vector<uint16_t>& cosp, const std::vector<uint16_t>& sinp,
                          const std::vector<uint16_t>& outpub,
                          uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (!c.pow2()) return fail("config must be pow2");
    if (pf.seq != c.seq || pf.d != c.d || pf.nh != c.nh || pf.dh != c.dh
        || pf.dff != c.dff || pf.batch != c.batch)
        return fail("dims mismatch");
    const uint32_t A = c.A(), nh = c.nh;
    if ((int)pf.qn.size() != c.nqn() || (int)pf.mm.size() != c.nmm()
        || pf.rp.size() != 2 * A || pf.sm.size() != A
        || (int)pf.rCodes.size() != c.nqn() || (int)pf.rScales.size() != c.nqn()
        || (int)pf.rMX.size() != c.nmm() || (int)pf.rMW.size() != c.nmm()
        || pf.rRopeQ.size() != 2 * A || pf.rRopeO.size() != 2 * A
        || pf.rSmP.size() != A || pf.rVtX.size() != A
        || (int)pf.mmY.size() != c.nmm())
        return fail("proof shape");
    if (x0pub.size() != (size_t)c.T() * c.d || outpub.size() != (size_t)c.T() * c.d)
        return fail("public i/o size");
    if (!(pf.rX0 == rX0)) return fail("input root mismatch");
    const bool zk = p3zkc::G.on;
    if (!zk) for (int i = 0; i < c.nmm(); i++) {
        uint32_t Bm = c.T(), N = c.d;
        if (i >= 3 && i < 3 + (int)A) { Bm = c.seq; N = c.seq; }          // QK
        else if (i >= 3 + (int)A && i < 3 + 2 * (int)A) { Bm = c.seq; N = c.dh; } // PV
        else if (i == c.mmWG() || i == c.mmWU()) N = c.dff;
        if (pf.mmY[i].size() != (size_t)Bm * N) return fail("mmY size");
    }
    const uint32_t lseq = c.lseq(), ld = c.ld(), ldh = c.ldh(), ldff = c.ldff(),
                   lT = c.lT(), lnh = c.lnh(), lB = c.lbb();

    uint32_t hdr[6] = {c.seq, c.d, c.nh, c.dh, c.dff, c.batch};
    tr.absorb("tf-dims", hdr, sizeof hdr);

    p3lu::VCtx vctx;
    p3bo::VLedger& vlg = vctx.vlg;
    const std::vector<uint8_t> causal = [&] {
        std::vector<uint8_t> m((size_t)c.seq * c.seq, 0);
        for (uint32_t i = 0; i < c.seq; i++)
            for (uint32_t j = 0; j <= i; j++) m[(size_t)i * c.seq + j] = 1;
        return m;
    }();

    // matmul output roots: the committed O_YB columns ARE the chained operands
    auto rY = [&](int i) -> const Hash& { return pf.mm[i].rdo[p3hwl::O_YB]; };

    // -- sub-verifies, same order, same pinned dims --
    if (!p3rms::verify(tr, T.rms, a, pf.rms[0], rX0, wr.G1, pf.rRmsY[0],
                       c.T(), ld, Q_pub, R_pub, why, &vctx)) return false;
    if (!p3qnt::verify(tr, T.qnt, pf.qn[QN_H1], pf.rRmsY[0], pf.rCodes[QN_H1],
                       pf.rScales[QN_H1], c.T(), ld, Q_pub, R_pub, why, &vctx)) return false;
    {
        static const int WIDX[3] = {W_Q, W_K, W_V};
        int j = 0;
        for (int i : {MM_WQ, MM_WK, MM_WV}) {
            if (!(pf.rMX[i] == pf.rMX[MM_WQ])) return fail("h1 X root not shared");
            if (!(pf.rMW[i] == wr.Wc[WIDX[j]])) return fail("weight root mismatch");
            if (!p3hwl::verify(tr, T.hw, pf.mm[i], pf.rMX[i], wr.Wc[WIDX[j]],
                               pf.rScales[QN_H1], wr.Ws[WIDX[j]], pf.mmY[i],
                               c.T(), c.d, c.d, Q_pub, R_pub, why, &vctx)) return false;
            j++;
        }
    }
    for (uint32_t ai = 0; ai < A; ai++) {
        if (!p3rope::verify(tr, T.rope, pf.rp[2 * ai], cosp, sinp, pf.rRopeQ[2 * ai],
                            pf.rRopeO[2 * ai], c.seq, c.dh, Q_pub, R_pub, why, &vctx)) return false;
        if (!p3rope::verify(tr, T.rope, pf.rp[2 * ai + 1], cosp, sinp, pf.rRopeQ[2 * ai + 1],
                            pf.rRopeO[2 * ai + 1], c.seq, c.dh, Q_pub, R_pub, why, &vctx)) return false;
        if (!p3qnt::verify(tr, T.qnt, pf.qn[c.qnRQ(ai)], pf.rRopeO[2 * ai], pf.rCodes[c.qnRQ(ai)],
                           pf.rScales[c.qnRQ(ai)], c.seq, ldh, Q_pub, R_pub, why, &vctx)) return false;
        if (!p3qnt::verify(tr, T.qnt, pf.qn[c.qnRK(ai)], pf.rRopeO[2 * ai + 1], pf.rCodes[c.qnRK(ai)],
                           pf.rScales[c.qnRK(ai)], c.seq, ldh, Q_pub, R_pub, why, &vctx)) return false;
        if (!p3hwl::verify(tr, T.hw, pf.mm[c.mmQK(ai)], pf.rMX[c.mmQK(ai)], pf.rMW[c.mmQK(ai)],
                           pf.rScales[c.qnRQ(ai)], pf.rScales[c.qnRK(ai)], pf.mmY[c.mmQK(ai)],
                           c.seq, c.dh, c.seq, Q_pub, R_pub, why, &vctx)) return false;
        if (!p3smx::verify(tr, T.smx, a, pf.sm[ai], causal, rY(c.mmQK(ai)), pf.rSmP[ai],
                           c.seq, c.seq, Q_pub, R_pub, why, &vctx)) return false;
        if (!p3qnt::verify(tr, T.qnt, pf.qn[c.qnPB(ai)], pf.rSmP[ai], pf.rCodes[c.qnPB(ai)],
                           pf.rScales[c.qnPB(ai)], c.seq, lseq, Q_pub, R_pub, why, &vctx)) return false;
        if (!p3qnt::verify(tr, T.qnt, pf.qn[c.qnVT(ai)], pf.rVtX[ai], pf.rCodes[c.qnVT(ai)],
                           pf.rScales[c.qnVT(ai)], c.dh, lseq, Q_pub, R_pub, why, &vctx)) return false;
        if (!p3hwl::verify(tr, T.hw, pf.mm[c.mmPV(ai)], pf.rMX[c.mmPV(ai)], pf.rMW[c.mmPV(ai)],
                           pf.rScales[c.qnPB(ai)], pf.rScales[c.qnVT(ai)], pf.mmY[c.mmPV(ai)],
                           c.seq, c.seq, c.dh, Q_pub, R_pub, why, &vctx)) return false;
    }
    if (!p3qnt::verify(tr, T.qnt, pf.qn[c.qnAT()], pf.rAtX, pf.rCodes[c.qnAT()],
                       pf.rScales[c.qnAT()], c.T(), ld, Q_pub, R_pub, why, &vctx)) return false;
    if (!(pf.rMW[c.mmWO()] == wr.Wc[W_O])) return fail("weight root mismatch");
    if (!p3hwl::verify(tr, T.hw, pf.mm[c.mmWO()], pf.rMX[c.mmWO()], wr.Wc[W_O],
                       pf.rScales[c.qnAT()], wr.Ws[W_O], pf.mmY[c.mmWO()],
                       c.T(), c.d, c.d, Q_pub, R_pub, why, &vctx)) return false;
    if (!p3bfa::verify(tr, T.bfa, pf.res[0], rX0, rY(c.mmWO()), pf.rRes1,
                       c.T() * c.d, Q_pub, R_pub, why, &vctx)) return false;
    if (!p3rms::verify(tr, T.rms, a, pf.rms[1], pf.rRes1, wr.G2, pf.rRmsY[1],
                       c.T(), ld, Q_pub, R_pub, why, &vctx)) return false;
    if (!p3qnt::verify(tr, T.qnt, pf.qn[c.qnH2()], pf.rRmsY[1], pf.rCodes[c.qnH2()],
                       pf.rScales[c.qnH2()], c.T(), ld, Q_pub, R_pub, why, &vctx)) return false;
    {
        static const int WIDX[2] = {W_G, W_U};
        int j = 0;
        for (int i : {c.mmWG(), c.mmWU()}) {
            if (!(pf.rMX[i] == pf.rMX[c.mmWG()])) return fail("h2 X root not shared");
            if (!(pf.rMW[i] == wr.Wc[WIDX[j]])) return fail("weight root mismatch");
            if (!p3hwl::verify(tr, T.hw, pf.mm[i], pf.rMX[i], wr.Wc[WIDX[j]],
                               pf.rScales[c.qnH2()], wr.Ws[WIDX[j]], pf.mmY[i],
                               c.T(), c.d, c.dff, Q_pub, R_pub, why, &vctx)) return false;
            j++;
        }
    }
    if (!p3swg::verify(tr, T.swg, pf.sw, rY(c.mmWG()), rY(c.mmWU()), pf.rSwM,
                       c.T() * c.dff, Q_pub, R_pub, why, &vctx)) return false;
    if (!p3qnt::verify(tr, T.qnt, pf.qn[c.qnSW()], pf.rSwM, pf.rCodes[c.qnSW()],
                       pf.rScales[c.qnSW()], c.T(), ldff, Q_pub, R_pub, why, &vctx)) return false;
    if (!(pf.rMW[c.mmWD()] == wr.Wc[W_D])) return fail("weight root mismatch");
    if (!p3hwl::verify(tr, T.hw, pf.mm[c.mmWD()], pf.rMX[c.mmWD()], wr.Wc[W_D],
                       pf.rScales[c.qnSW()], wr.Ws[W_D], pf.mmY[c.mmWD()],
                       c.T(), c.dff, c.d, Q_pub, R_pub, why, &vctx)) return false;
    if (!p3bfa::verify(tr, T.bfa, pf.res[1], pf.rRes1, rY(c.mmWD()), pf.rOut,
                       c.T() * c.d, Q_pub, R_pub, why, &vctx)) return false;

    // -- merged-lookup flush (mirror of the prover's group pass) --
    if (!p3lu::lu_verify_flush(tr, vctx, pf.lug, Q_pub, R_pub, why)) return false;

    // -- seam checks (mirror of the prover's claim sequence) --
    size_t si = 0;
    bool seam_short = false;
    auto vclp = [&](const Hash& root, const std::vector<gl_t>& z) -> gl_t {
        if (si >= pf.seam.size()) { seam_short = true; return 0; }
        gl_t y = pf.seam[si++];
        return claimv(tr, vlg, root, p3zkc::zpt(z), y);
    };
    auto vclx = [&](const Hash& root, const std::vector<gl_t>& z, gl_t zex) -> gl_t {
        if (si >= pf.seam.size()) { seam_short = true; return 0; }
        gl_t y = pf.seam[si++];
        return claimv(tr, vlg, root, p3zkc::xpt(z, zex), y);
    };
    auto seam_pair = [&](const Hash& rA, const std::vector<gl_t>& zA,
                         const Hash& rB, const std::vector<gl_t>& zB) {
        gl_t zex = zk ? p3lu::chal(tr) : 0;
        gl_t y1 = vclx(rA, zA, zex), y2 = vclx(rB, zB, zex);
        return !seam_short && y1 == y2;
    };
    auto seam_codes = [&](const Hash& rC, const Hash& rX, uint32_t ldr, uint32_t lk,
                          uint32_t lb) {
        std::vector<gl_t> z = chal_vec(tr, ldr + lb);
        if (!seam_pair(rC, z, rX, ins_zeros(z, ldr, lk - ldr))) return false;
        for (uint32_t t = ldr; t < lk; t++) {
            std::vector<gl_t> zt = chal_vec(tr, t + lb);
            if (vclp(rX, pad_cube_pt(zt, t, lk)) != 0 || seam_short) return false;
        }
        return true;
    };
    const uint32_t lkH1 = ilog2(p3hwl::make_dims(c.T(), c.d, c.d).Kpad);
    const uint32_t lkQK = ilog2(p3hwl::make_dims(c.seq, c.dh, c.seq).Kpad);
    const uint32_t lkPV = ilog2(p3hwl::make_dims(c.seq, c.seq, c.dh).Kpad);
    const uint32_t lkSW = ilog2(p3hwl::make_dims(c.T(), c.dff, c.d).Kpad);
    {   // input binding: committed layer input == the PUBLIC x
        std::vector<gl_t> z = chal_vec(tr, ld + lT);
        std::vector<gl_t> xp(x0pub.begin(), x0pub.end());
        if (vclp(rX0, z) != mle_eval(xp, z) || seam_short)
            return fail("public input binding");
    }
    if (!seam_codes(pf.rCodes[QN_H1], pf.rMX[MM_WQ], ld, lkH1, lT))
        return fail("seam: h1 codes");
    for (uint32_t b = 0; b < c.batch; b++)
    for (uint32_t h = 0; h < nh; h++) {
        const uint32_t ai = b * nh + h;
        {
            std::vector<gl_t> z = chal_vec(tr, ldh + lseq);
            if (!seam_pair(pf.rRopeQ[2 * ai], z, rY(MM_WQ), slice_pt(z, ldh, lnh, h, lB, b)))
                return fail("seam: rope-q slice");
            std::vector<gl_t> z2 = chal_vec(tr, ldh + lseq);
            if (!seam_pair(pf.rRopeQ[2 * ai + 1], z2, rY(MM_WK), slice_pt(z2, ldh, lnh, h, lB, b)))
                return fail("seam: rope-k slice");
        }
        if (!seam_codes(pf.rCodes[c.qnRQ(ai)], pf.rMX[c.mmQK(ai)], ldh, lkQK, lseq))
            return fail("seam: qk x codes");
        if (!seam_codes(pf.rCodes[c.qnRK(ai)], pf.rMW[c.mmQK(ai)], ldh, lkQK, lseq))
            return fail("seam: qk w codes");
        {
            std::vector<gl_t> z = chal_vec(tr, lseq + ldh);
            std::vector<gl_t> zt(z.begin() + lseq, z.end());
            for (uint32_t i = 0; i < lnh; i++) zt.push_back((h >> i) & 1);
            zt.insert(zt.end(), z.begin(), z.begin() + lseq);
            for (uint32_t i = 0; i < lB; i++) zt.push_back((b >> i) & 1);
            if (!seam_pair(pf.rVtX[ai], z, rY(MM_WV), zt))
                return fail("seam: v^T transpose");
        }
        if (!seam_codes(pf.rCodes[c.qnPB(ai)], pf.rMX[c.mmPV(ai)], lseq, lkPV, lseq))
            return fail("seam: pv x codes");
        if (!seam_codes(pf.rCodes[c.qnVT(ai)], pf.rMW[c.mmPV(ai)], lseq, lkPV, ldh))
            return fail("seam: pv w codes");
        {
            std::vector<gl_t> z = chal_vec(tr, ldh + lseq);
            if (!seam_pair(pf.rAtX, slice_pt(z, ldh, lnh, h, lB, b), rY(c.mmPV(ai)), z))
                return fail("seam: concat half");
        }
    }
    if (!seam_codes(pf.rCodes[c.qnAT()], pf.rMX[c.mmWO()], ld, lkH1, lT))
        return fail("seam: attn codes");
    if (!seam_codes(pf.rCodes[c.qnH2()], pf.rMX[c.mmWG()], ld, lkH1, lT))
        return fail("seam: h2 codes");
    if (!seam_codes(pf.rCodes[c.qnSW()], pf.rMX[c.mmWD()], ldff, lkSW, lT))
        return fail("seam: swiglu codes");
    {   // output binding: committed final residual == the PUBLIC out, bitwise
        std::vector<gl_t> z = chal_vec(tr, ld + lT);
        std::vector<gl_t> op(outpub.begin(), outpub.end());
        if (vclp(pf.rOut, z) != mle_eval(op, z) || seam_short)
            return fail("public output binding");
    }
    if (si != pf.seam.size()) return fail("seam claim count");

    // -- the ONE shared batched opening per size class --
    if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
    for (size_t i = 0; i < vlg.cls.size(); i++)
        if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                "tf-bo" + std::to_string(i), why)) return false;

    if (why) *why = "ok";
    return true;
}

// ---------------- proof size (serialized-payload bytes) ----------------
static inline size_t proof_size(const TfProof& pf) {
    size_t s = 24;
    for (int i = 0; i < 2; i++) s += p3rms::proof_size(pf.rms[i]);
    for (auto& q : pf.qn) s += p3qnt::proof_size(q);
    for (auto& m : pf.mm) s += p3hwl::proof_size(m);
    for (auto& r : pf.rp) s += p3rope::proof_size(r);
    for (auto& m : pf.sm) s += p3smx::proof_size(m);
    for (int i = 0; i < 2; i++) s += p3bfa::proof_size(pf.res[i]);
    s += p3swg::proof_size(pf.sw);
    s += 32 * (size_t)(1 + 2 + 2 * pf.qn.size() + 2 * pf.mm.size()
                       + 2 * pf.rp.size() + 2 * pf.sm.size() + 4);
    for (auto& y : pf.mmY) s += y.size() * 2;
    for (auto& g : pf.lug) s += p3lu::sz_group(g);
    s += pf.seam.size() * 8;
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3tf
