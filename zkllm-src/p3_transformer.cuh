// Composed FULL-TRANSFORMER-LAYER prover+verifier (design doc section 11.6
// R3+R5): proves the ENTIRE tiny-layer dataflow of transformer_ref.py,
//
//   x - rmsnorm(g1) - quant -[Wq,Wk,Wv]- rope(Q,K) -
//     per head [ quant - QK^T - softmax - quant - (quant V^T) - P.V ] - concat -
//   quant - Wo - residual - rmsnorm(g2) - quant -[Wg,Wu]- swiglu - quant - Wd -
//   residual - out
//
// as ONE proof: 34 gadget instances (2 rmsnorm, 12 quantize, 11 Hawkeye
// matmuls, 4 rope, 2 softmax, 2 residual bf16-adds, 1 swiglu) over ONE
// Fiat-Shamir transcript with ONE shared opening ledger (p3lu::XCtx) and a
// single per-size-class batched-opening pass at the end (the R3 merge).
//
// CHAINING (the section-11.5 vocabulary, all four moves used):
//  * shared roots: wherever producer and consumer commit the SAME values on the
//    SAME domain, the composed verifier passes ONE root hash to both gadget
//    verifiers (rmsnorm.Y == quant.X; quant.SCALES == matmul.XS; rope.OUT ==
//    quant.X; softmax.P == quant.X; swiglu.M == quant.X; bfadd.OUT ==
//    rmsnorm.X == bfadd.X1; matmul output column rdo[O_YB] == softmax.S ==
//    swiglu.GATE/UP == bfadd.X2 -- the matmul's committed output witness
//    column IS the downstream operand commitment).  No re-commitment, no
//    equality argument needed: one root, both sub-proofs bound to it.
//  * restriction seams: quant CODES (B x d) vs matmul X (B x Kpad, canonical
//    zero padding): CODES~(z) == X~(z with the high k-bits fixed to 0), plus
//    one zero-claim X~(.)==0 per padding subcube [2^t, 2^{t+1}) -- so the
//    padded operand is PINNED to be the canonical extension of the quantizer
//    output (matters: PV's group-0 padding k in [seq,32) feeds real outputs).
//  * slice / transpose / concat seams: head slice = insert the head index bit
//    into the opening point; V^T = swap the variable groups; concat = open the
//    parent at the half-fixed point.  Each seam = TWO ledger claims that must
//    agree at a fresh random point (Schwartz-Zippel over the shared
//    transcript, drawn AFTER all commitments).
//  * public statement bindings: committed layer input == public x (verifier
//    evaluates the public MLE itself), committed final residual output ==
//    public out (bitwise binding of the accepted output to the golden trace).
//
// Public statement: input patterns + root, g1/g2 roots, the 7 weight
// (codes+scales) roots, the pinned rope cos/sin + canonical table artifacts,
// and the OUTPUT patterns.  Intermediate activation roots ride in the proof,
// pinned by the chain; the matmuls' bf16 outputs additionally appear as
// public vectors (the Hawkeye atom's native output binding) -- non-ZK, as
// stated (ZK wiring = R7).
//
// Soundness of the composition = soundness of each gadget + the seam argument
// above; every opened evaluation across all 34 instances is authenticated by
// the SHARED batched opening against its commitment root, and every root is
// either caller-pinned or pinned by exactly one proof field that both its
// producer and consumer sub-verifiers absorb.
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

// ---------------- fixed instance indices ----------------
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
    uint32_t seq = 4, d = 64, nh = 2, dh = 32, dff = 128;
    uint32_t lseq() const { return ilog2(seq); }
    uint32_t ld()   const { return ilog2(d); }
    uint32_t ldh()  const { return ilog2(dh); }
    uint32_t ldff() const { return ilog2(dff); }
    bool pow2() const {
        return (1u << lseq()) == seq && (1u << ld()) == d && (1u << ldh()) == dh
            && (1u << ldff()) == dff && nh * dh == d && nh == 2;
    }
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
    W.cfg.dh = (uint32_t)hdr[5]; W.cfg.dff = (uint32_t)hdr[6];
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
    p3qnt::Wit qn[NQN];
    p3hwl::LayerWit mm[NMM];
    p3rope::GoldenSet rgs;                        // cos/sin container
    p3rope::Wit rp[4];                            // q0,k0,q1,k1
    p3smx::Wit sm[2];
    std::vector<uint8_t> smmask;                  // causal mask bytes (seq*seq)
    p3bfa::Wit res[2];
    p3swg::Wit sw;
    // chained activation patterns (REAL row-major grids)
    std::vector<uint16_t> x0, rms1y, ropq[2], ropk[2], probs[2], vt[2],
                          attn, res1o, rms2y, swm, outp;
    std::vector<uint16_t> mmY[NMM];               // honest matmul outputs
    std::vector<uint16_t> mmYpub[NMM];            // the PUBLIC per-matmul claims
};

// build the full chained witness from (input, weights): each gadget's canonical
// replay output feeds the next gadget's input -- the layer is COMPUTED by the
// gadget replays themselves, so a tamper anywhere propagates honestly and the
// proof must reject exactly at the owning gadget or seam.
static inline TfWit build_witness(const Config& cfg, const std::vector<uint16_t>& x0,
                                  const Weights& W, const Art& a, int tamper = TFT_NONE) {
    if (!cfg.pow2()) throw std::runtime_error("tf: config must be pow2, nh=2");
    TfWit w; w.cfg = cfg;
    const uint32_t seq = cfg.seq, d = cfg.d, dh = cfg.dh, dff = cfg.dff;
    w.x0 = x0;
    auto flip = [](std::vector<uint16_t>& v, size_t i) { v[i] ^= 1; };

    // -- rmsnorm 1 --
    {
        p3rms::Golden g; g.B = seq; g.d = d; g.x = w.x0; g.g = W.g1;
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
        p3qnt::Golden g; g.B = seq; g.d = d; g.x = w.rms1y;
        p3qnt::QTamper qt{p3qnt::QT_MAG, 0, 3};
        w.qn[QN_H1] = p3qnt::gen_witness(g, a, tamper == TFT_GW_QNT ? &qt : nullptr);
    }

    auto mmgold = [](uint32_t B, uint32_t K, uint32_t N,
                     const std::vector<uint8_t>& x, const std::vector<uint32_t>& xs,
                     const std::vector<uint8_t>& wc, const std::vector<uint32_t>& ws) {
        p3hwl::Golden L; L.B = B; L.K = K; L.N = N;
        L.x = x; L.w = wc; L.xs = xs; L.ws = ws; L.y.assign((size_t)B * N, 0);
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
            mmgold(seq, d, d, cx, w.qn[QN_H1].S, W.w[W_Q].codes, W.w[W_Q].scales), true,
            tamper == TFT_GW_MM ? &hm : nullptr);
        w.mm[MM_WK] = p3hwl::gen_witness(
            mmgold(seq, d, d, cx, w.qn[QN_H1].S, W.w[W_K].codes, W.w[W_K].scales));
        w.mm[MM_WV] = p3hwl::gen_witness(
            mmgold(seq, d, d, cx, w.qn[QN_H1].S, W.w[W_V].codes, W.w[W_V].scales));
        for (int i : {MM_WQ, MM_WK, MM_WV}) w.mmY[i] = w.mm[i].Y;
    }

    // -- attention, per head --
    w.rgs.seq = seq; w.rgs.dh = dh; w.rgs.cos = W.cos; w.rgs.sin = W.sin;
    w.smmask.assign((size_t)seq * seq, 0);
    for (uint32_t i = 0; i < seq; i++)
        for (uint32_t j = 0; j <= i; j++) w.smmask[(size_t)i * seq + j] = 1;
    w.attn.assign((size_t)seq * d, 0);
    for (int h = 0; h < 2; h++) {
        auto slice_case = [&](const std::vector<uint16_t>& Y) {
            p3rope::GoldenSet::Case c; c.flags = 0; c.q.resize((size_t)seq * dh);
            for (uint32_t p = 0; p < seq; p++)
                for (uint32_t j = 0; j < dh; j++)
                    c.q[(size_t)p * dh + j] = Y[(size_t)p * d + (uint32_t)h * dh + j];
            return c;
        };
        // rope q
        {
            p3rope::GoldenSet::Case cq = slice_case(w.mmY[MM_WQ]);
            if (tamper == TFT_SEAM_ROPEQ && h == 0) cq.q[3] ^= 1;
            w.rp[2 * h] = p3rope::gen_witness(w.rgs, cq, a);
            w.ropq[h].assign(w.rp[2 * h].OUT.begin(), w.rp[2 * h].OUT.end());
            if (tamper == TFT_ROPE_OUT && h == 0) {
                flip(w.ropq[h], 2);
                w.rp[2 * h].opat[2] = gl_add(w.rp[2 * h].opat[2], 1ULL);
            }
        }
        // rope k
        {
            p3rope::GoldenSet::Case ck = slice_case(w.mmY[MM_WK]);
            p3rope::RpTamper rt{p3rope::RPT_MUL, 0};
            w.rp[2 * h + 1] = p3rope::gen_witness(w.rgs, ck, a,
                (tamper == TFT_GW_ROPE && h == 1) ? &rt : nullptr);
            w.ropk[h].assign(w.rp[2 * h + 1].OUT.begin(), w.rp[2 * h + 1].OUT.end());
        }
        // quantize rotated q / k
        {
            p3qnt::Golden g; g.B = seq; g.d = dh; g.x = w.ropq[h];
            w.qn[QN_RQ[h]] = p3qnt::gen_witness(g, a);
            g.x = w.ropk[h];
            w.qn[QN_RK[h]] = p3qnt::gen_witness(g, a);
        }
        // scores = QK^T
        w.mm[MM_QK[h]] = p3hwl::gen_witness(
            mmgold(seq, dh, seq, w.qn[QN_RQ[h]].C, w.qn[QN_RQ[h]].S,
                   w.qn[QN_RK[h]].C, w.qn[QN_RK[h]].S));
        w.mmY[MM_QK[h]] = w.mm[MM_QK[h]].Y;
        // softmax
        {
            p3smx::Golden g; g.B = seq; g.n = seq;
            g.s = w.mmY[MM_QK[h]]; g.msk = w.smmask;
            g.p.assign((size_t)seq * seq, 0);
            p3smx::SmTamper st{p3smx::SMT_MASKLEAK, 2, 3};
            w.sm[h] = p3smx::gen_witness(g, a, (tamper == TFT_GW_SMX && h == 0) ? &st : nullptr);
        }
        w.probs[h].assign(w.sm[h].P.begin(), w.sm[h].P.end());
        if (tamper == TFT_SMX_P && h == 0) {
            flip(w.probs[h], 1);
            w.sm[h].ppat[1] = gl_add(w.sm[h].ppat[1], 1ULL);
        }
        // quantize probs
        {
            p3qnt::Golden g; g.B = seq; g.d = seq; g.x = w.probs[h];
            w.qn[QN_PB[h]] = p3qnt::gen_witness(g, a);
        }
        // V^T head slice
        w.vt[h].assign((size_t)dh * seq, 0);
        for (uint32_t j = 0; j < dh; j++)
            for (uint32_t p = 0; p < seq; p++)
                w.vt[h][(size_t)j * seq + p] = w.mmY[MM_WV][(size_t)p * d + (uint32_t)h * dh + j];
        if (tamper == TFT_SEAM_VT && h == 0) flip(w.vt[h], 5);
        {
            p3qnt::Golden g; g.B = dh; g.d = seq; g.x = w.vt[h];
            w.qn[QN_VT[h]] = p3qnt::gen_witness(g, a);
        }
        // attnout = P.V
        {
            p3hwl::Golden L = mmgold(seq, seq, dh, w.qn[QN_PB[h]].C, w.qn[QN_PB[h]].S,
                                     w.qn[QN_VT[h]].C, w.qn[QN_VT[h]].S);
            if (tamper == TFT_SEAM_PVPAD && h == 0) {
                // smuggle a nonzero code into group-0 k-padding: self-consistent
                // matmul of a value the quantizer never produced
                p3hwl::Dims dd = p3hwl::make_dims(seq, seq, dh);
                std::vector<uint8_t> ovr((size_t)dd.Bpad * dd.Kpad, 0);
                for (uint32_t b = 0; b < seq; b++)
                    for (uint32_t k = 0; k < seq; k++)
                        ovr[(size_t)b * dd.Kpad + k] = L.x[(size_t)b * seq + k];
                ovr[0 * dd.Kpad + 5] = 0x30;
                w.mm[MM_PV[h]] = p3hwl::gen_witness(L, true, nullptr, &ovr);
            } else {
                w.mm[MM_PV[h]] = p3hwl::gen_witness(L);
            }
        }
        w.mmY[MM_PV[h]] = w.mm[MM_PV[h]].Y;
        for (uint32_t p = 0; p < seq; p++)
            for (uint32_t j = 0; j < dh; j++)
                w.attn[(size_t)p * d + (uint32_t)h * dh + j] = w.mmY[MM_PV[h]][(size_t)p * dh + j];
    }
    if (tamper == TFT_SEAM_CONCAT) flip(w.attn, (size_t)1 * d + dh + 2);   // head-1 region

    // -- quantize attn, Wo, residual 1 --
    {
        p3qnt::Golden g; g.B = seq; g.d = d; g.x = w.attn;
        w.qn[QN_AT] = p3qnt::gen_witness(g, a);
    }
    w.mm[MM_WO] = p3hwl::gen_witness(
        mmgold(seq, d, d, w.qn[QN_AT].C, w.qn[QN_AT].S, W.w[W_O].codes, W.w[W_O].scales));
    w.mmY[MM_WO] = w.mm[MM_WO].Y;
    {
        p3bfa::Golden g; g.n = seq * d; g.flags = 0;
        g.a = w.x0; g.b = w.mmY[MM_WO]; g.o.assign(g.n, 0);
        w.res[0] = p3bfa::gen_witness(g);
    }
    w.res1o.assign(w.res[0].O.begin(), w.res[0].O.end());
    if (tamper == TFT_RES1_OUT) {
        flip(w.res1o, 9);
        w.res[0].opat[9] = gl_add(w.res[0].opat[9], 1ULL);
    }

    // -- rmsnorm 2, quantize h2, Wg / Wu --
    {
        p3rms::Golden g; g.B = seq; g.d = d; g.x = w.res1o; g.g = W.g2;
        w.rms[1] = p3rms::gen_witness(g, a);
    }
    w.rms2y = w.rms[1].Y;
    {
        p3qnt::Golden g; g.B = seq; g.d = d; g.x = w.rms2y;
        w.qn[QN_H2] = p3qnt::gen_witness(g, a);
    }
    w.mm[MM_WG] = p3hwl::gen_witness(
        mmgold(seq, d, dff, w.qn[QN_H2].C, w.qn[QN_H2].S, W.w[W_G].codes, W.w[W_G].scales));
    w.mm[MM_WU] = p3hwl::gen_witness(
        mmgold(seq, d, dff, w.qn[QN_H2].C, w.qn[QN_H2].S, W.w[W_U].codes, W.w[W_U].scales));
    w.mmY[MM_WG] = w.mm[MM_WG].Y; w.mmY[MM_WU] = w.mm[MM_WU].Y;

    // -- swiglu, quantize, Wd, residual 2 --
    {
        p3swg::Golden g; g.n = seq * dff;
        g.gate = w.mmY[MM_WG]; g.up = w.mmY[MM_WU]; g.m.assign(g.n, 0);
        p3swg::STamper st{p3swg::ST_SILU, 6};
        w.sw = p3swg::gen_witness(g, a, tamper == TFT_GW_SWG ? &st : nullptr);
    }
    w.swm.assign(w.sw.M.begin(), w.sw.M.end());
    if (tamper == TFT_SWG_M) {
        flip(w.swm, 4);
        w.sw.mpat[4] = gl_add(w.sw.mpat[4], 1ULL);
    }
    {
        p3qnt::Golden g; g.B = seq; g.d = dff; g.x = w.swm;
        w.qn[QN_SW] = p3qnt::gen_witness(g, a);
    }
    w.mm[MM_WD] = p3hwl::gen_witness(
        mmgold(seq, dff, d, w.qn[QN_SW].C, w.qn[QN_SW].S, W.w[W_D].codes, W.w[W_D].scales));
    w.mmY[MM_WD] = w.mm[MM_WD].Y;
    {
        p3bfa::Golden g; g.n = seq * d; g.flags = 0;
        g.a = w.res1o; g.b = w.mmY[MM_WD]; g.o.assign(g.n, 0);
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

    for (int i = 0; i < NMM; i++) w.mmYpub[i] = w.mmY[i];
    if (tamper == TFT_MM_YPUB) w.mmYpub[MM_WO][3] ^= 1;   // committed column stays honest
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
static inline TfTables build_tables(const Art& a) {
    TfTables T;
    T.hw = p3hwl::build_tables();
    T.rms = p3rms::build_tables(a);
    T.qnt = p3qnt::build_tables(a);
    T.rope = p3rope::build_tables(a);
    T.smx = p3smx::build_tables(a);
    T.bfa = p3bfa::build_tables();
    T.swg = p3swg::build_tables(a);
    return T;
}

// ---------------- operand commitments (the chain, committed once) ----------------
struct TfOps {
    p3rms::Operands rms[2];
    p3qnt::Operands qn[NQN];
    p3hwl::Operands mm[NMM];
    p3rope::Operands rp[4];
    p3smx::Operands sm[2];
    p3bfa::Operands res[2];
    p3swg::Operands sw;
    Col Ymm[NMM];                                 // matmul output columns [n|b]
};
static inline Col commit_u8(const std::vector<uint8_t>& v, uint32_t R) {
    std::vector<gl_t> g(v.begin(), v.end());
    return commit_col_nc(std::move(g), R);
}
static inline TfOps commit_all(const TfWit& w, uint32_t R) {
    TfOps o;
    // rms1: X = committed layer input, G = g1, Y fresh
    o.rms[0] = p3rms::commit_operands(w.rms[0], R);
    // quant h1 shares rms1.Y as its X
    o.qn[QN_H1].X = o.rms[0].Y;
    o.qn[QN_H1].CODES = commit_col_nc(w.qn[QN_H1].cpat, R);
    o.qn[QN_H1].SCALES = commit_col_nc(w.qn[QN_H1].spat, R);
    // Wq/Wk/Wv: ONE padded X-code commitment, shared scales
    {
        Col X1 = commit_u8(w.mm[MM_WQ].xcodes, R);
        for (int i : {MM_WQ, MM_WK, MM_WV}) {
            o.mm[i].X = X1;
            o.mm[i].W = commit_u8(w.mm[i].wcodes, R);
            o.mm[i].XS = o.qn[QN_H1].SCALES;
            o.mm[i].WS = commit_col_nc(w.mm[i].wsb, R);
        }
    }
    for (int i : {MM_WQ, MM_WK, MM_WV})
        o.Ymm[i] = commit_col_nc(w.mm[i].dob[p3hwl::O_YB], R);
    for (int h = 0; h < 2; h++) {
        o.rp[2 * h] = p3rope::commit_operands(w.rp[2 * h], R);
        o.rp[2 * h + 1] = p3rope::commit_operands(w.rp[2 * h + 1], R);
        o.qn[QN_RQ[h]].X = o.rp[2 * h].OUT;
        o.qn[QN_RQ[h]].CODES = commit_col_nc(w.qn[QN_RQ[h]].cpat, R);
        o.qn[QN_RQ[h]].SCALES = commit_col_nc(w.qn[QN_RQ[h]].spat, R);
        o.qn[QN_RK[h]].X = o.rp[2 * h + 1].OUT;
        o.qn[QN_RK[h]].CODES = commit_col_nc(w.qn[QN_RK[h]].cpat, R);
        o.qn[QN_RK[h]].SCALES = commit_col_nc(w.qn[QN_RK[h]].spat, R);
        o.mm[MM_QK[h]].X = commit_u8(w.mm[MM_QK[h]].xcodes, R);
        o.mm[MM_QK[h]].W = commit_u8(w.mm[MM_QK[h]].wcodes, R);
        o.mm[MM_QK[h]].XS = o.qn[QN_RQ[h]].SCALES;
        o.mm[MM_QK[h]].WS = o.qn[QN_RK[h]].SCALES;
        o.Ymm[MM_QK[h]] = commit_col_nc(w.mm[MM_QK[h]].dob[p3hwl::O_YB], R);
        o.sm[h].S = o.Ymm[MM_QK[h]];
        o.sm[h].P = commit_col_nc(w.sm[h].ppat, R);
        o.qn[QN_PB[h]].X = o.sm[h].P;
        o.qn[QN_PB[h]].CODES = commit_col_nc(w.qn[QN_PB[h]].cpat, R);
        o.qn[QN_PB[h]].SCALES = commit_col_nc(w.qn[QN_PB[h]].spat, R);
        o.qn[QN_VT[h]].X = commit_col_nc(w.qn[QN_VT[h]].xpat, R);
        o.qn[QN_VT[h]].CODES = commit_col_nc(w.qn[QN_VT[h]].cpat, R);
        o.qn[QN_VT[h]].SCALES = commit_col_nc(w.qn[QN_VT[h]].spat, R);
        o.mm[MM_PV[h]].X = commit_u8(w.mm[MM_PV[h]].xcodes, R);
        o.mm[MM_PV[h]].W = commit_u8(w.mm[MM_PV[h]].wcodes, R);
        o.mm[MM_PV[h]].XS = o.qn[QN_PB[h]].SCALES;
        o.mm[MM_PV[h]].WS = o.qn[QN_VT[h]].SCALES;
        o.Ymm[MM_PV[h]] = commit_col_nc(w.mm[MM_PV[h]].dob[p3hwl::O_YB], R);
    }
    o.qn[QN_AT].X = commit_col_nc(w.qn[QN_AT].xpat, R);
    o.qn[QN_AT].CODES = commit_col_nc(w.qn[QN_AT].cpat, R);
    o.qn[QN_AT].SCALES = commit_col_nc(w.qn[QN_AT].spat, R);
    o.mm[MM_WO].X = commit_u8(w.mm[MM_WO].xcodes, R);
    o.mm[MM_WO].W = commit_u8(w.mm[MM_WO].wcodes, R);
    o.mm[MM_WO].XS = o.qn[QN_AT].SCALES;
    o.mm[MM_WO].WS = commit_col_nc(w.mm[MM_WO].wsb, R);
    o.Ymm[MM_WO] = commit_col_nc(w.mm[MM_WO].dob[p3hwl::O_YB], R);
    o.res[0].X1 = o.rms[0].X;
    o.res[0].X2 = o.Ymm[MM_WO];
    o.res[0].OUT = commit_col_nc(w.res[0].opat, R);
    o.rms[1].X = o.res[0].OUT;
    o.rms[1].G = commit_col_nc(w.rms[1].gpat, R);
    o.rms[1].Y = commit_col_nc(w.rms[1].ypat, R);
    o.qn[QN_H2].X = o.rms[1].Y;
    o.qn[QN_H2].CODES = commit_col_nc(w.qn[QN_H2].cpat, R);
    o.qn[QN_H2].SCALES = commit_col_nc(w.qn[QN_H2].spat, R);
    {
        Col X2 = commit_u8(w.mm[MM_WG].xcodes, R);
        for (int i : {MM_WG, MM_WU}) {
            o.mm[i].X = X2;
            o.mm[i].W = commit_u8(w.mm[i].wcodes, R);
            o.mm[i].XS = o.qn[QN_H2].SCALES;
            o.mm[i].WS = commit_col_nc(w.mm[i].wsb, R);
            o.Ymm[i] = commit_col_nc(w.mm[i].dob[p3hwl::O_YB], R);
        }
    }
    o.sw.GATE = o.Ymm[MM_WG];
    o.sw.UP = o.Ymm[MM_WU];
    o.sw.M = commit_col_nc(w.sw.mpat, R);
    o.qn[QN_SW].X = o.sw.M;
    o.qn[QN_SW].CODES = commit_col_nc(w.qn[QN_SW].cpat, R);
    o.qn[QN_SW].SCALES = commit_col_nc(w.qn[QN_SW].spat, R);
    o.mm[MM_WD].X = commit_u8(w.mm[MM_WD].xcodes, R);
    o.mm[MM_WD].W = commit_u8(w.mm[MM_WD].wcodes, R);
    o.mm[MM_WD].XS = o.qn[QN_SW].SCALES;
    o.mm[MM_WD].WS = commit_col_nc(w.mm[MM_WD].wsb, R);
    o.Ymm[MM_WD] = commit_col_nc(w.mm[MM_WD].dob[p3hwl::O_YB], R);
    o.res[1].X1 = o.res[0].OUT;
    o.res[1].X2 = o.Ymm[MM_WD];
    o.res[1].OUT = commit_col_nc(w.res[1].opat, R);
    return o;
}

// the caller's pinned weight-side commitments (the public statement): computed
// independently of the prover from the weights file + canonical padding
struct WeightRoots { Hash Wc[NW], Ws[NW], G1, G2; };
static inline WeightRoots weight_roots(const Weights& W, uint32_t R) {
    WeightRoots wr;
    const Config& c = W.cfg;
    const uint32_t Bmm[NW] = {c.seq, c.seq, c.seq, c.seq, c.seq, c.seq, c.seq};
    for (int i = 0; i < NW; i++) {
        p3hwl::Dims dd = p3hwl::make_dims(Bmm[i], W.w[i].K, W.w[i].N);
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
    uint32_t seq = 0, d = 0, nh = 0, dh = 0, dff = 0;
    // sub-proofs (fixed protocol order)
    p3rms::RmsProof rms[2];
    p3qnt::QntProof qn[NQN];
    p3hwl::LayerProof mm[NMM];
    p3rope::RopeProof rp[4];
    p3smx::SmxProof sm[2];
    p3bfa::BfaProof res[2];
    p3swg::SwgProof sw;
    // chained intermediate roots (pinned by BOTH neighbours' sub-verifies)
    Hash rX0;                                     // committed layer input
    Hash rRmsY[2];
    Hash rCodes[NQN], rScales[NQN];
    Hash rMX[NMM], rMW[NMM];                      // padded matmul operand columns
    Hash rRopeQ[4], rRopeO[4];
    Hash rSmP[2], rVtX[2], rAtX;
    Hash rRes1, rSwM, rOut;
    // public per-matmul output claims (the Hawkeye atom's native binding)
    std::vector<uint16_t> mmY[NMM];
    // seam claim evaluations, fixed order
    std::vector<gl_t> seam;
    // ONE batched opening per size class for the WHOLE layer (R3)
    std::vector<p3bo::BatchProof> batches;
};

// per-stage prover timings (ms)
struct TfProf {
    double commit = 0, rms = 0, qnt = 0, mm = 0, rope = 0, smx = 0, bfa = 0,
           swg = 0, seam = 0, batch = 0, total = 0;
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

// ==================== prover ====================
static inline TfProof prove(fs::Transcript& tr, const TfWit& w, const TfOps& o,
                            const TfTables& T, const Art& a, uint32_t R, uint32_t Q,
                            bool strict = true, TfProf* prof = nullptr) {
    const Config& c = w.cfg;
    TfProof pf;
    pf.seq = c.seq; pf.d = c.d; pf.nh = c.nh; pf.dh = c.dh; pf.dff = c.dff;
    TfProf pl; TfProf& P = prof ? *prof : pl;
    double tall = now_ms(), tp;

    uint32_t hdr[5] = {c.seq, c.d, c.nh, c.dh, c.dff};
    tr.absorb("tf-dims", hdr, sizeof hdr);

    p3lu::XCtx xc;

    // roots into the proof (the chain the verifier re-pins)
    pf.rX0 = o.rms[0].X.root;
    for (int i = 0; i < 2; i++) pf.rRmsY[i] = o.rms[i].Y.root;
    for (int i = 0; i < NQN; i++) { pf.rCodes[i] = o.qn[i].CODES.root; pf.rScales[i] = o.qn[i].SCALES.root; }
    for (int i = 0; i < NMM; i++) { pf.rMX[i] = o.mm[i].X.root; pf.rMW[i] = o.mm[i].W.root; }
    for (int i = 0; i < 4; i++) { pf.rRopeQ[i] = o.rp[i].Q.root; pf.rRopeO[i] = o.rp[i].OUT.root; }
    for (int h = 0; h < 2; h++) { pf.rSmP[h] = o.sm[h].P.root; pf.rVtX[h] = o.qn[QN_VT[h]].X.root; }
    pf.rAtX = o.qn[QN_AT].X.root;
    pf.rRes1 = o.res[0].OUT.root;
    pf.rSwM = o.sw.M.root;
    pf.rOut = o.res[1].OUT.root;
    for (int i = 0; i < NMM; i++) pf.mmY[i] = w.mmYpub[i];

    // -- sub-proofs, fixed order, one transcript, one ledger --
    tp = now_ms();
    pf.rms[0] = p3rms::prove(tr, w.rms[0], T.rms, a, o.rms[0], R, Q, strict, &xc);
    P.rms += now_ms() - tp; tp = now_ms();
    pf.qn[QN_H1] = p3qnt::prove(tr, w.qn[QN_H1], T.qnt, o.qn[QN_H1], R, Q, strict, &xc);
    P.qnt += now_ms() - tp; tp = now_ms();
    for (int i : {MM_WQ, MM_WK, MM_WV})
        pf.mm[i] = p3hwl::prove(tr, w.mm[i], T.hw, o.mm[i], R, Q, true, strict,
                                nullptr, &w.mmYpub[i], &xc);
    P.mm += now_ms() - tp;
    for (int h = 0; h < 2; h++) {
        tp = now_ms();
        pf.rp[2 * h] = p3rope::prove(tr, w.rp[2 * h], T.rope, o.rp[2 * h], R, Q, strict, &xc);
        pf.rp[2 * h + 1] = p3rope::prove(tr, w.rp[2 * h + 1], T.rope, o.rp[2 * h + 1], R, Q, strict, &xc);
        P.rope += now_ms() - tp; tp = now_ms();
        pf.qn[QN_RQ[h]] = p3qnt::prove(tr, w.qn[QN_RQ[h]], T.qnt, o.qn[QN_RQ[h]], R, Q, strict, &xc);
        pf.qn[QN_RK[h]] = p3qnt::prove(tr, w.qn[QN_RK[h]], T.qnt, o.qn[QN_RK[h]], R, Q, strict, &xc);
        P.qnt += now_ms() - tp; tp = now_ms();
        pf.mm[MM_QK[h]] = p3hwl::prove(tr, w.mm[MM_QK[h]], T.hw, o.mm[MM_QK[h]], R, Q,
                                       true, strict, nullptr, &w.mmYpub[MM_QK[h]], &xc);
        P.mm += now_ms() - tp; tp = now_ms();
        pf.sm[h] = p3smx::prove(tr, w.sm[h], T.smx, a, o.sm[h], R, Q, strict, &xc);
        P.smx += now_ms() - tp; tp = now_ms();
        pf.qn[QN_PB[h]] = p3qnt::prove(tr, w.qn[QN_PB[h]], T.qnt, o.qn[QN_PB[h]], R, Q, strict, &xc);
        pf.qn[QN_VT[h]] = p3qnt::prove(tr, w.qn[QN_VT[h]], T.qnt, o.qn[QN_VT[h]], R, Q, strict, &xc);
        P.qnt += now_ms() - tp; tp = now_ms();
        pf.mm[MM_PV[h]] = p3hwl::prove(tr, w.mm[MM_PV[h]], T.hw, o.mm[MM_PV[h]], R, Q,
                                       true, strict, nullptr, &w.mmYpub[MM_PV[h]], &xc);
        P.mm += now_ms() - tp;
    }
    tp = now_ms();
    pf.qn[QN_AT] = p3qnt::prove(tr, w.qn[QN_AT], T.qnt, o.qn[QN_AT], R, Q, strict, &xc);
    P.qnt += now_ms() - tp; tp = now_ms();
    pf.mm[MM_WO] = p3hwl::prove(tr, w.mm[MM_WO], T.hw, o.mm[MM_WO], R, Q, true, strict,
                                nullptr, &w.mmYpub[MM_WO], &xc);
    P.mm += now_ms() - tp; tp = now_ms();
    pf.res[0] = p3bfa::prove(tr, w.res[0], T.bfa, o.res[0], R, Q, strict, &xc);
    P.bfa += now_ms() - tp; tp = now_ms();
    pf.rms[1] = p3rms::prove(tr, w.rms[1], T.rms, a, o.rms[1], R, Q, strict, &xc);
    P.rms += now_ms() - tp; tp = now_ms();
    pf.qn[QN_H2] = p3qnt::prove(tr, w.qn[QN_H2], T.qnt, o.qn[QN_H2], R, Q, strict, &xc);
    P.qnt += now_ms() - tp; tp = now_ms();
    for (int i : {MM_WG, MM_WU})
        pf.mm[i] = p3hwl::prove(tr, w.mm[i], T.hw, o.mm[i], R, Q, true, strict,
                                nullptr, &w.mmYpub[i], &xc);
    P.mm += now_ms() - tp; tp = now_ms();
    pf.sw = p3swg::prove(tr, w.sw, T.swg, o.sw, R, Q, strict, &xc);
    P.swg += now_ms() - tp; tp = now_ms();
    pf.qn[QN_SW] = p3qnt::prove(tr, w.qn[QN_SW], T.qnt, o.qn[QN_SW], R, Q, strict, &xc);
    P.qnt += now_ms() - tp; tp = now_ms();
    pf.mm[MM_WD] = p3hwl::prove(tr, w.mm[MM_WD], T.hw, o.mm[MM_WD], R, Q, true, strict,
                                nullptr, &w.mmYpub[MM_WD], &xc);
    P.mm += now_ms() - tp; tp = now_ms();
    pf.res[1] = p3bfa::prove(tr, w.res[1], T.bfa, o.res[1], R, Q, strict, &xc);
    P.bfa += now_ms() - tp;

    // -- seam claims (all roots are in the transcript; fresh challenges) --
    tp = now_ms();
    auto cl = [&](const Col& col, const std::vector<gl_t>& z) {
        gl_t y = claimc(tr, xc.lg, col, z);
        pf.seam.push_back(y);
        return y;
    };
    auto seam_pair = [&](const Col& A, const std::vector<gl_t>& zA,
                         const Col& B, const std::vector<gl_t>& zB, const char* what) {
        gl_t y1 = cl(A, zA), y2 = cl(B, zB);
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
            gl_t y = cl(X, pad_cube_pt(zt, t, lk));
            if (strict && y != 0)
                throw std::runtime_error(std::string("tf: padding not zero: ") + what);
        }
    };
    const uint32_t lseq = c.lseq(), ld = c.ld(), ldh = c.ldh(), ldff = c.ldff();
    const uint32_t lkH1 = ilog2(w.mm[MM_WQ].d.Kpad), lkQK = ilog2(w.mm[MM_QK0].d.Kpad),
                   lkPV = ilog2(w.mm[MM_PV0].d.Kpad), lkSW = ilog2(w.mm[MM_WD].d.Kpad);
    // input binding
    {
        std::vector<gl_t> z = chal_vec(tr, ld + lseq);
        cl(o.rms[0].X, z);
    }
    seam_codes(o.qn[QN_H1].CODES, o.mm[MM_WQ].X, ld, lkH1, lseq, "h1 codes");
    for (int h = 0; h < 2; h++) {
        {   // rope q/k operands = head slices of the Wq/Wk outputs
            std::vector<gl_t> z = chal_vec(tr, ldh + lseq);
            seam_pair(o.rp[2 * h].Q, z, o.Ymm[MM_WQ], ins_pt(z, ldh, h), "rope-q slice");
            std::vector<gl_t> z2 = chal_vec(tr, ldh + lseq);
            seam_pair(o.rp[2 * h + 1].Q, z2, o.Ymm[MM_WK], ins_pt(z2, ldh, h), "rope-k slice");
        }
        seam_codes(o.qn[QN_RQ[h]].CODES, o.mm[MM_QK[h]].X, ldh, lkQK, lseq, "qk x codes");
        seam_codes(o.qn[QN_RK[h]].CODES, o.mm[MM_QK[h]].W, ldh, lkQK, lseq, "qk w codes");
        {   // V^T: swap the variable groups of the Wv output, fix the head bit
            std::vector<gl_t> z = chal_vec(tr, lseq + ldh);
            std::vector<gl_t> zt(z.begin() + lseq, z.end());        // dh bits -> n low
            zt.push_back(h ? 1 : 0);                                // head bit -> n high
            zt.insert(zt.end(), z.begin(), z.begin() + lseq);       // seq bits -> b
            seam_pair(o.qn[QN_VT[h]].X, z, o.Ymm[MM_WV], zt, "v^T transpose");
        }
        seam_codes(o.qn[QN_PB[h]].CODES, o.mm[MM_PV[h]].X, lseq, lkPV, lseq, "pv x codes");
        seam_codes(o.qn[QN_VT[h]].CODES, o.mm[MM_PV[h]].W, lseq, lkPV, ldh, "pv w codes");
        {   // concat: attn head-h half == attnout_h
            std::vector<gl_t> z = chal_vec(tr, ldh + lseq);
            seam_pair(o.qn[QN_AT].X, ins_pt(z, ldh, h), o.Ymm[MM_PV[h]], z, "concat half");
        }
    }
    seam_codes(o.qn[QN_AT].CODES, o.mm[MM_WO].X, ld, lkH1, lseq, "attn codes");
    seam_codes(o.qn[QN_H2].CODES, o.mm[MM_WG].X, ld, lkH1, lseq, "h2 codes");
    seam_codes(o.qn[QN_SW].CODES, o.mm[MM_WD].X, ldff, lkSW, lseq, "swiglu codes");
    // output binding
    {
        std::vector<gl_t> z = chal_vec(tr, ld + lseq);
        cl(o.res[1].OUT, z);
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
    if (!c.pow2()) return fail("config must be pow2, nh=2");
    if (pf.seq != c.seq || pf.d != c.d || pf.nh != c.nh || pf.dh != c.dh || pf.dff != c.dff)
        return fail("dims mismatch");
    if (x0pub.size() != (size_t)c.seq * c.d || outpub.size() != (size_t)c.seq * c.d)
        return fail("public i/o size");
    if (!(pf.rX0 == rX0)) return fail("input root mismatch");
    for (int i = 0; i < NMM; i++) {
        uint32_t B = c.seq, N = (i == MM_QK0 || i == MM_QK1) ? c.seq
                              : (i == MM_PV0 || i == MM_PV1) ? c.dh
                              : (i == MM_WG || i == MM_WU) ? c.dff : c.d;
        if (pf.mmY[i].size() != (size_t)B * N) return fail("mmY size");
    }
    const uint32_t lseq = c.lseq(), ld = c.ld(), ldh = c.ldh(), ldff = c.ldff();

    uint32_t hdr[5] = {c.seq, c.d, c.nh, c.dh, c.dff};
    tr.absorb("tf-dims", hdr, sizeof hdr);

    p3bo::VLedger vlg;
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
                       c.seq, ld, Q_pub, R_pub, why, &vlg)) return false;
    if (!p3qnt::verify(tr, T.qnt, pf.qn[QN_H1], pf.rRmsY[0], pf.rCodes[QN_H1],
                       pf.rScales[QN_H1], c.seq, ld, Q_pub, R_pub, why, &vlg)) return false;
    {
        static const int WIDX[3] = {W_Q, W_K, W_V};
        int j = 0;
        for (int i : {MM_WQ, MM_WK, MM_WV}) {
            if (!(pf.rMX[i] == pf.rMX[MM_WQ])) return fail("h1 X root not shared");
            if (!(pf.rMW[i] == wr.Wc[WIDX[j]])) return fail("weight root mismatch");
            if (!p3hwl::verify(tr, T.hw, pf.mm[i], pf.rMX[i], wr.Wc[WIDX[j]],
                               pf.rScales[QN_H1], wr.Ws[WIDX[j]], pf.mmY[i],
                               c.seq, c.d, c.d, Q_pub, R_pub, why, &vlg)) return false;
            j++;
        }
    }
    for (int h = 0; h < 2; h++) {
        if (!p3rope::verify(tr, T.rope, pf.rp[2 * h], cosp, sinp, pf.rRopeQ[2 * h],
                            pf.rRopeO[2 * h], c.seq, c.dh, Q_pub, R_pub, why, &vlg)) return false;
        if (!p3rope::verify(tr, T.rope, pf.rp[2 * h + 1], cosp, sinp, pf.rRopeQ[2 * h + 1],
                            pf.rRopeO[2 * h + 1], c.seq, c.dh, Q_pub, R_pub, why, &vlg)) return false;
        if (!p3qnt::verify(tr, T.qnt, pf.qn[QN_RQ[h]], pf.rRopeO[2 * h], pf.rCodes[QN_RQ[h]],
                           pf.rScales[QN_RQ[h]], c.seq, ldh, Q_pub, R_pub, why, &vlg)) return false;
        if (!p3qnt::verify(tr, T.qnt, pf.qn[QN_RK[h]], pf.rRopeO[2 * h + 1], pf.rCodes[QN_RK[h]],
                           pf.rScales[QN_RK[h]], c.seq, ldh, Q_pub, R_pub, why, &vlg)) return false;
        if (!p3hwl::verify(tr, T.hw, pf.mm[MM_QK[h]], pf.rMX[MM_QK[h]], pf.rMW[MM_QK[h]],
                           pf.rScales[QN_RQ[h]], pf.rScales[QN_RK[h]], pf.mmY[MM_QK[h]],
                           c.seq, c.dh, c.seq, Q_pub, R_pub, why, &vlg)) return false;
        if (!p3smx::verify(tr, T.smx, a, pf.sm[h], causal, rY(MM_QK[h]), pf.rSmP[h],
                           c.seq, c.seq, Q_pub, R_pub, why, &vlg)) return false;
        if (!p3qnt::verify(tr, T.qnt, pf.qn[QN_PB[h]], pf.rSmP[h], pf.rCodes[QN_PB[h]],
                           pf.rScales[QN_PB[h]], c.seq, lseq, Q_pub, R_pub, why, &vlg)) return false;
        if (!p3qnt::verify(tr, T.qnt, pf.qn[QN_VT[h]], pf.rVtX[h], pf.rCodes[QN_VT[h]],
                           pf.rScales[QN_VT[h]], c.dh, lseq, Q_pub, R_pub, why, &vlg)) return false;
        if (!p3hwl::verify(tr, T.hw, pf.mm[MM_PV[h]], pf.rMX[MM_PV[h]], pf.rMW[MM_PV[h]],
                           pf.rScales[QN_PB[h]], pf.rScales[QN_VT[h]], pf.mmY[MM_PV[h]],
                           c.seq, c.seq, c.dh, Q_pub, R_pub, why, &vlg)) return false;
    }
    if (!p3qnt::verify(tr, T.qnt, pf.qn[QN_AT], pf.rAtX, pf.rCodes[QN_AT],
                       pf.rScales[QN_AT], c.seq, ld, Q_pub, R_pub, why, &vlg)) return false;
    if (!(pf.rMW[MM_WO] == wr.Wc[W_O])) return fail("weight root mismatch");
    if (!p3hwl::verify(tr, T.hw, pf.mm[MM_WO], pf.rMX[MM_WO], wr.Wc[W_O],
                       pf.rScales[QN_AT], wr.Ws[W_O], pf.mmY[MM_WO],
                       c.seq, c.d, c.d, Q_pub, R_pub, why, &vlg)) return false;
    if (!p3bfa::verify(tr, T.bfa, pf.res[0], rX0, rY(MM_WO), pf.rRes1,
                       c.seq * c.d, Q_pub, R_pub, why, &vlg)) return false;
    if (!p3rms::verify(tr, T.rms, a, pf.rms[1], pf.rRes1, wr.G2, pf.rRmsY[1],
                       c.seq, ld, Q_pub, R_pub, why, &vlg)) return false;
    if (!p3qnt::verify(tr, T.qnt, pf.qn[QN_H2], pf.rRmsY[1], pf.rCodes[QN_H2],
                       pf.rScales[QN_H2], c.seq, ld, Q_pub, R_pub, why, &vlg)) return false;
    {
        static const int WIDX[2] = {W_G, W_U};
        int j = 0;
        for (int i : {MM_WG, MM_WU}) {
            if (!(pf.rMX[i] == pf.rMX[MM_WG])) return fail("h2 X root not shared");
            if (!(pf.rMW[i] == wr.Wc[WIDX[j]])) return fail("weight root mismatch");
            if (!p3hwl::verify(tr, T.hw, pf.mm[i], pf.rMX[i], wr.Wc[WIDX[j]],
                               pf.rScales[QN_H2], wr.Ws[WIDX[j]], pf.mmY[i],
                               c.seq, c.d, c.dff, Q_pub, R_pub, why, &vlg)) return false;
            j++;
        }
    }
    if (!p3swg::verify(tr, T.swg, pf.sw, rY(MM_WG), rY(MM_WU), pf.rSwM,
                       c.seq * c.dff, Q_pub, R_pub, why, &vlg)) return false;
    if (!p3qnt::verify(tr, T.qnt, pf.qn[QN_SW], pf.rSwM, pf.rCodes[QN_SW],
                       pf.rScales[QN_SW], c.seq, ldff, Q_pub, R_pub, why, &vlg)) return false;
    if (!(pf.rMW[MM_WD] == wr.Wc[W_D])) return fail("weight root mismatch");
    if (!p3hwl::verify(tr, T.hw, pf.mm[MM_WD], pf.rMX[MM_WD], wr.Wc[W_D],
                       pf.rScales[QN_SW], wr.Ws[W_D], pf.mmY[MM_WD],
                       c.seq, c.dff, c.d, Q_pub, R_pub, why, &vlg)) return false;
    if (!p3bfa::verify(tr, T.bfa, pf.res[1], pf.rRes1, rY(MM_WD), pf.rOut,
                       c.seq * c.d, Q_pub, R_pub, why, &vlg)) return false;

    // -- seam checks (mirror of the prover's claim sequence) --
    size_t si = 0;
    bool seam_short = false;
    auto vcl = [&](const Hash& root, const std::vector<gl_t>& z) -> gl_t {
        if (si >= pf.seam.size()) { seam_short = true; return 0; }
        gl_t y = pf.seam[si++];
        return claimv(tr, vlg, root, z, y);
    };
    auto seam_pair = [&](const Hash& rA, const std::vector<gl_t>& zA,
                         const Hash& rB, const std::vector<gl_t>& zB) {
        gl_t y1 = vcl(rA, zA), y2 = vcl(rB, zB);
        return !seam_short && y1 == y2;
    };
    auto seam_codes = [&](const Hash& rC, const Hash& rX, uint32_t ldr, uint32_t lk,
                          uint32_t lb) {
        std::vector<gl_t> z = chal_vec(tr, ldr + lb);
        if (!seam_pair(rC, z, rX, ins_zeros(z, ldr, lk - ldr))) return false;
        for (uint32_t t = ldr; t < lk; t++) {
            std::vector<gl_t> zt = chal_vec(tr, t + lb);
            if (vcl(rX, pad_cube_pt(zt, t, lk)) != 0 || seam_short) return false;
        }
        return true;
    };
    const uint32_t lkH1 = ilog2(p3hwl::make_dims(c.seq, c.d, c.d).Kpad);
    const uint32_t lkQK = ilog2(p3hwl::make_dims(c.seq, c.dh, c.seq).Kpad);
    const uint32_t lkPV = ilog2(p3hwl::make_dims(c.seq, c.seq, c.dh).Kpad);
    const uint32_t lkSW = ilog2(p3hwl::make_dims(c.seq, c.dff, c.d).Kpad);
    {   // input binding: committed layer input == the PUBLIC x
        std::vector<gl_t> z = chal_vec(tr, ld + lseq);
        std::vector<gl_t> xp(x0pub.begin(), x0pub.end());
        if (vcl(rX0, z) != mle_eval(xp, z) || seam_short)
            return fail("public input binding");
    }
    if (!seam_codes(pf.rCodes[QN_H1], pf.rMX[MM_WQ], ld, lkH1, lseq))
        return fail("seam: h1 codes");
    for (int h = 0; h < 2; h++) {
        {
            std::vector<gl_t> z = chal_vec(tr, ldh + lseq);
            if (!seam_pair(pf.rRopeQ[2 * h], z, rY(MM_WQ), ins_pt(z, ldh, h)))
                return fail("seam: rope-q slice");
            std::vector<gl_t> z2 = chal_vec(tr, ldh + lseq);
            if (!seam_pair(pf.rRopeQ[2 * h + 1], z2, rY(MM_WK), ins_pt(z2, ldh, h)))
                return fail("seam: rope-k slice");
        }
        if (!seam_codes(pf.rCodes[QN_RQ[h]], pf.rMX[MM_QK[h]], ldh, lkQK, lseq))
            return fail("seam: qk x codes");
        if (!seam_codes(pf.rCodes[QN_RK[h]], pf.rMW[MM_QK[h]], ldh, lkQK, lseq))
            return fail("seam: qk w codes");
        {
            std::vector<gl_t> z = chal_vec(tr, lseq + ldh);
            std::vector<gl_t> zt(z.begin() + lseq, z.end());
            zt.push_back(h ? 1 : 0);
            zt.insert(zt.end(), z.begin(), z.begin() + lseq);
            if (!seam_pair(pf.rVtX[h], z, rY(MM_WV), zt))
                return fail("seam: v^T transpose");
        }
        if (!seam_codes(pf.rCodes[QN_PB[h]], pf.rMX[MM_PV[h]], lseq, lkPV, lseq))
            return fail("seam: pv x codes");
        if (!seam_codes(pf.rCodes[QN_VT[h]], pf.rMW[MM_PV[h]], lseq, lkPV, ldh))
            return fail("seam: pv w codes");
        {
            std::vector<gl_t> z = chal_vec(tr, ldh + lseq);
            if (!seam_pair(pf.rAtX, ins_pt(z, ldh, h), rY(MM_PV[h]), z))
                return fail("seam: concat half");
        }
    }
    if (!seam_codes(pf.rCodes[QN_AT], pf.rMX[MM_WO], ld, lkH1, lseq))
        return fail("seam: attn codes");
    if (!seam_codes(pf.rCodes[QN_H2], pf.rMX[MM_WG], ld, lkH1, lseq))
        return fail("seam: h2 codes");
    if (!seam_codes(pf.rCodes[QN_SW], pf.rMX[MM_WD], ldff, lkSW, lseq))
        return fail("seam: swiglu codes");
    {   // output binding: committed final residual == the PUBLIC out, bitwise
        std::vector<gl_t> z = chal_vec(tr, ld + lseq);
        std::vector<gl_t> op(outpub.begin(), outpub.end());
        if (vcl(pf.rOut, z) != mle_eval(op, z) || seam_short)
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
    size_t s = 20;
    for (int i = 0; i < 2; i++) s += p3rms::proof_size(pf.rms[i]);
    for (int i = 0; i < NQN; i++) s += p3qnt::proof_size(pf.qn[i]);
    for (int i = 0; i < NMM; i++) s += p3hwl::proof_size(pf.mm[i]);
    for (int i = 0; i < 4; i++) s += p3rope::proof_size(pf.rp[i]);
    for (int i = 0; i < 2; i++) s += p3smx::proof_size(pf.sm[i]);
    for (int i = 0; i < 2; i++) s += p3bfa::proof_size(pf.res[i]);
    s += p3swg::proof_size(pf.sw);
    s += 32 * (size_t)(1 + 2 + 2 * NQN + 2 * NMM + 8 + 2 + 2 + 1 + 1 + 1 + 1);
    for (int i = 0; i < NMM; i++) s += pf.mmY[i].size() * 2;
    s += pf.seam.size() * 8;
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3tf
