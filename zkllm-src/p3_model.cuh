// Composed FULL-MODEL prover+verifier (design doc section 18): proves an
// ENTIRE multi-layer forward pass of transformer_ref.py's TinyModel,
//
//   public token ids -> embedding lookup (committed table) ->
//     N chained transformer layers (p3_transformer.cuh, whole dataflow each) ->
//   final RMSNorm -> LM head Hawkeye matmul -> PUBLIC logits
//
// as ONE proof over ONE Fiat-Shamir transcript with ONE shared opening ledger,
// ONE model-level merged-lookup flush and ONE batched-opening pass:
//
//  * INTER-LAYER SEAM = ROOT EQUALITY.  Layer i's committed final-residual
//    column (res[1].OUT) IS layer i+1's committed input column (rms1.X): the
//    SAME hiding commitment object is handed to both layers' gadgets, and the
//    verifier checks lay[i+1].rX0 == lay[i].rOut.  No evaluation of any
//    intermediate activation is ever revealed -- in zk mode the column carries
//    its own mask slices and every gadget claim on it opens at masked points.
//  * EMBEDDING = GATHER SEAMS at PUBLIC token ids.  The embedding table E
//    (vocab x d) is committed (a model weight, secret values / public root);
//    for each token slot t the proof claims X0~(z_d, bits(t)) ==
//    E~(z_d, bits(id_t)) at a shared fresh ex-coordinate.  In zk, X0's mask
//    slice 1 is the SAME gather of E's mask slice 1, so the claim algebra
//    holds slice-by-slice while each opened value is uniform.
//  * LM HEAD = final rmsnorm (gain gF, X = last layer's OUT commitment) ->
//    quantize -> Hawkeye matmul against the committed head matrix -> the
//    committed logits column is bound to the PUBLIC logits by one real-slice
//    claim (clp) checked against the verifier's own MLE of the statement.
//
// Public statement: dims, token ids, embedding root, per-layer weight roots,
// gF root, head code+scale roots, pinned rope/canonical tables, Q/R -- and the
// claimed LOGITS.  Everything between the ids and the logits is hidden.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "p3_transformer.cuh"

namespace p3mdl {

using p3lu::Col; using p3lu::commit_col_nc; using p3lu::chal_vec; using p3lu::ilog2;
using p3hwl::claimc; using p3hwl::claimv; using p3hwl::now_ms;
using p3rms::Art;
using p3rope::mle_eval;
using p3tf::Config; using p3tf::Weights; using p3tf::WeightMat; using p3tf::WeightRoots;
using p3tf::TfWit; using p3tf::TfOps; using p3tf::TfProof; using p3tf::TfProf;
using p3tf::TfTables; using p3tf::commit_u8; using p3tf::restr_mask;
using p3tf::ins_zeros; using p3tf::pad_cube_pt;
using Hash = p3fri::Hash;

// ---------------- model weights (transformer_ref.py --dump-model-weights) ----------------
struct ModelWeights {
    uint32_t nlayers = 0, vocab = 0;
    Config cfg;                             // per-layer dims (batch = 1)
    std::vector<uint16_t> emb;              // vocab*d bf16 patterns
    std::vector<Weights> lw;                // per-layer weights (cos/sin replicated)
    std::vector<uint16_t> gF;               // final rmsnorm gain
    WeightMat wh;                           // LM head, vocab x d
};
static inline bool load_model_weights(const char* path, ModelWeights& M) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[8];
    if (fread(hdr, 8, 8, f) != 8 || hdr[0] != 0x54464D57) { fclose(f); return false; }
    M.nlayers = (uint32_t)hdr[1]; M.vocab = (uint32_t)hdr[2];
    M.cfg.seq = (uint32_t)hdr[3]; M.cfg.d = (uint32_t)hdr[4]; M.cfg.nh = (uint32_t)hdr[5];
    M.cfg.dh = (uint32_t)hdr[6]; M.cfg.dff = (uint32_t)hdr[7]; M.cfg.batch = 1;
    size_t ne = (size_t)M.vocab * M.cfg.d;
    M.emb.resize(ne);
    if (fread(M.emb.data(), 2, ne, f) != ne) { fclose(f); return false; }
    M.lw.resize(M.nlayers);
    for (uint32_t li = 0; li < M.nlayers; li++) {
        Weights& W = M.lw[li];
        W.cfg = M.cfg;
        for (int i = 0; i < p3tf::NW; i++) {
            int64_t nk[2];
            if (fread(nk, 8, 2, f) != 2) { fclose(f); return false; }
            W.w[i].N = (uint32_t)nk[0]; W.w[i].K = (uint32_t)nk[1];
            size_t n = (size_t)W.w[i].N * W.w[i].K;
            W.w[i].codes.resize(n); W.w[i].scales.resize(W.w[i].N);
            if (fread(W.w[i].codes.data(), 1, n, f) != n ||
                fread(W.w[i].scales.data(), 4, W.w[i].N, f) != W.w[i].N) { fclose(f); return false; }
        }
        W.g1.resize(M.cfg.d); W.g2.resize(M.cfg.d);
        if (fread(W.g1.data(), 2, M.cfg.d, f) != M.cfg.d ||
            fread(W.g2.data(), 2, M.cfg.d, f) != M.cfg.d) { fclose(f); return false; }
    }
    M.gF.resize(M.cfg.d);
    if (fread(M.gF.data(), 2, M.cfg.d, f) != M.cfg.d) { fclose(f); return false; }
    int64_t nk[2];
    if (fread(nk, 8, 2, f) != 2) { fclose(f); return false; }
    M.wh.N = (uint32_t)nk[0]; M.wh.K = (uint32_t)nk[1];
    size_t nh_ = (size_t)M.wh.N * M.wh.K;
    M.wh.codes.resize(nh_); M.wh.scales.resize(M.wh.N);
    if (fread(M.wh.codes.data(), 1, nh_, f) != nh_ ||
        fread(M.wh.scales.data(), 4, M.wh.N, f) != M.wh.N) { fclose(f); return false; }
    size_t nc = (size_t)M.cfg.seq * (M.cfg.dh / 2);
    std::vector<uint16_t> cos(nc), sin(nc);
    bool ok = fread(cos.data(), 2, nc, f) == nc && fread(sin.data(), 2, nc, f) == nc;
    fclose(f);
    for (uint32_t li = 0; li < M.nlayers; li++) { M.lw[li].cos = cos; M.lw[li].sin = sin; }
    return ok;
}

// ---------------- golden model trace (--dump-model-trace) ----------------
struct ModelTrace {
    uint32_t seq = 0, vocab = 0;
    std::vector<uint32_t> ids;
    std::vector<std::string> names;
    std::vector<std::vector<uint16_t>> vals;
};
static inline bool load_model_trace(const char* path, ModelTrace& T) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t hdr[4];
    if (fread(hdr, 8, 4, f) != 4 || hdr[0] != 0x54464D54) { fclose(f); return false; }
    T.seq = (uint32_t)hdr[2]; T.vocab = (uint32_t)hdr[3];
    std::vector<int64_t> ids64(T.seq);
    if (fread(ids64.data(), 8, T.seq, f) != T.seq) { fclose(f); return false; }
    T.ids.assign(ids64.begin(), ids64.end());
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
static inline const std::vector<uint16_t>* mtrace_get(const ModelTrace& T, const std::string& nm) {
    for (size_t i = 0; i < T.names.size(); i++)
        if (T.names[i] == nm) return &T.vals[i];
    return nullptr;
}

// ---------------- model tamper vocabulary ----------------
// MDT_L0+t / MDT_L1+t forward the p3tf tamper t into that layer only.
enum { MDT_NONE = 0,
       MDT_X0,           // embedded input != E[ids] (embedding gather seam)
       MDT_CHAIN,        // layer-1 input != layer-0 output (root-equality chain)
       MDT_HF,           // final rmsnorm output flip (rms zero-check)
       MDT_HEAD_CODES,   // head matmul X code != final quant CODES (restriction seam)
       MDT_HEAD_MM,      // head matmul accumulator teleport (chain sumcheck)
       MDT_LOGITS_W,     // prover's public logits claim flipped (Y binding)
       MDT_L0 = 100, MDT_L1 = 200 };

// ---------------- chained model witness ----------------
struct MdlWit {
    Config cfg;
    uint32_t nlayers = 0, vocab = 0;
    std::vector<uint32_t> ids;
    std::vector<uint16_t> x0;                  // embedded input (T x d patterns)
    std::vector<TfWit> lay;
    p3rms::Wit rmsF;
    p3qnt::Wit qnF;
    p3hwl::LayerWit mmH;
    std::vector<uint16_t> hFy, logits;         // honest head values
    std::vector<uint16_t> logitsPub;           // the PUBLIC logits claim
};

static inline MdlWit build_model_witness(const ModelWeights& MW,
                                         const std::vector<uint32_t>& ids,
                                         const Art& a, int tamper = MDT_NONE) {
    const Config& c = MW.cfg;
    if (!c.pow2()) throw std::runtime_error("mdl: config must be pow2");
    if ((1u << ilog2(MW.vocab)) != MW.vocab) throw std::runtime_error("mdl: vocab must be pow2");
    if (ids.size() != c.T()) throw std::runtime_error("mdl: ids size");
    MdlWit w; w.cfg = c; w.nlayers = MW.nlayers; w.vocab = MW.vocab; w.ids = ids;
    const uint32_t T = c.T(), d = c.d, V = MW.vocab;

    // -- embedding gather (the op IS the gather; proven by the gather seams) --
    w.x0.assign((size_t)T * d, 0);
    for (uint32_t t = 0; t < T; t++) {
        if (ids[t] >= V) throw std::runtime_error("mdl: token id out of range");
        memcpy(&w.x0[(size_t)t * d], &MW.emb[(size_t)ids[t] * d], 2 * d);
    }
    if (tamper == MDT_X0) w.x0[5] ^= 1;

    // -- N chained layers: each layer's canonical replay output feeds the next --
    w.lay.resize(MW.nlayers);
    std::vector<uint16_t> xi = w.x0;
    for (uint32_t li = 0; li < MW.nlayers; li++) {
        int tft = p3tf::TFT_NONE;
        if (tamper >= MDT_L0 && tamper < MDT_L0 + 100 && li == 0) tft = tamper - MDT_L0;
        if (tamper >= MDT_L1 && tamper < MDT_L1 + 100 && li == 1) tft = tamper - MDT_L1;
        if (tamper == MDT_CHAIN && li == 1) xi[3] ^= 1;   // break the hand-off
        w.lay[li] = p3tf::build_witness(c, xi, MW.lw[li], a, tft);
        xi = w.lay[li].outp;
    }

    // -- LM head: final rmsnorm -> quantize -> Hawkeye matmul -> logits --
    {
        p3rms::Golden g; g.B = T; g.d = d; g.x = xi; g.g = MW.gF;
        w.rmsF = p3rms::gen_witness(g, a);
    }
    if (tamper == MDT_HF) {
        w.rmsF.Y[5] ^= 1;
        w.rmsF.ypat[5] = gl_add(w.rmsF.ypat[5], 1ULL);
    }
    w.hFy = w.rmsF.Y;
    {
        p3qnt::Golden g; g.B = T; g.d = d; g.x = w.hFy;
        w.qnF = p3qnt::gen_witness(g, a);
    }
    {
        std::vector<uint8_t> cx = w.qnF.C;
        if (tamper == MDT_HEAD_CODES) cx[7] ^= 1;
        p3hwl::Golden L; L.B = T; L.K = d; L.N = V;
        L.x = cx; L.xs = w.qnF.S; L.w = MW.wh.codes; L.ws = MW.wh.scales;
        L.y.assign((size_t)T * V, 0);
        p3hwl::Tamper hm{p3hwl::TM_STATE, 1, 1, 0};
        w.mmH = p3hwl::gen_witness(L, true, tamper == MDT_HEAD_MM ? &hm : nullptr);
    }
    w.logits = w.mmH.Y;
    w.logitsPub = w.logits;
    if (tamper == MDT_LOGITS_W) w.logitsPub[3] ^= 1;
    return w;
}

// ---------------- operand commitments (the model chain, committed once) ----------------
struct MdlOps {
    Col E, X0;                                 // embedding table + embedded input
    std::vector<TfOps> lay;
    p3rms::Operands rmsF;
    p3qnt::Operands qnF;
    p3hwl::Operands mmH;
    Col YH;                                    // committed logits column [v | tok]
};
static inline MdlOps commit_model(const MdlWit& w, const ModelWeights& MW, uint32_t R) {
    MdlOps o;
    const bool zk = p3zkc::G.on;
    const Config& c = w.cfg;
    const uint32_t ld = c.ld(), lT = c.lT(), T = c.T(), d = c.d;
    // embedding table (secret values, public root)
    std::vector<gl_t> ep(MW.emb.begin(), MW.emb.end());
    o.E = commit_col_nc(std::move(ep), R);
    // embedded input, mask slice 1 = the SAME gather of E's mask slice 1
    std::vector<gl_t> x0m;
    if (zk) {
        std::vector<gl_t> e1 = p3tf::psl1(o.E);
        std::vector<gl_t> cons1((size_t)1 << (ld + lT), 0);
        for (uint32_t t = 0; t < T; t++)
            for (uint32_t j = 0; j < d; j++)
                cons1[j | ((size_t)t << ld)] = e1[j | ((size_t)w.ids[t] << ld)];
        x0m = p3zkc::mk_linked(ld + lT, cons1);
    }
    std::vector<gl_t> xp(w.x0.begin(), w.x0.end());
    o.X0 = commit_col_nc(std::move(xp), R, (zk && !p3zkc::G.nolink) ? &x0m : nullptr);
    // chained layers: layer i+1's input col IS layer i's output col (except a
    // deliberately-broken chain tamper, where the prover commits what it used)
    o.lay.resize(w.nlayers);
    for (uint32_t li = 0; li < w.nlayers; li++) {
        const Col* xin = li == 0 ? &o.X0 : &o.lay[li - 1].res[1].OUT;
        Col fresh;
        if (li > 0 && w.lay[li].x0 != w.lay[li - 1].outp) {
            std::vector<gl_t> fx(w.lay[li].x0.begin(), w.lay[li].x0.end());
            fresh = commit_col_nc(std::move(fx), R);
            xin = &fresh;
        }
        o.lay[li] = p3tf::commit_all(w.lay[li], R, xin);
    }
    // head: rmsF.X = last layer's OUT commitment (shared root)
    o.rmsF.X = o.lay[w.nlayers - 1].res[1].OUT;
    std::vector<gl_t> gp(w.rmsF.gpat), yp(w.rmsF.ypat);
    o.rmsF.G = commit_col_nc(std::move(gp), R);
    o.rmsF.Y = commit_col_nc(std::move(yp), R);
    o.qnF.X = o.rmsF.Y;
    o.qnF.CODES = commit_col_nc(std::vector<gl_t>(w.qnF.cpat), R);
    o.qnF.SCALES = commit_col_nc(std::vector<gl_t>(w.qnF.spat), R);
    {
        const uint32_t lkH = ilog2(w.mmH.d.Kpad);
        std::vector<gl_t> xm = restr_mask(o.qnF.CODES, ld, lkH, lT);
        o.mmH.X = commit_u8(w.mmH.xcodes, R, (zk && !p3zkc::G.nolink) ? &xm : nullptr);
    }
    o.mmH.W = commit_u8(w.mmH.wcodes, R);      // static head weight (secret leaf)
    o.mmH.XS = o.qnF.SCALES;
    o.mmH.WS = commit_col_nc(std::vector<gl_t>(w.mmH.wsb), R);
    o.YH = commit_col_nc(std::vector<gl_t>(w.mmH.dob[p3hwl::O_YB]), R);
    return o;
}

// ---------------- the model's pinned statement roots ----------------
struct MdlRoots {
    Hash E, GF, WHc, WHs;
    std::vector<WeightRoots> lw;
};
// non-zk: recomputed independently from the weights file (deterministic
// commitments); zk: pinned from the prover's masked commitments (the published
// model commitment), like zk_layer_smoke's roots_from_ops.
static inline MdlRoots model_roots(const ModelWeights& MW, uint32_t R) {
    MdlRoots mr;
    std::vector<gl_t> e(MW.emb.begin(), MW.emb.end());
    mr.E = commit_col_nc(std::move(e), R).root;
    std::vector<gl_t> gf(MW.gF.begin(), MW.gF.end());
    mr.GF = commit_col_nc(std::move(gf), R).root;
    p3hwl::Dims dd = p3hwl::make_dims(MW.cfg.T(), MW.wh.K, MW.wh.N);
    std::vector<gl_t> wc((size_t)dd.Npad * dd.Kpad, 0), ws(dd.Npad, 0);
    for (uint32_t n = 0; n < MW.wh.N; n++)
        for (uint32_t k = 0; k < MW.wh.K; k++)
            wc[(size_t)n * dd.Kpad + k] = MW.wh.codes[(size_t)n * MW.wh.K + k];
    for (uint32_t n = 0; n < MW.wh.N; n++) ws[n] = MW.wh.scales[n];
    mr.WHc = commit_col_nc(std::move(wc), R).root;
    mr.WHs = commit_col_nc(std::move(ws), R).root;
    for (uint32_t li = 0; li < MW.nlayers; li++)
        mr.lw.push_back(p3tf::weight_roots(MW.lw[li], R, MW.cfg.batch));
    return mr;
}
static inline MdlRoots roots_from_ops(const MdlOps& o, const Config& c, uint32_t nlayers) {
    MdlRoots mr;
    mr.E = o.E.root; mr.GF = o.rmsF.G.root;
    mr.WHc = o.mmH.W.root; mr.WHs = o.mmH.WS.root;
    const int MMW[p3tf::NW] = {p3tf::MM_WQ, p3tf::MM_WK, p3tf::MM_WV,
                               c.mmWO(), c.mmWG(), c.mmWU(), c.mmWD()};
    mr.lw.resize(nlayers);
    for (uint32_t li = 0; li < nlayers; li++) {
        for (int i = 0; i < p3tf::NW; i++) {
            mr.lw[li].Wc[i] = o.lay[li].mm[MMW[i]].W.root;
            mr.lw[li].Ws[i] = o.lay[li].mm[MMW[i]].WS.root;
        }
        mr.lw[li].G1 = o.lay[li].rms[0].G.root;
        mr.lw[li].G2 = o.lay[li].rms[1].G.root;
    }
    return mr;
}

// ---------------- proof object ----------------
struct MdlProof {
    uint32_t nlayers = 0, vocab = 0, seq = 0, d = 0, batch = 1;
    std::vector<TfProof> lay;
    Hash rEmb;
    // head sub-proofs + chained roots
    p3rms::RmsProof rmsF;
    p3qnt::QntProof qnF;
    p3hwl::LayerProof mmH;
    Hash rRmsFY, rCodesF, rScalesF, rMXH, rMWH;
    std::vector<uint16_t> mmYH;                // public logits claim (zk: empty)
    // model-level seam claims (embedding gathers, head codes, logits binding)
    std::vector<gl_t> seam;
    std::vector<p3lu::GroupProof> lug;
    std::vector<p3bo::BatchProof> batches;
};

// ==================== prover ====================
static inline MdlProof prove_model(fs::Transcript& tr, const MdlWit& w, const MdlOps& o,
                                   const TfTables& T, const Art& a, uint32_t R, uint32_t Q,
                                   bool strict = true, TfProf* prof = nullptr) {
    const Config& c = w.cfg;
    const bool zk = p3zkc::G.on;
    MdlProof pf;
    pf.nlayers = w.nlayers; pf.vocab = w.vocab; pf.seq = c.seq; pf.d = c.d; pf.batch = c.batch;
    pf.lay.resize(w.nlayers);
    TfProf pl; TfProf& P = prof ? *prof : pl;
    double tall = now_ms(), tp;

    uint32_t hdr[8] = {w.nlayers, w.vocab, c.seq, c.d, c.nh, c.dh, c.dff, c.batch};
    tr.absorb("mdl-dims", hdr, sizeof hdr);
    tr.absorb("mdl-ids", w.ids.data(), 4 * w.ids.size());
    tr.absorb("mdl-emb", o.E.root.data(), 32);
    pf.rEmb = o.E.root;

    p3lu::XCtx xc;

    // -- every layer's sub-proofs, then the head's, one transcript, one ledger --
    for (uint32_t li = 0; li < w.nlayers; li++)
        p3tf::prove_subs(tr, w.lay[li], o.lay[li], T, a, R, Q, strict, xc, pf.lay[li], P);
    tp = now_ms();
    pf.rmsF = p3rms::prove(tr, w.rmsF, T.rms, a, o.rmsF, R, Q, strict, &xc);
    P.rms += now_ms() - tp; tp = now_ms();
    pf.qnF = p3qnt::prove(tr, w.qnF, T.qnt, o.qnF, R, Q, strict, &xc);
    P.qnt += now_ms() - tp; tp = now_ms();
    pf.mmH = p3hwl::prove(tr, w.mmH, T.hw, o.mmH, R, Q, true, strict,
                          nullptr, &w.logitsPub, &xc, zk ? &o.YH : nullptr);
    P.mm += now_ms() - tp;
    pf.rRmsFY = o.rmsF.Y.root;
    pf.rCodesF = o.qnF.CODES.root; pf.rScalesF = o.qnF.SCALES.root;
    pf.rMXH = o.mmH.X.root; pf.rMWH = o.mmH.W.root;
    pf.mmYH = zk ? std::vector<uint16_t>() : w.logitsPub;

    // -- ONE merged-lookup flush for the WHOLE MODEL --
    tp = now_ms();
    pf.lug = p3lu::lu_flush(tr, xc, R, Q, strict);
    P.lug += now_ms() - tp;

    // -- seams: per-layer (no public IO bindings), then the model's own --
    tp = now_ms();
    for (uint32_t li = 0; li < w.nlayers; li++)
        p3tf::prove_seams(tr, w.lay[li], o.lay[li], strict, xc, pf.lay[li],
                          /*bind_in=*/false, /*bind_out=*/false);
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
            throw std::runtime_error(std::string("mdl: seam mismatch: ") + what);
    };
    const uint32_t ld = c.ld(), lT = c.lT(), lv = ilog2(w.vocab);
    const uint32_t lkH = ilog2(w.mmH.d.Kpad);
    // embedding gather seams: X0 token slot t == E row id_t
    for (uint32_t t = 0; t < c.T(); t++) {
        std::vector<gl_t> z = chal_vec(tr, ld);
        std::vector<gl_t> zx = z, ze = z;
        for (uint32_t i = 0; i < lT; i++) zx.push_back((t >> i) & 1);
        for (uint32_t i = 0; i < lv; i++) ze.push_back((w.ids[t] >> i) & 1);
        seam_pair(o.X0, zx, o.E, ze, "embedding gather");
    }
    // head codes restriction + canonical-zero padding
    {
        std::vector<gl_t> z = chal_vec(tr, ld + lT);
        seam_pair(o.qnF.CODES, z, o.mmH.X, ins_zeros(z, ld, lkH - ld), "head codes");
        for (uint32_t t2 = ld; t2 < lkH; t2++) {
            std::vector<gl_t> zt = chal_vec(tr, t2 + lT);
            gl_t y = clp(o.mmH.X, pad_cube_pt(zt, t2, lkH));
            if (strict && y != 0)
                throw std::runtime_error("mdl: head padding not zero");
        }
    }
    // public logits binding: committed head output == the PUBLIC logits
    {
        std::vector<gl_t> z = chal_vec(tr, lv + lT);
        clp(o.YH, z);
    }
    P.seam += now_ms() - tp;

    // -- ONE batched opening pass per size class for the WHOLE MODEL --
    tp = now_ms();
    for (size_t i = 0; i < xc.lg.cls.size(); i++)
        pf.batches.push_back(p3bo::prove_class(tr, xc.lg.cls[i], R, Q,
                                               "tf-bo" + std::to_string(i)));
    P.batch += now_ms() - tp;
    P.total += now_ms() - tall;
    return pf;
}

// ==================== verifier ====================
// PUBLIC statement, all pinned by the CALLER: dims + nlayers + vocab, the
// token ids, the embedding root, per-layer weight roots, gF root, head roots,
// the pinned rope/canonical tables, Q/R -- and the claimed LOGITS.  Every
// activation between the ids and the logits stays hidden: layers chain by
// commitment-root equality and the only public value claims are the logits.
static inline bool verify_model(fs::Transcript& tr, const MdlProof& pf, const TfTables& T,
                                const Art& a, const Config& c, uint32_t nlayers, uint32_t vocab,
                                const std::vector<uint32_t>& ids, const MdlRoots& mr,
                                const std::vector<uint16_t>& cosp, const std::vector<uint16_t>& sinp,
                                const std::vector<uint16_t>& logitspub,
                                uint32_t Q_pub, uint32_t R_pub, const char** why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (Q_pub < 20 || R_pub < 1) return fail("insecure params");
    if (!c.pow2() || (1u << ilog2(vocab)) != vocab) return fail("dims must be pow2");
    if (nlayers < 1 || pf.nlayers != nlayers || pf.vocab != vocab
        || pf.seq != c.seq || pf.d != c.d || pf.batch != c.batch
        || pf.lay.size() != nlayers || mr.lw.size() != nlayers)
        return fail("model shape");
    if (ids.size() != c.T()) return fail("ids size");
    for (uint32_t t = 0; t < c.T(); t++) if (ids[t] >= vocab) return fail("token id range");
    if (logitspub.size() != (size_t)c.T() * vocab) return fail("logits size");
    const bool zk = p3zkc::G.on;
    if (!zk && pf.mmYH != logitspub) return fail("public logits claim mismatch");
    if (!(pf.rEmb == mr.E)) return fail("embedding root mismatch");
    const uint32_t ld = c.ld(), lT = c.lT(), lv = ilog2(vocab);

    uint32_t hdr[8] = {nlayers, vocab, c.seq, c.d, c.nh, c.dh, c.dff, c.batch};
    tr.absorb("mdl-dims", hdr, sizeof hdr);
    tr.absorb("mdl-ids", ids.data(), 4 * ids.size());
    tr.absorb("mdl-emb", mr.E.data(), 32);

    p3lu::VCtx vctx;
    p3bo::VLedger& vlg = vctx.vlg;

    // -- layer sub-verifies: INTER-LAYER SEAM = ROOT EQUALITY --
    for (uint32_t li = 0; li < nlayers; li++) {
        const Hash& rin = li == 0 ? pf.lay[0].rX0 : pf.lay[li - 1].rOut;
        if (li > 0 && !(pf.lay[li].rX0 == rin)) return fail("inter-layer chain root");
        if (!p3tf::verify_subs(tr, pf.lay[li], T, a, c, rin, mr.lw[li], cosp, sinp,
                               Q_pub, R_pub, why, vctx)) return false;
    }
    // -- head sub-verifies (rmsF.X pinned to the LAST layer's output root) --
    if (!p3rms::verify(tr, T.rms, a, pf.rmsF, pf.lay[nlayers - 1].rOut, mr.GF, pf.rRmsFY,
                       c.T(), ld, Q_pub, R_pub, why, &vctx)) return false;
    if (!p3qnt::verify(tr, T.qnt, pf.qnF, pf.rRmsFY, pf.rCodesF, pf.rScalesF,
                       c.T(), ld, Q_pub, R_pub, why, &vctx)) return false;
    if (!(pf.rMWH == mr.WHc)) return fail("head weight root mismatch");
    if (!zk && pf.mmYH.size() != (size_t)c.T() * vocab) return fail("mmYH size");
    if (!p3hwl::verify(tr, T.hw, pf.mmH, pf.rMXH, mr.WHc, pf.rScalesF, mr.WHs, pf.mmYH,
                       c.T(), c.d, vocab, Q_pub, R_pub, why, &vctx)) return false;

    // -- merged-lookup flush (whole model) --
    if (!p3lu::lu_verify_flush(tr, vctx, pf.lug, Q_pub, R_pub, why)) return false;

    // -- seams: per-layer (no public IO bindings), then the model's own --
    for (uint32_t li = 0; li < nlayers; li++)
        if (!p3tf::verify_seams(tr, pf.lay[li], c, nullptr, nullptr, why, vctx))
            return false;
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
    const uint32_t lkH = ilog2(p3hwl::make_dims(c.T(), c.d, vocab).Kpad);
    for (uint32_t t = 0; t < c.T(); t++) {
        std::vector<gl_t> z = chal_vec(tr, ld);
        std::vector<gl_t> zx = z, ze = z;
        for (uint32_t i = 0; i < lT; i++) zx.push_back((t >> i) & 1);
        for (uint32_t i = 0; i < lv; i++) ze.push_back((ids[t] >> i) & 1);
        if (!seam_pair(pf.lay[0].rX0, zx, mr.E, ze))
            return fail("seam: embedding gather");
    }
    {
        std::vector<gl_t> z = chal_vec(tr, ld + lT);
        if (!seam_pair(pf.rCodesF, z, pf.rMXH, ins_zeros(z, ld, lkH - ld)))
            return fail("seam: head codes");
        for (uint32_t t2 = ld; t2 < lkH; t2++) {
            std::vector<gl_t> zt = chal_vec(tr, t2 + lT);
            if (vclp(pf.rMXH, pad_cube_pt(zt, t2, lkH)) != 0 || seam_short)
                return fail("seam: head padding");
        }
    }
    {   // public logits binding (real-slice claim vs the verifier's own MLE)
        std::vector<gl_t> z = chal_vec(tr, lv + lT);
        std::vector<gl_t> lp(logitspub.begin(), logitspub.end());
        if (vclp(pf.mmH.rdo[p3hwl::O_YB], z) != mle_eval(lp, z) || seam_short)
            return fail("public logits binding");
    }
    if (si != pf.seam.size()) return fail("model seam claim count");

    // -- the ONE shared batched opening per size class --
    if (pf.batches.size() != vlg.cls.size()) return fail("batch count");
    for (size_t i = 0; i < vlg.cls.size(); i++)
        if (!p3bo::verify_class(tr, vlg.cls[i], pf.batches[i], Q_pub, R_pub,
                                "tf-bo" + std::to_string(i), why)) return false;

    if (why) *why = "ok";
    return true;
}

// ---------------- proof size (serialized-payload bytes) ----------------
static inline size_t proof_size_model(const MdlProof& pf) {
    size_t s = 40 + 32;                        // dims + rEmb
    for (auto& l : pf.lay) {
        s += p3tf::proof_size(l);
        // per-layer lug/batches are empty (model-level); subtract nothing
    }
    s += p3rms::proof_size(pf.rmsF) + p3qnt::proof_size(pf.qnF) + p3hwl::proof_size(pf.mmH);
    s += 32 * 5;
    s += pf.mmYH.size() * 2;
    s += pf.seam.size() * 8;
    for (auto& g : pf.lug) s += p3lu::sz_group(g);
    for (auto& b : pf.batches) s += p3bo::sz_batch(b);
    return s;
}

} // namespace p3mdl
