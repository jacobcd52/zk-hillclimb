// Composed FULL-MODEL test suite (p3_model.cuh) -- teeth.
//
//   python3 transformer_ref.py --dump-tables        p3_rmsnorm_tables.bin
//   python3 transformer_ref.py --dump-model-weights transformer_model_weights.bin
//   python3 transformer_ref.py --dump-model-trace   transformer_model.bin
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_model_test.cu -o /root/p3_model_test
//   cd /root/zkllm && /root/p3_model_test
//
// HONEST bar: ONE proof for the ENTIRE forward pass (embedding -> N layers ->
// head) VERIFIES, and EVERY chained intermediate of EVERY layer AND the final
// logits are BITWISE equal to transformer_model.bin.  Teeth: each adversarial
// case breaks a DIFFERENT stage, seam, layer hand-off or statement item and
// must reject.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_rmsnorm.cuh"
#include "p3_bfadd.cuh"
#include "p3_rope.cuh"
#include "p3_quant.cuh"
#include "p3_softmax.cuh"
#include "p3_swiglu.cuh"
#include "p3_transformer.cuh"
#include "p3_model.cuh"
using namespace p3mdl;
using std::vector;

static int np_ = 0, nf_ = 0;
static void ck(const char* n, bool c, const char* why = nullptr) {
    printf("  [%s] %s%s%s%s\n", c ? "PASS" : "FAIL", n,
           why ? "  (why=" : "", why ? why : "", why ? ")" : "");
    fflush(stdout);
    if (c) np_++; else nf_++;
}
static const uint32_t R = 2, Q = 24;
static Art A;
static TfTables* TT;
static ModelWeights MW;
static MdlRoots MR;
static ModelTrace MT;

static bool eq16(const vector<uint16_t>& a, const vector<uint16_t>& b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), 2 * a.size()) == 0;
}

// bitwise comparison of EVERY chained intermediate of EVERY layer + the head
static bool model_bitwise(const MdlWit& w, std::string& bad) {
    auto chk = [&](const std::string& nm, const vector<uint16_t>& v) {
        const vector<uint16_t>* t = mtrace_get(MT, nm);
        if (!t || !eq16(*t, v)) { bad = nm; return false; }
        return true;
    };
    if (!chk("x0", w.x0)) return false;
    for (uint32_t li = 0; li < w.nlayers; li++) {
        const TfWit& L = w.lay[li];
        std::string p = "L" + std::to_string(li) + ".";
        if (!chk(p + "rms1", L.rms1y)) return false;
        if (!chk(p + "q", L.mmY[p3tf::MM_WQ]) || !chk(p + "k", L.mmY[p3tf::MM_WK])
            || !chk(p + "v", L.mmY[p3tf::MM_WV])) return false;
        for (int h = 0; h < 2; h++) {
            std::string hs = std::to_string(h);
            if (!chk(p + "ropeq" + hs, L.ropq[h])) return false;
            if (!chk(p + "ropek" + hs, L.ropk[h])) return false;
            if (!chk(p + "scores" + hs, L.mmY[p3tf::MM_QK[h]])) return false;
            if (!chk(p + "probs" + hs, L.probs[h])) return false;
            if (!chk(p + "attnout" + hs, L.mmY[p3tf::MM_PV[h]])) return false;
        }
        if (!chk(p + "oproj", L.mmY[p3tf::MM_WO])) return false;
        if (!chk(p + "resid1", L.res1o)) return false;
        if (!chk(p + "rms2", L.rms2y)) return false;
        if (!chk(p + "gate", L.mmY[p3tf::MM_WG]) || !chk(p + "up", L.mmY[p3tf::MM_WU])) return false;
        if (!chk(p + "swiglu", L.swm)) return false;
        if (!chk(p + "down", L.mmY[p3tf::MM_WD])) return false;
        if (!chk(p + "out", L.outp)) return false;
    }
    if (!chk("hF", w.hFy)) return false;
    if (!chk("logits", w.logits)) return false;
    return true;
}

// build + prove + verify one tampered model; expect REJECT
static bool tamper_rejects(int mode, const char** why) {
    *why = nullptr;
    MdlWit w;
    try {
        w = build_model_witness(MW, MT.ids, A, mode);
    } catch (const std::exception& e) {
        *why = "witness builder threw (out of domain)";
        return true;
    }
    MdlOps o = commit_model(w, MW, R);
    MdlProof pf;
    try {
        fs::Transcript tp("tf-model");
        pf = prove_model(tp, w, o, *TT, A, R, Q, /*strict=*/false);
    } catch (const std::exception& e) {
        *why = "prover threw (out of supported domain)";
        return true;
    }
    fs::Transcript tv("tf-model");
    bool ok = verify_model(tv, pf, *TT, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, MR,
                           MW.lw[0].cos, MW.lw[0].sin, w.logitsPub, Q, R, why);
    return !ok;
}

int main() {
    printf("=== COMPOSED FULL-MODEL selftest (multi-layer forward pass, canonical = transformer_ref.py TinyModel) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    if (!p3rms::load_art("p3_rmsnorm_tables.bin", A)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-tables p3_rmsnorm_tables.bin\n");
        return 1;
    }
    if (!load_model_weights("transformer_model_weights.bin", MW)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-model-weights transformer_model_weights.bin\n");
        return 1;
    }
    if (!load_model_trace("transformer_model.bin", MT)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-model-trace transformer_model.bin\n");
        return 1;
    }
    printf("config: nlayers=%u vocab=%u seq=%u d=%u nh=%u dh=%u dff=%u; ids=[",
           MW.nlayers, MW.vocab, MW.cfg.seq, MW.cfg.d, MW.cfg.nh, MW.cfg.dh, MW.cfg.dff);
    for (auto t : MT.ids) printf("%u ", t);
    printf("]; trace arrays=%zu\n", MT.names.size());

    TfTables T = p3tf::build_tables(A); TT = &T;
    MR = model_roots(MW, R);

    // ================= HONEST ACCEPT =================
    printf("-- honest accept (ONE proof for the ENTIRE multi-layer forward pass) --\n");
    MdlProof hpf;
    vector<uint16_t> hlogits;
    {
        double t0 = p3hwl::now_ms();
        MdlWit w = build_model_witness(MW, MT.ids, A);
        double t1 = p3hwl::now_ms();
        std::string bad;
        bool bit = model_bitwise(w, bad);
        ck("EVERY chained intermediate of EVERY layer bitwise == transformer_model.bin",
           bit, bit ? nullptr : bad.c_str());
        MdlOps o = commit_model(w, MW, R);
        double t2 = p3hwl::now_ms();
        // inter-layer sharing is REAL: the same commitment object
        bool chain = true;
        for (uint32_t li = 1; li < MW.nlayers; li++)
            chain = chain && (o.lay[li].rms[0].X.root == o.lay[li - 1].res[1].OUT.root);
        chain = chain && (o.rmsF.X.root == o.lay[MW.nlayers - 1].res[1].OUT.root);
        ck("layer i+1 input commitment IS layer i output commitment (+ head)", chain);
        TfProf prof;
        fs::Transcript tp("tf-model");
        hpf = prove_model(tp, w, o, T, A, R, Q, /*strict=*/true, &prof);
        double t3 = p3hwl::now_ms();
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, hpf, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, MR,
                               MW.lw[0].cos, MW.lw[0].sin, w.logitsPub, Q, R, &why);
        double t4 = p3hwl::now_ms();
        hlogits = w.logitsPub;
        const vector<uint16_t>* glog = mtrace_get(MT, "logits");
        ck("composed model proof VERIFIES and accepted logits == golden bitwise",
           ok && glog && eq16(*glog, w.logits), ok ? nullptr : why);
        size_t psz = proof_size_model(hpf);
        printf("  timings: witness %.2f s, commits %.2f s, prove %.2f s, verify %.2f s\n",
               (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3);
        printf("  proof %.2f MB (%zu model seams, %zu batch classes)\n",
               psz / 1048576.0, hpf.seam.size(), hpf.batches.size());
    }

    // ============ MUST-REJECT: model-level seams and stages ============
    printf("-- must-reject: embedding / chain / head tampers --\n");
    struct Case { int mode; const char* name; };
    const Case mcases[] = {
        {MDT_X0,         "embedded input != E[ids] (embedding gather seam)"},
        {MDT_CHAIN,      "layer-1 input != layer-0 output (inter-layer chain root)"},
        {MDT_HF,         "final rmsnorm output flipped (rms zero-check)"},
        {MDT_HEAD_CODES, "head matmul X != final quant CODES (restriction seam)"},
        {MDT_HEAD_MM,    "head matmul accumulator teleport (chain sumcheck)"},
        {MDT_LOGITS_W,   "prover's public logits claim flipped (Y binding)"},
    };
    for (auto& cse : mcases) {
        bool rej = tamper_rejects(cse.mode, &why);
        ck(cse.name, rej, why);
    }

    // ============ MUST-REJECT: per-layer tampers inside the composed model ============
    printf("-- must-reject: in-layer tampers (owning layer catches, either layer) --\n");
    const Case lcases[] = {
        {MDT_L0 + p3tf::TFT_RMS1_Y,      "L0 rms1 output flipped"},
        {MDT_L0 + p3tf::TFT_SEAM_MMX,    "L0 matmul X != quant codes (restriction seam)"},
        {MDT_L0 + p3tf::TFT_GW_MM,       "L0 matmul state teleport"},
        {MDT_L0 + p3tf::TFT_SEAM_ROPEQ,  "L0 rope operand != Wq head slice"},
        {MDT_L1 + p3tf::TFT_SMX_P,       "L1 softmax prob flipped"},
        {MDT_L1 + p3tf::TFT_SEAM_CONCAT, "L1 concat half != attnout1"},
        {MDT_L1 + p3tf::TFT_GW_SWG,      "L1 swiglu silu-table forge"},
        {MDT_L1 + p3tf::TFT_RES1_OUT,    "L1 residual-1 sum flipped"},
    };
    for (auto& cse : lcases) {
        bool rej = tamper_rejects(cse.mode, &why);
        ck(cse.name, rej, why);
    }

    // ============ MUST-REJECT: statement / proof-object tampers ============
    printf("-- must-reject: statement and proof-object tampers --\n");
    {
        vector<uint16_t> bad = hlogits; bad[5] ^= 1;
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, hpf, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, MR,
                               MW.lw[0].cos, MW.lw[0].sin, bad, Q, R, &why);
        ck("flipped PUBLIC logits reject at the logits binding", !ok, why);
    }
    {
        vector<uint32_t> bad = MT.ids; bad[2] = (bad[2] + 1) % MW.vocab;
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, hpf, T, A, MW.cfg, MW.nlayers, MW.vocab, bad, MR,
                               MW.lw[0].cos, MW.lw[0].sin, hlogits, Q, R, &why);
        ck("wrong PUBLIC token ids reject (embedding gather)", !ok, why);
    }
    {
        MdlRoots bad = MR; bad.E.data()[0] ^= 1;
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, hpf, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, bad,
                               MW.lw[0].cos, MW.lw[0].sin, hlogits, Q, R, &why);
        ck("wrong pinned embedding root rejects", !ok, why);
    }
    {
        MdlRoots bad = MR; bad.lw[1].Wc[p3tf::W_G].data()[3] ^= 1;
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, hpf, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, bad,
                               MW.lw[0].cos, MW.lw[0].sin, hlogits, Q, R, &why);
        ck("wrong pinned LAYER-1 weight root rejects", !ok, why);
    }
    {
        MdlRoots bad = MR; bad.WHc.data()[7] ^= 1;
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, hpf, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, bad,
                               MW.lw[0].cos, MW.lw[0].sin, hlogits, Q, R, &why);
        ck("wrong pinned head weight root rejects", !ok, why);
    }
    {
        auto p2 = hpf; p2.lay[1].rX0.data()[5] ^= 1;
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, p2, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, MR,
                               MW.lw[0].cos, MW.lw[0].sin, hlogits, Q, R, &why);
        ck("tampered inter-layer chain root rejects", !ok, why);
    }
    {
        auto p2 = hpf; p2.seam[1] = gl_add(p2.seam[1], 1ULL);   // an embedding gather claim
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, p2, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, MR,
                               MW.lw[0].cos, MW.lw[0].sin, hlogits, Q, R, &why);
        ck("tampered embedding seam claim rejects", !ok, why);
    }
    {
        auto p2 = hpf; p2.batches[0].ystar[0] = gl_add(p2.batches[0].ystar[0], 1ULL);
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, p2, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, MR,
                               MW.lw[0].cos, MW.lw[0].sin, hlogits, Q, R, &why);
        ck("tampered shared batch opening rejects", !ok, why);
    }
    {
        auto p2 = hpf; p2.lay[0].sm[1].mDe[0].s2 = gl_add(p2.lay[0].sm[1].mDe[0].s2, 1ULL);
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, p2, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, MR,
                               MW.lw[0].cos, MW.lw[0].sin, hlogits, Q, R, &why);
        ck("tampered in-layer sub-proof message rejects", !ok, why);
    }

    printf("\nCOMPOSED-MODEL: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
