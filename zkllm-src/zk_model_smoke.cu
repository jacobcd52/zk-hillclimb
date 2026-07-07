// zk_model_smoke.cu -- composed FULL-MODEL prover with the zk path ON:
// honest-accept + soundness tampers must still reject, no cleartext activation
// (of ANY layer or the inter-layer hand-offs) ships in the proof, and only the
// token ids + logits are publicly bound.  (The hiding-law battery is
// p3_model_zk_test.cu; this is the soundness+wiring smoke.)
//   nvcc -arch=sm_89 -std=c++17 -O2 zk_model_smoke.cu -o /root/zk_model_smoke
#include <cstdio>
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

int main() {
    printf("=== zk composed FULL-MODEL smoke (G.on, multi-layer forward pass) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    p3rms::Art A;
    if (!p3rms::load_art("p3_rmsnorm_tables.bin", A)) { printf("need tables\n"); return 1; }
    ModelWeights MW; if (!load_model_weights("transformer_model_weights.bin", MW)) { printf("need model weights\n"); return 1; }
    ModelTrace MT; if (!load_model_trace("transformer_model.bin", MT)) { printf("need model trace\n"); return 1; }
    TfTables T = p3tf::build_tables(A);

    p3zkc::G.on = true; p3zkc::G.Q = Q;

    // ---- honest accept ----
    MdlProof hpf; vector<uint16_t> hlogits; MdlRoots MR;
    {
        p3zkc::G.seed = 20260707; p3zkc::G.ctr = 0;
        double t0 = p3hwl::now_ms();
        MdlWit w = build_model_witness(MW, MT.ids, A);
        MdlOps o = commit_model(w, MW, R);
        MR = roots_from_ops(o, MW.cfg, MW.nlayers);
        double t1 = p3hwl::now_ms();
        TfProf prof; fs::Transcript tp("tf-model");
        hpf = prove_model(tp, w, o, T, A, R, Q, true, &prof);
        double t2 = p3hwl::now_ms();
        const char* why = nullptr;
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, hpf, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, MR,
                               MW.lw[0].cos, MW.lw[0].sin, w.logitsPub, Q, R, &why);
        double t3 = p3hwl::now_ms();
        ck("zk composed MODEL proof VERIFIES (honest)", ok, ok ? nullptr : why);
        hlogits = w.logitsPub;
        const vector<uint16_t>* glog = mtrace_get(MT, "logits");
        ck("accepted PUBLIC logits == golden bitwise",
           glog && hlogits == *glog);
        size_t psz = proof_size_model(hpf);
        printf("  commit %.2f s, prove %.2f s, verify %.2f s, proof %.2f MB\n",
               (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, psz / 1048576.0);
        // NO cleartext activation ships: every per-matmul public vector of
        // every layer is dropped, the head's public logits vector is dropped
        // (bound by the real-slice claim instead), and neither x0 nor any
        // layer output appears as a public value anywhere in the proof
        bool dropped = hpf.mmYH.empty();
        for (auto& l : hpf.lay)
            for (auto& y : l.mmY) dropped = dropped && y.empty();
        ck("zk: ALL cleartext activation vectors dropped (every layer + head)", dropped);
    }

    // ---- soundness tampers (must reject) ----
    auto tamper = [&](int mode, const char* nm) {
        const char* why = nullptr;
        MdlWit w;
        try { w = build_model_witness(MW, MT.ids, A, mode); }
        catch (const std::exception&) { ck(nm, true, "witness threw (out of domain)"); return; }
        MdlOps o = commit_model(w, MW, R);
        MdlRoots mr = roots_from_ops(o, MW.cfg, MW.nlayers);
        MdlProof pf;
        try { fs::Transcript tp("tf-model"); pf = prove_model(tp, w, o, T, A, R, Q, false); }
        catch (const std::exception&) { ck(nm, true, "prover threw"); return; }
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, pf, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, mr,
                               MW.lw[0].cos, MW.lw[0].sin, w.logitsPub, Q, R, &why);
        ck(nm, !ok, why);
    };
    printf("-- soundness (zk on): tampers must reject --\n");
    tamper(MDT_X0,          "embedded input != E[ids] (embedding gather seam)");
    tamper(MDT_CHAIN,       "layer-1 input != layer-0 output (chain root)");
    tamper(MDT_HF,          "final rmsnorm output flip");
    tamper(MDT_HEAD_CODES,  "head matmul X != final quant codes (restriction seam)");
    tamper(MDT_LOGITS_W,    "prover's logits claim flip (real-slice binding)");
    tamper(MDT_L0 + p3tf::TFT_RMS1_Y,      "L0 rms1 output flip");
    tamper(MDT_L0 + p3tf::TFT_SEAM_MMX,    "L0 matmul X != quant codes");
    tamper(MDT_L1 + p3tf::TFT_SMX_P,       "L1 softmax prob flip");
    tamper(MDT_L1 + p3tf::TFT_SEAM_CONCAT, "L1 concat half != attnout1");
    tamper(MDT_L1 + p3tf::TFT_GW_SWG,      "L1 swiglu silu forge");

    // ---- statement tampers ----
    {
        const char* why = nullptr;
        vector<uint16_t> bad = hlogits; bad[7] ^= 1;
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, hpf, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, MR,
                               MW.lw[0].cos, MW.lw[0].sin, bad, Q, R, &why);
        ck("flipped public logits reject", !ok, why);
    }
    {
        const char* why = nullptr;
        vector<uint32_t> bad = MT.ids; bad[0] = (bad[0] + 1) % MW.vocab;
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, hpf, T, A, MW.cfg, MW.nlayers, MW.vocab, bad, MR,
                               MW.lw[0].cos, MW.lw[0].sin, hlogits, Q, R, &why);
        ck("wrong public token ids reject", !ok, why);
    }
    {
        const char* why = nullptr;
        MdlRoots bad = MR; bad.E.data()[0] ^= 1;
        fs::Transcript tv("tf-model");
        bool ok = verify_model(tv, hpf, T, A, MW.cfg, MW.nlayers, MW.vocab, MT.ids, bad,
                               MW.lw[0].cos, MW.lw[0].sin, hlogits, Q, R, &why);
        ck("wrong pinned embedding root rejects", !ok, why);
    }

    printf("\nZK-MODEL-SMOKE: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
