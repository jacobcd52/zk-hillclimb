// zk_layer_smoke.cu -- composed FULL-LAYER prover with the zk path ON:
// honest-accept + soundness tampers must still reject.  (The hiding battery is
// p3_transformer_zk_test.cu; this is the soundness+wiring smoke.)
//   nvcc -arch=sm_89 -std=c++17 -O2 zk_layer_smoke.cu -o /root/zk_layer_smoke
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
using namespace p3tf;
using std::vector;
static int np_=0, nf_=0;
static void ck(const char* n, bool c, const char* why=nullptr){
    printf("  [%s] %s%s%s%s\n", c?"PASS":"FAIL", n, why?"  (why=":"", why?why:"", why?")":""); if(c)np_++; else nf_++; }
static const uint32_t R=2, Q=24;

// pin the weight/g roots from the prover's (masked) commitments -- in zk the
// commitment roots are the published statement (the prover fixes them once)
static WeightRoots roots_from_ops(const TfOps& o){
    WeightRoots wr;
    const int MMW[NW] = {MM_WQ,MM_WK,MM_WV,MM_WO,MM_WG,MM_WU,MM_WD};
    for(int i=0;i<NW;i++){ wr.Wc[i]=o.mm[MMW[i]].W.root; wr.Ws[i]=o.mm[MMW[i]].WS.root; }
    wr.G1=o.rms[0].G.root; wr.G2=o.rms[1].G.root;
    return wr;
}

int main(){
    printf("=== zk composed FULL-LAYER smoke (G.on) ===\n");
    p3fri::g_gpu_merkle=true; p3bf::p3_enable_mempool();
    p3rms::Art A;
    if(!p3rms::load_art("p3_rmsnorm_tables.bin", A)){ printf("need tables\n"); return 1; }
    Weights WW; if(!load_weights("transformer_weights.bin", WW)){ printf("need weights\n"); return 1; }
    Trace TR; if(!load_trace("transformer_layer.bin", TR)){ printf("need layer\n"); return 1; }
    Config CFG = WW.cfg;
    const vector<uint16_t>* xin = trace_get(TR, "input");
    vector<uint16_t> X0PUB = *xin;
    TfTables T = build_tables(A);

    p3zkc::G.on=true; p3zkc::G.Q=Q;

    // ---- honest accept ----
    TfProof hpf; vector<uint16_t> hout; Hash RX0; WeightRoots WR;
    {
        p3zkc::G.seed=12345; p3zkc::G.ctr=0;
        double t0=p3hwl::now_ms();
        TfWit w = build_witness(CFG, X0PUB, WW, A);
        TfOps o = commit_all(w, R);
        RX0 = o.rms[0].X.root; WR = roots_from_ops(o);
        double t1=p3hwl::now_ms();
        TfProf prof; fs::Transcript tp("tf-layer");
        hpf = prove(tp, w, o, T, A, R, Q, true, &prof);
        double t2=p3hwl::now_ms();
        const char* why=nullptr;
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, hpf, T, A, CFG, X0PUB, RX0, WR, WW.cos, WW.sin, w.outp, Q, R, &why);
        double t3=p3hwl::now_ms();
        ck("zk composed proof VERIFIES (honest)", ok, ok?nullptr:why);
        hout = w.outp;
        size_t psz = proof_size(hpf);
        printf("  prove %.2f s, verify %.2f s, proof %.2f MB (%zu seam claims)\n",
               (t2-t1)/1e3, (t3-t2)/1e3, psz/1048576.0, hpf.seam.size());
        // confirm mmY dropped (no cleartext activations shipped)
        bool dropped=true; for(int i=0;i<NMM;i++) dropped = dropped && hpf.mmY[i].empty();
        ck("zk: per-matmul cleartext output vectors dropped", dropped);
    }

    // ---- soundness tampers (must reject) ----
    auto tamper = [&](int mode, const char* nm){
        const char* why=nullptr;
        TfWit w;
        try { w = build_witness(CFG, X0PUB, WW, A, mode); }
        catch(const std::exception&){ ck(nm, true, "witness threw (out of domain)"); return; }
        TfOps o = commit_all(w, R);
        Hash rx = o.rms[0].X.root; WeightRoots wr = roots_from_ops(o);
        TfProof pf;
        try { fs::Transcript tp("tf-layer"); pf = prove(tp, w, o, T, A, R, Q, false); }
        catch(const std::exception&){ ck(nm, true, "prover threw"); return; }
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, pf, T, A, CFG, X0PUB, rx, wr, WW.cos, WW.sin, w.outp, Q, R, &why);
        ck(nm, !ok, why);
    };
    printf("-- soundness (zk on): tampers must reject --\n");
    tamper(TFT_RMS1_Y,   "rms1 output flip");
    tamper(TFT_SMX_P,    "softmax prob flip");
    tamper(TFT_SEAM_MMX, "matmul X != quant codes (restriction seam)");
    tamper(TFT_SEAM_PVPAD,"nonzero smuggled into PV k-padding (zero seam)");
    tamper(TFT_SEAM_ROPEQ,"rope operand != Wq head slice (slice seam)");
    tamper(TFT_SEAM_CONCAT,"concat half != attnout1 (concat seam)");
    tamper(TFT_SEAM_VT,  "V^T operand != transposed V (transpose seam)");
    tamper(TFT_GW_MM,    "matmul state teleport");
    tamper(TFT_GW_RMS,   "rmsnorm rsqrt forge");
    tamper(TFT_GW_SMX,   "softmax lane-gate forge");
    tamper(TFT_GW_SWG,   "swiglu silu forge");

    // ---- statement tampers ----
    {
        const char* why=nullptr;
        vector<uint16_t> bad = hout; bad[7]^=1;
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, hpf, T, A, CFG, X0PUB, RX0, WR, WW.cos, WW.sin, bad, Q, R, &why);
        ck("flipped public output rejects", !ok, why);
    }
    {
        const char* why=nullptr;
        WeightRoots wr2 = WR; wr2.Wc[W_Q].data()[0]^=1;
        fs::Transcript tv("tf-layer");
        bool ok = verify(tv, hpf, T, A, CFG, X0PUB, RX0, wr2, WW.cos, WW.sin, hout, Q, R, &why);
        ck("wrong pinned weight root rejects", !ok, why);
    }

    printf("\nZK-LAYER-SMOKE: %d passed, %d failed -> %s\n", np_, nf_, nf_==0?"ALL PASS":"FAIL");
    return nf_==0?0:1;
}
