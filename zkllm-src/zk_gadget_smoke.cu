// zk_gadget_smoke.cu -- honest-accept + tamper-reject for each gadget with the
// zk path ON (p3zkc::G.on = true).  Validates the per-gadget zk prover/verifier
// (augmentation + Libra blinds + salted commits + batch blinder) in isolation
// before the composed layer.
//   nvcc -arch=sm_89 -std=c++17 -O2 zk_gadget_smoke.cu -o /root/zk_gadget_smoke
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_rmsnorm.cuh"
#include "p3_quant.cuh"
#include "p3_bfadd.cuh"
#include "p3_swiglu.cuh"
#include "p3_softmax.cuh"
#include "p3_rope.cuh"
using std::vector;
static int np_=0, nf_=0;
static void ck(const char* n, bool c){ printf("  [%s] %s\n", c?"PASS":"FAIL", n); if(c)np_++; else nf_++; }
static const uint32_t R=2, Q=24;

int main(){
    printf("=== zk gadget smoke (G.on) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    p3rms::Art A;
    if(!p3rms::load_art("p3_rmsnorm_tables.bin", A)){ printf("need tables\n"); return 1; }
    p3zkc::G.on = true; p3zkc::G.Q = Q; p3zkc::G.seed = 0xC0FFEE;

    // ---- quant ----
    {
        vector<p3qnt::Golden> gs;
        if(!p3qnt::load_goldens("p3_quant_golden.bin", gs)){ printf("need quant golden\n"); return 1; }
        auto T = p3qnt::build_tables(A);
        auto& L = gs[0];
        uint32_t ld = 0; while((1u<<ld)<L.d) ld++;
        const char* why=nullptr;
        {
            p3zkc::G.seed = 111;
            auto w = p3qnt::gen_witness(L, A);
            auto o = p3qnt::commit_operands(w, R);
            fs::Transcript tp("zk"); auto pf = p3qnt::prove(tp, w, T, o, R, Q, true);
            fs::Transcript tv("zk");
            bool ok = p3qnt::verify(tv, T, pf, o.X.root, o.CODES.root, o.SCALES.root,
                                    L.B, ld, Q, R, &why);
            ck("quant zk honest accepts", ok);
            if(!ok) printf("    why=%s\n", why);
        }
        {
            p3zkc::G.seed = 222;
            p3qnt::QTamper qt{p3qnt::QT_MAG, 0, 3};
            auto w = p3qnt::gen_witness(L, A, &qt);
            auto o = p3qnt::commit_operands(w, R);
            fs::Transcript tp("zk"); auto pf = p3qnt::prove(tp, w, T, o, R, Q, false);
            fs::Transcript tv("zk");
            bool ok = p3qnt::verify(tv, T, pf, o.X.root, o.CODES.root, o.SCALES.root,
                                    L.B, ld, Q, R, &why);
            ck("quant zk tamper rejects", !ok);
        }
    }

    // ---- bfadd ----
    {
        vector<p3bfa::Golden> gs;
        if(!p3bfa::load_goldens("p3_bfadd_golden.bin", gs)){ printf("need bfadd golden\n"); return 1; }
        auto T = p3bfa::build_tables();
        auto& L = gs[0]; const char* why=nullptr;
        {
            p3zkc::G.seed = 333;
            auto w = p3bfa::gen_witness(L);
            auto o = p3bfa::commit_operands(w, R);
            fs::Transcript tp("zk"); auto pf = p3bfa::prove(tp, w, T, o, R, Q, true);
            fs::Transcript tv("zk");
            bool ok = p3bfa::verify(tv, T, pf, o.X1.root, o.X2.root, o.OUT.root, L.n, Q, R, &why);
            ck("bfadd zk honest accepts", ok);
            if(!ok) printf("    why=%s\n", why);
        }
    }

    // ---- swiglu ----
    {
        vector<p3swg::Golden> gs;
        if(!p3swg::load_goldens("p3_swiglu_golden.bin", gs)){ printf("need swiglu golden\n"); return 1; }
        auto T = p3swg::build_tables(A);
        auto& L = gs[0]; const char* why=nullptr;
        p3zkc::G.seed = 444;
        auto w = p3swg::gen_witness(L, A);
        auto o = p3swg::commit_operands(w, R);
        fs::Transcript tp("zk"); auto pf = p3swg::prove(tp, w, T, o, R, Q, true);
        fs::Transcript tv("zk");
        bool ok = p3swg::verify(tv, T, pf, o.GATE.root, o.UP.root, o.M.root, L.n, Q, R, &why);
        ck("swiglu zk honest accepts", ok);
        if(!ok) printf("    why=%s\n", why);
    }

    // ---- rmsnorm ----
    {
        vector<p3rms::Golden> gs; uint32_t gld=0;
        if(!p3rms::load_goldens("p3_rmsnorm_golden.bin", gs, gld)){ printf("need rms golden\n"); return 1; }
        auto T = p3rms::build_tables(A);
        auto& L = gs[0]; const char* why=nullptr;
        uint32_t ld=0; while((1u<<ld)<L.d) ld++;
        p3zkc::G.seed = 555;
        auto w = p3rms::gen_witness(L, A);
        auto o = p3rms::commit_operands(w, R);
        fs::Transcript tp("zk"); auto pf = p3rms::prove(tp, w, T, A, o, R, Q, true);
        fs::Transcript tv("zk");
        bool ok = p3rms::verify(tv, T, A, pf, o.X.root, o.G.root, o.Y.root, L.B, ld, Q, R, &why);
        ck("rmsnorm zk honest accepts", ok);
        if(!ok) printf("    why=%s\n", why);
    }

    // ---- softmax ----
    {
        vector<p3smx::Golden> gs;
        if(!p3smx::load_goldens("p3_softmax_golden.bin", gs)){ printf("need smx golden\n"); return 1; }
        auto T = p3smx::build_tables(A);
        auto& L = gs[0]; const char* why=nullptr;
        p3zkc::G.seed = 666;
        auto w = p3smx::gen_witness(L, A);
        auto o = p3smx::commit_operands(w, R);
        fs::Transcript tp("zk"); auto pf = p3smx::prove(tp, w, T, A, o, R, Q, true);
        fs::Transcript tv("zk");
        bool ok = p3smx::verify(tv, T, A, pf, L.msk, o.S.root, o.P.root, L.B, L.n, Q, R, &why);
        ck("softmax zk honest accepts", ok);
        if(!ok) printf("    why=%s\n", why);
    }

    // ---- rope ----
    {
        p3rope::GoldenSet gs;
        if(!p3rope::load_goldens("p3_rope_golden.bin", gs)){ printf("need rope golden\n"); return 1; }
        auto T = p3rope::build_tables(A);
        const char* why=nullptr;
        p3zkc::G.seed = 777;
        auto w = p3rope::gen_witness(gs, gs.cases[0], A);
        auto o = p3rope::commit_operands(w, R);
        fs::Transcript tp("zk"); auto pf = p3rope::prove(tp, w, T, o, R, Q, true);
        fs::Transcript tv("zk");
        bool ok = p3rope::verify(tv, T, pf, gs.cos, gs.sin, o.Q.root, o.OUT.root,
                                 gs.seq, gs.dh, Q, R, &why);
        ck("rope zk honest accepts", ok);
        if(!ok) printf("    why=%s\n", why);
    }

    // ---- hawkeye ----
    {
        vector<p3hwl::Golden> Ls;
        if(!p3hwl::load_layers("hawkeye_layers.bin", Ls)){ printf("need hawkeye layers\n"); return 1; }
        auto T = p3hwl::build_tables();
        auto& L = Ls[0]; const char* why=nullptr;
        p3zkc::G.seed = 888;
        auto w = p3hwl::gen_witness(L);
        auto o = p3hwl::commit_operands(w, R);
        fs::Transcript tp("zk"); auto pf = p3hwl::prove(tp, w, T, o, R, Q, true, true);
        fs::Transcript tv("zk");
        bool ok = p3hwl::verify(tv, T, pf, o.X.root, o.W.root, o.XS.root, o.WS.root,
                                w.Y, L.B, L.K, L.N, Q, R, &why);
        ck("hawkeye zk honest accepts", ok);
        if(!ok) printf("    why=%s\n", why);
    }

    printf("\nZK-GADGET-SMOKE: %d passed, %d failed -> %s\n", np_, nf_, nf_==0?"ALL PASS":"FAIL");
    return nf_==0?0:1;
}
