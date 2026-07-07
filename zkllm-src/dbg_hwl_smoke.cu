// Smoke: full-layer Hawkeye proof, honest accept on one tiny golden layer.
#include <cstdio>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
using namespace p3hwl;

static const uint32_t R = 2, Q = 24;

int main(int argc, char** argv) {
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    std::vector<Golden> Ls;
    if (!load_layers("hawkeye_layers.bin", Ls)) { printf("FATAL: no layers file\n"); return 1; }
    int li = argc > 1 ? atoi(argv[1]) : 3;   // default: minimal 1x1x1
    Golden& L = Ls[li];
    printf("layer %d: B=%u K=%u N=%u\n", li, L.B, L.K, L.N);
    Tables T = build_tables();
    LayerWit wt = gen_witness(L);
    printf("witness ok (P=%zu G=%zu Opad=%u)\n", wt.d.P, wt.d.G, wt.d.Opad);
    Operands ops = commit_operands(wt, R);
    double t0 = now_ms();
    Prof prof;
    fs::Transcript tp("hwl");
    LayerProof pf = prove(tp, wt, T, ops, R, Q, true, true, &prof);
    double t_prove = now_ms() - t0;
    t0 = now_ms();
    fs::Transcript tv("hwl");
    const char* why = "?";
    bool ok = verify(tv, T, pf, ops.X.root, ops.W.root, ops.XS.root, ops.WS.root,
                     L.y, L.B, L.K, L.N, Q, R, &why);
    double t_verify = now_ms() - t0;
    printf("prove %.0f ms | verify %.0f ms | size %.2f MB | %s (why=%s)\n",
           t_prove, t_verify, proof_size(pf) / 1048576.0, ok ? "ACCEPT" : "REJECT", why);
    printf("  prof: commit=%.0f luP=%.0f luG=%.0f luO=%.0f luS=%.0f zcP=%.0f zcG=%.0f "
           "chain=%.0f gs=%.0f zcO=%.0f zcS=%.0f slice=%.0f y=%.0f openP=%.0f openG=%.0f openO=%.0f\n",
           prof.commit_wit, prof.lu_dp, prof.lu_dg, prof.lu_do, prof.lu_sc, prof.zc_dp,
           prof.zc_dg, prof.chain, prof.gsum, prof.zc_do, prof.zc_ds, prof.slice, prof.ybind,
           prof.open_dp, prof.open_dg, prof.open_do);
    return ok ? 0 : 1;
}
