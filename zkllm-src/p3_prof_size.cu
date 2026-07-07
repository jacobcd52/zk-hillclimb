// Proof-size profiler for the composed transformer-layer proof.
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_prof_size.cu -o /root/p3_prof_size
//   cd /root/zkllm && /root/p3_prof_size
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
using namespace p3tf;
using std::vector;

int main() {
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    Art A;
    if (!p3rms::load_art("p3_rmsnorm_tables.bin", A)) { printf("no tables\n"); return 1; }
    Weights WW;
    if (!load_weights("transformer_weights.bin", WW)) { printf("no weights\n"); return 1; }
    Trace TR;
    if (!load_trace("transformer_layer.bin", TR)) { printf("no trace\n"); return 1; }
    Config CFG = WW.cfg;
    const vector<uint16_t>* xin = trace_get(TR, "input");
    vector<uint16_t> X0PUB = *xin;
    TfTables T = p3tf::build_tables(A);

    const uint32_t R = 2, Q = 24;
    TfWit w = build_witness(CFG, X0PUB, WW, A);
    TfOps o = commit_all(w, R);

    // intercept the ledger by reproving with our own transcript
    fs::Transcript tp("tf-layer");
    TfProf prof;
    TfProof pf = prove(tp, w, o, T, A, R, Q, true, &prof);

    auto MB = [](size_t s) { return s / 1048576.0; };
    size_t s_rms = 0, s_qnt = 0, s_mm = 0, s_rope = 0, s_smx = 0, s_bfa = 0,
           s_swg = 0, s_seam = pf.seam.size() * 8, s_bat = 0, s_mmy = 0;
    for (int i = 0; i < 2; i++) s_rms += p3rms::proof_size(pf.rms[i]);
    for (int i = 0; i < NQN; i++) s_qnt += p3qnt::proof_size(pf.qn[i]);
    for (int i = 0; i < NMM; i++) s_mm += p3hwl::proof_size(pf.mm[i]);
    for (int i = 0; i < 4; i++) s_rope += p3rope::proof_size(pf.rp[i]);
    for (int i = 0; i < 2; i++) s_smx += p3smx::proof_size(pf.sm[i]);
    for (int i = 0; i < 2; i++) s_bfa += p3bfa::proof_size(pf.res[i]);
    s_swg = p3swg::proof_size(pf.sw);
    for (int i = 0; i < NMM; i++) s_mmy += pf.mmY[i].size() * 2;
    for (auto& b : pf.batches) s_bat += p3bo::sz_batch(b);
    printf("total %.2f MB\n", MB(proof_size(pf)));
    printf("  rms   %8.3f MB\n", MB(s_rms));
    printf("  qnt   %8.3f MB (%d inst)\n", MB(s_qnt), NQN);
    printf("  mm    %8.3f MB (%d inst)\n", MB(s_mm), NMM);
    printf("  rope  %8.3f MB\n", MB(s_rope));
    printf("  smx   %8.3f MB\n", MB(s_smx));
    printf("  bfa   %8.3f MB\n", MB(s_bfa));
    printf("  swg   %8.3f MB\n", MB(s_swg));
    printf("  mmY   %8.3f MB\n", MB(s_mmy));
    printf("  seams %8.3f MB (%zu claims)\n", MB(s_seam), pf.seam.size());
    printf("  batch %8.3f MB (%zu classes)\n", MB(s_bat), pf.batches.size());
    for (size_t i = 0; i < pf.batches.size(); i++) {
        auto& b = pf.batches[i];
        size_t sq0 = 0, sqr = 0;
        for (auto& q : b.queries)
            for (auto& r : q.rounds) sqr += 16 + (r.pa.size() + r.pb.size()) * 32;
        for (auto& z : b.q0c)
            sq0 += z.vals.size() * 8 + z.salts.size() * 32 + z.nodes.size() * 32;
        printf("    class %2zu: v=%2u nc=%4u  total %7.3f MB  q0 %7.3f MB  rounds %7.3f MB\n",
               i, b.v, b.nc, MB(p3bo::sz_batch(b)), MB(sq0), MB(sqr));
    }
    printf("prove profile: total prove: rms %.2f qnt %.2f mm %.2f rope %.2f smx %.2f bfa %.2f swg %.2f batch %.2f s\n",
           prof.rms/1e3, prof.qnt/1e3, prof.mm/1e3, prof.rope/1e3, prof.smx/1e3,
           prof.bfa/1e3, prof.swg/1e3, prof.batch/1e3);
    printf("p3lu: prove_v %.2f s over %ld calls (%.2f ms/call); commit_col_nc %.2f s over %ld commits\n",
           p3lu::g_lustats.ms/1e3, p3lu::g_lustats.calls,
           p3lu::g_lustats.calls ? p3lu::g_lustats.ms/p3lu::g_lustats.calls : 0.0,
           p3lu::g_lustats.commit_ms/1e3, p3lu::g_lustats.commits);
    printf("p3lu split: inv %.2f  scA %.2f  scT %.2f s\n",
           p3lu::g_lustats.inv_ms/1e3, p3lu::g_lustats.sc_ms/1e3, p3lu::g_lustats.scg_ms/1e3);
    {
        size_t slug = 0;
        for (auto& g : pf.lug) slug += p3lu::sz_group(g);
        printf("  lug   %8.3f MB (%zu merged groups from 781 lookups)\n",
               MB(slug), pf.lug.size());
        printf("  prove lug stage %.2f s\n", prof.lug / 1e3);
    }
    return 0;
}
