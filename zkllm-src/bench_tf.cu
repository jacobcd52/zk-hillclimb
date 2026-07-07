// Composed-layer benchmark: one honest prove+verify with a detailed proof-size
// and batch-class breakdown (documentation numbers for design doc section 11).
//   nvcc -arch=sm_89 -std=c++17 -O2 bench_tf.cu -o /root/bench_tf
#include <cstdio>
#include <cstdint>
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

static size_t peak_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256]; size_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "VmHWM: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb;
}

int main() {
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    Art A; Weights W; Trace TR;
    if (!p3rms::load_art("p3_rmsnorm_tables.bin", A) ||
        !load_weights("transformer_weights.bin", W) ||
        !load_trace("transformer_layer.bin", TR)) { printf("FATAL: artifacts\n"); return 1; }
    const uint32_t R = 2, Q = 24;
    Config cfg = W.cfg;
    TfTables T = p3tf::build_tables(A);
    WeightRoots wr = weight_roots(W, R);
    const std::vector<uint16_t>& x0 = *trace_get(TR, "input");

    double t0 = p3hwl::now_ms();
    TfWit w = build_witness(cfg, x0, W, A);
    double t1 = p3hwl::now_ms();
    TfOps o = commit_all(w, R);
    double t2 = p3hwl::now_ms();
    TfProf prof;
    fs::Transcript tp("tf-layer");
    TfProof pf = prove(tp, w, o, T, A, R, Q, true, &prof);
    double t3 = p3hwl::now_ms();
    fs::Transcript tv("tf-layer");
    const char* why = nullptr;
    bool ok = verify(tv, pf, T, A, cfg, x0, o.rms[0].X.root, wr, W.cos, W.sin,
                     w.outp, Q, R, &why);
    double t4 = p3hwl::now_ms();
    printf("verify: %s (%s)\n", ok ? "ACCEPT" : "REJECT", why);
    printf("witness %.2f s | commits %.2f s | prove %.2f s | verify %.2f s | peakRSS %.2f GB\n",
           (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3,
           peak_rss_kb() / 1048576.0);
    printf("prove: rms %.2f qnt %.2f mm %.2f rope %.2f smx %.2f bfa %.2f swg %.2f seam %.2f batch %.2f s\n",
           prof.rms / 1e3, prof.qnt / 1e3, prof.mm / 1e3, prof.rope / 1e3, prof.smx / 1e3,
           prof.bfa / 1e3, prof.swg / 1e3, prof.seam / 1e3, prof.batch / 1e3);

    size_t total = proof_size(pf);
    size_t sbat = 0;
    for (auto& b : pf.batches) sbat += p3bo::sz_batch(b);
    size_t srms = 0, sqn = 0, smm = 0, srp = 0, ssm = 0, sbf = 0;
    for (int i = 0; i < 2; i++) srms += p3rms::proof_size(pf.rms[i]);
    for (int i = 0; i < NQN; i++) sqn += p3qnt::proof_size(pf.qn[i]);
    for (int i = 0; i < NMM; i++) smm += p3hwl::proof_size(pf.mm[i]);
    for (int i = 0; i < 4; i++) srp += p3rope::proof_size(pf.rp[i]);
    for (int i = 0; i < 2; i++) ssm += p3smx::proof_size(pf.sm[i]);
    for (int i = 0; i < 2; i++) sbf += p3bfa::proof_size(pf.res[i]);
    size_t ssw = p3swg::proof_size(pf.sw);
    printf("proof %.2f MB = batch %.2f + mm %.2f + rope %.2f + smx %.2f + rms %.2f"
           " + qnt %.2f + bfa %.2f + swg %.2f MB\n",
           total / 1048576.0, sbat / 1048576.0, (smm) / 1048576.0, srp / 1048576.0,
           ssm / 1048576.0, srms / 1048576.0, sqn / 1048576.0, sbf / 1048576.0,
           ssw / 1048576.0);
    printf("batch classes (%zu):\n", pf.batches.size());
    for (auto& b : pf.batches)
        printf("  v=%2u rows=%7u  nc=%4u distinct cols  %.2f MB\n",
               b.v, 1u << b.v, b.nc, p3bo::sz_batch(b) / 1048576.0);
    return ok ? 0 : 1;
}
