// Composed INTEGER-layer ZKP benchmark at arbitrary pow2 dims (the measured
// integer baseline for the fp8-vs-int overhead plot; INT_LAYER_LOG.md).
//
//   nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp -I. p3_int_layer_bench.cu -o /root/p3_int_layer_bench
//   cd /root/zkllm && /root/p3_int_layer_bench <seq> <d> <nh> <dh> <dff> <batch> <zk 0|1> <tables:-> [seed]
//
// Args mirror p3_transformer_bench (the <tables> slot is accepted for CLI
// parity; the int tables are built in-process from the config).  Generates an
// in-range random integer layer (weights/gains/int rope tables/input), builds
// the chained witness via the gadget replays (bitwise canonical; retries with
// progressively smaller weights if a random draw overflows a range window),
// commits, PROVES the composed layer, VERIFIES it, and prints:
//   BENCH  seq d nh dh dff batch tokens zk verify_ok commit prove verify proof_mb rss_gb
//   STAGES rms qnt mm rope smx bfa swg lug seam batch
// (stage names match the fp8 bench: qnt = the integer rescale stage, bfa =
//  the integer residual adds, seam = the public-IO binding claims).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_int_layer.cuh"
using std::vector;
using namespace p3itf;
using p3ig::gsig;

static size_t peak_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256]; size_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "VmHWM: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb;
}

static uint64_t rng_s;
static inline uint64_t rnd64() {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s;
}
static inline int64_t rnds(int64_t b) {
    return (int64_t)(rnd64() % (2 * (uint64_t)b - 1)) - (b - 1);
}

static Weights gen_weights(const Config& cfg, double wscale) {
    Weights W; W.cfg = cfg;
    int64_t wb = (int64_t)llround(65536.0 * wscale / sqrt((double)cfg.d));
    int64_t wbd = (int64_t)llround(65536.0 * wscale / sqrt((double)cfg.dff));
    for (int i = 0; i < NW; i++) {
        uint32_t K, N; wshape(cfg, i, K, N);
        W.w[i].resize((size_t)K * N);
        for (auto& v : W.w[i]) v = gsig(rnds(i == W_D ? wbd : wb));
    }
    W.g1.resize(cfg.d); W.g2.resize(cfg.d);
    for (auto& v : W.g1) v = gsig(49152 + rnds(16384));
    for (auto& v : W.g2) v = gsig(49152 + rnds(16384));
    W.rp.lseq = cfg.lseq(); W.rp.ldh = cfg.ldh();
    uint32_t dh2 = cfg.dh / 2;
    W.rp.ct.resize((size_t)cfg.seq * dh2); W.rp.st.resize((size_t)cfg.seq * dh2);
    for (uint32_t s = 0; s < cfg.seq; s++)
        for (uint32_t j = 0; j < dh2; j++) {
            double ang = (double)s * pow(10000.0, -2.0 * j / (double)cfg.dh);
            W.rp.ct[(size_t)s * dh2 + j] = (int32_t)llround(cos(ang) * 16384.0);
            W.rp.st[(size_t)s * dh2 + j] = (int32_t)llround(sin(ang) * 16384.0);
        }
    return W;
}

int main(int argc, char** argv) {
    if (argc < 9) {
        printf("usage: %s seq d nh dh dff batch zk tables [seed]\n", argv[0]);
        return 1;
    }
    Config cfg;
    cfg.seq = atoi(argv[1]); cfg.d = atoi(argv[2]); cfg.nh = atoi(argv[3]);
    cfg.dh = atoi(argv[4]); cfg.dff = atoi(argv[5]); cfg.batch = atoi(argv[6]);
    const bool zk = atoi(argv[7]) != 0;
    uint64_t seed = argc > 9 ? strtoull(argv[9], nullptr, 10) : 42;
    if (!cfg.pow2()) { printf("FATAL: dims must be pow2 with nh*dh==d\n"); return 1; }

    p3fri::g_gpu_merkle = true;
    p3bf::p3_enable_mempool();
    p3zkc::G.on = zk;
    const uint32_t R = 2, Q = 24;

    double t0 = now_ms();
    Tables T = layer_tables(cfg);
    printf("# tables %.2f s\n", (now_ms() - t0) / 1e3);

    // in-range witness: retry seeds, then shrink weights
    TfWit w; Weights W; bool ok = false;
    double wscale = 0.9;
    for (int tries = 0; tries < 9 && !ok; tries++) {
        rng_s = seed + 1000003ULL * tries;
        if (tries && tries % 3 == 0) wscale *= 0.7;
        W = gen_weights(cfg, wscale);
        vector<gl_t> x0((size_t)cfg.T() * cfg.d);
        for (auto& v : x0) v = gsig(rnds(1LL << 16));
        double tw = now_ms();
        try { w = build_witness(cfg, x0, W, T); ok = true; }
        catch (const std::exception& e) {
            printf("# witness retry %d (wscale %.2f): %s\n", tries, wscale, e.what());
            continue;
        }
        printf("# witness %.2f s (try %d, wscale %.2f)\n",
               (now_ms() - tw) / 1e3, tries, wscale);
    }
    if (!ok) { printf("FATAL: no in-range witness\n"); return 1; }

    double t1 = now_ms();
    TfOps o = commit_all(w, W, R);
    double t2 = now_ms();
    WeightRoots WR;
    if (!zk) WR = weight_roots(W, R);
    else { for (int i = 0; i < NW; i++) WR.Wc[i] = o.W[i].root;
           WR.G1 = o.G1.root; WR.G2 = o.G2.root; }
    Hash RX0 = o.X0.root;

    TfProf prof;
    fs::Transcript tp("itf-layer");
    TfProof pf = prove(tp, w, o, W, T, R, Q, /*strict=*/true, &prof);
    double t3 = now_ms();
    const char* why = nullptr;
    fs::Transcript tv("itf-layer");
    bool vok = verify(tv, pf, T, cfg, w.x0, RX0, WR, W.rp, w.out, Q, R, &why);
    double t4 = now_ms();
    size_t psz = proof_size(pf);
    printf("BENCH seq=%u d=%u nh=%u dh=%u dff=%u batch=%u tokens=%u zk=%d "
           "verify_ok=%d commit=%.3f prove=%.3f verify=%.3f proof_mb=%.3f rss_gb=%.3f\n",
           cfg.seq, cfg.d, cfg.nh, cfg.dh, cfg.dff, cfg.batch, cfg.T(), zk ? 1 : 0,
           vok ? 1 : 0, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3,
           psz / 1048576.0, peak_rss_kb() / 1048576.0);
    printf("STAGES rms=%.3f qnt=%.3f mm=%.3f rope=%.3f smx=%.3f bfa=%.3f swg=%.3f "
           "lug=%.3f seam=%.3f batch=%.3f\n",
           prof.rms / 1e3, prof.irs / 1e3, prof.mm / 1e3, prof.rope / 1e3,
           prof.smx / 1e3, prof.add / 1e3, prof.swg / 1e3, prof.lug / 1e3,
           prof.io / 1e3, prof.batch / 1e3);
    if (!vok) { printf("FATAL: verify failed: %s\n", why ? why : "?"); return 1; }
    return 0;
}
