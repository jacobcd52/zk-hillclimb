// Composed transformer-layer ZKP benchmark at arbitrary pow2 dims.
//
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_transformer_bench.cu -o /root/p3_transformer_bench
//   cd /root/zkllm && /root/p3_transformer_bench <seq> <d> <nh> <dh> <dff> <batch> <zk 0|1> <tables.bin> [seed]
//
// Generates an in-domain random layer (weights, gains, canonical rope cos/sin,
// bf16 input), builds the chained witness via the gadget replays (bitwise
// canonical by construction), commits, PROVES the composed layer, VERIFIES it,
// and prints one machine-parseable result line: witness/commit/prove/verify
// seconds, per-stage prove breakdown, proof MB, peak RSS GB.  One config per
// process so VmHWM is per-point.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

static size_t peak_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256]; size_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "VmHWM: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb;
}

// xorshift PRNG (deterministic per seed)
static uint64_t rng_s;
static inline uint64_t rnd64() {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s;
}
// float -> bf16 RNE
static inline uint16_t f2bf(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t lsb = (u >> 16) & 1;
    u += 0x7FFF + lsb;
    return (uint16_t)(u >> 16);
}
// moderate-range e4m3 code: exponent field 4..8 -> |v| in [0.13, 2.5]
static inline uint8_t rnd_code() {
    uint32_t s = rnd64() & 1, e = 4 + (rnd64() % 5), m = rnd64() & 7;
    return (uint8_t)((s << 7) | (e << 3) | m);
}
// bf16 value in +-[0.25, 2): exponent 125..127, random mantissa
static inline uint16_t rnd_bf16() {
    uint32_t s = rnd64() & 1, e = 125 + (rnd64() % 3), m = rnd64() & 0x7F;
    return (uint16_t)((s << 15) | (e << 7) | m);
}

static Weights gen_weights(const Config& cfg) {
    Weights W; W.cfg = cfg;
    const uint32_t shapes[NW][2] = {                 // {N, K}
        {cfg.d, cfg.d}, {cfg.d, cfg.d}, {cfg.d, cfg.d}, {cfg.d, cfg.d},
        {cfg.dff, cfg.d}, {cfg.dff, cfg.d}, {cfg.d, cfg.dff}};
    for (int i = 0; i < NW; i++) {
        W.w[i].N = shapes[i][0]; W.w[i].K = shapes[i][1];
        W.w[i].codes.resize((size_t)W.w[i].N * W.w[i].K);
        for (auto& cd : W.w[i].codes) cd = rnd_code();
        W.w[i].scales.assign(W.w[i].N, 0x3F800000u);   // 1.0f per row
    }
    W.g1.resize(cfg.d); W.g2.resize(cfg.d);
    for (auto& g : W.g1) g = f2bf(0.5f + (float)(rnd64() % 1000) / 1000.0f);
    for (auto& g : W.g2) g = f2bf(0.5f + (float)(rnd64() % 1000) / 1000.0f);
    // canonical rope tables: theta = 10000, angle = m * theta^(-2j/dh)
    size_t nh2 = (size_t)cfg.seq * (cfg.dh / 2);
    W.cos.resize(nh2); W.sin.resize(nh2);
    for (uint32_t m = 0; m < cfg.seq; m++)
        for (uint32_t j = 0; j < cfg.dh / 2; j++) {
            double ang = (double)m * pow(10000.0, -2.0 * j / cfg.dh);
            W.cos[(size_t)m * (cfg.dh / 2) + j] = f2bf((float)cos(ang));
            W.sin[(size_t)m * (cfg.dh / 2) + j] = f2bf((float)sin(ang));
        }
    return W;
}

int main(int argc, char** argv) {
    if (argc < 9) {
        printf("usage: %s seq d nh dh dff batch zk tables.bin [seed]\n", argv[0]);
        return 1;
    }
    Config cfg;
    cfg.seq = atoi(argv[1]); cfg.d = atoi(argv[2]); cfg.nh = atoi(argv[3]);
    cfg.dh = atoi(argv[4]); cfg.dff = atoi(argv[5]); cfg.batch = atoi(argv[6]);
    const bool zk = atoi(argv[7]) != 0;
    const char* tables = argv[8];
    uint64_t seed = argc > 9 ? strtoull(argv[9], nullptr, 10) : 42;
    if (!cfg.pow2()) { printf("FATAL: dims must be pow2 with nh*dh==d\n"); return 1; }

    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    p3zkc::G.on = zk;

    Art A;
    if (!p3rms::load_art(tables, A)) { printf("FATAL: bad tables %s\n", tables); return 1; }
    if ((uint32_t)A.ld != cfg.ld()) { printf("FATAL: artifact ld=%d != %u\n", A.ld, cfg.ld()); return 1; }

    const uint32_t R = 2, Q = 24;
    TfTables T = p3tf::build_tables(A);

    // retry a few seeds if a random draw lands outside a gadget v1 domain
    TfWit w; Weights W; bool ok = false; int tries = 0;
    for (; tries < 6 && !ok; tries++) {
        rng_s = seed + 1000003ULL * tries;
        W = gen_weights(cfg);
        vector<uint16_t> x0((size_t)cfg.T() * cfg.d);
        for (auto& x : x0) x = rnd_bf16();
        double t0 = p3hwl::now_ms();
        try { w = build_witness(cfg, x0, W, A); ok = true; }
        catch (const std::exception& e) {
            printf("# witness retry %d: %s\n", tries, e.what());
            continue;
        }
        printf("# witness %.2f s (try %d)\n", (p3hwl::now_ms() - t0) / 1e3, tries);
    }
    if (!ok) { printf("FATAL: no in-domain witness in %d tries\n", tries); return 1; }

    WeightRoots WR;   // zk: pinned from the prover's (masked) commitments below

    double t0 = p3hwl::now_ms();
    TfOps o = commit_all(w, R);
    double t1 = p3hwl::now_ms();
    Hash RX0 = o.rms[0].X.root;
    {   // public weight/gain roots: recomputed independently when non-zk, pinned
        // from the prover's salted commitments in zk (the published statement)
        if (!zk) WR = weight_roots(W, R, cfg.batch);
        else {
            const int MMW[NW] = {MM_WQ, MM_WK, MM_WV, cfg.mmWO(), cfg.mmWG(),
                                 cfg.mmWU(), cfg.mmWD()};
            for (int i = 0; i < NW; i++) { WR.Wc[i] = o.mm[MMW[i]].W.root; WR.Ws[i] = o.mm[MMW[i]].WS.root; }
            WR.G1 = o.rms[0].G.root; WR.G2 = o.rms[1].G.root;
        }
    }
    TfProf prof;
    fs::Transcript tp("tf-layer");
    TfProof pf = prove(tp, w, o, T, A, R, Q, /*strict=*/true, &prof);
    double t2 = p3hwl::now_ms();
    const char* why = nullptr;
    fs::Transcript tv("tf-layer");
    bool vok = verify(tv, pf, T, A, cfg, w.x0, RX0, WR, W.cos, W.sin,
                      w.outp, Q, R, &why);
    double t3 = p3hwl::now_ms();
    size_t psz = proof_size(pf);
    printf("BENCH seq=%u d=%u nh=%u dh=%u dff=%u batch=%u tokens=%u zk=%d "
           "verify_ok=%d commit=%.3f prove=%.3f verify=%.3f proof_mb=%.3f rss_gb=%.3f\n",
           cfg.seq, cfg.d, cfg.nh, cfg.dh, cfg.dff, cfg.batch, cfg.T(), zk ? 1 : 0,
           vok ? 1 : 0, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3,
           psz / 1048576.0, peak_rss_kb() / 1048576.0);
    printf("STAGES rms=%.3f qnt=%.3f mm=%.3f rope=%.3f smx=%.3f bfa=%.3f swg=%.3f "
           "lug=%.3f seam=%.3f batch=%.3f\n",
           prof.rms / 1e3, prof.qnt / 1e3, prof.mm / 1e3, prof.rope / 1e3,
           prof.smx / 1e3, prof.bfa / 1e3, prof.swg / 1e3, prof.lug / 1e3,
           prof.seam / 1e3, prof.batch / 1e3);
    if (!vok) { printf("FATAL: verify failed: %s\n", why ? why : "?"); return 1; }
    return 0;
}
