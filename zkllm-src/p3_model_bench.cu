// Composed FULL-MODEL ZKP benchmark at arbitrary pow2 dims (design doc
// sections 18 + 22): embedding gather -> N chained transformer layers ->
// final rmsnorm -> LM head matmul -> public logits, ONE proof.
//
//   nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler -fopenmp p3_model_bench.cu -o /root/p3_model_bench
//   cd /root/zkllm && /root/p3_model_bench <nlayers> <seq> <d> <nh> <dh> <dff> <vocab> <zk 0|1> <tables.bin> [seed]
//
// Generates an in-domain random MODEL (embedding, per-layer weights, head),
// builds the chained model witness, commits, PROVES, VERIFIES, and prints one
// machine-parseable line: witness/commit/prove/verify seconds, per-stage prove
// breakdown, proof MB, peak RSS GB.  One config per process (VmHWM per point).
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
#include "p3_model.cuh"
using namespace p3mdl;
using p3tf::Config; using p3tf::Weights; using p3tf::NW;
using p3tf::W_Q; using p3tf::W_K; using p3tf::W_V; using p3tf::W_O;
using p3tf::W_G; using p3tf::W_U; using p3tf::W_D;
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

static uint64_t rng_s;
static inline uint64_t rnd64() {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s;
}
static inline uint16_t f2bf(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t lsb = (u >> 16) & 1;
    u += 0x7FFF + lsb;
    return (uint16_t)(u >> 16);
}
static inline uint8_t rnd_code() {
    uint32_t s = rnd64() & 1, e = 4 + (rnd64() % 5), m = rnd64() & 7;
    return (uint8_t)((s << 7) | (e << 3) | m);
}
static inline uint16_t rnd_bf16() {
    uint32_t s = rnd64() & 1, e = 125 + (rnd64() % 3), m = rnd64() & 0x7F;
    return (uint16_t)((s << 15) | (e << 7) | m);
}

static Weights gen_layer_weights(const Config& cfg) {
    Weights W; W.cfg = cfg;
    const uint32_t shapes[NW][2] = {
        {cfg.d, cfg.d}, {cfg.d, cfg.d}, {cfg.d, cfg.d}, {cfg.d, cfg.d},
        {cfg.dff, cfg.d}, {cfg.dff, cfg.d}, {cfg.d, cfg.dff}};
    for (int i = 0; i < NW; i++) {
        W.w[i].N = shapes[i][0]; W.w[i].K = shapes[i][1];
        W.w[i].codes.resize((size_t)W.w[i].N * W.w[i].K);
        for (auto& cd : W.w[i].codes) cd = rnd_code();
        W.w[i].scales.assign(W.w[i].N, 0x3F800000u);
    }
    W.g1.resize(cfg.d); W.g2.resize(cfg.d);
    for (auto& g : W.g1) g = f2bf(0.5f + (float)(rnd64() % 1000) / 1000.0f);
    for (auto& g : W.g2) g = f2bf(0.5f + (float)(rnd64() % 1000) / 1000.0f);
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

static ModelWeights gen_model(const Config& cfg, uint32_t nlayers, uint32_t vocab) {
    ModelWeights M;
    M.nlayers = nlayers; M.vocab = vocab; M.cfg = cfg;
    M.emb.resize((size_t)vocab * cfg.d);
    for (auto& x : M.emb) x = rnd_bf16();
    M.lw.resize(nlayers);
    for (uint32_t li = 0; li < nlayers; li++) M.lw[li] = gen_layer_weights(cfg);
    M.gF.resize(cfg.d);
    for (auto& g : M.gF) g = f2bf(0.5f + (float)(rnd64() % 1000) / 1000.0f);
    M.wh.N = vocab; M.wh.K = cfg.d;
    M.wh.codes.resize((size_t)vocab * cfg.d);
    for (auto& cd : M.wh.codes) cd = rnd_code();
    M.wh.scales.assign(vocab, 0x3F800000u);
    return M;
}

int main(int argc, char** argv) {
    if (argc < 10) {
        printf("usage: %s nlayers seq d nh dh dff vocab zk tables.bin [seed]\n", argv[0]);
        return 1;
    }
    uint32_t nlayers = atoi(argv[1]);
    Config cfg;
    cfg.seq = atoi(argv[2]); cfg.d = atoi(argv[3]); cfg.nh = atoi(argv[4]);
    cfg.dh = atoi(argv[5]); cfg.dff = atoi(argv[6]); cfg.batch = 1;
    uint32_t vocab = atoi(argv[7]);
    const bool zk = atoi(argv[8]) != 0;
    const char* tables = argv[9];
    uint64_t seed = argc > 10 ? strtoull(argv[10], nullptr, 10) : 42;
    if (!cfg.pow2()) { printf("FATAL: dims must be pow2 with nh*dh==d\n"); return 1; }
    if ((1u << p3lu::ilog2(vocab)) != vocab) { printf("FATAL: vocab must be pow2\n"); return 1; }

    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    p3zkc::G.on = zk;
    p3hwl::g_free_dp = true;
    p3lu::g_free_idx = true;

    Art A;
    if (!p3rms::load_art(tables, A)) { printf("FATAL: bad tables %s\n", tables); return 1; }
    if ((uint32_t)A.ld != cfg.ld()) { printf("FATAL: artifact ld=%d != %u\n", A.ld, cfg.ld()); return 1; }

    const uint32_t R = 2, Q = 24;
    uint32_t lseq = p3lu::ilog2(p3hwl::pow2_at_least(cfg.seq));
    uint32_t smx_wmax = 16 + lseq > 23 ? 16 + lseq : 23;
    TfTables T = p3tf::build_tables(A, smx_wmax);

    MdlWit w; ModelWeights MW; bool ok = false; int tries = 0;
    double tw0 = p3hwl::now_ms();
    for (; tries < 6 && !ok; tries++) {
        rng_s = seed + 1000003ULL * tries;
        MW = gen_model(cfg, nlayers, vocab);
        vector<uint32_t> ids(cfg.T());
        for (auto& id : ids) id = (uint32_t)(rnd64() % vocab);
        try { w = build_model_witness(MW, ids, A); ok = true; }
        catch (const std::exception& e) {
            printf("# witness retry %d: %s\n", tries, e.what());
            continue;
        }
    }
    if (!ok) { printf("FATAL: no in-domain witness in %d tries\n", tries); return 1; }
    double tw = (p3hwl::now_ms() - tw0) / 1e3;
    printf("# witness %.2f s (try %d)\n", tw, tries - 1);

    double t0 = p3hwl::now_ms();
    MdlOps o = commit_model(w, MW, R);
    double t1 = p3hwl::now_ms();
    // statement roots: recomputed independently when non-zk, pinned from the
    // prover's salted commitments in zk (the published model commitment)
    MdlRoots mr = zk ? roots_from_ops(o, cfg, nlayers) : model_roots(MW, R);
    if (zk) mr.E = o.E.root;

    TfProf prof;
    fs::Transcript tp("tf-model");
    MdlProof pf = prove_model(tp, w, o, T, A, R, Q, /*strict=*/true, &prof);
    double t2 = p3hwl::now_ms();
    if (p3zp::on()) p3zp::report(stdout);
    const char* why = nullptr;
    fs::Transcript tv("tf-model");
    bool vok = verify_model(tv, pf, T, A, cfg, nlayers, vocab, w.ids, mr,
                            MW.lw[0].cos, MW.lw[0].sin, w.logitsPub, Q, R, &why);
    double t3 = p3hwl::now_ms();
    size_t psz = proof_size_model(pf);
    printf("MBENCH nlayers=%u seq=%u d=%u nh=%u dh=%u dff=%u vocab=%u tokens=%u zk=%d "
           "verify_ok=%d witness=%.3f commit=%.3f prove=%.3f verify=%.3f proof_mb=%.3f rss_gb=%.3f\n",
           nlayers, cfg.seq, cfg.d, cfg.nh, cfg.dh, cfg.dff, vocab, cfg.T(), zk ? 1 : 0,
           vok ? 1 : 0, tw, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3,
           psz / 1048576.0, peak_rss_kb() / 1048576.0);
    printf("STAGES rms=%.3f qnt=%.3f mm=%.3f rope=%.3f smx=%.3f bfa=%.3f swg=%.3f "
           "lug=%.3f seam=%.3f batch=%.3f\n",
           prof.rms / 1e3, prof.qnt / 1e3, prof.mm / 1e3, prof.rope / 1e3,
           prof.smx / 1e3, prof.bfa / 1e3, prof.swg / 1e3, prof.lug / 1e3,
           prof.seam / 1e3, prof.batch / 1e3);
    if (!vok) { printf("FATAL: verify failed: %s\n", why ? why : "?"); return 1; }
    return 0;
}
