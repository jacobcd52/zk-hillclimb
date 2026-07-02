// Benchmark + per-phase profile of the full-layer Hawkeye ZKP.
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_hawkeye_bench.cu -o p3_hawkeye_bench
//   ./p3_hawkeye_bench B K N   (e.g. 1 768 3072 for llama-68m up_proj at B=1)
// Generates a random supported-domain layer, proves, verifies, prints the
// timing table.  The witness generator's internal float cross-check plus the
// suite's golden battery cover correctness; this binary measures cost.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/resource.h>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
using namespace p3hwl;

static const uint32_t R = 2, Q = 24;

static double peak_rss_gb() {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1048576.0;
}

int main(int argc, char** argv) {
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    uint32_t B = argc > 1 ? atoi(argv[1]) : 1;
    uint32_t K = argc > 2 ? atoi(argv[2]) : 768;
    uint32_t N = argc > 3 ? atoi(argv[3]) : 3072;
    Golden L; L.B = B; L.K = K; L.N = N;
    std::mt19937_64 rng(20260702);
    L.x.resize((size_t)B * K); L.w.resize((size_t)N * K);
    for (auto& c : L.x) c = (uint8_t)rng();
    for (auto& c : L.w) c = (uint8_t)rng();
    std::normal_distribution<float> nd(0.f, 2.f);
    L.xs.resize(B); L.ws.resize(N);
    for (auto& s : L.xs) { float f = expf(nd(rng)); memcpy(&s, &f, 4); }
    for (auto& s : L.ws) { float f = expf(nd(rng)); memcpy(&s, &f, 4); }

    Dims d = make_dims(B, K, N);
    printf("=== Hawkeye full-layer bench: B=%u K=%u N=%u ===\n", B, K, N);
    printf("padded: Bpad=%u Npad=%u Gpad=%u (NG=%u)  products P=%zu (real %zu)  groups G=%zu  Opad=%u\n",
           d.Bpad, d.Npad, d.Gpad, d.NG, d.P, (size_t)B * N * ((K + 31) / 32) * 32, d.G, d.Opad);
    size_t committed = (size_t)NDP * d.P + (size_t)NDG * d.G + (size_t)NDO * d.Opad
                     + (size_t)NDS * (d.Bpad + d.Npad)
                     + d.P * 4 + d.G * 12 + d.Opad * 16;   // + logUp hA helpers (approx)
    printf("committed witness (approx incl. logUp helpers): %.1f M elts\n", committed / 1e6);

    double t0 = now_ms();
    Tables T = build_tables();
    printf("tables: %.0f ms\n", now_ms() - t0);

    t0 = now_ms();
    LayerWit wt = gen_witness(L);
    printf("witness gen (+float cross-check): %.0f ms\n", now_ms() - t0);

    t0 = now_ms();
    Operands ops = commit_operands(wt, R);
    printf("operand commits: %.0f ms\n", now_ms() - t0);

    Prof prof;
    t0 = now_ms();
    fs::Transcript tp("hwl");
    LayerProof pf = prove(tp, wt, T, ops, R, Q, true, true, &prof);
    double t_prove = now_ms() - t0;

    t0 = now_ms();
    fs::Transcript tv("hwl");
    const char* why = "?";
    bool ok = verify(tv, T, pf, ops.X.root, ops.W.root, ops.XS.root, ops.WS.root,
                     L.y.empty() ? wt.Y : L.y, B, K, N, Q, R, &why);
    double t_verify = now_ms() - t0;

    printf("\nPROVE  %.2f s   VERIFY %.2f s   proof %.2f MB   %s (why=%s)   peak RSS %.1f GB\n",
           t_prove / 1e3, t_verify / 1e3, proof_size(pf) / 1048576.0,
           ok ? "ACCEPT" : "REJECT", why, peak_rss_gb());
    printf("\nper-phase prover profile (ms):\n");
    printf("  witness commits        %8.0f\n", prof.commit_wit);
    printf("  lookups: products      %8.0f\n", prof.lu_dp);
    printf("  lookups: groups        %8.0f\n", prof.lu_dg);
    printf("  lookups: outputs       %8.0f\n", prof.lu_do);
    printf("  lookups: scales        %8.0f\n", prof.lu_sc);
    printf("  zero-check Dp sumcheck %8.0f\n", prof.zc_dp);
    printf("  zero-check Dp openings %8.0f\n", prof.open_dp);
    printf("  zero-check Dg sumcheck %8.0f\n", prof.zc_dg);
    printf("  zero-check Dg openings %8.0f\n", prof.open_dg);
    printf("  chain sumcheck+open    %8.0f\n", prof.chain);
    printf("  group-sum binding      %8.0f\n", prof.gsum);
    printf("  zero-check Do sumcheck %8.0f\n", prof.zc_do);
    printf("  zero-check Do openings %8.0f\n", prof.open_do);
    printf("  scale decomp checks    %8.0f\n", prof.zc_ds);
    printf("  slice bindings         %8.0f\n", prof.slice);
    printf("  Y binding              %8.0f\n", prof.ybind);
    printf("  batched openings       %8.0f\n", prof.batch);
    return ok ? 0 : 1;
}
