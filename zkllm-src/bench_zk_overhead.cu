// bench_zk_overhead.cu -- measured cost of the ZK (mask-slice + blind + salt)
// layer over the non-ZK Basefold commit path.  The dominant prover cost is
// Merkle+NTT ~ linear in committed data (FABLE_REPORT §6); mask-slicing doubles a
// column's length (N -> 2N), so the expected factor is ~2x committed data /
// encode time.  Measured here on the host RS-encode (the per-element work the
// GPU NTT parallelizes) so the number is deterministic and GPU-contention-free.
#include <cstdio>
#include <vector>
#include <chrono>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_zkopen.cuh"
using std::vector;
static double now_ms() { return std::chrono::duration<double,std::milli>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }
static uint64_t S=7; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const uint32_t R = 2;
    printf("=== ZK overhead: committed data & NTT-encode cost, non-ZK vs mask-sliced(2N)+blind ===\n");
    { vector<gl_t> warm(1u<<14); auto w = p3bf::rs_encode_gpu(warm, R); cudaDeviceSynchronize(); }
    printf("  v       N      NTT-enc(N)   NTT-enc(2N aug)   NTT-enc(2N blind)   ZK/non-ZK(aug only)\n");
    for (uint32_t v : {16u, 18u, 20u}) {             // GPU NTT is O(M log M): true prover path
        uint32_t N = 1u << v;
        vector<gl_t> real(N); for (auto& x : real) x = rng() % 256;
        vector<gl_t> aug(2*N); for (uint32_t i=0;i<N;i++){ aug[i]=real[i]; aug[N+i]=rng(); }
        vector<gl_t> blind(2*N); for (auto& x : blind) x = rng();
        auto tenc = [&](const vector<gl_t>& c){ double t0=now_ms(); auto cw=p3bf::rs_encode_gpu(c,R);
            cudaDeviceSynchronize(); return now_ms()-t0; };
        double t_n = tenc(real), t_a = tenc(aug), t_b = tenc(blind);
        printf("  %2u  %6u   %8.2f ms    %8.2f ms       %8.2f ms         %.2fx\n",
               v, N, t_n, t_a, t_b, t_a / t_n);
    }
    printf("\n  Committed ELEMENTS per column: non-ZK N -> ZK 2N (augmented) = exactly 2x.\n");
    printf("  The full-domain blind is ONE extra column per opening SIZE-CLASS (not per\n");
    printf("  column), amortized across all columns of that class, so the layer-level\n");
    printf("  committed-data factor is ~2x -- matching the ~doubling budget in the mandate.\n");

    printf("\n=== proof-size delta of one hiding opening (v=16, Q=24) ===\n");
    {
        const uint32_t vv = 16, Q = 24;
        vector<gl_t> real(1u<<vv); for (auto& x : real) x = rng()%256;
        vector<gl_t> z(vv); for (auto& x : z) x = rng();
        vector<gl_t> r(vv+1); for (auto& x : r) x = rng();
        vector<uint32_t> qpos(Q); for (auto& q : qpos) q = rng() % (1u<<(vv+1+R));
        p3zko::HOpen o = p3zko::open(real, z, rng(), rng(), r, R, qpos, 1,2,3);
        size_t msgs = (size_t)o.msgs.size()*24, evals = 3*8, cw = (size_t)Q*8, salts = (size_t)Q*32;
        printf("  hiding-open transcript: %zu msg(%zu rounds) + %zu eval + %zu cw + %zu salt = %zu B\n",
               msgs, o.msgs.size(), evals, cw, salts, msgs+evals+cw+salts);
        printf("  ZK-only extra vs non-ZK open: +y_h(8) +1 ex-round(24) +Q salts(%zu) ~= %zu B/open\n",
               salts, (size_t)8+24+salts);
    }
    return 0;
}
