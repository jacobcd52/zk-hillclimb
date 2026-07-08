// device LCG jump-ahead vs host zprng chain: bit-identical check
#include <cstdio>
#include "p3_zkc.cuh"

int main() {
    p3zkc::G.blind_on = true;
    int fail = 0;
    for (uint64_t seed : {1ULL, 0x9E3779B97F4A7C15ULL, 123456789ULL, ~0ULL}) {
        for (size_t n : {1ULL, 255ULL, 4096ULL, (1ULL << 20) + 7, 1ULL << 24}) {
            std::vector<gl_t> h(n);
            p3zkc::blind_col_aug_into(0, seed, h.data(), n);
            gl_t* d; cudaMalloc(&d, n * 8);
            p3zkc::blind_col_aug_dev(seed, d, n);
            std::vector<gl_t> g(n);
            cudaMemcpy(g.data(), d, n * 8, cudaMemcpyDeviceToHost);
            cudaFree(d);
            size_t bad = 0, first = 0;
            for (size_t i = 0; i < n; i++)
                if (h[i] != g[i]) { if (!bad) first = i; bad++; }
            printf("seed=%016llx n=%zu -> %s (bad=%zu first=%zu)\n",
                   (unsigned long long)seed, n, bad ? "FAIL" : "PASS", bad, first);
            if (bad) fail = 1;
        }
    }
    printf("\nLCGDEV: %s\n", fail ? "FAIL" : "ALL PASS");
    return fail;
}
