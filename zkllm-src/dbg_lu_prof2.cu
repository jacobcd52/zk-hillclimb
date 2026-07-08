// per-layer GKR profile
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <omp.h>
#include "p3_binius_logup.cuh"
using std::vector;
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static uint64_t rs_ = 12345;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }
int main() {
#ifndef __CUDA_ARCH__
    bf16_tab();
#endif
    #pragma omp parallel
    { volatile int wa = omp_get_thread_num(); (void)wa; }
    { void* d; cudaMalloc(&d, 4); cudaFree(d); }
    const int L = 21;
    vector<bf128_t> lv0((size_t)1 << L);
    for (auto& x : lv0) { x.lo = rnd() | 1; x.hi = rnd(); }
    for (int it = 0; it < 2; it++) {
        printf("--- it %d ---\n", it);
        auto leaves = lv0;
        double t0 = now_ms();
        vector<vector<bf128_t>> lv(L + 1);
        lv[0] = std::move(leaves);
        for (int h = 1; h <= L; h++) {
            size_t n = (size_t)1 << (L - h);
            lv[h].resize(n);
            const vector<bf128_t>& c = lv[h - 1];
            vector<bf128_t>& o = lv[h];
            #pragma omp parallel for schedule(static) if (n >= 16384)
            for (int64_t y = 0; y < (int64_t)n; y++)
                o[y] = bf128_mul(c[2 * y], c[2 * y + 1]);
        }
        printf("levels: %.0f ms\n", now_ms() - t0);
        fs::Transcript tr("prof2");
        bf128_t hdr[3] = {lv[L][0], lv[L-1][0], lv[L-1][1]};
        tr.absorb("t", hdr, sizeof hdr);
        bf128_t mu = bf_chal128(tr);
        vector<bf128_t> z{mu};
        for (int h = L - 1; h >= 1; h--) {
            size_t n = (size_t)1 << (L - h);
            double ta = now_ms();
            vector<bf128_t> A(n), B(n);
            const vector<bf128_t>& c = lv[h - 1];
            #pragma omp parallel for schedule(static) if (n >= 16384)
            for (int64_t y = 0; y < (int64_t)n; y++) { A[y] = c[2*y]; B[y] = c[2*y+1]; }
            double tb = now_ms();
            BfScProof pf; vector<bf128_t> zeta;
            bflu::bflu_sc3(z, A, B, tr, pf, zeta, true);
            double tc = now_ms();
            if (L - h >= 10)
                printf("layer l=%2d: split %.0f ms, sc %.0f ms\n", L - h, tb - ta, tc - tb);
            bf128_t mu2 = bf_chal128(tr);
            z.assign(1, mu2);
            z.insert(z.end(), zeta.begin(), zeta.end());
        }
    }
    return 0;
}
