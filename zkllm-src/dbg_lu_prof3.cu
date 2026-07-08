#include <cstdio>
#include <vector>
#include <chrono>
#include <omp.h>
#include "p3_binius_logup.cuh"
using std::vector;
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static uint64_t rs_ = 999;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }
int main() {
#ifndef __CUDA_ARCH__
    bf16_tab();
#endif
    #pragma omp parallel
    { volatile int wa = omp_get_thread_num(); (void)wa; }
    { void* d; cudaMalloc(&d, 4); cudaFree(d); }
    for (int l : {14, 17, 20}) {
        size_t n = (size_t)1 << l;
        vector<bf128_t> A0(n), B0(n), z(l);
        for (auto& x : A0) { x.lo = rnd(); x.hi = rnd(); }
        for (auto& x : B0) { x.lo = rnd(); x.hi = rnd(); }
        for (auto& x : z) { x.lo = rnd(); x.hi = rnd(); }
        for (int it = 0; it < 3; it++) {
            fs::Transcript tr("p3");
            double t0 = now_ms();
            BfScDev dv; bfsc_dev_alloc(dv, 3, l);
            double t1 = now_ms();
            bfsc_dev_eq(dv.a, z.data(), l);
            cudaDeviceSynchronize();
            double t2 = now_ms();
            auto A = A0, B = B0;
            double t3 = now_ms();
            cudaMemcpy(dv.a + dv.n, A.data(), n * sizeof(bf128_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dv.a + 2 * dv.n, B.data(), n * sizeof(bf128_t), cudaMemcpyHostToDevice);
            double t4 = now_ms();
            bflu::Bf3Prod cf;
            BfScProof pf; vector<bf128_t> zeta;
            bf_sumcheck_prove_gpu(dv, cf, tr, pf, zeta);
            cudaDeviceSynchronize();
            double t5 = now_ms();
            bfsc_dev_free(dv);
            double t6 = now_ms();
            printf("l=%d it=%d: alloc %.1f eq %.1f hostcopy %.1f upload %.1f rounds %.1f free %.1f\n",
                   l, it, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5);
        }
    }
    return 0;
}
