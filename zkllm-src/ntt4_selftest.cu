// radix-4 (fused stage-pair) NTT vs the pure radix-2 stage chain: outputs must
// be BITWISE identical (the fused kernel performs the same field ops).
#include <cstdio>
#include <vector>
#include "p3_ntt.cuh"

int main() {
    int fail = 0;
    for (uint32_t logn : {1u, 2u, 3u, 7u, 12u, 13u, 20u, 25u}) {
        uint32_t n = 1u << logn;
        P3Ntt ntt(logn);
        std::vector<gl_t> in(n);
        uint64_t s = 42 + logn;
        for (auto& x : in) { s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                             x = (s ^ (s >> 31)) % GL_P; }
        gl_t *d_in, *d_a, *d_b;
        cudaMalloc(&d_in, (size_t)n * 8); cudaMalloc(&d_a, (size_t)n * 8); cudaMalloc(&d_b, (size_t)n * 8);
        cudaMemcpy(d_in, in.data(), (size_t)n * 8, cudaMemcpyHostToDevice);
        // reference: pure radix-2 chain (the pre-change run())
        {
            uint32_t halfblocks = (n/2 + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
            uint32_t nblocks    = (n   + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
            p3_bitrev_kernel<<<nblocks, P3_NTT_THREADS>>>(d_in, d_a, n, logn);
            for (uint32_t m = 2; m <= n; m <<= 1)
                p3_ntt_stage_kernel<<<halfblocks, P3_NTT_THREADS>>>(d_a, ntt.d_Wf, ntt.hb, n, m);
        }
        ntt.run(d_in, d_b, true);
        std::vector<gl_t> a(n), b(n);
        cudaMemcpy(a.data(), d_a, (size_t)n * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, (size_t)n * 8, cudaMemcpyDeviceToHost);
        size_t bad = 0;
        for (uint32_t i = 0; i < n; i++) if (a[i] != b[i]) bad++;
        // inverse roundtrip through the fused path too
        gl_t* d_c; cudaMalloc(&d_c, (size_t)n * 8);
        ntt.run(d_b, d_c, false);
        std::vector<gl_t> c(n);
        cudaMemcpy(c.data(), d_c, (size_t)n * 8, cudaMemcpyDeviceToHost);
        size_t badr = 0;
        for (uint32_t i = 0; i < n; i++) if (c[i] != in[i]) badr++;
        // batched forward (B=3) vs single
        size_t badb = 0;
        if (logn <= 20) {
            uint32_t B = 3;
            gl_t *d_bi, *d_bo;
            cudaMalloc(&d_bi, (size_t)B * n * 8); cudaMalloc(&d_bo, (size_t)B * n * 8);
            for (uint32_t k = 0; k < B; k++)
                cudaMemcpy(d_bi + (size_t)k * n, d_in, (size_t)n * 8, cudaMemcpyDeviceToDevice);
            ntt.run_batch(d_bi, d_bo, B);
            std::vector<gl_t> bb((size_t)B * n);
            cudaMemcpy(bb.data(), d_bo, (size_t)B * n * 8, cudaMemcpyDeviceToHost);
            for (uint32_t k = 0; k < B; k++)
                for (uint32_t i = 0; i < n; i++)
                    if (bb[(size_t)k * n + i] != b[i]) badb++;
            cudaFree(d_bi); cudaFree(d_bo);
        }
        printf("logn=%2u fwd=%s (bad=%zu)  inv-roundtrip=%s (bad=%zu)  batch=%s (bad=%zu)\n",
               logn, bad ? "FAIL" : "PASS", bad, badr ? "FAIL" : "PASS", badr,
               badb ? "FAIL" : "PASS", badb);
        if (bad || badr || badb) fail = 1;
        cudaFree(d_in); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }
    printf("\nNTT4: %s\n", fail ? "FAIL" : "ALL PASS");
    return fail;
}
