// P3.2 validation: NTT correctness + commitment-cost  hash-PCS vs EC Pedersen.
//   ./p3_commit_bench
// Part A: NTT round-trip + naive-DFT cross-check (small n).
// Part B: commit N field elements two ways and time it:
//   - hash-PCS : Reed-Solomon encode (NTT, blowup 2) + GPU Merkle (SHA-256)
//   - EC Pedersen: the existing Pippenger Commitment::commit (P2)
#include <cstdio>
#include <cstdint>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_ntt.cuh"
#include "p3_merkle.cuh"
#include "commitment.cuh"
#include "fr-tensor.cuh"

// ---------- Part A: NTT correctness ----------
static int ntt_correctness() {
    int fails = 0;
    // naive DFT cross-check at n=8
    {
        const uint32_t logn = 3, n = 8;
        std::vector<gl_t> coeff(n);
        uint64_t s = 7;
        for (uint32_t i = 0; i < n; i++) { s = s*6364136223846793005ULL+1; coeff[i] = s % GL_P; }
        gl_t w = gl_root_of_unity(logn);
        std::vector<gl_t> ref(n);
        for (uint32_t k = 0; k < n; k++) {
            gl_t acc = 0, wk = gl_pow(w, k);
            gl_t cur = 1;
            for (uint32_t j = 0; j < n; j++) { acc = gl_add(acc, gl_mul(coeff[j], cur)); cur = gl_mul(cur, wk); }
            ref[k] = acc;
        }
        gl_t *d_in, *d_out; cudaMalloc(&d_in, n*sizeof(gl_t)); cudaMalloc(&d_out, n*sizeof(gl_t));
        cudaMemcpy(d_in, coeff.data(), n*sizeof(gl_t), cudaMemcpyHostToDevice);
        P3Ntt ntt(logn); ntt.run(d_in, d_out, true);
        std::vector<gl_t> got(n); cudaMemcpy(got.data(), d_out, n*sizeof(gl_t), cudaMemcpyDeviceToHost);
        for (uint32_t k = 0; k < n; k++) if (got[k] != ref[k]) fails++;
        cudaFree(d_in); cudaFree(d_out);
    }
    // round-trip at n=2^16
    {
        const uint32_t logn = 16, n = 1u << logn;
        std::vector<gl_t> in(n); uint64_t s = 99;
        for (uint32_t i = 0; i < n; i++) { s = s*6364136223846793005ULL+1; in[i] = s % GL_P; }
        gl_t *d_a,*d_b,*d_c; cudaMalloc(&d_a,n*sizeof(gl_t)); cudaMalloc(&d_b,n*sizeof(gl_t)); cudaMalloc(&d_c,n*sizeof(gl_t));
        cudaMemcpy(d_a, in.data(), n*sizeof(gl_t), cudaMemcpyHostToDevice);
        P3Ntt ntt(logn);
        ntt.run(d_a, d_b, true);
        ntt.run(d_b, d_c, false);
        std::vector<gl_t> out(n); cudaMemcpy(out.data(), d_c, n*sizeof(gl_t), cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < n; i++) if (out[i] != in[i]) { fails++; }
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }
    printf("Part A (NTT correctness): %s  (%d failures)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails;
}

int main() {
    printf("=== P3.2  hash-PCS commit cost vs EC Pedersen ===\n");
    int fails = ntt_correctness();

    const uint32_t logN = 22;            // commit N = 2^22 ~ 4.19M field elements
    const uint32_t N    = 1u << logN;
    const uint32_t logM = logN + 1;      // blowup 2
    const uint32_t M    = 1u << logM;    // codeword length 2^23

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms;

    // ---- hash-PCS commit ----
    gl_t* d_coeff; cudaMalloc(&d_coeff, M*sizeof(gl_t));     // zero-padded coeffs (length M)
    gl_t* d_code;  cudaMalloc(&d_code,  M*sizeof(gl_t));
    cudaMemset(d_coeff, 0, M*sizeof(gl_t));
    { std::vector<gl_t> data(N); uint64_t s=1234;
      for (uint32_t i=0;i<N;i++){ s=s*6364136223846793005ULL+1; data[i]=s%GL_P; }
      cudaMemcpy(d_coeff, data.data(), N*sizeof(gl_t), cudaMemcpyHostToDevice); }
    uint8_t *d_sA, *d_sB; cudaMalloc(&d_sA, (size_t)M*32); cudaMalloc(&d_sB, (size_t)M*32);
    P3Ntt ntt(logM);
    uint8_t root[32];
    // warmup
    ntt.run(d_coeff, d_code, true); p3_merkle_build(d_code, M, d_sA, d_sB, root); cudaDeviceSynchronize();
    cudaEventRecord(e0);
    ntt.run(d_coeff, d_code, true);                          // RS encode
    p3_merkle_build(d_code, M, d_sA, d_sB, root);            // Merkle commit
    cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
    float ms_hash = ms;
    printf("\nhash-PCS commit (N=2^%u, RS blowup2 -> 2^%u, SHA-256 Merkle): %.2f ms\n", logN, logM, ms_hash);
    printf("  root = %02x%02x%02x%02x...\n", root[0],root[1],root[2],root[3]);
    cudaFree(d_coeff); cudaFree(d_code); cudaFree(d_sA); cudaFree(d_sB);

    // ---- EC Pedersen commit (existing Pippenger) ----
    const uint32_t rowlen = 2048, m = N / rowlen;           // 2048 generators, 2048 rows
    Commitment gen = Commitment::random(rowlen);
    FrTensor t = FrTensor::random(N);
    auto warm = gen.commit(t); cudaDeviceSynchronize();
    cudaEventRecord(e0);
    auto com = gen.commit(t);
    cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
    float ms_ec = ms;
    printf("EC Pedersen commit (N=2^%u, %u rows x %u, Pippenger MSM): %.2f ms  [%u pts]\n",
           logN, m, rowlen, ms_ec, (uint32_t)com.size);

    printf("\n>>> hash-PCS commit is %.1fx faster than EC Pedersen (same N) <<<\n", ms_ec / ms_hash);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 2; }
    printf("\n%s\n", fails==0 ? "P3.2 COMMIT: ALL PASS" : "P3.2 COMMIT: NTT CORRECTNESS FAIL");
    return fails==0 ? 0 : 1;
}
