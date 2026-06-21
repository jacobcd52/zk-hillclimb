// P3.1 validation: Goldilocks correctness + field-multiply throughput vs BLS12-381.
//
//   ./p3_field_bench
//
// Part A: host correctness -- gl_mul / gl_add / gl_sub / gl_inv / roots cross-checked
//         against an independent __int128 modular reference.
// Part B: GPU throughput -- many threads, each a 2-accumulator multiply chain.
//         Reports Gmul/s for Goldilocks vs blstrs__scalar__Scalar_mul and the ratio.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "p3_goldilocks.cuh"
#include "bls12-381.cuh"

typedef blstrs__scalar__Scalar Fr;

// ---------- independent host reference ----------
static uint64_t ref_mul(uint64_t a, uint64_t b) {
    return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) % (unsigned __int128)GL_P);
}
static uint64_t ref_add(uint64_t a, uint64_t b) {
    return (uint64_t)(((unsigned __int128)a + (unsigned __int128)b) % (unsigned __int128)GL_P);
}
static uint64_t splitmix(uint64_t& s) {
    s += 0x9e3779b97f4a7c15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static int host_correctness() {
    int fails = 0;
    uint64_t s = 12345;
    for (int i = 0; i < 2000000; i++) {
        uint64_t a = splitmix(s) % GL_P, b = splitmix(s) % GL_P;
        if (gl_mul(a, b) != ref_mul(a, b)) { if (fails<5) printf("  MUL mismatch a=%llu b=%llu\n",(unsigned long long)a,(unsigned long long)b); fails++; }
        if (gl_add(a, b) != ref_add(a, b)) { if (fails<5) printf("  ADD mismatch\n"); fails++; }
        uint64_t d = gl_sub(a, b);
        if (gl_add(d, b) != a) { fails++; }                       // (a-b)+b == a
        if (d >= GL_P) fails++;                                    // canonical
    }
    // inverse
    s = 99;
    for (int i = 0; i < 100000; i++) {
        uint64_t a = (splitmix(s) % (GL_P - 1)) + 1;               // nonzero
        if (gl_mul(a, gl_inv(a)) != 1ULL) fails++;
    }
    // roots of unity: order exactly 2^logn
    for (uint32_t logn = 1; logn <= 24; logn++) {
        gl_t w = gl_root_of_unity(logn);
        gl_t full = gl_pow(w, 1ULL << logn);
        gl_t half = gl_pow(w, 1ULL << (logn - 1));
        if (full != 1ULL) { printf("  root order: w^(2^%u) != 1\n", logn); fails++; }
        if (half == 1ULL) { printf("  root not primitive at logn=%u\n", logn); fails++; }
    }
    printf("Part A (host correctness): %s  (%d failures)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails;
}

// ---------- GPU throughput kernels ----------
__global__ void gl_bench_kernel(gl_t* out, uint64_t K, uint64_t seed) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    gl_t a = (seed + tid + 1) % GL_P;
    gl_t c = (seed * 2654435761ULL + tid * 40503ULL + 7) % GL_P;
    gl_t b = (a ^ 0x5bf03635ULL) % GL_P;
    for (uint64_t i = 0; i < K; i++) { a = gl_mul(a, b); c = gl_mul(c, b); }
    out[tid] = gl_add(a, c);
}

__global__ void bls_bench_kernel(Fr* out, uint64_t K, uint32_t seed) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    Fr a, b, c;
    #pragma unroll
    for (int i = 0; i < blstrs__scalar__Scalar_LIMBS; i++) {
        a.val[i] = (seed + tid + 1) * (i + 1) + 0x9e37u;
        b.val[i] = (seed ^ (tid * 2654435761u)) + i * 7u + 1u;
        c.val[i] = (seed + tid * 40503u) ^ (i * 0x85ebca6bu);
    }
    for (uint64_t i = 0; i < K; i++) {
        a = blstrs__scalar__Scalar_mul(a, b);
        c = blstrs__scalar__Scalar_mul(c, b);
    }
    out[tid] = blstrs__scalar__Scalar_add(a, c);
}

int main() {
    printf("=== P3.1  Goldilocks field  ===\n");
    int fails = host_correctness();

    const uint32_t THREADS = 256;
    const uint32_t BLOCKS  = 4096;            // ~1.05M threads
    const uint64_t NT      = (uint64_t)THREADS * BLOCKS;
    const uint64_t K       = 2000;            // muls per accumulator per thread
    const double   total_muls = (double)NT * K * 2.0;

    gl_t* d_gl; cudaMalloc(&d_gl, NT * sizeof(gl_t));
    Fr*   d_fr; cudaMalloc(&d_fr, NT * sizeof(Fr));
    cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);

    // warmup
    gl_bench_kernel<<<BLOCKS, THREADS>>>(d_gl, 10, 1);
    bls_bench_kernel<<<BLOCKS, THREADS>>>(d_fr, 10, 1);
    cudaDeviceSynchronize();

    float ms_gl = 0, ms_fr = 0;
    cudaEventRecord(s0);
    gl_bench_kernel<<<BLOCKS, THREADS>>>(d_gl, K, 123456789ULL);
    cudaEventRecord(s1); cudaEventSynchronize(s1); cudaEventElapsedTime(&ms_gl, s0, s1);

    cudaEventRecord(s0);
    bls_bench_kernel<<<BLOCKS, THREADS>>>(d_fr, K, 123456789u);
    cudaEventRecord(s1); cudaEventSynchronize(s1); cudaEventElapsedTime(&ms_fr, s0, s1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 2; }

    double gl_gms = total_muls / (ms_gl * 1e6);   // G mul / s
    double fr_gms = total_muls / (ms_fr * 1e6);
    printf("\nPart B (GPU throughput, %.2e multiplies each):\n", total_muls);
    printf("  Goldilocks (64-bit):   %7.2f ms   %6.2f Gmul/s\n", ms_gl, gl_gms);
    printf("  BLS12-381  (256-bit):  %7.2f ms   %6.2f Gmul/s\n", ms_fr, fr_gms);
    printf("  >>> Goldilocks is %.1fx faster per multiply <<<\n", ms_fr / ms_gl);

    cudaFree(d_gl); cudaFree(d_fr);
    printf("\n%s\n", fails == 0 ? "P3.1 FIELD: ALL PASS" : "P3.1 FIELD: CORRECTNESS FAIL");
    return fails == 0 ? 0 : 1;
}
