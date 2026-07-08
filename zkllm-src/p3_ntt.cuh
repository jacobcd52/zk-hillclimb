// P3.2 Goldilocks NTT (Cooley-Tukey, decimation-in-time, iterative).
// Used for Reed-Solomon encoding: N coeffs -> 2N (or blowup*N) evaluations, the
// codeword that the hash-PCS Merkle-commits.  All arithmetic in Goldilocks.
#pragma once
#include <cstdint>
#include <vector>
#include "p3_goldilocks.cuh"

#define P3_NTT_THREADS 256

__global__ void p3_bitrev_kernel(const gl_t* in, gl_t* out, uint32_t n, uint32_t logn) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t r = __brev(i) >> (32 - logn);
    out[r] = in[i];
}
// tiled bit-reversal permutation (same out[rev(i)] = in[i] mapping): 32x32
// shared-memory tiles make both the gather and the scatter segment-coalesced
// (the naive kernel's random 8-byte scatter was ~1/3 of big-NTT time).
// blockIdx.y = batch column (column b at offset b*n).
__global__ void p3_bitrev_tiled_kernel(const gl_t* in, gl_t* out, uint32_t logn) {
    __shared__ gl_t tile[32][33];
    const uint32_t nmid = logn - 10;
    uint32_t mid = blockIdx.x;
    size_t cb = (size_t)blockIdx.y << logn;
    uint32_t x = threadIdx.x, y = threadIdx.y;
    tile[y][x] = in[cb | ((size_t)y << (logn - 5)) | ((size_t)mid << 5) | x];
    __syncthreads();
    uint32_t rmid = nmid ? (__brev(mid) >> (32 - nmid)) : 0;
    uint32_t rx = __brev(x) >> 27, ry = __brev(y) >> 27;
    out[cb | ((size_t)ry << (logn - 5)) | ((size_t)rmid << 5) | rx] = tile[x][y];
}

// one DIT stage: m = 2^s butterfly span, half = m/2; W is the length-(n/2) table
// of powers of the primitive n-th root, indexed with stride n/m.
__global__ void p3_ntt_stage_kernel(gl_t* a, const gl_t* W, uint32_t n, uint32_t m) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = m >> 1;
    if (tid >= (n >> 1)) return;
    uint32_t blk = tid / half;
    uint32_t i   = tid - blk * half;
    uint32_t base = blk * m + i;
    gl_t w = W[(size_t)i * (n / m)];
    gl_t u = a[base];
    gl_t v = gl_mul(a[base + half], w);
    a[base]        = gl_add(u, v);
    a[base + half] = gl_sub(u, v);
}

// batched variants: B independent same-size columns laid out contiguously
// (column b occupies [b*n, (b+1)*n)); identical butterflies/twiddles per
// column, so each column's output is bitwise the single-column kernel's.
__global__ void p3_bitrev_batch_kernel(const gl_t* in, gl_t* out, uint32_t n,
                                       uint32_t logn, uint32_t B) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * B) return;
    uint32_t b = t / n, i = t - b * n;
    uint32_t r = __brev(i) >> (32 - logn);
    out[(size_t)b * n + r] = in[t];
}
__global__ void p3_ntt_stage_batch_kernel(gl_t* a, const gl_t* W, uint32_t n,
                                          uint32_t m, uint32_t B) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = m >> 1, per = n >> 1;
    if (t >= per * B) return;
    uint32_t b = t / per, tid = t - b * per;
    uint32_t blk = tid / half;
    uint32_t i   = tid - blk * half;
    size_t base = (size_t)b * n + blk * m + i;
    gl_t w = W[(size_t)i * (n / m)];
    gl_t u = a[base];
    gl_t v = gl_mul(a[base + half], w);
    a[base]        = gl_add(u, v);
    a[base + half] = gl_sub(u, v);
}

// FUSED pair of consecutive DIT stages (spans m/2 and m): each thread computes
// the EXACT same mul/add sequence the two radix-2 stage launches would, on 4
// elements -- bitwise-identical results, HALF the global-memory traffic.
__global__ void p3_ntt_stage4_kernel(gl_t* a, const gl_t* W, uint32_t n, uint32_t m) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t q = m >> 2;
    if (tid >= (n >> 2)) return;
    uint32_t blk = tid / q;
    uint32_t i   = tid - blk * q;
    uint32_t base = blk * m + i;
    gl_t x0 = a[base], x1 = a[base + q], x2 = a[base + 2*q], x3 = a[base + 3*q];
    gl_t w1 = W[(size_t)i * (n / (m >> 1))];
    gl_t v1 = gl_mul(x1, w1), v3 = gl_mul(x3, w1);
    gl_t t0 = gl_add(x0, v1), t1 = gl_sub(x0, v1);
    gl_t t2 = gl_add(x2, v3), t3 = gl_sub(x2, v3);
    gl_t u2 = gl_mul(t2, W[(size_t)i * (n / m)]);
    gl_t u3 = gl_mul(t3, W[(size_t)(i + q) * (n / m)]);
    a[base]         = gl_add(t0, u2);
    a[base + 2*q]   = gl_sub(t0, u2);
    a[base + q]     = gl_add(t1, u3);
    a[base + 3*q]   = gl_sub(t1, u3);
}
__global__ void p3_ntt_stage4_batch_kernel(gl_t* a, const gl_t* W, uint32_t n,
                                           uint32_t m, uint32_t B) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t q = m >> 2, per = n >> 2;
    if (t >= per * B) return;
    uint32_t b = t / per, tid = t - b * per;
    uint32_t blk = tid / q;
    uint32_t i   = tid - blk * q;
    size_t base = (size_t)b * n + blk * m + i;
    gl_t x0 = a[base], x1 = a[base + q], x2 = a[base + 2*q], x3 = a[base + 3*q];
    gl_t w1 = W[(size_t)i * (n / (m >> 1))];
    gl_t v1 = gl_mul(x1, w1), v3 = gl_mul(x3, w1);
    gl_t t0 = gl_add(x0, v1), t1 = gl_sub(x0, v1);
    gl_t t2 = gl_add(x2, v3), t3 = gl_sub(x2, v3);
    gl_t u2 = gl_mul(t2, W[(size_t)i * (n / m)]);
    gl_t u3 = gl_mul(t3, W[(size_t)(i + q) * (n / m)]);
    a[base]         = gl_add(t0, u2);
    a[base + 2*q]   = gl_sub(t0, u2);
    a[base + q]     = gl_add(t1, u3);
    a[base + 3*q]   = gl_sub(t1, u3);
}

__global__ void p3_scale_kernel(gl_t* a, gl_t c, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = gl_mul(a[i], c);
}

// Build a twiddle table W[k] = base^k on the GPU (each thread an independent pow).
// Replaces the host O(n) sequential prefix-product + a big H2D upload, which
// dominated commit time for large layers (the W matrix builds ~2^25 twiddles).
__global__ void p3_twiddle_kernel(gl_t* W, gl_t base, uint32_t cnt) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= cnt) return;
    W[k] = gl_pow(base, (uint64_t)k);
}

// Forward/inverse NTT over a device buffer of length n=2^logn (in place).
// Twiddle tables are built on the GPU (no host loop / upload).
struct P3Ntt {
    uint32_t n, logn;
    gl_t* d_Wf = nullptr;   // forward twiddles (powers of omega_n)
    gl_t* d_Wi = nullptr;   // inverse twiddles (powers of omega_n^{-1})
    gl_t  ninv;

    P3Ntt(uint32_t logn_) : logn(logn_) {
        n = 1u << logn;
        gl_t w  = gl_root_of_unity(logn);
        gl_t wi = gl_inv(w);
        uint32_t half = n / 2, blk = (half + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
        cudaMalloc(&d_Wf, (n/2) * sizeof(gl_t));
        cudaMalloc(&d_Wi, (n/2) * sizeof(gl_t));
        p3_twiddle_kernel<<<blk, P3_NTT_THREADS>>>(d_Wf, w,  half);
        p3_twiddle_kernel<<<blk, P3_NTT_THREADS>>>(d_Wi, wi, half);
        ninv = gl_inv((gl_t)n);
    }
    ~P3Ntt() { if (d_Wf) cudaFree(d_Wf); if (d_Wi) cudaFree(d_Wi); }

    // out and in may differ; both length n on device. forward=true => evaluate.
    void run(const gl_t* d_in, gl_t* d_out, bool forward) const {
        uint32_t halfblocks = (n/2 + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
        uint32_t qblocks    = (n/4 + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
        uint32_t nblocks     = (n   + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
        if (logn >= 10)
            p3_bitrev_tiled_kernel<<<dim3(1u << (logn - 10), 1), dim3(32, 32)>>>(d_in, d_out, logn);
        else
            p3_bitrev_kernel<<<nblocks, P3_NTT_THREADS>>>(d_in, d_out, n, logn);
        const gl_t* W = forward ? d_Wf : d_Wi;
        // fused stage pairs (radix-4 traffic); odd logn takes one radix-2 first
        uint32_t m = 2;
        if (logn & 1) {
            p3_ntt_stage_kernel<<<halfblocks, P3_NTT_THREADS>>>(d_out, W, n, m);
            m <<= 1;
        }
        for (m <<= 1; m <= n; m <<= 2)
            p3_ntt_stage4_kernel<<<qblocks, P3_NTT_THREADS>>>(d_out, W, n, m);
        if (!forward) p3_scale_kernel<<<nblocks, P3_NTT_THREADS>>>(d_out, ninv, n);
    }
    // B contiguous same-size columns in one launch series (forward only);
    // per-column results bitwise identical to run()
    void run_batch(const gl_t* d_in, gl_t* d_out, uint32_t B) const {
        uint32_t halfblocks = (n / 2 * B + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
        uint32_t qblocks    = (n / 4 * B + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
        uint32_t nblocks    = (n * B     + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
        if (logn >= 10)
            p3_bitrev_tiled_kernel<<<dim3(1u << (logn - 10), B), dim3(32, 32)>>>(d_in, d_out, logn);
        else
            p3_bitrev_batch_kernel<<<nblocks, P3_NTT_THREADS>>>(d_in, d_out, n, logn, B);
        uint32_t m = 2;
        if (logn & 1) {
            p3_ntt_stage_batch_kernel<<<halfblocks, P3_NTT_THREADS>>>(d_out, d_Wf, n, m, B);
            m <<= 1;
        }
        for (m <<= 1; m <= n; m <<= 2)
            p3_ntt_stage4_batch_kernel<<<qblocks, P3_NTT_THREADS>>>(d_out, d_Wf, n, m, B);
    }
};
