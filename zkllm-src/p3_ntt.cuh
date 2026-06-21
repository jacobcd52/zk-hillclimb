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

__global__ void p3_scale_kernel(gl_t* a, gl_t c, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = gl_mul(a[i], c);
}

// Forward/inverse NTT over a device buffer of length n=2^logn (in place).
// Twiddle tables are built once on host and cached on device.
struct P3Ntt {
    uint32_t n, logn;
    gl_t* d_Wf = nullptr;   // forward twiddles (powers of omega_n)
    gl_t* d_Wi = nullptr;   // inverse twiddles (powers of omega_n^{-1})
    gl_t  ninv;

    P3Ntt(uint32_t logn_) : logn(logn_) {
        n = 1u << logn;
        gl_t w  = gl_root_of_unity(logn);
        gl_t wi = gl_inv(w);
        std::vector<gl_t> Wf(n / 2), Wi(n / 2);
        Wf[0] = 1; Wi[0] = 1;
        for (uint32_t k = 1; k < n / 2; k++) { Wf[k] = gl_mul(Wf[k-1], w); Wi[k] = gl_mul(Wi[k-1], wi); }
        cudaMalloc(&d_Wf, (n/2) * sizeof(gl_t));
        cudaMalloc(&d_Wi, (n/2) * sizeof(gl_t));
        cudaMemcpy(d_Wf, Wf.data(), (n/2)*sizeof(gl_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Wi, Wi.data(), (n/2)*sizeof(gl_t), cudaMemcpyHostToDevice);
        ninv = gl_inv((gl_t)n);
    }
    ~P3Ntt() { if (d_Wf) cudaFree(d_Wf); if (d_Wi) cudaFree(d_Wi); }

    // out and in may differ; both length n on device. forward=true => evaluate.
    void run(const gl_t* d_in, gl_t* d_out, bool forward) const {
        uint32_t halfblocks = (n/2 + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
        uint32_t nblocks     = (n   + P3_NTT_THREADS - 1) / P3_NTT_THREADS;
        p3_bitrev_kernel<<<nblocks, P3_NTT_THREADS>>>(d_in, d_out, n, logn);
        const gl_t* W = forward ? d_Wf : d_Wi;
        for (uint32_t m = 2; m <= n; m <<= 1)
            p3_ntt_stage_kernel<<<halfblocks, P3_NTT_THREADS>>>(d_out, W, n, m);
        if (!forward) p3_scale_kernel<<<nblocks, P3_NTT_THREADS>>>(d_out, ninv, n);
    }
};
