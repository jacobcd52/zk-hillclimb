// Generic GPU multi-column sumcheck prover rounds.
//
// Proves sum_b F(cols(b)) for a per-pair polynomial F of degree < NT in each
// variable: every round sends the NT evaluations of the round univariate at
// t = 0..NT-1 (same message layout/bytes as the host sc_prove / sc5_prove
// loops, so transcripts and verifiers are unchanged), then MLE-binds all
// columns with the round challenge, LSB-first.  Field ops are exact, so the
// block-reduced sums are bit-identical to the host accumulation.
//
// F is supplied as a functor type with  static __device__ gl_t eval(const
// gl_t* cols, const gl_t* par)  -- cols = the nc column values at a vertex,
// par = the caller's lambda/parameter vector (device-resident).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_goldilocks.cuh"
#include "fs_transcript.hpp"

namespace p3sg {

static inline gl_t chal(fs::Transcript& tr) {
    uint8_t b[32]; tr.challenge_bytes(b);
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v % GL_P;
}

__global__ void p3sg_fill_kernel(gl_t* out, gl_t val, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    out[i] = val;
}
__global__ void p3sg_bind_kernel(const gl_t* in, gl_t* out, uint32_t half, gl_t a) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= half) return;
    out[i] = gl_add(in[2 * i], gl_mul(a, gl_sub(in[2 * i + 1], in[2 * i])));
}
// fused bind of ALL nc columns in one launch (same element math as
// p3sg_bind_kernel; the per-column launch train dominated small chains)
__global__ void p3sg_bindn_kernel(gl_t* const* in, gl_t* const* out, uint32_t nc,
                                  uint32_t half, gl_t a) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t k = j / half, i = j % half;
    if (k >= nc) return;
    const gl_t* src = in[k]; gl_t* dst = out[k];
    dst[i] = gl_add(src[2 * i], gl_mul(a, gl_sub(src[2 * i + 1], src[2 * i])));
}

template <typename FF, int NT, int MAXC>
__global__ void p3sg_msg_kernel(gl_t* const* cols, uint32_t nc, const gl_t* par,
                                gl_t* out, uint32_t half) {
    __shared__ gl_t sh[NT * 256];
    uint32_t tid = threadIdx.x;
    gl_t acc[NT];
    for (int t = 0; t < NT; t++) acc[t] = 0;
    gl_t cur[MAXC], dd[MAXC];
    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < half; i += gridDim.x * blockDim.x) {
        for (uint32_t k = 0; k < nc; k++) {
            gl_t lo = cols[k][2 * i], hi = cols[k][2 * i + 1];
            cur[k] = lo; dd[k] = gl_sub(hi, lo);
        }
        for (int t = 0; t < NT; t++) {
            acc[t] = gl_add(acc[t], FF::eval(cur, par));
            if (t < NT - 1) for (uint32_t k = 0; k < nc; k++) cur[k] = gl_add(cur[k], dd[k]);
        }
    }
    for (int t = 0; t < NT; t++) sh[t * 256 + tid] = acc[t];
    __syncthreads();
    for (uint32_t s = 128; s > 0; s >>= 1) {
        if (tid < s)
            for (int t = 0; t < NT; t++)
                sh[t * 256 + tid] = gl_add(sh[t * 256 + tid], sh[t * 256 + tid + s]);
        __syncthreads();
    }
    if (tid == 0) for (int t = 0; t < NT; t++) out[t * gridDim.x + blockIdx.x] = sh[t * 256];
}

// dcols: device column arrays (bound in place round by round; the caller frees
// what remains).  Returns the round challenges; appends messages (absorbed
// with `tag` exactly like the host provers).
template <typename FF, typename MsgT, int NT, int MAXC>
static inline std::vector<gl_t> sc_prove_gpu(fs::Transcript& tr, const char* tag,
        std::vector<gl_t*>& dcols, uint32_t N, const gl_t* par, uint32_t npar,
        std::vector<MsgT>& msgs) {
    static_assert(sizeof(MsgT) == (size_t)NT * sizeof(gl_t), "msg layout");
    uint32_t v = 0; while ((1u << v) < N) v++;
    const uint32_t nc = (uint32_t)dcols.size(), NB = 256;
    gl_t* d_par; cudaMallocAsync(&d_par, (size_t)(npar ? npar : 1) * 8, 0);
    if (npar) cudaMemcpy(d_par, par, (size_t)npar * 8, cudaMemcpyHostToDevice);
    gl_t** d_ptrs; cudaMallocAsync(&d_ptrs, (size_t)nc * sizeof(gl_t*), 0);
    gl_t** d_optrs; cudaMallocAsync(&d_optrs, (size_t)nc * sizeof(gl_t*), 0);
    gl_t* d_out; cudaMallocAsync(&d_out, (size_t)NT * NB * 8, 0);
    std::vector<gl_t> hout((size_t)NT * NB);
    std::vector<gl_t*> ncols(nc);
    std::vector<gl_t> r; r.reserve(v);
    uint32_t n = N;
    for (uint32_t rd = 0; rd < v; rd++) {
        uint32_t half = n / 2;
        cudaMemcpy(d_ptrs, dcols.data(), (size_t)nc * sizeof(gl_t*), cudaMemcpyHostToDevice);
        p3sg_msg_kernel<FF, NT, MAXC><<<NB, 256>>>(d_ptrs, nc, d_par, d_out, half);
        cudaMemcpy(hout.data(), d_out, (size_t)NT * NB * 8, cudaMemcpyDeviceToHost);
        gl_t s[NT]; MsgT m;
        for (int t = 0; t < NT; t++) {
            s[t] = 0;
            for (uint32_t b = 0; b < NB; b++) s[t] = gl_add(s[t], hout[(size_t)t * NB + b]);
        }
        memcpy(&m, s, sizeof m);
        msgs.push_back(m); tr.absorb(tag, &m, sizeof m);
        gl_t a = chal(tr); r.push_back(a);
        for (uint32_t k = 0; k < nc; k++) cudaMallocAsync(&ncols[k], (size_t)half * 8, 0);
        cudaMemcpy(d_optrs, ncols.data(), (size_t)nc * sizeof(gl_t*), cudaMemcpyHostToDevice);
        p3sg_bindn_kernel<<<((size_t)nc * half + 255) / 256, 256>>>(d_ptrs, d_optrs, nc, half, a);
        for (uint32_t k = 0; k < nc; k++) { cudaFreeAsync(dcols[k], 0); dcols[k] = ncols[k]; }
        n = half;
    }
    cudaFreeAsync(d_par, 0); cudaFreeAsync(d_ptrs, 0); cudaFreeAsync(d_optrs, 0);
    cudaFreeAsync(d_out, 0);
    return r;
}

} // namespace p3sg
