// Binius substrate step 2 (design doc section 21.2): additive NTT over
// GF(2^16) in the LCH novel polynomial basis, host + GPU batch.
//
// Basis: beta_0..beta_{m-1} spanning the evaluation domain (default: the F_2
// bit-basis beta_i = 1<<i).  Subspace polynomials W_0(x) = x, W_{i+1}(x) =
// W_i(x)*(W_i(x) + W_i(beta_i)); normalized What_i = W_i / W_i(beta_i) is
// F_2-LINEAR, vanishes on span(beta_0..beta_{i-1}) and What_i(beta_i) = 1.
// Novel basis polynomial Xhat_k(x) = prod_{i: bit i of k} What_i(x), with
// deg Xhat_k = k, so a message in coefficients d_0..d_{n-1} is a polynomial
// of degree < n; zero-padding the coefficient vector and running the NTT on a
// larger domain IS Reed-Solomon encoding (distance (N - n + 1)/N).
//
// Transform (in place, coefficients -> evaluations d[u] = f(sum u_i beta_i)):
//   for stage s = m-1 .. 0, block h < n>>(s+1), lane j < 2^s:
//     t = What_s(omega_h),  omega_h = sum_{i} h_i beta_{s+1+i}   (F_2-linear
//         => t is an XOR-fold of the precomputed What_s(beta_i) table)
//     lo = d[h<<(s+1) | j];  hi = d[h<<(s+1) | 1<<s | j];
//     lo ^= t*hi;  hi ^= lo;
// Butterflies at one stage are independent -> one GPU kernel per stage,
// twiddles precomputed per stage on host (n-1 total field elements).
// Correctness: p3_binius_ntt_test.cu checks the transform bitwise against
// brute-force Xhat evaluation with W_i computed BY DEFINITION (product over
// the whole subspace), plus RS distance teeth and GPU == host.
#pragma once
#include <cstdint>
#include <vector>
#include "p3_binius_field.cuh"

struct BfNtt {
    int m = 0;                      // log2(domain size)
    std::vector<bf16_t> beta;       // basis
    std::vector<bf16_t> twid;       // per-stage twiddles, stage s at off[s], len n>>(s+1)
    std::vector<uint32_t> off;      // m entries
};

// host precompute for a size-2^m domain on basis beta_i = 1<<i
static inline void bfntt_init(BfNtt& nt, int m) {
    nt.m = m;
    nt.beta.resize(m);
    for (int i = 0; i < m; i++) nt.beta[i] = (bf16_t)(1u << i);
    // wv[i] = W_s(beta_i), advanced stage by stage
    std::vector<uint32_t> wv(m);
    for (int i = 0; i < m; i++) wv[i] = nt.beta[i];
    // What_s(beta_i) for i > s (What_s(beta_s) = 1 by normalization)
    std::vector<std::vector<uint32_t>> what(m);
    for (int s = 0; s < m; s++) {
        uint32_t ninv = bf16_inv(wv[s]);
        what[s].assign(m, 0);
        for (int i = s + 1; i < m; i++) what[s][i] = bf16_mul(wv[i], ninv);
        for (int i = s + 1; i < m; i++) wv[i] = bf16_mul(wv[i], wv[i] ^ wv[s]);
    }
    // per-stage twiddle arrays: tw_s[h] = What_s(sum h_i beta_{s+1+i}), built
    // incrementally off the lowest set bit (F_2-linearity)
    nt.off.resize(m);
    uint32_t total = 0;
    for (int s = m - 1; s >= 0; s--) { nt.off[s] = total; total += 1u << (m - 1 - s); }
    nt.twid.assign(total, 0);
    for (int s = 0; s < m; s++) {
        bf16_t* tw = nt.twid.data() + nt.off[s];
        uint32_t cnt = 1u << (m - 1 - s);
        tw[0] = 0;
        for (uint32_t h = 1; h < cnt; h++) {
            int lb = __builtin_ctz(h);
            tw[h] = tw[h & (h - 1)] ^ (bf16_t)what[s][s + 1 + lb];
        }
    }
}

// in-place forward NTT of one row (host)
static inline void bfntt_fwd_host(const BfNtt& nt, bf16_t* d) {
    int m = nt.m;
    for (int s = m - 1; s >= 0; s--) {
        const bf16_t* tw = nt.twid.data() + nt.off[s];
        uint32_t half = 1u << s;
        for (uint32_t h = 0; h < (1u << (m - 1 - s)); h++) {
            uint32_t t = tw[h], base = h << (s + 1);
            for (uint32_t j = 0; j < half; j++) {
                uint32_t lo = d[base + j], hi = d[base + half + j];
                lo ^= bf16_mul(t, hi);
                d[base + j] = (bf16_t)lo;
                d[base + half + j] = (bf16_t)(lo ^ hi);
            }
        }
    }
}

// ---- GPU batch: R independent rows, row-major, in place ----
__global__ void bfntt_stage_kernel(bf16_t* rows, const bf16_t* tw, uint32_t n,
                                   uint32_t s, uint64_t nbf) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nbf) return;                       // nbf = rows * n/2 butterflies
    uint64_t r = idx / (n >> 1);
    uint32_t b = (uint32_t)(idx % (n >> 1));
    uint32_t h = b >> s, j = b & ((1u << s) - 1);
    bf16_t* d = rows + r * n + ((uint64_t)h << (s + 1));
    uint32_t t = tw[h];
    uint32_t lo = d[j], hi = d[j + (1u << s)];
    lo ^= bf16_mul(t, hi);
    d[j] = (bf16_t)lo;
    d[j + (1u << s)] = (bf16_t)(lo ^ hi);
}

struct BfNttDev {
    int m = 0;
    bf16_t* d_twid = nullptr;
    std::vector<uint32_t> off;
};

static inline void bfntt_to_device(const BfNtt& nt, BfNttDev& dv) {
    dv.m = nt.m; dv.off = nt.off;
    cudaMalloc(&dv.d_twid, nt.twid.size() * sizeof(bf16_t));
    cudaMemcpy(dv.d_twid, nt.twid.data(), nt.twid.size() * sizeof(bf16_t),
               cudaMemcpyHostToDevice);
}

// in-place forward NTT of `rows` rows of length 2^m at d_rows (device)
static inline void bfntt_fwd_gpu(const BfNttDev& dv, bf16_t* d_rows, uint32_t rows) {
    uint32_t n = 1u << dv.m;
    uint64_t nbf = (uint64_t)rows * (n >> 1);
    uint32_t blocks = (uint32_t)((nbf + 255) / 256);
    for (int s = dv.m - 1; s >= 0; s--)
        bfntt_stage_kernel<<<blocks, 256>>>(d_rows, dv.d_twid + dv.off[s], n, (uint32_t)s, nbf);
}
