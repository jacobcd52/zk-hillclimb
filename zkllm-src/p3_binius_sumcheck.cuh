// Binius substrate step 4 (design doc section 21.4): multilinear sumcheck
// over the binary tower.  Proves  sum_{x in {0,1}^l} C(W_0(x),..,W_{K-1}(x))
// == claim  for a degree-D composition C of K multilinear columns, all
// arithmetic in T_128 = GF(2^128).
//
// Characteristic-2 notes (this is where a prime-field port would silently
// break): "sum over the hypercube" means XOR of field elements; the round
// polynomial is sent as evaluations at the D+1 tower points {0,1,2,..,D}
// (distinct field elements -- bitstrings, NOT integers mod p) and the
// verifier interpolates with Lagrange weights whose denominators are
// products of XOR-differences of those points; a column's value on the
// z-extension of a pair is V0 + z*(V0+V1) with z a small subfield element
// (bf128_smul16 fast path).  Zerocheck = include the eq(rz,.) table as
// column 0 and claim 0; the verifier recomputes eq(rz, zeta) itself.
//
// The verifier half only replays rounds and returns the expected integrand
// value E at the final point zeta; the CALLER must (a) check E ==
// C(finals), (b) validate finals against its own eq computation / PCS
// openings.  Teeth in p3_binius_sumcheck_test.cu.
#pragma once
#include <cstdint>
#include <vector>
#include "fs_transcript.hpp"
#include "p3_binius_pcs.cuh"      // bf_eq_table, bf_chal128

typedef bf128_t (*BfConstraintFn)(const bf128_t* w, const void* ctx);

struct BfScProof {
    int l = 0, D = 0, K = 0;
    std::vector<bf128_t> rounds;    // l * (D+1) evaluations
    std::vector<bf128_t> finals;    // K column values at the final point
};

// prove; cols are consumed (folded in place).  Returns the final point.
static inline void bf_sumcheck_prove(int l, int K, std::vector<std::vector<bf128_t>>& cols,
                                     int D, BfConstraintFn C, const void* ctx,
                                     fs::Transcript& tr, BfScProof& pf,
                                     std::vector<bf128_t>& zeta) {
    pf.l = l; pf.D = D; pf.K = K;
    pf.rounds.assign((size_t)l * (D + 1), bf128_zero());
    zeta.assign(l, bf128_zero());
    for (int s = 0; s < l; s++) {
        size_t half = (size_t)1 << (l - 1 - s);
        bf128_t* rp = pf.rounds.data() + (size_t)s * (D + 1);
        #pragma omp parallel
        {
            std::vector<bf128_t> acc(D + 1, bf128_zero());
            std::vector<bf128_t> w(K);
            #pragma omp for schedule(static)
            for (int64_t y = 0; y < (int64_t)half; y++) {
                for (int z = 0; z <= D; z++) {
                    for (int k = 0; k < K; k++) {
                        bf128_t v0 = cols[k][2 * y], v1 = cols[k][2 * y + 1];
                        w[k] = bf128_add(v0, bf128_smul16(bf128_add(v0, v1), (uint32_t)z));
                    }
                    acc[z] = bf128_add(acc[z], C(w.data(), ctx));
                }
            }
            #pragma omp critical
            for (int z = 0; z <= D; z++) rp[z] = bf128_add(rp[z], acc[z]);
        }
        tr.absorb("bfsc-round", rp, (D + 1) * sizeof(bf128_t));
        bf128_t zs = bf_chal128(tr);
        zeta[s] = zs;
        // fold in place; ascending y within one column is safe (position y is
        // last read at iteration y/2 <= y), so parallelize across columns only
        #pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < K; k++)
            for (size_t y = 0; y < half; y++) {
                bf128_t v0 = cols[k][2 * y], v1 = cols[k][2 * y + 1];
                cols[k][y] = bf128_add(v0, bf128_mul(zs, bf128_add(v0, v1)));
            }
    }
    pf.finals.resize(K);
    for (int k = 0; k < K; k++) pf.finals[k] = cols[k][0];
    tr.absorb("bfsc-finals", pf.finals.data(), K * sizeof(bf128_t));
}

// ---- GPU prover (design doc section 21.6 item 1) --------------------------
// Same protocol, same transcript, byte-identical proof: the round polynomial
// is an XOR of per-row contributions (order-independent), and the fold is a
// per-element exact field map, so GPU and host provers agree bitwise (teeth in
// p3_binius_sumcheck_test.cu / p3_binius_e2e_test.cu).  The constraint is a
// functor type CF with static constexpr int K, D and a __device__
// operator()(const bf128_t* w) const, passed by value into the kernels.
// Columns live in one K x 2^l device buffer (stride 2^l); the fold ping-pongs
// between two buffers because the in-place host order (read 2y before write y,
// ascending) has no parallel counterpart.

struct BfScDev {
    bf128_t *a = nullptr, *b = nullptr;   // ping-pong column buffers, K * n each
    bf128_t *part = nullptr;              // per-block round partials
    int K = 0, l = 0;
    size_t n = 0;                         // 2^l = column stride
};
#define BFSC_MAXBLK 512
static inline void bfsc_dev_alloc(BfScDev& d, int K, int l) {
    d.K = K; d.l = l; d.n = (size_t)1 << l;
    cudaMalloc(&d.a, (size_t)K * d.n * sizeof(bf128_t));
    cudaMalloc(&d.b, (size_t)K * d.n * sizeof(bf128_t));
    cudaMalloc(&d.part, (size_t)BFSC_MAXBLK * 8 * sizeof(bf128_t));   // D <= 7
}
static inline void bfsc_dev_free(BfScDev& d) {
    cudaFree(d.a); cudaFree(d.b); cudaFree(d.part);
    d.a = d.b = d.part = nullptr;
}

// eq(r, .) table built in place on device -- the bf_eq_table recurrence level
// by level (same multiplication tree, so identical field values)
static __global__ void bfsc_eq_level_kernel(bf128_t* out, bf128_t r, size_t half) {
    size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (x >= half) return;
    bf128_t e = out[x];
    bf128_t hi = bf128_mul(e, r);
    out[x | half] = hi;
    out[x] = bf128_add(e, hi);
}
static inline void bfsc_dev_eq(bf128_t* d_dst, const bf128_t* r, int k) {
    bf128_t one = bf128_one();
    cudaMemcpy(d_dst, &one, sizeof(one), cudaMemcpyHostToDevice);
    for (int t = 0; t < k; t++) {
        size_t half = (size_t)1 << t;
        bfsc_eq_level_kernel<<<(uint32_t)((half + 255) / 256), 256>>>(d_dst, r[t], half);
    }
}
// expand a 0/1 byte witness into T_128 columns (contiguous cols, stride n)
static __global__ void bfsc_bits_kernel(const uint8_t* bits, bf128_t* dst, size_t total) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < total) dst[i] = (bits[i] & 1) ? bf128_one() : bf128_zero();
}

template <class CF>
static __global__ void bfsc_round_kernel(const bf128_t* cols, size_t stride, size_t half,
                                         CF cf, bf128_t* part) {
    bf128_t acc[CF::D + 1];
    for (int z = 0; z <= CF::D; z++) acc[z] = bf128_zero();
    bf128_t a0[CF::K], ad[CF::K], w[CF::K];
    for (size_t y = blockIdx.x * (size_t)blockDim.x + threadIdx.x; y < half;
         y += (size_t)gridDim.x * blockDim.x) {
        for (int k = 0; k < CF::K; k++) {
            bf128_t v0 = cols[(size_t)k * stride + 2 * y];
            bf128_t v1 = cols[(size_t)k * stride + 2 * y + 1];
            a0[k] = v0; ad[k] = bf128_add(v0, v1);
        }
        for (int z = 0; z <= CF::D; z++) {
            for (int k = 0; k < CF::K; k++)
                w[k] = z ? bf128_add(a0[k], bf128_smul16(ad[k], (uint32_t)z)) : a0[k];
            acc[z] = bf128_add(acc[z], cf(w));
        }
    }
    __shared__ bf128_t sm[256];
    for (int z = 0; z <= CF::D; z++) {
        sm[threadIdx.x] = acc[z];
        __syncthreads();
        for (int s = blockDim.x / 2; s; s >>= 1) {
            if ((int)threadIdx.x < s)
                sm[threadIdx.x] = bf128_add(sm[threadIdx.x], sm[threadIdx.x + s]);
            __syncthreads();
        }
        if (!threadIdx.x) part[(size_t)blockIdx.x * (CF::D + 1) + z] = sm[0];
        __syncthreads();
    }
}
static __global__ void bfsc_fold_kernel(const bf128_t* src, bf128_t* dst, size_t stride,
                                        int K, size_t half, bf128_t zs) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= (size_t)K * half) return;
    size_t k = idx / half, y = idx - k * half;
    bf128_t v0 = src[k * stride + 2 * y], v1 = src[k * stride + 2 * y + 1];
    dst[k * stride + y] = bf128_add(v0, bf128_mul(zs, bf128_add(v0, v1)));
}

// prove on device; dv.a holds the K columns on entry (consumed).
template <class CF>
static inline void bf_sumcheck_prove_gpu(BfScDev& dv, const CF& cf,
                                         fs::Transcript& tr, BfScProof& pf,
                                         std::vector<bf128_t>& zeta) {
    const int D = CF::D, K = CF::K;
    int l = dv.l;
    pf.l = l; pf.D = D; pf.K = K;
    pf.rounds.assign((size_t)l * (D + 1), bf128_zero());
    zeta.assign(l, bf128_zero());
    bf128_t *cur = dv.a, *oth = dv.b;
    std::vector<bf128_t> part;
    for (int s = 0; s < l; s++) {
        size_t half = (size_t)1 << (l - 1 - s);
        uint32_t nb = (uint32_t)((half + 255) / 256);
        if (nb > BFSC_MAXBLK) nb = BFSC_MAXBLK;
        bfsc_round_kernel<CF><<<nb, 256>>>(cur, dv.n, half, cf, dv.part);
        part.resize((size_t)nb * (D + 1));
        cudaMemcpy(part.data(), dv.part, part.size() * sizeof(bf128_t), cudaMemcpyDeviceToHost);
        bf128_t* rp = pf.rounds.data() + (size_t)s * (D + 1);
        for (uint32_t b = 0; b < nb; b++)
            for (int z = 0; z <= D; z++)
                rp[z] = bf128_add(rp[z], part[(size_t)b * (D + 1) + z]);
        tr.absorb("bfsc-round", rp, (D + 1) * sizeof(bf128_t));
        bf128_t zs = bf_chal128(tr);
        zeta[s] = zs;
        size_t tot = (size_t)K * half;
        bfsc_fold_kernel<<<(uint32_t)((tot + 255) / 256), 256>>>(cur, oth, dv.n, K, half, zs);
        bf128_t* t = cur; cur = oth; oth = t;
    }
    pf.finals.resize(K);
    cudaMemcpy2D(pf.finals.data(), sizeof(bf128_t), cur, dv.n * sizeof(bf128_t),
                 sizeof(bf128_t), K, cudaMemcpyDeviceToHost);
    tr.absorb("bfsc-finals", pf.finals.data(), K * sizeof(bf128_t));
}

// replay rounds; returns false on a broken chain.  On success *expected is
// the required integrand value at zeta (caller checks against C(finals)).
static inline bool bf_sumcheck_verify(const BfScProof& pf, bf128_t claim,
                                      fs::Transcript& tr, std::vector<bf128_t>& zeta,
                                      bf128_t* expected) {
    int l = pf.l, D = pf.D;
    if ((int)pf.rounds.size() != l * (D + 1) || (int)pf.finals.size() != pf.K) return false;
    // Lagrange denominators at nodes {0..D}: d_z = prod_{w != z} (z ^ w)
    std::vector<bf128_t> dinv(D + 1);
    for (int z = 0; z <= D; z++) {
        bf128_t d = bf128_one();
        for (int w = 0; w <= D; w++)
            if (w != z) d = bf128_mul(d, bf128_from64((uint64_t)(z ^ w)));
        dinv[z] = bf128_inv(d);
    }
    zeta.assign(l, bf128_zero());
    for (int s = 0; s < l; s++) {
        const bf128_t* rp = pf.rounds.data() + (size_t)s * (D + 1);
        if (!bf128_eq(bf128_add(rp[0], rp[1]), claim)) return false;
        tr.absorb("bfsc-round", rp, (D + 1) * sizeof(bf128_t));
        bf128_t zs = bf_chal128(tr);
        zeta[s] = zs;
        // claim <- p(zs) by Lagrange at nodes {0..D}
        bf128_t nxt = bf128_zero();
        for (int z = 0; z <= D; z++) {
            bf128_t num = bf128_one();
            for (int w = 0; w <= D; w++)
                if (w != z) num = bf128_mul(num, bf128_add(zs, bf128_from64((uint64_t)w)));
            nxt = bf128_add(nxt, bf128_mul(rp[z], bf128_mul(num, dinv[z])));
        }
        claim = nxt;
    }
    tr.absorb("bfsc-finals", pf.finals.data(), pf.K * sizeof(bf128_t));
    *expected = claim;
    return true;
}
