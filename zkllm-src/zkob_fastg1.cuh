// zkob_fastg1.cuh — Stage C2 fast-helpers for the per-row homomorphic G1
// host loops (TRANSPORT_REBUILD_DESIGN §6 Stage C contingency: "fast-helpers
// on round-check host loops before declaring"). The measured Stage-C2 verify
// hot spots are the per-row 1-thread loops:
//   - zkob_rescale affine link  g1_eq(X[j], sf*Xr[j] + rem[j])   ~1.70 s/call
//   - zkob_glu / zkob_softmax / zkob_softmax8 comb loops
//     comb[j] = G[j] + r*S[j]                                    ~4.4 s/call
//
// NO new G1 kernel shape (the -dlto rule): these helpers launch the SAME
// 1-thread k_g1_mul / k_g1_addsub kernels (vrf_common.cuh) the sequential
// h_mul/h_add/g1_eq wrappers use — same compiled device code, same
// per-element inputs and argument order, bit-identical per-element results —
// but CONCURRENTLY across a CUDA stream pool instead of one synchronous
// malloc+launch+memcpy round-trip per element. Pure scheduling, no protocol
// or algebra change.
//
// Validation, house style:
//   - per-call cross-check: two sample rows are recomputed with the original
//     sequential helpers and compared BYTE-exact (same kernel => identical
//     Jacobian coordinates); mismatch throws (fail-closed).
//   - ZKOB_SLOW_G1LOOP=1 forces the original sequential loop outright.
#ifndef ZKOB_FASTG1_CUH
#define ZKOB_FASTG1_CUH

#include "vrf_common.cuh"
#include <cstdlib>
#include <stdexcept>
#include <vector>

static bool hb_slow_on() {
    return getenv("ZKOB_SLOW_G1LOOP") != nullptr;
}

static const int HB_NSTREAMS = 128;   // Ada concurrent-kernel limit

struct HbStreamPool {
    cudaStream_t s[HB_NSTREAMS];
    HbStreamPool() { for (int i = 0; i < HB_NSTREAMS; i++) cudaStreamCreate(&s[i]); }
};
static cudaStream_t hb_stream(uint j) {
    static HbStreamPool pool;   // lazy: first use is after CUDA context exists
    return pool.s[j % HB_NSTREAMS];
}

static bool hb_bytes_eq(const G1Jacobian_t& a, const G1Jacobian_t& b) {
    const uint32_t* pa = (const uint32_t*)&a;
    const uint32_t* pb = (const uint32_t*)&b;
    for (size_t i = 0; i < sizeof(G1Jacobian_t) / 4; i++)
        if (pa[i] != pb[i]) return false;
    return true;
}

// out[j] = mul_first ? h_add(h_mul(a[j], r), b[j])
//                    : h_add(b[j], h_mul(a[j], r))
// (exact h_add argument order of the loop being replaced)
static void hb_addmul(const std::vector<G1Jacobian_t>& a, Fr_t r,
                      const std::vector<G1Jacobian_t>& b, bool mul_first,
                      std::vector<G1Jacobian_t>& out) {
    const uint n = (uint)a.size();
    out.resize(n);
    if (hb_slow_on()) {
        for (uint j = 0; j < n; j++)
            out[j] = mul_first ? h_add(h_mul(a[j], r), b[j])
                               : h_add(b[j], h_mul(a[j], r));
        return;
    }
    G1Jacobian_t* d;
    cudaMalloc(&d, n * sizeof(G1Jacobian_t));
    for (uint j = 0; j < n; j++)
        k_g1_mul<<<1, 1, 0, hb_stream(j)>>>(a[j], r, d + j);
    cudaDeviceSynchronize();
    std::vector<G1Jacobian_t> m(n);
    cudaMemcpy(m.data(), d, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    for (uint j = 0; j < n; j++) {
        if (mul_first) k_g1_addsub<<<1, 1, 0, hb_stream(j)>>>(m[j], b[j], 0, d + j);
        else           k_g1_addsub<<<1, 1, 0, hb_stream(j)>>>(b[j], m[j], 0, d + j);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cudaFree(d);
    // cross-check two sample rows against the proven sequential helpers
    for (uint j : {0u, n - 1}) {
        G1Jacobian_t want = mul_first ? h_add(h_mul(a[j], r), b[j])
                                      : h_add(b[j], h_mul(a[j], r));
        if (!hb_bytes_eq(out[j], want))
            throw std::runtime_error("hb_addmul cross-check failed (fastg1)");
    }
}

// per-row n_bad count of !g1_eq(x[j], y[j]) — exact g1_eq semantics
// (x - y must fold to the point at infinity, z == 0)
static uint hb_neq_count(const std::vector<G1Jacobian_t>& x,
                         const std::vector<G1Jacobian_t>& y) {
    const uint n = (uint)x.size();
    if (hb_slow_on()) {
        uint bad = 0;
        for (uint j = 0; j < n; j++) if (!g1_eq(x[j], y[j])) bad++;
        return bad;
    }
    G1Jacobian_t* d;
    cudaMalloc(&d, n * sizeof(G1Jacobian_t));
    for (uint j = 0; j < n; j++)
        k_g1_addsub<<<1, 1, 0, hb_stream(j)>>>(x[j], y[j], 1, d + j);
    cudaDeviceSynchronize();
    std::vector<G1Jacobian_t> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cudaFree(d);
    uint bad = 0;
    for (uint j = 0; j < n; j++) {
        bool z0 = true;
        for (int i = 0; i < 12; i++) if (h[j].z.val[i] != 0) { z0 = false; break; }
        if (!z0) bad++;
    }
    // cross-check two sample rows against the proven sequential helper
    for (uint j : {0u, n - 1})
        if (g1_eq(x[j], y[j]) != ([&]{ bool z0 = true;
                for (int i = 0; i < 12; i++) if (h[j].z.val[i] != 0) z0 = false;
                return z0; }()))
            throw std::runtime_error("hb_neq_count cross-check failed (fastg1)");
    return bad;
}

#endif
