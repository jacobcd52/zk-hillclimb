// COORDINATOR-BUILT (do not let submission agents edit).
// Shared verification library for the real zkprove/zkverify drivers.
// Everything here was pinned at toy scale against the REAL upstream provers:
//   vrf_toy_open.cu   (§7  commitment fold chain, raw-limb C_0, -dlto workaround)
//   vrf_toy_matmul.cu (§8  zkip sumcheck, Lagrange-3, opening glue)
//   vrf_toy_lookup.cu (§9  logUp lookup, Lagrange-4, recomputable terminals)
//   vrf_toy_ipa.cu    (§10 me_open steering attack -> Fiat-Shamir IPA)
// All Fr arithmetic is in the kernels' "mont limbs as integers" view.
//
// -dlto MISCOMPILATION rules (vrf_toy_debug2.cu): a kernel with TWO branches
// that each call G1Jacobian_mul miscompiles. Straight-line two-mul is CLEAN
// (probed in vrf_toy_ipa.cu and re-probed by drivers' selftest). Therefore:
// fold kernels = single branch + identity padding; the IPA g-fold uses the
// (probed-clean) straight-line two-mul shape and is cross-checked in selftest.
#pragma once
#include "commitment.cuh"
#include "proof.cuh"
#include "polynomial.cuh"
#include "fs_transcript.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// ---- integer-view constants ----
static const Fr_t F_ZERO = {0, 0, 0, 0, 0, 0, 0, 0};
static const Fr_t F_ONE  = {1, 0, 0, 0, 0, 0, 0, 0};
static const Fr_t F_TWO  = {2, 0, 0, 0, 0, 0, 0, 0};
// INV2 = (r+1)/2; self-check with vrf_selfcheck() before use.
static const Fr_t F_INV2 = {2147483649u, 2147483647u, 2147429887u, 2849952257u,
                            80800770u, 429714436u, 2496577188u, 972477353u};

// ---- 1-thread device helpers (host-side field/point ops, bit-identical) ----
KERNEL void k_scalar_op(Fr_t a, Fr_t b, int op, GLOBAL Fr_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    if (op == 0) *out = blstrs__scalar__Scalar_add(a, b);
    if (op == 1) *out = blstrs__scalar__Scalar_sub(a, b);
    if (op == 2) *out = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a, b)); // plain a*b (int view)
}
KERNEL void k_g1_mul(G1Jacobian_t g, Fr_t c, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = G1Jacobian_mul(g, c);
}
KERNEL void k_g1_addsub(G1Jacobian_t a, G1Jacobian_t b, int sub, GLOBAL G1Jacobian_t* out) {
    if (GET_GLOBAL_ID() > 0) return;
    *out = blstrs__g1__G1Affine_add(a, sub ? G1Jacobian_minus(b) : b);
}
// pair-fold, scalar/ME orientation: c' = c0 + u*(c1 - c0). SINGLE branch.
KERNEL void k_com_me(GLOBAL G1Jacobian_t* c, Fr_t u, GLOBAL G1Jacobian_t* cout, uint new_size) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    cout[gid] = blstrs__g1__G1Affine_add(c[2*gid],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(c[2*gid+1], G1Jacobian_minus(c[2*gid])), u));
}
// pair-fold, me_open generator orientation: g' = g1 + u*(g0 - g1). SINGLE branch.
KERNEL void k_gen_fold(GLOBAL G1Jacobian_t* g, Fr_t u, GLOBAL G1Jacobian_t* gout, uint new_size) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;
    gout[gid] = blstrs__g1__G1Affine_add(g[2*gid+1],
        G1Jacobian_mul(blstrs__g1__G1Affine_add(g[2*gid], G1Jacobian_minus(g[2*gid+1])), u));
}
// IPA front/back-half folds (sizes always pow2 inside the IPA)
KERNEL void k_fr_fold2(GLOBAL Fr_t* a, Fr_t x, Fr_t xi, GLOBAL Fr_t* out, uint half) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= half) return;
    out[gid] = blstrs__scalar__Scalar_add(
        blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(x, a[gid])),
        blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(xi, a[gid + half])));
}
// straight-line two-mul (probed clean; re-probed by driver selftest)
KERNEL void k_g1_fold2(GLOBAL G1Jacobian_t* g, Fr_t xi, Fr_t x, GLOBAL G1Jacobian_t* out, uint half) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= half) return;
    out[gid] = blstrs__g1__G1Affine_add(G1Jacobian_mul(g[gid], xi),
                                        G1Jacobian_mul(g[gid + half], x));
}
// elementwise scale + pairwise-add reduce (MSM building blocks, pow2 sizes)
KERNEL void k_g1_scale(GLOBAL G1Jacobian_t* g, GLOBAL Fr_t* a, GLOBAL G1Jacobian_t* out, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    out[gid] = G1Jacobian_mul(g[gid], a[gid]);
}
KERNEL void k_g1_add_pairs(GLOBAL G1Jacobian_t* src, GLOBAL G1Jacobian_t* dst, uint half) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= half) return;
    dst[gid] = blstrs__g1__G1Affine_add(src[gid], src[gid + half]);
}
KERNEL void k_fr_emul(GLOBAL Fr_t* a, GLOBAL Fr_t* b, GLOBAL Fr_t* out, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    out[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a[gid], b[gid]));
}
KERNEL void k_fr_add_pairs(GLOBAL Fr_t* src, GLOBAL Fr_t* dst, uint half) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= half) return;
    dst[gid] = blstrs__scalar__Scalar_add(src[gid], src[gid + half]);
}

// ---- host wrappers ----
static Fr_t h_scalar(Fr_t a, Fr_t b, int op) {
    Fr_t *d, h; cudaMalloc(&d, sizeof(Fr_t));
    k_scalar_op<<<1,1>>>(a, b, op, d);
    cudaMemcpy(&h, d, sizeof(Fr_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
static G1Jacobian_t h_mul(G1Jacobian_t g, Fr_t c) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_mul<<<1,1>>>(g, c, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
static G1Jacobian_t h_add(G1Jacobian_t a, G1Jacobian_t b) {
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_addsub<<<1,1>>>(a, b, 0, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d); return h;
}
static bool g1_eq(G1Jacobian_t a, G1Jacobian_t b) {
    // equal iff a - b == infinity (z == 0); add formula yields z3=0 for P+(-P)
    G1Jacobian_t *d, h; cudaMalloc(&d, sizeof(G1Jacobian_t));
    k_g1_addsub<<<1,1>>>(a, b, 1, d);
    cudaMemcpy(&h, d, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost); cudaFree(d);
    for (int i = 0; i < 12; i++) if (h.z.val[i] != 0) return false;
    return true;
}
static bool fr_eq(const Fr_t& a, const Fr_t& b) {
    for (int i = 0; i < 8; i++) if (a.val[i] != b.val[i]) return false;
    return true;
}

// Lagrange-3 evaluation from {p(0), p(1), p(2)} at u (degree-2 sumcheck rounds):
// p(u) = p0*(u-1)(u-2)/2 + p1*u(2-u) + p2*u(u-1)/2
static Fr_t lagrange3(const Fr_t& p0, const Fr_t& p1, const Fr_t& p2, const Fr_t& u) {
    Fr_t um1 = h_scalar(u, F_ONE, 1);
    Fr_t um2 = h_scalar(u, F_TWO, 1);
    Fr_t l0 = h_scalar(h_scalar(um1, um2, 2), F_INV2, 2);
    Fr_t l1 = h_scalar(u, h_scalar(F_TWO, u, 1), 2);
    Fr_t l2 = h_scalar(h_scalar(u, um1, 2), F_INV2, 2);
    Fr_t acc = h_scalar(p0, l0, 2);
    acc = h_scalar(acc, h_scalar(p1, l1, 2), 0);
    acc = h_scalar(acc, h_scalar(p2, l2, 2), 0);
    return acc;
}

// generic G1 fold chain with identity-padding for odd levels.
// orientation 0 = scalar/ME (k_com_me), 1 = me_open generators (k_gen_fold).
static G1Jacobian_t fold_chain(const G1Jacobian_t* dev_src, uint size,
                               const std::vector<Fr_t>& us, int orientation) {
    uint cap = size + 1;
    G1Jacobian_t* d_a; cudaMalloc(&d_a, cap * sizeof(G1Jacobian_t));
    G1Jacobian_t* d_b; cudaMalloc(&d_b, cap * sizeof(G1Jacobian_t));
    cudaMemcpy(d_a, dev_src, size * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    uint sz = size;
    for (auto& u : us) {
        if (sz % 2) { cudaMemset(d_a + sz, 0, sizeof(G1Jacobian_t)); sz += 1; }
        uint nsz = sz / 2;
        if (orientation == 0) k_com_me<<<(nsz + 31) / 32, 32>>>(d_a, u, d_b, nsz);
        else                  k_gen_fold<<<(nsz + 31) / 32, 32>>>(d_a, u, d_b, nsz);
        cudaDeviceSynchronize();
        std::swap(d_a, d_b); sz = nsz;
    }
    G1Jacobian_t out;
    cudaMemcpy(&out, d_a, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b);
    return out;
}

// device MSM <g, a> over pow2 n: elementwise scale then pairwise-add reduce
static G1Jacobian_t dev_msm(const G1Jacobian_t* d_g, const Fr_t* d_a, uint n) {
    G1Jacobian_t *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(G1Jacobian_t));
    cudaMalloc(&d_y, (n / 2 + 1) * sizeof(G1Jacobian_t));
    k_g1_scale<<<(n + 63) / 64, 64>>>(const_cast<G1Jacobian_t*>(d_g), const_cast<Fr_t*>(d_a), d_x, n);
    cudaDeviceSynchronize();
    uint sz = n;
    while (sz > 1) {
        uint half = sz / 2;
        k_g1_add_pairs<<<(half + 63) / 64, 64>>>(d_x, d_y, half);
        cudaDeviceSynchronize();
        std::swap(d_x, d_y); sz = half;
    }
    G1Jacobian_t out;
    cudaMemcpy(&out, d_x, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y);
    return out;
}
// device inner product <a, b> over pow2 n (integer view)
static Fr_t dev_ip(const Fr_t* d_a, const Fr_t* d_b, uint n) {
    Fr_t *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(Fr_t));
    cudaMalloc(&d_y, (n / 2 + 1) * sizeof(Fr_t));
    k_fr_emul<<<(n + 63) / 64, 64>>>(const_cast<Fr_t*>(d_a), const_cast<Fr_t*>(d_b), d_x, n);
    cudaDeviceSynchronize();
    uint sz = n;
    while (sz > 1) {
        uint half = sz / 2;
        k_fr_add_pairs<<<(half + 63) / 64, 64>>>(d_x, d_y, half);
        cudaDeviceSynchronize();
        std::swap(d_x, d_y); sz = half;
    }
    Fr_t out;
    cudaMemcpy(&out, d_x, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y);
    return out;
}

// ME weight vector of u: b_k = prod_i (k>>i & 1 ? u[i] : 1 - u[i]), size 2^|u|.
// Bit order pinned in vrf_toy_ipa.cu: <t_row, b> == me_open eval bit-exact.
static std::vector<Fr_t> me_weights(const std::vector<Fr_t>& u) {
    uint L = u.size(), n = 1u << L;
    std::vector<Fr_t> b(n);
    for (uint k = 0; k < n; k++) {
        Fr_t w = F_ONE;
        for (uint i = 0; i < L; i++)
            w = h_scalar(w, ((k >> i) & 1) ? u[i] : h_scalar(F_ONE, u[i], 1), 2);
        b[k] = w;
    }
    return b;
}

// ---- Fiat-Shamir glue ----
static void absorb_fr(fs::Transcript& tr, const std::string& label, const Fr_t& x) {
    tr.absorb(label, &x, sizeof(Fr_t));
}
static void absorb_g1(fs::Transcript& tr, const std::string& label, const G1Jacobian_t& p) {
    tr.absorb(label, &p, sizeof(G1Jacobian_t));
}
static void absorb_u32(fs::Transcript& tr, const std::string& label, uint32_t v) {
    tr.absorb(label, &v, sizeof(v));
}
static void absorb_g1_tensor(fs::Transcript& tr, const std::string& label,
                             const G1TensorJacobian& t) {
    std::vector<G1Jacobian_t> h(t.size);
    cudaMemcpy(h.data(), t.gpu_data, t.size * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    tr.absorb(label, h.data(), h.size() * sizeof(G1Jacobian_t));
}
// challenge -> Fr: 8 LE uint32 limbs, top % 1944954707 (random_vec distribution)
static Fr_t fs_challenge_fr(fs::Transcript& tr) {
    while (true) {
        uint8_t buf[32]; tr.challenge_bytes(buf);
        Fr_t x;
        for (int i = 0; i < 8; i++)
            x.val[i] = uint32_t(buf[4*i]) | (uint32_t(buf[4*i+1]) << 8)
                     | (uint32_t(buf[4*i+2]) << 16) | (uint32_t(buf[4*i+3]) << 24);
        x.val[7] %= 1944954707;
        for (int i = 0; i < 8; i++) if (x.val[i]) return x;
    }
}
static std::vector<Fr_t> fs_challenge_vec(fs::Transcript& tr, uint n) {
    std::vector<Fr_t> out(n);
    for (uint i = 0; i < n; i++) out[i] = fs_challenge_fr(tr);
    return out;
}

// ---- serialization (POD vectors with magic+count headers) ----
template <typename T>
static void write_pod_vec(FILE* f, const std::vector<T>& v) {
    uint32_t n = (uint32_t)v.size();
    fwrite(&n, sizeof(n), 1, f);
    if (n) fwrite(v.data(), sizeof(T), n, f);
}
template <typename T>
static std::vector<T> read_pod_vec(FILE* f) {
    uint32_t n = 0;
    if (fread(&n, sizeof(n), 1, f) != 1) throw std::runtime_error("read_pod_vec: header");
    std::vector<T> v(n);
    if (n && fread(v.data(), sizeof(T), n, f) != n) throw std::runtime_error("read_pod_vec: body");
    return v;
}
static FILE* open_or_die(const std::string& path, const char* mode) {
    FILE* f = fopen(path.c_str(), mode);
    if (!f) throw std::runtime_error("cannot open " + path);
    return f;
}

// ---- Fiat-Shamir IPA (pinned in vrf_toy_ipa.cu) ----
// Statement: P0 = C0 + eval*Q with C0 = <g, a>, eval = <a, b>, b public.
// Round: L = <a_lo, g_hi> + <a_lo, b_hi>Q ; R = <a_hi, g_lo> + <a_hi, b_lo>Q
//        absorb L,R -> x ; a' = x a_lo + x^-1 a_hi ; b' = x^-1 b_lo + x b_hi ;
//        g' = x^-1 g_lo + x g_hi ; P' = x^2 L + P + x^-2 R
// Final: a_f ; check P_L == a_f g_f + (a_f b_f) Q with g_f via s-vector MSM,
//        s_i = prod_r (bit_{MSB-r}(i) ? x_r : x_r^-1).
struct IpaProof {
    std::vector<G1Jacobian_t> L, R;
    Fr_t a_final;
};
static void write_ipa(const std::string& path, const IpaProof& pf) {
    FILE* f = open_or_die(path, "wb");
    write_pod_vec(f, pf.L); write_pod_vec(f, pf.R);
    fwrite(&pf.a_final, sizeof(Fr_t), 1, f);
    fclose(f);
}
static IpaProof read_ipa(const std::string& path) {
    FILE* f = open_or_die(path, "rb");
    IpaProof pf;
    pf.L = read_pod_vec<G1Jacobian_t>(f);
    pf.R = read_pod_vec<G1Jacobian_t>(f);
    if (fread(&pf.a_final, sizeof(Fr_t), 1, f) != 1) throw std::runtime_error("read_ipa: a_final");
    fclose(f);
    return pf;
}

// prove: a = within-row vector (device, size n pow2), b = host ME weights,
// g = device generators (size n). Continues the caller's transcript.
static IpaProof ipa_prove(const Fr_t* d_a_in, const std::vector<Fr_t>& b_in,
                          const G1Jacobian_t* d_g_in, const G1Jacobian_t& Q,
                          uint n, fs::Transcript& tr) {
    if (n & (n - 1)) throw std::runtime_error("ipa_prove: n not a power of two");
    Fr_t *d_a, *d_a2; G1Jacobian_t *d_g, *d_g2; Fr_t *d_b, *d_b2;
    cudaMalloc(&d_a, n * sizeof(Fr_t));   cudaMalloc(&d_a2, (n/2+1) * sizeof(Fr_t));
    cudaMalloc(&d_b, n * sizeof(Fr_t));   cudaMalloc(&d_b2, (n/2+1) * sizeof(Fr_t));
    cudaMalloc(&d_g, n * sizeof(G1Jacobian_t)); cudaMalloc(&d_g2, (n/2+1) * sizeof(G1Jacobian_t));
    cudaMemcpy(d_a, d_a_in, n * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_b, b_in.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, d_g_in, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);
    IpaProof pf;
    uint sz = n;
    while (sz > 1) {
        uint half = sz / 2;
        G1Jacobian_t Lp = h_add(dev_msm(d_g + half, d_a, half),
                                h_mul(Q, dev_ip(d_a, d_b + half, half)));
        G1Jacobian_t Rp = h_add(dev_msm(d_g, d_a + half, half),
                                h_mul(Q, dev_ip(d_a + half, d_b, half)));
        absorb_g1(tr, "L", Lp); absorb_g1(tr, "R", Rp);
        Fr_t x = fs_challenge_fr(tr), xi = inv(x);
        k_fr_fold2<<<(half + 63) / 64, 64>>>(d_a, x, xi, d_a2, half);
        k_fr_fold2<<<(half + 63) / 64, 64>>>(d_b, xi, x, d_b2, half);
        k_g1_fold2<<<(half + 63) / 64, 64>>>(d_g, xi, x, d_g2, half);
        cudaDeviceSynchronize();
        std::swap(d_a, d_a2); std::swap(d_b, d_b2); std::swap(d_g, d_g2);
        pf.L.push_back(Lp); pf.R.push_back(Rp);
        sz = half;
    }
    cudaMemcpy(&pf.a_final, d_a, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_a2); cudaFree(d_b); cudaFree(d_b2); cudaFree(d_g); cudaFree(d_g2);
    return pf;
}

// verify: witness-free. g = device pp generators (size n), b from u (public),
// P0 = C0 + eval*Q computed by the CALLER from public commitments.
static bool ipa_verify(const G1Jacobian_t* d_g, uint n, const G1Jacobian_t& Q,
                       const G1Jacobian_t& P0, const std::vector<Fr_t>& u_b,
                       const IpaProof& pf, fs::Transcript& tr) {
    uint rounds = pf.L.size();
    if (rounds != pf.R.size()) return false;
    if (n != (1u << rounds)) return false;
    if ((1u << u_b.size()) != n) return false;
    std::vector<Fr_t> xs(rounds), xis(rounds);
    G1Jacobian_t P = P0;
    std::vector<Fr_t> b = me_weights(u_b);
    for (uint r = 0; r < rounds; r++) {
        absorb_g1(tr, "L", pf.L[r]); absorb_g1(tr, "R", pf.R[r]);
        xs[r] = fs_challenge_fr(tr); xis[r] = inv(xs[r]);
        P = h_add(h_add(h_mul(pf.L[r], h_scalar(xs[r], xs[r], 2)), P),
                  h_mul(pf.R[r], h_scalar(xis[r], xis[r], 2)));
        uint half = b.size() / 2;
        for (uint k = 0; k < half; k++)
            b[k] = h_scalar(h_scalar(xis[r], b[k], 2), h_scalar(xs[r], b[k + half], 2), 0);
        b.resize(half);
    }
    // g_f via the s-vector MSM (pinned == explicit fold in vrf_toy_ipa.cu)
    std::vector<Fr_t> s(n);
    for (uint i = 0; i < n; i++) {
        Fr_t w = F_ONE;
        for (uint r = 0; r < rounds; r++)
            w = h_scalar(w, ((i >> (rounds - 1 - r)) & 1) ? xs[r] : xis[r], 2);
        s[i] = w;
    }
    Fr_t* d_s; cudaMalloc(&d_s, n * sizeof(Fr_t));
    cudaMemcpy(d_s, s.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);
    G1Jacobian_t g_f = dev_msm(d_g, d_s, n);
    cudaFree(d_s);
    return g1_eq(P, h_add(h_mul(g_f, pf.a_final), h_mul(Q, h_scalar(pf.a_final, b[0], 2))));
}

// startup self-checks for the hardcoded constants
static void vrf_selfcheck() {
    if (!fr_eq(h_scalar(F_INV2, F_TWO, 2), F_ONE))
        throw std::runtime_error("INV2 self-check failed");
}
