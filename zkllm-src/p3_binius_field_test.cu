// Selftest for p3_binius_field.cuh (design doc section 21.1).
// Teeth: the branch-free tower is cross-checked BITWISE against an independent
// implementation (the recursive Fan-Paar reference from binius_proto.cu, which
// itself passed 800/800 axiom checks per level), exhaustively at 4/8 bits and
// randomly at 16..128 bits; inverses vs Fermat; embeddings; host == device.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_binius_field.cuh"

static int np_ = 0, nf_ = 0;
static void ck(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) np_++; else nf_++;
}

// ---- independent reference: recursive Fan-Paar on 128-bit strings ----
// (verbatim math from binius_proto.cu; kept separate from the production path)
struct U128 { uint64_t lo = 0, hi = 0; };
static inline U128 x128(U128 a, U128 b) { return {a.lo ^ b.lo, a.hi ^ b.hi}; }
static inline U128 rhalf_lo(U128 a, int k) {
    int nb = 1 << (k - 1);
    if (nb >= 64) return {a.lo, 0};
    return {a.lo & ((1ULL << nb) - 1), 0};
}
static inline U128 rhalf_hi(U128 a, int k) {
    int nb = 1 << (k - 1);
    if (nb >= 64) return {a.hi, 0};
    return {(a.lo >> nb) & ((1ULL << nb) - 1), 0};
}
static inline U128 rmk(U128 lo, U128 hi, int k) {
    int nb = 1 << (k - 1);
    if (nb >= 64) return {lo.lo, hi.lo};
    return {lo.lo | (hi.lo << nb), 0};
}
static U128 rmulgen(U128 a, int k);
static U128 rmul(U128 a, U128 b, int k) {
    if (k == 0) return {a.lo & b.lo & 1, 0};
    U128 a0 = rhalf_lo(a, k), a1 = rhalf_hi(a, k);
    U128 b0 = rhalf_lo(b, k), b1 = rhalf_hi(b, k);
    U128 p00 = rmul(a0, b0, k - 1), p11 = rmul(a1, b1, k - 1);
    U128 pm = rmul(x128(a0, a1), x128(b0, b1), k - 1);
    U128 hi = x128(x128(pm, p00), p11);
    U128 lo = x128(p00, p11);
    if (k >= 2) hi = x128(hi, rmulgen(p11, k - 1));
    else        hi = x128(hi, p11);
    return rmk(lo, hi, k);
}
static U128 rmulgen(U128 a, int k) {
    U128 c0 = rhalf_lo(a, k), c1 = rhalf_hi(a, k);
    U128 hi = c0;
    if (k >= 2) hi = x128(hi, rmulgen(c1, k - 1));
    else        hi = x128(hi, c1);
    return rmk(c1, hi, k);
}
static uint64_t rs_ = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }
static bf128_t rnd128() { return {rnd(), rnd()}; }

// dispatch the production mul/mg/inv at runtime level (test-only shim)
static bf128_t pmul(bf128_t a, bf128_t b, int k) {
    switch (k) {
        case 1: return {bf2_mul((uint32_t)a.lo, (uint32_t)b.lo), 0};
        case 2: return {bf4_mul((uint32_t)a.lo, (uint32_t)b.lo), 0};
        case 3: return {bf8_mul((uint32_t)a.lo, (uint32_t)b.lo), 0};
        case 4: return {bf16_mul((uint32_t)a.lo, (uint32_t)b.lo), 0};
        case 5: return {bf32_mul((uint32_t)a.lo, (uint32_t)b.lo), 0};
        case 6: return {bf64_mul(a.lo, b.lo), 0};
        default: return bf128_mul(a, b);
    }
}
static bf128_t pinv(bf128_t a, int k) {
    switch (k) {
        case 1: return {bf2_inv((uint32_t)a.lo), 0};
        case 2: return {bf4_inv((uint32_t)a.lo), 0};
        case 3: return {bf8_inv((uint32_t)a.lo), 0};
        case 4: return {bf16_inv((uint32_t)a.lo), 0};
        case 5: return {bf32_inv((uint32_t)a.lo), 0};
        case 6: return {bf64_inv(a.lo), 0};
        default: return bf128_inv(a);
    }
}
static bf128_t trand(int k) {
    int nb = 1 << k;
    bf128_t r = {rnd(), rnd()};
    if (nb < 64) r = {r.lo & ((1ULL << nb) - 1), 0};
    else if (nb == 64) r.hi = 0;
    return r;
}

__global__ void dev_mul_kernel(const bf128_t* a, const bf128_t* b, bf128_t* m128,
                               bf128_t* i128, uint32_t* m16, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    m128[i] = bf128_mul(a[i], b[i]);
    i128[i] = bf128_is0(a[i]) ? bf128_zero() : bf128_inv(a[i]);
    m16[i] = bf16_mul((uint32_t)(a[i].lo & 0xffff), (uint32_t)(b[i].lo & 0xffff));
}

int main() {
    // ---- exhaustive cross-check vs reference at 4 and 8 bits ----
    {
        int bad = 0;
        for (uint32_t a = 0; a < 16; a++) for (uint32_t b = 0; b < 16; b++)
            if (bf4_mul(a, b) != (uint32_t)rmul({a, 0}, {b, 0}, 2).lo) bad++;
        ck("bf4_mul == reference (exhaustive 256 pairs)", bad == 0);
        bad = 0;
        for (uint32_t a = 0; a < 256; a++) for (uint32_t b = 0; b < 256; b++)
            if (bf8_mul(a, b) != (uint32_t)rmul({a, 0}, {b, 0}, 3).lo) bad++;
        ck("bf8_mul == reference (exhaustive 65536 pairs)", bad == 0);
        bad = 0;
        for (uint32_t a = 1; a < 16; a++)
            if (bf4_mul(a, bf4_inv(a)) != 1) bad++;
        for (uint32_t a = 1; a < 256; a++)
            if (bf8_mul(a, bf8_inv(a)) != 1) bad++;
        ck("bf4/bf8 inverses (exhaustive)", bad == 0);
    }
    // ---- random cross-check + axioms + inverse at 16..128 bits ----
    for (int k = 4; k <= 7; k++) {
        int bad = 0;
        for (int it = 0; it < 20000; it++) {
            bf128_t a = trand(k), b = trand(k), c = trand(k);
            bf128_t ab = pmul(a, b, k);
            U128 ra = rmul({a.lo, a.hi}, {b.lo, b.hi}, k);
            if (ab.lo != ra.lo || ab.hi != ra.hi) bad++;
            if (!bf128_eq(ab, pmul(b, a, k))) bad++;
            if (!bf128_eq(pmul(ab, c, k), pmul(a, pmul(b, c, k), k))) bad++;
            if (!bf128_eq(pmul(a, bf128_add(b, c), k),
                          bf128_add(pmul(a, b, k), pmul(a, c, k)))) bad++;
            if (!bf128_is0(a)) {
                bf128_t ia = pinv(a, k);
                if (!bf128_eq(pmul(a, ia, k), bf128_one())) bad++;
            }
        }
        char msg[96];
        snprintf(msg, sizeof msg, "T_%d (%3d-bit): ref-match + axioms + inverse (100k checks)",
                 k, 1 << k);
        ck(msg, bad == 0);
    }
    // ---- Fermat: a^(2^n - 1) == 1, and bf16_inv == a^(2^16-2) ----
    {
        int bad = 0;
        for (int it = 0; it < 200; it++) {
            uint32_t a = (uint32_t)(rnd() & 0xffff);
            if (!a) continue;
            if (bf16_pow(a, 0xffff) != 1) bad++;
            if (bf16_inv(a) != bf16_pow(a, 0xfffe)) bad++;
        }
        ck("bf16 Fermat + inv == pow(2^16-2)", bad == 0);
        bad = 0;
        for (int it = 0; it < 50; it++) {
            bf128_t a = rnd128();
            if (bf128_is0(a)) continue;
            bf128_t f = bf128_pow(a, ~0ULL, ~0ULL);   // a^(2^128 - 1)
            if (!bf128_eq(f, bf128_one())) bad++;
        }
        ck("bf128 Fermat a^(2^128-1) == 1", bad == 0);
    }
    // ---- embedding + scalar-action consistency ----
    {
        int bad = 0;
        for (int it = 0; it < 20000; it++) {
            uint32_t s = (uint32_t)(rnd() & 0xffff), t = (uint32_t)(rnd() & 0xffff);
            // subfield embedding is multiplicative
            bf128_t e = bf128_mul(bf128_from16(s), bf128_from16(t));
            if (!bf128_eq(e, bf128_from16(bf16_mul(s, t)))) bad++;
            // limb-wise scalar action == full embedded mul
            bf128_t a = rnd128();
            if (!bf128_eq(bf128_smul16(a, s), bf128_mul(a, bf128_from16(s)))) bad++;
            // bit scalar
            if (!bf128_eq(bf128_smul1(a, 1), a) || !bf128_is0(bf128_smul1(a, 0))) bad++;
        }
        ck("T_16 -> T_128 embedding + smul16/smul1 (60k checks)", bad == 0);
    }
    // ---- host == device, bitwise ----
    {
        const uint32_t n = 65536;
        std::vector<bf128_t> a(n), b(n), m(n), iv(n);
        std::vector<uint32_t> m16(n);
        for (uint32_t i = 0; i < n; i++) { a[i] = rnd128(); b[i] = rnd128(); }
        bf128_t *da, *db, *dm, *di; uint32_t* d16;
        cudaMalloc(&da, n * sizeof(bf128_t)); cudaMalloc(&db, n * sizeof(bf128_t));
        cudaMalloc(&dm, n * sizeof(bf128_t)); cudaMalloc(&di, n * sizeof(bf128_t));
        cudaMalloc(&d16, n * sizeof(uint32_t));
        cudaMemcpy(da, a.data(), n * sizeof(bf128_t), cudaMemcpyHostToDevice);
        cudaMemcpy(db, b.data(), n * sizeof(bf128_t), cudaMemcpyHostToDevice);
        dev_mul_kernel<<<(n + 255) / 256, 256>>>(da, db, dm, di, d16, n);
        cudaMemcpy(m.data(), dm, n * sizeof(bf128_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(iv.data(), di, n * sizeof(bf128_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(m16.data(), d16, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaError_t err = cudaGetLastError();
        int bad = (err != cudaSuccess);
        for (uint32_t i = 0; i < n; i++) {
            if (!bf128_eq(m[i], bf128_mul(a[i], b[i]))) bad++;
            bf128_t hi = bf128_is0(a[i]) ? bf128_zero() : bf128_inv(a[i]);
            if (!bf128_eq(iv[i], hi)) bad++;
            if (m16[i] != bf16_mul((uint32_t)(a[i].lo & 0xffff), (uint32_t)(b[i].lo & 0xffff))) bad++;
        }
        ck("device bf16_mul/bf128_mul/bf128_inv == host, 64k lanes", bad == 0);
        cudaFree(da); cudaFree(db); cudaFree(dm); cudaFree(di); cudaFree(d16);
    }

    printf("\nBINIUS-FIELD: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
