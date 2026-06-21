// P3 (speed lever 2): Goldilocks small field  p = 2^64 - 2^32 + 1.
//
// Why: the BLS12-381 scalar field is 256-bit (8 limbs); every model int8
// multiply becomes an 8x8-word Montgomery multiply in the proof. Goldilocks is
// 64-bit with a near-free reduction (2^64 = 2^32 - 1 mod p), and has a 2^32
// order 2-adic subgroup (cheap NTT / Reed-Solomon for hash-based PCS).
//
// Canonical representation: every gl_t held in [0, p). No Montgomery form, so a
// field element is literally its integer value (1 == 1u64) -- simpler chaining.
//
// reduce128 is the standard Plonky2/Goldilocks reduction; gl_mul matches the
// independent (a*b) mod p reference (see p3_field_bench selftest).
#pragma once
#include <cstdint>

#define GL_P   0xFFFFFFFF00000001ULL   // 2^64 - 2^32 + 1
#define GL_EPS 0x00000000FFFFFFFFULL   // 2^32 - 1  ( == 2^64 mod p )

typedef uint64_t gl_t;

// Reduce a 128-bit product (hi:lo) to [0,p).  Uses 2^64 = 2^32-1, 2^96 = -1.
static __host__ __device__ __forceinline__ gl_t gl_reduce128(uint64_t lo, uint64_t hi) {
    uint32_t hi_hi = (uint32_t)(hi >> 32);   // coeff of 2^96  (= -1 mod p)
    uint32_t hi_lo = (uint32_t)hi;           // coeff of 2^64  (= EPS mod p)
    uint64_t t0 = lo - (uint64_t)hi_hi;
    if (lo < (uint64_t)hi_hi) t0 -= GL_EPS;          // borrow: -2^64 == -EPS  => +p
    uint64_t t1 = (uint64_t)hi_lo * GL_EPS;          // < 2^64 (32b*32b)
    uint64_t t2 = t0 + t1;
    if (t2 < t0) t2 += GL_EPS;                        // carry: +2^64 == +EPS
    if (t2 >= GL_P) t2 -= GL_P;
    return t2;
}

static __host__ __device__ __forceinline__ gl_t gl_add(gl_t a, gl_t b) {
    uint64_t s = a + b;
    if (s < a) s += GL_EPS;       // wrapped 2^64
    if (s >= GL_P) s -= GL_P;
    return s;
}

static __host__ __device__ __forceinline__ gl_t gl_sub(gl_t a, gl_t b) {
    uint64_t d = a - b;
    if (a < b) d -= GL_EPS;       // -2^64 == -EPS  => +p, lands in (0,p)
    return d;
}

static __host__ __device__ __forceinline__ gl_t gl_mul(gl_t a, gl_t b) {
#ifdef __CUDA_ARCH__
    uint64_t hi = __umul64hi(a, b);
    uint64_t lo = a * b;
#else
    unsigned __int128 p = (unsigned __int128)a * (unsigned __int128)b;
    uint64_t lo = (uint64_t)p;
    uint64_t hi = (uint64_t)(p >> 64);
#endif
    return gl_reduce128(lo, hi);
}

static __host__ __device__ __forceinline__ gl_t gl_pow(gl_t a, uint64_t e) {
    gl_t r = 1ULL, base = a;
    while (e) { if (e & 1ULL) r = gl_mul(r, base); base = gl_mul(base, base); e >>= 1; }
    return r;
}

// multiplicative inverse via Fermat: a^(p-2).  (a != 0)
static __host__ __device__ __forceinline__ gl_t gl_inv(gl_t a) { return gl_pow(a, GL_P - 2ULL); }

// 7 is a multiplicative generator of Goldilocks; (p-1) = 2^32 * (2^32-1), so the
// 2-adic subgroup has order 2^32.  Return a primitive 2^logn-th root of unity.
static __host__ __device__ __forceinline__ gl_t gl_root_of_unity(uint32_t logn) {
    gl_t g2_32 = gl_pow(7ULL, GL_EPS);            // order exactly 2^32
    return gl_pow(g2_32, 1ULL << (32 - logn));    // order 2^logn
}
