// Binius substrate step 1 (design doc section 21): Fan-Paar binary tower field
// library, host + device, branch-free bit arithmetic (no tables, no init).
//
// Tower: T_0 = GF(2), T_{k+1} = T_k[X_k]/(X_k^2 + X_k*X_{k-1} + 1), with
// X_0^2 = X_0 + 1 at the base.  An element of T_k is a 2^k-bit string whose
// bit u is the coefficient of the u-th F_2 basis monomial (products of tower
// generators), so T_j embeds in T_k (j<k) as literal zero-extension, and the
// packed-bit PCS "unpack" is literally reading bits.  Validated against the
// independent recursive prototype (binius_proto.cu) by p3_binius_field_test.
//
// Levels used by the prover: T_4 = GF(2^16) (NTT / packing field, bf16_t) and
// T_7 = GF(2^128) (challenge field, bf128_t).  Addition everywhere is XOR;
// integer arithmetic does NOT exist here -- carry logic is rebuilt as explicit
// bit constraints (section 21 carry note).
//
// mul cost model: level k mul = 3 muls at k-1 (Karatsuba) + one mulgen, all
// inlined; bf16_mul ~ 27 GF(4) muls of a few LOP3s each, bf128_mul = 27
// bf16-scale blocks.  bf128_smul16 (T_16 scalar times T_128) exploits that
// the 8 16-bit limbs are a T_16-basis: 8 bf16_muls, ~27x cheaper than full
// bf128_mul -- the sumcheck workhorse.
#pragma once
#include <cstdint>

typedef uint16_t bf16_t;
struct bf128_t { uint64_t lo, hi; };

#define BF_HD static __host__ __device__ __forceinline__

// ---- T_1 = GF(4), 2 bits, X0^2 = X0 + 1 ----
BF_HD uint32_t bf2_mul(uint32_t a, uint32_t b) {
    uint32_t a0 = a & 1, a1 = (a >> 1) & 1, b0 = b & 1, b1 = (b >> 1) & 1;
    uint32_t p00 = a0 & b0, p11 = a1 & b1, pm = (a0 ^ a1) & (b0 ^ b1);
    return (p00 ^ p11) | ((pm ^ p00) << 1);
}
BF_HD uint32_t bf2_mg(uint32_t a) {              // * X0
    uint32_t c0 = a & 1, c1 = (a >> 1) & 1;
    return c1 | ((c0 ^ c1) << 1);
}
BF_HD uint32_t bf2_inv(uint32_t a) { return (0x2310u >> (4 * a)) & 3; }

// ---- T_2 = GF(16), 4 bits, X1^2 = X1*X0 + 1 ----
BF_HD uint32_t bf4_mul(uint32_t a, uint32_t b) {
    uint32_t a0 = a & 3, a1 = (a >> 2) & 3, b0 = b & 3, b1 = (b >> 2) & 3;
    uint32_t p00 = bf2_mul(a0, b0), p11 = bf2_mul(a1, b1);
    uint32_t pm  = bf2_mul(a0 ^ a1, b0 ^ b1);
    return (p00 ^ p11) | ((pm ^ p00 ^ p11 ^ bf2_mg(p11)) << 2);
}
BF_HD uint32_t bf4_mg(uint32_t a) {              // * X1
    uint32_t c0 = a & 3, c1 = (a >> 2) & 3;
    return c1 | ((c0 ^ bf2_mg(c1)) << 2);
}
BF_HD uint32_t bf4_inv(uint32_t a) {
    uint32_t a0 = a & 3, a1 = (a >> 2) & 3;
    uint32_t t = a0 ^ bf2_mg(a1);                        // a0 + a1*X0
    uint32_t d = bf2_mul(a0, t) ^ bf2_mul(a1, a1);       // norm in T_1
    uint32_t di = bf2_inv(d);
    return bf2_mul(t, di) | (bf2_mul(a1, di) << 2);
}

// ---- T_3 = GF(2^8), X2^2 = X2*X1 + 1 ----
BF_HD uint32_t bf8_mul(uint32_t a, uint32_t b) {
    uint32_t a0 = a & 15, a1 = (a >> 4) & 15, b0 = b & 15, b1 = (b >> 4) & 15;
    uint32_t p00 = bf4_mul(a0, b0), p11 = bf4_mul(a1, b1);
    uint32_t pm  = bf4_mul(a0 ^ a1, b0 ^ b1);
    return (p00 ^ p11) | ((pm ^ p00 ^ p11 ^ bf4_mg(p11)) << 4);
}
BF_HD uint32_t bf8_mg(uint32_t a) {              // * X2
    uint32_t c0 = a & 15, c1 = (a >> 4) & 15;
    return c1 | ((c0 ^ bf4_mg(c1)) << 4);
}
BF_HD uint32_t bf8_inv(uint32_t a) {
    uint32_t a0 = a & 15, a1 = (a >> 4) & 15;
    uint32_t t = a0 ^ bf4_mg(a1);
    uint32_t d = bf4_mul(a0, t) ^ bf4_mul(a1, a1);
    uint32_t di = bf4_inv(d);
    return bf4_mul(t, di) | (bf4_mul(a1, di) << 4);
}

// ---- T_4 = GF(2^16), X3^2 = X3*X2 + 1  (NTT / packing field) ----
BF_HD uint32_t bf16_mul(uint32_t a, uint32_t b) {
    uint32_t a0 = a & 255, a1 = (a >> 8) & 255, b0 = b & 255, b1 = (b >> 8) & 255;
    uint32_t p00 = bf8_mul(a0, b0), p11 = bf8_mul(a1, b1);
    uint32_t pm  = bf8_mul(a0 ^ a1, b0 ^ b1);
    return (p00 ^ p11) | ((pm ^ p00 ^ p11 ^ bf8_mg(p11)) << 8);
}
BF_HD uint32_t bf16_mg(uint32_t a) {             // * X3
    uint32_t c0 = a & 255, c1 = (a >> 8) & 255;
    return c1 | ((c0 ^ bf8_mg(c1)) << 8);
}
BF_HD uint32_t bf16_inv(uint32_t a) {
    uint32_t a0 = a & 255, a1 = (a >> 8) & 255;
    uint32_t t = a0 ^ bf8_mg(a1);
    uint32_t d = bf8_mul(a0, t) ^ bf8_mul(a1, a1);
    uint32_t di = bf8_inv(d);
    return bf8_mul(t, di) | (bf8_mul(a1, di) << 8);
}
BF_HD uint32_t bf16_pow(uint32_t a, uint32_t e) {
    uint32_t r = 1, b = a;
    while (e) { if (e & 1) r = bf16_mul(r, b); b = bf16_mul(b, b); e >>= 1; }
    return r;
}

// ---- T_5 = GF(2^32), T_6 = GF(2^64) ----
BF_HD uint32_t bf32_mul(uint32_t a, uint32_t b) {
    uint32_t a0 = a & 0xffff, a1 = a >> 16, b0 = b & 0xffff, b1 = b >> 16;
    uint32_t p00 = bf16_mul(a0, b0), p11 = bf16_mul(a1, b1);
    uint32_t pm  = bf16_mul(a0 ^ a1, b0 ^ b1);
    return (p00 ^ p11) | ((pm ^ p00 ^ p11 ^ bf16_mg(p11)) << 16);
}
BF_HD uint32_t bf32_mg(uint32_t a) {             // * X4
    uint32_t c0 = a & 0xffff, c1 = a >> 16;
    return c1 | ((c0 ^ bf16_mg(c1)) << 16);
}
BF_HD uint32_t bf32_inv(uint32_t a) {
    uint32_t a0 = a & 0xffff, a1 = a >> 16;
    uint32_t t = a0 ^ bf16_mg(a1);
    uint32_t d = bf16_mul(a0, t) ^ bf16_mul(a1, a1);
    uint32_t di = bf16_inv(d);
    return bf16_mul(t, di) | (bf16_mul(a1, di) << 16);
}
BF_HD uint64_t bf64_mul(uint64_t a, uint64_t b) {
    uint32_t a0 = (uint32_t)a, a1 = (uint32_t)(a >> 32);
    uint32_t b0 = (uint32_t)b, b1 = (uint32_t)(b >> 32);
    uint32_t p00 = bf32_mul(a0, b0), p11 = bf32_mul(a1, b1);
    uint32_t pm  = bf32_mul(a0 ^ a1, b0 ^ b1);
    return (uint64_t)(p00 ^ p11) | ((uint64_t)(pm ^ p00 ^ p11 ^ bf32_mg(p11)) << 32);
}
BF_HD uint64_t bf64_mg(uint64_t a) {             // * X5
    uint32_t c0 = (uint32_t)a, c1 = (uint32_t)(a >> 32);
    return (uint64_t)c1 | ((uint64_t)(c0 ^ bf32_mg(c1)) << 32);
}
BF_HD uint64_t bf64_inv(uint64_t a) {
    uint32_t a0 = (uint32_t)a, a1 = (uint32_t)(a >> 32);
    uint32_t t = a0 ^ bf32_mg(a1);
    uint32_t d = bf32_mul(a0, t) ^ bf32_mul(a1, a1);
    uint32_t di = bf32_inv(d);
    return (uint64_t)bf32_mul(t, di) | ((uint64_t)bf32_mul(a1, di) << 32);
}

// ---- T_7 = GF(2^128)  (challenge field) ----
BF_HD bf128_t bf128_add(bf128_t a, bf128_t b) { return {a.lo ^ b.lo, a.hi ^ b.hi}; }
BF_HD bf128_t bf128_mul(bf128_t a, bf128_t b) {
    uint64_t p00 = bf64_mul(a.lo, b.lo), p11 = bf64_mul(a.hi, b.hi);
    uint64_t pm  = bf64_mul(a.lo ^ a.hi, b.lo ^ b.hi);
    return {p00 ^ p11, pm ^ p00 ^ p11 ^ bf64_mg(p11)};
}
BF_HD bf128_t bf128_inv(bf128_t a) {
    uint64_t t = a.lo ^ bf64_mg(a.hi);
    uint64_t d = bf64_mul(a.lo, t) ^ bf64_mul(a.hi, a.hi);
    uint64_t di = bf64_inv(d);
    return {bf64_mul(t, di), bf64_mul(a.hi, di)};
}
BF_HD bool bf128_eq(bf128_t a, bf128_t b) { return a.lo == b.lo && a.hi == b.hi; }
BF_HD bool bf128_is0(bf128_t a) { return !(a.lo | a.hi); }
BF_HD bf128_t bf128_zero() { return {0, 0}; }
BF_HD bf128_t bf128_one()  { return {1, 0}; }
BF_HD bf128_t bf128_from16(uint32_t x) { return {(uint64_t)(x & 0xffff), 0}; }
BF_HD bf128_t bf128_from64(uint64_t x) { return {x, 0}; }

// T_16-scalar times T_128: the 8 16-bit limbs of a T_128 element are its
// coordinates over a T_16 basis, so scalar action is limb-wise bf16_mul.
BF_HD bf128_t bf128_smul16(bf128_t a, uint32_t s) {
    uint64_t lo = 0, hi = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        lo |= (uint64_t)bf16_mul((uint32_t)((a.lo >> (16 * i)) & 0xffff), s) << (16 * i);
        hi |= (uint64_t)bf16_mul((uint32_t)((a.hi >> (16 * i)) & 0xffff), s) << (16 * i);
    }
    return {lo, hi};
}
// GF(2)-scalar times T_128 (bit witness fast path): select or zero.
BF_HD bf128_t bf128_smul1(bf128_t a, uint32_t bit) {
    uint64_t m = (uint64_t)0 - (uint64_t)(bit & 1);
    return {a.lo & m, a.hi & m};
}

BF_HD bf128_t bf128_pow(bf128_t a, uint64_t elo, uint64_t ehi) {
    bf128_t r = {1, 0}, b = a;
    for (int i = 0; i < 64; i++) { if ((elo >> i) & 1) r = bf128_mul(r, b); b = bf128_mul(b, b); }
    for (int i = 0; i < 64; i++) { if ((ehi >> i) & 1) r = bf128_mul(r, b); b = bf128_mul(b, b); }
    return r;
}
