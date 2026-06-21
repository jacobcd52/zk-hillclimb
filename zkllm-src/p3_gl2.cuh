// P3 soundness upgrade: degree-2 extension GL2 = Goldilocks[u]/(u^2 - 7).
//
// The base field is 64-bit, so drawing Fiat-Shamir challenges from it gives sumcheck/FRI
// soundness ~ rounds*deg / 2^64 ~ 2^-58.  Drawing challenges from GL2 (~2^128 elements)
// lifts this to ~2^-116.  Witness/codeword stay in the base field; only the bound/folded
// values and challenges live in GL2 (base embeds as (x,0)).
//
// 7 is a quadratic non-residue mod p (checked in the selftest), so x^2-7 is irreducible.
#pragma once
#include <cstdint>
#include "p3_goldilocks.cuh"

struct gl2_t { gl_t a, b; };   // a + b*u,  u^2 = 7

#define GL2_NONRES 7ULL

static __host__ __device__ __forceinline__ gl2_t gl2_from(gl_t x) { return gl2_t{ x, 0ULL }; }
static __host__ __device__ __forceinline__ gl2_t gl2_zero() { return gl2_t{ 0ULL, 0ULL }; }
static __host__ __device__ __forceinline__ gl2_t gl2_one()  { return gl2_t{ 1ULL, 0ULL }; }
static __host__ __device__ __forceinline__ bool gl2_eq(gl2_t x, gl2_t y) { return x.a==y.a && x.b==y.b; }

static __host__ __device__ __forceinline__ gl2_t gl2_add(gl2_t x, gl2_t y) {
    return gl2_t{ gl_add(x.a,y.a), gl_add(x.b,y.b) };
}
static __host__ __device__ __forceinline__ gl2_t gl2_sub(gl2_t x, gl2_t y) {
    return gl2_t{ gl_sub(x.a,y.a), gl_sub(x.b,y.b) };
}
// (a0+a1 u)(b0+b1 u) = (a0b0 + 7 a1b1) + (a0b1 + a1b0) u
static __host__ __device__ __forceinline__ gl2_t gl2_mul(gl2_t x, gl2_t y) {
    gl_t a0b0 = gl_mul(x.a, y.a);
    gl_t a1b1 = gl_mul(x.b, y.b);
    gl_t cross = gl_add(gl_mul(x.a, y.b), gl_mul(x.b, y.a));
    gl_t lo = gl_add(a0b0, gl_mul(GL2_NONRES, a1b1));
    return gl2_t{ lo, cross };
}
static __host__ __device__ __forceinline__ gl2_t gl2_scale(gl2_t x, gl_t s) {  // base * ext
    return gl2_t{ gl_mul(x.a, s), gl_mul(x.b, s) };
}
// inverse: conjugate / norm,  norm = a0^2 - 7 a1^2  in the base field
static __host__ __device__ __forceinline__ gl2_t gl2_inv(gl2_t x) {
    gl_t norm = gl_sub(gl_mul(x.a, x.a), gl_mul(GL2_NONRES, gl_mul(x.b, x.b)));
    gl_t ninv = gl_inv(norm);
    return gl2_t{ gl_mul(x.a, ninv), gl_mul(gl_sub(0ULL, x.b), ninv) };
}
