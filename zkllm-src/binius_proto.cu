// Binius de-risk prototype (design doc section 17 scoping, section 20.8):
// standalone binary tower field (Fan-Paar) + additive NTT (Gao-Mateer),
// host-only.  Validates the two mathematical primitives a Binius-style
// commitment for the Hawkeye bit-witness would stand on:
//   (1) the recursive tower  T_0=GF(2), T_{k+1}=T_k[X_k]/(X_k^2+X_k*X_{k-1}+1)
//       (X_0^2+X_0+1 at the base) with Karatsuba multiplication,
//   (2) the additive FFT evaluating a GF(2^m)[x] polynomial on an F2-linear
//       subspace in O(n log n) via Taylor expansion in y = x^2+x.
// Selftests: field axioms + Fermat/inverse at 8..128 bits; additive NTT vs
// brute-force evaluation at n = 2^4..2^9 over T_4 (GF(2^16)).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>

// ---- Fan-Paar tower on bitstrings of length 2^k (k <= 7 -> 128 bits) ----
struct U128 { uint64_t lo = 0, hi = 0; };
static inline U128 x128(U128 a, U128 b) { return {a.lo ^ b.lo, a.hi ^ b.hi}; }
static inline bool z128(U128 a) { return !(a.lo | a.hi); }
static inline bool eq128(U128 a, U128 b) { return a.lo == b.lo && a.hi == b.hi; }

// extract/insert 2^(k-1)-bit halves of a 2^k-bit element (k >= 1)
static inline U128 half_lo(U128 a, int k) {
    int nb = 1 << (k - 1);
    if (nb >= 64) return {a.lo, 0};
    return {a.lo & ((1ULL << nb) - 1), 0};
}
static inline U128 half_hi(U128 a, int k) {
    int nb = 1 << (k - 1);
    if (nb >= 64) return {a.hi, 0};
    return {(a.lo >> nb) & ((1ULL << nb) - 1), 0};
}
static inline U128 mk(U128 lo, U128 hi, int k) {
    int nb = 1 << (k - 1);
    if (nb >= 64) return {lo.lo, hi.lo};
    return {lo.lo | (hi.lo << nb), 0};
}
// multiply by the generator X_{k-1} inside T_k (recursive)
static U128 mulgen(U128 a, int k);
// full tower multiplication in T_k
static U128 tmul(U128 a, U128 b, int k) {
    if (k == 0) return {a.lo & b.lo & 1, 0};
    U128 a0 = half_lo(a, k), a1 = half_hi(a, k);
    U128 b0 = half_lo(b, k), b1 = half_hi(b, k);
    U128 p00 = tmul(a0, b0, k - 1);
    U128 p11 = tmul(a1, b1, k - 1);
    U128 pmid = tmul(x128(a0, a1), x128(b0, b1), k - 1);   // Karatsuba
    // (a0+a1X)(b0+b1X) = p00 + (pmid+p00+p11) X + p11 X^2,
    // X^2 = X*X_{k-2} + 1   (X_{-1} := 1 at the base level)
    U128 hi = x128(x128(pmid, p00), p11);
    U128 lo = x128(p00, p11);
    if (k >= 2) hi = x128(hi, mulgen(p11, k - 1));
    else        hi = x128(hi, p11);            // base: X_0^2 = X_0 + 1
    return mk(lo, hi, k);
}
static U128 mulgen(U128 a, int k) {
    // X_{k-1} * (c0 + c1 X_{k-1}) = c1 + (c0 + c1 X_{k-2}) X_{k-1}
    U128 c0 = half_lo(a, k), c1 = half_hi(a, k);
    U128 hi = c0;
    if (k >= 2) hi = x128(hi, mulgen(c1, k - 1));
    else        hi = x128(hi, c1);
    return mk(c1, hi, k);
}
static U128 tpow(U128 a, U128 e_unused, uint64_t elo, uint64_t ehi, int k) {
    (void)e_unused;
    U128 r = {1, 0}, b = a;
    for (int i = 0; i < 64; i++) { if ((elo >> i) & 1) r = tmul(r, b, k); b = tmul(b, b, k); }
    for (int i = 0; i < 64; i++) { if ((ehi >> i) & 1) r = tmul(r, b, k); b = tmul(b, b, k); }
    return r;
}
static uint64_t rng_s = 0x1234abcd5678ULL;
static uint64_t rnd() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static U128 trand(int k) {
    int nb = 1 << k;
    U128 r = {rnd(), rnd()};
    if (nb < 64) r = {r.lo & ((1ULL << nb) - 1), 0};
    else if (nb == 64) r.hi = 0;
    return r;
}

// ---- additive NTT over T_4 = GF(2^16) ----
typedef uint32_t gf16;                          // 16-bit elements
static inline gf16 gmul(gf16 a, gf16 b) { return (gf16)tmul({a, 0}, {b, 0}, 4).lo; }
// Taylor expansion of f (deg < n, n = 2^t) in y = x^2 + x: pairs out[2i], out[2i+1]
// such that f(x) = sum_i (out[2i] + out[2i+1] x) y^i.  In place, recursive.
static void taylor(gf16* f, size_t n) {
    if (n <= 2) return;
    size_t t = n / 4;                            // y^t = x^{2t} + x^t, halves in y
    // divide by y^{n/4}: q = f div (x^{2t}+x^t) processed from the top
    for (size_t i = n - 1; i >= 2 * t; i--) f[i - t] ^= f[i];   // r update; q sits in f[2t..)
    // f now holds r in [0,2t) and q in [2t,n): recurse on both halves
    taylor(f, 2 * t);
    taylor(f + 2 * t, 2 * t);
}
// evaluate f (coeffs, deg < n = 2^m) at all points of span(beta[0..m-1]),
// REQUIRES beta[m-1] == 1.  out[u] = f(sum_i u_i beta_i), out[u + n/2] adds 1.
// (positional index: u's bit i selects beta_i, i < m-1; bit m-1 adds 1.)
static void afft(std::vector<gf16> f, std::vector<gf16> beta, std::vector<gf16>& out) {
    size_t n = f.size(), m = beta.size();
    if (n == 1) { out = {f[0]}; return; }
    if (beta[m - 1] != 1) { printf("afft: basis not normalized\n"); exit(1); }
    taylor(f.data(), n);
    std::vector<gf16> g0(n / 2), g1(n / 2);
    for (size_t i = 0; i < n / 2; i++) { g0[i] = f[2 * i]; g1[i] = f[2 * i + 1]; }
    // image basis gamma_i = beta_i^2 + beta_i (i < m-1); normalize by delta = gamma[m-2]
    std::vector<gf16> gam(m - 1);
    for (size_t i = 0; i + 1 < m; i++) gam[i] = gmul(beta[i], beta[i]) ^ beta[i];
    std::vector<gf16> G0, G1;
    if (m == 1) { G0 = {g0[0]}; G1 = {g1[0]}; }
    else {
        gf16 delta = gam[m - 2];
        // scale: g'(x) = g(delta x) evaluated on gam/delta = g on gam (same index)
        gf16 dinv = (gf16)tpow({delta, 0}, {}, (1ULL << 16) - 2, 0, 4).lo;
        std::vector<gf16> nb2(m - 1);
        for (size_t i = 0; i + 1 < m; i++) nb2[i] = gmul(gam[i], dinv);
        gf16 dp = 1;
        std::vector<gf16> s0 = g0, s1 = g1;
        for (size_t i = 0; i < n / 2; i++) { s0[i] = gmul(s0[i], dp); s1[i] = gmul(s1[i], dp); dp = gmul(dp, delta); }
        afft(std::move(s0), nb2, G0);
        afft(std::move(s1), nb2, G1);
    }
    out.assign(n, 0);
    for (size_t u = 0; u < n / 2; u++) {
        gf16 om = 0;                              // omega_u = sum u_i beta_i
        for (size_t i = 0; i + 1 < m; i++) if ((u >> i) & 1) om ^= beta[i];
        gf16 e0 = G0[u] ^ gmul(om, G1[u]);        // f(omega) = g0(delta) + omega g1(delta)
        out[u] = e0;
        out[u + n / 2] = e0 ^ G1[u];              // f(omega + 1)
    }
}

int main() {
    int fail = 0;
    // ---- tower field axioms + Fermat inverses at 8..128 bits ----
    for (int k : {3, 4, 5, 6, 7}) {
        int bad = 0;
        for (int it = 0; it < 200; it++) {
            U128 a = trand(k), b = trand(k), c = trand(k);
            if (!eq128(tmul(a, b, k), tmul(b, a, k))) bad++;
            if (!eq128(tmul(tmul(a, b, k), c, k), tmul(a, tmul(b, c, k), k))) bad++;
            if (!eq128(tmul(a, x128(b, c), k), x128(tmul(a, b, k), tmul(a, c, k)))) bad++;
            if (!z128(a)) {
                // a^(2^n - 1) == 1  (n = 2^k)
                int nbits = 1 << k;
                uint64_t elo = nbits >= 64 ? ~0ULL : (1ULL << nbits) - 1;
                uint64_t ehi = nbits > 64 ? (nbits == 128 ? ~0ULL : (1ULL << (nbits - 64)) - 1) : 0;
                U128 f1 = tpow(a, {}, elo, ehi, k);
                if (!(f1.lo == 1 && f1.hi == 0)) bad++;
            }
        }
        printf("tower T_%d (%3d-bit): %s (bad=%d/800)\n", k, 1 << k, bad ? "FAIL" : "PASS", bad);
        if (bad) fail = 1;
    }
    // ---- additive NTT vs brute force over GF(2^16) ----
    for (int m : {4, 6, 8, 9}) {
        size_t n = (size_t)1 << m;
        // tower basis with last element 1: (X-basis products indices 1..m-1, then 1)
        std::vector<gf16> beta(m);
        for (int i = 0; i + 1 < m; i++) beta[i] = (gf16)(2u << i);   // X_0, X_0X_1(bit2)... distinct basis bits
        beta[m - 1] = 1;
        std::vector<gf16> f(n);
        for (auto& x : f) x = (gf16)(rnd() & 0xffff);
        std::vector<gf16> out;
        afft(f, beta, out);
        size_t bad = 0;
        for (size_t u = 0; u < n; u++) {
            gf16 om = (u >> (m - 1)) & 1;                       // +1 for the top bit
            for (int i = 0; i + 1 < m; i++) if ((u >> i) & 1) om ^= beta[i];
            gf16 acc = 0, xp = 1;
            for (size_t i = 0; i < n; i++) { acc ^= gmul(f[i], xp); xp = gmul(xp, om); }
            if (acc != out[u]) bad++;
        }
        printf("additive-NTT n=2^%d over GF(2^16): %s (bad=%zu/%zu)\n", m, bad ? "FAIL" : "PASS", bad, n);
        if (bad) fail = 1;
    }
    // ---- throughput sanity: 128-bit tower mult on host ----
    {
        U128 a = trand(7), b = trand(7);
        auto t0 = std::chrono::steady_clock::now();
        const int N = 200000;
        for (int i = 0; i < N; i++) { a = tmul(a, b, 7); a.lo ^= i; }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("host T_7 (128-bit) mult: %.0f ns each (recursive scalar reference; the real\n"
               "  implementation uses CLMUL/GPU bit-slicing -- this is a correctness anchor)\n",
               ms * 1e6 / N);
    }
    printf("\nBINIUS-PROTO: %s\n", fail ? "FAIL" : "ALL PASS");
    return fail;
}
