// Selftest for p3_binius_sumcheck.cuh (design doc section 21.4).
// Teeth: honest sum accepted with finals matching independent multilinear
// evaluation; tampered rounds/claims reject; a cheating prover that patches
// round 0 to hide a false claim (so every p(0)+p(1) chain check passes) is
// caught by the final integrand check; zerocheck with the eq column catches
// a single violated constraint row.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_binius_sumcheck.cuh"

static int np_ = 0, nf_ = 0;
static void ck(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) np_++; else nf_++;
}
static uint64_t rs_ = 0xfeedface12345678ULL;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }
static bf128_t rnd128() { return {rnd(), rnd()}; }

// multilinear (values on hypercube) evaluated at an arbitrary point
static bf128_t ml_eval(const std::vector<bf128_t>& vals, int l, const bf128_t* r) {
    std::vector<bf128_t> eq;
    bf_eq_table(r, l, eq);
    bf128_t acc = bf128_zero();
    for (size_t x = 0; x < vals.size(); x++)
        acc = bf128_add(acc, bf128_mul(eq[x], vals[x]));
    return acc;
}

static bf128_t C_prod(const bf128_t* w, const void*) {          // w0*w1 + w2, D=2
    return bf128_add(bf128_mul(w[0], w[1]), w[2]);
}
static bf128_t C_zc(const bf128_t* w, const void*) {            // eq*(w1*w2 + w3), D=3
    return bf128_mul(w[0], bf128_add(bf128_mul(w[1], w[2]), w[3]));
}

int main() {
    // ---- honest sum, K=3, D=2, l=10 ----
    {
        int l = 10, K = 3;
        size_t n = (size_t)1 << l;
        std::vector<std::vector<bf128_t>> cols(K, std::vector<bf128_t>(n)), keep;
        for (auto& c : cols) for (auto& x : c) x = rnd128();
        keep = cols;
        bf128_t claim = bf128_zero();
        for (size_t x = 0; x < n; x++) {
            bf128_t w[3] = {cols[0][x], cols[1][x], cols[2][x]};
            claim = bf128_add(claim, C_prod(w, nullptr));
        }
        fs::Transcript tp("bfsc-test");
        BfScProof pf; std::vector<bf128_t> zp;
        bf_sumcheck_prove(l, K, cols, 2, C_prod, nullptr, tp, pf, zp);
        auto vfy = [&](const BfScProof& q, bf128_t cl, bool check_final) {
            fs::Transcript tv("bfsc-test");
            std::vector<bf128_t> zv; bf128_t E;
            if (!bf_sumcheck_verify(q, cl, tv, zv, &E)) return false;
            if (!check_final) return true;
            return bf128_eq(E, C_prod(q.finals.data(), nullptr));
        };
        ck("honest sum accepts (rounds + final integrand)", vfy(pf, claim, true));
        {
            int bad = 0;
            for (int k = 0; k < 3; k++)
                if (!bf128_eq(pf.finals[k], ml_eval(keep[k], l, zp.data()))) bad++;
            ck("finals == independent multilinear evaluation at zeta", bad == 0);
        }
        { auto q = pf; q.rounds[5].lo ^= 1;
          ck("tampered round polynomial rejects", !vfy(q, claim, true)); }
        { bf128_t c2 = claim; c2.hi ^= 2;
          ck("wrong claim rejects", !vfy(pf, c2, true)); }
        { auto q = pf; q.finals[1].lo ^= 4;
          ck("tampered finals reject (integrand check)", !vfy(q, claim, true)); }
        // cheating prover: false claim, round 0 patched so THAT round's chain
        // check p(0)+p(1)==claim passes; Fiat-Shamir reshuffles the later
        // challenges, so the lie must die somewhere downstream
        {
            bf128_t lie = bf128_add(claim, bf128_from64(7));
            auto q = pf;
            q.rounds[0] = bf128_add(q.rounds[0], bf128_from64(7));
            ck("patched round-0 cheat rejects", !vfy(q, lie, true));
        }
    }
    // ---- zerocheck shape with eq column, one violated row ----
    {
        int l = 8, K = 4;
        size_t n = (size_t)1 << l;
        std::vector<bf128_t> rz(l);
        for (auto& x : rz) x = rnd128();
        std::vector<std::vector<bf128_t>> cols(K, std::vector<bf128_t>(n));
        bf_eq_table(rz.data(), l, cols[0]);
        // satisfy w3 = w1*w2 everywhere
        for (size_t x = 0; x < n; x++) {
            cols[1][x] = rnd128(); cols[2][x] = rnd128();
            cols[3][x] = bf128_mul(cols[1][x], cols[2][x]);
        }
        auto run = [&](std::vector<std::vector<bf128_t>> cs) {
            fs::Transcript tp("bfzc-test");
            BfScProof pf; std::vector<bf128_t> zp;
            bf_sumcheck_prove(l, K, cs, 3, C_zc, nullptr, tp, pf, zp);
            fs::Transcript tv("bfzc-test");
            std::vector<bf128_t> zv; bf128_t E;
            if (!bf_sumcheck_verify(pf, bf128_zero(), tv, zv, &E)) return false;
            if (!bf128_eq(E, C_zc(pf.finals.data(), nullptr))) return false;
            // verifier-side eq validation (column 0 is public): over char 2 the
            // eq factor rz*z + (1+rz)(1+z) collapses to 1 + rz + z
            bf128_t eqv = bf128_one();
            for (int t = 0; t < l; t++)
                eqv = bf128_mul(eqv, bf128_add(bf128_one(), bf128_add(rz[t], zv[t])));
            return bf128_eq(pf.finals[0], eqv);
        };
        ck("zerocheck: satisfying witness accepts (incl. verifier eq check)", run(cols));
        auto bad = cols;
        bad[3][n / 3].lo ^= 1;                    // violate ONE row
        ck("zerocheck: single violated row rejects", !run(bad));
    }

    printf("\nBINIUS-SUMCHECK: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
