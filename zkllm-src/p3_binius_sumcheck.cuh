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
