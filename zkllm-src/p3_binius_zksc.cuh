// p3_binius_zksc.cuh -- REUSABLE zk-blinded zerocheck over the tower (design
// doc section 21.14, factored from the validated primitive).  Wraps the plain
// tower sumcheck so a degree-D zerocheck's round polynomials are HIDING: the
// char-2 Libra blind adds  gamma * ( B_0 + E*B_1 + ... + E^(D-1)*B_{D-1} )  to
// the integrand, with E = eq(rz,.) the zerocheck weight (its powers keep the
// blind alive under the tail-cube XOR that would zero a bare additive blind in
// char 2) and B_j fresh uniform multilinear columns supplied by the caller.
// gamma is drawn AFTER H = XOR_x blind(x) is absorbed, so the chain runs from
// claim0 + gamma*H and a cheating prover cannot adapt the blind to gamma.
//
// The caller owns the B_j columns (commits them on the hiding PCS and opens
// B_j(zeta) to discharge the terminal claim); this wrapper only does the
// round-message blinding + bookkeeping.  Legacy callers that pass nb==0 get the
// bit-identical non-zk sumcheck.
#pragma once
#include <cstdint>
#include <vector>
#include "p3_binius_sumcheck.cuh"

namespace bfz {

// integrand for the blinded zerocheck.  Column layout handed to bf_sumcheck:
//   [ user columns 0..Kc-1 (col 0 = E = eq(rz,.)) | B_0 .. B_{nb-1} ]
// ctx = ZcCtx: the user constraint (over the first Kc columns) + gamma + Kc/nb.
struct ZcCtx {
    BfConstraintFn user;     // user zerocheck integrand over w[0..Kc-1] (incl. E=w[0])
    const void*    uctx;
    bf128_t        gamma;
    int            Kc;       // user column count (E included)
    int            nb;       // number of blind columns
    int            eidx;     // index of the E=eq weight column (usually 0)
};
static inline bf128_t bfz_zc_integrand(const bf128_t* w, const void* c) {
    const ZcCtx* z = (const ZcCtx*)c;
    bf128_t v = z->user(w, z->uctx);              // E * C'(W)  (the real zerocheck)
    if (z->nb) {
        bf128_t e = w[z->eidx], ep = bf128_one(), bl = bf128_zero();
        for (int j = 0; j < z->nb; j++) {
            bl = bf128_add(bl, bf128_mul(ep, w[z->Kc + j]));   // E^j * B_j
            ep = bf128_mul(ep, e);
        }
        v = bf128_add(v, bf128_mul(z->gamma, bl));
    }
    return v;
}

// H = XOR_x ( B_0 + E*B_1 + ... + E^(nb-1)*B_{nb-1} )  over the l-cube
static inline bf128_t bfz_blind_sum(int l, const std::vector<bf128_t>& E,
                                    const std::vector<std::vector<bf128_t>>& B) {
    int nb = (int)B.size();
    bf128_t H = bf128_zero();
    for (size_t x = 0; x < ((size_t)1 << l); x++) {
        bf128_t e = E[x], ep = bf128_one(), bl = bf128_zero();
        for (int j = 0; j < nb; j++) { bl = bf128_add(bl, bf128_mul(ep, B[j][x])); ep = bf128_mul(ep, e); }
        H = bf128_add(H, bl);
    }
    return H;
}

// PROVE a zk-blinded zerocheck.  user_cols[0] must be the E=eq(rz,.) column and
// user_cols[1..] the constrained columns; the real integrand `user` has degree
// D over them.  B (nb columns, each length 2^l, fresh uniform) is appended and
// the wrapped integrand has degree max(D, nb).  Publishes H (absorbed) before
// gamma; on return, pf.finals holds [user finals | B finals] and *H_out = H.
static inline void bfz_zc_prove(int l, std::vector<std::vector<bf128_t>>& user_cols,
                                int D, BfConstraintFn user, const void* uctx,
                                std::vector<std::vector<bf128_t>>& B, int eidx,
                                fs::Transcript& tr, BfScProof& pf,
                                std::vector<bf128_t>& zeta, bf128_t& H_out) {
    int Kc = (int)user_cols.size(), nb = (int)B.size();
    int Dz = D > nb ? D : nb;
    H_out = nb ? bfz_blind_sum(l, user_cols[eidx], B) : bf128_zero();
    tr.absorb("bfz-zc-H", &H_out, sizeof H_out);
    ZcCtx z{user, uctx, nb ? bf_chal128(tr) : bf128_zero(), Kc, nb, eidx};
    std::vector<std::vector<bf128_t>> cols = std::move(user_cols);
    for (int j = 0; j < nb; j++) cols.push_back(std::move(B[j]));
    bf_sumcheck_prove(l, Kc + nb, cols, Dz, bfz_zc_integrand, &z, tr, pf, zeta);
}

// VERIFY: replays the chain from claim0 + gamma*H; returns the expected
// integrand value at zeta in *expected (the caller checks it == the wrapped
// integrand of the finals, having validated E_final and the B/user finals
// against its own eq recomputation and the hiding PCS openings).
static inline bool bfz_zc_verify(const BfScProof& pf, bf128_t claim0, int nb, int eidx,
                                 fs::Transcript& tr, std::vector<bf128_t>& zeta,
                                 bf128_t H, bf128_t* expected, bf128_t* gamma_out) {
    bf128_t claim = claim0;
    tr.absorb("bfz-zc-H", &H, sizeof H);
    bf128_t gamma = bf128_zero();
    if (nb) { gamma = bf_chal128(tr); claim = bf128_add(claim, bf128_mul(gamma, H)); }
    if (gamma_out) *gamma_out = gamma;
    (void)eidx;
    return bf_sumcheck_verify(pf, claim, tr, zeta, expected);
}

// helper: recompute the wrapped integrand at the finals (verifier side), given
// gamma and the E-final value; user_expected is E_f*C'(finals) computed by the
// caller, B_finals are pf.finals[Kc..Kc+nb).
static inline bf128_t bfz_zc_terminal(bf128_t user_expected, bf128_t gamma, bf128_t Efinal,
                                      const bf128_t* Bfinals, int nb) {
    bf128_t ep = bf128_one(), bl = bf128_zero();
    for (int j = 0; j < nb; j++) { bl = bf128_add(bl, bf128_mul(ep, Bfinals[j])); ep = bf128_mul(ep, Efinal); }
    return bf128_add(user_expected, bf128_mul(gamma, bl));
}

} // namespace bfz
