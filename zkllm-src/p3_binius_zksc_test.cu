// Standalone teeth for the ZK-BLINDED SUMCHECK over the tower (design doc
// section 21.14) -- the second half of lever 5.  A sumcheck's round
// polynomials m_s(z) = XOR_{y} C(w(y,z)) are witness functionals and LEAK; to
// make the whole composed proof zero-knowledge every round message must be
// hidden.  This is the Libra "degree-matched blind" (p3_zkc.cuh mechanism 2)
// specialised to a char-2 tower ZEROCHECK, validated in isolation.
//
// Statement (a genuine zerocheck that HOLDS): W2 = W0 & W1 (bit AND embedded in
// T_128), so C'(W) = W2 + W0*W1 == 0 everywhere and XOR_x eq(rz,x)*C'(W) = 0.
//
// Blind: add  gamma * ( B0 + E*B1 + E^2*B2 )  with B0,B1,B2 FRESH UNIFORM
// multilinear columns and E = eq(rz,x) the zerocheck weight.  The E^j factors
// are the char-2 fix: a bare additive blind sum_j B_j would VANISH under the
// tail-cube XOR (2^k = 0 in char 2) in every round but the last, leaving those
// round messages unblinded; multiplying by powers of the NON-CONSTANT weight E
// keeps the blind alive in every round and, being degree 0..D in E, spans all
// D+1 round-message coefficients.  The prover publishes H = XOR_x blind(x)
// BEFORE gamma is drawn; the chain runs from claim0 + gamma*H = gamma*H (the
// real zerocheck sums to 0) and ends at C'(finals)*E_f + gamma*blind(finals).
//
// Teeth:
//  SOUNDNESS: honest blinded zerocheck accepts (expected == C_zk(finals), and
//  E_f == eq(rz,zeta) independently); a false witness (one W2 bit wrong) makes
//  the real sum != 0 so the chain's round-0 check rp[0]+rp[1] != claim -> reject.
//  HIDING: round messages m_s(z) are UNIFORM across blind seeds (chi-square,
//  1 dof) in EVERY round -- including the early rounds a bare additive blind
//  would leave deterministic -- and collapse to a single deterministic value
//  under the blind_on negative control.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <set>
#include "p3_binius_sumcheck.cuh"
#include "p3_binius_zksc.cuh"

static int np_ = 0, nf_ = 0;
static void ck(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) np_++; else nf_++;
}
static uint64_t rs_ = 0xC0FFEE1234567ULL;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }
static bf128_t rnd128() { return {rnd(), rnd()}; }
static uint64_t mix(uint64_t z){ z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }

// column layout: 0=W0 1=W1 2=W2 3=E 4=B0 5=B1 6=B2 ; ctx = &gamma
static bf128_t Czk(const bf128_t* w, const void* ctx) {
    bf128_t g = *(const bf128_t*)ctx;
    bf128_t real = bf128_add(w[2], bf128_mul(w[0], w[1]));        // W2 + W0*W1  (==0 honest)
    bf128_t ic = bf128_mul(w[3], real);                          // E * C'
    bf128_t e = w[3];
    bf128_t blind = bf128_add(w[4], bf128_mul(e, w[5]));         // B0 + E*B1
    blind = bf128_add(blind, bf128_mul(bf128_mul(e, e), w[6]));  // + E^2*B2
    return bf128_add(ic, bf128_mul(g, blind));
}

int main() {
    const int l = 12, K = 7, D = 3;

    // ---- fixed witness: W0,W1 random bits, W2 = W0 & W1 ; rz random ----
    std::vector<uint8_t> b0(1u<<l), b1(1u<<l), b2(1u<<l);
    for (size_t i = 0; i < (1u<<l); i++) { b0[i]=rnd()&1; b1[i]=rnd()&1; b2[i]=b0[i]&b1[i]; }
    std::vector<bf128_t> rz(l); for (auto& x: rz) x = rnd128();
    std::vector<bf128_t> E; bf_eq_table(rz.data(), l, E);

    // build the (unfolded) column set for a given blind seed (B random or zero)
    auto build_cols = [&](uint64_t seed, bool blind_on, std::vector<std::vector<bf128_t>>& cols,
                          bf128_t& H) {
        cols.assign(K, std::vector<bf128_t>(1u<<l));
        for (size_t i=0;i<(1u<<l);i++){
            cols[0][i]=b0[i]?bf128_one():bf128_zero();
            cols[1][i]=b1[i]?bf128_one():bf128_zero();
            cols[2][i]=b2[i]?bf128_one():bf128_zero();
            cols[3][i]=E[i];
        }
        for (int j=0;j<3;j++) for (size_t i=0;i<(1u<<l);i++){
            cols[4+j][i] = blind_on ? bf128_t{mix(seed+0x100*(j+1)+i*2654435761u),
                                              mix(seed*3+0x55*(j+1)+i*40503u)} : bf128_zero();
        }
        // H = XOR_x ( B0 + E*B1 + E^2*B2 )
        H = bf128_zero();
        for (size_t i=0;i<(1u<<l);i++){
            bf128_t e=cols[3][i];
            bf128_t bl=bf128_add(cols[4][i], bf128_mul(e,cols[5][i]));
            bl=bf128_add(bl, bf128_mul(bf128_mul(e,e),cols[6][i]));
            H=bf128_add(H,bl);
        }
    };

    // one honest blinded prove; returns proof + the gamma used + accept flag
    auto prove_verify = [&](uint64_t seed, bool blind_on, BfScProof& pf, bf128_t& gamma,
                            bool& accept) {
        std::vector<std::vector<bf128_t>> cols; bf128_t H;
        build_cols(seed, blind_on, cols, H);
        fs::Transcript tp("bfzksc");
        tp.absorb("H", &H, sizeof H);
        gamma = bf_chal128(tp);                 // gamma AFTER H (soundness)
        std::vector<bf128_t> zeta;
        bf_sumcheck_prove(l, K, cols, D, Czk, &gamma, tp, pf, zeta);
        // verify
        fs::Transcript tv("bfzksc");
        tv.absorb("H", &H, sizeof H);
        bf128_t g2 = bf_chal128(tv);
        bf128_t claim = bf128_mul(g2, H);       // claim0 (=0) + gamma*H
        std::vector<bf128_t> zv; bf128_t expected;
        if (!bf_sumcheck_verify(pf, claim, tv, zv, &expected)) { accept=false; return; }
        // expected == C_zk(finals); and E_final == eq(rz, zeta) independently.
        // char-2 eq: eq(a,b) = prod_t (1 + a_t + b_t)
        bf128_t Cf = Czk(pf.finals.data(), &g2);
        bf128_t ef = bf128_one();
        for (int t = 0; t < l; t++)
            ef = bf128_mul(ef, bf128_add(bf128_one(), bf128_add(rz[t], zv[t])));
        accept = bf128_eq(expected, Cf) && bf128_eq(ef, pf.finals[3]);
    };

    // ---- SOUNDNESS ----
    { BfScProof pf; bf128_t g; bool acc; prove_verify(0xA1, true, pf, g, acc);
      ck("honest blinded zerocheck accepts (expected==C_zk(finals), E_f==eq(rz,zeta))", acc); }
    { BfScProof pf; bf128_t g; bool acc; prove_verify(0xA1, false, pf, g, acc);
      ck("honest zerocheck accepts with blind OFF (correctness preserved)", acc); }
    // false witness: flip one W2 bit -> real zerocheck != 0 -> reject
    {
        auto sav = b2[123]; b2[123] ^= 1;
        BfScProof pf; bf128_t g; bool acc; prove_verify(0xA1, true, pf, g, acc);
        ck("false witness (one W2 bit wrong) REJECTS through the blind", !acc);
        b2[123] = sav;
    }

    // ---- HIDING: round-message uniformity in EVERY round ----
    const int N = 512;
    // collect m_s[z] over N blind seeds, for a probe in an EARLY round (s=2,z=2)
    // -- the round a bare additive blind would leave deterministic -- and a mid
    // round (s=6,z=1).
    auto chi2 = [&](const std::vector<uint64_t>& xs){
        double worst=0;
        for (int bt=0; bt<32; bt++){ long c1=0; for(auto x:xs)c1+=(x>>bt)&1; long c0=(long)xs.size()-c1;
            double e=xs.size()/2.0, chi=(c0-e)*(c0-e)/e+(c1-e)*(c1-e)/e; if(chi>worst)worst=chi; }
        return worst;
    };
    auto probe = [&](int s, int z, bool blind_on, std::vector<uint64_t>& out){
        out.clear();
        for (int t=0;t<N;t++){
            std::vector<std::vector<bf128_t>> cols; bf128_t H;
            build_cols(0x5000 + t, blind_on, cols, H);
            fs::Transcript tp("bfzksc"); tp.absorb("H",&H,sizeof H);
            bf128_t g = bf_chal128(tp);
            BfScProof pf; std::vector<bf128_t> zeta;
            bf_sumcheck_prove(l, K, cols, D, Czk, &g, tp, pf, zeta);
            out.push_back(pf.rounds[(size_t)s*(D+1)+z].lo);
        }
    };
    for (auto sr : std::vector<std::pair<int,int>>{{2,2},{6,1},{0,3},{10,0}}) {
        std::vector<uint64_t> on, off;
        probe(sr.first, sr.second, true, on);
        probe(sr.first, sr.second, false, off);
        char m[160];
        double c = chi2(on);
        snprintf(m,sizeof m,"round m_%d(z=%d) UNIFORM under blind (chi2 %.1f < 16, N=%d)",sr.first,sr.second,c,N);
        ck(m, c < 16.0);
        snprintf(m,sizeof m,"round m_%d(z=%d) DETERMINISTIC without blind (leak the blind removes)",sr.first,sr.second);
        ck(m, std::set<uint64_t>(off.begin(),off.end()).size()==1);
    }

    // ---- REUSABLE HEADER (p3_binius_zksc.cuh) drop-in: same statement via
    //      bfz_zc_prove/verify with the blind columns supplied by the caller ----
    {
        // user cols: 0=E, 1=W0, 2=W1, 3=W2 ; user integrand E*(W2+W0*W1)
        auto userC = [](const bf128_t* w, const void*)->bf128_t {
            return bf128_mul(w[0], bf128_add(w[3], bf128_mul(w[1], w[2])));
        };
        auto run_hdr = [&](uint64_t seed, bool blind_on, bool false_wit, BfScProof& pf,
                           bf128_t& gamma, bool& accept){
            std::vector<std::vector<bf128_t>> uc(4, std::vector<bf128_t>(1u<<l));
            for (size_t i=0;i<(1u<<l);i++){ uint8_t w2=b2[i]; if(false_wit && i==77) w2^=1;
                uc[0][i]=E[i]; uc[1][i]=b0[i]?bf128_one():bf128_zero();
                uc[2][i]=b1[i]?bf128_one():bf128_zero(); uc[3][i]=w2?bf128_one():bf128_zero(); }
            std::vector<std::vector<bf128_t>> B(3, std::vector<bf128_t>(1u<<l, bf128_zero()));
            if (blind_on) for(int j=0;j<3;j++) for(size_t i=0;i<(1u<<l);i++)
                B[j][i]=bf128_t{mix(seed+0x777*(j+1)+i*11),mix(seed*5+0x33*(j+1)+i*7)};
            fs::Transcript tp("bfzsc-hdr"); bf128_t H; std::vector<bf128_t> zeta;
            bfz::bfz_zc_prove(l, uc, 3, userC, nullptr, B, 0, tp, pf, zeta, H);
            // verify
            fs::Transcript tv("bfzsc-hdr");
            std::vector<bf128_t> zv; bf128_t expected, g;
            if(!bfz::bfz_zc_verify(pf, bf128_zero(), 3, 0, tv, zv, H, &expected, &g)){accept=false;return;}
            gamma=g;
            // E_final, user_expected = E_f*(W2_f + W0_f*W1_f) from finals (0..3), B finals 4..6
            bf128_t ef=bf128_one(); for(int t=0;t<l;t++) ef=bf128_mul(ef,bf128_add(bf128_one(),bf128_add(rz[t],zv[t])));
            bf128_t uexp=bf128_mul(pf.finals[0], bf128_add(pf.finals[3], bf128_mul(pf.finals[1],pf.finals[2])));
            bf128_t term=bfz::bfz_zc_terminal(uexp, g, ef, &pf.finals[4], 3);
            accept = bf128_eq(expected, term) && bf128_eq(ef, pf.finals[0]);
        };
        { BfScProof pf; bf128_t g; bool a; run_hdr(0xB1,true,false,pf,g,a);
          ck("header bfz_zc: honest blinded zerocheck accepts", a); }
        { BfScProof pf; bf128_t g; bool a; run_hdr(0xB1,true,true,pf,g,a);
          ck("header bfz_zc: false witness REJECTS", !a); }
        // header round-message uniformity in an early round
        std::vector<uint64_t> on;
        for(int t=0;t<N;t++){ BfScProof pf; bf128_t g; bool a; run_hdr(0x6000+t,true,false,pf,g,a);
            on.push_back(pf.rounds[(size_t)2*(3+1)+2].lo); }
        double c=chi2(on);
        char hm[128]; snprintf(hm,sizeof hm,"header bfz_zc: round m_2(z=2) UNIFORM (chi2 %.1f < 16)",c);
        ck(hm, c<16.0);
    }

    printf("\nBINIUS-ZKSC: %d passed, %d failed -> %s\n", np_, nf_, nf_==0?"ALL PASS":"FAIL");
    return nf_==0?0:1;
}
