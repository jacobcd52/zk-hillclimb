// Selftest for p3_binius_fri.cuh (design doc section 21.17): the additive-NTT
// FRI low-degree test over the tower.  Teeth: a genuine rate-1/4 codeword folds
// to a constant and ACCEPTS; a codeword with one corrupted symbol (Merkle
// rebuilt honestly so paths pass) is caught by the fold-consistency queries and
// REJECTS; a fully random word REJECTS; tampering a root / final constant /
// query leaf REJECTS.  Also checks proof size is logarithmic.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_binius_fri.cuh"

static int np_=0,nf_=0;
static void ck(const char* w,bool ok){printf("  [%s] %s\n",ok?"PASS":"FAIL",w);if(ok)np_++;else nf_++;}
static uint64_t rs_=0x1234abcd5678ef01ULL;
static uint64_t rnd(){rs_^=rs_<<13;rs_^=rs_>>7;rs_^=rs_<<17;return rs_;}
static bf128_t rnd128(){return {rnd(),rnd()};}

// encode a message (low k coeffs nonzero) into a rate-1/4 codeword of length
// 2^lm via the T_128 limb-wise additive NTT (novel-basis coeffs -> evaluations)
static std::vector<bf128_t> encode(const std::vector<bf128_t>& msg_low, int lm) {
    size_t n = (size_t)1 << lm;
    std::vector<bf128_t> c(n, bf128_zero());
    for (size_t i = 0; i < msg_low.size(); i++) c[i] = msg_low[i];
    BfNtt nt; bfntt_init(nt, lm);
    bfntt_fwd_host128(nt, c.data());
    return c;
}

int main(){
    size_t proofsz[2]={0,0}, cwsz[2]={0,0};
    for (int cfg=0; cfg<2; cfg++){
        int lm = cfg?14:10, R = BFFRI_RATE_LOG, Q=40;
        size_t n=(size_t)1<<lm, k=n>>R;            // message length (rate 1/4)
        std::vector<bf128_t> msg(k); for(auto&x:msg)x=rnd128();
        std::vector<bf128_t> cw = encode(msg, lm);

        fs::Transcript tp("bffri-test");
        BfFriProof pf; bffri_prove(cw, lm, R, Q, tp, pf);
        auto vfy=[&](const BfFriProof& q){ fs::Transcript tv("bffri-test"); return bffri_verify(q,R,tv); };
        char m[160];
        snprintf(m,sizeof m,"honest rate-1/4 codeword folds to a constant + ACCEPTS (lm=%d, %d layers, proof %.1f KB vs cw %.1f KB)",
                 lm, pf.nlayers, pf.bytes()/1024.0, n*16/1024.0);
        ck(m, vfy(pf));
        proofsz[cfg]=pf.bytes(); cwsz[cfg]=n*16;

        if(cfg==0) continue;
        // corrupted codeword (one symbol), Merkle rebuilt honestly -> fold queries catch it
        {
            std::vector<bf128_t> bad=cw; bad[123].lo^=1;
            fs::Transcript t2("bffri-test"); BfFriProof p2; bffri_prove(bad, lm, R, Q, t2, p2);
            ck("corrupted codeword (1 symbol, honest Merkle) REJECTS", !vfy(p2));
        }
        // fully random word rejects
        {
            std::vector<bf128_t> rw(n); for(auto&x:rw)x=rnd128();
            fs::Transcript t3("bffri-test"); BfFriProof p3; bffri_prove(rw, lm, R, Q, t3, p3);
            ck("fully random word REJECTS (not low-degree)", !vfy(p3));
        }
        { auto q=pf; q.roots[40]^=1; ck("tampered layer root REJECTS", !vfy(q)); }
        { auto q=pf; q.final_const.lo^=1; ck("tampered final constant REJECTS", !vfy(q)); }
        { auto q=pf; q.qd[5].hi^=1; ck("tampered query leaf REJECTS", !vfy(q)); }
    }
    // FRI's defining property: proof scales LOGARITHMICALLY, not linearly. The
    // codeword grew 16x (lm 10->14) but the proof should grow far less (layer
    // count + query-chain length are both O(log n)).
    { double cwr=(double)cwsz[1]/cwsz[0], pr=(double)proofsz[1]/proofsz[0];
      char m[160]; snprintf(m,sizeof m,"proof scales LOGARITHMICALLY: codeword 16x bigger -> proof only %.1fx (vs %.0fx linear)",pr,cwr);
      ck(m, pr < cwr/4); }
    printf("\nBINIUS-FRI: %d passed, %d failed -> %s\n",np_,nf_,nf_==0?"ALL PASS":"FAIL");
    return nf_==0?0:1;
}
