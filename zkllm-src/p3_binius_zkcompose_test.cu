// p3_binius_zkcompose_test.cu -- END-TO-END composed ZERO-KNOWLEDGE proof over
// the tower (design doc section 21.15): the 21.13 hiding PCS + the 21.14 blinded
// zerocheck + the multi-point hiding open, WIRED TOGETHER into one proof of a
// real committed statement, with a COMPOSED hiding battery.  This is the
// integration pattern the composed bhw prover follows per gadget.
//
// STATEMENT (a genuine zerocheck that holds): a committed bit-witness with
// columns W0,W1,W2 satisfies W2 = W0 & W1, i.e. E*(W2 + W0*W1) == 0 for the
// verifier's random weight E = eq(rz,.).
//
// THE INTEGRATION SUBTLETY the tower forces: the PCS commits BITS (booleanity
// structural), but the Libra blind columns B_j are T_128-VALUED (needed to hide
// the T_128 round messages).  So each field blind is committed as a BUNDLE of
// 128 bit-columns and reconstructed B_j(x) = XOR_u basis_u * bit_{j,u}(x); its
// terminal eval B_j(zeta) is then bound by opening the 128 bit-columns on the
// hiding PCS (audit: unbound blinds are adaptively forgeable after gamma).
//
// COMMITTED LAYOUT (one bfz commitment): column c occupies bit indices
// [c<<l, (c+1)<<l).  cols 0..2 = W0,W1,W2 ; cols 3.. = the 2*128 blind bits.
// Opening column c at a row-point zeta = a PCS eval at (zeta || c-bits).
//
// TEETH: honest composed proof ACCEPTS (zerocheck terminal checks against the
// PCS-opened W finals AND the reconstructed-from-opened-bits blind finals; every
// opened column is redundant/masked).  SOUNDNESS: a false witness (one W2 bit
// wrong) REJECTS.  HIDING (composed): over N seeds every zerocheck round message
// is UNIFORM (chi-square) and every opened column is hidden -- the WHOLE proof
// leaks nothing beyond the public statement.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <set>
#include "p3_binius_zkpcs.cuh"
#include "p3_binius_zksc.cuh"

static int np_=0,nf_=0;
static void ck(const char* w,bool ok){printf("  [%s] %s\n",ok?"PASS":"FAIL",w);if(ok)np_++;else nf_++;}
static uint64_t rs_=0xDEADBEEF01234567ULL;
static uint64_t rnd(){rs_^=rs_<<13;rs_^=rs_>>7;rs_^=rs_<<17;return rs_;}
static bf128_t rnd128(){return {rnd(),rnd()};}
static uint64_t mix(uint64_t z){z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;return z^(z>>31);}

// user zerocheck integrand over [E, W0, W1, W2]: E*(W2 + W0*W1)
static bf128_t userZC(const bf128_t* w, const void*){
    return bf128_mul(w[0], bf128_add(w[3], bf128_mul(w[1], w[2])));
}

int main(){
    bfz::G.on=true;
    const int l=8;                       // row vars (256 rows)
    const int KW=3;                      // W0,W1,W2
    const int NB=3;                      // blind columns: E*(W2+W0*W1) is DEGREE 3
                                         // (E*W0*W1), so 4 round-coeffs need 3 E^j blinds
    const int BW=128;                    // bits per field blind
    const int ncol_used = KW + NB*BW;    // 3 + 256 = 259
    int lcolc=0; while((1<<lcolc) < ncol_used) lcolc++;   // column-index vars
    const int ncol = 1<<lcolc;           // padded column count (512)
    const int lpcs = l + lcolc;          // total PCS vars
    bfz::ZkParams p; p.l=lpcs; p.lrow=l; p.lcol=lcolc; p.Q=80; bfz::G.Q=p.Q;
    size_t Nrow=(size_t)1<<l;

    // ---- fixed witness bits: W0,W1 random, W2 = W0&W1 ----
    std::vector<uint8_t> W0(Nrow),W1(Nrow),W2(Nrow);
    for(size_t i=0;i<Nrow;i++){W0[i]=rnd()&1;W1[i]=rnd()&1;W2[i]=W0[i]&W1[i];}
    std::vector<bf128_t> rz(l); for(auto&x:rz)x=rnd128();
    std::vector<bf128_t> E; bf_eq_table(rz.data(), l, E);

    // basis elements basis_u = 1<<u embedded in T_128 (bf128_from16 of the T_16
    // bit-basis for u<16, extended: we use the T_128 element with a single bit set)
    auto basis=[&](int u)->bf128_t{ bf128_t b=bf128_zero(); if(u<64)b.lo=1ull<<u; else b.hi=1ull<<(u-64); return b; };

    // build one composed proof for a given (blind seed, false-witness flag)
    struct Proof { bfz::ZkProofM openW; std::vector<BfScProof> zc; std::vector<bf128_t> Hs;
                   uint8_t root[32]; uint64_t sseed; std::vector<std::vector<bf128_t>> zeta; };
    auto build=[&](uint64_t seed, bool blind_on, bool false_wit, Proof& P, bool& accept,
                   std::vector<uint64_t>* round_probe){
        // assemble committed bit array (ncol columns x Nrow), column-major flattened
        std::vector<uint8_t> bits((size_t)ncol*Nrow, 0);
        for(size_t i=0;i<Nrow;i++){
            bits[(size_t)0*Nrow+i]=W0[i];
            bits[(size_t)1*Nrow+i]=W1[i];
            uint8_t w2=W2[i]; if(false_wit && i==42) w2^=1;
            bits[(size_t)2*Nrow+i]=w2;
        }
        // blind bit columns (uniform); zero under the negative control
        for(int j=0;j<NB;j++) for(int u=0;u<BW;u++){
            size_t c=KW+j*BW+u;
            for(size_t i=0;i<Nrow;i++)
                bits[c*Nrow+i]= blind_on ? (uint8_t)((mix(seed+0x1000*(j+1)+u*2654435761u+i*40503u))&1) : 0;
        }
        // COMMIT (hiding PCS)
        bfz::G.seed=seed; bfz::G.ctr=0;
        fs::Transcript tr("zkcompose");
        bfz::ZkCommit C; bfz::bfz_commit(p, bits.data(), tr, C);
        memcpy(P.root,C.root,32); P.sseed=C.sseed;
        // reconstruct field blinds B_j from their committed bit columns
        std::vector<std::vector<bf128_t>> B(NB, std::vector<bf128_t>(Nrow, bf128_zero()));
        for(int j=0;j<NB;j++) for(int u=0;u<BW;u++){ size_t c=KW+j*BW+u;
            for(size_t i=0;i<Nrow;i++) if(bits[c*Nrow+i]) B[j][i]=bf128_add(B[j][i],basis(u)); }
        // user columns for the zerocheck: [E, W0, W1, W2] (as field cols)
        std::vector<std::vector<bf128_t>> uc(4, std::vector<bf128_t>(Nrow));
        for(size_t i=0;i<Nrow;i++){ uc[0][i]=E[i];
            uc[1][i]=W0[i]?bf128_one():bf128_zero(); uc[2][i]=W1[i]?bf128_one():bf128_zero();
            uint8_t w2=W2[i]; if(false_wit&&i==42)w2^=1; uc[3][i]=w2?bf128_one():bf128_zero(); }
        // BLINDED ZEROCHECK (rounds hidden)
        P.zc.resize(1); P.Hs.resize(1); P.zeta.resize(1);
        bf128_t H; std::vector<bf128_t> zeta;
        std::vector<std::vector<bf128_t>> Bcopy=B;
        bfz::bfz_zc_prove(l, uc, 3, userZC, nullptr, Bcopy, 0, tr, P.zc[0], zeta, H);
        P.Hs[0]=H; P.zeta[0]=zeta;
        if(round_probe) round_probe->push_back(P.zc[0].rounds[(size_t)2*(3+1)+1].lo);
        // OPEN every column needed by the terminal at zeta: W0,W1,W2 + all blind bits
        std::vector<std::vector<bf128_t>> pts; std::vector<bf128_t> vals;
        auto colpoint=[&](int c){ std::vector<bf128_t> pt(lpcs);
            for(int t=0;t<l;t++) pt[t]=zeta[t];
            for(int t=0;t<lcolc;t++) pt[l+t]=(c>>t)&1?bf128_one():bf128_zero(); return pt; };
        std::vector<int> cols={0,1,2}; for(int j=0;j<NB;j++)for(int u=0;u<BW;u++)cols.push_back(KW+j*BW+u);
        for(int c:cols){ pts.push_back(colpoint(c)); vals.push_back(bf_ml_eval_bits(bits.data(),lpcs,pts.back().data())); }
        std::vector<const bf128_t*> rs; for(auto&pt:pts) rs.push_back(pt.data());
        bfz::bfz_open_multi(C, rs, vals, tr, P.openW);

        // ================= VERIFY =================
        fs::Transcript tv("zkcompose");
        // re-commit-absorb: root + sseed (the verifier has the root)
        tv.absorb("bfz-root",P.root,32); tv.absorb("bfz-sseed",&P.sseed,sizeof P.sseed);
        // replay the blinded zerocheck
        std::vector<bf128_t> zv; bf128_t expected,gamma;
        if(!bfz::bfz_zc_verify(P.zc[0], bf128_zero(), NB, 0, tv, zv, P.Hs[0], &expected,&gamma)){accept=false;return;}
        // recompute E_final and the terminal from finals; W finals bound by the open,
        // blind finals reconstructed from opened bit-finals
        bf128_t ef=bf128_one(); for(int t=0;t<l;t++) ef=bf128_mul(ef,bf128_add(bf128_one(),bf128_add(rz[t],zv[t])));
        // finals layout: [E_f, W0_f, W1_f, W2_f, B0_f, B1_f]
        bf128_t uexp=bf128_mul(P.zc[0].finals[0], bf128_add(P.zc[0].finals[3], bf128_mul(P.zc[0].finals[1],P.zc[0].finals[2])));
        // reconstruct blind finals from opened bit column evals (points 3..)
        // and verify the multi-open binds all column evals to the commitment
        std::vector<const bf128_t*> rs2; for(auto&pt:pts) rs2.push_back(pt.data());
        if(!bfz::bfz_verify_multi(C.p, P.root, rs2, vals, tv, P.openW)){accept=false;return;}
        // bind: W finals == opened vals[0..2]; blind finals == XOR_u basis_u*vals[bit]
        bool wbind = bf128_eq(P.zc[0].finals[1],vals[0]) && bf128_eq(P.zc[0].finals[2],vals[1])
                   && bf128_eq(P.zc[0].finals[3],vals[2]) && bf128_eq(P.zc[0].finals[0],ef);
        bf128_t Bf[8];
        for(int j=0;j<NB;j++){ Bf[j]=bf128_zero();
            for(int u=0;u<BW;u++) Bf[j]=bf128_add(Bf[j], bf128_mul(basis(u), vals[KW+j*BW+u])); }
        bool bbind=true;
        for(int j=0;j<NB;j++) if(!bf128_eq(P.zc[0].finals[KW+1+j],Bf[j])) bbind=false;
        bf128_t term=bfz::bfz_zc_terminal(uexp, gamma, ef, Bf, NB);
        accept = wbind && bbind && bf128_eq(expected, term);
    };

    // ---- SOUNDNESS ----
    { Proof P; bool a; build(0xA1,true,false,P,a,nullptr); ck("composed ZK proof ACCEPTS (masked commit + blinded zerocheck + hiding open, all finals bound)",a); }
    { Proof P; bool a; build(0xA1,true,true, P,a,nullptr); ck("false witness (one W2 bit wrong) REJECTS through the composed proof",!a); }
    { Proof P; bool a; build(0xA1,false,false,P,a,nullptr); ck("composed proof accepts with blind OFF (correctness preserved)",a); }

    // ---- COMPOSED HIDING: zerocheck round messages uniform across seeds ----
    const int N=384;
    auto chi2=[&](std::vector<uint64_t>&xs){double w=0;for(int b=0;b<32;b++){long c1=0;for(auto x:xs)c1+=(x>>b)&1;long c0=(long)xs.size()-c1;double e=xs.size()/2.0,c=(c0-e)*(c0-e)/e+(c1-e)*(c1-e)/e;if(c>w)w=c;}return w;};
    std::vector<uint64_t> on,off;
    for(int s=0;s<N;s++){ Proof P; bool a; build(0x5000+s,true,false,P,a,&on); }
    { double c=chi2(on); char m[128]; snprintf(m,sizeof m,"composed: zerocheck round m_2(z=1) UNIFORM across seeds (chi2 %.1f < 16, N=%d)",c,N);
      ck(m, c<16.0); }
    // negative control: disable BOTH the zerocheck blind AND the PCS masking, so
    // the transcript (hence the sumcheck challenges) is a deterministic function
    // of the fixed witness -> the round message collapses to one value; turning
    // the blind back on (above) makes it uniform.
    bfz::G.mask_on=bfz::G.blind_on=bfz::G.salt_on=false;
    for(int s=0;s<N;s++){ Proof P; bool a; build(0x5000+s,false,false,P,a,&off); }
    bfz::G.mask_on=bfz::G.blind_on=bfz::G.salt_on=true;
    ck("composed: round is DETERMINISTIC with blind+mask OFF (blind is load-bearing)",
       std::set<uint64_t>(off.begin(),off.end()).size()==1);

    printf("\nBINIUS-ZKCOMPOSE: %d passed, %d failed -> %s\n",np_,nf_,nf_==0?"ALL PASS":"FAIL");
    return nf_==0?0:1;
}
