// p3_binius_zkprove_test.cu -- FULL zero-knowledge prover for the REAL Hawkeye
// matmul zerocheck (design doc section 21.16).  Takes the actual committed
// witness (128 stacked slots, hawkeye_prod_vectors.bin) and proves the main
// HwF constraint -- the complete per-product fp8 semantics, K=90 D=3 zerocheck
// over the NCON=89 constrained columns -- with FULL zero-knowledge: masked
// hiding commitment (21.13.1) + char-2 Libra blinded sumcheck (21.14) + hiding
// multi-point openings (21.13) binding every final, field blinds committed as
// bit-bundles (21.15).  This is the 21.15 template applied to the real gadget.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <set>
#include "p3_binius_hawkeye.cuh"
#include "p3_binius_zkpcs.cuh"
#include "p3_binius_zksc.cuh"

using namespace bhw;
static int np_=0,nf_=0;
static void ck(const char* w,bool ok){printf("  [%s] %s\n",ok?"PASS":"FAIL",w);if(ok)np_++;else nf_++;}
static uint64_t mix(uint64_t z){z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;return z^(z>>31);}

static bool load_vectors(const char* path, std::vector<int64_t> raw[10]){
    FILE* f=fopen(path,"rb"); if(!f) return false;
    int64_t n; if(fread(&n,8,1,f)!=1){fclose(f);return false;}
    for(int c=0;c<10;c++){ raw[c].resize(n); if(fread(raw[c].data(),8,n,f)!=(size_t)n){fclose(f);return false;} }
    fclose(f); return true;
}
static bf128_t hwfC(const bf128_t* w, const void* ctx){ return (*(const HwF*)ctx)(w); }

int main(){
    bfz::G.on=true;
    std::vector<int64_t> raw[10];
    if(!load_vectors("hawkeye_prod_vectors.bin", raw)){ printf("FATAL: hawkeye_prod_vectors.bin missing\n"); return 1; }
    std::vector<uint8_t> bits; size_t n_real;
    int lN = bhw_build_bits(raw, bits, &n_real);
    size_t N=(size_t)1<<lN;
    const int NBLD=3;               // deg-3 zerocheck -> 3 E^j blinds
    const int BW=128;

    auto basis=[&](int u)->bf128_t{ bf128_t b=bf128_zero(); if(u<64)b.lo=1ull<<u; else b.hi=1ull<<(u-64); return b; };

    // ---------- FULL PROVE ----------
    // returns accept; if round_probe!=null just runs the blinded zerocheck and
    // records one round message (fast path, no commit/open).
    auto run=[&](uint64_t seed, bool blind_on, bool false_wit, std::vector<uint64_t>* round_probe)->bool{
        std::vector<uint8_t> wb = bits;
        if(false_wit) wb[(size_t)(LMAG+3)*N + 17] ^= 1;     // corrupt one committed bit
        // field blinds over lN (uniform) + their bit decomposition
        std::vector<std::vector<bf128_t>> B(NBLD, std::vector<bf128_t>(N, bf128_zero()));
        std::vector<std::vector<uint8_t>> Bb(NBLD*BW, std::vector<uint8_t>(N,0));
        for(int j=0;j<NBLD;j++) for(int u=0;u<BW;u++){ auto& col=Bb[j*BW+u];
            for(size_t i=0;i<N;i++){ uint8_t bit = blind_on ? (uint8_t)(mix(seed+0x1000*(j+1)+u*40503u+i*2654435761u)&1):0;
                col[i]=bit; if(bit) B[j][i]=bf128_add(B[j][i], basis(u)); } }

        fs::Transcript tr("bhwzk");
        // gammas need rz first; mirror bhw: absorb a stmt tag, draw rz, gammas
        tr.absorb("bhwzk-stmt",&lN,sizeof lN);
        // COMMIT witness (masked) unless round-probe fast path
        bfz::ZkParams pw; pw.l=lN+LOGSTK; pw.lrow=(lN+LOGSTK)-(lN+LOGSTK)/2; pw.lcol=(lN+LOGSTK)/2; pw.Q=32; bfz::G.Q=32;
        bfz::ZkCommit Cw;
        bfz::ZkParams pb; { int lb=lN; int slots=NBLD*BW; int ls=0; while((1<<ls)<slots)ls++; pb.l=lN+ls; pb.lrow=(lN+ls)-(lN+ls)/2; pb.lcol=(lN+ls)/2; pb.Q=32; }
        bfz::ZkCommit Cb;
        if(!round_probe){
            bfz::G.seed=seed*7+1; bfz::G.ctr=0;
            bfz::bfz_commit(pw, wb.data(), tr, Cw);
            // pack blind bits into a slot-major bit array (NBLD*BW slots x N)
            int nslot = 1<<(pb.l - lN);                    // slots available
            std::vector<uint8_t> bbits((size_t)nslot*N,0);
            for(int c=0;c<NBLD*BW;c++) memcpy(&bbits[(size_t)c*N], Bb[c].data(), N);
            bfz::G.seed=seed*7+2; bfz::G.ctr=0;
            bfz::bfz_commit(pb, bbits.data(), tr, Cb);
        }
        // rz, gammas
        std::vector<bf128_t> rz(lN); for(auto&x:rz)x=bf_chal128(tr);
        std::vector<bf128_t> g; bhw_gammas(tr,g);
        // build user columns [eq | 89 constrained columns]
        std::vector<std::vector<bf128_t>> uc(1+NCON, std::vector<bf128_t>(N));
        bf_eq_table(rz.data(), lN, uc[0]);
        for(int j=0;j<NCON;j++) for(size_t i=0;i<N;i++) uc[1+j][i]= wb[(size_t)j*N+i]?bf128_one():bf128_zero();
        HwF cf; cf.g=g.data();
        std::vector<std::vector<bf128_t>> Bcopy=B;
        BfScProof zc; std::vector<bf128_t> zeta; bf128_t H;
        bfz::bfz_zc_prove(lN, uc, HwF::D, hwfC, &cf, Bcopy, 0, tr, zc, zeta, H);
        if(round_probe){ round_probe->push_back(zc.rounds[(size_t)2*(HwF::D+1)+1].lo); return true; }

        // OPEN: constrained columns (slots 0..88) at zeta from Cw ; blind bits from Cb
        auto ptW=[&](int slot){ std::vector<bf128_t> p(pw.l); for(int t=0;t<lN;t++)p[t]=zeta[t];
            for(int t=0;t<pw.l-lN;t++) p[lN+t]=(slot>>t)&1?bf128_one():bf128_zero(); return p; };
        auto ptB=[&](int slot){ std::vector<bf128_t> p(pb.l); for(int t=0;t<lN;t++)p[t]=zeta[t];
            for(int t=0;t<pb.l-lN;t++) p[lN+t]=(slot>>t)&1?bf128_one():bf128_zero(); return p; };
        std::vector<std::vector<bf128_t>> ptsW,ptsB; std::vector<bf128_t> vW,vB;
        // wb is NSTK(128) slots x N; pw.l = lN+LOGSTK covers exactly 128 slots.
        for(int j=0;j<NCON;j++){ ptsW.push_back(ptW(j)); vW.push_back(bf_ml_eval_bits(wb.data(), pw.l, ptsW.back().data())); }
        { std::vector<uint8_t> bbits; int nslot=1<<(pb.l-lN); bbits.assign((size_t)nslot*N,0);
          for(int c=0;c<NBLD*BW;c++) memcpy(&bbits[(size_t)c*N], Bb[c].data(), N);
          for(int c=0;c<NBLD*BW;c++){ ptsB.push_back(ptB(c)); vB.push_back(bf_ml_eval_bits(bbits.data(), pb.l, ptsB.back().data())); } }
        std::vector<const bf128_t*> rsW,rsB; for(auto&p:ptsW)rsW.push_back(p.data()); for(auto&p:ptsB)rsB.push_back(p.data());
        bfz::ZkProofM opW,opB;
        bfz::bfz_open_multi(Cw, rsW, vW, tr, opW);
        bfz::bfz_open_multi(Cb, rsB, vB, tr, opB);

        // ---------- VERIFY ----------
        fs::Transcript tv("bhwzk"); tv.absorb("bhwzk-stmt",&lN,sizeof lN);
        tv.absorb("bfz-root",Cw.root,32); tv.absorb("bfz-sseed",&Cw.sseed,sizeof Cw.sseed);
        tv.absorb("bfz-root",Cb.root,32); tv.absorb("bfz-sseed",&Cb.sseed,sizeof Cb.sseed);
        std::vector<bf128_t> rzv(lN); for(auto&x:rzv)x=bf_chal128(tv);
        std::vector<bf128_t> gv; bhw_gammas(tv,gv);
        std::vector<bf128_t> zv; bf128_t expected,gamma;
        if(!bfz::bfz_zc_verify(zc, bf128_zero(), NBLD, 0, tv, zv, H, &expected,&gamma)) return false;
        // E_final = eq(rz, zeta)
        bf128_t ef=bf128_one(); for(int t=0;t<lN;t++) ef=bf128_mul(ef,bf128_add(bf128_one(),bf128_add(rzv[t],zv[t])));
        // open-verify: binds the column evals to the commitments
        std::vector<const bf128_t*> rsW2,rsB2; for(auto&p:ptsW)rsW2.push_back(p.data()); for(auto&p:ptsB)rsB2.push_back(p.data());
        if(!bfz::bfz_verify_multi(Cw.p,Cw.root,rsW2,vW,tv,opW)) return false;
        if(!bfz::bfz_verify_multi(Cb.p,Cb.root,rsB2,vB,tv,opB)) return false;
        // reconstruct the HwF finals from opened column evals: finals[0]=E_f,
        // finals[1+j]=column j eval, finals[Kc+b]=blind b = XOR_u basis_u*biteval
        int Kc=1+NCON;
        std::vector<bf128_t> fin(Kc+NBLD);
        fin[0]=ef; for(int j=0;j<NCON;j++) fin[1+j]=vW[j];
        bf128_t Bf[8];
        for(int b=0;b<NBLD;b++){ Bf[b]=bf128_zero(); for(int u=0;u<BW;u++) Bf[b]=bf128_add(Bf[b],bf128_mul(basis(u),vB[b*BW+u])); fin[Kc+b]=Bf[b]; }
        // bind: the sumcheck's own finals must equal these opened/reconstructed values
        bool bind=true;
        for(int j=0;j<NCON;j++) if(!bf128_eq(zc.finals[1+j],vW[j])) bind=false;
        if(!bf128_eq(zc.finals[0],ef)) bind=false;
        for(int b=0;b<NBLD;b++) if(!bf128_eq(zc.finals[Kc+b],Bf[b])) bind=false;
        // terminal: expected == HwF(finals) + gamma*blind(finals)
        HwF cfv; cfv.g=gv.data();
        bf128_t uexp=cfv(fin.data());
        bf128_t term=bfz::bfz_zc_terminal(uexp, gamma, ef, Bf, NBLD);
        return bind && bf128_eq(expected, term);
    };

    ck("FULL-ZK Hawkeye matmul zerocheck ACCEPTS on the REAL witness (masked commit + blinded sumcheck + hiding opens, all 89 columns + blinds bound)", run(0xA1,true,false,nullptr));
    ck("false witness (one committed MAG bit flipped) REJECTS", !run(0xA1,true,true,nullptr));
    ck("accepts with blind OFF (correctness preserved)", run(0xA1,false,false,nullptr));

    // hiding: main-zerocheck round message uniform across blind seeds (fast path)
    const int Nn=256;
    auto chi2=[&](std::vector<uint64_t>&xs){double w=0;for(int b=0;b<32;b++){long c1=0;for(auto x:xs)c1+=(x>>b)&1;long c0=(long)xs.size()-c1;double e=xs.size()/2.0,c=(c0-e)*(c0-e)/e+(c1-e)*(c1-e)/e;if(c>w)w=c;}return w;};
    std::vector<uint64_t> on;
    for(int s=0;s<Nn;s++) run(0x5000+s,true,false,&on);
    { double c=chi2(on); char m[128]; snprintf(m,sizeof m,"FULL-ZK: main HwF round m_2(z=1) UNIFORM across seeds (chi2 %.1f < 16, N=%d)",c,Nn); ck(m,c<16.0); }

    printf("\nBINIUS-ZKPROVE: %d passed, %d failed -> %s\n",np_,nf_,nf_==0?"ALL PASS":"FAIL");
    return nf_==0?0:1;
}
