// P3.5 capstone test: private FC prover Y=X.W (hides X,W,Y). ./p3_private_fc_test
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_private_fc.cuh"
using namespace p3pfc; using std::vector;
static uint64_t S=3; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }
static int np=0,nf=0; static void ck(const char* n,bool c){ printf("  [%s] %s\n",c?"PASS":"FAIL",n); if(c)np++; else nf++; }

static vector<gl_t> matmul(const vector<gl_t>&X,const vector<gl_t>&W,uint32_t B,uint32_t IN,uint32_t OUT){
    vector<gl_t> Y((size_t)B*OUT,0);
    for(uint32_t i=0;i<B;i++)for(uint32_t k=0;k<OUT;k++){ gl_t a=0; for(uint32_t j=0;j<IN;j++) a=gl_add(a,gl_mul(X[i*IN+j],W[j*OUT+k])); Y[i*OUT+k]=a; }
    return Y;
}
int main(){
    printf("=== P3.5 private FC prover test ===\n");
    uint32_t bb=2,ii=3,oo=2, B=1u<<bb,IN=1u<<ii,OUT=1u<<oo, R=2,Q=24;
    vector<gl_t> X((size_t)B*IN),W((size_t)IN*OUT); for(auto&x:X)x=rng()%257; for(auto&x:W)x=rng()%257;
    auto Y=matmul(X,W,B,IN,OUT);
    vector<gl_t> RX((size_t)B*IN),RW((size_t)IN*OUT),RY((size_t)B*OUT);
    for(auto&x:RX)x=rng(); for(auto&x:RW)x=rng(); for(auto&x:RY)x=rng();
    const char* why=nullptr;

    { auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,111);
      ck("honest private proof accepts", verify(pf,Q,R,&why)); if(why&&std::string(why)!="ok")printf("      why=%s\n",why); }

    { auto Y2=Y; Y2[0]=gl_add(Y2[0],1); auto pf=prove(X,W,Y2,RX,RW,RY,bb,ii,oo,R,Q,111);
      ck("wrong product Y!=X.W rejects", !verify(pf,Q,R,&why)); printf("      why=%s\n",why); }

    { auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,111); pf.msgs[0].s0=gl_add(pf.msgs[0].s0,1);
      ck("tampered sumcheck rejects", !verify(pf,Q,R,&why)); printf("      why=%s\n",why); }

    { auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,111); pf.openX.y=gl_add(pf.openX.y,1);
      ck("tampered opened value rejects", !verify(pf,Q,R,&why)); printf("      why=%s\n",why); }

    { auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,111); pf.openW.queries[0].rounds[0].a=gl_add(pf.openW.queries[0].rounds[0].a,1);
      ck("tampered opening codeword rejects", !verify(pf,Q,R,&why)); printf("      why=%s\n",why); }

    // hiding sanity: the opened value mX must NOT equal the real X~(ri,rj) (it's mask-mixed)
    { auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,111);
      auto pf0=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,111);   // same -> deterministic check it reproduces
      ck("deterministic (same inputs/seed -> same opened mX)", pf.openX.y==pf0.openX.y);
      // different random slice -> different opened value (mask actually mixes in)
      vector<gl_t> RX2(RX.size()); for(auto&x:RX2)x=rng();
      auto pf2=prove(X,W,Y,RX2,RW,RY,bb,ii,oo,R,Q,111);
      ck("opened mX depends on random slice (mask active)", pf.openX.y!=pf2.openX.y); printf("      mX=%llu mX'=%llu\n",(unsigned long long)pf.openX.y,(unsigned long long)pf2.openX.y); }

    // red-team CRITICAL-1 regression: Q=0 vacuous-accept forgery must be rejected
    { auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,111);
      pf.openX.Q=0; pf.openX.queries.clear();
      ck("Q=0 forgery rejected (params pinned to public Q,R)", !verify(pf,Q,R,&why)); printf("      why=%s\n",why);
      // also: wrong public R rejected
      ck("mismatched public R rejected", !verify(prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,111),Q,R+1,&why)); }

    printf("\nP3.5 PRIVATE-FC: %d passed, %d failed -> %s\n", np,nf, nf==0?"ALL PASS":"FAIL");
    return nf==0?0:1;
}
