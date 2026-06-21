// P3.4b selftest: FC-layer matmul argument Y = X.W.
//   ./p3_matmul_selftest
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
#include "p3_basefold.cuh"
#include "p3_matmul.cuh"
using namespace p3mm;

static uint64_t rng_s = 0x5151;
static uint64_t rng(){ rng_s=rng_s*6364136223846793005ULL+1442695040888963407ULL; uint64_t z=rng_s; z^=z>>31; return z; }

static std::vector<gl_t> matmul(const std::vector<gl_t>& X, const std::vector<gl_t>& W,
                                uint32_t B, uint32_t IN, uint32_t OUT) {
    std::vector<gl_t> Y((size_t)B*OUT, 0);
    for (uint32_t i=0;i<B;i++) for (uint32_t k=0;k<OUT;k++){
        gl_t acc=0; for (uint32_t j=0;j<IN;j++) acc=gl_add(acc, gl_mul(X[i*IN+j], W[j*OUT+k]));
        Y[i*OUT+k]=acc;
    }
    return Y;
}

static int npass=0,nfail=0;
static void check(const char* n,bool c){ printf("  [%s] %s\n",c?"PASS":"FAIL",n); if(c)npass++; else nfail++; }

int main(){
    printf("=== P3.4b  FC matmul argument selftest ===\n");
    // a few shapes
    for (auto cfg : std::vector<std::vector<uint32_t>>{{1,2,1},{2,3,2},{3,2,3}}) {
        uint32_t bb=cfg[0],ii=cfg[1],oo=cfg[2], B=1u<<bb,IN=1u<<ii,OUT=1u<<oo, R=2,Q=24;
        std::vector<gl_t> X((size_t)B*IN), W((size_t)IN*OUT);
        for (auto& x:X) x=rng()%257; for (auto& x:W) x=rng()%257;
        auto Y=matmul(X,W,B,IN,OUT);
        auto pf=prove(X,W,Y,bb,ii,oo,R,Q);
        const char* why=nullptr; char nm[80]; snprintf(nm,sizeof nm,"honest B=%u IN=%u OUT=%u",B,IN,OUT);
        check(nm, verify(pf,&why)); if(why&&std::string(why)!="ok") printf("      why=%s\n",why);
    }

    // tamper battery on a fixed shape
    uint32_t bb=2,ii=3,oo=2, B=1u<<bb,IN=1u<<ii,OUT=1u<<oo, R=2,Q=24;
    std::vector<gl_t> X((size_t)B*IN), W((size_t)IN*OUT);
    for (auto& x:X) x=rng()%257; for (auto& x:W) x=rng()%257;
    auto Y=matmul(X,W,B,IN,OUT);
    const char* why=nullptr;

    { auto pf=prove(X,W,Y,bb,ii,oo,R,Q); check("baseline honest", verify(pf,&why)); }

    { auto Y2=Y; Y2[0]=gl_add(Y2[0],1);                       // committed Y != X.W
      auto pf=prove(X,W,Y2,bb,ii,oo,R,Q);
      check("wrong product (Y != X.W) -> reject", !verify(pf,&why)); printf("      why=%s\n",why); }

    { auto pf=prove(X,W,Y,bb,ii,oo,R,Q); pf.mm[0].s0=gl_add(pf.mm[0].s0,1);
      check("tamper matmul sumcheck msg -> reject", !verify(pf,&why)); printf("      why=%s\n",why); }

    { auto pf=prove(X,W,Y,bb,ii,oo,R,Q); pf.openX.y=gl_add(pf.openX.y,1);
      check("tamper opened X value -> reject", !verify(pf,&why)); printf("      why=%s\n",why); }

    { auto pf=prove(X,W,Y,bb,ii,oo,R,Q); pf.openW.queries[0].rounds[0].a=gl_add(pf.openW.queries[0].rounds[0].a,1);
      check("tamper W opening codeword -> reject", !verify(pf,&why)); printf("      why=%s\n",why); }

    printf("\nP3.4b MATMUL: %d passed, %d failed -> %s\n", npass,nfail, nfail==0?"ALL PASS":"FAIL");
    return nfail==0?0:1;
}
