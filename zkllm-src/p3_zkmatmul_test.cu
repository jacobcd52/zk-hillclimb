// P3.5 ZK matmul-sumcheck test (honest, negative-controlled). ./p3_zkmatmul_test
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_zkmatmul.cuh"
using namespace p3zkmm;
static uint64_t S=5; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }
static int np=0,nf=0; static void ck(const char* n,bool c){ printf("  [%s] %s\n",c?"PASS":"FAIL",n); if(c)np++; else nf++; }
static const int B=257, M=20000; static const double UNIF_HI=370.0;
static double chisq(const std::vector<gl_t>& v){ std::vector<long> h(B,0); for(auto x:v) h[(int)(x%B)]++;
    double e=(double)v.size()/B,c=0; for(int i=0;i<B;i++){ double d=h[i]-e; c+=d*d/e;} return c; }

int main(){
    printf("=== P3.5 ZK matmul-sumcheck honest test ===\n");
    uint32_t v=8, N=1u<<v; std::string seed="zkmm";
    std::vector<gl_t> A(N),Bv(N); for(auto&x:A)x=rng(); for(auto&x:Bv)x=rng();
    gl_t c=0; for(uint32_t j=0;j<N;j++) c=gl_add(c,gl_mul(A[j],Bv[j]));   // c = sum A*B

    // soundness
    { std::vector<gl_t> r; auto pf=prove(A,Bv,c,seed,1,r);
      gl_t Ar=mle(A,r),Br=mle(Bv,r);
      ck("honest accepts", verify(pf,c,seed,v,Ar,Br));
      ck("wrong c rejects", !verify(pf,gl_add(c,1),seed,v,Ar,Br));
      auto pf2=pf; pf2.msgs[0].s0=gl_add(pf2.msgs[0].s0,1);
      ck("tampered message rejects", !verify(pf2,c,seed,v,Ar,Br));
      ck("wrong final eval rejects", !verify(pf,c,seed,v,gl_add(Ar,1),Br)); }

    // HVZK by simulation
    { std::vector<gl_t> r; gl_t Ar,Br; auto sp=simulate(c,seed,v,7,r,Ar,Br);
      ck("simulated (witnessless) transcript ACCEPTS", verify(sp,c,seed,v,Ar,Br)); }

    // ZK: round-0 message uniform vs simulated; negative control
    std::vector<gl_t> real(M),sim(M),ctl(M);
    for(int i=0;i<M;i++){ std::vector<gl_t> r; auto pf=prove(A,Bv,c,seed,1000+i,r); real[i]=pf.msgs[0].s0; }
    for(int i=0;i<M;i++){ std::vector<gl_t> r; gl_t Ar,Br; auto sp=simulate(c,seed,v,9000+i,r,Ar,Br); sim[i]=sp.msgs[0].s0; }
    for(int i=0;i<M;i++){ std::vector<gl_t> r; auto pf=prove(A,Bv,c,seed,1000+i,r,/*mask=*/false); ctl[i]=pf.msgs[0].s0; }
    double cr=chisq(real), cs=chisq(sim), cc=chisq(ctl);
    printf("    chi-sq real=%.0f sim=%.0f control(unmasked)=%.0f\n", cr, cs, cc);
    ck("real transcript uniform", cr<UNIF_HI);
    ck("simulated matches (uniform)", cs<UNIF_HI);
    ck("negative control: unmasked NON-uniform (teeth)", cc>10000.0);

    // witness independence: A2,B2 with same c
    std::vector<gl_t> A2(N),B2(N); for(auto&x:A2)x=rng(); for(auto&x:B2)x=rng();
    gl_t c2=0; for(uint32_t j=0;j<N;j++) c2=gl_add(c2,gl_mul(A2[j],B2[j]));
    A2[0]=gl_add(A2[0], gl_mul(gl_sub(c,c2), gl_inv(B2[0])));   // force sum A2*B2 = c
    std::vector<gl_t> r2(M); for(int i=0;i<M;i++){ std::vector<gl_t> r; auto pf=prove(A2,B2,c,seed,1000+i,r); r2[i]=pf.msgs[0].s0; }
    ck("different witness -> same uniform dist", chisq(r2)<UNIF_HI);

    printf("\nP3.5 ZK-MATMUL: %d passed, %d failed -> %s\n", np,nf, nf==0?"ALL PASS":"FAIL");
    return nf==0?0:1;
}
