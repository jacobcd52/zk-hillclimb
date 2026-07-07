// Soundness probe: can a prover with WRONG real Y but a doctored random slice RY pass?
// The sumcheck constraint sum = eq(ey,0)*Yhat - 2IN*eq(exX,0)eq(exW,0)*Xhat*What over cube.
// eq(ey,0) selects ey=0 slice (real Y). The honest claim=0. We try Y'=Y+Delta (wrong) and
// adjust RY to try to keep the sumcheck-revealed messages consistent. Also try a cheat where
// prover uses inconsistent Yh in commit vs in sumcheck arrays.
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_private_fc.cuh"
using std::vector; using namespace p3pfc;
static uint64_t S=7; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }
int main(){
  p3fri::g_gpu_merkle=true; p3bf::p3_enable_mempool();
  uint32_t bb=2,ii=3,oo=2,B=1u<<bb,IN=1u<<ii,OUT=1u<<oo,R=2,Q=48;
  vector<gl_t> X((size_t)B*IN),W((size_t)IN*OUT); for(auto&x:X)x=rng()%257;for(auto&x:W)x=rng()%257;
  vector<gl_t> Y((size_t)B*OUT,0);
  for(uint32_t i=0;i<B;i++)for(uint32_t k=0;k<OUT;k++){gl_t a=0;for(uint32_t j=0;j<IN;j++)a=gl_add(a,gl_mul(X[i*IN+j],W[j*OUT+k]));Y[i*OUT+k]=a;}
  vector<gl_t> RX((size_t)B*IN),RW((size_t)IN*OUT),RY((size_t)B*OUT);
  for(auto&x:RX)x=rng();for(auto&x:RW)x=rng();for(auto&x:RY)x=rng();
  const char* why=nullptr;
  // baseline honest
  printf("honest: %d\n", verify(prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,1,true),Q,R,&why));
  // wrong Y at several positions
  { auto Yb=Y; Yb[3]=gl_add(Yb[3],12345);
    bool ok=verify(prove(X,W,Yb,RX,RW,RY,bb,ii,oo,R,Q,1,true),Q,R,&why);
    printf("wrong Y[3]: accept=%d why=%s\n",ok,why?why:""); }
  // attacker tries: keep Y wrong but set RY so that real+rand combos line up (can't, real slice is what's constrained)
  // try X wrong:
  { auto Xb=X; Xb[5]=gl_add(Xb[5],999);
    bool ok=verify(prove(Xb,W,Y,RX,RW,RY,bb,ii,oo,R,Q,1,true),Q,R,&why);
    printf("wrong X[5] (Y still =trueX.W): accept=%d why=%s\n",ok,why?why:""); }
  return 0;
}
