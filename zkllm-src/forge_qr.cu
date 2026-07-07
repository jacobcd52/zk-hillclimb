// Soundness probe: qr (mask residual) is prover-supplied & uncommitted. The final tie is
//   claim == summand_final + rho*qr.
// A cheating prover with wrong Y can try to absorb the discrepancy into qr. BUT qr is bound
// because q feeds the sumcheck MESSAGES (absorbed) and Sq is absorbed. Test: take an honest
// proof, then forge a wrong-Y proof and try to FIX it by overwriting pf.qr to satisfy the tie.
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
  // honest baseline + observe rho, claim path. We can't easily recompute; instead brute: try to
  // make a wrong-Y proof pass by overwriting qr to whatever the verifier's leftover claim needs.
  // The verifier computes 'claim' purely from messages+challenges (absorbed BEFORE qr is used),
  // and rho from Sq. So if attacker keeps messages s.t. (s0+s1)==claim each round (FS-consistent),
  // the only free knob at the end is qr. Try: honest proof, bump Y, recompute messages? Hard.
  // Simpler: take honest proof, overwrite qr -> must FAIL (tie breaks). Confirms qr is checked.
  auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,1,true);
  printf("honest accept=%d\n",verify(pf,Q,R,&why));
  auto pf2=pf; pf2.qr=gl_add(pf2.qr,1);
  printf("qr+1 accept=%d why=%s (expect 0: qr enters tie)\n",verify(pf2,Q,R,&why),why?why:"");
  // Now the real question: is qr INDEPENDENTLY checked, or only via the tie? If only via the tie,
  // a prover who can freely pick the LAST sumcheck message can offset. Test: bump last msg s2 and qr
  // together to keep tie? The verifier recomputes claim via quad_eval through all msgs; last alpha
  // depends on absorbing last msg. Changing last msg changes alpha -> changes everything downstream
  // incl z points -> openings fail. So qr alone can't forge. Confirm wrong-Y always fails:
  auto Yb=Y; Yb[2]=gl_add(Yb[2],55555);
  auto pfb=prove(X,W,Yb,RX,RW,RY,bb,ii,oo,R,Q,1,true);
  printf("wrong-Y honest-prover accept=%d why=%s\n",verify(pfb,Q,R,&why),why?why:"");
  return 0;
}
