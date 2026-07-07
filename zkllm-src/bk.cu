#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_private_fc.cuh"
using namespace p3pfc; using std::vector;
static uint64_t S=1; static gl_t rng(){S=S*6364136223846793005ULL+1;uint64_t z=S;z^=z>>31;return z%GL_P;}
int main(int c,char**v){ uint32_t bb=atoi(v[1]),ii=atoi(v[2]),oo=atoi(v[3]),R=1,Q=64;
 uint32_t B=1u<<bb,IN=1u<<ii,OUT=1u<<oo;
 vector<gl_t> X((size_t)B*IN),W((size_t)IN*OUT),Y((size_t)B*OUT,0);
 for(auto&x:X)x=rng()%257;for(auto&x:W)x=rng()%257;
 for(uint32_t i=0;i<B;i++)for(uint32_t k=0;k<OUT;k++){gl_t a=0;for(uint32_t j=0;j<IN;j++)a=gl_add(a,gl_mul(X[i*IN+j],W[j*OUT+k]));Y[i*OUT+k]=a;}
 vector<gl_t> RX((size_t)B*IN),RW((size_t)IN*OUT),RY((size_t)B*OUT);
 for(auto&x:RX)x=rng();for(auto&x:RW)x=rng();for(auto&x:RY)x=rng();
 p3fri::g_gpu_merkle=true; p3bf::p3_enable_mempool();
 { vector<gl_t> d(1024,1ull),t; p3bf::commit_gpu(d,1,t); cudaDeviceSynchronize(); }
 Timing tm; auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,123,true,&tm); cudaDeviceSynchronize();
 printf("commit=%.1f prep=%.1f sumcheck=%.1f open=%.1f\n",tm.commit_ms,tm.prep_ms,tm.sumcheck_ms,tm.open_ms);
 return 0;}
