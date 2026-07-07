#include <cstdio>
#include <vector>
#include <chrono>
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
#include "p3_basefold.cuh"
using clk=std::chrono::high_resolution_clock;
static double ms(clk::time_point a,clk::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static uint64_t rs=1; static uint64_t rng(){rs=rs*6364136223846793005ULL+1;return rs;}
int main(){
  // W-sized opening: v=14 (IN*OUT=2^14), R=2, Q=32
  uint32_t v=14,R=2,Q=32,N=1u<<v; std::vector<gl_t> c(N); for(auto&x:c)x=rng()%257;
  std::vector<gl_t> z(v); for(auto&x:z)x=rng()%GL_P;
  gl_t y=p3bf::eval_h(c,p3bf::build_eq(z));

  auto t0=clk::now(); auto cw=p3bf::rs_encode_gpu(c,R); auto t1=clk::now();        // gpu encode
  p3fri::Merkle mk; mk.build(cw); auto t2=clk::now();                               // host merkle (commit)
  auto pf=p3bf::prove_eval(c,z,y,R,Q,cw,"x"); auto t3=clk::now();                   // full opening (rebuilds trees+folds)
  printf("gpu encode (2^%u): %6.1f ms\n",v+R,ms(t0,t1));
  printf("host commit Merkle: %6.1f ms\n",ms(t1,t2));
  printf("full opening (host merkle/round + folds + queries): %6.1f ms\n",ms(t2,t3));
  return 0;
}
