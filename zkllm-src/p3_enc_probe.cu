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
  // host encode of one W-sized poly (N=2^14, R=2)
  uint32_t v=14,R=2,N=1u<<v; std::vector<gl_t> c(N); for(auto&x:c)x=rng()%257;
  auto t0=clk::now(); auto cw=p3bf::rs_encode(c,R); auto t1=clk::now();
  p3fri::Merkle mk; mk.build(cw); auto t2=clk::now();
  printf("host rs_encode N=2^%u -> 2^%u : %8.1f ms\n",v,v+R,ms(t0,t1));
  printf("host Merkle build (2^%u leaves): %8.1f ms\n",v+R,ms(t1,t2));
  printf("(GPU NTT encode of 2^23 measured earlier ~10 ms; this poly is 2^%u, ~%.0fx smaller)\n",v+R,(double)(1<<23)/(1<<(v+R)));
  return 0;
}
