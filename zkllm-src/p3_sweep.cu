// Parametric speed sweep for the ZK private FC prover.
//   ./p3_sweep <bb> <ii> <oo> <R> <Q>   (dims = 2^bb x 2^ii -> 2^oo, padded powers of 2)
// Generates random int8-range data, runs the GPU private prover + verify, prints one CSV line:
//   B,IN,OUT,prove_ms,verify_ms,proof_kb,ok
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_fri.cuh"
#include "p3_private_fc.cuh"
using namespace p3pfc; using std::vector;
using clk=std::chrono::high_resolution_clock; static double ms(clk::time_point a,clk::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static uint64_t S=1; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }
static size_t evb(const p3bf::EvalProof&e){ size_t b=e.roots.size()*32+e.msgs.size()*24+e.final_word.size()*8+e.z.size()*8+8;
    for(auto&q:e.queries)for(auto&r:q.rounds) b+=16+r.pa.size()*32+r.pb.size()*32; return b; }
int main(int argc,char**argv){
    if(argc<6){ fprintf(stderr,"usage: bb ii oo R Q\n"); return 2; }
    uint32_t bb=atoi(argv[1]),ii=atoi(argv[2]),oo=atoi(argv[3]),R=atoi(argv[4]),Q=atoi(argv[5]);
    uint32_t B=1u<<bb,IN=1u<<ii,OUT=1u<<oo;
    vector<gl_t> X((size_t)B*IN),W((size_t)IN*OUT);
    for(auto&x:X)x=rng()%257; for(auto&x:W)x=rng()%257;
    vector<gl_t> Y((size_t)B*OUT,0);
    for(uint32_t i=0;i<B;i++)for(uint32_t k=0;k<OUT;k++){gl_t a=0;for(uint32_t j=0;j<IN;j++)a=gl_add(a,gl_mul(X[i*IN+j],W[j*OUT+k]));Y[i*OUT+k]=a;}
    vector<gl_t> RX((size_t)B*IN),RW((size_t)IN*OUT),RY((size_t)B*OUT);
    for(auto&x:RX)x=rng();for(auto&x:RW)x=rng();for(auto&x:RY)x=rng();
    p3fri::g_gpu_merkle=true; p3bf::p3_enable_mempool();
    { vector<gl_t> t; p3bf::commit_gpu(W,R,t); cudaDeviceSynchronize(); }   // prewarm
    auto t0=clk::now(); auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,12345,true); cudaDeviceSynchronize(); auto t1=clk::now();
    const char* why=nullptr; bool ok=verify(pf,Q,R,&why); auto t2=clk::now();
    size_t bytes = pf.msgs.empty()?0 : evb(pf.openX)+evb(pf.openW)+evb(pf.openY)+pf.msgs.size()*24+128;
    printf("%u,%u,%u,%.1f,%.1f,%.1f,%d\n",B,IN,OUT,ms(t0,t1),ms(t1,t2),bytes/1024.0,ok?1:0);
    return ok?0:1;
}
