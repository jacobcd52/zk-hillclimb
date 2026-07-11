// Parameterized integer-GEMM ZK prover timing (the "integerized" comparison
// point): proves Y = X.W over integers (plain sumcheck matmul, no fp8 accum
// semantics), GPU encode + GPU Merkle.  Args: <logB> <logIN> <logOUT> [R] [Q].
// Prints one parseable line: IMM B=.. IN=.. OUT=.. prove_ms=.. verify_ms=.. proof_kb=..
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
#include "p3_basefold.cuh"
#include "p3_matmul.cuh"
using namespace p3mm;
using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
static uint64_t rs=1; static uint64_t rng(){ rs=rs*6364136223846793005ULL+1; return rs; }
static size_t ev_bytes(const p3bf::EvalProof& e){
    size_t b = e.roots.size()*32 + e.msgs.size()*24 + e.final_word.size()*8 + e.z.size()*8 + 8;
    for(auto& q:e.queries) for(auto& r:q.rounds) b += 16 + r.pa.size()*32 + r.pb.size()*32;
    return b;
}
static std::vector<gl_t> mm(const std::vector<gl_t>& X,const std::vector<gl_t>& W,uint32_t B,uint32_t IN,uint32_t OUT){
    std::vector<gl_t> Y((size_t)B*OUT,0);
    for(uint32_t i=0;i<B;i++)for(uint32_t k=0;k<OUT;k++){gl_t a=0;for(uint32_t j=0;j<IN;j++)a=gl_add(a,gl_mul(X[i*IN+j],W[j*OUT+k]));Y[i*OUT+k]=a;}
    return Y;
}
int main(int argc,char**argv){
    if(argc<4){printf("usage: %s logB logIN logOUT [R] [Q]\n",argv[0]);return 1;}
    uint32_t bb=atoi(argv[1]),ii=atoi(argv[2]),oo=atoi(argv[3]);
    uint32_t R=argc>4?atoi(argv[4]):2, Q=argc>5?atoi(argv[5]):32;
    uint32_t B=1u<<bb,IN=1u<<ii,OUT=1u<<oo;
    std::vector<gl_t> X((size_t)B*IN),W((size_t)IN*OUT);
    for(auto&x:X)x=rng()%257; for(auto&x:W)x=rng()%257;
    auto Y=mm(X,W,B,IN,OUT);
    { std::vector<gl_t> a; p3bf::commit_gpu(W,R,a); cudaDeviceSynchronize(); }  // prewarm
    p3fri::g_gpu_merkle = true;
    auto t0=clk::now(); auto pf=prove(X,W,Y,bb,ii,oo,R,Q,true); cudaDeviceSynchronize(); auto t1=clk::now();
    const char* why=nullptr; bool ok=verify(pf,&why); auto t2=clk::now();
    size_t bytes = ev_bytes(pf.openX)+ev_bytes(pf.openW)+ev_bytes(pf.openY)+pf.mm.size()*24+96;
    printf("IMM B=%u IN=%u OUT=%u verify_ok=%d prove_ms=%.1f verify_ms=%.1f proof_kb=%.1f\n",
           B,IN,OUT,ok?1:0,ms(t0,t1),ms(t1,t2),bytes/1024.0);
    return ok?0:1;
}
