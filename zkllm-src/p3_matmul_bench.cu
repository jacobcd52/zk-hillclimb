// P3.4 timing: integrity FC matmul prove/verify wall time + proof size (host).
#include <cstdio>
#include <vector>
#include <chrono>
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
#include "p3_basefold.cuh"
#include "p3_matmul.cuh"
using namespace p3mm;
using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }

static uint64_t rs=1; static uint64_t rng(){ rs=rs*6364136223846793005ULL+1; uint64_t z=rs; z^=z>>31; return z; }
static std::vector<gl_t> mm(const std::vector<gl_t>& X,const std::vector<gl_t>& W,uint32_t B,uint32_t IN,uint32_t OUT){
    std::vector<gl_t> Y((size_t)B*OUT,0);
    for(uint32_t i=0;i<B;i++)for(uint32_t k=0;k<OUT;k++){gl_t a=0;for(uint32_t j=0;j<IN;j++)a=gl_add(a,gl_mul(X[i*IN+j],W[j*OUT+k]));Y[i*OUT+k]=a;}
    return Y;
}
static size_t ev_bytes(const p3bf::EvalProof& e){
    size_t b = e.roots.size()*32 + e.msgs.size()*24 + e.final_word.size()*8 + e.z.size()*8 + 8;
    for(auto& q:e.queries) for(auto& r:q.rounds) b += 16 + r.pa.size()*32 + r.pb.size()*32;
    return b;
}

int main(){
    uint32_t bb=4,ii=10,oo=4, B=1u<<bb,IN=1u<<ii,OUT=1u<<oo, R=2,Q=32;
    printf("=== P3.4 matmul timing  B=%u IN=%u OUT=%u  R=%u Q=%u ===\n",B,IN,OUT,R,Q);
    std::vector<gl_t> X((size_t)B*IN),W((size_t)IN*OUT);
    for(auto&x:X)x=rng()%257; for(auto&x:W)x=rng()%257;
    auto Y=mm(X,W,B,IN,OUT);

    // sanity: GPU encode must give the same commitment as host encode
    { std::vector<gl_t> a,b; auto rh=p3bf::commit(W,R,a); auto rg=p3bf::commit_gpu(W,R,b);
      printf("gpu-encode == host-encode commitment: %s\n", (rh==rg)?"YES":"NO"); }

    auto t0=clk::now(); auto pf=prove(X,W,Y,bb,ii,oo,R,Q,/*gpu=*/true); auto t1=clk::now();
    const char* why=nullptr; bool ok=verify(pf,&why); auto t2=clk::now();

    size_t bytes = ev_bytes(pf.openX)+ev_bytes(pf.openW)+ev_bytes(pf.openY)+pf.mm.size()*24+96;
    printf("verify: %s (%s)\n", ok?"ACCEPT":"REJECT", why);
    printf("prove (GPU encode + host Merkle/fold): %8.1f ms\n", ms(t0,t1));
    printf("verify: %8.1f ms\n", ms(t1,t2));
    printf("proof : %8.2f KB\n", bytes/1024.0);
    return ok?0:1;
}
