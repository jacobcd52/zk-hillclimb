// P3.5 capstone on a REAL llama-68m FC layer (up_proj, 768->3072, int8 weights).
//   ./p3_private_fc_llama          (reads llama_fc/{X.bin,W.bin,dims.txt})
// Pads IN,OUT to powers of 2, proves Y=X.W privately, verifies, prints timing breakdown.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_fri.cuh"
#include "p3_private_fc.cuh"
using namespace p3pfc; using std::vector;
using clk=std::chrono::high_resolution_clock;
static double ms(clk::time_point a,clk::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
static uint64_t S=1; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }
static gl_t fromi(int64_t v){ return v<0 ? (gl_t)(GL_P+(uint64_t)v) : (gl_t)v; }   // small |v| < p
static uint32_t clog2(uint32_t n){ uint32_t l=0; while((1u<<l)<n) l++; return l; }

int main(){
    FILE* fd=fopen("llama_fc/dims.txt","r"); if(!fd){ printf("run the python export first\n"); return 1; }
    uint32_t B,IN0,OUT0; if(fscanf(fd,"%u %u %u",&B,&IN0,&OUT0)!=3){return 1;} fclose(fd);
    uint32_t bb=clog2(B), ii=clog2(IN0), oo=clog2(OUT0), IN=1u<<ii, OUT=1u<<oo;
    printf("=== private FC on llama-68m up_proj: B=%u IN=%u(->%u) OUT=%u(->%u) ===\n",B,IN0,IN,OUT0,OUT);

    vector<int64_t> Xr((size_t)B*IN0), Wr((size_t)IN0*OUT0);
    { FILE* f=fopen("llama_fc/X.bin","rb"); fread(Xr.data(),8,Xr.size(),f); fclose(f); }
    { FILE* f=fopen("llama_fc/W.bin","rb"); fread(Wr.data(),8,Wr.size(),f); fclose(f); }

    // load + zero-pad to powers of 2
    vector<gl_t> X((size_t)B*IN,0), W((size_t)IN*OUT,0);
    for(uint32_t i=0;i<B;i++) for(uint32_t j=0;j<IN0;j++) X[i*IN+j]=fromi(Xr[i*IN0+j]);
    for(uint32_t j=0;j<IN0;j++) for(uint32_t k=0;k<OUT0;k++) W[j*OUT+k]=fromi(Wr[j*OUT0+k]);
    // Y = X.W over the field
    vector<gl_t> Y((size_t)B*OUT,0);
    for(uint32_t i=0;i<B;i++) for(uint32_t k=0;k<OUT;k++){ gl_t a=0; for(uint32_t j=0;j<IN;j++) a=gl_add(a,gl_mul(X[i*IN+j],W[j*OUT+k])); Y[i*OUT+k]=a; }
    // random mask slices
    vector<gl_t> RX((size_t)B*IN),RW((size_t)IN*OUT),RY((size_t)B*OUT);
    for(auto&x:RX)x=rng(); for(auto&x:RW)x=rng(); for(auto&x:RY)x=rng();

    uint32_t R=2, Q=32;
    p3fri::g_gpu_merkle=true;
    { vector<gl_t> tmp; p3bf::commit_gpu(W,R,tmp); cudaDeviceSynchronize(); }  // prewarm CUDA

    Timing tm;
    auto t0=clk::now();
    auto pf=prove(X,W,Y,RX,RW,RY,bb,ii,oo,R,Q,12345,/*gpu=*/true,&tm);
    cudaDeviceSynchronize(); auto t1=clk::now();
    const char* why=nullptr; bool ok=verify(pf,&why); auto t2=clk::now();

    // proof size
    auto evb=[&](const p3bf::EvalProof&e){ size_t b=e.roots.size()*32+e.msgs.size()*24+e.final_word.size()*8+e.z.size()*8+8;
        for(auto&q:e.queries)for(auto&r:q.rounds) b+=16+r.pa.size()*32+r.pb.size()*32; return b; };
    size_t bytes=evb(pf.openX)+evb(pf.openW)+evb(pf.openY)+pf.msgs.size()*24+3*32+16;

    printf("verify: %s (%s)\n", ok?"ACCEPT":"REJECT", why);
    printf("--- prover timing breakdown ---\n");
    printf("  commit (3 augmented polys, GPU NTT+Merkle): %8.1f ms\n", tm.commit_ms);
    printf("  prep   (contractions + factor build)      : %8.1f ms\n", tm.prep_ms);
    printf("  sumcheck (combined ZK, %u rounds)          : %8.1f ms\n", ii+3, tm.sumcheck_ms);
    printf("  openings (3 ZK Basefold)                  : %8.1f ms\n", tm.open_ms);
    printf("  prove TOTAL                               : %8.1f ms\n", ms(t0,t1));
    printf("  verify                                    : %8.1f ms\n", ms(t1,t2));
    printf("  proof size                                : %8.2f KB\n", bytes/1024.0);
    return ok?0:1;
}
