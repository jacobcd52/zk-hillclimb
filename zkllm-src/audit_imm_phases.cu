// Audit tool: p3_matmul_bench2 with phase split (commit / sumcheck / open)
// and Q as a knob.  Same prove path as p3mm::prove but with timers.
// Args: <logB> <logIN> <logOUT> [R] [Q]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
#include "p3_basefold.cuh"
#include "p3_matmul.cuh"
#include "fs_transcript.hpp"
using namespace p3mm;
using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
static uint64_t rs=1; static uint64_t rng(){ rs=rs*6364136223846793005ULL+1; return rs; }
static std::vector<gl_t> mm_ref(const std::vector<gl_t>& X,const std::vector<gl_t>& W,uint32_t B,uint32_t IN,uint32_t OUT){
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
    auto Y=mm_ref(X,W,B,IN,OUT);
    { std::vector<gl_t> a; p3bf::commit_gpu(W,R,a); cudaDeviceSynchronize(); }  // prewarm
    p3fri::g_gpu_merkle = true;

    auto t0=clk::now();
    MatmulProof pf; pf.bb=bb; pf.ii=ii; pf.oo=oo; pf.R=R; pf.Q=Q;
    std::vector<gl_t> cwX, cwW, cwY;
    pf.rootX = p3bf::commit_gpu(X, R, cwX);
    pf.rootW = p3bf::commit_gpu(W, R, cwW);
    pf.rootY = p3bf::commit_gpu(Y, R, cwY);
    cudaDeviceSynchronize();
    auto t1=clk::now();
    fs::Transcript tr("p3-mm");
    tr.absorb("rX", pf.rootX.data(), 32); tr.absorb("rW", pf.rootW.data(), 32); tr.absorb("rY", pf.rootY.data(), 32);
    std::vector<gl_t> ri = chal_vec(tr, bb), rk = chal_vec(tr, oo);
    auto eqi = p3bf::build_eq(ri), eqk = p3bf::build_eq(rk);
    std::vector<gl_t> A(IN,0), Bv(IN,0);
    for (uint32_t j = 0; j < IN; j++) {
        gl_t a=0,b=0;
        for (uint32_t i = 0; i < B;   i++) a = gl_add(a, gl_mul(X[i*IN+j], eqi[i]));
        for (uint32_t k = 0; k < OUT; k++) b = gl_add(b, gl_mul(W[j*OUT+k], eqk[k]));
        A[j]=a; Bv[j]=b;
    }
    std::vector<gl_t> curA=A, curB=Bv, rj;
    for (uint32_t r = 0; r < ii; r++) {
        uint32_t half = (uint32_t)curA.size()/2;
        gl_t s0=0,s1=0,s2=0;
        for (uint32_t b=0;b<half;b++){
            gl_t cl=curA[2*b],ch=curA[2*b+1],wl=curB[2*b],wh=curB[2*b+1];
            s0=gl_add(s0,gl_mul(cl,wl)); s1=gl_add(s1,gl_mul(ch,wh));
            gl_t c2=gl_sub(gl_add(ch,ch),cl), w2=gl_sub(gl_add(wh,wh),wl);
            s2=gl_add(s2,gl_mul(c2,w2));
        }
        p3bf::SumMsg m{s0,s1,s2}; pf.mm.push_back(m); tr.absorb("mm",&m,sizeof(m));
        gl_t a=p3bf::alpha_from(tr); rj.push_back(a);
        std::vector<gl_t> nA(half), nB(half);
        for (uint32_t b=0;b<half;b++){
            nA[b]=gl_add(curA[2*b], gl_mul(a, gl_sub(curA[2*b+1],curA[2*b])));
            nB[b]=gl_add(curB[2*b], gl_mul(a, gl_sub(curB[2*b+1],curB[2*b])));
        }
        curA=nA; curB=nB;
    }
    auto t2=clk::now();
    std::vector<gl_t> zX = concat(rj, ri), zW = concat(rk, rj), zY = concat(rk, ri);
    gl_t yX = p3bf::eval_h(X, p3bf::build_eq(zX));
    gl_t yW = p3bf::eval_h(W, p3bf::build_eq(zW));
    gl_t yY = p3bf::eval_h(Y, p3bf::build_eq(zY));
    pf.openX = p3bf::prove_eval(X, zX, yX, R, Q, cwX, "p3-mm-X");
    pf.openW = p3bf::prove_eval(W, zW, yW, R, Q, cwW, "p3-mm-W");
    pf.openY = p3bf::prove_eval(Y, zY, yY, R, Q, cwY, "p3-mm-Y");
    cudaDeviceSynchronize();
    auto t3=clk::now();
    const char* why=nullptr; bool ok=verify(pf,&why);
    printf("PHASES B=%u IN=%u OUT=%u R=%u Q=%u ok=%d commit_ms=%.1f sumcheck_ms=%.1f open_ms=%.1f total_ms=%.1f\n",
           B,IN,OUT,R,Q,ok?1:0,ms(t0,t1),ms(t1,t2),ms(t2,t3),ms(t0,t3));
    return ok?0:1;
}
