// P3 GL2 Basefold opening selftest (soundness-upgraded, ~2^-116). ./p3_basefold_gl2_selftest
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_gl2.cuh"
#include "p3_basefold.cuh"
#include "p3_basefold_gl2.cuh"
using namespace p3bf2;
static uint64_t s=0xABCD; static uint64_t rng(){ s=s*6364136223846793005ULL+1; uint64_t z=s; z^=z>>31; return z; }
static int np=0,nf=0; static void ck(const char* n,bool c){ printf("  [%s] %s\n",c?"PASS":"FAIL",n); if(c)np++; else nf++; }
int main(){
    printf("=== P3 GL2 Basefold opening selftest ===\n");
    for(auto cfg:std::vector<std::pair<uint32_t,uint32_t>>{{4,1},{6,2},{8,1}}){
        uint32_t v=cfg.first,R=cfg.second,N=1u<<v;
        std::vector<gl_t> cb(N); for(auto&x:cb) x=rng()%GL_P;          // base-field witness
        std::vector<gl2_t> z(v); for(auto&x:z) x=gl2_t{rng()%GL_P,rng()%GL_P}; // GL2 opening point
        std::vector<gl2_t> ce(N); for(uint32_t i=0;i<N;i++) ce[i]=gl2_from(cb[i]);
        gl2_t y=eval_h(ce,build_eq(z));
        std::vector<gl2_t> cw; auto root=commit(cb,R,cw);
        auto pf=prove_eval(cb,z,y,R,32,cw); const char* why=nullptr; char nm[64];
        snprintf(nm,sizeof nm,"honest open v=%u R=%u",v,R); ck(nm,verify_eval(pf,"p3-bf2",&why));
        if(why&&std::string(why)!="ok") printf("      why=%s\n",why);
    }
    // tamper battery
    uint32_t v=8,R=2,N=1u<<v; std::vector<gl_t> cb(N); for(auto&x:cb) x=rng()%GL_P;
    std::vector<gl2_t> z(v); for(auto&x:z) x=gl2_t{rng()%GL_P,rng()%GL_P};
    std::vector<gl2_t> ce(N); for(uint32_t i=0;i<N;i++) ce[i]=gl2_from(cb[i]);
    gl2_t y=eval_h(ce,build_eq(z)); std::vector<gl2_t> cw; commit(cb,R,cw); const char* why=nullptr;
    { auto pf=prove_eval(cb,z,y,R,32,cw); ck("baseline honest",verify_eval(pf,"p3-bf2",&why)); }
    { auto pf=prove_eval(cb,z,gl2_add(y,gl2_one()),R,32,cw); ck("wrong value -> reject",!verify_eval(pf,"p3-bf2",&why)); printf("      why=%s\n",why); }
    { auto pf=prove_eval(cb,z,y,R,32,cw); pf.msgs[0].s0=gl2_add(pf.msgs[0].s0,gl2_one()); ck("tamper sumcheck -> reject",!verify_eval(pf,"p3-bf2",&why)); printf("      why=%s\n",why); }
    { auto pf=prove_eval(cb,z,y,R,32,cw); pf.queries[0].rounds[0].a=gl2_add(pf.queries[0].rounds[0].a,gl2_one()); ck("tamper codeword -> reject",!verify_eval(pf,"p3-bf2",&why)); printf("      why=%s\n",why); }
    { auto pf=prove_eval(cb,z,y,R,32,cw); pf.final_word[0]=gl2_add(pf.final_word[0],gl2_one()); ck("tamper final -> reject",!verify_eval(pf,"p3-bf2",&why)); printf("      why=%s\n",why); }
    printf("\nP3 GL2-BASEFOLD: %d passed, %d failed -> %s\n", np,nf, nf==0?"ALL PASS":"FAIL");
    return nf==0?0:1;
}
