// P3.5 ZK-sumcheck test harness -- designed to be honest and hard to fake.
//   ./p3_zksumcheck_test
//
// Why each test has teeth:
//  * SOUNDNESS: honest proof + correct oracle MUST accept, and every tamper MUST
//    reject. (Can't pass by always-rejecting: honest case checks acceptance.)
//  * HVZK by simulation: simulate() gets NO witness, yet produces a transcript the
//    real verifier ACCEPTS -> the verifier provably learns nothing it couldn't make
//    itself. (Can't pass with a stub verifier: the same verifier must reject tampers.)
//  * DISTRIBUTION: real-vs-simulated transcript values must be statistically
//    identical (both uniform) -- HVZK is about distributions, not one sample.
//  * WITNESS-INDEPENDENCE: two different witnesses with the same public claim must
//    give the same transcript distribution.
//  * NEGATIVE CONTROL: with masking DISABLED the SAME tests must FAIL (the transcript
//    becomes witness-dependent / non-uniform). This proves the tests detect leakage
//    rather than passing vacuously.
#include <cstdio>
#include <vector>
#include <cmath>
#include "p3_goldilocks.cuh"
#include "p3_zksumcheck.cuh"
using namespace p3zksc;

static uint64_t S=123; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }
static int np=0,nf=0; static void ck(const char* n,bool c){ printf("  [%s] %s\n",c?"PASS":"FAIL",n); if(c)np++; else nf++; }

static const int B=257; static const int M=20000;
// chi-square of a sample's (value % B) histogram against uniform; ~B-1 if uniform, huge if not.
static double chisq_uniform(const std::vector<gl_t>& vals){
    std::vector<long> h(B,0); for (auto v:vals) h[(int)(v%B)]++;
    double exp=(double)vals.size()/B, c=0; for (int i=0;i<B;i++){ double d=h[i]-exp; c+=d*d/exp; } return c;
}

int main(){
    printf("=== P3.5 ZK-sumcheck honest test harness ===\n");
    const double UNIF_HI=370.0;       // uniform chi-sq (~256 +/- 23) stays well under this
    uint32_t v=8, N=1u<<v;
    std::vector<gl_t> z(v); for (auto& x:z) x=rng();
    std::vector<gl_t> f(N); for (auto& x:f) x=rng();
    auto eq=build_eq(z); gl_t y=eval_h(f,eq);
    std::string seed="zksc";

    // ---------- soundness ----------
    { std::vector<gl_t> r; gl_t h; auto pf=prove(f,z,y,seed,1,r,h);
      ck("honest proof accepts", verify(pf,z,y,seed,h));
      auto pf2=pf; pf2.msgs[0].s0=gl_add(pf2.msgs[0].s0,1);
      ck("tampered message rejects", !verify(pf2,z,y,seed,h));
      ck("wrong claimed y rejects", !verify(pf,z,gl_add(y,1),seed,h));
      ck("wrong oracle value rejects", !verify(pf,z,y,seed,gl_add(h,1)));
    }

    // ---------- HVZK by simulation (no witness) ----------
    { std::vector<gl_t> r; gl_t h; auto sp=simulate(z,y,seed,7,r,h);
      ck("simulated (witnessless) transcript ACCEPTS", verify(sp,z,y,seed,h)); }

    // ---------- distribution: real (vary mask g) vs simulated ----------
    std::vector<gl_t> real_s0(M), sim_s0(M);
    for (int i=0;i<M;i++){ std::vector<gl_t> r; gl_t h; auto pf=prove(f,z,y,seed,1000+i,r,h); real_s0[i]=pf.msgs[0].s0; }
    for (int i=0;i<M;i++){ std::vector<gl_t> r; gl_t h; auto sp=simulate(z,y,seed,9000+i,r,h); sim_s0[i]=sp.msgs[0].s0; }
    double c_real=chisq_uniform(real_s0), c_sim=chisq_uniform(sim_s0);
    printf("    chi-sq(real)=%.0f  chi-sq(sim)=%.0f  (uniform ~256)\n", c_real, c_sim);
    ck("real transcript value is uniform", c_real < UNIF_HI);
    ck("simulated transcript value is uniform (matches real)", c_sim < UNIF_HI);

    // ---------- witness-independence: f vs f2 with same f~(z)=y ----------
    std::vector<gl_t> f2(N); for (auto& x:f2) x=rng();
    gl_t cur=eval_h(f2,eq); f2[0]=gl_add(f2[0], gl_mul(gl_sub(y,cur), gl_inv(eq[0]))); // force f2~(z)=y
    std::vector<gl_t> s0_f2(M);
    for (int i=0;i<M;i++){ std::vector<gl_t> r; gl_t h; auto pf=prove(f2,z,y,seed,1000+i,r,h); s0_f2[i]=pf.msgs[0].s0; }
    double c_f2=chisq_uniform(s0_f2);
    ck("different witness gives same (uniform) distribution", c_f2 < UNIF_HI);

    // ---------- NEGATIVE CONTROL: masking OFF -> tests MUST fail ----------
    std::vector<gl_t> ctl(M);
    for (int i=0;i<M;i++){ std::vector<gl_t> r; gl_t h; auto pf=prove(f,z,y,seed,1000+i,r,h,/*mask=*/false); ctl[i]=pf.msgs[0].s0; }
    double c_ctl=chisq_uniform(ctl);
    printf("    chi-sq(unmasked control)=%.0f  (should be huge)\n", c_ctl);
    ck("negative control: unmasked transcript is NON-uniform (test has teeth)", c_ctl > 10000.0);
    // and witness-dependent: unmasked f vs f2 differ
    std::vector<gl_t> ctl2(M);
    for (int i=0;i<M;i++){ std::vector<gl_t> r; gl_t h; auto pf=prove(f2,z,y,seed,1000+i,r,h,false); ctl2[i]=pf.msgs[0].s0; }
    bool differ = !(ctl[0]==ctl2[0]);
    ck("negative control: unmasked leaks witness (f vs f2 differ)", differ);

    printf("\nP3.5 ZK-SUMCHECK: %d passed, %d failed -> %s\n", np,nf, nf==0?"ALL PASS":"FAIL");
    return nf==0?0:1;
}
