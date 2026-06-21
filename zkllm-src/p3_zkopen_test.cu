// P3.5 ZK query-opening hiding test (the second leakage channel).
//   ./p3_zkopen_test
//
// A FRI/Basefold opening reveals Q codeword positions.  Each codeword value is a
// linear combination of the whole witness, so revealing them leaks.  Masking: commit
// a random codeword cw_e, draw lambda != 0, reveal positions of cw_f + lambda*cw_e.
// With cw_e uniform the revealed value is a one-time pad -> uniform, witness-independent.
//
// Checks the HIDING NECESSARY CONDITION (revealed values uniform & witness-independent)
// with a NEGATIVE CONTROL (no mask -> reveals cw_f -> witness-dependent / non-uniform ->
// the test fails, proving it has teeth).  Binding of the combined codeword to the
// witness commitment is argued in the random-oracle model (not unit-testable) -- see
// P3_PRIVACY_DESIGN.md.  A codeword position is a single point-evaluation poly(w^pos).
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
using std::vector;

static uint64_t S=99; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }
static int np=0,nf=0; static void ck(const char* n,bool c){ printf("  [%s] %s\n",c?"PASS":"FAIL",n); if(c)np++; else nf++; }
static const int B=257, M=20000; static const double UNIF_HI=370.0;
static double chisq(const vector<gl_t>& v){ vector<long> h(B,0); for(auto x:v) h[(int)(x%B)]++;
    double e=(double)v.size()/B,c=0; for(int i=0;i<B;i++){ double d=h[i]-e; c+=d*d/e;} return c; }

// poly(x) by Horner over coeff vector
static gl_t poly_eval(const vector<gl_t>& c, gl_t x){ gl_t p=0; for(int i=(int)c.size()-1;i>=0;i--) p=gl_add(gl_mul(p,x),c[i]); return p; }

int main(){
    printf("=== P3.5 ZK query-opening hiding test ===\n");
    uint32_t v=8, R=2, N=1u<<v, pos=13;
    gl_t xpos = gl_pow(gl_root_of_unity(v+R), pos);   // the domain point for codeword index `pos`
    vector<gl_t> f(N); for(auto&x:f) x=rng();

    // revealed codeword value at `pos` of cw_f + lambda*cw_e (mask=false -> just cw_f)
    auto sample=[&](const vector<gl_t>& w, uint64_t eseed, bool mask)->gl_t{
        gl_t base=poly_eval(w,xpos);
        if(!mask) return base;
        uint64_t s=eseed; vector<gl_t> e(w.size()); for(auto&x:e){ s=s*6364136223846793005ULL+1; x=(s^(s>>31))%GL_P; }
        s=s*2862933555777941757ULL+3; gl_t lam=((s^(s>>29))%(GL_P-1))+1;   // nonzero
        return gl_add(base, gl_mul(lam, poly_eval(e,xpos)));
    };

    vector<gl_t> masked(M); for(int i=0;i<M;i++) masked[i]=sample(f,3000+i,true);
    double c_mask=chisq(masked);
    printf("    chi-sq(masked reveal)=%.0f  (uniform ~256)\n", c_mask);
    ck("masked revealed value is uniform", c_mask < UNIF_HI);

    vector<gl_t> f2(N); for(auto&x:f2) x=rng();
    vector<gl_t> masked2(M); for(int i=0;i<M;i++) masked2[i]=sample(f2,3000+i,true);
    ck("different witness -> still uniform (independent)", chisq(masked2) < UNIF_HI);

    vector<gl_t> ctl(M); for(int i=0;i<M;i++) ctl[i]=sample(f,3000+i,false);
    double c_ctl=chisq(ctl);
    printf("    chi-sq(unmasked control)=%.0f  (should be huge)\n", c_ctl);
    ck("negative control: unmasked reveal is NON-uniform (test has teeth)", c_ctl > 10000.0);
    vector<gl_t> ctl2(M); for(int i=0;i<M;i++) ctl2[i]=sample(f2,3000+i,false);
    ck("negative control: unmasked leaks witness (f vs f2 differ)", !(ctl[0]==ctl2[0]));

    printf("\nP3.5 ZK-OPEN: %d passed, %d failed -> %s\n", np,nf, nf==0?"ALL PASS":"FAIL");
    return nf==0?0:1;
}
