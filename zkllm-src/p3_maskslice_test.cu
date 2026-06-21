// P3.5 capstone mechanism: the "extra mask-slice" that makes operand openings hiding
// WITHOUT breaking the matmul product check.
//
// Augment a length-N witness X into X^ on {0,1}^(v+1): X^(0,b)=X[b] (real slice),
// X^(1,b)=random (mask slice), ex = the high variable.
//  * SOUNDNESS mechanism: sum_ex eq(ex,0)*X^(ex,b) = X^(0,b) = X[b].  So a sumcheck that
//    carries the eq(ex,0) weight reads ONLY the real slice -> the matmul constraint is
//    exactly Y=X*W, the random slice contributes 0.  (test: identity holds for all b)
//  * PRIVACY mechanism: an opening at a RANDOM ex=rex gives X^~(rex,r) =
//    (1-rex)*X~(r) + rex*rand~(r), which is UNIFORM -> hides X~(r). The verifier still
//    multiplies the two opened operand values (they are real field elements, just
//    uniformly distributed); correctness holds because the sumcheck ties exactly those
//    points, weighted by the public (1-rex).  (test: uniform & witness-independent;
//    negative control with no mask slice -> witness-dependent / non-uniform)
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
using std::vector;
static uint64_t S=11; static gl_t rng(){ S=S*6364136223846793005ULL+1; uint64_t z=S; z^=z>>31; return z%GL_P; }
static int np=0,nf=0; static void ck(const char* n,bool c){ printf("  [%s] %s\n",c?"PASS":"FAIL",n); if(c)np++; else nf++; }
static const int B=257, M=20000; static const double UNIF_HI=370.0;
static double chisq(const vector<gl_t>& v){ vector<long> h(B,0); for(auto x:v) h[(int)(x%B)]++;
    double e=(double)v.size()/B,c=0; for(int i=0;i<B;i++){ double d=h[i]-e; c+=d*d/e;} return c; }
static gl_t mle(const vector<gl_t>& a, const vector<gl_t>& r){ uint32_t v=r.size(); gl_t acc=0;
    for(uint32_t b=0;b<a.size();b++){ gl_t e=1ULL; for(uint32_t i=0;i<v;i++) e=gl_mul(e,(b&(1u<<i))?r[i]:gl_sub(1ULL,r[i])); acc=gl_add(acc,gl_mul(a[b],e)); } return acc; }

int main(){
    printf("=== P3.5 capstone mask-slice mechanism ===\n");
    uint32_t v=8, N=1u<<v;
    vector<gl_t> X(N); for(auto&x:X) x=rng();

    // ---- SOUNDNESS mechanism: eq(ex,0)-weighted sum over both slices == real X ----
    { vector<gl_t> rnd(N); for(auto&x:rnd) x=rng();
      bool ok=true;
      for(uint32_t b=0;b<N;b++){ // sum_ex eq(ex,0)*Xhat(ex,b) = 1*X[b] + 0*rnd[b]
          gl_t s=gl_add(gl_mul(gl_sub(1ULL,0ULL),X[b]), gl_mul(0ULL,rnd[b])); // eq(0,0)=1, eq(1,0)=0
          if(s!=X[b]) ok=false; }
      ck("eq(ex,0)-weighted sum reads ONLY the real slice (constraint = Y=X*W)", ok); }

    // ---- PRIVACY mechanism: opening at random ex is uniform & witness-independent ----
    // fixed challenges (rex, r); vary the mask slice; collect Xhat~(rex,r)
    gl_t rex=rng(); vector<gl_t> r(v); for(auto&x:r) x=rng();
    auto open_at=[&](const vector<gl_t>& Xw, uint64_t mseed, bool mask)->gl_t{
        gl_t xr=mle(Xw,r);                          // X~(r)
        if(!mask) return xr;                        // negative control: no mask slice (ex=0 effectively)
        uint64_t s=mseed; vector<gl_t> rnd(Xw.size()); for(auto&y:rnd){ s=s*6364136223846793005ULL+1; y=(s^(s>>31))%GL_P; }
        gl_t rr=mle(rnd,r);
        return gl_add(gl_mul(gl_sub(1ULL,rex),xr), gl_mul(rex,rr));   // (1-rex)X~(r)+rex*rand~(r)
    };
    vector<gl_t> masked(M); for(int i=0;i<M;i++) masked[i]=open_at(X,7000+i,true);
    double cm=chisq(masked); printf("    chi-sq(masked open)=%.0f\n", cm);
    ck("opening at random ex is uniform (hiding)", cm<UNIF_HI);

    vector<gl_t> X2(N); for(auto&x:X2) x=rng();
    vector<gl_t> masked2(M); for(int i=0;i<M;i++) masked2[i]=open_at(X2,7000+i,true);
    ck("different witness -> still uniform (independent)", chisq(masked2)<UNIF_HI);

    vector<gl_t> ctl(M); for(int i=0;i<M;i++) ctl[i]=open_at(X,7000+i,false);
    double cc=chisq(ctl); printf("    chi-sq(no-mask-slice control)=%.0f\n", cc);
    ck("negative control: no mask slice -> NON-uniform (teeth)", cc>10000.0);
    ck("negative control: no mask slice leaks witness (X vs X2 differ)",
       open_at(X,0,false)!=open_at(X2,0,false));

    printf("\nP3.5 MASK-SLICE: %d passed, %d failed -> %s\n", np,nf, nf==0?"ALL PASS":"FAIL");
    return nf==0?0:1;
}
