// P3.5 privacy check (red-team HIGH-1): does opening an AUGMENTED poly [real|random]
// hide the WHOLE opening transcript (sumcheck messages, query codeword values, final
// folded value) -- not just the eval value?
//   ./p3_opening_zk_test
//
// Mechanism: a codeword position / sumcheck message / fold is a linear function
// L_real(real) + L_rand(rand) of the committed coeffs. With the random slice uniform,
// L_rand(rand) is uniform, so the revealed value = (anything) + uniform = uniform over
// GF(p), INDEPENDENT of the real witness (shift-invariance of the uniform dist).  We
// test uniformity + witness-independence at FIXED challenges, with a NEGATIVE CONTROL
// (no random slice -> revealed values become witness-dependent / non-uniform).
#include <cstdio>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"   // build_eq, fold, rs_encode
using std::vector;
static uint64_t St=9; static gl_t rng(){ St=St*6364136223846793005ULL+1; uint64_t z=St; z^=z>>31; return z%GL_P; }
static int np=0,nf=0; static void ck(const char* n,bool c){ printf("  [%s] %s\n",c?"PASS":"FAIL",n); if(c)np++; else nf++; }
static const int B=257, M=20000; static const double UNIF_HI=370.0;
static double chisq(const vector<gl_t>& v){ vector<long> h(B,0); for(auto x:v) h[(int)(x%B)]++;
    double e=(double)v.size()/B,c=0; for(int i=0;i<B;i++){ double d=h[i]-e; c+=d*d/e;} return c; }

// build augmented coeff vector [real | random]; vr = log2(len(real)); returns length 2*Nreal
static vector<gl_t> augment(const vector<gl_t>& real, uint64_t rseed, bool mask){
    uint32_t N=real.size(); vector<gl_t> f(2*N);
    for(uint32_t i=0;i<N;i++) f[i]=real[i];
    uint64_t s=rseed; for(uint32_t i=0;i<N;i++) f[N+i]= mask? (s=s*6364136223846793005ULL+1,(s^(s>>31))%GL_P) : 0ULL;
    return f;
}

int main(){
    printf("=== P3.5 opening-transcript privacy (augmentation masks transcript?) ===\n");
    uint32_t vr=8, Nreal=1u<<vr, R=2;            // real poly 2^8; augmented 2^9; codeword 2^11
    vector<gl_t> real(Nreal); for(auto&x:real) x=rng()%257;   // small-domain "weights"
    uint32_t v=vr+1, N=1u<<v;                      // augmented length
    // fixed opening point z (length v) incl. random ex coordinate; fixed pos
    vector<gl_t> z(v); for(auto&x:z) x=rng();
    uint32_t pos=37; gl_t xpos=gl_pow(gl_root_of_unity(v+R), pos);

    auto eq=p3bf::build_eq(z);
    // revealed quantities as functions of the augmented poly, at FIXED z / pos / challenges:
    //  (a) round-0 sumcheck message s0 = sum_b f[2b]*eq[2b]
    //  (b) codeword query value  cw[pos] = poly_f(xpos)
    //  (c) final fold value f~(z) (eval)  -- already known masked, included for completeness
    auto reveal=[&](const vector<gl_t>& real_,uint64_t rs,bool mask,gl_t&s0,gl_t&cwpos,gl_t&ev){
        auto f=augment(real_,rs,mask);
        s0=0; for(uint32_t b=0;b<N/2;b++) s0=gl_add(s0,gl_mul(f[2*b],eq[2*b]));
        cwpos=0; for(int i=(int)f.size()-1;i>=0;i--) cwpos=gl_add(gl_mul(cwpos,xpos),f[i]);  // poly_f(xpos)
        ev=0; for(uint32_t b=0;b<N;b++) ev=gl_add(ev,gl_mul(f[b],eq[b]));
    };

    for(int part=0;part<3;part++){
        const char* nm = part==0?"sumcheck msg s0" : part==1?"codeword query value" : "final eval value";
        vector<gl_t> mk(M), mk2(M), ctl(M); gl_t a,b,c;
        vector<gl_t> real2(Nreal); for(auto&x:real2) x=rng()%257;
        for(int i=0;i<M;i++){ reveal(real,3000+i,true,a,b,c);  mk[i]= part==0?a:part==1?b:c; }
        for(int i=0;i<M;i++){ reveal(real2,3000+i,true,a,b,c); mk2[i]=part==0?a:part==1?b:c; }
        for(int i=0;i<M;i++){ reveal(real,3000+i,false,a,b,c); ctl[i]=part==0?a:part==1?b:c; }
        double cm=chisq(mk), cc=chisq(ctl);
        printf("  [%s] masked chi-sq=%.0f  control(no-rand)=%.0f\n", nm, cm, cc);
        char n1[80]; snprintf(n1,sizeof n1,"%s: masked uniform",nm); ck(n1, cm<UNIF_HI);
        snprintf(n1,sizeof n1,"%s: witness-independent",nm); ck(n1, chisq(mk2)<UNIF_HI);
        snprintf(n1,sizeof n1,"%s: negative control leaks (teeth)",nm); ck(n1, cc>10000.0);
    }
    printf("\nP3.5 OPENING-ZK: %d passed, %d failed -> %s\n", np,nf, nf==0?"ALL PASS":"FAIL");
    return nf==0?0:1;
}
