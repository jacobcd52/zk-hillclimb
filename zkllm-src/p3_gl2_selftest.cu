// P3 GL2 extension field selftest.  ./p3_gl2_selftest
#include <cstdio>
#include "p3_goldilocks.cuh"
#include "p3_gl2.cuh"
static uint64_t s=7; static uint64_t rng(){ s=s*6364136223846793005ULL+1; uint64_t z=s; z^=z>>31; return z; }
static gl2_t rnd(){ return gl2_t{ rng()%GL_P, rng()%GL_P }; }
static int np=0,nf=0; static void ck(const char* n,bool c){ printf("  [%s] %s\n",c?"PASS":"FAIL",n); if(c)np++; else nf++; }
int main(){
    printf("=== P3 GL2 extension field selftest ===\n");
    // 7 must be a quadratic non-residue: 7^((p-1)/2) == p-1
    ck("7 is a non-residue (x^2-7 irreducible)", gl_pow(7ULL,(GL_P-1)/2)==GL_P-1);
    int fails=0;
    for(int i=0;i<500000;i++){
        gl2_t x=rnd(), y=rnd(), z=rnd();
        // commutativity & distributivity
        if(!gl2_eq(gl2_mul(x,y),gl2_mul(y,x))) fails++;
        if(!gl2_eq(gl2_mul(x,gl2_add(y,z)), gl2_add(gl2_mul(x,y),gl2_mul(x,z)))) fails++;
        // (x-y)+y == x
        if(!gl2_eq(gl2_add(gl2_sub(x,y),y),x)) fails++;
        // base-field consistency: embed(a)*embed(b) == embed(a*b)
        gl_t a=x.a,b=y.a;
        if(!gl2_eq(gl2_mul(gl2_from(a),gl2_from(b)), gl2_from(gl_mul(a,b)))) fails++;
    }
    ck("field axioms + base embedding (500k)", fails==0);
    int invf=0;
    for(int i=0;i<200000;i++){ gl2_t x=rnd(); if(x.a==0&&x.b==0) continue;
        if(!gl2_eq(gl2_mul(x,gl2_inv(x)), gl2_one())) invf++; }
    ck("inverse: x * x^-1 == 1 (200k)", invf==0);
    // scale consistency: scale(x,s) == mul(x, embed(s))
    int sf=0; for(int i=0;i<100000;i++){ gl2_t x=rnd(); gl_t sc=rng()%GL_P;
        if(!gl2_eq(gl2_scale(x,sc), gl2_mul(x,gl2_from(sc)))) sf++; }
    ck("scale == mul by embedded base (100k)", sf==0);
    printf("\nP3 GL2: %d passed, %d failed -> %s\n", np,nf, nf==0?"ALL PASS":"FAIL");
    return nf==0?0:1;
}
