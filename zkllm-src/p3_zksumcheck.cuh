// P3.5 ZK-sumcheck (Libra-style masking) for the eq-weighted claim  f~(z) = y.
//
// Hides the witness multilinear f.  Prover samples a random mask multilinear g,
// reveals only y_g = g~(z), gets challenge rho, and runs the degree-2 sumcheck on
// h = f + rho*g (claim y + rho*y_g).  Every round message and the final value are
// blinded by rho*(g-part), which is uniform -> the transcript leaks nothing about f.
//
// This is honest-verifier zero-knowledge: simulate() below produces an accepting
// transcript from (z, y, challenges) ALONE -- no witness -- identically distributed
// to a real one.  The test harness checks real==simulated in distribution AND that
// the property FAILS when masking is disabled (negative control).
//
// The final value h(r) is checked here against a supplied oracle value; in the full
// system that oracle is a (ZK) PCS opening of f+rho*g at r.  Soundness of the
// reduction (message chain) is fully checked.  Challenges shown in base field for
// clarity; production draws them from GL2 (p3_gl2.cuh) for ~2^-116.
#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include "p3_goldilocks.cuh"
#include "fs_transcript.hpp"

namespace p3zksc {

struct SumMsg { gl_t s0, s1, s2; };
struct Proof  { gl_t yg; std::vector<SumMsg> msgs; };

static inline std::vector<gl_t> build_eq(const std::vector<gl_t>& z) {
    uint32_t v=(uint32_t)z.size(), N=1u<<v; std::vector<gl_t> w(N,1ULL);
    for (uint32_t b=0;b<N;b++){ gl_t p=1ULL;
        for (uint32_t i=0;i<v;i++) p=gl_mul(p,(b&(1u<<i))?z[i]:gl_sub(1ULL,z[i])); w[b]=p; }
    return w;
}
static inline gl_t eval_h(const std::vector<gl_t>& f, const std::vector<gl_t>& eq){
    gl_t a=0; for (size_t b=0;b<f.size();b++) a=gl_add(a,gl_mul(f[b],eq[b])); return a;
}
static inline gl_t eq_point(const std::vector<gl_t>& r, const std::vector<gl_t>& z){
    gl_t p=1ULL; for (size_t i=0;i<r.size();i++)
        p=gl_mul(p, gl_add(gl_mul(r[i],z[i]), gl_mul(gl_sub(1ULL,r[i]),gl_sub(1ULL,z[i])))); return p;
}
static inline gl_t chal(fs::Transcript& tr){ uint8_t c[32]; tr.challenge_bytes(c);
    uint64_t v=0; for(int i=0;i<8;i++) v|=(uint64_t)c[i]<<(8*i); return v%GL_P; }
static inline gl_t quad_eval(gl_t s0,gl_t s1,gl_t s2,gl_t t){
    gl_t inv2=gl_inv(2ULL), t1=gl_sub(t,1ULL), t2=gl_sub(t,2ULL);
    gl_t L0=gl_mul(gl_mul(t1,t2),inv2), L1=gl_sub(0ULL,gl_mul(t,t2)), L2=gl_mul(gl_mul(t,t1),inv2);
    return gl_add(gl_add(gl_mul(s0,L0),gl_mul(s1,L1)),gl_mul(s2,L2));
}
// small deterministic PRNG so tests can drive the mask randomness explicitly
static inline gl_t prng(uint64_t& s){ s=s*6364136223846793005ULL+1442695040888963407ULL;
    uint64_t z=s; z^=z>>31; return z%GL_P; }

// ---- prover.  mask=false zeros g (NEGATIVE CONTROL: makes the transcript non-ZK) ----
static inline Proof prove(const std::vector<gl_t>& f, const std::vector<gl_t>& z, gl_t y,
                          const std::string& seed, uint64_t gseed,
                          std::vector<gl_t>& r_out, gl_t& h_at_r_out, bool mask=true) {
    uint32_t v=(uint32_t)z.size(), N=1u<<v;
    std::vector<gl_t> g(N,0ULL);
    if (mask){ uint64_t s=gseed; for (auto& x:g) x=prng(s); }
    auto eq = build_eq(z);
    Proof pf; pf.yg = eval_h(g, eq);
    fs::Transcript tr(seed); tr.absorb("z",z.data(),z.size()*sizeof(gl_t)); tr.absorb("y",&y,sizeof y);
    tr.absorb("yg",&pf.yg,sizeof pf.yg);
    gl_t rho = chal(tr);
    std::vector<gl_t> h(N); for (uint32_t i=0;i<N;i++) h[i]=gl_add(f[i], gl_mul(rho,g[i]));
    std::vector<gl_t> ch=h, ce=eq, r;
    gl_t claim = gl_add(y, gl_mul(rho, pf.yg));
    for (uint32_t rd=0; rd<v; rd++){
        uint32_t half=(uint32_t)ch.size()/2; gl_t s0=0,s1=0,s2=0;
        for (uint32_t b=0;b<half;b++){
            gl_t h0=ch[2*b],h1=ch[2*b+1],e0=ce[2*b],e1=ce[2*b+1];
            s0=gl_add(s0,gl_mul(h0,e0)); s1=gl_add(s1,gl_mul(h1,e1));
            gl_t h2=gl_sub(gl_add(h1,h1),h0), e2=gl_sub(gl_add(e1,e1),e0); s2=gl_add(s2,gl_mul(h2,e2));
        }
        SumMsg m{s0,s1,s2}; pf.msgs.push_back(m); tr.absorb("sc",&m,sizeof m);
        gl_t a=chal(tr); r.push_back(a);
        std::vector<gl_t> nh(half),ne(half);
        for (uint32_t b=0;b<half;b++){ nh[b]=gl_add(ch[2*b],gl_mul(a,gl_sub(ch[2*b+1],ch[2*b])));
            ne[b]=gl_add(ce[2*b],gl_mul(a,gl_sub(ce[2*b+1],ce[2*b]))); }
        ch=nh; ce=ne; claim=quad_eval(s0,s1,s2,a);
    }
    r_out=r; h_at_r_out=ch[0];     // h(r) = f~(r)+rho*g~(r); in the full system, a ZK PCS opening
    return pf;
}

// ---- verifier.  h_at_r is the oracle/PCS-opened value of f+rho*g at r ----
static inline bool verify(const Proof& pf, const std::vector<gl_t>& z, gl_t y,
                          const std::string& seed, gl_t h_at_r, const char** why=nullptr){
    auto fail=[&](const char* m){ if(why)*why=m; return false; };
    uint32_t v=(uint32_t)z.size(); if (pf.msgs.size()!=v) return fail("msg count");
    fs::Transcript tr(seed); tr.absorb("z",z.data(),z.size()*sizeof(gl_t)); tr.absorb("y",&y,sizeof y);
    tr.absorb("yg",&pf.yg,sizeof pf.yg);
    gl_t rho=chal(tr), claim=gl_add(y,gl_mul(rho,pf.yg));
    std::vector<gl_t> r;
    for (uint32_t rd=0; rd<v; rd++){ const SumMsg& m=pf.msgs[rd];
        if (gl_add(m.s0,m.s1)!=claim) return fail("sumcheck claim");
        tr.absorb("sc",&m,sizeof m); gl_t a=chal(tr); r.push_back(a); claim=quad_eval(m.s0,m.s1,m.s2,a);
    }
    if (claim != gl_mul(h_at_r, eq_point(r,z))) return fail("final oracle check");
    if (why)*why="ok"; return true;
}

// ---- HVZK simulator: builds an ACCEPTING transcript from (z,y,seed) with NO witness ----
static inline Proof simulate(const std::vector<gl_t>& z, gl_t y, const std::string& seed,
                             uint64_t simseed, std::vector<gl_t>& r_out, gl_t& h_at_r_out){
    uint32_t v=(uint32_t)z.size(); uint64_t s=simseed;
    Proof pf; pf.yg = prng(s);                       // y_g uniform, as in the real proof
    fs::Transcript tr(seed); tr.absorb("z",z.data(),z.size()*sizeof(gl_t)); tr.absorb("y",&y,sizeof y);
    tr.absorb("yg",&pf.yg,sizeof pf.yg);
    gl_t rho=chal(tr), claim=gl_add(y,gl_mul(rho,pf.yg));
    std::vector<gl_t> r;
    for (uint32_t rd=0; rd<v; rd++){
        gl_t s0=prng(s), s2=prng(s), s1=gl_sub(claim,s0);   // free coords uniform; s1 fixed by claim
        SumMsg m{s0,s1,s2}; pf.msgs.push_back(m); tr.absorb("sc",&m,sizeof m);
        gl_t a=chal(tr); r.push_back(a); claim=quad_eval(s0,s1,s2,a);
    }
    r_out=r; h_at_r_out=gl_mul(claim, gl_inv(eq_point(r,z)));  // pick h(r) so verify accepts
    return pf;
}

} // namespace p3zksc
