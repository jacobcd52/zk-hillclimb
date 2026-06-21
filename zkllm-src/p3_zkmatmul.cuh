// P3.5 ZK matmul-sumcheck: prove  sum_j A[j]*B[j] = c  in zero-knowledge.
//
// This is the reduction at the heart of the FC matmul (A[j]=X~(r_i,j),
// B[j]=W~(j,r_k), c=Y~(r_i,r_k)).  Masked with a random multilinear q so every round
// message and intermediate claim is uniform -> the verifier learns nothing about the
// partial sums of A*B (i.e. nothing about X/W beyond the public statement).  The two
// final evaluations A~(r), B~(r) are supplied by an oracle (ZK PCS opening in the full
// system).  HVZK is validated by a witnessless simulator; soundness by the message chain.
#pragma once
#include <cstdint>
#include <vector>
#include "p3_goldilocks.cuh"
#include "fs_transcript.hpp"

namespace p3zkmm {

struct SumMsg { gl_t s0, s1, s2; };
struct Proof  { gl_t Sq, qr; std::vector<SumMsg> msgs; };   // Sq=sum q, qr=q~(r) (both random->hide nothing)

static inline gl_t chal(fs::Transcript& tr){ uint8_t c[32]; tr.challenge_bytes(c);
    uint64_t v=0; for(int i=0;i<8;i++) v|=(uint64_t)c[i]<<(8*i); return v%GL_P; }
static inline gl_t quad_eval(gl_t s0,gl_t s1,gl_t s2,gl_t t){
    gl_t inv2=gl_inv(2ULL), t1=gl_sub(t,1ULL), t2=gl_sub(t,2ULL);
    gl_t L0=gl_mul(gl_mul(t1,t2),inv2), L1=gl_sub(0ULL,gl_mul(t,t2)), L2=gl_mul(gl_mul(t,t1),inv2);
    return gl_add(gl_add(gl_mul(s0,L0),gl_mul(s1,L1)),gl_mul(s2,L2));
}
static inline gl_t prng(uint64_t& s){ s=s*6364136223846793005ULL+1442695040888963407ULL; uint64_t z=s; z^=z>>31; return z%GL_P; }
static inline gl_t mle(const std::vector<gl_t>& a, const std::vector<gl_t>& r){    // a~(r) = sum_b a[b] eq(b,r)
    uint32_t v=(uint32_t)r.size(); gl_t acc=0;
    for (uint32_t b=0;b<a.size();b++){ gl_t e=1ULL; for(uint32_t i=0;i<v;i++) e=gl_mul(e,(b&(1u<<i))?r[i]:gl_sub(1ULL,r[i])); acc=gl_add(acc,gl_mul(a[b],e)); }
    return acc;
}

// prove sum_j A[j]B[j] = c.  mask=false -> q=0 (NEGATIVE CONTROL: non-ZK).
static inline Proof prove(const std::vector<gl_t>& A, const std::vector<gl_t>& B, gl_t c,
                          const std::string& seed, uint64_t qseed,
                          std::vector<gl_t>& r_out, bool mask=true){
    uint32_t N=(uint32_t)A.size(), v=0; while((1u<<v)<N) v++;
    std::vector<gl_t> q(N,0ULL); if(mask){ uint64_t s=qseed; for(auto&x:q) x=prng(s); }
    Proof pf; pf.Sq=0; for(auto x:q) pf.Sq=gl_add(pf.Sq,x);
    fs::Transcript tr(seed); tr.absorb("c",&c,sizeof c); tr.absorb("Sq",&pf.Sq,sizeof pf.Sq);
    gl_t rho=chal(tr);
    std::vector<gl_t> a=A,b=B,qq=q,r; gl_t claim=gl_add(c, gl_mul(rho,pf.Sq));
    for(uint32_t rd=0;rd<v;rd++){
        uint32_t half=(uint32_t)a.size()/2; gl_t s0=0,s1=0,s2=0;
        for(uint32_t i=0;i<half;i++){
            gl_t a0=a[2*i],a1=a[2*i+1],b0=b[2*i],b1=b[2*i+1],q0=qq[2*i],q1=qq[2*i+1];
            // summand(t) = a(t)*b(t) + rho*q(t)
            s0=gl_add(s0, gl_add(gl_mul(a0,b0), gl_mul(rho,q0)));
            s1=gl_add(s1, gl_add(gl_mul(a1,b1), gl_mul(rho,q1)));
            gl_t a2=gl_sub(gl_add(a1,a1),a0), b2=gl_sub(gl_add(b1,b1),b0), q2=gl_sub(gl_add(q1,q1),q0);
            s2=gl_add(s2, gl_add(gl_mul(a2,b2), gl_mul(rho,q2)));
        }
        SumMsg m{s0,s1,s2}; pf.msgs.push_back(m); tr.absorb("sc",&m,sizeof m);
        gl_t al=chal(tr); r.push_back(al);
        uint32_t h=half; std::vector<gl_t> na(h),nb(h),nq(h);
        for(uint32_t i=0;i<h;i++){ na[i]=gl_add(a[2*i],gl_mul(al,gl_sub(a[2*i+1],a[2*i])));
            nb[i]=gl_add(b[2*i],gl_mul(al,gl_sub(b[2*i+1],b[2*i])));
            nq[i]=gl_add(qq[2*i],gl_mul(al,gl_sub(qq[2*i+1],qq[2*i]))); }
        a=na; b=nb; qq=nq; claim=quad_eval(s0,s1,s2,al);
    }
    pf.qr=qq[0]; r_out=r;     // q~(r); a[0]=A~(r), b[0]=B~(r) are the oracle finals
    return pf;
}

// verify with oracle finals Ar=A~(r), Br=B~(r) (from ZK openings in the full system).
static inline bool verify(const Proof& pf, gl_t c, const std::string& seed, uint32_t v,
                          gl_t Ar, gl_t Br, const char** why=nullptr){
    auto fail=[&](const char* m){ if(why)*why=m; return false; };
    if(pf.msgs.size()!=v) return fail("msg count");
    fs::Transcript tr(seed); tr.absorb("c",&c,sizeof c); tr.absorb("Sq",&pf.Sq,sizeof pf.Sq);
    gl_t rho=chal(tr), claim=gl_add(c,gl_mul(rho,pf.Sq));
    for(uint32_t rd=0;rd<v;rd++){ const SumMsg& m=pf.msgs[rd];
        if(gl_add(m.s0,m.s1)!=claim) return fail("matmul sumcheck claim");
        tr.absorb("sc",&m,sizeof m); gl_t al=chal(tr); claim=quad_eval(m.s0,m.s1,m.s2,al);
    }
    // final: claim == A~(r)*B~(r) + rho*q~(r)
    if(claim != gl_add(gl_mul(Ar,Br), gl_mul(rho,pf.qr))) return fail("final tie");
    if(why)*why="ok"; return true;
}

// HVZK simulator: accepting transcript from (c,seed) with NO witness.
static inline Proof simulate(gl_t c, const std::string& seed, uint32_t v, uint64_t simseed,
                             std::vector<gl_t>& r_out, gl_t& Ar_out, gl_t& Br_out){
    uint64_t s=simseed; Proof pf; pf.Sq=prng(s); pf.qr=prng(s);
    fs::Transcript tr(seed); tr.absorb("c",&c,sizeof c); tr.absorb("Sq",&pf.Sq,sizeof pf.Sq);
    gl_t rho=chal(tr), claim=gl_add(c,gl_mul(rho,pf.Sq)); std::vector<gl_t> r;
    for(uint32_t rd=0;rd<v;rd++){ gl_t s0=prng(s), s2=prng(s), s1=gl_sub(claim,s0);
        SumMsg m{s0,s1,s2}; pf.msgs.push_back(m); tr.absorb("sc",&m,sizeof m);
        gl_t al=chal(tr); r.push_back(al); claim=quad_eval(s0,s1,s2,al); }
    // pick Ar, Br consistent with final tie: Ar*Br = claim - rho*qr ; set Br=1, Ar=that.
    gl_t prod=gl_sub(claim, gl_mul(rho,pf.qr)); Br_out=1ULL; Ar_out=prod; r_out=r;
    return pf;
}

} // namespace p3zkmm
