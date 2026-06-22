// P3.5 CAPSTONE: zero-knowledge private FC layer  Y = X . W  (hides X, W, Y).
//
// Composes the validated mechanisms:
//  - augment X,W,Y with an extra "ex" slice (real | random)            [p3_maskslice]
//  - one ZK-masked sumcheck over (j, ex_X, ex_W, ey), PUBLIC claim 0   [p3_zkmatmul/zksumcheck]
//    summand = eq(ey,0)*Yhat(ey) - 2*IN*eq(ex_X,0)eq(ex_W,0)*Xhat(ex_X,j)*What(ex_W,j)
//    sum over the cube = 4*IN*(Y~(ri,rk) - sum_j X~(ri,j)W~(j,rk)) = 0  iff  Y=X.W
//  - open Xhat,What,Yhat at random ex -> uniform values mX,mW,mY (hide the boundary evals)
//    final check: claim == (1-rey)mY - 2IN(1-rexX)(1-rexW)mX mW + rho*qr
//
// Challenges base-field here (GL2 swap = production soundness).  Openings via p3bf
// (eval value mX is uniform by the mask slice; query-masking/salted leaves are the
// remaining hardening, see P3_PRIVACY_DESIGN.md).  Built to be RED-TEAMED.
#pragma once
#include <cstdint>
#include <vector>
#include <chrono>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "fs_transcript.hpp"

namespace p3pfc {

struct Timing { double commit_ms=0, prep_ms=0, sumcheck_ms=0, open_ms=0; };
static inline double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }

struct SumMsg { gl_t s0,s1,s2; };
struct Proof {
    uint32_t bb,ii,oo,R,Q;
    p3fri::Hash rootX,rootW,rootY;
    gl_t Sq, qr;
    std::vector<SumMsg> msgs;            // V = ii+3 rounds
    p3bf::EvalProof openX,openW,openY;
};

static inline gl_t chal(fs::Transcript& tr){ uint8_t c[32]; tr.challenge_bytes(c);
    uint64_t v=0; for(int i=0;i<8;i++) v|=(uint64_t)c[i]<<(8*i); return v%GL_P; }
static inline std::vector<gl_t> chal_vec(fs::Transcript& tr,uint32_t n){ std::vector<gl_t> v(n); for(auto&x:v)x=chal(tr); return v; }
static inline gl_t quad_eval(gl_t s0,gl_t s1,gl_t s2,gl_t t){
    gl_t inv2=gl_inv(2ULL),t1=gl_sub(t,1ULL),t2=gl_sub(t,2ULL);
    gl_t L0=gl_mul(gl_mul(t1,t2),inv2),L1=gl_sub(0ULL,gl_mul(t,t2)),L2=gl_mul(gl_mul(t,t1),inv2);
    return gl_add(gl_add(gl_mul(s0,L0),gl_mul(s1,L1)),gl_mul(s2,L2)); }
static inline gl_t prng(uint64_t&s){ s=s*6364136223846793005ULL+1442695040888963407ULL; uint64_t z=s; z^=z>>31; return z%GL_P; }
static inline std::vector<gl_t> cat(std::vector<gl_t> a,const std::vector<gl_t>& b){ a.insert(a.end(),b.begin(),b.end()); return a; }

// bind one variable (LSB) of a length-2m array with challenge a
static inline void bind(std::vector<gl_t>& f, gl_t a){ uint32_t h=f.size()/2; std::vector<gl_t> n(h);
    for(uint32_t i=0;i<h;i++) n[i]=gl_add(f[2*i],gl_mul(a,gl_sub(f[2*i+1],f[2*i]))); f=n; }

// ---------------- prover ----------------
static inline Proof prove(const std::vector<gl_t>& X,const std::vector<gl_t>& W,const std::vector<gl_t>& Y,
                          const std::vector<gl_t>& RX,const std::vector<gl_t>& RW,const std::vector<gl_t>& RY,
                          uint32_t bb,uint32_t ii,uint32_t oo,uint32_t R,uint32_t Q,uint64_t seed_q,bool gpu=false,
                          Timing* tm=nullptr){
    uint32_t B=1u<<bb, IN=1u<<ii, OUT=1u<<oo;
    Proof pf; pf.bb=bb;pf.ii=ii;pf.oo=oo;pf.R=R;pf.Q=Q;
    double t0=now_ms();
    // augment: index ex*base + real_index
    std::vector<gl_t> Xh(2u*B*IN), Wh(2u*IN*OUT), Yh(2u*B*OUT);
    for(uint32_t i=0;i<B*IN;i++){ Xh[i]=X[i]; Xh[B*IN+i]=RX[i]; }
    for(uint32_t i=0;i<IN*OUT;i++){ Wh[i]=W[i]; Wh[IN*OUT+i]=RW[i]; }
    for(uint32_t i=0;i<B*OUT;i++){ Yh[i]=Y[i]; Yh[B*OUT+i]=RY[i]; }
    std::vector<gl_t> cwX,cwW,cwY;
    pf.rootX = gpu?p3bf::commit_gpu(Xh,R,cwX):p3bf::commit(Xh,R,cwX);
    pf.rootW = gpu?p3bf::commit_gpu(Wh,R,cwW):p3bf::commit(Wh,R,cwW);
    pf.rootY = gpu?p3bf::commit_gpu(Yh,R,cwY):p3bf::commit(Yh,R,cwY);

    cudaDeviceSynchronize(); double t1=now_ms(); if(tm) tm->commit_ms=t1-t0;
    fs::Transcript tr("p3-pfc");
    tr.absorb("rX",pf.rootX.data(),32); tr.absorb("rW",pf.rootW.data(),32); tr.absorb("rY",pf.rootY.data(),32);
    std::vector<gl_t> r_i=chal_vec(tr,bb), r_k=chal_vec(tr,oo);

    auto eqi=p3bf::build_eq(r_i), eqk=p3bf::build_eq(r_k);
    std::vector<gl_t> AX0(IN,0),AX1(IN,0),BW0(IN,0),BW1(IN,0);
    for(uint32_t j=0;j<IN;j++){ gl_t a0=0,a1=0;
        for(uint32_t i=0;i<B;i++){ a0=gl_add(a0,gl_mul(X[i*IN+j],eqi[i])); a1=gl_add(a1,gl_mul(RX[i*IN+j],eqi[i])); }
        gl_t b0=0,b1=0; for(uint32_t k=0;k<OUT;k++){ b0=gl_add(b0,gl_mul(W[j*OUT+k],eqk[k])); b1=gl_add(b1,gl_mul(RW[j*OUT+k],eqk[k])); }
        AX0[j]=a0;AX1[j]=a1;BW0[j]=b0;BW1[j]=b1; }
    gl_t Y0=0,Y1=0; for(uint32_t i=0;i<B;i++)for(uint32_t k=0;k<OUT;k++){ gl_t w=gl_mul(eqi[i],eqk[k]);
        Y0=gl_add(Y0,gl_mul(Y[i*OUT+k],w)); Y1=gl_add(Y1,gl_mul(RY[i*OUT+k],w)); }

    // factor arrays over V=ii+3 cube (order: j[ii], ex_X, ex_W, ey)
    uint32_t V=ii+3, SZ=1u<<V;
    std::vector<gl_t> f1a(SZ),f1b(SZ),f2a(SZ),f2b(SZ),f2c(SZ),f2d(SZ),q(SZ);
    uint64_t sq=seed_q;
    for(uint32_t b=0;b<SZ;b++){ uint32_t j=b&(IN-1),exX=(b>>ii)&1,exW=(b>>(ii+1))&1,ey=(b>>(ii+2))&1;
        f1a[b]=gl_sub(1ULL,(gl_t)ey); f1b[b]= ey?Y1:Y0;
        f2a[b]=gl_sub(1ULL,(gl_t)exX); f2b[b]=gl_sub(1ULL,(gl_t)exW);
        f2c[b]= exX?AX1[j]:AX0[j]; f2d[b]= exW?BW1[j]:BW0[j];
        q[b]=prng(sq); }
    pf.Sq=0; for(auto x:q) pf.Sq=gl_add(pf.Sq,x);
    tr.absorb("Sq",&pf.Sq,sizeof pf.Sq); gl_t rho=chal(tr);
    gl_t c2=(gl_t)((2ull*IN)%GL_P);
    double t2=now_ms(); if(tm) tm->prep_ms=t2-t1;

    std::vector<gl_t> r;
    for(uint32_t rd=0;rd<V;rd++){ uint32_t h=f1a.size()/2; gl_t s0=0,s1=0,s2=0;
        for(uint32_t i=0;i<h;i++){
            // evaluate masked summand at t=0,1,2 for this pair
            for(int t=0;t<3;t++){
                auto I=[&](const std::vector<gl_t>&f)->gl_t{ gl_t lo=f[2*i],hi=f[2*i+1];
                    if(t==0) return lo; if(t==1) return hi; return gl_sub(gl_add(hi,hi),lo); };
                gl_t term1=gl_mul(I(f1a),I(f1b));
                gl_t term2=gl_mul(gl_mul(I(f2a),I(f2b)),gl_mul(I(f2c),I(f2d)));
                gl_t val=gl_add(gl_sub(term1,gl_mul(c2,term2)), gl_mul(rho,I(q)));
                if(t==0)s0=gl_add(s0,val); else if(t==1)s1=gl_add(s1,val); else s2=gl_add(s2,val);
            }
        }
        SumMsg m{s0,s1,s2}; pf.msgs.push_back(m); tr.absorb("sc",&m,sizeof m);
        gl_t al=chal(tr); r.push_back(al);
        bind(f1a,al);bind(f1b,al);bind(f2a,al);bind(f2b,al);bind(f2c,al);bind(f2d,al);bind(q,al);
    }
    pf.qr=q[0];
    double t3=now_ms(); if(tm) tm->sumcheck_ms=t3-t2;
    // openings at the random ex points
    std::vector<gl_t> r_j(r.begin(),r.begin()+ii);
    gl_t rexX=r[ii],rexW=r[ii+1],rey=r[ii+2];
    auto zX=cat(cat(r_j,r_i),std::vector<gl_t>{rexX});      // (j, i, exX)
    auto zW=cat(cat(r_k,r_j),std::vector<gl_t>{rexW});      // (k, j, exW)
    auto zY=cat(cat(r_k,r_i),std::vector<gl_t>{rey});       // (k, i, ey)
    gl_t yX = gpu ? p3bf::eval_h_gpu(Xh,zX) : p3bf::eval_h(Xh,p3bf::build_eq(zX));
    gl_t yW = gpu ? p3bf::eval_h_gpu(Wh,zW) : p3bf::eval_h(Wh,p3bf::build_eq(zW));
    gl_t yY = gpu ? p3bf::eval_h_gpu(Yh,zY) : p3bf::eval_h(Yh,p3bf::build_eq(zY));
    pf.openX=p3bf::prove_eval(Xh,zX,yX,R,Q,cwX,"pfc-X");
    pf.openW=p3bf::prove_eval(Wh,zW,yW,R,Q,cwW,"pfc-W");
    pf.openY=p3bf::prove_eval(Yh,zY,yY,R,Q,cwY,"pfc-Y");
    cudaDeviceSynchronize(); if(tm) tm->open_ms=now_ms()-t3;
    return pf;
}

// ---------------- verifier ----------------
// Q_pub, R_pub are PUBLIC parameters fixed by the verifier (NOT read from the proof) --
// the FRI query count and rate are the basis of soundness; trusting the proof's copy
// lets a malicious prover set Q=0 for a vacuous accept (red-team CRITICAL-1).
static inline bool verify(const Proof& pf, uint32_t Q_pub, uint32_t R_pub, const char** why=nullptr){
    auto fail=[&](const char* m){ if(why)*why=m; return false; };
    uint32_t bb=pf.bb,ii=pf.ii,oo=pf.oo,V=ii+3, IN=1u<<ii;
    if(pf.msgs.size()!=V) return fail("msg count");
    // pin FRI params to public values on every opening (defeats Q=0 / R=0 forgery)
    const uint32_t Q_MIN=20;
    if(Q_pub<Q_MIN || R_pub<1) return fail("insecure params");
    auto chkpar=[&](const p3bf::EvalProof& e, uint32_t logN_exp)->bool{
        return e.Q==Q_pub && e.R==R_pub && e.logN==logN_exp; };
    if(!chkpar(pf.openX, bb+ii+1) || !chkpar(pf.openW, ii+oo+1) || !chkpar(pf.openY, bb+oo+1))
        return fail("opening params != public (Q/R/logN)");
    fs::Transcript tr("p3-pfc");
    tr.absorb("rX",pf.rootX.data(),32); tr.absorb("rW",pf.rootW.data(),32); tr.absorb("rY",pf.rootY.data(),32);
    std::vector<gl_t> r_i=chal_vec(tr,bb), r_k=chal_vec(tr,oo);
    tr.absorb("Sq",&pf.Sq,sizeof pf.Sq); gl_t rho=chal(tr);
    gl_t claim=gl_mul(rho,pf.Sq); std::vector<gl_t> r;
    for(uint32_t rd=0;rd<V;rd++){ const SumMsg& m=pf.msgs[rd];
        if(gl_add(m.s0,m.s1)!=claim) return fail("sumcheck claim");
        tr.absorb("sc",&m,sizeof m); gl_t al=chal(tr); r.push_back(al); claim=quad_eval(m.s0,m.s1,m.s2,al); }
    std::vector<gl_t> r_j(r.begin(),r.begin()+ii);
    gl_t rexX=r[ii],rexW=r[ii+1],rey=r[ii+2];
    auto zX=cat(cat(r_j,r_i),std::vector<gl_t>{rexX});
    auto zW=cat(cat(r_k,r_j),std::vector<gl_t>{rexW});
    auto zY=cat(cat(r_k,r_i),std::vector<gl_t>{rey});
    if(pf.openX.roots.empty()||!(pf.openX.roots[0]==pf.rootX)||pf.openX.z!=zX) return fail("X opening bind");
    if(pf.openW.roots.empty()||!(pf.openW.roots[0]==pf.rootW)||pf.openW.z!=zW) return fail("W opening bind");
    if(pf.openY.roots.empty()||!(pf.openY.roots[0]==pf.rootY)||pf.openY.z!=zY) return fail("Y opening bind");
    if(!p3bf::verify_eval(pf.openX,"pfc-X",why)) return false;
    if(!p3bf::verify_eval(pf.openW,"pfc-W",why)) return false;
    if(!p3bf::verify_eval(pf.openY,"pfc-Y",why)) return false;
    gl_t mX=pf.openX.y,mW=pf.openW.y,mY=pf.openY.y;
    gl_t c2=(gl_t)((2ull*IN)%GL_P);
    gl_t term1=gl_mul(gl_sub(1ULL,rey),mY);
    gl_t term2=gl_mul(gl_mul(gl_sub(1ULL,rexX),gl_sub(1ULL,rexW)),gl_mul(mX,mW));
    gl_t summand_final=gl_sub(term1,gl_mul(c2,term2));
    if(claim != gl_add(summand_final, gl_mul(rho,pf.qr))) return fail("final tie");
    if(why)*why="ok"; return true;
}

} // namespace p3pfc
