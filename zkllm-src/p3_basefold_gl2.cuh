// P3 soundness-upgraded Basefold eval opening: challenges + folded values in GL2
// (degree-2 extension), giving ~2^-116 soundness vs the base-field ~2^-58.
//
// Uniform-embedding approach: the base-field witness/codeword embeds as (x,0); the
// evaluation domain stays base field (embedded per use); challenges and all folded /
// bound values live in GL2.  Mirrors p3_basefold.cuh exactly, retyped gl_t -> gl2_t.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include "p3_goldilocks.cuh"
#include "p3_gl2.cuh"
#include "p3_basefold.cuh"   // reuse p3bf::rs_encode (base-field), ilog2
#include "fs_transcript.hpp"

namespace p3bf2 {
typedef std::array<uint8_t,32> Hash;

static inline Hash leaf_hash(gl2_t v) {
    uint8_t b[16];
    for (int k=0;k<8;k++){ b[k]=(uint8_t)(v.a>>(8*k)); b[8+k]=(uint8_t)(v.b>>(8*k)); }
    Hash h; fs::sha256(b,16,h.data()); return h;
}
static inline Hash node_hash(const Hash& l,const Hash& r){
    uint8_t b[64]; memcpy(b,l.data(),32); memcpy(b+32,r.data(),32); Hash h; fs::sha256(b,64,h.data()); return h;
}
struct Merkle {
    std::vector<std::vector<Hash>> levels;
    void build(const std::vector<gl2_t>& cw){
        std::vector<Hash> lv(cw.size()); for(size_t i=0;i<cw.size();i++) lv[i]=leaf_hash(cw[i]);
        levels.assign(1,lv);
        while(levels.back().size()>1){ const auto& c=levels.back(); std::vector<Hash> n(c.size()/2);
            for(size_t i=0;i<n.size();i++) n[i]=node_hash(c[2*i],c[2*i+1]); levels.push_back(n); }
    }
    Hash root() const { return levels.back()[0]; }
    std::vector<Hash> path(size_t i) const { std::vector<Hash> p;
        for(size_t d=0;d+1<levels.size();d++){ p.push_back(levels[d][i^1]); i>>=1; } return p; }
};
static inline bool verify_path(gl2_t v,size_t idx,const std::vector<Hash>& path,const Hash& root){
    Hash h=leaf_hash(v); for(auto& s:path){ h=(idx&1)?node_hash(s,h):node_hash(h,s); idx>>=1; } return h==root;
}

// MLE fold of a coset pair (a=f(x), b=f(-x)) at base-field x (embedded):
//  E=(a+b)/2, O=(a-b)/(2x), out=(1-beta)E + beta*O
static inline gl2_t fold_pair(gl2_t a, gl2_t b, gl_t invx, gl2_t beta){
    gl2_t inv2 = gl2_from(gl_inv(2ULL));
    gl2_t E = gl2_mul(gl2_add(a,b), inv2);
    gl2_t O = gl2_scale(gl2_mul(gl2_sub(a,b), inv2), invx); // (a-b)/(2x), x base-field
    return gl2_add(gl2_mul(gl2_sub(gl2_one(),beta),E), gl2_mul(beta,O));
}
static inline std::vector<gl2_t> fold(const std::vector<gl2_t>& f, gl_t wM, gl2_t beta){
    uint32_t M=f.size(),half=M/2; gl_t winv=gl_inv(wM), invx=1ULL; std::vector<gl2_t> out(half);
    for(uint32_t c=0;c<half;c++){ out[c]=fold_pair(f[c],f[c+half],invx,beta); invx=gl_mul(invx,winv); }
    return out;
}
struct RoundOpen { gl2_t a,b; std::vector<Hash> pa,pb; };
struct QueryProof { std::vector<RoundOpen> rounds; };
struct SumMsg { gl2_t s0,s1,s2; };
struct EvalProof {
    uint32_t logN,R,Q; std::vector<Hash> roots; std::vector<SumMsg> msgs;
    std::vector<gl2_t> final_word; std::vector<QueryProof> queries; std::vector<gl2_t> z; gl2_t y;
};

static inline std::vector<gl2_t> build_eq(const std::vector<gl2_t>& z){
    uint32_t v=z.size(),N=1u<<v; std::vector<gl2_t> w(N,gl2_one());
    for(uint32_t b=0;b<N;b++){ gl2_t p=gl2_one();
        for(uint32_t i=0;i<v;i++) p=gl2_mul(p,(b&(1u<<i))?z[i]:gl2_sub(gl2_one(),z[i])); w[b]=p; }
    return w;
}
static inline gl2_t eval_h(const std::vector<gl2_t>& c,const std::vector<gl2_t>& eq){
    gl2_t a=gl2_zero(); for(size_t b=0;b<c.size();b++) a=gl2_add(a,gl2_mul(c[b],eq[b])); return a;
}
static inline gl2_t chal(fs::Transcript& tr){ uint8_t c[32]; tr.challenge_bytes(c);
    uint64_t a=0,b=0; for(int i=0;i<8;i++){ a|=(uint64_t)c[i]<<(8*i); b|=(uint64_t)c[8+i]<<(8*i);} return gl2_t{a%GL_P,b%GL_P}; }
static inline uint64_t idx_from(fs::Transcript& tr,uint64_t mod){ uint8_t c[32]; tr.challenge_bytes(c);
    uint64_t v=0; for(int i=0;i<8;i++) v|=(uint64_t)c[i]<<(8*i); return v%mod; }
static inline gl2_t quad_eval(gl2_t s0,gl2_t s1,gl2_t s2,gl2_t t){
    gl2_t inv2=gl2_from(gl_inv(2ULL)), t1=gl2_sub(t,gl2_one()), t2=gl2_sub(t,gl2_from(2ULL));
    gl2_t L0=gl2_mul(gl2_mul(t1,t2),inv2);
    gl2_t L1=gl2_sub(gl2_zero(),gl2_mul(t,t2));
    gl2_t L2=gl2_mul(gl2_mul(t,t1),inv2);
    return gl2_add(gl2_add(gl2_mul(s0,L0),gl2_mul(s1,L1)),gl2_mul(s2,L2));
}
static inline gl2_t eq_point(const std::vector<gl2_t>& al,const std::vector<gl2_t>& z){
    gl2_t p=gl2_one(); for(size_t i=0;i<al.size();i++){
        gl2_t t=gl2_add(gl2_mul(al[i],z[i]),gl2_mul(gl2_sub(gl2_one(),al[i]),gl2_sub(gl2_one(),z[i]))); p=gl2_mul(p,t);} return p;
}

// commit a base-field coeff array, embed codeword into GL2.  Returns root, fills cw.
static inline Hash commit(const std::vector<gl_t>& c_base, uint32_t R, std::vector<gl2_t>& cw){
    auto cwb = p3bf::rs_encode(c_base, R);
    cw.resize(cwb.size()); for(size_t i=0;i<cwb.size();i++) cw[i]=gl2_from(cwb[i]);
    Merkle mk; mk.build(cw); return mk.root();
}

static inline EvalProof prove_eval(const std::vector<gl_t>& c_base, const std::vector<gl2_t>& z, gl2_t y,
                                   uint32_t R, uint32_t Q, const std::vector<gl2_t>& cw0, const std::string& seed="p3-bf2"){
    uint32_t v=z.size(), logM0=v+R, M0=1u<<logM0;
    EvalProof pf; pf.logN=v; pf.R=R; pf.Q=Q; pf.z=z; pf.y=y;
    fs::Transcript tr(seed); tr.absorb("z",z.data(),z.size()*sizeof(gl2_t)); tr.absorb("y",&y,sizeof(gl2_t));
    std::vector<std::vector<gl2_t>> words; words.push_back(cw0);
    std::vector<Merkle> trees;
    std::vector<gl2_t> cur_c(c_base.size()); for(size_t i=0;i<c_base.size();i++) cur_c[i]=gl2_from(c_base[i]);
    std::vector<gl2_t> cur_w=build_eq(z), al;
    for(uint32_t r=0;r<v;r++){
        Merkle mk; mk.build(words[r]); trees.push_back(mk); pf.roots.push_back(mk.root());
        tr.absorb("root",mk.root().data(),32);
        uint32_t n=cur_c.size(),half=n/2; gl2_t s0=gl2_zero(),s1=gl2_zero(),s2=gl2_zero();
        for(uint32_t b=0;b<half;b++){ gl2_t cl=cur_c[2*b],ch=cur_c[2*b+1],wl=cur_w[2*b],wh=cur_w[2*b+1];
            s0=gl2_add(s0,gl2_mul(cl,wl)); s1=gl2_add(s1,gl2_mul(ch,wh));
            gl2_t c2=gl2_sub(gl2_add(ch,ch),cl), w2=gl2_sub(gl2_add(wh,wh),wl); s2=gl2_add(s2,gl2_mul(c2,w2)); }
        SumMsg m{s0,s1,s2}; pf.msgs.push_back(m); tr.absorb("sc",&m,sizeof(m));
        gl2_t a=chal(tr); al.push_back(a);
        std::vector<gl2_t> nc(half),nw(half);
        for(uint32_t b=0;b<half;b++){ nc[b]=gl2_add(cur_c[2*b],gl2_mul(a,gl2_sub(cur_c[2*b+1],cur_c[2*b])));
            nw[b]=gl2_add(cur_w[2*b],gl2_mul(a,gl2_sub(cur_w[2*b+1],cur_w[2*b]))); }
        cur_c=nc; cur_w=nw;
        words.push_back(fold(words[r], gl_root_of_unity(logM0-r), a));
    }
    pf.final_word=words[v]; tr.absorb("final",pf.final_word.data(),pf.final_word.size()*sizeof(gl2_t));
    for(uint32_t q=0;q<Q;q++){ uint32_t c0=(uint32_t)idx_from(tr,M0/2); QueryProof qp; uint32_t p=c0;
        for(uint32_t r=0;r<v;r++){ uint32_t Mr=M0>>r,h=Mr/2,cc=p%h; RoundOpen ro;
            ro.a=words[r][cc]; ro.b=words[r][cc+h]; ro.pa=trees[r].path(cc); ro.pb=trees[r].path(cc+h);
            qp.rounds.push_back(ro); p=cc; }
        pf.queries.push_back(qp); }
    return pf;
}

static inline bool verify_eval(const EvalProof& pf, const std::string& seed="p3-bf2", const char** why=nullptr){
    auto fail=[&](const char* m){ if(why)*why=m; return false; };
    uint32_t v=pf.logN,R=pf.R,logM0=v+R,M0=1u<<logM0;
    if(pf.roots.size()!=v||pf.msgs.size()!=v) return fail("size");
    if(pf.final_word.size()!=(1u<<R)) return fail("final size");
    if(pf.queries.size()!=pf.Q) return fail("query count");
    fs::Transcript tr(seed); tr.absorb("z",pf.z.data(),pf.z.size()*sizeof(gl2_t)); tr.absorb("y",&pf.y,sizeof(gl2_t));
    std::vector<gl2_t> al(v); gl2_t claim=pf.y;
    for(uint32_t r=0;r<v;r++){ tr.absorb("root",pf.roots[r].data(),32); const SumMsg& m=pf.msgs[r];
        if(!gl2_eq(gl2_add(m.s0,m.s1),claim)) return fail("sumcheck claim");
        tr.absorb("sc",&m,sizeof(m)); gl2_t a=chal(tr); al[r]=a; claim=quad_eval(m.s0,m.s1,m.s2,a); }
    tr.absorb("final",pf.final_word.data(),pf.final_word.size()*sizeof(gl2_t));
    for(size_t i=1;i<pf.final_word.size();i++) if(!gl2_eq(pf.final_word[i],pf.final_word[0])) return fail("final not constant");
    gl2_t C=pf.final_word[0];
    if(!gl2_eq(claim, gl2_mul(C, eq_point(al,pf.z)))) return fail("eval tie");
    std::vector<uint32_t> c0s(pf.Q); for(uint32_t q=0;q<pf.Q;q++) c0s[q]=(uint32_t)idx_from(tr,M0/2);
    for(uint32_t q=0;q<pf.Q;q++){ const QueryProof& qp=pf.queries[q]; if(qp.rounds.size()!=v) return fail("round count");
        uint32_t p=c0s[q];
        for(uint32_t r=0;r<v;r++){ uint32_t Mr=M0>>r,half=Mr/2,c=p%half; const RoundOpen& ro=qp.rounds[r];
            if(!verify_path(ro.a,c,ro.pa,pf.roots[r])) return fail("merkle a");
            if(!verify_path(ro.b,c+half,ro.pb,pf.roots[r])) return fail("merkle b");
            gl_t w=gl_root_of_unity(logM0-r), x=gl_pow(w,c), invx=gl_inv(x);
            gl2_t folded=fold_pair(ro.a,ro.b,invx,al[r]);
            if(r+1<v){ uint32_t nhalf=(M0>>(r+1))/2; gl2_t val=(c<nhalf)?qp.rounds[r+1].a:qp.rounds[r+1].b;
                if(!gl2_eq(val,folded)) return fail("fold link"); }
            else if(!gl2_eq(pf.final_word[c],folded)) return fail("fold to final");
            p=c; }
    }
    if(why)*why="ok"; return true;
}
} // namespace p3bf2
