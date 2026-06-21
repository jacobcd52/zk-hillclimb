// P3.4b FC-layer matmul argument:  Y = X . W   (X: B x IN, W: IN x OUT, Y: B x OUT).
//
// Y[i][k] = sum_j X[i][j] W[j][k].  Reduce  Y~(r_i,r_k) = sum_j X~(r_i,j) W~(j,r_k)
// by a sumcheck over the IN contraction variables j -> X~(r_i,r_j)*W~(r_j,r_k), then
// open all three operands at the resulting points with Basefold.
//
// Index layout (low bits first): X[i*IN+j] (j low, i high), W[j*OUT+k] (k low, j high),
// Y[i*OUT+k] (k low, i high).  Sumcheck binds j-bits LSB-first, matching Basefold's
// variable order, so opening points concatenate as zX=[r_j,r_i], zW=[r_k,r_j], zY=[r_k,r_i].
#pragma once
#include <cstdint>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "fs_transcript.hpp"

namespace p3mm {

struct MatmulProof {
    uint32_t bb, ii, oo, R, Q;
    p3fri::Hash rootX, rootW, rootY;
    std::vector<p3bf::SumMsg> mm;          // ii sumcheck messages over j
    p3bf::EvalProof openX, openW, openY;
};

static inline std::vector<gl_t> chal_vec(fs::Transcript& tr, uint32_t n) {
    std::vector<gl_t> v(n); for (uint32_t i = 0; i < n; i++) v[i] = p3bf::alpha_from(tr); return v;
}
static inline std::vector<gl_t> concat(const std::vector<gl_t>& lo, const std::vector<gl_t>& hi) {
    std::vector<gl_t> z = lo; z.insert(z.end(), hi.begin(), hi.end()); return z;
}

// ---------------- prover ----------------
static inline MatmulProof prove(const std::vector<gl_t>& X, const std::vector<gl_t>& W, const std::vector<gl_t>& Y,
                                uint32_t bb, uint32_t ii, uint32_t oo, uint32_t R, uint32_t Q,
                                bool gpu = false) {
    uint32_t B = 1u<<bb, IN = 1u<<ii, OUT = 1u<<oo;
    MatmulProof pf; pf.bb=bb; pf.ii=ii; pf.oo=oo; pf.R=R; pf.Q=Q;
    std::vector<gl_t> cwX, cwW, cwY;
    pf.rootX = gpu ? p3bf::commit_gpu(X, R, cwX) : p3bf::commit(X, R, cwX);
    pf.rootW = gpu ? p3bf::commit_gpu(W, R, cwW) : p3bf::commit(W, R, cwW);
    pf.rootY = gpu ? p3bf::commit_gpu(Y, R, cwY) : p3bf::commit(Y, R, cwY);

    fs::Transcript tr("p3-mm");
    tr.absorb("rX", pf.rootX.data(), 32); tr.absorb("rW", pf.rootW.data(), 32); tr.absorb("rY", pf.rootY.data(), 32);
    std::vector<gl_t> ri = chal_vec(tr, bb), rk = chal_vec(tr, oo);

    // A[j] = X~(r_i,j) = sum_i X[i][j] eq(i,r_i);  Bv[j] = W~(j,r_k) = sum_k W[j][k] eq(k,r_k)
    auto eqi = p3bf::build_eq(ri), eqk = p3bf::build_eq(rk);
    std::vector<gl_t> A(IN,0), Bv(IN,0);
    for (uint32_t j = 0; j < IN; j++) {
        gl_t a=0,b=0;
        for (uint32_t i = 0; i < B;   i++) a = gl_add(a, gl_mul(X[i*IN+j], eqi[i]));
        for (uint32_t k = 0; k < OUT; k++) b = gl_add(b, gl_mul(W[j*OUT+k], eqk[k]));
        A[j]=a; Bv[j]=b;
    }
    // sumcheck over j of sum_j A[j]*Bv[j]
    std::vector<gl_t> curA=A, curB=Bv, rj;
    for (uint32_t r = 0; r < ii; r++) {
        uint32_t half = (uint32_t)curA.size()/2;
        gl_t s0=0,s1=0,s2=0;
        for (uint32_t b=0;b<half;b++){
            gl_t cl=curA[2*b],ch=curA[2*b+1],wl=curB[2*b],wh=curB[2*b+1];
            s0=gl_add(s0,gl_mul(cl,wl)); s1=gl_add(s1,gl_mul(ch,wh));
            gl_t c2=gl_sub(gl_add(ch,ch),cl), w2=gl_sub(gl_add(wh,wh),wl);
            s2=gl_add(s2,gl_mul(c2,w2));
        }
        p3bf::SumMsg m{s0,s1,s2}; pf.mm.push_back(m); tr.absorb("mm",&m,sizeof(m));
        gl_t a=p3bf::alpha_from(tr); rj.push_back(a);
        std::vector<gl_t> nA(half), nB(half);
        for (uint32_t b=0;b<half;b++){
            nA[b]=gl_add(curA[2*b], gl_mul(a, gl_sub(curA[2*b+1],curA[2*b])));
            nB[b]=gl_add(curB[2*b], gl_mul(a, gl_sub(curB[2*b+1],curB[2*b])));
        }
        curA=nA; curB=nB;
    }
    // opening points and values
    std::vector<gl_t> zX = concat(rj, ri), zW = concat(rk, rj), zY = concat(rk, ri);
    gl_t yX = p3bf::eval_h(X, p3bf::build_eq(zX));
    gl_t yW = p3bf::eval_h(W, p3bf::build_eq(zW));
    gl_t yY = p3bf::eval_h(Y, p3bf::build_eq(zY));
    pf.openX = p3bf::prove_eval(X, zX, yX, R, Q, cwX, "p3-mm-X");
    pf.openW = p3bf::prove_eval(W, zW, yW, R, Q, cwW, "p3-mm-W");
    pf.openY = p3bf::prove_eval(Y, zY, yY, R, Q, cwY, "p3-mm-Y");
    return pf;
}

// ---------------- verifier ----------------
static inline bool verify(const MatmulProof& pf, const char** why = nullptr) {
    auto fail = [&](const char* m){ if (why) *why = m; return false; };
    uint32_t bb=pf.bb, ii=pf.ii, oo=pf.oo;
    if (pf.mm.size() != ii) return fail("mm count");

    fs::Transcript tr("p3-mm");
    tr.absorb("rX", pf.rootX.data(), 32); tr.absorb("rW", pf.rootW.data(), 32); tr.absorb("rY", pf.rootY.data(), 32);
    std::vector<gl_t> ri = chal_vec(tr, bb), rk = chal_vec(tr, oo);

    // Y opening pins the initial sumcheck claim Y~(r_i,r_k)
    std::vector<gl_t> zY = concat(rk, ri);
    if (pf.openY.roots.empty() || !(pf.openY.roots[0] == pf.rootY)) return fail("Y commitment");
    if (pf.openY.z != zY) return fail("Y point");
    if (!p3bf::verify_eval(pf.openY, "p3-mm-Y", why)) return false;
    gl_t claim = pf.openY.y;

    std::vector<gl_t> rj;
    for (uint32_t r = 0; r < ii; r++) {
        const p3bf::SumMsg& m = pf.mm[r];
        if (gl_add(m.s0, m.s1) != claim) return fail("matmul sumcheck claim");
        tr.absorb("mm",&m,sizeof(m));
        gl_t a = p3bf::alpha_from(tr); rj.push_back(a);
        claim = p3bf::quad_eval(m.s0, m.s1, m.s2, a);
    }
    // final claim must equal X~(r_i,r_j) * W~(r_j,r_k)
    std::vector<gl_t> zX = concat(rj, ri), zW = concat(rk, rj);
    if (pf.openX.roots.empty() || !(pf.openX.roots[0] == pf.rootX)) return fail("X commitment");
    if (pf.openW.roots.empty() || !(pf.openW.roots[0] == pf.rootW)) return fail("W commitment");
    if (pf.openX.z != zX) return fail("X point");
    if (pf.openW.z != zW) return fail("W point");
    if (!p3bf::verify_eval(pf.openX, "p3-mm-X", why)) return false;
    if (!p3bf::verify_eval(pf.openW, "p3-mm-W", why)) return false;
    if (claim != gl_mul(pf.openX.y, pf.openW.y)) return fail("matmul final tie");
    if (why) *why = "ok";
    return true;
}

} // namespace p3mm
