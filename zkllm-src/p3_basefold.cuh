// P3.4 Basefold multilinear-evaluation opening over Goldilocks.
//
// Commit: RS-encode the length-N=2^v coefficient vector c (root = roots[0]); the
//   committed polynomial is h = multilinear extension of the array c on the cube.
// Open at z in F^v with claimed value y = h(z) = sum_b c[b]*eq(b,z):
//   v rounds, each binding variable r.  Per round we (a) send the degree-2
//   sumcheck message of  sum_b c[b]*eq(b,z), get challenge alpha_r, then (b)
//   MLE-fold the codeword with the SAME alpha_r.  The MLE fold tracks the
//   alpha-binding of c, so the final folded constant C = c~(alpha) and the
//   running sumcheck claim must equal C * eq(alpha,z).
//   Q FRI queries authenticate the folds against roots[0..v-1] (the commitment).
//
// Soundness note: challenges are drawn from the 64-bit base field, giving ~2^-58
// for our round counts; a degree-2 extension (mechanical) lifts this to ~2^-116
// and is the one production change flagged for this module.
#pragma once
#include <cstdint>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
#include "p3_ntt.cuh"
#include "fs_transcript.hpp"

namespace p3bf {
using p3fri::Hash; using p3fri::Merkle; using p3fri::RoundOpen; using p3fri::QueryProof;
using p3fri::leaf_hash; using p3fri::verify_path; using p3fri::fold; using p3fri::idx_from;

struct SumMsg { gl_t s0, s1, s2; };           // sumcheck univariate at t=0,1,2
struct EvalProof {
    uint32_t logN, R, Q;                       // v = logN variables
    std::vector<Hash>     roots;               // v round roots (roots[0] = commitment)
    std::vector<SumMsg>   msgs;                 // v sumcheck messages
    std::vector<gl_t>     final_word;           // length 2^R, in clear
    std::vector<QueryProof> queries;
    std::vector<gl_t>     z;                     // opening point (v values)
    gl_t                  y;                     // claimed g(z)
};

// eq weights over the hypercube: w[b] = eq(b,z) = prod_i (b_i?z_i:(1-z_i)), length 2^v.
static inline std::vector<gl_t> build_eq(const std::vector<gl_t>& z) {
    uint32_t v = (uint32_t)z.size(), N = 1u << v;
    std::vector<gl_t> w(N, 1ULL);
    for (uint32_t b = 0; b < N; b++) {
        gl_t prod = 1ULL;
        for (uint32_t i = 0; i < v; i++)
            prod = gl_mul(prod, (b & (1u << i)) ? z[i] : gl_sub(1ULL, z[i]));
        w[b] = prod;
    }
    return w;
}
// committed-poly value h(z) = c~(z) = sum_b c[b]*eq(b,z).
static inline gl_t eval_h(const std::vector<gl_t>& c, const std::vector<gl_t>& eq) {
    gl_t acc = 0; for (size_t b = 0; b < c.size(); b++) acc = gl_add(acc, gl_mul(c[b], eq[b])); return acc;
}

// log2 of a power-of-two.
static inline uint32_t ilog2(uint32_t n) { uint32_t l = 0; while ((1u << l) < n) l++; return l; }

// RS-encode a length-N=2^v coefficient array onto the size-2^(v+R) subgroup (Horner).
// O(N*2^(v+R)); fine for tests, replaced by GPU NTT in the perf pass.
static inline std::vector<gl_t> rs_encode(const std::vector<gl_t>& c, uint32_t R) {
    uint32_t v = ilog2((uint32_t)c.size()), logM0 = v + R, M0 = 1u << logM0;
    gl_t w = gl_root_of_unity(logM0);
    std::vector<gl_t> cw(M0);
    for (uint32_t jx = 0; jx < M0; jx++) {
        gl_t xj = gl_pow(w, jx), p = 0;
        for (int i = (int)c.size() - 1; i >= 0; i--) p = gl_add(gl_mul(p, xj), c[i]);
        cw[jx] = p;
    }
    return cw;
}
// Commit to coefficient array c: returns Merkle root, fills cw with the codeword.
static inline Hash commit(const std::vector<gl_t>& c, uint32_t R, std::vector<gl_t>& cw) {
    cw = rs_encode(c, R); Merkle mk; mk.build(cw); return mk.root();
}

// GPU Reed-Solomon encode (forward NTT of the zero-padded coeff vector) -- same
// codeword as rs_encode but in ms instead of the O(N*M) host Horner.
static inline std::vector<gl_t> rs_encode_gpu(const std::vector<gl_t>& c, uint32_t R) {
    uint32_t v = ilog2((uint32_t)c.size()), logM0 = v + R, M0 = 1u << logM0;
    gl_t *d_in, *d_out;
    cudaMalloc(&d_in, (size_t)M0 * sizeof(gl_t)); cudaMalloc(&d_out, (size_t)M0 * sizeof(gl_t));
    cudaMemset(d_in, 0, (size_t)M0 * sizeof(gl_t));
    cudaMemcpy(d_in, c.data(), c.size() * sizeof(gl_t), cudaMemcpyHostToDevice);
    { P3Ntt ntt(logM0); ntt.run(d_in, d_out, true); }
    std::vector<gl_t> cw(M0);
    cudaMemcpy(cw.data(), d_out, (size_t)M0 * sizeof(gl_t), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
    return cw;
}
static inline Hash commit_gpu(const std::vector<gl_t>& c, uint32_t R, std::vector<gl_t>& cw) {
    cw = rs_encode_gpu(c, R);
    p3fri::DeviceMerkle mk; mk.build(cw); Hash r = mk.root(); mk.free_(); return r;  // root only, no host levels
}

// GPU MLE fold of a codeword (openings use mle=true): out[c] = (1-beta)E + beta*O,
// E=(f[c]+f[c+half])/2, O=(f[c]-f[c+half])/(2*x), x=w_M^c.  Replaces the O(M) host fold.
__global__ void p3bf_fold_kernel(const gl_t* f, gl_t* out, uint32_t half, gl_t winv, gl_t beta, gl_t inv2) {
    uint32_t c = blockIdx.x*blockDim.x + threadIdx.x; if (c >= half) return;
    gl_t invx = gl_pow(winv, c), a = f[c], b = f[c+half];
    gl_t E = gl_mul(gl_add(a,b), inv2);
    gl_t O = gl_mul(gl_mul(gl_sub(a,b), inv2), invx);
    out[c] = gl_add(gl_mul(gl_sub(1ULL,beta), E), gl_mul(beta, O));
}
static inline std::vector<gl_t> fold_gpu(const std::vector<gl_t>& f, gl_t w_M, gl_t beta) {
    uint32_t M=(uint32_t)f.size(), half=M/2;
    gl_t *d_in,*d_out; cudaMalloc(&d_in,(size_t)M*sizeof(gl_t)); cudaMalloc(&d_out,(size_t)half*sizeof(gl_t));
    cudaMemcpy(d_in,f.data(),(size_t)M*sizeof(gl_t),cudaMemcpyHostToDevice);
    gl_t winv=gl_inv(w_M), inv2=gl_inv(2ULL);
    p3bf_fold_kernel<<<(half+255)/256,256>>>(d_in,d_out,half,winv,beta,inv2);
    std::vector<gl_t> out(half); cudaMemcpy(out.data(),d_out,(size_t)half*sizeof(gl_t),cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out); return out;
}

static inline gl_t alpha_from(fs::Transcript& tr) {
    uint8_t b[32]; tr.challenge_bytes(b);
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v % GL_P;
}
// evaluate the degree-2 poly through (0,s0),(1,s1),(2,s2) at t.
static inline gl_t quad_eval(gl_t s0, gl_t s1, gl_t s2, gl_t t) {
    gl_t inv2 = gl_inv(2ULL);
    gl_t t1 = gl_sub(t, 1ULL), t2 = gl_sub(t, 2ULL);
    gl_t L0 = gl_mul(gl_mul(t1, t2), inv2);                 // (t-1)(t-2)/2
    gl_t L1 = gl_sub(0ULL, gl_mul(t, t2));                  // -t(t-2)
    gl_t L2 = gl_mul(gl_mul(t, t1), inv2);                  // t(t-1)/2
    return gl_add(gl_add(gl_mul(s0, L0), gl_mul(s1, L1)), gl_mul(s2, L2));
}
// eq(alpha,z) = prod_i (a_i z_i + (1-a_i)(1-z_i))
static inline gl_t eq_point(const std::vector<gl_t>& alpha, const std::vector<gl_t>& z) {
    gl_t prod = 1ULL;
    for (size_t i = 0; i < alpha.size(); i++) {
        gl_t t = gl_add(gl_mul(alpha[i], z[i]), gl_mul(gl_sub(1ULL, alpha[i]), gl_sub(1ULL, z[i])));
        prod = gl_mul(prod, t);
    }
    return prod;
}

// ---------------- prover ----------------
static inline EvalProof prove_eval(std::vector<gl_t> c, const std::vector<gl_t>& z, gl_t y,
                                   uint32_t R, uint32_t Q,
                                   const std::vector<gl_t>& cw0, const std::string& seed = "p3-bf") {
    uint32_t v = (uint32_t)z.size(), logM0 = v + R, M0 = 1u << logM0;
    EvalProof pf; pf.logN = v; pf.R = R; pf.Q = Q; pf.z = z; pf.y = y;
    fs::Transcript tr(seed);
    tr.absorb("z", z.data(), z.size() * sizeof(gl_t));
    tr.absorb("y", &y, sizeof(gl_t));

    std::vector<std::vector<gl_t>> words; words.push_back(cw0);
    bool GM = p3fri::g_gpu_merkle;                 // device-resident Merkle for large openings
    std::vector<Merkle> trees;
    std::vector<p3fri::DeviceMerkle> dtrees;
    std::vector<gl_t> cur_c = c, cur_w = build_eq(z), alphas;

    for (uint32_t r = 0; r < v; r++) {
        Hash root;
        if (GM) { p3fri::DeviceMerkle mk; mk.build(words[r]); root=mk.root(); dtrees.push_back(mk); }
        else    { Merkle mk; mk.build(words[r]); root=mk.root(); trees.push_back(mk); }
        pf.roots.push_back(root);
        tr.absorb("root", root.data(), 32);
        // sumcheck message over the LSB split of cur_c, cur_w
        uint32_t n = (uint32_t)cur_c.size(), half = n / 2;
        gl_t s0 = 0, s1 = 0, s2 = 0;
        for (uint32_t b = 0; b < half; b++) {
            gl_t cl = cur_c[2*b], ch = cur_c[2*b+1], wl = cur_w[2*b], wh = cur_w[2*b+1];
            s0 = gl_add(s0, gl_mul(cl, wl));
            s1 = gl_add(s1, gl_mul(ch, wh));
            gl_t c2 = gl_sub(gl_add(ch, ch), cl), w2 = gl_sub(gl_add(wh, wh), wl);  // val at t=2
            s2 = gl_add(s2, gl_mul(c2, w2));
        }
        SumMsg msg{s0, s1, s2}; pf.msgs.push_back(msg);
        tr.absorb("sc", &msg, sizeof(SumMsg));
        gl_t a = alpha_from(tr); alphas.push_back(a);
        // bind cur_c, cur_w to alpha (LSB) ; fold codeword with same alpha
        std::vector<gl_t> nc(half), nw(half);
        for (uint32_t b = 0; b < half; b++) {
            nc[b] = gl_add(cur_c[2*b], gl_mul(a, gl_sub(cur_c[2*b+1], cur_c[2*b])));
            nw[b] = gl_add(cur_w[2*b], gl_mul(a, gl_sub(cur_w[2*b+1], cur_w[2*b])));
        }
        cur_c = nc; cur_w = nw;
        gl_t w = gl_root_of_unity(logM0 - r);
        words.push_back(GM ? fold_gpu(words[r], w, a) : fold(words[r], w, a, /*mle=*/true));
    }
    pf.final_word = words[v];
    tr.absorb("final", pf.final_word.data(), pf.final_word.size() * sizeof(gl_t));

    for (uint32_t q = 0; q < Q; q++) {
        uint32_t c0 = (uint32_t)idx_from(tr, M0 / 2);
        QueryProof qp; uint32_t p = c0;
        for (uint32_t r = 0; r < v; r++) {
            uint32_t Mr = M0 >> r, h = Mr / 2, cc = p % h;
            RoundOpen ro; ro.a = words[r][cc]; ro.b = words[r][cc + h];
            if (GM) { ro.pa = dtrees[r].path(cc); ro.pb = dtrees[r].path(cc + h); }
            else    { ro.pa = trees[r].path(cc);  ro.pb = trees[r].path(cc + h); }
            qp.rounds.push_back(ro); p = cc;
        }
        pf.queries.push_back(qp);
    }
    if (GM) for (auto& t : dtrees) t.free_();
    return pf;
}

// ---------------- verifier ----------------
static inline bool verify_eval(const EvalProof& pf, const std::string& seed = "p3-bf", const char** why = nullptr) {
    auto fail = [&](const char* m){ if (why) *why = m; return false; };
    uint32_t v = pf.logN, R = pf.R, M0 = 1u << (v + R);
    if (pf.roots.size() != v || pf.msgs.size() != v) return fail("size");
    if (pf.final_word.size() != (1u << R)) return fail("final size");
    if (pf.queries.size() != pf.Q) return fail("query count");
    if (pf.z.size() != v) return fail("z size");

    fs::Transcript tr(seed);
    tr.absorb("z", pf.z.data(), pf.z.size() * sizeof(gl_t));
    tr.absorb("y", &pf.y, sizeof(gl_t));

    std::vector<gl_t> alphas(v);
    gl_t claim = pf.y;
    for (uint32_t r = 0; r < v; r++) {
        tr.absorb("root", pf.roots[r].data(), 32);
        const SumMsg& m = pf.msgs[r];
        if (gl_add(m.s0, m.s1) != claim) return fail("sumcheck claim");   // s(0)+s(1)==H
        tr.absorb("sc", &m, sizeof(SumMsg));
        gl_t a = alpha_from(tr); alphas[r] = a;
        claim = quad_eval(m.s0, m.s1, m.s2, a);
    }
    tr.absorb("final", pf.final_word.data(), pf.final_word.size() * sizeof(gl_t));

    for (size_t i = 1; i < pf.final_word.size(); i++)
        if (pf.final_word[i] != pf.final_word[0]) return fail("final not constant");

    // tie: final sumcheck claim == c~(alpha) * eq(alpha,z), with c~(alpha)=final constant C
    gl_t C = pf.final_word[0];
    if (claim != gl_mul(C, eq_point(alphas, pf.z))) return fail("eval tie");

    std::vector<uint32_t> c0s(pf.Q);
    for (uint32_t q = 0; q < pf.Q; q++) c0s[q] = (uint32_t)idx_from(tr, M0 / 2);
    if (!p3fri::check_queries(pf.roots, pf.final_word, alphas, pf.queries, v, R, c0s, why, /*mle=*/true)) return false;
    if (why) *why = "ok";
    return true;
}

} // namespace p3bf
