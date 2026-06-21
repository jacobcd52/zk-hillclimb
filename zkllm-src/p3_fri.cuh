// P3.3 FRI low-degree test over Goldilocks: Merkle-committed codewords, per-round
// folding with Fiat-Shamir challenges, Q authenticated queries.  This is the
// hash-PCS proximity engine that P3.4 (Basefold eval) and the matmul argument
// build on.  Soundness-critical -> validated by p3_fri_selftest (honest accept,
// every tamper rejects).
//
// Domain: multiplicative subgroup of order M_0 = 2^(logN+R), w = M_0-th root.
// f_0[j] = poly(w^j), poly of degree < N_0 = 2^logN (rate 1/2^R).
// Fold:  f'(x^2) = (f(x)+f(-x))/2 + beta*(f(x)-f(-x))/(2x),  x=-x at index j+M/2.
// k = logN fold rounds -> final codeword length 2^R must be constant (degree 0).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include "p3_goldilocks.cuh"
#include "fs_transcript.hpp"

namespace p3fri {

typedef std::array<uint8_t, 32> Hash;

static inline void gl_to_le(gl_t v, uint8_t b[8]) { for (int k = 0; k < 8; k++) b[k] = (uint8_t)(v >> (8 * k)); }

static inline Hash leaf_hash(gl_t v) {
    uint8_t b[8]; gl_to_le(v, b); Hash h; fs::sha256(b, 8, h.data()); return h;
}
static inline Hash node_hash(const Hash& l, const Hash& r) {
    uint8_t b[64]; memcpy(b, l.data(), 32); memcpy(b + 32, r.data(), 32);
    Hash h; fs::sha256(b, 64, h.data()); return h;
}

// host Merkle tree over a codeword (size must be power of 2)
struct Merkle {
    std::vector<std::vector<Hash>> levels;   // levels[0] = leaves
    void build(const std::vector<gl_t>& cw) {
        levels.clear();
        std::vector<Hash> lv(cw.size());
        for (size_t i = 0; i < cw.size(); i++) lv[i] = leaf_hash(cw[i]);
        levels.push_back(lv);
        while (levels.back().size() > 1) {
            const auto& cur = levels.back();
            std::vector<Hash> nxt(cur.size() / 2);
            for (size_t i = 0; i < nxt.size(); i++) nxt[i] = node_hash(cur[2*i], cur[2*i+1]);
            levels.push_back(nxt);
        }
    }
    Hash root() const { return levels.back()[0]; }
    std::vector<Hash> path(size_t idx) const {
        std::vector<Hash> p;
        for (size_t d = 0; d + 1 < levels.size(); d++) { p.push_back(levels[d][idx ^ 1]); idx >>= 1; }
        return p;
    }
};

static inline bool verify_path(Hash leaf, size_t idx, const std::vector<Hash>& path, const Hash& root) {
    Hash h = leaf;
    for (const auto& sib : path) { h = (idx & 1) ? node_hash(sib, h) : node_hash(h, sib); idx >>= 1; }
    return h == root;
}

struct RoundOpen { gl_t a, b; std::vector<Hash> pa, pb; };   // values at coset idx c and c+half
struct QueryProof { std::vector<RoundOpen> rounds; };
struct FriProof {
    uint32_t logN, R, Q;
    std::vector<Hash> roots;        // k = logN round roots (f_0..f_{k-1})
    std::vector<gl_t> final_word;   // f_k, length 2^R, sent in clear
    std::vector<QueryProof> queries;
};

// honest fold of a round codeword (size M) with challenge beta; domain root w_M.
static inline std::vector<gl_t> fold(const std::vector<gl_t>& f, gl_t w_M, gl_t beta) {
    uint32_t M = (uint32_t)f.size(), half = M / 2;
    gl_t inv2 = gl_inv(2ULL);
    gl_t winv = gl_inv(w_M);          // w_M^{-1}
    std::vector<gl_t> out(half);
    gl_t inv_x = 1ULL;                 // w_M^{-c}
    for (uint32_t c = 0; c < half; c++) {
        gl_t a = f[c], b = f[c + half];
        gl_t s = gl_mul(gl_add(a, b), inv2);                       // (a+b)/2
        gl_t d = gl_mul(gl_mul(gl_sub(a, b), inv2), inv_x);        // (a-b)/(2x)
        out[c] = gl_add(s, gl_mul(beta, d));
        inv_x = gl_mul(inv_x, winv);
    }
    return out;
}

static inline gl_t beta_from(fs::Transcript& tr) {
    uint8_t c[32]; tr.challenge_bytes(c);
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)c[i] << (8 * i);
    return v % GL_P;
}
static inline uint64_t idx_from(fs::Transcript& tr, uint64_t mod) {
    uint8_t c[32]; tr.challenge_bytes(c);
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)c[i] << (8 * i);
    return v % mod;
}

// ---------------- prover ----------------
static inline FriProof prove(std::vector<gl_t> cw0, uint32_t logN, uint32_t R, uint32_t Q,
                             const std::string& seed = "p3-fri") {
    FriProof pf; pf.logN = logN; pf.R = R; pf.Q = Q;
    uint32_t logM0 = logN + R, M0 = 1u << logM0;
    fs::Transcript tr(seed);

    std::vector<std::vector<gl_t>> words;   // words[r] = f_r, r=0..k (k=logN)
    std::vector<Merkle> trees;              // trees[r] for r=0..k-1
    std::vector<gl_t> betas;
    words.push_back(cw0);
    for (uint32_t r = 0; r < logN; r++) {
        Merkle mk; mk.build(words[r]); trees.push_back(mk);
        pf.roots.push_back(mk.root());
        tr.absorb("root", mk.root().data(), 32);
        gl_t beta = beta_from(tr); betas.push_back(beta);
        uint32_t logMr = logM0 - r;
        gl_t w = gl_root_of_unity(logMr);
        words.push_back(fold(words[r], w, beta));
    }
    pf.final_word = words[logN];            // length 2^R
    tr.absorb("final", pf.final_word.data(), pf.final_word.size() * sizeof(gl_t));

    for (uint32_t q = 0; q < Q; q++) {
        uint32_t c0 = (uint32_t)idx_from(tr, M0 / 2);
        QueryProof qp;
        uint32_t p = c0;
        for (uint32_t r = 0; r < logN; r++) {
            uint32_t Mr = M0 >> r, half = Mr / 2, c = p % half;
            RoundOpen ro;
            ro.a = words[r][c]; ro.b = words[r][c + half];
            ro.pa = trees[r].path(c); ro.pb = trees[r].path(c + half);
            qp.rounds.push_back(ro);
            p = c;
        }
        pf.queries.push_back(qp);
    }
    return pf;
}

// ---------------- verifier ----------------
static inline bool verify(const FriProof& pf, const std::string& seed = "p3-fri", const char** why = nullptr) {
    auto fail = [&](const char* m){ if (why) *why = m; return false; };
    uint32_t logN = pf.logN, R = pf.R, logM0 = logN + R, M0 = 1u << logM0;
    if (pf.roots.size() != logN) return fail("root count");
    if (pf.final_word.size() != (1u << R)) return fail("final size");
    if (pf.queries.size() != pf.Q) return fail("query count");

    fs::Transcript tr(seed);
    std::vector<gl_t> betas(logN);
    for (uint32_t r = 0; r < logN; r++) {
        tr.absorb("root", pf.roots[r].data(), 32);
        betas[r] = beta_from(tr);
    }
    tr.absorb("final", pf.final_word.data(), pf.final_word.size() * sizeof(gl_t));

    // final codeword must be constant (degree 0)
    for (size_t i = 1; i < pf.final_word.size(); i++)
        if (pf.final_word[i] != pf.final_word[0]) return fail("final not constant");

    gl_t inv2 = gl_inv(2ULL);
    for (uint32_t q = 0; q < pf.Q; q++) {
        uint32_t c0 = (uint32_t)idx_from(tr, M0 / 2);
        const QueryProof& qp = pf.queries[q];
        if (qp.rounds.size() != logN) return fail("round count");
        uint32_t p = c0;
        for (uint32_t r = 0; r < logN; r++) {
            uint32_t Mr = M0 >> r, half = Mr / 2, c = p % half;
            const RoundOpen& ro = qp.rounds[r];
            // Merkle authentication of the revealed pair
            if (!verify_path(leaf_hash(ro.a), c,        ro.pa, pf.roots[r])) return fail("merkle a");
            if (!verify_path(leaf_hash(ro.b), c + half, ro.pb, pf.roots[r])) return fail("merkle b");
            // fold at x = w_Mr^c
            uint32_t logMr = logM0 - r;
            gl_t w = gl_root_of_unity(logMr);
            gl_t x = gl_pow(w, c), invx = gl_inv(x);
            gl_t s = gl_mul(gl_add(ro.a, ro.b), inv2);
            gl_t d = gl_mul(gl_mul(gl_sub(ro.a, ro.b), inv2), invx);
            gl_t folded = gl_add(s, gl_mul(betas[r], d));
            // the folded value must equal f_{r+1}[c]
            if (r + 1 < logN) {
                uint32_t nhalf = (M0 >> (r + 1)) / 2;
                uint32_t cn = c % nhalf;
                gl_t val = (c < nhalf) ? qp.rounds[r+1].a : qp.rounds[r+1].b;
                (void)cn;
                if (val != folded) return fail("fold link");
            } else {
                if (pf.final_word[c] != folded) return fail("fold to final");
            }
            p = c;
        }
    }
    if (why) *why = "ok";
    return true;
}

} // namespace p3fri
