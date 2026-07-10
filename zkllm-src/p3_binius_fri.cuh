// p3_binius_fri.cuh -- FRI low-degree test / opening over the binary tower
// (design doc section 21.17, lever 4).  Replaces the Ligero O(sqrt n) query
// phase with a logarithmic FRI fold on the additive-NTT (LCH novel-basis)
// codeword, giving polylog proof size / verify.
//
// THE ADDITIVE FOLD.  A rate-2^-R codeword d of length 2^m is the novel-basis
// encoding of a message with only its low 2^(m-R) coefficients nonzero:
// d[u] = f(P_u), P_u = sum_i u_i * B[i] over the layer's subspace basis B.
// Writing f(X) = f0(q0(X)) + What0(X)*f1(q0(X)) with q0(X) = X*(X+b0) (kernel
// span(b0), b0 = B[0]) and What0(X) = X/b0, the pair (2j, 2j+1) -- domain points
// differing by b0 -- gives
//     f1 = d[2j] ^ d[2j+1]
//     d'[j] = d[2j] ^ (P_{2j}/b0 + r) * f1        (= f0 + r*f1, the fold at r)
// and the folded domain has basis B'[i] = q0(B[i+1]) (F_2-linear, so the folded
// index j addresses q0(P_{2j})).  After m-R folds the message is one coefficient
// and the codeword is CONSTANT -- that constant-collapse IS the low-degree test.
// A prover far from any low-degree codeword survives a fold with bounded
// probability, so Q random query chains (each checking one fold triple + Merkle
// paths, layer to layer) give the standard FRI soundness.
#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include "fs_transcript.hpp"
#include "p3_binius_field.cuh"
#include "p3_binius_ntt.cuh"
#include "p3_binius_pcs.cuh"   // bf_chal128, bfntt_fwd_host128
#include "p3_merkle.cuh"

#define BFFRI_RATE_LOG 2                 // rate 1/4 (match bfpcs)

// domain point for index u in basis B (T_16): XOR of B[i] over set bits i of u
static inline bf16_t bffri_point(uint32_t u, const std::vector<bf16_t>& B) {
    uint32_t p = 0;
    while (u) { int i = __builtin_ctz(u); u &= u - 1; p ^= B[i]; }
    return (bf16_t)p;
}
// one fold layer: cw (length 2^lm, T_128) + basis B (lm elts) -> cw' (2^(lm-1)),
// B' (lm-1 elts), at challenge r
static inline void bffri_fold(const std::vector<bf128_t>& cw, std::vector<bf16_t>& B,
                              bf128_t r, std::vector<bf128_t>& out) {
    size_t half = cw.size() / 2;
    bf16_t b0 = B[0], b0inv = (bf16_t)bf16_inv(b0);
    out.resize(half);
    for (size_t j = 0; j < half; j++) {
        bf128_t d0 = cw[2 * j], d1 = cw[2 * j + 1];
        bf128_t f1 = bf128_add(d0, d1);
        bf16_t pj = bffri_point((uint32_t)(2 * j), B);
        bf128_t mult = bf128_add(bf128_from16((bf16_t)bf16_mul(pj, b0inv)), r);
        out[j] = bf128_add(d0, bf128_mul(mult, f1));
    }
    std::vector<bf16_t> Bn(B.size() - 1);
    for (size_t i = 0; i + 1 < B.size(); i++) {           // B'[i] = q0(B[i+1])
        bf16_t x = B[i + 1];
        Bn[i] = (bf16_t)(bf16_mul(x, x) ^ bf16_mul(x, b0));
    }
    B.swap(Bn);
}

// ---- Merkle over a T_128 codeword (leaf = SHA256 of the 16-byte element) ----
static inline void bffri_tree(const std::vector<bf128_t>& cw,
                              std::vector<std::vector<uint8_t>>& lvl, uint8_t root[32]) {
    size_t n = cw.size();
    std::vector<uint8_t> leaves(n * 32);
    for (size_t i = 0; i < n; i++) fs::sha256(&cw[i], sizeof(bf128_t), leaves.data() + i * 32);
    lvl.clear(); lvl.push_back(std::move(leaves));
    while (lvl.back().size() > 32) {
        const auto& prev = lvl.back();
        std::vector<uint8_t> nx(prev.size() / 2);
        for (size_t i = 0; i < nx.size() / 32; i++)
            p3_sha256_compress64(prev.data() + i * 64, nx.data() + i * 32);
        lvl.push_back(std::move(nx));
    }
    memcpy(root, lvl.back().data(), 32);
}
static inline void bffri_path(const std::vector<std::vector<uint8_t>>& lvl, uint32_t idx,
                              std::vector<uint8_t>& path) {
    int depth = (int)lvl.size() - 1;
    path.assign((size_t)depth * 32, 0);
    uint32_t k = idx;
    for (int d = 0; d < depth; d++) { memcpy(path.data() + (size_t)d * 32, lvl[d].data() + (size_t)(k ^ 1) * 32, 32); k >>= 1; }
}
static inline bool bffri_check_path(const uint8_t* leaf, uint32_t idx, const uint8_t* path,
                                    int depth, const uint8_t root[32]) {
    uint8_t h[32]; memcpy(h, leaf, 32); uint32_t k = idx;
    for (int d = 0; d < depth; d++) {
        uint8_t cat[64];
        if (k & 1) { memcpy(cat, path + (size_t)d * 32, 32); memcpy(cat + 32, h, 32); }
        else       { memcpy(cat, h, 32); memcpy(cat + 32, path + (size_t)d * 32, 32); }
        p3_sha256_compress64(cat, h); k >>= 1;
    }
    return memcmp(h, root, 32) == 0;
}

struct BfFriProof {
    std::vector<uint8_t> roots;                 // 32 bytes per folded layer
    bf128_t final_const = bf128_zero();         // the collapsed constant
    int nlayers = 0, lm0 = 0, Q = 0;
    // per query: for each layer, (leaf pair d0,d1) + two paths
    std::vector<bf128_t> qd;                    // Q * nlayers * 2 leaves
    std::vector<uint8_t> qpath;                 // Q * nlayers * 2 * depth32 (varying depth per layer)
    std::vector<uint32_t> qoff;                 // path offsets per (query,layer)
    size_t bytes() const { return roots.size() + sizeof final_const + qd.size()*sizeof(bf128_t) + qpath.size(); }
};

// PROVE the low-degree test for a rate-2^-R codeword cw (length 2^lm0, basis
// B0 = {1,2,4,...}); folds to a constant, Q query chains.
static inline void bffri_prove(std::vector<bf128_t> cw, int lm0, int R, int Q,
                               fs::Transcript& tr, BfFriProof& pf) {
    pf.lm0 = lm0; pf.Q = Q;
    std::vector<bf16_t> B(lm0); for (int i = 0; i < lm0; i++) B[i] = (bf16_t)(1u << i);
    std::vector<std::vector<bf128_t>> layers; layers.push_back(cw);
    std::vector<std::vector<std::vector<uint8_t>>> trees;
    int nf = lm0 - R;                            // folds to length 2^R
    pf.nlayers = nf;
    std::vector<bf128_t> rr(nf);
    std::vector<bf128_t> cur = cw;
    for (int L = 0; L < nf; L++) {
        std::vector<std::vector<uint8_t>> lvl; uint8_t root[32];
        bffri_tree(cur, lvl, root);
        pf.roots.insert(pf.roots.end(), root, root + 32);
        tr.absorb("bffri-root", root, 32);
        trees.push_back(lvl);
        bf128_t r = bf_chal128(tr); rr[L] = r;
        std::vector<bf128_t> nxt; bffri_fold(cur, B, r, nxt);
        cur.swap(nxt); layers.push_back(cur);
    }
    // final layer must be constant (all equal); send that constant
    pf.final_const = cur[0];
    tr.absorb("bffri-final", &pf.final_const, sizeof(bf128_t));
    // queries
    pf.qd.clear(); pf.qpath.clear(); pf.qoff.clear();
    for (int q = 0; q < Q; q++) {
        uint8_t b[32]; tr.challenge_bytes(b);
        uint32_t idx; memcpy(&idx, b, 4);
        idx &= (1u << (lm0 - 1)) - 1;            // index into the FIRST folded half
        uint32_t cidx = idx;
        for (int L = 0; L < nf; L++) {
            const auto& lay = layers[L];         // length 2^(lm0-L)
            uint32_t base = cidx * 2;
            pf.qd.push_back(lay[base]); pf.qd.push_back(lay[base + 1]);
            std::vector<uint8_t> p0, p1;
            bffri_path(trees[L], base, p0); bffri_path(trees[L], base + 1, p1);
            pf.qoff.push_back((uint32_t)pf.qpath.size());
            pf.qpath.insert(pf.qpath.end(), p0.begin(), p0.end());
            pf.qpath.insert(pf.qpath.end(), p1.begin(), p1.end());
            cidx = cidx >> 1;                    // fold index of next layer
            if (L + 1 < nf) idx = base >> 1;     // (unused; kept for clarity)
        }
    }
}

static inline bool bffri_verify(const BfFriProof& pf, int R, fs::Transcript& tr) {
    int lm0 = pf.lm0, nf = pf.nlayers, Q = pf.Q;
    if ((int)pf.roots.size() != nf * 32) return false;
    std::vector<bf16_t> B(lm0); for (int i = 0; i < lm0; i++) B[i] = (bf16_t)(1u << i);
    std::vector<std::vector<bf16_t>> Blayers; Blayers.push_back(B);
    std::vector<bf128_t> rr(nf);
    for (int L = 0; L < nf; L++) {
        tr.absorb("bffri-root", pf.roots.data() + (size_t)L * 32, 32);
        rr[L] = bf_chal128(tr);
        std::vector<bf16_t> Bn = Blayers[L]; bf16_t b0 = Bn[0];
        std::vector<bf16_t> B2(Bn.size() - 1);
        for (size_t i = 0; i + 1 < Bn.size(); i++) { bf16_t x = Bn[i + 1]; B2[i] = (bf16_t)(bf16_mul(x, x) ^ bf16_mul(x, b0)); }
        Blayers.push_back(B2);
    }
    tr.absorb("bffri-final", &pf.final_const, sizeof(bf128_t));
    // recompute the query indices and check every fold triple + Merkle path
    size_t di = 0, oi = 0;
    for (int q = 0; q < Q; q++) {
        uint8_t b[32]; tr.challenge_bytes(b);
        uint32_t idx; memcpy(&idx, b, 4);
        idx &= (1u << (lm0 - 1)) - 1;
        uint32_t cidx = idx;
        bf128_t carry; bool have_carry = false;
        for (int L = 0; L < nf; L++) {
            uint32_t base = cidx * 2;
            bf128_t d0 = pf.qd[di], d1 = pf.qd[di + 1]; di += 2;
            int depth = lm0 - L;                 // layer length 2^(lm0-L) -> depth = lm0-L
            const uint8_t* p0 = pf.qpath.data() + pf.qoff[oi];
            const uint8_t* p1 = p0 + (size_t)depth * 32;
            oi++;
            uint8_t lf0[32], lf1[32];
            fs::sha256(&d0, sizeof(bf128_t), lf0); fs::sha256(&d1, sizeof(bf128_t), lf1);
            const uint8_t* root = pf.roots.data() + (size_t)L * 32;
            if (!bffri_check_path(lf0, base, p0, depth, root)) return false;
            if (!bffri_check_path(lf1, base + 1, p1, depth, root)) return false;
            // the carry folded down from the previous layer sits at position
            // (idx>>(L-1))&1 of this layer's opened pair
            if (have_carry) {
                bf128_t want = ((idx >> (L - 1)) & 1) ? d1 : d0;
                if (!bf128_eq(want, carry)) return false;
            }
            // fold this triple with rr[L]
            bf16_t b0 = Blayers[L][0], b0inv = (bf16_t)bf16_inv(b0);
            bf128_t f1 = bf128_add(d0, d1);
            bf16_t pj = bffri_point(base, Blayers[L]);
            bf128_t mult = bf128_add(bf128_from16((bf16_t)bf16_mul(pj, b0inv)), rr[L]);
            carry = bf128_add(d0, bf128_mul(mult, f1)); have_carry = true;
            cidx >>= 1;
        }
        // after nf folds the carry is the folded value at the final layer; it
        // must equal the (constant) final codeword.
        if (!bf128_eq(carry, pf.final_const)) return false;
    }
    return true;
}
