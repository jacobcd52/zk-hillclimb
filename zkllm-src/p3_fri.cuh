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
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
#include <array>
#include "p3_goldilocks.cuh"
#include "p3_merkle.cuh"
#include "fs_transcript.hpp"

namespace p3fri {

typedef std::array<uint8_t, 32> Hash;

// When set, Merkle::build offloads to the GPU (byte-identical SHA-256 leaves/nodes
// as the host path, so roots/paths/openings are unchanged).  Off by default so the
// pure-host selftests stay host-only.
static bool g_gpu_merkle = false;

static inline void gl_to_le(gl_t v, uint8_t b[8]) { for (int k = 0; k < 8; k++) b[k] = (uint8_t)(v >> (8 * k)); }

static inline Hash leaf_hash(gl_t v) {
    uint8_t b[8]; gl_to_le(v, b); Hash h; fs::sha256(b, 8, h.data()); return h;
}
static inline Hash node_hash(const Hash& l, const Hash& r) {
    uint8_t b[64]; memcpy(b, l.data(), 32); memcpy(b + 32, r.data(), 32);
    Hash h; p3_sha256_compress64(b, h.data()); return h;   // single-block: matches device internal kernel
}

// host Merkle tree over a codeword (size must be power of 2)
struct Merkle {
    std::vector<std::vector<Hash>> levels;   // levels[0] = leaves
    void build(const std::vector<gl_t>& cw) {
        if (g_gpu_merkle && cw.size() >= 1024) { build_gpu(cw); return; }
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
    // GPU build: SHA-256 leaf/internal kernels (p3_merkle.cuh), each level copied to host.
    void build_gpu(const std::vector<gl_t>& cw) {
        uint32_t M = (uint32_t)cw.size();
        gl_t* d_cw; cudaMalloc(&d_cw, (size_t)M * sizeof(gl_t));
        cudaMemcpy(d_cw, cw.data(), (size_t)M * sizeof(gl_t), cudaMemcpyHostToDevice);
        uint8_t *d_a, *d_b; cudaMalloc(&d_a, (size_t)M * 32); cudaMalloc(&d_b, (size_t)M * 32);
        uint32_t blk = (M + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS;
        p3_merkle_leaf_kernel<<<blk, P3_MERKLE_THREADS>>>(d_cw, d_a, M);
        levels.clear();
        std::vector<Hash> lv0(M); cudaMemcpy(lv0.data(), d_a, (size_t)M * 32, cudaMemcpyDeviceToHost);
        levels.push_back(std::move(lv0));
        uint8_t* cur = d_a; uint8_t* nxt = d_b;
        for (uint32_t cnt = M; cnt > 1; ) {
            uint32_t half = cnt / 2;
            uint32_t b = (half + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS;
            p3_merkle_internal_kernel<<<b, P3_MERKLE_THREADS>>>(cur, nxt, half);
            std::vector<Hash> lv(half); cudaMemcpy(lv.data(), nxt, (size_t)half * 32, cudaMemcpyDeviceToHost);
            levels.push_back(std::move(lv));
            uint8_t* t = cur; cur = nxt; nxt = t; cnt = half;
        }
        cudaFree(d_cw); cudaFree(d_a); cudaFree(d_b);
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

// Device-resident Merkle for openings at scale: all levels stay on the GPU; only the
// root (32B) and the O(log M) sibling hashes per queried index are copied to host.
// Avoids the ~2*M*32-byte host materialization that dominated large openings.
// No destructor (shallow-copyable into a vector); call free_() once when done.
struct DeviceMerkle {
    std::vector<uint8_t*> dlev; std::vector<uint32_t> sz;
    uint8_t* slab = nullptr;                 // one allocation for ALL internal levels
    void build(const std::vector<gl_t>& cw) {
        uint32_t M = (uint32_t)cw.size();
        gl_t* d_cw; cudaMalloc(&d_cw, (size_t)M*sizeof(gl_t));
        cudaMemcpy(d_cw, cw.data(), (size_t)M*sizeof(gl_t), cudaMemcpyHostToDevice);
        build_dev(d_cw, M); cudaFree(d_cw); cudaDeviceSynchronize();
    }
    // build from a device-resident codeword (does not upload or free d_cw).
    // Uses the stream-ordered memory pool (cudaMallocAsync) so the GBs of tree memory
    // are RECYCLED across operands/rounds instead of round-tripping to the driver.
    void build_dev(const gl_t* d_cw, uint32_t M) {
        uint8_t* d0; cudaMallocAsync(&d0, (size_t)M*32, 0);
        p3_merkle_leaf_kernel<<<(M+P3_MERKLE_THREADS-1)/P3_MERKLE_THREADS,P3_MERKLE_THREADS>>>(d_cw,d0,M);
        build_leaves(d0, M);
    }
    // build from a precomputed device leaf-hash buffer (takes ownership of d0).
    // All internal levels live in ONE slab allocation and are built two levels
    // per launch (p3_merkle_internal2_kernel) -- identical tree bytes, ~1/4 the
    // GPU API calls (the per-level alloc+launch latency dominated the thousands
    // of small commits in a layer proof).
    void build_leaves(uint8_t* d0, uint32_t M) {
        dlev.assign(1,d0); sz.assign(1,M);
        size_t tot = 0; for (uint32_t c = M/2; ; c /= 2) { tot += c; if (c <= 1) break; }
        cudaMallocAsync(&slab, tot*32, 0);
        uint8_t* p = slab;
        for (uint32_t cnt=M; cnt>1;) {
            if (cnt >= 4) {
                uint32_t half = cnt/2, q = cnt/4;
                uint8_t* l1 = p; p += (size_t)half*32;
                uint8_t* l2 = p; p += (size_t)q*32;
                p3_merkle_internal2_kernel<<<(q+P3_MERKLE_THREADS-1)/P3_MERKLE_THREADS,P3_MERKLE_THREADS>>>(dlev.back(),l1,l2,q);
                dlev.push_back(l1); sz.push_back(half);
                dlev.push_back(l2); sz.push_back(q);
                cnt = q;
            } else {
                uint8_t* l1 = p; p += 32;
                p3_merkle_internal_kernel<<<1,P3_MERKLE_THREADS>>>(dlev.back(),l1,1);
                dlev.push_back(l1); sz.push_back(1); cnt = 1;
            }
        }
    }
    Hash root() const { Hash h; cudaMemcpy(h.data(), dlev.back(), 32, cudaMemcpyDeviceToHost); return h; }
    std::vector<Hash> path(size_t idx) const {
        std::vector<Hash> p;
        for (size_t d=0; d+1<dlev.size(); d++){ Hash h; cudaMemcpy(h.data(), dlev[d]+(size_t)(idx^1)*32, 32, cudaMemcpyDeviceToHost); p.push_back(h); idx>>=1; }
        return p;
    }
    void free_(){ if(!dlev.empty()) cudaFreeAsync(dlev[0],0);
        if(slab){ cudaFreeAsync(slab,0); slab=nullptr; }
        dlev.clear(); sz.clear(); }

    // batch-extract authentication paths for many leaf indices in ONE kernel + ONE D2H
    // (replaces O(#idx * logM) tiny latency-bound copies).
    std::vector<std::vector<Hash>> paths_batch(const std::vector<uint32_t>& idxs) const;
};

// thread per (index, level): copy the sibling hash into a contiguous output buffer
__global__ void p3_path_gather_kernel(uint8_t** lev, const uint32_t* idx, uint8_t* out, uint32_t n, uint32_t P) {
    uint32_t t = blockIdx.x*blockDim.x + threadIdx.x; if (t >= n*P) return;
    uint32_t i = t / P, l = t % P;
    uint32_t sib = (idx[i] >> l) ^ 1u;
    const uint8_t* src = lev[l] + (size_t)sib*32; uint8_t* dst = out + (size_t)t*32;
    #pragma unroll
    for (int k=0;k<32;k++) dst[k]=src[k];
}
inline std::vector<std::vector<Hash>> DeviceMerkle::paths_batch(const std::vector<uint32_t>& idxs) const {
    uint32_t L=(uint32_t)dlev.size(), n=(uint32_t)idxs.size();
    std::vector<std::vector<Hash>> res(n);
    if (L<=1) return res;
    uint32_t P=L-1;
    uint8_t** d_lev; cudaMallocAsync(&d_lev,(size_t)L*sizeof(uint8_t*),0);
    cudaMemcpy(d_lev, dlev.data(), (size_t)L*sizeof(uint8_t*), cudaMemcpyHostToDevice);
    uint32_t* d_idx; cudaMallocAsync(&d_idx,(size_t)n*4,0); cudaMemcpy(d_idx, idxs.data(), (size_t)n*4, cudaMemcpyHostToDevice);
    uint8_t* d_out; cudaMallocAsync(&d_out,(size_t)n*P*32,0);
    p3_path_gather_kernel<<<(n*P+255)/256,256>>>(d_lev,d_idx,d_out,n,P);
    std::vector<uint8_t> hout((size_t)n*P*32); cudaMemcpy(hout.data(), d_out, (size_t)n*P*32, cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_lev,0); cudaFreeAsync(d_idx,0); cudaFreeAsync(d_out,0);
    for (uint32_t i=0;i<n;i++){ res[i].resize(P); for(uint32_t l=0;l<P;l++) memcpy(res[i][l].data(), &hout[((size_t)i*P+l)*32], 32); }
    return res;
}

// ==================== streamed Merkle (bounded device memory) ====================
// A full DeviceMerkle over an M-leaf codeword holds ~64*M bytes of device hash
// levels; at llama-68m dims a single 2^29-leaf tree exceeds the 24 GB card.  The
// streamed variant reduces pow2-aligned contiguous CHUNKS to their subtree roots
// (bit-identical to the flat tree's internal nodes at the chunk boundary), keeps
// only the per-chunk subroots plus the HOST upper tree, and rebuilds individual
// chunks on demand for path extraction.  Roots and paths are byte-identical to
// DeviceMerkle, so verify_path and all verifiers are unchanged.
static const uint32_t P3_STREAM_LCH = 22;      // leaves per chunk (2^22 -> 128 MB hash buf)

struct StreamTree {
    size_t M = 0; uint32_t lch = 0;            // leaf count, log2 leaves-per-chunk
    std::vector<std::vector<Hash>> upper;      // upper[0] = chunk subroots (host)
    Hash root() const { return upper.back()[0]; }
    void build_upper(std::vector<Hash> sub) {
        upper.assign(1, std::move(sub));
        while (upper.back().size() > 1) {
            const auto& cur = upper.back();
            std::vector<Hash> nx(cur.size() / 2);
            for (size_t i = 0; i < nx.size(); i++) nx[i] = node_hash(cur[2*i], cur[2*i+1]);
            upper.push_back(std::move(nx));
        }
    }
};

// LeafFn(off, len, d_out): write the leaf hashes of codeword positions
// [off, off+len) into the device buffer d_out (len*32 bytes).
template <class LeafFn>
static inline StreamTree stream_tree_build(size_t M, LeafFn leaf_chunk) {
    StreamTree st; st.M = M;
    size_t CH = (size_t)1 << P3_STREAM_LCH; if (CH > M) CH = M;
    st.lch = 0; while (((size_t)1 << st.lch) < CH) st.lch++;
    const size_t nch = M / CH;
    uint8_t *ha, *hb;
    cudaMallocAsync(&ha, CH * 32, 0);
    cudaMallocAsync(&hb, (CH / 2 ? CH / 2 : 1) * 32, 0);
    std::vector<Hash> sub(nch);
    for (size_t c = 0; c < nch; c++) {
        leaf_chunk(c * CH, (uint32_t)CH, ha);
        uint8_t *cur = ha, *nxt = hb;
        for (size_t cnt = CH; cnt > 1; cnt /= 2) {
            uint32_t half = (uint32_t)(cnt / 2);
            p3_merkle_internal_kernel<<<(half + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS,
                                        P3_MERKLE_THREADS>>>(cur, nxt, half);
            uint8_t* t = cur; cur = nxt; nxt = t;
        }
        cudaMemcpy(sub[c].data(), cur, 32, cudaMemcpyDeviceToHost);
    }
    cudaFreeAsync(ha, 0); cudaFreeAsync(hb, 0);
    st.build_upper(std::move(sub));
    return st;
}

// full authentication paths (aligned with DeviceMerkle::paths_batch) for a list
// of leaf positions; rebuilds ONLY the chunks containing positions.
template <class LeafFn>
static inline std::vector<std::vector<Hash>> stream_tree_paths(const StreamTree& st, LeafFn leaf_chunk,
                                                               const std::vector<uint32_t>& pos) {
    const size_t CH = std::min<size_t>((size_t)1 << st.lch, st.M);
    std::vector<std::vector<Hash>> res(pos.size());
    std::map<uint32_t, std::vector<uint32_t>> byc;
    for (uint32_t i = 0; i < (uint32_t)pos.size(); i++) byc[(uint32_t)(pos[i] >> st.lch)].push_back(i);
    for (auto& cc : byc) {
        uint8_t* d0; cudaMallocAsync(&d0, CH * 32, 0);
        leaf_chunk((size_t)cc.first << st.lch, (uint32_t)CH, d0);
        DeviceMerkle mk; mk.build_leaves(d0, (uint32_t)CH);      // takes ownership of d0
        cudaDeviceSynchronize();
        std::vector<uint32_t> loc; loc.reserve(cc.second.size());
        for (auto i : cc.second) loc.push_back(pos[i] & (uint32_t)(CH - 1));
        auto lp = mk.paths_batch(loc);
        mk.free_();
        for (size_t q = 0; q < cc.second.size(); q++) {
            auto& r = res[cc.second[q]]; r = std::move(lp[q]);   // in-chunk levels
            uint32_t u = (uint32_t)(pos[cc.second[q]] >> st.lch);
            for (size_t l = 0; l + 1 < st.upper.size(); l++) { r.push_back(st.upper[l][u ^ 1]); u >>= 1; }
        }
    }
    return res;
}

// ONE-PASS variant for positions KNOWN at build time (round-0 subset opens):
// chunks containing positions keep their sublevels transiently and yield paths
// during the same pass that produces the subroots -- saves the second full
// hashing pass of stream_tree_build + stream_tree_paths.
template <class LeafFn>
static inline std::vector<std::vector<Hash>> stream_tree_paths_onepass(
        size_t M, LeafFn leaf_chunk, const std::vector<uint32_t>& pos) {
    StreamTree st; st.M = M;
    size_t CH = (size_t)1 << P3_STREAM_LCH; if (CH > M) CH = M;
    st.lch = 0; while (((size_t)1 << st.lch) < CH) st.lch++;
    const size_t nch = M / CH;
    std::map<uint32_t, std::vector<uint32_t>> byc;
    for (uint32_t i = 0; i < (uint32_t)pos.size(); i++) byc[(uint32_t)(pos[i] >> st.lch)].push_back(i);
    uint8_t *ha, *hb;
    cudaMallocAsync(&ha, CH * 32, 0);
    cudaMallocAsync(&hb, (CH / 2 ? CH / 2 : 1) * 32, 0);
    std::vector<Hash> sub(nch);
    std::vector<std::vector<Hash>> res(pos.size());
    for (size_t c = 0; c < nch; c++) {
        auto hit = byc.find((uint32_t)c);
        if (hit == byc.end()) {
            leaf_chunk(c * CH, (uint32_t)CH, ha);
            uint8_t *cur = ha, *nxt = hb;
            for (size_t cnt = CH; cnt > 1; cnt /= 2) {
                uint32_t half = (uint32_t)(cnt / 2);
                p3_merkle_internal_kernel<<<(half + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS,
                                            P3_MERKLE_THREADS>>>(cur, nxt, half);
                uint8_t* t = cur; cur = nxt; nxt = t;
            }
            cudaMemcpy(sub[c].data(), cur, 32, cudaMemcpyDeviceToHost);
        } else {
            uint8_t* d0; cudaMallocAsync(&d0, CH * 32, 0);
            leaf_chunk(c * CH, (uint32_t)CH, d0);
            DeviceMerkle mk; mk.build_leaves(d0, (uint32_t)CH);
            cudaDeviceSynchronize();
            sub[c] = mk.root();
            std::vector<uint32_t> loc; loc.reserve(hit->second.size());
            for (auto i : hit->second) loc.push_back(pos[i] & (uint32_t)(CH - 1));
            auto lp = mk.paths_batch(loc);
            mk.free_();
            for (size_t q = 0; q < hit->second.size(); q++) res[hit->second[q]] = std::move(lp[q]);
        }
    }
    cudaFreeAsync(ha, 0); cudaFreeAsync(hb, 0);
    st.build_upper(std::move(sub));
    for (uint32_t i = 0; i < (uint32_t)pos.size(); i++) {
        uint32_t u = (uint32_t)(pos[i] >> st.lch);
        for (size_t l = 0; l + 1 < st.upper.size(); l++) { res[i].push_back(st.upper[l][u ^ 1]); u >>= 1; }
    }
    return res;
}

struct RoundOpen { gl_t a, b; std::vector<Hash> pa, pb; };   // values at coset idx c and c+half
struct QueryProof { std::vector<RoundOpen> rounds; };
struct FriProof {
    uint32_t logN, R, Q;
    std::vector<Hash> roots;        // k = logN round roots (f_0..f_{k-1})
    std::vector<gl_t> final_word;   // f_k, length 2^R, sent in clear
    std::vector<QueryProof> queries;
};

// Fold value from a coset pair (a=f(x), b=f(-x)) at x:
//   E = (a+b)/2 [even part], O = (a-b)/(2x) [odd part]
//   coeff fold (FRI LDT):  E + beta*O           binds coefficients c'[k]=c[2k]+beta*c[2k+1]
//   MLE   fold (Basefold): (1-beta)*E + beta*O  binds MLE c'[k]=(1-beta)c[2k]+beta*c[2k+1]
static inline gl_t fold_pair(gl_t a, gl_t b, gl_t invx, gl_t beta, bool mle) {
    gl_t inv2 = gl_inv(2ULL);
    gl_t E = gl_mul(gl_add(a, b), inv2);
    gl_t O = gl_mul(gl_mul(gl_sub(a, b), inv2), invx);
    if (mle) return gl_add(gl_mul(gl_sub(1ULL, beta), E), gl_mul(beta, O));
    return gl_add(E, gl_mul(beta, O));
}

// honest fold of a round codeword (size M) with challenge beta; domain root w_M.
static inline std::vector<gl_t> fold(const std::vector<gl_t>& f, gl_t w_M, gl_t beta, bool mle = false) {
    uint32_t M = (uint32_t)f.size(), half = M / 2;
    gl_t winv = gl_inv(w_M);          // w_M^{-1}
    std::vector<gl_t> out(half);
    gl_t inv_x = 1ULL;                 // w_M^{-c}
    for (uint32_t c = 0; c < half; c++) {
        out[c] = fold_pair(f[c], f[c + half], inv_x, beta, mle);
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

// Shared query authentication: for each query coset index c0s[q], verify the
// revealed pairs are Merkle-consistent with `roots` and that folding with `betas`
// links each round to the next and finally to `final_word`.  Used by both the FRI
// LDT and the Basefold evaluation opening (which differ only in how betas arise).
static inline bool check_queries(const std::vector<Hash>& roots, const std::vector<gl_t>& final_word,
                                 const std::vector<gl_t>& betas, const std::vector<QueryProof>& queries,
                                 uint32_t logN, uint32_t R, const std::vector<uint32_t>& c0s,
                                 const char** why, bool mle = false) {
    auto fail = [&](const char* m){ if (why) *why = m; return false; };
    uint32_t logM0 = logN + R, M0 = 1u << logM0;
    for (uint32_t q = 0; q < queries.size(); q++) {
        const QueryProof& qp = queries[q];
        if (qp.rounds.size() != logN) return fail("round count");
        uint32_t p = c0s[q];
        for (uint32_t r = 0; r < logN; r++) {
            uint32_t Mr = M0 >> r, half = Mr / 2, c = p % half;
            const RoundOpen& ro = qp.rounds[r];
            if (!verify_path(leaf_hash(ro.a), c,        ro.pa, roots[r])) return fail("merkle a");
            if (!verify_path(leaf_hash(ro.b), c + half, ro.pb, roots[r])) return fail("merkle b");
            uint32_t logMr = logM0 - r;
            gl_t w = gl_root_of_unity(logMr);
            gl_t x = gl_pow(w, c), invx = gl_inv(x);
            gl_t folded = fold_pair(ro.a, ro.b, invx, betas[r], mle);
            if (r + 1 < logN) {
                uint32_t nhalf = (M0 >> (r + 1)) / 2;
                gl_t val = (c < nhalf) ? qp.rounds[r+1].a : qp.rounds[r+1].b;
                if (val != folded) return fail("fold link");
            } else {
                if (final_word[c] != folded) return fail("fold to final");
            }
            p = c;
        }
    }
    return true;
}

// ---------------- verifier ----------------
static inline bool verify(const FriProof& pf, const std::string& seed = "p3-fri", const char** why = nullptr) {
    auto fail = [&](const char* m){ if (why) *why = m; return false; };
    uint32_t logN = pf.logN, R = pf.R, M0 = 1u << (logN + R);
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

    for (size_t i = 1; i < pf.final_word.size(); i++)
        if (pf.final_word[i] != pf.final_word[0]) return fail("final not constant");

    std::vector<uint32_t> c0s(pf.Q);
    for (uint32_t q = 0; q < pf.Q; q++) c0s[q] = (uint32_t)idx_from(tr, M0 / 2);
    if (!check_queries(pf.roots, pf.final_word, betas, pf.queries, logN, R, c0s, why)) return false;
    if (why) *why = "ok";
    return true;
}

} // namespace p3fri
