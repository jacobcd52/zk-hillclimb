// Binius substrate step 3 (design doc section 21.3): small-field polynomial
// commitment over the binary tower -- commit BITS at their true width.
//
// Commit: a multilinear over F_2 with 2^l coefficients (witness bits, index
// bit t = variable t) is arranged as a 2^lrow x 2^lcol bit matrix (row index
// = high variables), each row packed 16 bits per GF(2^16) symbol (the F_2
// bit-basis of T_16, so "unpack" is literally reading bits -- a committed
// column can only ever contain F_2 values: booleanity is STRUCTURAL, no
// booleanity constraints needed).  Rows are RS-encoded at rate 1/4 by the
// additive NTT (GPU) and a SHA-256 Merkle tree is built over codeword
// COLUMNS.  Committed data = codeword + tree; both scale with the TRUE bit
// content of the witness (vs 64-bit Goldilocks elements per bit today).
//
// Open at r in T_128^l (Ligero/Brakedown-style; T_128-combinations of packed
// T_16 data commute with the encoding because Enc is T_16-linear and T_128 is
// a free T_16-module -- checked bitwise by the selftest):
//   t[j']  = sum_i eq(r_hi,i) * M[i][j']   (the eval row, over T_128)
//   u[j']  = sum_i rho_i      * M[i][j']   (proximity row, rho from transcript)
//   Q spot columns from the transcript: Merkle paths + column data; verifier
//   re-encodes pack(t), pack(u) over T_128 and checks both combinations at
//   every spot column, then checks v == sum_j' eq(r_lo,j') * t[j'].
// Proof size O(sqrt(n)); FRI-style polylog opening is a later migration step
// (section 21 handoff) -- commit format and the committed-data win are
// unchanged by that swap.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include "fs_transcript.hpp"
#include "p3_binius_ntt.cuh"
#include "p3_merkle.cuh"          // host-callable p3_sha256_compress64

#define BFPCS_RATE_LOG 2          // rate 1/4

// ---- eq tables and multilinear helpers over T_128 ----
static inline void bf_eq_table(const bf128_t* r, int k, std::vector<bf128_t>& out) {
    out.assign((size_t)1 << k, bf128_zero());
    out[0] = bf128_one();
    for (int t = 0; t < k; t++) {
        size_t half = (size_t)1 << t;
        for (size_t x = 0; x < half; x++) {
            bf128_t e = out[x];
            bf128_t hi = bf128_mul(e, r[t]);              // x_t = 1 branch
            out[x | half] = hi;
            out[x] = bf128_add(e, hi);                    // (1+r_t)*e
        }
    }
}
// reference evaluation of the F_2-coefficient multilinear at r (test/verifier aid)
static inline bf128_t bf_ml_eval_bits(const uint8_t* bits, int l, const bf128_t* r) {
    std::vector<bf128_t> eq;
    bf_eq_table(r, l, eq);
    bf128_t acc = bf128_zero();
    for (size_t x = 0; x < ((size_t)1 << l); x++)
        if (bits[x] & 1) acc = bf128_add(acc, eq[x]);
    return acc;
}
static inline bf128_t bf_chal128(fs::Transcript& tr) {
    uint8_t b[32]; tr.challenge_bytes(b);
    bf128_t r; memcpy(&r.lo, b, 8); memcpy(&r.hi, b + 8, 8);
    return r;
}

// pack 16 unpacked bit-positions into one T_16 symbol; over T_128 rows the
// same packing is smul16 by the bit-basis element 1<<u
static inline void bf_pack_bits(const uint8_t* bits, size_t nbits, bf16_t* out) {
    for (size_t j = 0; j < nbits / 16; j++) {
        uint32_t s = 0;
        for (int u = 0; u < 16; u++) s |= (uint32_t)(bits[16 * j + u] & 1) << u;
        out[j] = (bf16_t)s;
    }
}
static inline void bf_pack128(const bf128_t* t, size_t nu, std::vector<bf128_t>& w) {
    w.assign(nu / 16, bf128_zero());
    for (size_t j = 0; j < nu / 16; j++)
        for (int u = 0; u < 16; u++)
            w[j] = bf128_add(w[j], bf128_smul16(t[16 * j + u], 1u << u));
}

// forward NTT of a T_128 row (limb-wise T_16 butterflies; used by the verifier)
static inline void bfntt_fwd_host128(const BfNtt& nt, bf128_t* d) {
    int m = nt.m;
    for (int s = m - 1; s >= 0; s--) {
        const bf16_t* tw = nt.twid.data() + nt.off[s];
        uint32_t half = 1u << s;
        for (uint32_t h = 0; h < (1u << (m - 1 - s)); h++) {
            uint32_t t = tw[h];
            uint64_t base = (uint64_t)h << (s + 1);
            for (uint32_t j = 0; j < half; j++) {
                bf128_t lo = d[base + j], hi = d[base + half + j];
                lo = bf128_add(lo, bf128_smul16(hi, t));
                d[base + j] = lo;
                d[base + half + j] = bf128_add(lo, hi);
            }
        }
    }
}

struct BfPcsParams {
    int l = 0;                    // log2 unpacked coefficient (bit) count
    int lrow = 0, lcol = 0;       // l = lrow + lcol, lcol >= 4
    int Q = 100;                  // spot-check columns
};
struct BfPcsProof {
    std::vector<bf128_t> t, u;    // eval + proximity rows, length 2^lcol
    std::vector<bf16_t> cols;     // Q * n_rows column data
    std::vector<uint8_t> paths;   // Q * depth * 32 sibling hashes
    size_t bytes() const {
        return (t.size() + u.size()) * sizeof(bf128_t) +
               cols.size() * sizeof(bf16_t) + paths.size();
    }
};
struct BfPcsCommit {
    BfPcsParams p;
    uint32_t n_rows = 0, pc = 0, nc = 0;          // rows, packed cols, code cols
    std::vector<bf16_t> msg;                      // n_rows * pc packed message
    std::vector<bf16_t> cw;                       // n_rows * nc codeword
    std::vector<std::vector<uint8_t>> lvl;        // Merkle levels, leaves first
    uint8_t root[32];
    size_t committed_bytes = 0;                   // codeword + tree
    double commit_ms = 0;
};

// Merkle over codeword columns (host; leaf = SHA-256 of the column bytes)
static inline void bfpcs_tree(BfPcsCommit& C) {
    C.lvl.clear();
    std::vector<uint8_t> leaves((size_t)C.nc * 32);
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < (int64_t)C.nc; j++) {
        std::vector<bf16_t> col(C.n_rows);
        for (uint32_t i = 0; i < C.n_rows; i++) col[i] = C.cw[(size_t)i * C.nc + j];
        fs::sha256(col.data(), C.n_rows * sizeof(bf16_t), leaves.data() + (size_t)j * 32);
    }
    C.lvl.push_back(std::move(leaves));
    while (C.lvl.back().size() > 32) {
        const std::vector<uint8_t>& prev = C.lvl.back();
        std::vector<uint8_t> next(prev.size() / 2);
        for (size_t i = 0; i < next.size() / 32; i++)
            p3_sha256_compress64(prev.data() + i * 64, next.data() + i * 32);
        C.lvl.push_back(std::move(next));
    }
    memcpy(C.root, C.lvl.back().data(), 32);
    C.committed_bytes = (size_t)C.n_rows * C.nc * sizeof(bf16_t);
    for (auto& L : C.lvl) C.committed_bytes += L.size();
}

// commit witness bits (one byte per bit, length 2^l) under params p
static inline void bfpcs_commit(const BfPcsParams& p, const uint8_t* bits,
                                fs::Transcript& tr, BfPcsCommit& C) {
    C.p = p;
    C.n_rows = 1u << p.lrow;
    C.pc = 1u << (p.lcol - 4);
    C.nc = C.pc << BFPCS_RATE_LOG;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // pack rows and RS-encode on GPU (rows zero-padded to code length)
    C.msg.assign((size_t)C.n_rows * C.pc, 0);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)C.n_rows; i++)
        bf_pack_bits(bits + ((size_t)i << p.lcol), (size_t)1 << p.lcol,
                     C.msg.data() + (size_t)i * C.pc);
    cudaEventRecord(e0);
    C.cw.assign((size_t)C.n_rows * C.nc, 0);
    for (uint32_t i = 0; i < C.n_rows; i++)
        memcpy(C.cw.data() + (size_t)i * C.nc, C.msg.data() + (size_t)i * C.pc,
               C.pc * sizeof(bf16_t));
    BfNtt nt; bfntt_init(nt, p.lcol - 4 + BFPCS_RATE_LOG);
    BfNttDev dv; bfntt_to_device(nt, dv);
    bf16_t* d_rows;
    cudaMalloc(&d_rows, C.cw.size() * sizeof(bf16_t));
    cudaMemcpy(d_rows, C.cw.data(), C.cw.size() * sizeof(bf16_t), cudaMemcpyHostToDevice);
    bfntt_fwd_gpu(dv, d_rows, C.n_rows);
    cudaMemcpy(C.cw.data(), d_rows, C.cw.size() * sizeof(bf16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_rows); cudaFree(dv.d_twid);
    bfpcs_tree(C);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    C.commit_ms = ms;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    tr.absorb("bfpcs-root", C.root, 32);
}

// shared challenge schedule: rho after t, queries after u
static inline void bfpcs_rho(fs::Transcript& tr, uint32_t n_rows, std::vector<bf128_t>& rho) {
    rho.resize(n_rows);
    for (uint32_t i = 0; i < n_rows; i++) rho[i] = bf_chal128(tr);
}
static inline void bfpcs_queries(fs::Transcript& tr, uint32_t nc, int Q,
                                 std::vector<uint32_t>& q) {
    q.clear();
    while ((int)q.size() < Q) {
        uint8_t b[32]; tr.challenge_bytes(b);
        for (int k = 0; k < 8 && (int)q.size() < Q; k++) {
            uint32_t v; memcpy(&v, b + 4 * k, 4);
            q.push_back(v & (nc - 1));                 // nc is a power of two
        }
    }
}

// combined row over T_128: out[j'] = sum_i coef[i] * bit(i,j')
static inline void bfpcs_combine(const BfPcsCommit& C, const std::vector<bf128_t>& coef,
                                 std::vector<bf128_t>& out) {
    size_t nu = (size_t)1 << C.p.lcol;
    out.assign(nu, bf128_zero());
    #pragma omp parallel for schedule(static)
    for (int64_t jp = 0; jp < (int64_t)C.pc; jp++)
        for (uint32_t i = 0; i < C.n_rows; i++) {
            uint32_t m = C.msg[(size_t)i * C.pc + jp];
            while (m) {
                int u = __builtin_ctz(m); m &= m - 1;
                out[16 * jp + u] = bf128_add(out[16 * jp + u], coef[i]);
            }
        }
}

static inline void bfpcs_open(const BfPcsCommit& C, const bf128_t* r, bf128_t v,
                              fs::Transcript& tr, BfPcsProof& pf) {
    const BfPcsParams& p = C.p;
    tr.absorb("bfpcs-point", r, p.l * sizeof(bf128_t));
    tr.absorb("bfpcs-value", &v, sizeof(v));
    std::vector<bf128_t> eqrow;
    bf_eq_table(r + p.lcol, p.lrow, eqrow);
    bfpcs_combine(C, eqrow, pf.t);
    tr.absorb("bfpcs-t", pf.t.data(), pf.t.size() * sizeof(bf128_t));
    std::vector<bf128_t> rho;
    bfpcs_rho(tr, C.n_rows, rho);
    bfpcs_combine(C, rho, pf.u);
    tr.absorb("bfpcs-u", pf.u.data(), pf.u.size() * sizeof(bf128_t));
    std::vector<uint32_t> q;
    bfpcs_queries(tr, C.nc, p.Q, q);
    int depth = 0; while ((1u << depth) < C.nc) depth++;
    pf.cols.assign((size_t)p.Q * C.n_rows, 0);
    pf.paths.assign((size_t)p.Q * depth * 32, 0);
    for (int k = 0; k < p.Q; k++) {
        uint32_t j = q[k];
        for (uint32_t i = 0; i < C.n_rows; i++)
            pf.cols[(size_t)k * C.n_rows + i] = C.cw[(size_t)i * C.nc + j];
        uint32_t idx = j;
        for (int dl = 0; dl < depth; dl++) {
            memcpy(pf.paths.data() + ((size_t)k * depth + dl) * 32,
                   C.lvl[dl].data() + (size_t)(idx ^ 1u) * 32, 32);
            idx >>= 1;
        }
    }
}

static inline bool bfpcs_verify(const BfPcsParams& p, const uint8_t root[32],
                                const bf128_t* r, bf128_t v,
                                fs::Transcript& tr, const BfPcsProof& pf) {
    uint32_t n_rows = 1u << p.lrow, pc = 1u << (p.lcol - 4), nc = pc << BFPCS_RATE_LOG;
    size_t nu = (size_t)1 << p.lcol;
    if (pf.t.size() != nu || pf.u.size() != nu) return false;
    if (pf.cols.size() != (size_t)p.Q * n_rows) return false;
    tr.absorb("bfpcs-point", r, p.l * sizeof(bf128_t));
    tr.absorb("bfpcs-value", &v, sizeof(v));
    tr.absorb("bfpcs-t", pf.t.data(), pf.t.size() * sizeof(bf128_t));
    std::vector<bf128_t> rho;
    bfpcs_rho(tr, n_rows, rho);
    tr.absorb("bfpcs-u", pf.u.data(), pf.u.size() * sizeof(bf128_t));
    std::vector<uint32_t> q;
    bfpcs_queries(tr, nc, p.Q, q);
    int depth = 0; while ((1u << depth) < nc) depth++;
    if (pf.paths.size() != (size_t)p.Q * depth * 32) return false;
    // re-encode the claimed combined rows over T_128
    BfNtt nt; bfntt_init(nt, p.lcol - 4 + BFPCS_RATE_LOG);
    std::vector<bf128_t> W, U;
    bf_pack128(pf.t.data(), nu, W); W.resize(nc, bf128_zero()); bfntt_fwd_host128(nt, W.data());
    bf_pack128(pf.u.data(), nu, U); U.resize(nc, bf128_zero()); bfntt_fwd_host128(nt, U.data());
    std::vector<bf128_t> eqrow, eqcol;
    bf_eq_table(r + p.lcol, p.lrow, eqrow);
    bf_eq_table(r, p.lcol, eqcol);
    for (int k = 0; k < p.Q; k++) {
        uint32_t j = q[k];
        const bf16_t* col = pf.cols.data() + (size_t)k * n_rows;
        // Merkle path
        uint8_t h[32];
        fs::sha256(col, n_rows * sizeof(bf16_t), h);
        uint32_t idx = j;
        for (int dl = 0; dl < depth; dl++) {
            uint8_t cat[64];
            const uint8_t* sib = pf.paths.data() + ((size_t)k * depth + dl) * 32;
            if (idx & 1) { memcpy(cat, sib, 32); memcpy(cat + 32, h, 32); }
            else         { memcpy(cat, h, 32);   memcpy(cat + 32, sib, 32); }
            p3_sha256_compress64(cat, h);
            idx >>= 1;
        }
        if (memcmp(h, root, 32)) return false;
        // consistency of both combined rows with the opened column
        bf128_t se = bf128_zero(), sr = bf128_zero();
        for (uint32_t i = 0; i < n_rows; i++) {
            se = bf128_add(se, bf128_smul16(eqrow[i], col[i]));
            sr = bf128_add(sr, bf128_smul16(rho[i], col[i]));
        }
        if (!bf128_eq(se, W[j]) || !bf128_eq(sr, U[j])) return false;
    }
    // the evaluation claim itself
    bf128_t acc = bf128_zero();
    for (size_t jp = 0; jp < nu; jp++)
        acc = bf128_add(acc, bf128_mul(eqcol[jp], pf.t[jp]));
    return bf128_eq(acc, v);
}
