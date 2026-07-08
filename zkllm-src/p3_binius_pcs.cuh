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

// internal Merkle levels above the leaves (host; leaf count is O(sqrt n))
static inline void bfpcs_tree_levels(BfPcsCommit& C, std::vector<uint8_t>&& leaves) {
    C.lvl.clear();
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

// Merkle over codeword columns (host reference; leaf = SHA-256 of the column bytes)
static inline void bfpcs_tree(BfPcsCommit& C) {
    std::vector<uint8_t> leaves((size_t)C.nc * 32);
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < (int64_t)C.nc; j++) {
        std::vector<bf16_t> col(C.n_rows);
        for (uint32_t i = 0; i < C.n_rows; i++) col[i] = C.cw[(size_t)i * C.nc + j];
        fs::sha256(col.data(), C.n_rows * sizeof(bf16_t), leaves.data() + (size_t)j * 32);
    }
    bfpcs_tree_levels(C, std::move(leaves));
}

// ---- GPU column hashing: one thread streams one codeword column through a
// chained rolling-schedule SHA-256 (word assembly from row-strided bf16 loads,
// coalesced across the warp) -- bitwise-identical to fs::sha256 of the column
// bytes, checked by the selftest against bfpcs_tree ----
__device__ __forceinline__ void bfpcs_sha_blk_st(const uint32_t win[16], uint32_t st[8]) {
    static const uint32_t Kc[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = win[i];
    uint32_t a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],hh=st[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t wi;
        if (i < 16) wi = w[i];
        else {
            uint32_t w15 = w[(i-15) & 15], w2 = w[(i-2) & 15];
            uint32_t s0 = p3_ror(w15,7) ^ p3_ror(w15,18) ^ (w15 >> 3);
            uint32_t s1 = p3_ror(w2,17) ^ p3_ror(w2,19) ^ (w2 >> 10);
            wi = w[i & 15] = w[i & 15] + s0 + w[(i-7) & 15] + s1;
        }
        uint32_t S1 = p3_ror(e,6) ^ p3_ror(e,11) ^ p3_ror(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + Kc[i] + wi;
        uint32_t S0 = p3_ror(a,2) ^ p3_ror(a,13) ^ p3_ror(a,22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=hh;
}
static __global__ void bfpcs_leaf_kernel(const bf16_t* cw, uint32_t n_rows, uint32_t nc,
                                         uint8_t* leaves) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nc) return;
    uint32_t st[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                      0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint64_t total = (uint64_t)n_rows * 2;                    // column bytes (LE bf16)
    uint32_t full = (uint32_t)(total / 64);
    for (uint32_t b = 0; b < full; b++) {
        uint32_t w[16];
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            uint32_t r0 = cw[(size_t)(32 * b + 2 * k)     * nc + j];
            uint32_t r1 = cw[(size_t)(32 * b + 2 * k + 1) * nc + j];
            w[k] = p3_bswap32(r0 | (r1 << 16));
        }
        bfpcs_sha_blk_st(w, st);
    }
    // tail: remaining rows + 0x80 pad + 8-byte BE bit length (1 or 2 blocks)
    uint8_t buf[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) buf[i] = 0;
    uint32_t rem = (uint32_t)(total - (uint64_t)full * 64);
    for (uint32_t t = 0; t < rem / 2; t++) {
        uint32_t v = cw[(size_t)(32 * full + t) * nc + j];
        buf[2 * t] = (uint8_t)v; buf[2 * t + 1] = (uint8_t)(v >> 8);
    }
    buf[rem] = 0x80;
    int nblk = (rem + 9 <= 64) ? 1 : 2;
    uint64_t bits = total * 8;
    for (int t = 0; t < 8; t++) buf[nblk * 64 - 1 - t] = (uint8_t)(bits >> (8 * t));
    for (int b = 0; b < nblk; b++) {
        uint32_t w[16];
        #pragma unroll
        for (int k = 0; k < 16; k++)
            w[k] = ((uint32_t)buf[64*b+4*k] << 24) | ((uint32_t)buf[64*b+4*k+1] << 16) |
                   ((uint32_t)buf[64*b+4*k+2] << 8) | (uint32_t)buf[64*b+4*k+3];
        bfpcs_sha_blk_st(w, st);
    }
    uint32_t* o = (uint32_t*)(leaves + (size_t)j * 32);
    #pragma unroll
    for (int k = 0; k < 8; k++) o[k] = p3_bswap32(st[k]);
}
// GPU tree: leaves hashed on device from the (post-NTT, still-resident)
// codeword; internal levels host (leaf count is O(sqrt n))
static inline void bfpcs_tree_gpu(BfPcsCommit& C, const bf16_t* d_cw) {
    uint8_t* d_leaves;
    cudaMalloc(&d_leaves, (size_t)C.nc * 32);
    bfpcs_leaf_kernel<<<(C.nc + 127) / 128, 128>>>(d_cw, C.n_rows, C.nc, d_leaves);
    std::vector<uint8_t> leaves((size_t)C.nc * 32);
    cudaMemcpy(leaves.data(), d_leaves, leaves.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_leaves);
    bfpcs_tree_levels(C, std::move(leaves));
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
    bfpcs_tree_gpu(C, d_rows);                 // leaves hashed from the resident codeword
    cudaFree(d_rows); cudaFree(dv.d_twid);
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

// GPU combine: block per unpacked output position j' = 16*jp + u, threads
// XOR-reduce coef[i] over rows with bit u of msg[i][jp] set.  XOR accumulation
// is order-independent, so the result is bitwise-identical to bfpcs_combine.
static __global__ void bfpcs_combine_kernel(const bf16_t* msg, uint32_t n_rows, uint32_t pc,
                                            const bf128_t* coef, bf128_t* out) {
    uint32_t jp = blockIdx.x >> 4, u = blockIdx.x & 15;
    bf128_t acc = bf128_zero();
    for (uint32_t i = threadIdx.x; i < n_rows; i += blockDim.x)
        acc = bf128_add(acc, bf128_smul1(coef[i], (uint32_t)(msg[(size_t)i * pc + jp] >> u)));
    __shared__ bf128_t sm[128];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = 64; s; s >>= 1) {
        if ((int)threadIdx.x < s) sm[threadIdx.x] = bf128_add(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    if (!threadIdx.x) out[blockIdx.x] = sm[0];
}
// device-resident open state: msg uploaded once, both combined rows on GPU
static inline void bfpcs_combine_gpu(const bf16_t* d_msg, uint32_t n_rows, uint32_t pc,
                                     const std::vector<bf128_t>& coef, bf128_t* d_coef,
                                     bf128_t* d_out, std::vector<bf128_t>& out) {
    size_t nu = (size_t)pc * 16;
    cudaMemcpy(d_coef, coef.data(), n_rows * sizeof(bf128_t), cudaMemcpyHostToDevice);
    bfpcs_combine_kernel<<<(uint32_t)nu, 128>>>(d_msg, n_rows, pc, d_coef, d_out);
    out.resize(nu);
    cudaMemcpy(out.data(), d_out, nu * sizeof(bf128_t), cudaMemcpyDeviceToHost);
}

static inline void bfpcs_open(const BfPcsCommit& C, const bf128_t* r, bf128_t v,
                              fs::Transcript& tr, BfPcsProof& pf) {
    const BfPcsParams& p = C.p;
    tr.absorb("bfpcs-point", r, p.l * sizeof(bf128_t));
    tr.absorb("bfpcs-value", &v, sizeof(v));
    bf16_t* d_msg; bf128_t *d_coef, *d_out;
    cudaMalloc(&d_msg, C.msg.size() * sizeof(bf16_t));
    cudaMalloc(&d_coef, (size_t)C.n_rows * sizeof(bf128_t));
    cudaMalloc(&d_out, ((size_t)C.pc * 16) * sizeof(bf128_t));
    cudaMemcpy(d_msg, C.msg.data(), C.msg.size() * sizeof(bf16_t), cudaMemcpyHostToDevice);
    std::vector<bf128_t> eqrow;
    bf_eq_table(r + p.lcol, p.lrow, eqrow);
    bfpcs_combine_gpu(d_msg, C.n_rows, C.pc, eqrow, d_coef, d_out, pf.t);
    tr.absorb("bfpcs-t", pf.t.data(), pf.t.size() * sizeof(bf128_t));
    std::vector<bf128_t> rho;
    bfpcs_rho(tr, C.n_rows, rho);
    bfpcs_combine_gpu(d_msg, C.n_rows, C.pc, rho, d_coef, d_out, pf.u);
    tr.absorb("bfpcs-u", pf.u.data(), pf.u.size() * sizeof(bf128_t));
    cudaFree(d_msg); cudaFree(d_coef); cudaFree(d_out);
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

// ---- multi-point opening: M evaluation claims on ONE commitment share the
// proximity row, the spot-column draw, the column data and the Merkle paths
// (the expensive O(sqrt n) part); only the per-point combined eval rows are
// extra.  Soundness is the same Ligero argument: u ties the matrix to a
// nearby codeword once, and each t_m is checked against the SAME opened
// columns.  Transcript schedule: (point,value) x M, t x M, rho, u, queries. ----
struct BfPcsProofM {
    std::vector<std::vector<bf128_t>> t;  // M eval rows, each length 2^lcol
    std::vector<bf128_t> u;               // shared proximity row
    std::vector<bf16_t> cols;             // Q * n_rows shared column data
    std::vector<uint8_t> paths;           // Q * depth * 32 shared sibling hashes
    size_t bytes() const {
        size_t s = u.size() * sizeof(bf128_t) + cols.size() * sizeof(bf16_t) + paths.size();
        for (auto& tm : t) s += tm.size() * sizeof(bf128_t);
        return s;
    }
};
static inline void bfpcs_open_multi(const BfPcsCommit& C,
                                    const std::vector<const bf128_t*>& rs,
                                    const std::vector<bf128_t>& vs,
                                    fs::Transcript& tr, BfPcsProofM& pf) {
    const BfPcsParams& p = C.p;
    const int M = (int)rs.size();
    for (int m = 0; m < M; m++) {
        tr.absorb("bfpcs-point", rs[m], p.l * sizeof(bf128_t));
        tr.absorb("bfpcs-value", &vs[m], sizeof(bf128_t));
    }
    bf16_t* d_msg; bf128_t *d_coef, *d_out;
    cudaMalloc(&d_msg, C.msg.size() * sizeof(bf16_t));
    cudaMalloc(&d_coef, (size_t)C.n_rows * sizeof(bf128_t));
    cudaMalloc(&d_out, ((size_t)C.pc * 16) * sizeof(bf128_t));
    cudaMemcpy(d_msg, C.msg.data(), C.msg.size() * sizeof(bf16_t), cudaMemcpyHostToDevice);
    pf.t.resize(M);
    for (int m = 0; m < M; m++) {
        std::vector<bf128_t> eqrow;
        bf_eq_table(rs[m] + p.lcol, p.lrow, eqrow);
        bfpcs_combine_gpu(d_msg, C.n_rows, C.pc, eqrow, d_coef, d_out, pf.t[m]);
        tr.absorb("bfpcs-t", pf.t[m].data(), pf.t[m].size() * sizeof(bf128_t));
    }
    std::vector<bf128_t> rho;
    bfpcs_rho(tr, C.n_rows, rho);
    bfpcs_combine_gpu(d_msg, C.n_rows, C.pc, rho, d_coef, d_out, pf.u);
    tr.absorb("bfpcs-u", pf.u.data(), pf.u.size() * sizeof(bf128_t));
    cudaFree(d_msg); cudaFree(d_coef); cudaFree(d_out);
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
static inline bool bfpcs_verify_multi(const BfPcsParams& p, const uint8_t root[32],
                                      const std::vector<const bf128_t*>& rs,
                                      const std::vector<bf128_t>& vs,
                                      fs::Transcript& tr, const BfPcsProofM& pf) {
    const int M = (int)rs.size();
    uint32_t n_rows = 1u << p.lrow, pc = 1u << (p.lcol - 4), nc = pc << BFPCS_RATE_LOG;
    size_t nu = (size_t)1 << p.lcol;
    if ((int)pf.t.size() != M || pf.u.size() != nu) return false;
    for (int m = 0; m < M; m++) if (pf.t[m].size() != nu) return false;
    if (pf.cols.size() != (size_t)p.Q * n_rows) return false;
    for (int m = 0; m < M; m++) {
        tr.absorb("bfpcs-point", rs[m], p.l * sizeof(bf128_t));
        tr.absorb("bfpcs-value", &vs[m], sizeof(bf128_t));
    }
    for (int m = 0; m < M; m++)
        tr.absorb("bfpcs-t", pf.t[m].data(), pf.t[m].size() * sizeof(bf128_t));
    std::vector<bf128_t> rho;
    bfpcs_rho(tr, n_rows, rho);
    tr.absorb("bfpcs-u", pf.u.data(), pf.u.size() * sizeof(bf128_t));
    std::vector<uint32_t> q;
    bfpcs_queries(tr, nc, p.Q, q);
    int depth = 0; while ((1u << depth) < nc) depth++;
    if (pf.paths.size() != (size_t)p.Q * depth * 32) return false;
    BfNtt nt; bfntt_init(nt, p.lcol - 4 + BFPCS_RATE_LOG);
    std::vector<std::vector<bf128_t>> W(M);
    std::vector<bf128_t> U;
    for (int m = 0; m < M; m++) {
        bf_pack128(pf.t[m].data(), nu, W[m]);
        W[m].resize(nc, bf128_zero()); bfntt_fwd_host128(nt, W[m].data());
    }
    bf_pack128(pf.u.data(), nu, U); U.resize(nc, bf128_zero()); bfntt_fwd_host128(nt, U.data());
    std::vector<std::vector<bf128_t>> eqrow(M);
    for (int m = 0; m < M; m++) bf_eq_table(rs[m] + p.lcol, p.lrow, eqrow[m]);
    for (int k = 0; k < p.Q; k++) {
        uint32_t j = q[k];
        const bf16_t* col = pf.cols.data() + (size_t)k * n_rows;
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
        bf128_t sr = bf128_zero();
        for (uint32_t i = 0; i < n_rows; i++)
            sr = bf128_add(sr, bf128_smul16(rho[i], col[i]));
        if (!bf128_eq(sr, U[j])) return false;
        for (int m = 0; m < M; m++) {
            bf128_t se = bf128_zero();
            for (uint32_t i = 0; i < n_rows; i++)
                se = bf128_add(se, bf128_smul16(eqrow[m][i], col[i]));
            if (!bf128_eq(se, W[m][j])) return false;
        }
    }
    for (int m = 0; m < M; m++) {
        std::vector<bf128_t> eqcol;
        bf_eq_table(rs[m], p.lcol, eqcol);
        bf128_t acc = bf128_zero();
        for (size_t jp = 0; jp < nu; jp++)
            acc = bf128_add(acc, bf128_mul(eqcol[jp], pf.t[m][jp]));
        if (!bf128_eq(acc, vs[m])) return false;
    }
    return true;
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
