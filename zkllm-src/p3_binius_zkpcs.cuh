// p3_binius_zkpcs.cuh -- ZERO-KNOWLEDGE tower PCS (design doc section 21.13).
// Makes the Ligero/Brakedown-style bfpcs opening (section 21.3) HIDING: an
// opened proof reveals NOTHING about the committed witness bits beyond the
// public evaluation value v.  Three mechanisms, mirroring the Goldilocks
// p3_zkc.cuh core (mechanisms 1/3/4) but specialised to the packed-T_16 tower
// Ligero code:
//
//  (A) MASK-COLUMN AUGMENTATION.  The witness multilinear over l = lrow+lcol
//      bit-variables is committed over lcol_a = lcol + e augmented column
//      variables; the extra e coordinates (ex) index MASK columns filled with
//      fresh uniform bits.  The evaluation point is zero-extended in the ex
//      coordinates -- eq(0^e, ex) selects ONLY the real slice, so the augmented
//      eval equals the real v EXACTLY (soundness/correctness untouched).  Every
//      opened codeword column is a codeword symbol of [real | mask]: the
//      additive-NTT code mixes every message symbol into every codeword
//      position, so each opened symbol is one-time-padded by the mask and is
//      uniform.  e is sized so the mask packed-symbol dimension (2^e-1)*pc
//      exceeds the Q opened columns (Ligero's random-column count).
//
//  (B) MASKING-POLYNOMIAL BLIND (Brakedown-ZK).  The opening sends two full
//      combined rows -- the eval row t = sum_i eq(r_hi,i) M[i] and the proximity
//      row u = sum_i rho_i M[i] -- each a linear functional of the witness in
//      EVERY column direction, so each leaks.  The row block is bumped by one
//      (lrow_a = lrow+1); the HIGH HALF is a fresh uniform matrix g (a masking
//      polynomial of the witness's shape):
//        * proximity:  u' = sum_{ALL rows} rho_i M_aug[i].  rho spans the g
//          rows, so u' is one-time-padded by g and is uniform (its value is
//          unconstrained -- u is a proximity test only).
//        * eval:  prover sends y_g = <eqcol, t_g> (the masking poly's eval),
//          THEN a challenge lambda is drawn, THEN the combined row
//          tau = t_M + lambda*t_g is sent.  tau is uniform (t_g uniform), so it
//          reveals nothing; the verifier recovers v via <eqcol, tau> = v +
//          lambda*y_g.  Because lambda is drawn AFTER y_g is fixed, the single
//          random equation (v_true - v) + lambda*(y_g_true - y_g) = 0 forces
//          BOTH the public v and the sent y_g to their true values -- a cheating
//          prover cannot trade one against the other (this is what makes the
//          blind SOUND, unlike a free additive scalar).  y_g is a functional of
//          the uniform g, hence uniform, hence leak-free.
//      Consistency at each spot column ties tau and u' to the committed real
//      rows AND the committed g rows, so the Ligero proximity test catches any
//      inconsistent combined row.
//
//  (C) SALTED LEAVES.  Merkle leaf_j = SHA256(column_bytes || salt_j),
//      salt_j = SHA256(sseed_le8 || j_le8).  An opened column discloses
//      (uniform column data, salt, path); salting stops a low-entropy column
//      from being recognised by its leaf hash and blinds sibling-hash leakage.
//      sseed is fixed before the root, sent in the proof, and re-derived by the
//      verifier; the root binds every salt.
//
// The global bfz::G context toggles the master switch and the three
// negative-control flags (mask_on/blind_on/salt_on) the hiding battery flips to
// prove each mechanism is load-bearing.  This header is a HOST reference: it
// reuses bfpcs's tower helpers (bf_eq_table, bf_pack_bits, bfntt_fwd_host/128)
// and adds no CUDA kernels -- the committed-data / speed win is already measured
// on the non-zk path (section 21.11); zk cost is the (2^e)*2 blow-up plus the
// two combined rows, quantified by the selftest.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include "fs_transcript.hpp"
#include "p3_binius_ntt.cuh"
#include "p3_binius_pcs.cuh"
#include "p3_merkle.cuh"

namespace bfz {

struct ZkCtx {
    bool on = true;         // master zk switch
    bool mask_on = true;    // (A) mask-column augmentation values (0 under control)
    bool blind_on = true;   // (B) blind-row values (0 under control)
    bool salt_on = true;    // (C) salted Merkle leaves
    uint32_t Q = 100;       // spot-check columns (mask dimension sized to this)
    uint64_t seed = 0xB1A17;// master randomness for masks/blinds/salts
    uint64_t ctr = 0;       // per-draw counter
};
static ZkCtx G;

// ---- randomness: splitmix64 stream (deterministic per seed for the battery) ----
static inline uint64_t bfz_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline uint64_t bfz_next_seed() {
    return bfz_mix(G.seed += 0x9E3779B97F4A7C15ULL * (++G.ctr));
}
// fill n bits (one byte each, value 0/1) from seed s
static inline void bfz_fill_bits(uint64_t s, uint8_t* out, size_t n) {
    size_t i = 0;
    while (i < n) {
        uint64_t w = bfz_mix(s + 0x9E3779B97F4A7C15ULL * (i + 1));
        for (int k = 0; k < 64 && i < n; k++, i++) out[i] = (uint8_t)((w >> k) & 1);
    }
}

// ---- augmentation policy ----
// mask packed-symbol dimension (2^e - 1)*pc must exceed the Q opened columns so
// the Q codeword-position functionals are jointly independent over the mask
// (Ligero's "add ~Q random columns" made multiplicative on the packed width).
static inline uint32_t e_of(int lcol) {
    if (!G.on) return 0;
    uint32_t pc = 1u << (lcol - 4);
    uint32_t need = G.Q + 16;                       // + headroom
    uint32_t e = 1;
    while (((1u << e) - 1) * pc < need) e++;
    return e;
}

// ---- salted leaves ----
static inline void bfz_salt(uint64_t sseed, uint64_t j, uint8_t out[32]) {
    uint8_t b[16];
    for (int k = 0; k < 8; k++) { b[k] = (uint8_t)(sseed >> (8*k)); b[8+k] = (uint8_t)(j >> (8*k)); }
    fs::sha256(b, 16, out);
}
// leaf = SHA256(column_bytes || salt) if salt_on, else SHA256(column_bytes)
static inline void bfz_leaf(const bf16_t* col, uint32_t n_rows, uint64_t sseed, uint64_t j,
                            uint8_t out[32]) {
    if (!G.salt_on) { fs::sha256(col, n_rows * sizeof(bf16_t), out); return; }
    std::vector<uint8_t> buf(n_rows * sizeof(bf16_t) + 32);
    memcpy(buf.data(), col, n_rows * sizeof(bf16_t));
    bfz_salt(sseed, j, buf.data() + n_rows * sizeof(bf16_t));
    fs::sha256(buf.data(), buf.size(), out);
}

// ---- ZK commitment ----
struct ZkParams {
    int l = 0, lrow = 0, lcol = 0;   // REAL dims (as in BfPcsParams)
    int Q = 100;
    // derived (filled by commit):
    int e = 0;                       // mask-column augmentation
    int lcol_a = 0, lrow_a = 0;      // augmented dims
};
struct ZkCommit {
    ZkParams p;
    uint32_t n_rows = 0, pc = 0, nc = 0;         // augmented rows / packed cols / code cols
    uint64_t sseed = 0;
    std::vector<bf16_t> msg;                     // n_rows * pc  (packed augmented message)
    std::vector<bf16_t> cw;                      // n_rows * nc  (codeword)
    std::vector<std::vector<uint8_t>> lvl;       // Merkle levels (leaves first)
    uint8_t root[32] = {};
    size_t committed_bytes = 0;
    // row roles: [0, n_real) real; row n_real = w_u; n_real+1 = w_t; rest random
    uint32_t n_real = 0, row_wu = 0, row_wt = 0;
};
struct ZkProof {
    std::vector<bf128_t> t, u;       // combined eval row tau + proximity row u' (length 2^lcol_a)
    bf128_t yg = bf128_zero();       // masking-polynomial eval <eqcol, t_g> (sent before lambda)
    uint64_t sseed = 0;              // salt seed (verifier re-derives salts)
    std::vector<bf16_t> cols;        // Q * n_rows column data
    std::vector<uint8_t> paths;      // Q * depth * 32 sibling hashes
    size_t bytes() const {
        return (t.size() + u.size() + 1) * sizeof(bf128_t) +
               cols.size() * sizeof(bf16_t) + paths.size() + sizeof(sseed);
    }
};

// host RS-encode one packed message row (pc symbols) to nc codeword symbols
static inline void bfz_encode_row(const BfNtt& nt, const bf16_t* msg, uint32_t pc, uint32_t nc,
                                  bf16_t* cw) {
    memset(cw, 0, nc * sizeof(bf16_t));
    memcpy(cw, msg, pc * sizeof(bf16_t));
    bfntt_fwd_host(nt, cw);
}

static inline void bfz_build_tree(ZkCommit& C) {
    std::vector<uint8_t> leaves((size_t)C.nc * 32);
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < (int64_t)C.nc; j++) {
        std::vector<bf16_t> col(C.n_rows);
        for (uint32_t i = 0; i < C.n_rows; i++) col[i] = C.cw[(size_t)i * C.nc + j];
        bfz_leaf(col.data(), C.n_rows, C.sseed, (uint64_t)j, leaves.data() + (size_t)j * 32);
    }
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

// commit real witness bits (length 2^l) with masks + blind rows + salted tree
static inline void bfz_commit(const ZkParams& p_in, const uint8_t* bits,
                              fs::Transcript& tr, ZkCommit& C) {
    ZkParams p = p_in;
    p.e = e_of(p.lcol);
    p.lcol_a = p.lcol + p.e;
    p.lrow_a = p.lrow + (G.on ? 1 : 0);              // one extra row block for blind rows
    C.p = p;
    uint32_t n_real = 1u << p.lrow;
    C.n_rows = 1u << p.lrow_a;
    C.n_real = n_real;
    C.pc = 1u << (p.lcol_a - 4);
    C.nc = C.pc << BFPCS_RATE_LOG;
    C.row_wu = n_real;                               // first high row
    C.row_wt = n_real + 1;
    C.sseed = G.salt_on ? bfz_next_seed() : 0;

    uint32_t real_cols = 1u << p.lcol;
    uint32_t aug_cols  = 1u << p.lcol_a;
    // ---- assemble the augmented bit matrix ----
    std::vector<uint8_t> M((size_t)C.n_rows * aug_cols, 0);
    // real rows: [ real bits | mask bits ]
    for (uint32_t i = 0; i < n_real; i++) {
        uint8_t* row = M.data() + (size_t)i * aug_cols;
        memcpy(row, bits + (size_t)i * real_cols, real_cols);
        if (G.on && G.mask_on && aug_cols > real_cols)
            bfz_fill_bits(bfz_next_seed(), row + real_cols, aug_cols - real_cols);
    }
    // blind / high rows (only present when zk on)
    if (G.on) {
        for (uint32_t i = n_real; i < C.n_rows; i++) {
            uint8_t* row = M.data() + (size_t)i * aug_cols;
            if (G.blind_on) bfz_fill_bits(bfz_next_seed(), row, aug_cols);
            // else: zero rows (negative control -> u',t' unblinded)
        }
    }
    // ---- pack + encode every row ----
    BfNtt nt; bfntt_init(nt, p.lcol_a - 4 + BFPCS_RATE_LOG);
    C.msg.assign((size_t)C.n_rows * C.pc, 0);
    C.cw.assign((size_t)C.n_rows * C.nc, 0);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)C.n_rows; i++) {
        bf_pack_bits(M.data() + (size_t)i * aug_cols, aug_cols, C.msg.data() + (size_t)i * C.pc);
        bfz_encode_row(nt, C.msg.data() + (size_t)i * C.pc, C.pc, C.nc,
                       C.cw.data() + (size_t)i * C.nc);
    }
    bfz_build_tree(C);
    tr.absorb("bfz-root", C.root, 32);
    tr.absorb("bfz-sseed", &C.sseed, sizeof C.sseed);
}

// combined row over T_128 across a SUBSET of rows [lo,hi): out[jc]=sum coef[i]*bit(i,jc)
static inline void bfz_combine(const ZkCommit& C, const bf128_t* coef, uint32_t lo, uint32_t hi,
                               std::vector<bf128_t>& out) {
    size_t nu = (size_t)1 << C.p.lcol_a;
    out.assign(nu, bf128_zero());
    for (uint32_t i = lo; i < hi; i++)
        for (uint32_t jp = 0; jp < C.pc; jp++) {
            uint32_t m = C.msg[(size_t)i * C.pc + jp];
            while (m) { int uu = __builtin_ctz(m); m &= m - 1;
                out[16*jp+uu] = bf128_add(out[16*jp+uu], coef[i]); }
        }
}

static inline void bfz_open(const ZkCommit& C, const bf128_t* r, bf128_t v,
                            fs::Transcript& tr, ZkProof& pf) {
    const ZkParams& p = C.p;
    // eval point pieces: r[0..lcol) real col, r[lcol..l) real row.  Augmented
    // col point = [ r_lo | 0^e ] ; row eq uses r_hi over the low-half indexing,
    // shared by the witness rows (low half) and the masking rows g (high half).
    std::vector<bf128_t> rcol_a(p.lcol_a, bf128_zero());
    for (int i = 0; i < p.lcol; i++) rcol_a[i] = r[i];      // ex coords stay 0
    std::vector<bf128_t> eqcol, eqrow;
    bf_eq_table(rcol_a.data(), p.lcol_a, eqcol);            // length 2^lcol_a
    bf_eq_table(r + p.lcol, p.lrow, eqrow);                 // length 2^lrow

    pf.sseed = C.sseed;
    tr.absorb("bfz-point", r, p.l * sizeof(bf128_t));
    tr.absorb("bfz-value", &v, sizeof v);

    // ---- masking-polynomial eval: t_M (low half) and t_g (high half g) ----
    std::vector<bf128_t> t_M, t_g;
    bfz_combine(C, eqrow.data(), 0, C.n_real, t_M);                 // witness rows
    if (C.n_rows > C.n_real) {                                      // g = high half
        std::vector<bf128_t> coef(C.n_rows, bf128_zero());
        for (uint32_t i = 0; i < C.n_real; i++) coef[C.n_real + i] = eqrow[i];
        bfz_combine(C, coef.data(), C.n_real, C.n_rows, t_g);
    } else t_g.assign(1u << p.lcol_a, bf128_zero());
    // y_g = <eqcol, t_g>, sent BEFORE lambda is drawn
    pf.yg = bf128_zero();
    for (uint32_t j = 0; j < (1u << p.lcol_a); j++)
        pf.yg = bf128_add(pf.yg, bf128_mul(eqcol[j], t_g[j]));
    tr.absorb("bfz-yg", &pf.yg, sizeof pf.yg);
    bf128_t lam = bf_chal128(tr);
    // tau = t_M + lambda * t_g  (uniform, one-time-padded by lambda*t_g)
    pf.t.assign(1u << p.lcol_a, bf128_zero());
    for (uint32_t j = 0; j < (1u << p.lcol_a); j++)
        pf.t[j] = bf128_add(t_M[j], bf128_mul(lam, t_g[j]));
    tr.absorb("bfz-t", pf.t.data(), pf.t.size() * sizeof(bf128_t));

    // ---- proximity row u' = sum_{all rows} rho_i M_aug[i]  (g rows pad it) ----
    std::vector<bf128_t> rho(C.n_rows);
    for (uint32_t i = 0; i < C.n_rows; i++) rho[i] = bf_chal128(tr);
    bfz_combine(C, rho.data(), 0, C.n_rows, pf.u);
    tr.absorb("bfz-u", pf.u.data(), pf.u.size() * sizeof(bf128_t));

    // ---- spot columns ----
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

static inline bool bfz_verify(const ZkParams& p, const uint8_t root[32],
                              const bf128_t* r, bf128_t v, fs::Transcript& tr,
                              const ZkProof& pf) {
    uint32_t n_real = 1u << p.lrow;
    uint32_t n_rows = 1u << p.lrow_a;
    uint32_t pc = 1u << (p.lcol_a - 4), nc = pc << BFPCS_RATE_LOG;
    size_t nu = (size_t)1 << p.lcol_a;
    if (pf.t.size() != nu || pf.u.size() != nu) return false;
    if (pf.cols.size() != (size_t)p.Q * n_rows) return false;

    tr.absorb("bfz-point", r, p.l * sizeof(bf128_t));
    tr.absorb("bfz-value", &v, sizeof v);
    tr.absorb("bfz-yg", &pf.yg, sizeof pf.yg);
    bf128_t lam = bf_chal128(tr);
    tr.absorb("bfz-t", pf.t.data(), pf.t.size() * sizeof(bf128_t));
    std::vector<bf128_t> rho(n_rows);
    for (uint32_t i = 0; i < n_rows; i++) rho[i] = bf_chal128(tr);
    tr.absorb("bfz-u", pf.u.data(), pf.u.size() * sizeof(bf128_t));
    std::vector<uint32_t> q;
    bfpcs_queries(tr, nc, p.Q, q);
    int depth = 0; while ((1u << depth) < nc) depth++;
    if (pf.paths.size() != (size_t)p.Q * depth * 32) return false;

    // augmented col/row eq
    std::vector<bf128_t> rcol_a(p.lcol_a, bf128_zero());
    for (int i = 0; i < p.lcol; i++) rcol_a[i] = r[i];
    std::vector<bf128_t> eqcol, eqrow;
    bf_eq_table(rcol_a.data(), p.lcol_a, eqcol);
    bf_eq_table(r + p.lcol, p.lrow, eqrow);

    // re-encode tau, u' over T_128 (the verifier's limbwise encode)
    BfNtt nt; bfntt_init(nt, p.lcol_a - 4 + BFPCS_RATE_LOG);
    std::vector<bf128_t> W, U;
    bf_pack128(pf.t.data(), nu, W); W.resize(nc, bf128_zero()); bfntt_fwd_host128(nt, W.data());
    bf_pack128(pf.u.data(), nu, U); U.resize(nc, bf128_zero()); bfntt_fwd_host128(nt, U.data());

    for (int k = 0; k < p.Q; k++) {
        uint32_t j = q[k];
        const bf16_t* col = pf.cols.data() + (size_t)k * n_rows;
        // salted Merkle path
        uint8_t h[32];
        bfz_leaf(col, n_rows, pf.sseed, (uint64_t)j, h);
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
        // proximity consistency: sum_{all rows} rho_i col[i] == U[j]
        bf128_t sr = bf128_zero();
        for (uint32_t i = 0; i < n_rows; i++)
            sr = bf128_add(sr, bf128_smul16(rho[i], col[i]));
        if (!bf128_eq(sr, U[j])) return false;
        // eval consistency: (sum_low eq_i col[i]) + lambda*(sum_high eq_i col[hi]) == W[j]
        bf128_t sM = bf128_zero(), sG = bf128_zero();
        for (uint32_t i = 0; i < n_real; i++) {
            sM = bf128_add(sM, bf128_smul16(eqrow[i], col[i]));
            if (n_rows > n_real)
                sG = bf128_add(sG, bf128_smul16(eqrow[i], col[n_real + i]));
        }
        bf128_t se = bf128_add(sM, bf128_mul(lam, sG));
        if (!bf128_eq(se, W[j])) return false;
    }
    // evaluation claim: <eqcol, tau> == v + lambda*y_g  (lambda drawn after y_g)
    bf128_t acc = bf128_zero();
    for (size_t j = 0; j < nu; j++) acc = bf128_add(acc, bf128_mul(eqcol[j], pf.t[j]));
    return bf128_eq(acc, bf128_add(v, bf128_mul(lam, pf.yg)));
}

} // namespace bfz
