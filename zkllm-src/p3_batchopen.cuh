// Batched Basefold openings: many (column, point, claimed-eval) obligations,
// collapsed per SIZE CLASS (log2 rows) into
//   (1) one multi-point REDUCTION sumcheck   sum_b sum_t G_t(b)*eq(z_t,b)
//       with G_t = sum_{claims j at point t} mu^j * col_j  (mu from transcript),
//       claim0 = sum_j mu^j y_j, ending at a single point r*;
//   (2) per-column terminal claims y*_c = col_c(r*) (absorbed), combined with a
//       second challenge rho into U = sum_c rho^c col_c, opened ONCE at r* by
//       the standard Basefold fold/query protocol -- except round 0 carries no
//       new Merkle root: each query's round-0 values are opened PER COLUMN
//       against the columns' ORIGINAL commitment roots and the verifier forms
//       the combined value itself (classic batch-FRI round-0 authentication).
//
// Soundness: wrong y_j breaks the reduction terminal unless some y*_c is wrong;
// wrong y*_c makes yU != U(r*), caught by the RLC opening (Schwartz-Zippel over
// rho + FRI proximity / correlated agreement for the combined word).  Columns
// are deduplicated BY ROOT (equal roots => equal committed words, so a shared
// y* is sound).  Challenges are base-field, like the rest of the p3 stack
// (GL2 upgrade applies stack-wide).
//
// GPU-resident throughout (requires p3fri::g_gpu_merkle path conventions).
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
#include "p3_ntt.cuh"
#include "p3_basefold.cuh"
#include "fs_transcript.hpp"

namespace p3bo {

using p3fri::Hash;
using p3bf::SumMsg;

static inline gl_t chal(fs::Transcript& tr) {
    uint8_t b[32]; tr.challenge_bytes(b);
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v % GL_P;
}

// per-query round-0 openings: entry c = the c-th DISTINCT column of the class
struct BQ0 { std::vector<gl_t> a, b; std::vector<std::vector<Hash>> pa, pb; };

struct BatchProof {
    uint32_t v = 0, R = 0, Q = 0, nc = 0;      // rows=2^v; nc = #distinct columns
    std::vector<gl_t> ystar;                   // nc terminal per-column evals at r*
    std::vector<SumMsg> rmsgs;                 // v reduction-sumcheck messages
    std::vector<Hash> roots;                   // v-1 opening roots (rounds 1..v-1)
    std::vector<SumMsg> omsgs;                 // v opening-sumcheck messages
    std::vector<gl_t> final_word;              // 2^R
    std::vector<p3fri::QueryProof> queries;    // Q; rounds[i] = opening round i+1
    std::vector<BQ0> q0;                       // Q round-0 per-column openings
};

// prover ledger: values pointer + commitment root per obligation
struct PLedger {
    struct Ent { const std::vector<gl_t>* vals; Hash root; uint32_t zid; gl_t y; };
    struct Cls { uint32_t v; std::vector<std::vector<gl_t>> pts; std::vector<Ent> ents; };
    std::vector<Cls> cls;
    void add(const std::vector<gl_t>* vals, const Hash& root, const std::vector<gl_t>& z, gl_t y) {
        uint32_t vv = (uint32_t)z.size();
        for (auto& c : cls) if (c.v == vv) {
            uint32_t zid = 0;
            while (zid < c.pts.size() && c.pts[zid] != z) zid++;
            if (zid == c.pts.size()) c.pts.push_back(z);
            c.ents.push_back({vals, root, zid, y});
            return;
        }
        cls.push_back({vv, {z}, {{vals, root, 0, y}}});
    }
};
// verifier ledger: roots only
struct VLedger {
    struct Ent { Hash root; uint32_t zid; gl_t y; };
    struct Cls { uint32_t v; std::vector<std::vector<gl_t>> pts; std::vector<Ent> ents; };
    std::vector<Cls> cls;
    void add(const Hash& root, const std::vector<gl_t>& z, gl_t y) {
        uint32_t vv = (uint32_t)z.size();
        for (auto& c : cls) if (c.v == vv) {
            uint32_t zid = 0;
            while (zid < c.pts.size() && c.pts[zid] != z) zid++;
            if (zid == c.pts.size()) c.pts.push_back(z);
            c.ents.push_back({root, zid, y});
            return;
        }
        cls.push_back({vv, {z}, {{root, 0, y}}});
    }
};

__global__ void p3bo_axpy_kernel(gl_t* out, const gl_t* in, gl_t s, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    out[i] = gl_add(out[i], gl_mul(s, in[i]));
}

// device coeffs -> device codeword (zero-padded forward NTT)
static inline gl_t* bo_encode_dev(const gl_t* d_vals, uint32_t N, uint32_t logM0, const P3Ntt& ntt) {
    uint32_t M0 = 1u << logM0; gl_t *d_in, *d_out;
    cudaMallocAsync(&d_in, (size_t)M0 * 8, 0); cudaMemsetAsync(d_in, 0, (size_t)M0 * 8, 0);
    cudaMemcpyAsync(d_in, d_vals, (size_t)N * 8, cudaMemcpyDeviceToDevice, 0);
    cudaMallocAsync(&d_out, (size_t)M0 * 8, 0);
    ntt.run(d_in, d_out, true);
    cudaFreeAsync(d_in, 0);
    return d_out;
}

// degree-2 sumcheck message of sum_b c[b]*w[b] over the LSB split (device pair)
static inline SumMsg bo_scmsg(const gl_t* d_c, const gl_t* d_w, uint32_t half,
                              gl_t* db0, gl_t* db1, gl_t* db2,
                              std::vector<gl_t>& hb0, std::vector<gl_t>& hb1, std::vector<gl_t>& hb2) {
    const uint32_t NB = 256;
    p3bf::p3bf_scmsg_kernel<<<NB, 256>>>(d_c, d_w, db0, db1, db2, half);
    cudaMemcpy(hb0.data(), db0, NB * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb1.data(), db1, NB * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb2.data(), db2, NB * 8, cudaMemcpyDeviceToHost);
    gl_t s0 = 0, s1 = 0, s2 = 0;
    for (uint32_t i = 0; i < NB; i++) {
        s0 = gl_add(s0, hb0[i]); s1 = gl_add(s1, hb1[i]); s2 = gl_add(s2, hb2[i]);
    }
    return {s0, s1, s2};
}

// ==================== prover ====================
static inline BatchProof prove_class(fs::Transcript& tr, const PLedger::Cls& C,
                                     uint32_t R, uint32_t Q, const std::string& label) {
    const uint32_t v = C.v, N = 1u << v, T = (uint32_t)C.pts.size();
    const uint32_t logM0 = v + R, M0 = 1u << logM0;
    const size_t k = C.ents.size();
    BatchProof pf; pf.v = v; pf.R = R; pf.Q = Q;

    // distinct columns by root (first appearance), entry -> column map
    std::vector<const std::vector<gl_t>*> hcols;
    std::vector<Hash> droots;
    std::vector<uint32_t> colidx(k);
    for (size_t j = 0; j < k; j++) {
        uint32_t c = 0;
        while (c < droots.size() && !(droots[c] == C.ents[j].root)) c++;
        if (c == droots.size()) { droots.push_back(C.ents[j].root); hcols.push_back(C.ents[j].vals); }
        colidx[j] = c;
    }
    const uint32_t nc = (uint32_t)hcols.size();
    pf.nc = nc;

    gl_t mu = chal(tr);

    // upload distinct columns
    std::vector<gl_t*> dcol(nc);
    for (uint32_t c = 0; c < nc; c++) {
        cudaMallocAsync(&dcol[c], (size_t)N * 8, 0);
        cudaMemcpy(dcol[c], hcols[c]->data(), (size_t)N * 8, cudaMemcpyHostToDevice);
    }
    // per-point combined columns G_t and eq weights
    std::vector<gl_t*> dG(T), dEq(T);
    for (uint32_t t = 0; t < T; t++) {
        cudaMallocAsync(&dG[t], (size_t)N * 8, 0);
        cudaMemsetAsync(dG[t], 0, (size_t)N * 8, 0);
        gl_t* dz; cudaMallocAsync(&dz, (size_t)v * 8, 0);
        cudaMemcpy(dz, C.pts[t].data(), (size_t)v * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dEq[t], (size_t)N * 8, 0);
        p3bf::p3bf_eq_kernel<<<(N + 255) / 256, 256>>>(dz, dEq[t], v, N);
        cudaFreeAsync(dz, 0);
    }
    {
        gl_t w = 1ULL;
        for (size_t j = 0; j < k; j++) {
            p3bo_axpy_kernel<<<(N + 255) / 256, 256>>>(dG[C.ents[j].zid], dcol[colidx[j]], w, N);
            w = gl_mul(w, mu);
        }
    }
    const uint32_t NB = 256;
    gl_t *db0, *db1, *db2;
    cudaMallocAsync(&db0, NB * 8, 0); cudaMallocAsync(&db1, NB * 8, 0); cudaMallocAsync(&db2, NB * 8, 0);
    std::vector<gl_t> hb0(NB), hb1(NB), hb2(NB);

    // ---- reduction sumcheck ----
    std::vector<gl_t> rstar; rstar.reserve(v);
    for (uint32_t rd = 0; rd < v; rd++) {
        uint32_t half = (N >> rd) / 2;
        gl_t s0 = 0, s1 = 0, s2 = 0;
        for (uint32_t t = 0; t < T; t++) {
            SumMsg m = bo_scmsg(dG[t], dEq[t], half, db0, db1, db2, hb0, hb1, hb2);
            s0 = gl_add(s0, m.s0); s1 = gl_add(s1, m.s1); s2 = gl_add(s2, m.s2);
        }
        SumMsg msg{s0, s1, s2};
        pf.rmsgs.push_back(msg); tr.absorb("bo-rm", &msg, sizeof msg);
        gl_t a = chal(tr); rstar.push_back(a);
        for (uint32_t t = 0; t < T; t++) {
            gl_t *nG, *nE;
            cudaMallocAsync(&nG, (size_t)half * 8, 0); cudaMallocAsync(&nE, (size_t)half * 8, 0);
            p3bf::p3bf_bind_kernel<<<(half + 255) / 256, 256>>>(dG[t], nG, half, a);
            p3bf::p3bf_bind_kernel<<<(half + 255) / 256, 256>>>(dEq[t], nE, half, a);
            cudaFreeAsync(dG[t], 0); cudaFreeAsync(dEq[t], 0);
            dG[t] = nG; dEq[t] = nE;
        }
    }
    for (uint32_t t = 0; t < T; t++) { cudaFreeAsync(dG[t], 0); cudaFreeAsync(dEq[t], 0); }

    // ---- per-column terminal evals at r* ----
    gl_t* dEqr;
    {
        gl_t* dz; cudaMallocAsync(&dz, (size_t)v * 8, 0);
        cudaMemcpy(dz, rstar.data(), (size_t)v * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dEqr, (size_t)N * 8, 0);
        p3bf::p3bf_eq_kernel<<<(N + 255) / 256, 256>>>(dz, dEqr, v, N);
        cudaFreeAsync(dz, 0);
    }
    pf.ystar.resize(nc);
    for (uint32_t c = 0; c < nc; c++) {
        p3bf::p3bf_dot_kernel<<<NB, 256>>>(dcol[c], dEqr, db0, N);
        cudaMemcpy(hb0.data(), db0, NB * 8, cudaMemcpyDeviceToHost);
        gl_t s = 0; for (uint32_t i = 0; i < NB; i++) s = gl_add(s, hb0[i]);
        pf.ystar[c] = s;
    }
    tr.absorb("bo-ys", pf.ystar.data(), (size_t)nc * 8);
    gl_t rho = chal(tr);

    // ---- RLC column U and its claimed eval ----
    gl_t* dU; cudaMallocAsync(&dU, (size_t)N * 8, 0); cudaMemsetAsync(dU, 0, (size_t)N * 8, 0);
    gl_t yU = 0, w = 1ULL;
    for (uint32_t c = 0; c < nc; c++) {
        p3bo_axpy_kernel<<<(N + 255) / 256, 256>>>(dU, dcol[c], w, N);
        yU = gl_add(yU, gl_mul(w, pf.ystar[c]));
        w = gl_mul(w, rho);
    }

    // ---- Basefold opening of U at r* (round 0 uncommitted) ----
    P3Ntt ntt(logM0);
    std::vector<gl_t*> d_cw(v + 1, nullptr);
    d_cw[0] = bo_encode_dev(dU, N, logM0, ntt);
    fs::Transcript ot(label);
    ot.absorb("z", rstar.data(), (size_t)v * 8);
    ot.absorb("y", &yU, 8);
    gl_t* d_c = dU;            // consumed by the sumcheck binds
    gl_t* d_w = dEqr;
    std::vector<p3fri::DeviceMerkle> dtrees(v);   // [r] valid for r>=1
    for (uint32_t r = 0; r < v; r++) {
        uint32_t half = (N >> r) / 2, Mr = M0 >> r;
        if (r > 0) {
            dtrees[r].build_dev(d_cw[r], Mr); cudaDeviceSynchronize();
            Hash root = dtrees[r].root();
            pf.roots.push_back(root); ot.absorb("root", root.data(), 32);
        }
        SumMsg msg = bo_scmsg(d_c, d_w, half, db0, db1, db2, hb0, hb1, hb2);
        pf.omsgs.push_back(msg); ot.absorb("sc", &msg, sizeof msg);
        gl_t a = p3bf::alpha_from(ot);
        gl_t *ncf, *nwf;
        cudaMallocAsync(&ncf, (size_t)half * 8, 0); cudaMallocAsync(&nwf, (size_t)half * 8, 0);
        p3bf::p3bf_bind_kernel<<<(half + 255) / 256, 256>>>(d_c, ncf, half, a);
        p3bf::p3bf_bind_kernel<<<(half + 255) / 256, 256>>>(d_w, nwf, half, a);
        cudaFreeAsync(d_c, 0); cudaFreeAsync(d_w, 0); d_c = ncf; d_w = nwf;
        gl_t wr = gl_root_of_unity(logM0 - r), winv = gl_inv(wr), inv2 = gl_inv(2ULL);
        uint32_t fh = Mr / 2;
        cudaMallocAsync(&d_cw[r + 1], (size_t)fh * 8, 0);
        p3bf::p3bf_fold_kernel<<<(fh + 255) / 256, 256>>>(d_cw[r], d_cw[r + 1], fh, winv, a, inv2);
    }
    cudaFreeAsync(d_c, 0); cudaFreeAsync(d_w, 0);
    pf.final_word.resize(1u << R);
    cudaMemcpy(pf.final_word.data(), d_cw[v], (size_t)(1u << R) * 8, cudaMemcpyDeviceToHost);
    ot.absorb("final", pf.final_word.data(), pf.final_word.size() * 8);

    // query coset indices (same chain rule as prove_eval_dev)
    std::vector<std::vector<uint32_t>> ccq(Q, std::vector<uint32_t>(v));
    for (uint32_t q = 0; q < Q; q++) {
        uint32_t c0 = (uint32_t)p3fri::idx_from(ot, M0 / 2), p = c0;
        for (uint32_t r = 0; r < v; r++) { uint32_t h = (M0 >> r) / 2, cc = p % h; ccq[q][r] = cc; p = cc; }
    }
    // rounds >= 1 openings against the fresh fold roots
    pf.queries.assign(Q, {});
    for (auto& x : pf.queries) x.rounds.resize(v > 0 ? v - 1 : 0);
    for (uint32_t r = 1; r < v; r++) {
        uint32_t h = (M0 >> r) / 2;
        std::vector<uint32_t> idxs(2 * Q);
        for (uint32_t q = 0; q < Q; q++) { idxs[2 * q] = ccq[q][r]; idxs[2 * q + 1] = ccq[q][r] + h; }
        uint32_t* d_idx; cudaMallocAsync(&d_idx, (size_t)2 * Q * 4, 0);
        cudaMemcpy(d_idx, idxs.data(), (size_t)2 * Q * 4, cudaMemcpyHostToDevice);
        gl_t* d_val; cudaMallocAsync(&d_val, (size_t)2 * Q * 8, 0);
        p3bf::p3bf_gather_kernel<<<(2 * Q + 255) / 256, 256>>>(d_cw[r], d_idx, d_val, 2 * Q);
        std::vector<gl_t> vals(2 * Q);
        cudaMemcpy(vals.data(), d_val, (size_t)2 * Q * 8, cudaMemcpyDeviceToHost);
        cudaFreeAsync(d_idx, 0); cudaFreeAsync(d_val, 0);
        auto paths = dtrees[r].paths_batch(idxs);
        for (uint32_t q = 0; q < Q; q++) {
            auto& ro = pf.queries[q].rounds[r - 1];
            ro.a = vals[2 * q]; ro.b = vals[2 * q + 1];
            ro.pa = paths[2 * q]; ro.pb = paths[2 * q + 1];
        }
    }
    // round-0 per-column openings against the ORIGINAL commitments
    {
        std::vector<uint32_t> idxs0(2 * Q);
        for (uint32_t q = 0; q < Q; q++) { idxs0[2 * q] = ccq[q][0]; idxs0[2 * q + 1] = ccq[q][0] + M0 / 2; }
        uint32_t* d_idx; cudaMallocAsync(&d_idx, (size_t)2 * Q * 4, 0);
        cudaMemcpy(d_idx, idxs0.data(), (size_t)2 * Q * 4, cudaMemcpyHostToDevice);
        gl_t* d_val; cudaMallocAsync(&d_val, (size_t)2 * Q * 8, 0);
        pf.q0.assign(Q, {});
        for (uint32_t q = 0; q < Q; q++) {
            pf.q0[q].a.resize(nc); pf.q0[q].b.resize(nc);
            pf.q0[q].pa.resize(nc); pf.q0[q].pb.resize(nc);
        }
        std::vector<gl_t> vals(2 * Q);
        for (uint32_t c = 0; c < nc; c++) {
            gl_t* d_cwc = bo_encode_dev(dcol[c], N, logM0, ntt);
            p3fri::DeviceMerkle mk; mk.build_dev(d_cwc, M0); cudaDeviceSynchronize();
            p3bf::p3bf_gather_kernel<<<(2 * Q + 255) / 256, 256>>>(d_cwc, d_idx, d_val, 2 * Q);
            cudaMemcpy(vals.data(), d_val, (size_t)2 * Q * 8, cudaMemcpyDeviceToHost);
            auto paths = mk.paths_batch(idxs0);
            for (uint32_t q = 0; q < Q; q++) {
                pf.q0[q].a[c] = vals[2 * q]; pf.q0[q].b[c] = vals[2 * q + 1];
                pf.q0[q].pa[c] = paths[2 * q]; pf.q0[q].pb[c] = paths[2 * q + 1];
            }
            mk.free_(); cudaFreeAsync(d_cwc, 0);
        }
        cudaFreeAsync(d_idx, 0); cudaFreeAsync(d_val, 0);
    }
    for (uint32_t r = 0; r <= v; r++) if (d_cw[r]) cudaFreeAsync(d_cw[r], 0);
    for (uint32_t r = 1; r < v; r++) dtrees[r].free_();
    for (uint32_t c = 0; c < nc; c++) cudaFreeAsync(dcol[c], 0);
    cudaFreeAsync(db0, 0); cudaFreeAsync(db1, 0); cudaFreeAsync(db2, 0);
    cudaDeviceSynchronize();
    return pf;
}

// ==================== verifier ====================
static inline bool verify_class(fs::Transcript& tr, const VLedger::Cls& C, const BatchProof& pf,
                                uint32_t Q_pub, uint32_t R_pub, const std::string& label,
                                const char** why) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    const uint32_t v = C.v, T = (uint32_t)C.pts.size();
    const uint32_t logM0 = v + R_pub, M0 = 1u << logM0;
    const size_t k = C.ents.size();
    if (pf.v != v || pf.R != R_pub || pf.Q != Q_pub) return fail("batch params");

    // distinct columns by root, first appearance (must match prover)
    std::vector<Hash> droots;
    std::vector<uint32_t> colidx(k);
    for (size_t j = 0; j < k; j++) {
        uint32_t c = 0;
        while (c < droots.size() && !(droots[c] == C.ents[j].root)) c++;
        if (c == droots.size()) droots.push_back(C.ents[j].root);
        colidx[j] = c;
    }
    const uint32_t nc = (uint32_t)droots.size();
    if (pf.nc != nc || pf.ystar.size() != nc) return fail("batch col count");
    if (pf.rmsgs.size() != v || pf.omsgs.size() != v) return fail("batch msg count");
    if (pf.roots.size() != (v > 0 ? v - 1 : 0)) return fail("batch root count");
    if (pf.final_word.size() != (1u << R_pub)) return fail("batch final size");
    if (pf.queries.size() != Q_pub || pf.q0.size() != Q_pub) return fail("batch query count");

    gl_t mu = chal(tr);
    gl_t claim = 0, wj = 1ULL;
    for (size_t j = 0; j < k; j++) { claim = gl_add(claim, gl_mul(wj, C.ents[j].y)); wj = gl_mul(wj, mu); }

    // reduction sumcheck
    std::vector<gl_t> rstar(v);
    for (uint32_t rd = 0; rd < v; rd++) {
        const SumMsg& m = pf.rmsgs[rd];
        if (gl_add(m.s0, m.s1) != claim) return fail("batch reduction claim");
        tr.absorb("bo-rm", &m, sizeof m);
        gl_t a = chal(tr); rstar[rd] = a;
        claim = p3bf::quad_eval(m.s0, m.s1, m.s2, a);
    }
    tr.absorb("bo-ys", pf.ystar.data(), (size_t)nc * 8);
    // terminal: sum_t eq(z_t, r*) * sum_{j in t} mu^j * ystar[col(j)]
    {
        std::vector<gl_t> gt(T, 0);
        wj = 1ULL;
        for (size_t j = 0; j < k; j++) {
            gt[C.ents[j].zid] = gl_add(gt[C.ents[j].zid], gl_mul(wj, pf.ystar[colidx[j]]));
            wj = gl_mul(wj, mu);
        }
        gl_t end = 0;
        for (uint32_t t = 0; t < T; t++)
            end = gl_add(end, gl_mul(p3bf::eq_point(rstar, C.pts[t]), gt[t]));
        if (end != claim) return fail("batch reduction terminal");
    }
    gl_t rho = chal(tr);
    gl_t yU = 0, wc = 1ULL;
    std::vector<gl_t> rhop(nc);
    for (uint32_t c = 0; c < nc; c++) { rhop[c] = wc; yU = gl_add(yU, gl_mul(wc, pf.ystar[c])); wc = gl_mul(wc, rho); }

    // opening verify (round 0 uncommitted)
    fs::Transcript ot(label);
    ot.absorb("z", rstar.data(), (size_t)v * 8);
    ot.absorb("y", &yU, 8);
    std::vector<gl_t> alphas(v);
    gl_t oclaim = yU;
    for (uint32_t r = 0; r < v; r++) {
        if (r > 0) ot.absorb("root", pf.roots[r - 1].data(), 32);
        const SumMsg& m = pf.omsgs[r];
        if (gl_add(m.s0, m.s1) != oclaim) return fail("batch opening claim");
        ot.absorb("sc", &m, sizeof m);
        gl_t a = p3bf::alpha_from(ot); alphas[r] = a;
        oclaim = p3bf::quad_eval(m.s0, m.s1, m.s2, a);
    }
    ot.absorb("final", pf.final_word.data(), pf.final_word.size() * 8);
    for (size_t i = 1; i < pf.final_word.size(); i++)
        if (pf.final_word[i] != pf.final_word[0]) return fail("batch final not constant");
    if (oclaim != gl_mul(pf.final_word[0], p3bf::eq_point(alphas, rstar)))
        return fail("batch eval tie");

    // queries
    for (uint32_t q = 0; q < Q_pub; q++) {
        uint32_t c0 = (uint32_t)p3fri::idx_from(ot, M0 / 2);
        const BQ0& z0 = pf.q0[q];
        if (z0.a.size() != nc || z0.b.size() != nc || z0.pa.size() != nc || z0.pb.size() != nc)
            return fail("batch q0 size");
        uint32_t p = c0;
        for (uint32_t r = 0; r < v; r++) {
            uint32_t half = (M0 >> r) / 2, c = p % half;
            gl_t a, b;
            if (r == 0) {
                a = 0; b = 0;
                for (uint32_t cc = 0; cc < nc; cc++) {
                    if (!p3fri::verify_path(p3fri::leaf_hash(z0.a[cc]), c, z0.pa[cc], droots[cc]))
                        return fail("batch q0 merkle a");
                    if (!p3fri::verify_path(p3fri::leaf_hash(z0.b[cc]), c + half, z0.pb[cc], droots[cc]))
                        return fail("batch q0 merkle b");
                    a = gl_add(a, gl_mul(rhop[cc], z0.a[cc]));
                    b = gl_add(b, gl_mul(rhop[cc], z0.b[cc]));
                }
            } else {
                const p3fri::RoundOpen& ro = pf.queries[q].rounds[r - 1];
                if (!p3fri::verify_path(p3fri::leaf_hash(ro.a), c, ro.pa, pf.roots[r - 1]))
                    return fail("batch merkle a");
                if (!p3fri::verify_path(p3fri::leaf_hash(ro.b), c + half, ro.pb, pf.roots[r - 1]))
                    return fail("batch merkle b");
                a = ro.a; b = ro.b;
            }
            uint32_t logMr = logM0 - r;
            gl_t wr = gl_root_of_unity(logMr);
            gl_t x = gl_pow(wr, c), invx = gl_inv(x);
            gl_t folded = p3fri::fold_pair(a, b, invx, alphas[r], /*mle=*/true);
            if (r + 1 < v) {
                uint32_t nhalf = (M0 >> (r + 1)) / 2;
                const p3fri::RoundOpen& nro = pf.queries[q].rounds[r];
                gl_t val = (c < nhalf) ? nro.a : nro.b;
                if (val != folded) return fail("batch fold link");
            } else {
                if (pf.final_word[c] != folded) return fail("batch fold to final");
            }
            p = c;
        }
    }
    return true;
}

// serialized-payload size accounting
static inline size_t sz_batch(const BatchProof& pf) {
    size_t s = 16 + pf.ystar.size() * 8 + (pf.rmsgs.size() + pf.omsgs.size()) * 24
             + pf.roots.size() * 32 + pf.final_word.size() * 8;
    for (auto& q : pf.queries)
        for (auto& r : q.rounds) s += 16 + (r.pa.size() + r.pb.size()) * 32;
    for (auto& z : pf.q0) {
        s += (z.a.size() + z.b.size()) * 8;
        for (auto& p : z.pa) s += p.size() * 32;
        for (auto& p : z.pb) s += p.size() * 32;
    }
    return s;
}

} // namespace p3bo
