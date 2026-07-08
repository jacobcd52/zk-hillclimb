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
#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <tuple>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
#include "p3_ntt.cuh"
#include "p3_basefold.cuh"
#include "p3_zkc.cuh"
#include "fs_transcript.hpp"

namespace p3bo {

using p3fri::Hash;
using p3bf::SumMsg;

static inline gl_t chal(fs::Transcript& tr) {
    uint8_t b[32]; tr.challenge_bytes(b);
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v % GL_P;
}

// round-0 subset opening of ONE distinct column: values (+ zk salts) at the
// class's sorted-unique query positions, plus the PRUNED Merkle node stream --
// the 2Q overlapping authentication paths cost each tree node at most once.
struct BQ0C { std::vector<gl_t> vals; std::vector<p3zkc::Salt> salts;
              std::vector<Hash> nodes; };

struct BatchProof {
    uint32_t v = 0, R = 0, Q = 0, nc = 0;      // rows=2^v; nc = #distinct columns
    std::vector<gl_t> ystar;                   // nc terminal per-column evals at r*
    std::vector<SumMsg> rmsgs;                 // v reduction-sumcheck messages
    std::vector<Hash> roots;                   // v-1 opening roots (rounds 1..v-1)
    std::vector<SumMsg> omsgs;                 // v opening-sumcheck messages
    std::vector<gl_t> final_word;              // 2^R
    std::vector<p3fri::QueryProof> queries;    // Q; rounds[i] = opening round i+1
    std::vector<BQ0C> q0c;                     // nc round-0 subset openings
    Hash bl_root = {};                         // zk: class blinder column commitment
    gl_t bl_y = 0;                             // zk: blinder claim at the fresh point
};

// ---- Merkle subset proof (pruned multi-path) ----
// Deterministic bottom-up frontier walk over the sorted-unique leaf positions:
// where BOTH children of a parent are in the frontier the parent is computed;
// otherwise exactly one sibling hash is consumed from (verify) / emitted to
// (prove) the canonical node stream.  Same walk on both sides, so the stream
// order needs no indices.
// prover side: paths[i] = full auth path of leaf upos[i]; the sibling of a
// frontier node at level l is path node l of any covered leaf below it (the
// walk tracks the first covered leaf's slot).
static inline std::vector<Hash> subset_prove(uint32_t logM, const std::vector<uint32_t>& upos,
                                             const std::vector<Hash>& leaves,
                                             const std::vector<std::vector<Hash>>& paths,
                                             Hash* root_out = nullptr) {
    std::vector<Hash> nodes;
    // frontier of (nodeidx, hash, leafslot): leafslot = index into upos/paths
    std::vector<std::tuple<uint32_t, Hash, uint32_t>> fr(upos.size());
    for (size_t i = 0; i < upos.size(); i++) fr[i] = {upos[i], leaves[i], (uint32_t)i};
    for (uint32_t l = 0; l < logM; l++) {
        std::vector<std::tuple<uint32_t, Hash, uint32_t>> nx;
        size_t i = 0;
        while (i < fr.size()) {
            uint32_t idx = std::get<0>(fr[i]);
            uint32_t slot = std::get<2>(fr[i]);
            Hash h;
            if (!(idx & 1) && i + 1 < fr.size() && std::get<0>(fr[i + 1]) == (idx | 1)) {
                h = p3fri::node_hash(std::get<1>(fr[i]), std::get<1>(fr[i + 1]));
                i += 2;
            } else {
                Hash sib = paths[slot][l];
                nodes.push_back(sib);
                h = (idx & 1) ? p3fri::node_hash(sib, std::get<1>(fr[i]))
                              : p3fri::node_hash(std::get<1>(fr[i]), sib);
                i += 1;
            }
            nx.push_back({idx >> 1, h, slot});
        }
        fr = std::move(nx);
    }
    if (root_out) *root_out = std::get<1>(fr[0]);
    return nodes;
}
// verifier-side wrapper: reconstruct the root from leaves + pruned node stream
static inline bool subset_verify(uint32_t logM, const std::vector<uint32_t>& upos,
                                 const std::vector<Hash>& leaves,
                                 const std::vector<Hash>& nodes, const Hash& root) {
    std::vector<std::pair<uint32_t, Hash>> fr(upos.size());
    for (size_t i = 0; i < upos.size(); i++) fr[i] = {upos[i], leaves[i]};
    size_t ni = 0;
    for (uint32_t l = 0; l < logM; l++) {
        std::vector<std::pair<uint32_t, Hash>> nx;
        size_t i = 0;
        while (i < fr.size()) {
            uint32_t idx = fr[i].first;
            Hash h;
            if (!(idx & 1) && i + 1 < fr.size() && fr[i + 1].first == (idx | 1)) {
                h = p3fri::node_hash(fr[i].second, fr[i + 1].second);
                i += 2;
            } else {
                if (ni >= nodes.size()) return false;
                const Hash& sib = nodes[ni++];
                h = (idx & 1) ? p3fri::node_hash(sib, fr[i].second)
                              : p3fri::node_hash(fr[i].second, sib);
                i += 1;
            }
            nx.push_back({idx >> 1, h});
        }
        fr = std::move(nx);
    }
    return ni == nodes.size() && fr.size() == 1 && fr[0].second == root;
}

// prover ledger: values pointer + commitment root per obligation.
// `gen` (optional): a closure that REGENERATES the column values on demand.
// Columns with a generator can be dropped (vals cleared) after their claims are
// recorded -- prove_class rebuilds them transiently per use.  Used for the zk
// Libra blind columns (pure PRNG streams) and the logUp helper columns
// (recomputable from the member witness columns), which otherwise hold tens of
// GB across the whole proof just to be opened once at the end.
struct PLedger {
    // dropped-column regenerator: WRITES the n column values into the caller's
    // buffer (a per-class scratch reused across uses -- regenerating into a
    // fresh std::vector cost a 100s-of-MB mmap + page-fault + free cycle PER
    // USE at scale, which dominated the big batch classes)
    typedef std::function<void(gl_t*, size_t)> Gen;
    // DEVICE regenerator: fills a DEVICE buffer with the same n values (kernel
    // launches only) -- skips the host materialization + PCIe upload entirely.
    // Optional; when present it MUST produce bit-identical values to gen.
    typedef std::function<void(gl_t*, size_t)> DGen;
    struct Ent { const std::vector<gl_t>* vals; Hash root; uint32_t zid; gl_t y;
                 uint64_t sseed; Gen gen; DGen dgen; };
    struct Cls { uint32_t v; std::vector<std::vector<gl_t>> pts; std::vector<Ent> ents; };
    std::vector<Cls> cls;
    void add(const std::vector<gl_t>* vals, const Hash& root, const std::vector<gl_t>& z,
             gl_t y, uint64_t sseed = 0, Gen gen = {}, DGen dgen = {}) {
        uint32_t vv = (uint32_t)z.size();
        for (auto& c : cls) if (c.v == vv) {
            uint32_t zid = 0;
            while (zid < c.pts.size() && c.pts[zid] != z) zid++;
            if (zid == c.pts.size()) c.pts.push_back(z);
            c.ents.push_back({vals, root, zid, y, sseed, std::move(gen), std::move(dgen)});
            return;
        }
        cls.push_back({vv, {z}, {{vals, root, 0, y, sseed, std::move(gen), std::move(dgen)}}});
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

// ---- fused per-class reduction over STACKED per-point buffers ----
// The resident (!strCol && !strG) path used to launch one kernel per (point,
// round): T up to ~200 points x ~10 rounds x 3 kernels was pure launch
// latency at small dims.  These kernels process all T points in ONE launch
// per operation; every sum is the same exact field value (mod-p addition is
// associative/commutative), so messages and transcripts are unchanged.
__global__ void p3bo_gbuild_kernel(const gl_t* const* ecol, const gl_t* ew,
                                   const uint32_t* estart, gl_t* Gs,
                                   uint32_t T, uint32_t N) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= T * N) return;
    uint32_t t = j / N, i = j % N;
    gl_t acc = 0;
    for (uint32_t e = estart[t]; e < estart[t + 1]; e++)
        acc = gl_add(acc, gl_mul(ew[e], ecol[e][i]));
    Gs[j] = acc;
}
__global__ void p3bo_eqs_kernel(const gl_t* zs, gl_t* Es, uint32_t T, uint32_t v,
                                uint32_t N) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= T * N) return;
    uint32_t t = j / N, b = j % N;
    const gl_t* z = zs + (size_t)t * v;
    gl_t p = 1ULL;
    for (uint32_t i = 0; i < v; i++)
        p = gl_mul(p, (b & (1u << i)) ? z[i] : gl_sub(1ULL, z[i]));
    Es[j] = p;
}
__global__ void p3bo_msgs_kernel(const gl_t* Gs, const gl_t* Es, uint32_t L,
                                 uint32_t half, uint32_t T,
                                 gl_t* b0, gl_t* b1, gl_t* b2) {
    __shared__ gl_t s0[256], s1[256], s2[256];
    uint32_t tid = threadIdx.x; gl_t l0 = 0, l1 = 0, l2 = 0;
    size_t tot = (size_t)T * half;
    for (size_t j = (size_t)blockIdx.x * blockDim.x + tid; j < tot;
         j += (size_t)gridDim.x * blockDim.x) {
        size_t t = j / half, i = j % half;
        const gl_t* c = Gs + t * L; const gl_t* w = Es + t * L;
        gl_t cl = c[2*i], ch = c[2*i+1], wl = w[2*i], wh = w[2*i+1];
        l0 = gl_add(l0, gl_mul(cl, wl)); l1 = gl_add(l1, gl_mul(ch, wh));
        gl_t c2 = gl_sub(gl_add(ch, ch), cl), w2 = gl_sub(gl_add(wh, wh), wl);
        l2 = gl_add(l2, gl_mul(c2, w2));
    }
    s0[tid] = l0; s1[tid] = l1; s2[tid] = l2; __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s0[tid] = gl_add(s0[tid], s0[tid+s]);
                       s1[tid] = gl_add(s1[tid], s1[tid+s]);
                       s2[tid] = gl_add(s2[tid], s2[tid+s]); }
        __syncthreads();
    }
    if (tid == 0) { b0[blockIdx.x] = s0[0]; b1[blockIdx.x] = s1[0]; b2[blockIdx.x] = s2[0]; }
}
__global__ void p3bo_binds_kernel(const gl_t* Gs, const gl_t* Es, gl_t* nG, gl_t* nE,
                                  uint32_t L, uint32_t half, uint32_t T, gl_t a) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= T * half) return;
    uint32_t t = j / half, i = j % half;
    const gl_t* g = Gs + (size_t)t * L; const gl_t* e = Es + (size_t)t * L;
    nG[(size_t)t * half + i] = gl_add(g[2*i], gl_mul(a, gl_sub(g[2*i+1], g[2*i])));
    nE[(size_t)t * half + i] = gl_add(e[2*i], gl_mul(a, gl_sub(e[2*i+1], e[2*i])));
}

// device coeffs -> device codeword (zero-padded forward NTT)
static inline gl_t* bo_encode_dev(const gl_t* d_vals, uint32_t N, uint32_t logM0, const P3Ntt& ntt) {
    uint32_t M0 = 1u << logM0;
    gl_t* d_in = p3bf::dmalloc(M0, "bo_encode:in");
    cudaMemsetAsync(d_in, 0, (size_t)M0 * 8, 0);
    cudaMemcpyAsync(d_in, d_vals, (size_t)N * 8, cudaMemcpyDeviceToDevice, 0);
    gl_t* d_out = p3bf::dmalloc(M0, "bo_encode:out");
    ntt.run(d_in, d_out, true);
    cudaFreeAsync(d_in, 0);
    return d_out;
}
// host coeffs -> device codeword (upload straight into the padded NTT input,
// no intermediate device column buffer)
static inline gl_t* bo_encode_host(const gl_t* hvals, uint32_t N, uint32_t logM0, const P3Ntt& ntt) {
    uint32_t M0 = 1u << logM0;
    gl_t* d_in = p3bf::dmalloc(M0, "bo_encode_h:in");
    cudaMemsetAsync(d_in, 0, (size_t)M0 * 8, 0);
    cudaMemcpyAsync(d_in, hvals, (size_t)N * 8, cudaMemcpyHostToDevice, 0);
    gl_t* d_out = p3bf::dmalloc(M0, "bo_encode_h:out");
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
static inline BatchProof prove_class(fs::Transcript& tr, const PLedger::Cls& C_in,
                                     uint32_t R, uint32_t Q, const std::string& label) {
    const uint32_t v = C_in.v, N = 1u << v, T0 = (uint32_t)C_in.pts.size();
    const uint32_t logM0 = v + R, M0 = 1u << logM0;
    BatchProof pf; pf.v = v; pf.R = R; pf.Q = Q;

    // zk: append the class BLINDER -- one fresh full-domain uniform committed
    // column with one claim at a fresh point.  It enters the mu-combination and
    // the rho-RLC like a real column, one-time-padding the combined word U (the
    // opening messages, fold openings and final word become uniform) and
    // blinding the reduction messages through its own term.
    PLedger::Cls C = C_in;
    std::vector<gl_t> blv;
    if (p3zkc::G.on) {
        p3zp::T zt(p3zp::g.bo_blinder);
        blv.resize(N);
        { uint64_t s = p3zkc::next_seed();
          if (p3zkc::G.blind_on) for (auto& x : blv) x = p3zkc::zprng(s); }
        uint64_t blseed = p3zkc::next_seed();
        pf.bl_root = p3zkc::salted_commit_root(blv, R, blseed);
        tr.absorb("bo-blr", pf.bl_root.data(), 32);
        std::vector<gl_t> zD(v); for (auto& x : zD) x = chal(tr);
        pf.bl_y = p3bf::eval_h(blv, p3bf::build_eq(zD));
        tr.absorb("bo-bly", &pf.bl_y, 8);
        C.pts.push_back(zD);
        C.ents.push_back({&blv, pf.bl_root, T0, pf.bl_y, blseed});
    }
    const uint32_t T = (uint32_t)C.pts.size();
    const size_t k = C.ents.size();

    // distinct columns by root (first appearance), entry -> column map
    std::vector<const std::vector<gl_t>*> hcols;
    std::vector<PLedger::Gen> hgen;
    std::vector<PLedger::DGen> hdgen;
    std::vector<Hash> droots;
    std::vector<uint64_t> dsseed;
    std::vector<uint32_t> colidx(k);
    for (size_t j = 0; j < k; j++) {
        uint32_t c = 0;
        while (c < droots.size() && !(droots[c] == C.ents[j].root)) c++;
        if (c == droots.size()) { droots.push_back(C.ents[j].root); hcols.push_back(C.ents[j].vals);
                                  hgen.push_back(C.ents[j].gen);
                                  hdgen.push_back(C.ents[j].dgen);
                                  dsseed.push_back(C.ents[j].sseed); }
        colidx[j] = c;
    }
    const uint32_t nc = (uint32_t)hcols.size();
    pf.nc = nc;
    // dropped-column materialization: regenerate transiently per use, into ONE
    // reusable scratch allocation (no per-use alloc/fault/free churn)
    std::vector<gl_t> genscratch;
    auto col_host = [&](uint32_t c) -> const std::vector<gl_t>* {
        if (hcols[c] && hcols[c]->size() == (size_t)N) return hcols[c];
        if (genscratch.size() != (size_t)N) genscratch.resize(N);
        hgen[c](genscratch.data(), (size_t)N);
        return &genscratch;
    };

    // env-gated ledger validation (debug): every claimed y must equal the
    // column's MLE at its point -- prints the first mismatch per class
    if (getenv("P3_LUDBG")) {
        for (size_t j = 0; j < k; j++) {
            const std::vector<gl_t>* cv = col_host(colidx[j]);
            gl_t yy = p3bf::eval_h(*cv, p3bf::build_eq(C.pts[C.ents[j].zid]));
            if (yy != C.ents[j].y)
                fprintf(stderr, "# LUDBG class v=%u ent %zu/%zu zid=%u root=%02x%02x.. y=%llu evals-to %llu\n",
                        v, j, k, C.ents[j].zid, C.ents[j].root[0], C.ents[j].root[1],
                        (unsigned long long)C.ents[j].y, (unsigned long long)yy);
        }
    }

    gl_t mu = chal(tr);

    // ---- memory modes (transcript-identical; only WHERE arrays live differs) --
    // strCol: column values stay on host, uploaded per use (vs all nc resident).
    // strG:   the per-point combined columns G_t park on HOST between rounds and
    //         the eq columns are REBUILT per round -- eq weights factor per
    //         variable and rounds bind variables in order, so the bound eq array
    //         equals  prod_{i<rd} eq(a_i, z_i)  *  eq-array(z[rd..]).
    const size_t colbytes = (size_t)N * 8;
    const bool strCol = (size_t)nc * colbytes > ((size_t)2 << 30);
    const bool strG   = (size_t)2 * T * colbytes > ((size_t)3 << 30);
    // strG sub-mode: park the G_t on DEVICE when they fit (only T columns --
    // the eq columns are rebuilt per round); fall back to host parking (with
    // per-round round-trips) only when even that exceeds the card.  The
    // budget is ACTUAL free device memory (the card is nearly idle in the
    // batch phase; a fixed 12 GB left a 16 GB class parking on the host and
    // breached the 41 GB cgroup cap at d=1024 zk).
    size_t devfree = 0, devtot = 0;
    if (strG) {
        cudaMemGetInfo(&devfree, &devtot);
        // the async mempool retains freed blocks (release threshold inf), so
        // driver-free understates: add the pool's reusable slack
        cudaMemPool_t pool; cudaDeviceGetDefaultMemPool(&pool, 0);
        unsigned long long resv = 0, used = 0;
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &resv);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
        devfree += (size_t)(resv > used ? resv - used : 0);
    }
    const bool strGdev = strG &&
        (size_t)(T + 2) * colbytes <= (size_t)(devfree * 0.92);
    if (p3bf::memlog())
        fprintf(stderr, "# bo class %s: v=%u nc=%u k=%zu T=%u colMB=%.1f strCol=%d strG=%d strGdev=%d devfreeGB=%.1f needGB=%.1f\n",
                label.c_str(), v, nc, k, T, colbytes / 1048576.0, strCol ? 1 : 0, strG ? 1 : 0,
                strGdev ? 1 : 0, devfree / 1073741824.0,
                (double)(T + 2) * colbytes / 1073741824.0);

    auto bo_now = [] { using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count(); };
    double tp0 = bo_now(), tpG = 0, tpR = 0, tpY = 0, tpF = 0;

    std::vector<gl_t> muw(k);
    { gl_t w = 1ULL; for (size_t j = 0; j < k; j++) { muw[j] = w; w = gl_mul(w, mu); } }
    const uint32_t NB = 256;
    gl_t* db0 = p3bf::dmalloc(NB, "bo:db0");
    gl_t* db1 = p3bf::dmalloc(NB, "bo:db1");
    gl_t* db2 = p3bf::dmalloc(NB, "bo:db2");
    std::vector<gl_t> hb0(NB), hb1(NB), hb2(NB);

    // resident distinct columns (small classes only)
    std::vector<gl_t*> dcol(strCol ? 0 : nc, nullptr);
    if (!strCol)
        for (uint32_t c = 0; c < nc; c++) {
            dcol[c] = p3bf::dmalloc(N, "bo:dcol");
            if (hdgen[c]) hdgen[c](dcol[c], (size_t)N);
            else cudaMemcpy(dcol[c], col_host(c)->data(), colbytes, cudaMemcpyHostToDevice);
        }
    gl_t* colbuf = strCol ? p3bf::dmalloc(N, "bo:colbuf") : nullptr;
    auto upload_col = [&](uint32_t c) -> const gl_t* {
        if (!strCol) return dcol[c];
        if (hdgen[c]) { hdgen[c](colbuf, (size_t)N); cudaDeviceSynchronize(); }
        else cudaMemcpy(colbuf, col_host(c)->data(), colbytes, cudaMemcpyHostToDevice);
        return colbuf;
    };
    auto eq_dev = [&](const gl_t* zh, uint32_t nv, gl_t* out) {
        gl_t* dz = p3bf::dmalloc(nv ? nv : 1, "bo:z");
        cudaMemcpy(dz, zh, (size_t)nv * 8, cudaMemcpyHostToDevice);
        uint32_t L = 1u << nv;
        p3bf::p3bf_eq_kernel<<<(L + 255) / 256, 256>>>(dz, out, nv, L);
        cudaFreeAsync(dz, 0);
    };

    // ---- per-point combined columns G_t (+ eq columns when resident) ----
    const bool fused = !strCol && !strG;   // stacked one-launch-per-op path
    std::vector<gl_t*> dG(T, nullptr), dEq(T, nullptr);   // resident mode
    std::vector<std::vector<gl_t>> hG(strG ? T : 0);      // host-parked mode
    gl_t* dGs = nullptr; gl_t* dEs = nullptr;             // fused stacked buffers
    SumMsg msg0{0, 0, 0};                                 // round-0 message (streamed mode
                                                          // computes it during the build)
    if (fused) {
        // CSR of entries by point id, weights and resident column pointers
        std::vector<uint32_t> estart(T + 1, 0);
        for (size_t j = 0; j < k; j++) estart[C.ents[j].zid + 1]++;
        for (uint32_t t = 0; t < T; t++) estart[t + 1] += estart[t];
        std::vector<const gl_t*> ecol(k); std::vector<gl_t> ew(k);
        { std::vector<uint32_t> pos(estart.begin(), estart.end() - 1);
          for (size_t j = 0; j < k; j++) {
              uint32_t p = pos[C.ents[j].zid]++;
              ecol[p] = dcol[colidx[j]]; ew[p] = muw[j];
          } }
        const gl_t** d_ecol; cudaMallocAsync(&d_ecol, k * sizeof(gl_t*), 0);
        cudaMemcpy(d_ecol, ecol.data(), k * sizeof(gl_t*), cudaMemcpyHostToDevice);
        gl_t* d_ew = p3bf::dmalloc(k, "bo:ew");
        cudaMemcpy(d_ew, ew.data(), k * 8, cudaMemcpyHostToDevice);
        uint32_t* d_es; cudaMallocAsync(&d_es, (size_t)(T + 1) * 4, 0);
        cudaMemcpy(d_es, estart.data(), (size_t)(T + 1) * 4, cudaMemcpyHostToDevice);
        dGs = p3bf::dmalloc((size_t)T * N, "bo:Gs");
        p3bo_gbuild_kernel<<<((uint32_t)((size_t)T * N + 255)) / 256, 256>>>(
            d_ecol, d_ew, d_es, dGs, T, N);
        cudaFreeAsync((void*)d_ecol, 0); cudaFreeAsync(d_ew, 0); cudaFreeAsync(d_es, 0);
        std::vector<gl_t> zs((size_t)T * v);
        for (uint32_t t = 0; t < T; t++)
            memcpy(zs.data() + (size_t)t * v, C.pts[t].data(), (size_t)v * 8);
        gl_t* d_zs = p3bf::dmalloc(zs.size() ? zs.size() : 1, "bo:zs");
        cudaMemcpy(d_zs, zs.data(), zs.size() * 8, cudaMemcpyHostToDevice);
        dEs = p3bf::dmalloc((size_t)T * N, "bo:Es");
        p3bo_eqs_kernel<<<((uint32_t)((size_t)T * N + 255)) / 256, 256>>>(d_zs, dEs, T, v, N);
        cudaFreeAsync(d_zs, 0);
    } else if (!strG || strGdev) {
        // ALL G_t device-resident during the build: run COLUMN-major so each
        // distinct column is materialized (regen) + uploaded exactly ONCE and
        // axpy'd into every G_t that references it (was once per ENTRY).  Only
        // the accumulation ORDER inside each G_t changes -- exact field adds,
        // identical sums, identical transcript.  (upload_col's blocking
        // default-stream memcpy serializes against the prior axpy kernels, so
        // the shared colbuf reuse is safe.)
        for (uint32_t t = 0; t < T; t++) {
            dG[t] = p3bf::dmalloc(N, "bo:G");
            cudaMemsetAsync(dG[t], 0, colbytes, 0);
        }
        for (uint32_t c = 0; c < nc; c++) {
            const gl_t* src = nullptr;
            for (size_t j = 0; j < k; j++)
                if (colidx[j] == c) {
                    if (!src) src = upload_col(c);
                    p3bo_axpy_kernel<<<(N + 255) / 256, 256>>>(
                        dG[C.ents[j].zid], src, muw[j], N);
                }
        }
        if (strG) {                                    // strGdev: round-0 message now
            gl_t* eqbuf = p3bf::dmalloc(N, "bo:eqbuf");
            for (uint32_t t = 0; t < T; t++) {
                eq_dev(C.pts[t].data(), v, eqbuf);
                SumMsg m = bo_scmsg(dG[t], eqbuf, N / 2, db0, db1, db2, hb0, hb1, hb2);
                msg0.s0 = gl_add(msg0.s0, m.s0); msg0.s1 = gl_add(msg0.s1, m.s1);
                msg0.s2 = gl_add(msg0.s2, m.s2);
            }
            cudaFreeAsync(eqbuf, 0);
        } else {
            for (uint32_t t = 0; t < T; t++) {
                dEq[t] = p3bf::dmalloc(N, "bo:eq");
                eq_dev(C.pts[t].data(), v, dEq[t]);
            }
        }
    } else {
        // host-parked strG: the T G_t cannot all be device-resident, keep the
        // point-major build (park each G_t as it completes)
        gl_t* eqbuf = p3bf::dmalloc(N, "bo:eqbuf");
        for (uint32_t t = 0; t < T; t++) {
            gl_t* g = p3bf::dmalloc(N, "bo:G");
            cudaMemsetAsync(g, 0, colbytes, 0);
            for (size_t j = 0; j < k; j++)
                if (C.ents[j].zid == t)
                    p3bo_axpy_kernel<<<(N + 255) / 256, 256>>>(g, upload_col(colidx[j]), muw[j], N);
            eq_dev(C.pts[t].data(), v, eqbuf);
            SumMsg m = bo_scmsg(g, eqbuf, N / 2, db0, db1, db2, hb0, hb1, hb2);
            msg0.s0 = gl_add(msg0.s0, m.s0); msg0.s1 = gl_add(msg0.s1, m.s1);
            msg0.s2 = gl_add(msg0.s2, m.s2);
            hG[t].resize(N);
            cudaMemcpy(hG[t].data(), g, colbytes, cudaMemcpyDeviceToHost);
            cudaFreeAsync(g, 0);
        }
        cudaFreeAsync(eqbuf, 0);
    }
    p3bf::ckcuda("bo:G-build");
    tpG = bo_now();
    if (p3zp::on()) { p3zp::g.bo_G.ms += tpG - tp0; p3zp::g.bo_G.n++; }

    // ---- reduction sumcheck ----
    auto eq1 = [](gl_t a, gl_t z) {           // eq(a, z) over one variable
        return gl_add(gl_mul(a, z), gl_mul(gl_sub(1ULL, a), gl_sub(1ULL, z)));
    };
    std::vector<gl_t> rstar; rstar.reserve(v);
    std::vector<gl_t> pref(T, 1ULL);          // prod_{i<rd} eq(a_i, z_t[i])
    for (uint32_t rd = 0; rd < v; rd++) {
        uint32_t L = N >> rd, half = L / 2;
        SumMsg msg{0, 0, 0};
        if (fused) {
            const uint32_t NBm = 256;
            p3bo_msgs_kernel<<<NBm, 256>>>(dGs, dEs, L, half, T, db0, db1, db2);
            cudaMemcpy(hb0.data(), db0, NBm * 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(hb1.data(), db1, NBm * 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(hb2.data(), db2, NBm * 8, cudaMemcpyDeviceToHost);
            for (uint32_t i = 0; i < NBm; i++) {
                msg.s0 = gl_add(msg.s0, hb0[i]); msg.s1 = gl_add(msg.s1, hb1[i]);
                msg.s2 = gl_add(msg.s2, hb2[i]);
            }
        } else if (!strG) {
            for (uint32_t t = 0; t < T; t++) {
                SumMsg m = bo_scmsg(dG[t], dEq[t], half, db0, db1, db2, hb0, hb1, hb2);
                msg.s0 = gl_add(msg.s0, m.s0); msg.s1 = gl_add(msg.s1, m.s1);
                msg.s2 = gl_add(msg.s2, m.s2);
            }
        } else if (rd == 0) {
            msg = msg0;
        } else {
            gl_t* gbuf = strGdev ? nullptr : p3bf::dmalloc(L, "bo:gbuf");
            gl_t* ebuf = p3bf::dmalloc(L, "bo:ebuf");
            for (uint32_t t = 0; t < T; t++) {
                const gl_t* gsrc;
                if (strGdev) gsrc = dG[t];
                else {
                    cudaMemcpy(gbuf, hG[t].data(), (size_t)L * 8, cudaMemcpyHostToDevice);
                    gsrc = gbuf;
                }
                eq_dev(C.pts[t].data() + rd, v - rd, ebuf);
                SumMsg m = bo_scmsg(gsrc, ebuf, half, db0, db1, db2, hb0, hb1, hb2);
                msg.s0 = gl_add(msg.s0, gl_mul(pref[t], m.s0));
                msg.s1 = gl_add(msg.s1, gl_mul(pref[t], m.s1));
                msg.s2 = gl_add(msg.s2, gl_mul(pref[t], m.s2));
            }
            if (gbuf) cudaFreeAsync(gbuf, 0);
            cudaFreeAsync(ebuf, 0);
        }
        pf.rmsgs.push_back(msg); tr.absorb("bo-rm", &msg, sizeof msg);
        gl_t a = chal(tr); rstar.push_back(a);
        if (fused) {
            gl_t* nGs = p3bf::dmalloc((size_t)T * half, "bo:nGs");
            gl_t* nEs = p3bf::dmalloc((size_t)T * half, "bo:nEs");
            p3bo_binds_kernel<<<((uint32_t)((size_t)T * half + 255)) / 256, 256>>>(
                dGs, dEs, nGs, nEs, L, half, T, a);
            cudaFreeAsync(dGs, 0); cudaFreeAsync(dEs, 0);
            dGs = nGs; dEs = nEs;
        } else if (!strG) {
            for (uint32_t t = 0; t < T; t++) {
                gl_t *nG, *nE;
                nG = p3bf::dmalloc(half, "bo:nG"); nE = p3bf::dmalloc(half, "bo:nE");
                p3bf::p3bf_bind_kernel<<<(half + 255) / 256, 256>>>(dG[t], nG, half, a);
                p3bf::p3bf_bind_kernel<<<(half + 255) / 256, 256>>>(dEq[t], nE, half, a);
                cudaFreeAsync(dG[t], 0); cudaFreeAsync(dEq[t], 0);
                dG[t] = nG; dEq[t] = nE;
            }
        } else if (strGdev) {
            for (uint32_t t = 0; t < T; t++) {
                gl_t* nG = p3bf::dmalloc(half ? half : 1, "bo:nGd");
                p3bf::p3bf_bind_kernel<<<(half + 255) / 256, 256>>>(dG[t], nG, half, a);
                cudaFreeAsync(dG[t], 0); dG[t] = nG;
                pref[t] = gl_mul(pref[t], eq1(a, C.pts[t][rd]));
            }
        } else {
            gl_t* gbuf = p3bf::dmalloc(L, "bo:gbind");
            gl_t* nbuf = p3bf::dmalloc(half ? half : 1, "bo:gbind2");
            for (uint32_t t = 0; t < T; t++) {
                cudaMemcpy(gbuf, hG[t].data(), (size_t)L * 8, cudaMemcpyHostToDevice);
                p3bf::p3bf_bind_kernel<<<(half + 255) / 256, 256>>>(gbuf, nbuf, half, a);
                hG[t].resize(half);
                cudaMemcpy(hG[t].data(), nbuf, (size_t)half * 8, cudaMemcpyDeviceToHost);
                pref[t] = gl_mul(pref[t], eq1(a, C.pts[t][rd]));
            }
            cudaFreeAsync(gbuf, 0); cudaFreeAsync(nbuf, 0);
        }
    }
    if (fused) { cudaFreeAsync(dGs, 0); cudaFreeAsync(dEs, 0); }
    else if (!strG)
        for (uint32_t t = 0; t < T; t++) { cudaFreeAsync(dG[t], 0); cudaFreeAsync(dEq[t], 0); }
    else if (strGdev)
        for (uint32_t t = 0; t < T; t++) if (dG[t]) cudaFreeAsync(dG[t], 0);
    hG.clear(); hG.shrink_to_fit();
    p3bf::ckcuda("bo:reduction");
    tpR = bo_now();
    if (p3zp::on()) { p3zp::g.bo_red.ms += tpR - tpG; p3zp::g.bo_red.n++; }

    // ---- per-column terminal evals at r* ----
    gl_t* dEqr = p3bf::dmalloc(N, "bo:eqr");
    eq_dev(rstar.data(), v, dEqr);
    pf.ystar.resize(nc);
    for (uint32_t c = 0; c < nc; c++) {
        p3bf::p3bf_dot_kernel<<<NB, 256>>>(upload_col(c), dEqr, db0, N);
        cudaMemcpy(hb0.data(), db0, NB * 8, cudaMemcpyDeviceToHost);
        gl_t s = 0; for (uint32_t i = 0; i < NB; i++) s = gl_add(s, hb0[i]);
        pf.ystar[c] = s;
    }
    tr.absorb("bo-ys", pf.ystar.data(), (size_t)nc * 8);
    gl_t rho = chal(tr);

    // ---- RLC column U and its claimed eval ----
    gl_t* dU = p3bf::dmalloc(N, "bo:U"); cudaMemsetAsync(dU, 0, colbytes, 0);
    gl_t yU = 0, w = 1ULL;
    for (uint32_t c = 0; c < nc; c++) {
        p3bo_axpy_kernel<<<(N + 255) / 256, 256>>>(dU, upload_col(c), w, N);
        yU = gl_add(yU, gl_mul(w, pf.ystar[c]));
        w = gl_mul(w, rho);
    }
    p3bf::ckcuda("bo:rlc");
    tpY = bo_now();
    if (p3zp::on()) { p3zp::g.bo_ys.ms += tpY - tpR; p3zp::g.bo_ys.n++; }

    // ---- Basefold opening of U at r* (round 0 uncommitted) ----
    // strM: the retained per-round device trees would hold ~64*M0 bytes total
    // and the retained codewords 16*M0 -- stream instead: per round build a
    // chunked StreamTree (bit-identical root), PARK the round codeword on host,
    // and rebuild only the query-touched chunks for path extraction.
    const bool strM = M0 >= (1u << 24);
    const P3Ntt& ntt = p3bf::ntt_plan(logM0);
    auto plain_leaves = [](const gl_t* d_cwv) {
        return [d_cwv](size_t off, uint32_t len, uint8_t* out) {
            p3_merkle_leaf_kernel<<<(len + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS,
                                    P3_MERKLE_THREADS>>>(d_cwv + off, out, len);
        };
    };
    std::vector<gl_t*> d_cw(v + 1, nullptr);            // resident mode
    std::vector<std::vector<gl_t>> hcw(strM ? v : 0);   // host-parked round codewords
    std::vector<p3fri::StreamTree> strees(strM ? v : 0);
    std::vector<p3fri::DeviceMerkle> dtrees(strM ? 0 : v);   // [r] valid for r>=1
    gl_t* d_prev = bo_encode_dev(dU, N, logM0, ntt);
    if (!strM) d_cw[0] = d_prev;
    fs::Transcript ot(label);
    ot.absorb("z", rstar.data(), (size_t)v * 8);
    ot.absorb("y", &yU, 8);
    gl_t* d_c = dU;            // consumed by the sumcheck binds
    gl_t* d_w = dEqr;
    for (uint32_t r = 0; r < v; r++) {
        uint32_t half = (N >> r) / 2, Mr = M0 >> r;
        if (r > 0) {
            Hash root;
            if (strM) {
                strees[r] = p3fri::stream_tree_build(Mr, plain_leaves(d_prev));
                root = strees[r].root();
                hcw[r].resize(Mr);
                cudaMemcpy(hcw[r].data(), d_prev, (size_t)Mr * 8, cudaMemcpyDeviceToHost);
            } else {
                dtrees[r].build_dev(d_prev, Mr); cudaDeviceSynchronize();
                root = dtrees[r].root();
            }
            pf.roots.push_back(root); ot.absorb("root", root.data(), 32);
        }
        SumMsg msg = bo_scmsg(d_c, d_w, half, db0, db1, db2, hb0, hb1, hb2);
        pf.omsgs.push_back(msg); ot.absorb("sc", &msg, sizeof msg);
        gl_t a = p3bf::alpha_from(ot);
        gl_t *ncf, *nwf;
        ncf = p3bf::dmalloc(half ? half : 1, "bo:oc"); nwf = p3bf::dmalloc(half ? half : 1, "bo:ow");
        p3bf::p3bf_bind_kernel<<<(half + 255) / 256, 256>>>(d_c, ncf, half, a);
        p3bf::p3bf_bind_kernel<<<(half + 255) / 256, 256>>>(d_w, nwf, half, a);
        cudaFreeAsync(d_c, 0); cudaFreeAsync(d_w, 0); d_c = ncf; d_w = nwf;
        gl_t wr = gl_root_of_unity(logM0 - r), winv = gl_inv(wr), inv2 = gl_inv(2ULL);
        uint32_t fh = Mr / 2;
        gl_t* d_next = p3bf::dmalloc(fh, "bo:fold");
        p3bf::p3bf_fold_kernel<<<(fh + 255) / 256, 256>>>(d_prev, d_next, fh, winv, a, inv2);
        if (strM) cudaFreeAsync(d_prev, 0);
        else d_cw[r + 1] = d_next;
        d_prev = d_next;
    }
    cudaFreeAsync(d_c, 0); cudaFreeAsync(d_w, 0);
    pf.final_word.resize(1u << R);
    cudaMemcpy(pf.final_word.data(), d_prev, (size_t)(1u << R) * 8, cudaMemcpyDeviceToHost);
    if (strM) cudaFreeAsync(d_prev, 0);
    ot.absorb("final", pf.final_word.data(), pf.final_word.size() * 8);
    p3bf::ckcuda("bo:fold-loop");
    tpF = bo_now();
    if (p3zp::on()) { p3zp::g.bo_fold.ms += tpF - tpY; p3zp::g.bo_fold.n++; }

    // query coset indices (same chain rule as prove_eval_dev)
    std::vector<std::vector<uint32_t>> ccq(Q, std::vector<uint32_t>(v));
    for (uint32_t q = 0; q < Q; q++) {
        uint32_t c0 = (uint32_t)p3fri::idx_from(ot, M0 / 2), p = c0;
        for (uint32_t r = 0; r < v; r++) { uint32_t h = (M0 >> r) / 2, cc = p % h; ccq[q][r] = cc; p = cc; }
    }
    // rounds >= 1 openings against the fresh fold roots
    double zt_qr0 = p3zp::nowms();
    pf.queries.assign(Q, {});
    for (auto& x : pf.queries) x.rounds.resize(v > 0 ? v - 1 : 0);
    for (uint32_t r = 1; r < v; r++) {
        uint32_t h = (M0 >> r) / 2;
        std::vector<uint32_t> idxs(2 * Q);
        for (uint32_t q = 0; q < Q; q++) { idxs[2 * q] = ccq[q][r]; idxs[2 * q + 1] = ccq[q][r] + h; }
        std::vector<gl_t> vals(2 * Q);
        std::vector<std::vector<Hash>> paths;
        if (strM) {
            for (uint32_t i = 0; i < 2 * Q; i++) vals[i] = hcw[r][idxs[i]];
            const std::vector<gl_t>& hr = hcw[r];
            const size_t CH = std::min<size_t>((size_t)1 << strees[r].lch, hr.size());
            gl_t* chunkbuf = p3bf::dmalloc(CH, "bo:qchunk");
            paths = p3fri::stream_tree_paths(strees[r],
                [&](size_t off, uint32_t len, uint8_t* out) {
                    cudaMemcpy(chunkbuf, hr.data() + off, (size_t)len * 8, cudaMemcpyHostToDevice);
                    p3_merkle_leaf_kernel<<<(len + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS,
                                            P3_MERKLE_THREADS>>>(chunkbuf, out, len);
                }, idxs);
            cudaFreeAsync(chunkbuf, 0);
            hcw[r].clear(); hcw[r].shrink_to_fit();
        } else {
            uint32_t* d_idx; cudaMallocAsync(&d_idx, (size_t)2 * Q * 4, 0);
            cudaMemcpy(d_idx, idxs.data(), (size_t)2 * Q * 4, cudaMemcpyHostToDevice);
            gl_t* d_val; cudaMallocAsync(&d_val, (size_t)2 * Q * 8, 0);
            p3bf::p3bf_gather_kernel<<<(2 * Q + 255) / 256, 256>>>(d_cw[r], d_idx, d_val, 2 * Q);
            cudaMemcpy(vals.data(), d_val, (size_t)2 * Q * 8, cudaMemcpyDeviceToHost);
            cudaFreeAsync(d_idx, 0); cudaFreeAsync(d_val, 0);
            paths = dtrees[r].paths_batch(idxs);
        }
        for (uint32_t q = 0; q < Q; q++) {
            auto& ro = pf.queries[q].rounds[r - 1];
            ro.a = vals[2 * q]; ro.b = vals[2 * q + 1];
            ro.pa = paths[2 * q]; ro.pb = paths[2 * q + 1];
        }
    }
    // round-0 per-column SUBSET openings against the ORIGINAL commitments:
    // sorted-unique positions, one pruned Merkle node stream per column
    {
        if (p3zp::on()) { p3zp::g.bo_qr.ms += p3zp::nowms() - zt_qr0; p3zp::g.bo_qr.n++; }
        p3zp::T zt_q0(p3zp::g.bo_q0);
        if (colbuf) { cudaFreeAsync(colbuf, 0); colbuf = nullptr; }   // headroom for encodes
        std::vector<uint32_t> upos;
        for (uint32_t q = 0; q < Q; q++) { upos.push_back(ccq[q][0]); upos.push_back(ccq[q][0] + M0 / 2); }
        std::sort(upos.begin(), upos.end());
        upos.erase(std::unique(upos.begin(), upos.end()), upos.end());
        const uint32_t nu = (uint32_t)upos.size();
        uint32_t* d_idx; cudaMallocAsync(&d_idx, (size_t)nu * 4, 0);
        cudaMemcpy(d_idx, upos.data(), (size_t)nu * 4, cudaMemcpyHostToDevice);
        gl_t* d_val; cudaMallocAsync(&d_val, (size_t)nu * 8, 0);
        pf.q0c.assign(nc, {});
        const bool zk_salted = p3zkc::G.on && p3zkc::G.salt_on;
        // batched small-column path: encode a CHUNK of columns into one forest
        // buffer, hash all leaves, then run ONE internal-kernel series over the
        // concatenated subtrees + ONE path gather + ONE value gather for the
        // whole chunk.  Levels l < logM0 of the forest tree never cross a
        // column boundary (each column's subtree occupies a contiguous
        // pow2-aligned slice), so the extracted sibling hashes are bitwise
        // the per-column tree's -- the proof bytes are unchanged.  The
        // per-column loop below was pure launch/sync latency at small dims
        // (~0.8 ms x thousands of columns).
        if (!strM && !strCol) {
            const uint32_t CB = (uint32_t)std::min<size_t>(nc,
                std::max<size_t>(1, ((size_t)1 << 22) / M0));
            std::vector<uint32_t> gidx; std::vector<gl_t> gvals;
            for (uint32_t c0 = 0; c0 < nc; c0 += CB) {
                const uint32_t B = std::min(CB, nc - c0);
                double zq0 = p3zp::nowms();
                gl_t* dbuf = p3bf::dmalloc((size_t)B * M0, "bo:q0buf");
                gl_t* dbin = p3bf::dmalloc((size_t)B * M0, "bo:q0in");
                cudaMemsetAsync(dbin, 0, (size_t)B * M0 * 8, 0);
                for (uint32_t b = 0; b < B; b++)
                    cudaMemcpyAsync(dbin + (size_t)b * M0, dcol[c0 + b], (size_t)N * 8,
                                    cudaMemcpyDeviceToDevice, 0);
                ntt.run_batch(dbin, dbuf, B);
                cudaFreeAsync(dbin, 0);
                if (p3zp::on()) { cudaDeviceSynchronize();
                    p3zp::g.q0_enc.ms += p3zp::nowms() - zq0; p3zp::g.q0_enc.n++;
                    zq0 = p3zp::nowms(); }
                uint8_t* d0; cudaMallocAsync(&d0, (size_t)B * M0 * 32, 0);
                for (uint32_t b = 0; b < B; b++) {
                    if (zk_salted)
                        p3zkc::p3zkc_salted_leaf_off_kernel<<<(M0 + 255) / 256, 256>>>(
                            dbuf + (size_t)b * M0, dsseed[c0 + b],
                            d0 + (size_t)b * M0 * 32, 0, M0);
                    else
                        p3_merkle_leaf_kernel<<<(M0 + 255) / 256, 256>>>(
                            dbuf + (size_t)b * M0, d0 + (size_t)b * M0 * 32, M0);
                }
                p3fri::DeviceMerkle mk; mk.build_leaves(d0, B * M0);
                cudaDeviceSynchronize();
                p3bf::ckcuda("bo:q0-forest");
                if (p3zp::on()) { p3zp::g.q0_tree.ms += p3zp::nowms() - zq0;
                    p3zp::g.q0_tree.n++; zq0 = p3zp::nowms(); }
                gidx.assign((size_t)B * nu, 0);
                for (uint32_t b = 0; b < B; b++)
                    for (uint32_t i = 0; i < nu; i++)
                        gidx[(size_t)b * nu + i] = b * M0 + upos[i];
                auto allpaths = mk.paths_batch(gidx);
                uint32_t* d_gi; cudaMallocAsync(&d_gi, (size_t)B * nu * 4, 0);
                cudaMemcpy(d_gi, gidx.data(), (size_t)B * nu * 4, cudaMemcpyHostToDevice);
                gl_t* d_gv; cudaMallocAsync(&d_gv, (size_t)B * nu * 8, 0);
                p3bf::p3bf_gather_kernel<<<((uint32_t)(B * nu) + 255) / 256, 256>>>(
                    dbuf, d_gi, d_gv, B * nu);
                gvals.assign((size_t)B * nu, 0);
                cudaMemcpy(gvals.data(), d_gv, (size_t)B * nu * 8, cudaMemcpyDeviceToHost);
                cudaFreeAsync(d_gi, 0); cudaFreeAsync(d_gv, 0);
                mk.free_(); cudaFreeAsync(dbuf, 0);
                if (p3zp::on()) { p3zp::g.q0_path.ms += p3zp::nowms() - zq0;
                    p3zp::g.q0_path.n++; zq0 = p3zp::nowms(); }
                // per-column host assembly (salts, leaves, pruned subset
                // stream) -- columns are independent, results deterministic
                const int hthr = (int)std::min<size_t>(std::max<size_t>(B / 8, 1),
#ifdef _OPENMP
                                                       (size_t)omp_get_max_threads());
#else
                                                       (size_t)1);
#endif
                #pragma omp parallel for schedule(dynamic) if (hthr > 1) num_threads(hthr)
                for (uint32_t b = 0; b < B; b++) {
                    const uint32_t c = c0 + b;
                    BQ0C& oc = pf.q0c[c];
                    oc.vals.assign(gvals.begin() + (size_t)b * nu,
                                   gvals.begin() + (size_t)(b + 1) * nu);
                    std::vector<std::vector<Hash>> paths(nu);
                    for (uint32_t i = 0; i < nu; i++)
                        paths[i].assign(allpaths[(size_t)b * nu + i].begin(),
                                        allpaths[(size_t)b * nu + i].begin() + logM0);
                    std::vector<Hash> leaves(nu);
                    if (zk_salted) {
                        oc.salts.resize(nu);
                        for (uint32_t i = 0; i < nu; i++) {
                            oc.salts[i] = p3zkc::salt_of(dsseed[c], upos[i]);
                            leaves[i] = p3zkc::salted_leaf(oc.vals[i], oc.salts[i]);
                        }
                    } else {
                        for (uint32_t i = 0; i < nu; i++)
                            leaves[i] = p3fri::leaf_hash(oc.vals[i]);
                    }
                    oc.nodes = subset_prove(logM0, upos, leaves, paths);
                }
                if (p3zp::on()) { p3zp::g.q0_host.ms += p3zp::nowms() - zq0;
                    p3zp::g.q0_host.n++; }
            }
        } else
        for (uint32_t c = 0; c < nc; c++) {
            double zq0 = p3zp::nowms();
            gl_t* d_cwc;
            if (strCol && hdgen[c]) {
                gl_t* dtmp = p3bf::dmalloc(N, "bo:q0gen");
                hdgen[c](dtmp, (size_t)N);
                d_cwc = bo_encode_dev(dtmp, N, logM0, ntt);
                cudaFreeAsync(dtmp, 0);
            } else {
                d_cwc = strCol ? bo_encode_host(col_host(c)->data(), N, logM0, ntt)
                               : bo_encode_dev(dcol[c], N, logM0, ntt);
            }
            if (p3zp::on()) { cudaDeviceSynchronize();
                p3zp::g.q0_enc.ms += p3zp::nowms() - zq0; p3zp::g.q0_enc.n++;
                zq0 = p3zp::nowms(); }
            std::vector<std::vector<Hash>> paths;
            BQ0C& oc = pf.q0c[c];
            oc.vals.resize(nu);
            // commit-time retained tree top: rebuild only the query-touched
            // subtrees (path bytes identical to the full-tree rebuild)
            auto rit = p3fri::rettree_reg().find(droots[c]);
            const bool haveret = rit != p3fri::rettree_reg().end();
            if (zk_salted) {
                uint64_t ss = dsseed[c];
                auto sleaves = [&](size_t off, uint32_t len, uint8_t* out) {
                    p3zkc::p3zkc_salted_leaf_off_kernel<<<(len + 255) / 256, 256>>>(
                        d_cwc + off, ss, out, off, len);
                };
                if (haveret) {
                    paths = p3fri::retained_tree_paths(rit->second, sleaves, upos);
                } else if (strM) {
                    paths = p3fri::stream_tree_paths_onepass(M0, sleaves, upos);
                } else {
                    p3zkc::SaltedDevMerkle mk; mk.build_dev(d_cwc, M0, ss); cudaDeviceSynchronize();
                    paths = mk.paths_batch(upos);
                    mk.free_();
                }
                oc.salts.resize(nu);
                for (uint32_t i = 0; i < nu; i++) oc.salts[i] = p3zkc::salt_of(ss, upos[i]);
            } else {
                if (haveret) {
                    paths = p3fri::retained_tree_paths(rit->second, plain_leaves(d_cwc), upos);
                } else if (strM) {
                    paths = p3fri::stream_tree_paths_onepass(M0, plain_leaves(d_cwc), upos);
                } else {
                    p3fri::DeviceMerkle mk; mk.build_dev(d_cwc, M0); cudaDeviceSynchronize();
                    paths = mk.paths_batch(upos);
                    mk.free_();
                }
            }
            if (haveret) p3fri::rettree_reg().erase(rit);   // last use of this root
            if (p3zp::on()) { cudaDeviceSynchronize();
                p3zp::g.q0_tree.ms += p3zp::nowms() - zq0; p3zp::g.q0_tree.n++;
                zq0 = p3zp::nowms(); }
            p3bf::p3bf_gather_kernel<<<(nu + 255) / 256, 256>>>(d_cwc, d_idx, d_val, nu);
            cudaMemcpy(oc.vals.data(), d_val, (size_t)nu * 8, cudaMemcpyDeviceToHost);
            std::vector<Hash> leaves(nu);
            for (uint32_t i = 0; i < nu; i++)
                leaves[i] = zk_salted ? p3zkc::salted_leaf(oc.vals[i], oc.salts[i])
                                      : p3fri::leaf_hash(oc.vals[i]);
            oc.nodes = subset_prove(logM0, upos, leaves, paths);
            cudaFreeAsync(d_cwc, 0);
            if (p3zp::on()) { p3zp::g.q0_host.ms += p3zp::nowms() - zq0;
                p3zp::g.q0_host.n++; }
        }
        cudaFreeAsync(d_idx, 0); cudaFreeAsync(d_val, 0);
    }
    if (!strM) {
        for (uint32_t r = 0; r <= v; r++) if (d_cw[r]) cudaFreeAsync(d_cw[r], 0);
        for (uint32_t r = 1; r < v; r++) dtrees[r].free_();
    }
    for (auto p : dcol) cudaFreeAsync(p, 0);
    if (colbuf) cudaFreeAsync(colbuf, 0);
    cudaFreeAsync(db0, 0); cudaFreeAsync(db1, 0); cudaFreeAsync(db2, 0);
    cudaDeviceSynchronize();
    p3bf::ckcuda("bo:queries");
    if (p3bf::memlog() && (size_t)N * 8 >= ((size_t)1 << 22))
        fprintf(stderr, "# bo class %s timing: G=%.0f red=%.0f ys+rlc=%.0f fold=%.0f q=%.0f ms\n",
                label.c_str(), tpG - tp0, tpR - tpG, tpY - tpR, tpF - tpY, bo_now() - tpF);
    return pf;
}

// ==================== verifier ====================
static inline bool verify_class(fs::Transcript& tr, const VLedger::Cls& C_in, const BatchProof& pf,
                                uint32_t Q_pub, uint32_t R_pub, const std::string& label,
                                const char** why) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    const uint32_t v = C_in.v;
    const uint32_t logM0 = v + R_pub, M0 = 1u << logM0;
    if (pf.v != v || pf.R != R_pub || pf.Q != Q_pub) return fail("batch params");

    // zk: mirror the class blinder (root/claim from the proof, fresh point)
    VLedger::Cls C = C_in;
    if (p3zkc::G.on) {
        tr.absorb("bo-blr", pf.bl_root.data(), 32);
        std::vector<gl_t> zD(v); for (auto& x : zD) x = chal(tr);
        gl_t by = pf.bl_y;
        tr.absorb("bo-bly", &by, 8);
        C.ents.push_back({pf.bl_root, (uint32_t)C.pts.size(), by});
        C.pts.push_back(zD);
    }
    const uint32_t T = (uint32_t)C.pts.size();
    const size_t k = C.ents.size();

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
    if (pf.queries.size() != Q_pub || pf.q0c.size() != nc) return fail("batch query count");

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

    // query round-0 positions (transcript draw order unchanged), deduped
    std::vector<uint32_t> c0q(Q_pub);
    for (uint32_t q = 0; q < Q_pub; q++) c0q[q] = (uint32_t)p3fri::idx_from(ot, M0 / 2);
    std::vector<uint32_t> upos;
    for (uint32_t q = 0; q < Q_pub; q++) { upos.push_back(c0q[q]); upos.push_back(c0q[q] + M0 / 2); }
    std::sort(upos.begin(), upos.end());
    upos.erase(std::unique(upos.begin(), upos.end()), upos.end());
    const uint32_t nu = (uint32_t)upos.size();
    auto slot_of = [&](uint32_t pos) {
        return (uint32_t)(std::lower_bound(upos.begin(), upos.end(), pos) - upos.begin());
    };
    // authenticate every distinct column's subset opening against its root
    {
        const bool zk_salted = p3zkc::G.on && p3zkc::G.salt_on;
        for (uint32_t cc = 0; cc < nc; cc++) {
            const BQ0C& oc = pf.q0c[cc];
            if (oc.vals.size() != nu) return fail("batch q0 size");
            if (zk_salted && oc.salts.size() != nu) return fail("batch q0 salt count");
            std::vector<Hash> leaves(nu);
            for (uint32_t i = 0; i < nu; i++)
                leaves[i] = zk_salted ? p3zkc::salted_leaf(oc.vals[i], oc.salts[i])
                                      : p3fri::leaf_hash(oc.vals[i]);
            if (!subset_verify(logM0, upos, leaves, oc.nodes, droots[cc]))
                return fail("batch q0 merkle");
        }
    }
    for (uint32_t q = 0; q < Q_pub; q++) {
        uint32_t c0 = c0q[q];
        uint32_t p = c0;
        for (uint32_t r = 0; r < v; r++) {
            uint32_t half = (M0 >> r) / 2, c = p % half;
            gl_t a, b;
            if (r == 0) {
                a = 0; b = 0;
                uint32_t sa = slot_of(c), sb = slot_of(c + half);
                for (uint32_t cc = 0; cc < nc; cc++) {
                    a = gl_add(a, gl_mul(rhop[cc], pf.q0c[cc].vals[sa]));
                    b = gl_add(b, gl_mul(rhop[cc], pf.q0c[cc].vals[sb]));
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
    for (auto& z : pf.q0c)
        s += z.vals.size() * 8 + z.salts.size() * 32 + z.nodes.size() * 32;
    if (p3zkc::G.on) s += 40;                               // blinder root + claim
    return s;
}

} // namespace p3bo
