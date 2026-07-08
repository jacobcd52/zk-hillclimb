// P3 GKR fractional-sum tree over Goldilocks (logUp-GKR, Papini-Habock style).
//
// Statement: given leaf fraction pairs (p_i, q_i), i in [0, 2^L), all q_i != 0,
// the published root (P, Q) satisfies  P/Q = sum_i p_i/q_i.  The tree combines
// pairs   (p', q') = (p1*q2 + p2*q1, q1*q2)   level by level (so Q = prod q_i
// and P = sum_i p_i * prod_{j != i} q_j -- the fraction sum with common
// denominator).  NOTHING in the tree is committed: the root is published and a
// chain of L layer-reduction sumchecks (one per level, cubic rounds, standard
// GKR) reduces it to claims on the LEAF polynomials p~(.), q~(.) at a random
// point, which the CALLER binds to its own committed/public data.
//
// Layer h reduction (levels: h = L root .. 0 leaves; level h has 2^(L-h)
// nodes = polys over L-h variables, LSB-first, children of node y at 2y/2y+1
// i.e. child-selector = variable 0):
//   claims (cp, cq) at z in F^(L-h)  on level h;   lam = chal
//   sumcheck over y of  eq(z,y) * [ p(0,y)q(1,y) + p(1,y)q(0,y) + lam*q(0,y)q(1,y) ]
//   (claim0 = cp + lam*cq; degree 3, 4-point messages), terminal publishes the
//   four level-(h-1) claims p0,p1,q0,q1 at (0,r),(1,r);  mu = chal combines to
//   (cp', cq') = (lerp(p0,p1;mu), lerp(q0,q1;mu)) at z' = (mu, r).
// Root layer (0 rounds): the two published children must satisfy
//   p0*q1 + p1*q0 == P  and  q0*q1 == Q  (checked directly, no lam).
// Last layer: combine_last=true draws the final mu as well (single-point leaf
// claims at rfin, |rfin| = L); combine_last=false stops at the four separate
// claims at (0,rfin),(1,rfin), |rfin| = L-1 -- used by the zk mode where the
// leaf LSB splits real data (b0=0) from interleaved mask leaves (b0=1).
//
// Soundness: per layer, a false claim forces (whp over lam / the round
// challenges / mu) a false claim on the next level (sumcheck soundness + the
// degree-1 lerp bound); the chain ends at leaf claims the caller must check.
// The tree itself adds NO commitments and NO hiding requirement of its own;
// zk callers make level 1 uniform by interleaving committed uniform mask
// leaves (see p3_logup v3).
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_scgpu.cuh"
#include "fs_transcript.hpp"

namespace p3gkr {

struct Msg4 { gl_t s0, s1, s2, s3; };
// device functor of the layer-reduction summand: cols = [E, A0, A1, B0, B1],
// par = [lam]:  E * (A0*B1 + A1*B0 + lam*B0*B1)
struct FGkrGpu {
    static __device__ gl_t eval(const gl_t* c, const gl_t* p) {
        return gl_mul(c[0], gl_add(gl_add(gl_mul(c[1], c[4]), gl_mul(c[2], c[3])),
                                   gl_mul(p[0], gl_mul(c[3], c[4]))));
    }
};
// layers with a sumcheck domain >= this run device-resident (byte-identical
// messages -- exact field sums in either place); below it the host loop wins
// on launch overhead
static const size_t GKR_GPU_MIN = (size_t)1 << 13;

// fused round kernels (2 launches per round: message + 5-column bind)
__global__ void p3gkr_msg_kernel(const gl_t* E, const gl_t* A0, const gl_t* A1,
                                 const gl_t* B0, const gl_t* B1, gl_t lam,
                                 gl_t* out, uint32_t half) {
    __shared__ gl_t sh[4 * 256];
    uint32_t tid = threadIdx.x;
    gl_t acc[4]; for (int t = 0; t < 4; t++) acc[t] = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < half;
         i += gridDim.x * blockDim.x) {
        gl_t e = E[2*i],  de = gl_sub(E[2*i+1],  E[2*i]);
        gl_t a0 = A0[2*i], da0 = gl_sub(A0[2*i+1], A0[2*i]);
        gl_t a1 = A1[2*i], da1 = gl_sub(A1[2*i+1], A1[2*i]);
        gl_t b0 = B0[2*i], db0 = gl_sub(B0[2*i+1], B0[2*i]);
        gl_t b1 = B1[2*i], db1 = gl_sub(B1[2*i+1], B1[2*i]);
        for (int t = 0; t < 4; t++) {
            acc[t] = gl_add(acc[t], gl_mul(e,
                        gl_add(gl_add(gl_mul(a0, b1), gl_mul(a1, b0)),
                               gl_mul(lam, gl_mul(b0, b1)))));
            if (t < 3) { e = gl_add(e, de); a0 = gl_add(a0, da0); a1 = gl_add(a1, da1);
                         b0 = gl_add(b0, db0); b1 = gl_add(b1, db1); }
        }
    }
    for (int t = 0; t < 4; t++) sh[t * 256 + tid] = acc[t];
    __syncthreads();
    for (uint32_t s = 128; s > 0; s >>= 1) {
        if (tid < s)
            for (int t = 0; t < 4; t++)
                sh[t * 256 + tid] = gl_add(sh[t * 256 + tid], sh[t * 256 + tid + s]);
        __syncthreads();
    }
    if (tid == 0) for (int t = 0; t < 4; t++) out[t * gridDim.x + blockIdx.x] = sh[t * 256];
}
__global__ void p3gkr_bind5_kernel(const gl_t* i0, const gl_t* i1, const gl_t* i2,
                                   const gl_t* i3, const gl_t* i4,
                                   gl_t* o0, gl_t* o1, gl_t* o2, gl_t* o3, gl_t* o4,
                                   uint32_t half, gl_t a) {
    uint32_t gi = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t k = gi / half, i = gi % half;
    if (k >= 5) return;
    const gl_t* in = k == 0 ? i0 : k == 1 ? i1 : k == 2 ? i2 : k == 3 ? i3 : i4;
    gl_t* out      = k == 0 ? o0 : k == 1 ? o1 : k == 2 ? o2 : k == 3 ? o3 : o4;
    out[i] = gl_add(in[2*i], gl_mul(a, gl_sub(in[2*i+1], in[2*i])));
}
// fraction-combine one level: (op,oq)[y] = (cp[2y]cq[2y+1] + cp[2y+1]cq[2y],
// cq[2y]cq[2y+1])
__global__ void p3gkr_comb_kernel(const gl_t* cp, const gl_t* cq,
                                  gl_t* op, gl_t* oq, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    op[i] = gl_add(gl_mul(cp[2*i], cq[2*i+1]), gl_mul(cp[2*i+1], cq[2*i]));
    oq[i] = gl_mul(cq[2*i], cq[2*i+1]);
}
// strided child split for the layer sumcheck columns
__global__ void p3gkr_split_kernel(const gl_t* cp, const gl_t* cq,
                                   gl_t* a0, gl_t* a1, gl_t* b0, gl_t* b1, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    a0[i] = cp[2*i]; a1[i] = cp[2*i+1]; b0[i] = cq[2*i]; b1[i] = cq[2*i+1];
}
// full-device threshold: above this leaf count the tree lives on the GPU
static const size_t GKR_FULLGPU_MIN = (size_t)1 << 17;
struct Lay { std::vector<Msg4> msgs; gl_t p0 = 0, p1 = 0, q0 = 0, q1 = 0; };
struct Proof { gl_t P = 0, Q = 0; std::vector<Lay> lay; };

static inline size_t sz_proof(const Proof& pf) {
    size_t s = 16;
    for (auto& l : pf.lay) s += 8 + l.msgs.size() * sizeof(Msg4) + 4 * 8;
    return s;
}

static inline gl_t chal(fs::Transcript& tr) {
    uint8_t b[32]; tr.challenge_bytes(b);
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v % GL_P;
}
static inline gl_t gneg(gl_t x) { return gl_sub(0ULL, x); }
static inline gl_t cubic_eval(gl_t s0, gl_t s1, gl_t s2, gl_t s3, gl_t t) {
    gl_t inv2 = gl_inv(2ULL), inv6 = gl_inv(6ULL);
    gl_t t1 = gl_sub(t, 1ULL), t2 = gl_sub(t, 2ULL), t3 = gl_sub(t, 3ULL);
    gl_t L0 = gneg(gl_mul(gl_mul(gl_mul(t1, t2), t3), inv6));
    gl_t L1 = gl_mul(gl_mul(gl_mul(t, t2), t3), inv2);
    gl_t L2 = gneg(gl_mul(gl_mul(gl_mul(t, t1), t3), inv2));
    gl_t L3 = gl_mul(gl_mul(gl_mul(t, t1), t2), inv6);
    return gl_add(gl_add(gl_mul(s0, L0), gl_mul(s1, L1)),
                  gl_add(gl_mul(s2, L2), gl_mul(s3, L3)));
}
static inline void bind_lsb(std::vector<gl_t>& f, gl_t a) {
    size_t h = f.size() / 2; std::vector<gl_t> nf(h);
    #pragma omp parallel for schedule(static) if (h >= 65536) num_threads(p3bf::nthr(h))
    for (size_t i = 0; i < h; i++)
        nf[i] = gl_add(f[2*i], gl_mul(a, gl_sub(f[2*i+1], f[2*i])));
    f = std::move(nf);
}
static inline uint32_t ilog2(size_t n) { uint32_t l = 0; while ((1ull << l) < n) l++; return l; }

// one layer-reduction sumcheck: E=eq(z,.), split child arrays A0/A1 (p side)
// B0/B1 (q side), summand E*(A0*B1 + A1*B0 + lam*B0*B1).  Returns challenges;
// arrays end bound to length 1 (the four terminal claims).
static inline std::vector<gl_t> sc_layer(fs::Transcript& tr, const char* tag,
                                         std::vector<gl_t>& E,
                                         std::vector<gl_t>& A0, std::vector<gl_t>& A1,
                                         std::vector<gl_t>& B0, std::vector<gl_t>& B1,
                                         gl_t lam, std::vector<Msg4>& msgs) {
    uint32_t v = ilog2(E.size());
    std::vector<gl_t> r;
    for (uint32_t rd = 0; rd < v; rd++) {
        size_t half = E.size() / 2;
        gl_t s[4] = {0, 0, 0, 0};
        const int P = half >= 4096 ? 128 : 1;
        std::vector<std::array<gl_t, 4>> part(P, {0, 0, 0, 0});
        #pragma omp parallel for schedule(static) if (P > 1) num_threads(p3bf::nthr(half * 8))
        for (int p = 0; p < P; p++) {
            size_t lo = half * p / P, hi = half * (p + 1) / P;
            std::array<gl_t, 4>& sp = part[p];
            for (size_t i = lo; i < hi; i++) {
                gl_t e = E[2*i],  de = gl_sub(E[2*i+1],  E[2*i]);
                gl_t a0 = A0[2*i], da0 = gl_sub(A0[2*i+1], A0[2*i]);
                gl_t a1 = A1[2*i], da1 = gl_sub(A1[2*i+1], A1[2*i]);
                gl_t b0 = B0[2*i], db0 = gl_sub(B0[2*i+1], B0[2*i]);
                gl_t b1 = B1[2*i], db1 = gl_sub(B1[2*i+1], B1[2*i]);
                for (int t = 0; t < 4; t++) {
                    gl_t val = gl_mul(e, gl_add(gl_add(gl_mul(a0, b1), gl_mul(a1, b0)),
                                                gl_mul(lam, gl_mul(b0, b1))));
                    sp[t] = gl_add(sp[t], val);
                    e = gl_add(e, de); a0 = gl_add(a0, da0); a1 = gl_add(a1, da1);
                    b0 = gl_add(b0, db0); b1 = gl_add(b1, db1);
                }
            }
        }
        for (int p = 0; p < P; p++)
            for (int t = 0; t < 4; t++) s[t] = gl_add(s[t], part[p][t]);
        Msg4 m{s[0], s[1], s[2], s[3]};
        msgs.push_back(m); tr.absorb(tag, &m, sizeof m);
        gl_t a = chal(tr); r.push_back(a);
        bind_lsb(E, a); bind_lsb(A0, a); bind_lsb(A1, a); bind_lsb(B0, a); bind_lsb(B1, a);
    }
    return r;
}

// prover.  p/q: leaf arrays, size 2^L (L >= 1), all q != 0 (checked via the
// root: Q == prod q; throws on Q == 0 -- resample upstream randomness).
// rfin: the final leaf point (size L if combine_last else L-1).
// cp/cq_out (combine_last): the single-point leaf claims p~(rfin), q~(rfin).
static inline Proof prove(fs::Transcript& tr, const char* tag,
                          std::vector<gl_t> p, std::vector<gl_t> q,
                          bool combine_last, std::vector<gl_t>& rfin,
                          gl_t* cp_out = nullptr, gl_t* cq_out = nullptr,
                          bool gpu = true,
                          gl_t* dP0 = nullptr, gl_t* dQ0 = nullptr, size_t Ndev = 0) {
    const uint32_t L = dP0 ? ilog2(Ndev) : ilog2(p.size());
    if (!dP0 && (p.size() != q.size() || p.size() != ((size_t)1 << L) || L < 1))
        throw std::runtime_error("p3gkr: leaf size");
    Proof pf;
    // tree: lvp[h]/lvq[h] = level-h pair arrays (h=0 leaves .. L root).
    // devtree: the big levels live ONLY on the device (dlp/dlq); the first
    // level at or below GKR_GPU_MIN is downloaded and the small levels (and
    // their layers, and the root) run on the host as usual.  dP0/dQ0: leaves
    // already device-resident (ownership transferred; frees like dmalloc'd
    // levels) -- skips the host materialization + upload entirely.
    const bool devtree = dP0 ? true : (gpu && p.size() >= GKR_FULLGPU_MIN);
    std::vector<std::vector<gl_t>> lvp(L + 1), lvq(L + 1);
    std::vector<gl_t*> dlp(L + 1, nullptr), dlq(L + 1, nullptr);
    uint32_t HB = 0;                           // first host-resident level
    if (devtree) {
        if (dP0) {
            dlp[0] = dP0; dlq[0] = dQ0;
        } else {
        size_t N0 = p.size();
        dlp[0] = p3bf::dmalloc(N0, "gkr:lv"); dlq[0] = p3bf::dmalloc(N0, "gkr:lv");
        cudaMemcpy(dlp[0], p.data(), N0 * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dlq[0], q.data(), N0 * 8, cudaMemcpyHostToDevice);
        std::vector<gl_t>().swap(p); std::vector<gl_t>().swap(q);
        }
        uint32_t h = 1;
        for (;; h++) {
            size_t n = (size_t)1 << (L - h);
            dlp[h] = p3bf::dmalloc(n, "gkr:lv"); dlq[h] = p3bf::dmalloc(n, "gkr:lv");
            p3gkr_comb_kernel<<<((uint32_t)n + 255) / 256, 256>>>(
                dlp[h-1], dlq[h-1], dlp[h], dlq[h], (uint32_t)n);
            if (n <= GKR_GPU_MIN) break;
        }
        HB = h;
        size_t nh = (size_t)1 << (L - HB);
        lvp[HB].resize(nh); lvq[HB].resize(nh);
        cudaMemcpy(lvp[HB].data(), dlp[HB], nh * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(lvq[HB].data(), dlq[HB], nh * 8, cudaMemcpyDeviceToHost);
        cudaFreeAsync(dlp[HB], 0); cudaFreeAsync(dlq[HB], 0);
        dlp[HB] = dlq[HB] = nullptr;
        p3bf::ckcuda("gkr:devtree");
    } else {
        lvp[0] = std::move(p); lvq[0] = std::move(q);
    }
    for (uint32_t h = HB + 1; h <= L; h++) {
        size_t n = (size_t)1 << (L - h);
        lvp[h].resize(n); lvq[h].resize(n);
        const std::vector<gl_t>& cp = lvp[h-1];
        const std::vector<gl_t>& cq = lvq[h-1];
        std::vector<gl_t>& op = lvp[h];
        std::vector<gl_t>& oq = lvq[h];
        #pragma omp parallel for schedule(static) if (n >= 65536) num_threads(p3bf::nthr(n))
        for (size_t y = 0; y < n; y++) {
            op[y] = gl_add(gl_mul(cp[2*y], cq[2*y+1]), gl_mul(cp[2*y+1], cq[2*y]));
            oq[y] = gl_mul(cq[2*y], cq[2*y+1]);
        }
    }
    pf.P = lvp[L][0]; pf.Q = lvq[L][0];
    if (pf.Q == 0) throw std::runtime_error("p3gkr: zero leaf denominator");
    gl_t rt[2] = {pf.P, pf.Q};
    tr.absorb(tag, rt, sizeof rt);

    std::vector<gl_t> z;                       // current claim point (level h)
    pf.lay.resize(L);
    for (uint32_t h = L; h >= 1; h--) {
        Lay& ly = pf.lay[L - h];
        const std::vector<gl_t>& cp = lvp[h-1];
        const std::vector<gl_t>& cq = lvq[h-1];
        std::vector<gl_t> r;
        if (h == L) {                          // root: children published directly
            ly.p0 = cp[0]; ly.p1 = cp[1]; ly.q0 = cq[0]; ly.q1 = cq[1];
        } else {
            gl_t lam = chal(tr);
            size_t n = (size_t)1 << (L - h);   // sumcheck domain = level h
            const bool dchild = devtree && (h - 1) < HB && dlp[h-1];
            if (gpu && n >= GKR_GPU_MIN) {
                gl_t* dz = p3bf::dmalloc(z.size(), "gkr:z");
                cudaMemcpy(dz, z.data(), z.size() * 8, cudaMemcpyHostToDevice);
                gl_t* dE = p3bf::dmalloc(n, "gkr:E");
                p3bf::p3bf_eq_kernel<<<((uint32_t)n + 255) / 256, 256>>>(
                    dz, dE, (uint32_t)z.size(), (uint32_t)n);
                cudaFreeAsync(dz, 0);
                // double-buffered fused rounds: 2 launches + one 8KB D2H per round
                gl_t* cur[5]; gl_t* nxt[5];
                cur[0] = dE;
                for (int j = 0; j < 4; j++) cur[j+1] = p3bf::dmalloc(n, "gkr:c");
                if (dchild) {                  // children split device-side
                    p3gkr_split_kernel<<<((uint32_t)n + 255) / 256, 256>>>(
                        dlp[h-1], dlq[h-1], cur[1], cur[2], cur[3], cur[4], (uint32_t)n);
                    cudaFreeAsync(dlp[h-1], 0); cudaFreeAsync(dlq[h-1], 0);
                    dlp[h-1] = dlq[h-1] = nullptr;
                } else {
                    std::vector<gl_t> A0(n), A1(n), B0(n), B1(n);
                    #pragma omp parallel for schedule(static) if (n >= 65536) num_threads(p3bf::nthr(n))
                    for (size_t y = 0; y < n; y++) {
                        A0[y] = cp[2*y]; A1[y] = cp[2*y+1];
                        B0[y] = cq[2*y]; B1[y] = cq[2*y+1];
                    }
                    const std::vector<gl_t>* hs[4] = {&A0, &A1, &B0, &B1};
                    for (int j = 0; j < 4; j++)
                        cudaMemcpy(cur[j+1], hs[j]->data(), n * 8, cudaMemcpyHostToDevice);
                }
                for (int j = 0; j < 5; j++) nxt[j] = p3bf::dmalloc(n / 2, "gkr:c2");
                const uint32_t NB = 256;
                gl_t* dout = p3bf::dmalloc((size_t)4 * NB, "gkr:msg");
                std::vector<gl_t> hout((size_t)4 * NB);
                uint32_t v = ilog2(n);
                size_t half = n;
                for (uint32_t rd = 0; rd < v; rd++) {
                    half /= 2;
                    p3gkr_msg_kernel<<<NB, 256>>>(cur[0], cur[1], cur[2], cur[3], cur[4],
                                                  lam, dout, (uint32_t)half);
                    cudaMemcpy(hout.data(), dout, (size_t)4 * NB * 8, cudaMemcpyDeviceToHost);
                    Msg4 m{0, 0, 0, 0}; gl_t* sm = &m.s0;
                    for (int t = 0; t < 4; t++)
                        for (uint32_t b = 0; b < NB; b++)
                            sm[t] = gl_add(sm[t], hout[(size_t)t * NB + b]);
                    ly.msgs.push_back(m); tr.absorb(tag, &m, sizeof m);
                    gl_t a = chal(tr); r.push_back(a);
                    p3gkr_bind5_kernel<<<((uint32_t)(5 * half) + 255) / 256, 256>>>(
                        cur[0], cur[1], cur[2], cur[3], cur[4],
                        nxt[0], nxt[1], nxt[2], nxt[3], nxt[4], (uint32_t)half, a);
                    for (int j = 0; j < 5; j++) std::swap(cur[j], nxt[j]);
                }
                gl_t term[4];
                for (int j = 0; j < 4; j++)
                    cudaMemcpy(&term[j], cur[j+1], 8, cudaMemcpyDeviceToHost);
                ly.p0 = term[0]; ly.p1 = term[1]; ly.q0 = term[2]; ly.q1 = term[3];
                for (int j = 0; j < 5; j++) { cudaFreeAsync(cur[j], 0); cudaFreeAsync(nxt[j], 0); }
                cudaFreeAsync(dout, 0);
                p3bf::ckcuda("gkr:layer");
            } else {
                std::vector<gl_t> A0(n), A1(n), B0(n), B1(n);
                for (size_t y = 0; y < n; y++) {
                    A0[y] = cp[2*y]; A1[y] = cp[2*y+1];
                    B0[y] = cq[2*y]; B1[y] = cq[2*y+1];
                }
                std::vector<gl_t> E = p3bf::build_eq(z);
                r = sc_layer(tr, tag, E, A0, A1, B0, B1, lam, ly.msgs);
                ly.p0 = A0[0]; ly.p1 = A1[0]; ly.q0 = B0[0]; ly.q1 = B1[0];
            }
        }
        gl_t tm[4] = {ly.p0, ly.p1, ly.q0, ly.q1};
        tr.absorb(tag, tm, sizeof tm);
        lvp[h].clear(); lvp[h].shrink_to_fit(); lvq[h].clear(); lvq[h].shrink_to_fit();
        if (h > 1 || combine_last) {
            gl_t mu = chal(tr);
            z.clear(); z.push_back(mu);
            z.insert(z.end(), r.begin(), r.end());
            if (h == 1 && combine_last) {
                rfin = z;
                if (cp_out) *cp_out = gl_add(ly.p0, gl_mul(mu, gl_sub(ly.p1, ly.p0)));
                if (cq_out) *cq_out = gl_add(ly.q0, gl_mul(mu, gl_sub(ly.q1, ly.q0)));
            }
        } else {
            rfin = r;                          // four separate claims at (0,r),(1,r)
        }
    }
    return pf;
}

// verifier.  On success: rfin as in prove; out4 = {p0,p1,q0,q1}, the leaf
// claims -- combine_last: out4[0]/out4[2] hold the combined (cp,cq) and
// out4[1]/out4[3] are unused (=0).  The CALLER must (a) check pf.Q != 0 and
// use (pf.P, pf.Q) in its multiset identity, (b) bind the leaf claims.
static inline bool verify(fs::Transcript& tr, const char* tag, uint32_t L,
                          const Proof& pf, bool combine_last,
                          std::vector<gl_t>& rfin, gl_t out4[4], const char** why) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (L < 1 || pf.lay.size() != L) return fail("gkr layer count");
    if (pf.Q == 0) return fail("gkr zero root Q");
    gl_t rt[2] = {pf.P, pf.Q};
    tr.absorb(tag, rt, sizeof rt);
    std::vector<gl_t> z;
    gl_t cp = 0, cq = 0;                       // current claims at z (level h)
    for (uint32_t h = L; h >= 1; h--) {
        const Lay& ly = pf.lay[L - h];
        std::vector<gl_t> r;
        if (h == L) {
            if (!ly.msgs.empty()) return fail("gkr root msgs");
            if (gl_add(gl_mul(ly.p0, ly.q1), gl_mul(ly.p1, ly.q0)) != pf.P)
                return fail("gkr root P");
            if (gl_mul(ly.q0, ly.q1) != pf.Q) return fail("gkr root Q");
        } else {
            gl_t lam = chal(tr);
            uint32_t v = L - h;
            if (ly.msgs.size() != v) return fail("gkr layer msgs");
            gl_t claim = gl_add(cp, gl_mul(lam, cq));
            for (uint32_t rd = 0; rd < v; rd++) {
                const Msg4& m = ly.msgs[rd];
                if (gl_add(m.s0, m.s1) != claim) return fail("gkr round claim");
                tr.absorb(tag, &m, sizeof m);
                gl_t a = chal(tr); r.push_back(a);
                claim = cubic_eval(m.s0, m.s1, m.s2, m.s3, a);
            }
            gl_t end = gl_mul(p3bf::eq_point(z, r),
                              gl_add(gl_add(gl_mul(ly.p0, ly.q1), gl_mul(ly.p1, ly.q0)),
                                     gl_mul(lam, gl_mul(ly.q0, ly.q1))));
            if (claim != end) return fail("gkr layer terminal");
        }
        gl_t tm[4] = {ly.p0, ly.p1, ly.q0, ly.q1};
        tr.absorb(tag, tm, sizeof tm);
        if (h > 1 || combine_last) {
            gl_t mu = chal(tr);
            cp = gl_add(ly.p0, gl_mul(mu, gl_sub(ly.p1, ly.p0)));
            cq = gl_add(ly.q0, gl_mul(mu, gl_sub(ly.q1, ly.q0)));
            z.clear(); z.push_back(mu);
            z.insert(z.end(), r.begin(), r.end());
            if (h == 1) { rfin = z; out4[0] = cp; out4[1] = 0; out4[2] = cq; out4[3] = 0; }
        } else {
            rfin = r;
            out4[0] = ly.p0; out4[1] = ly.p1; out4[2] = ly.q0; out4[3] = ly.q1;
        }
    }
    if (why) *why = "ok";
    return true;
}

// device-leaf entry point: leaves already on the GPU (ownership transferred)
static inline Proof prove_dev(fs::Transcript& tr, const char* tag,
                              gl_t* dP, gl_t* dQ, size_t N,
                              bool combine_last, std::vector<gl_t>& rfin,
                              gl_t* cp_out = nullptr, gl_t* cq_out = nullptr) {
    return prove(tr, tag, {}, {}, combine_last, rfin, cp_out, cq_out, true, dP, dQ, N);
}

} // namespace p3gkr
