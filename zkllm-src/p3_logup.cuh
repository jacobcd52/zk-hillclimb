// P3 logUp lookup argument over Goldilocks, committed via Basefold (hash PCS).
//
// Ports the logUp LOGIC of the curve-based zkob_lookup.cuh onto the p3 stack:
// no G1/IPA anywhere -- witness, multiplicity and helper-inverse columns are
// Basefold-committed, the rational-sum identity is enforced by two batched
// cubic sumchecks, and terminals are Basefold openings.
//
// Statement: every row i of the c-column witness (W_0[i],..,W_{c-1}[i]) is a
// row of the PUBLIC fixed table (T_0[j],..,T_{c-1}[j]).  Rows are gamma-combined
// (A_i = sum_k g^k W_k[i], Tc_j = sum_k g^k T_k[j]); logUp identity with a
// random beta:
//        sum_i 1/(A_i+beta)  ==  sum_j cnt_j/(Tc_j+beta)   ( = S )
// Prover commits cnt (multiplicities), hA_i = 1/(A_i+beta), hT_j = cnt_j/(Tc_j+beta).
// Both sides then reduce to sumchecks batched with an eq-weighted zero-check
// binding the helper columns to their defining relations:
//   A side: sum_b [ lamA*hA(b) + eq(zA,b)*( hA(b)*(A(b)+beta) - 1     ) ] = lamA*S
//   T side: sum_b [ lamT*hT(b) + eq(zT,b)*( hT(b)*(Tc(b)+beta) - cnt(b)) ] = lamT*S
// The T-side table value Tc~(rT) is recomputed by the VERIFIER from the public
// table (no table commitment needed -- "fixed tables").
//
// Padding: N and M must be powers of two; the caller pads witness rows with a
// legitimate table row (padding rows are counted in cnt -- harmless).
// Challenges are base-field (the whole p3 stack's stated GL2 upgrade applies).
// Non-ZK: masking of hA/hT/openings is the increment-2 hardening, as in p3pfc.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_zkc.cuh"
#include "p3_batchopen.cuh"
#include "p3_scgpu.cuh"
#include "p3_gkr.cuh"
#include "fs_transcript.hpp"

namespace p3lu {

// device functor of the batched cubic sumcheck term:
// cols = [H, V, E, D], par = [lam, beta]:  lam*H + E*(H*(V+beta) - D)
struct FLuGpu {
    static __device__ gl_t eval(const gl_t* c, const gl_t* p) {
        return gl_add(gl_mul(p[0], c[0]),
                      gl_mul(c[2], gl_sub(gl_mul(c[0], gl_add(c[1], p[1])), c[3])));
    }
};
// zk variant: + rho*(B0 + E*(B1 + E*B2)); c = [H,V,E,D,B0,B1,B2], p = [lam,beta,rho]
struct FLuGpuZk {
    static __device__ gl_t eval(const gl_t* c, const gl_t* p) {
        gl_t f = gl_add(gl_mul(p[0], c[0]),
                        gl_mul(c[2], gl_sub(gl_mul(c[0], gl_add(c[1], p[1])), c[3])));
        return gl_add(f, gl_mul(p[2],
                 gl_add(c[4], gl_mul(c[2], gl_add(c[5], gl_mul(c[2], c[6]))))));
    }
};
// device regen of the GKR mask-leaf stream columns (bit-identical to the host
// ledger regenerators: zprng_at is __host__ __device__ with the same values)
__global__ void p3lu_pmgen_kernel(gl_t* out, uint64_t sd, uint64_t sq, size_t n, int mo) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mo ? gl_mul(p3zkc::zprng_at(sd, i), p3zkc::zprng_at(sq, i)) : 0ULL;
}
__global__ void p3lu_qmgen_kernel(gl_t* out, uint64_t sd, size_t n, int mo) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mo ? p3zkc::zprng_at(sd, i) : 1ULL;
}

// zk GKR leaf construction on device (even = real/witness rows, odd = mask
// siblings) -- bit-identical to the host loop (zprng_at is __host__ __device__)
__global__ void p3lu_zkleaf_kernel(const gl_t* Am, gl_t beta, uint64_t pseed, uint64_t qseed,
                                   int mo, size_t NR, gl_t* LP, gl_t* LQ, size_t NM) {
    size_t x = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= NM) return;
    LP[2*x]   = x < NR ? 1ULL : 0ULL;
    LQ[2*x]   = gl_add(Am[x], beta);
    gl_t qm = mo ? p3zkc::zprng_at(qseed, x) : 1ULL;
    LP[2*x+1] = mo ? gl_mul(p3zkc::zprng_at(pseed, x), qm) : 0ULL;
    LQ[2*x+1] = qm;
}
// device build of the merged combined witness Am (section 23): pad member
// rows (j >= k) hold the gamma-combined table row 0 value a0; real member
// rows accumulate gamma-axpy'd member columns.  Same exact field ops as the
// host column-major build -- only the per-element accumulation ORDER differs
// (exact adds: identical values, transcript unchanged).
__global__ void p3lu_amfill_kernel(gl_t* Am, gl_t a0, uint32_t n, uint32_t g,
                                   uint32_t k, size_t NM) {
    size_t x = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= NM) return;
    uint32_t j = (uint32_t)((x >> n) & (((size_t)1 << g) - 1));
    Am[x] = j >= k ? a0 : 0ULL;
}
__global__ void p3lu_amaxpy_kernel(gl_t* Am, const gl_t* src, gl_t gpt,
                                   uint32_t n, uint32_t g, uint32_t j, size_t NAf) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NAf) return;
    size_t ex = idx >> n, i = idx & (((size_t)1 << n) - 1);
    size_t x = (ex << (n + g)) | ((size_t)j << n) | i;
    Am[x] = gl_add(Am[x], gl_mul(gpt, src[idx]));
}

// device block-partial sum of the sm stream (the A-side mask fraction sum)
__global__ void p3lu_smsum_kernel(uint64_t pseed, int mo, gl_t* out, size_t n) {
    __shared__ gl_t sh[256];
    uint32_t tid = threadIdx.x;
    gl_t acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + tid; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        acc = gl_add(acc, mo ? p3zkc::zprng_at(pseed, i) : 0ULL);
    sh[tid] = acc; __syncthreads();
    for (uint32_t s = 128; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = gl_add(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sh[0];
}

// device reduction of the blind-sum H = sum_b B0 + E*(B1 + E*B2) (block partials)
__global__ void p3lu_blindsum_kernel(const gl_t* b0, const gl_t* b1, const gl_t* b2,
                                     const gl_t* e, gl_t* out, size_t n) {
    __shared__ gl_t sh[256];
    uint32_t tid = threadIdx.x;
    gl_t acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + tid; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        gl_t ez = e[i];
        acc = gl_add(acc, gl_add(b0[i], gl_mul(ez, gl_add(b1[i], gl_mul(ez, b2[i])))));
    }
    sh[tid] = acc; __syncthreads();
    for (uint32_t s = 128; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = gl_add(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sh[0];
}

// perf counters (profiling only; no protocol effect)
struct LuStats { double ms = 0, commit_ms = 0, sc_ms = 0, cnt_ms = 0, inv_ms = 0,
                 ev_ms = 0, scg_ms = 0; long calls = 0, commits = 0; };
static LuStats g_lustats;

// Free each member's lookup-index vector after its group is proven (the
// indices feed only the multiplicity count).  Off by default because it
// mutates the caller's witness; the scale bench opts in.
static bool g_free_idx = false;
static inline double lu_now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---- committed column: values + Basefold root + codeword (kept for opening) ----
// zk mode: v is the AUGMENTED array [real | mask] (p3_zkc mechanism 1), vreal
// the log2 of the real prefix, sseed the salt seed of the hiding Merkle leaves.
struct Col {
    std::vector<gl_t> v;
    p3fri::Hash root;
    std::vector<gl_t> cw;
    uint32_t vreal = 0;
    uint64_t sseed = 0;
    // ---- compact (dropped) storage, design doc section 22 ----
    // After a column's last direct use the owner may compact_col() it: the
    // real region is PACKED (sign-magnitude small ints), a fresh mask region
    // is dropped to its PRNG seed, a linked mask region is retained raw, and
    // v is freed.  mat_col_into() reproduces v bit-identically on demand (the
    // opening ledger resolver and the lookup flush read through it).
    p3zkc::Packed pk;              // packed real region [0, mstart)
    std::vector<gl_t> cmask;       // retained mask region (linked masks only)
    uint64_t mseed = 0;            // fresh-mask chain seed (regenerable)
    size_t mstart = 0, ctot = 0;   // real-region end, full committed length
};
// compact a committed column in place (no transcript effect: same values are
// reproduced at every later read).  False = kept raw (small, or wide values).
static inline bool compact_col(Col& c) {
    if (c.pk.on || c.v.empty()) return false;
    size_t ms = c.mstart ? c.mstart : c.v.size();
    if (!p3zkc::pack_ints(c.v.data(), ms, c.pk)) return false;
    if (ms < c.v.size() && !c.mseed)
        c.cmask.assign(c.v.begin() + ms, c.v.end());
    c.ctot = c.v.size();
    std::vector<gl_t>().swap(c.v);
    std::vector<gl_t>().swap(c.cw);
    return true;
}
// reproduce the committed column into out[0..n) (bit-identical to the
// original v, zero-extended past ctot)
static inline void mat_col_into(const Col& c, gl_t* out, size_t n) {
    if (!c.pk.on) throw std::runtime_error("p3lu: mat_col_into on raw column");
    if (n < c.ctot) throw std::runtime_error("p3lu: mat_col_into short buffer");
    p3zkc::unpack_ints(c.pk, out);
    size_t mlen = c.ctot - c.pk.n;
    if (mlen) {
        if (c.mseed) p3zkc::zprng_fill(c.mseed, out + c.pk.n, mlen);
        else if (!c.cmask.empty()) memcpy(out + c.pk.n, c.cmask.data(), mlen * 8);
        else memset(out + c.pk.n, 0, mlen * 8);          // masks-off control
    }
    if (n > c.ctot) memset(out + c.ctot, 0, (n - c.ctot) * 8);
}
// reproduce values [off, off+len) of a compacted committed column -- bit-
// identical to the corresponding mat_col_into slice.  Lets consumers stream
// chunks of a compacted column without materializing all of it (the huge
// product-domain zero-checks read through this).
static inline void mat_col_range(const Col& c, size_t off, gl_t* out, size_t len) {
    if (!c.pk.on) throw std::runtime_error("p3lu: mat_col_range on raw column");
    const size_t end = off + len;
    if (off < c.pk.n)
        p3zkc::unpack_ints_range(c.pk, off, std::min(end, c.pk.n) - off, out);
    if (end > c.pk.n && off < c.ctot) {
        size_t lo = std::max(off, c.pk.n), hi = std::min(end, c.ctot);
        gl_t* dst = out + (lo - off);
        if (c.mseed) p3zkc::zprng_fill_at(c.mseed, lo - c.pk.n, dst, hi - lo);
        else if (!c.cmask.empty()) memcpy(dst, c.cmask.data() + (lo - c.pk.n), (hi - lo) * 8);
        else memset(dst, 0, (hi - lo) * 8);
    }
    if (end > c.ctot) {
        size_t z0 = std::max(off, c.ctot);
        memset(out + (z0 - off), 0, (end - z0) * 8);
    }
}
// mat_col_range DIRECTLY INTO a device buffer: packed bytes upload + device
// unpack for the real region, jump-ahead device PRNG chain for a seeded mask
// region -- bit-identical values with 2-8x less PCIe than staging raw
// elements through the host
static inline void mat_col_range_dev(const Col& c, size_t off, gl_t* dout, size_t len) {
    if (!c.pk.on) throw std::runtime_error("p3lu: mat_col_range_dev on raw column");
    const size_t end = off + len;
    if (off < c.pk.n)
        p3zkc::unpack_ints_dev(c.pk, off, std::min(end, c.pk.n) - off, dout);
    if (end > c.pk.n && off < c.ctot) {
        size_t lo = std::max(off, c.pk.n), hi = std::min(end, c.ctot);
        gl_t* dst = dout + (lo - off);
        if (c.mseed) {
            static const p3zkc::LcgLadder L = p3zkc::lcg_ladder();
            p3zkc_lcgchain_kernel<<<(uint32_t)((hi - lo + 255) / 256), 256>>>(
                dst, p3zkc::lcg_jump(c.mseed, lo - c.pk.n, L), hi - lo, L);
        } else if (!c.cmask.empty()) {
            cudaMemcpy(dst, c.cmask.data() + (lo - c.pk.n), (hi - lo) * 8,
                       cudaMemcpyHostToDevice);
        } else {
            cudaMemsetAsync(dst, 0, (hi - lo) * 8, 0);
        }
    }
    if (end > c.ctot) {
        size_t z0 = std::max(off, c.ctot);
        cudaMemsetAsync(dout + (z0 - off), 0, (end - z0) * 8, 0);
    }
}
static inline size_t col_len(const Col& c) { return c.pk.on ? c.ctot : c.v.size(); }
static inline Col commit_col(std::vector<gl_t> vals, uint32_t R, bool gpu = true) {
    Col c; c.v = std::move(vals);
    c.root = gpu ? p3bf::commit_gpu(c.v, R, c.cw) : p3bf::commit(c.v, R, c.cw);
    return c;
}
// commit without materializing the host codeword (deferred/batched openings);
// zkmask (zk mode only) supplies a LINKED mask region (seam / row-sum bindings).
// cpk_dev (section 22b): produce the commitment PRE-COMPACTED -- the real
// region is packed on the host, the fresh mask region is generated ON THE
// DEVICE from its recorded seed (bit-identical chain, kernel above), and the
// augmented host column never exists.  Identical committed bytes -> identical
// root; falls back to the classic path when the values don't pack, the length
// is not a power of two, or a linked mask is supplied.
static inline Col commit_col_nc(std::vector<gl_t> vals, uint32_t R,
                                const std::vector<gl_t>* zkmask = nullptr,
                                bool cpk_dev = false) {
    struct Tm { double t0; Tm() : t0(lu_now_ms()) {}
                ~Tm() { g_lustats.commit_ms += lu_now_ms() - t0; g_lustats.commits++; } } tm_;
    Col c;
    if (p3zkc::G.on && cpk_dev && !zkmask && !vals.empty() &&
        (vals.size() & (vals.size() - 1)) == 0) {
        uint32_t vr = 0; while ((1ull << vr) < vals.size()) vr++;
        p3zkc::Packed pk;
        if (p3zkc::pack_ints(vals.data(), vals.size(), pk)) {
            const size_t Nr = vals.size();
            const size_t Naug = (size_t)1 << p3zkc::vfull(vr);
            c.vreal = vr; c.mstart = Nr;
            {
                p3zp::T zt(p3zp::g.mask_gen);
                c.mseed = p3zkc::G.mask_on ? p3zkc::next_seed() : 0;
            }
            gl_t* dm = p3bf::dmalloc(Naug, "ccnc:aug");
            cudaMemcpy(dm, vals.data(), Nr * 8, cudaMemcpyHostToDevice);
            if (c.mseed)
                p3zkc_lcgchain_kernel<<<(uint32_t)((Naug - Nr + 255) / 256), 256>>>(
                    dm + Nr, c.mseed, Naug - Nr, p3zkc::lcg_ladder());
            else
                cudaMemsetAsync(dm + Nr, 0, (Naug - Nr) * 8, 0);
            c.sseed = p3zkc::next_seed();
            {
                p3zp::T zt(p3zp::g.commit_salt);
                c.root = p3zkc::salted_commit_root_dev(dm, Naug, R, c.sseed);
            }
            cudaFreeAsync(dm, 0);
            p3zkc::spill_packed(pk);                       // no-op unless P3_PK_SPILL
            c.pk = std::move(pk); c.ctot = Naug;
            std::vector<gl_t>().swap(vals);
            return c;
        }
    }
    if (p3zkc::G.on) {
        uint32_t vr = 0; while ((1ull << vr) < vals.size()) vr++;
        c.vreal = vr;
        c.mstart = vals.size();
        {
            p3zp::T zt(p3zp::g.mask_gen);
            if (zkmask) {
                c.v = p3zkc::augment(vals, *zkmask);
            } else {
                // in-place augment: identical bytes to augment(vals,
                // fresh_mask(vr)) -- the mask region starts at vals.size()
                // and fills from the same next_seed() chain -- without the
                // separate mask vector + concat copy
                size_t Nr = vals.size();
                size_t ml = ((size_t)1 << p3zkc::vfull(vr)) - ((size_t)1 << vr);
                c.v = std::move(vals);
                c.v.resize(Nr + ml, 0);
                c.mseed = p3zkc::fresh_mask_into(c.v.data() + Nr, ml);
            }
        }
        c.sseed = p3zkc::next_seed();
        {
            p3zp::T zt(p3zp::g.commit_salt);
            c.root = p3zkc::salted_commit_root(c.v, R, c.sseed);
        }
        return c;
    }
    c.v = std::move(vals);
    { uint32_t vr = 0; while ((1ull << vr) < c.v.size()) vr++; c.vreal = vr; }
    c.mstart = c.v.size();
    {
        p3zp::T zt(p3zp::g.commit_plain);
        c.root = p3bf::commit_gpu_rootonly(c.v, R);
    }
    return c;
}

// commit an ALREADY-PACKED small-int column directly (section 23): the packed
// bytes upload + device unpack replace the cpk_dev path's host materialize +
// re-pack + raw upload.  The device augmented column is byte-identical to the
// cpk_dev path's (same unpack values, same mask chain, same seed draw order)
// -> identical root; the pack becomes the commitment's compact storage.
// zk mode only, no linked mask, power-of-two real length (caller checks).
static inline Col commit_pk_nc(p3zkc::Packed pk, uint32_t R) {
    struct Tm { double t0; Tm() : t0(lu_now_ms()) {}
                ~Tm() { g_lustats.commit_ms += lu_now_ms() - t0; g_lustats.commits++; } } tm_;
    Col c;
    uint32_t vr = 0; while ((1ull << vr) < pk.n) vr++;
    const size_t Nr = pk.n;
    const size_t Naug = (size_t)1 << p3zkc::vfull(vr);
    c.vreal = vr; c.mstart = Nr;
    {
        p3zp::T zt(p3zp::g.mask_gen);
        c.mseed = p3zkc::G.mask_on ? p3zkc::next_seed() : 0;
    }
    gl_t* dm = p3bf::dmalloc(Naug, "cpnc:aug");
    p3zkc::unpack_ints_dev(pk, 0, Nr, dm);
    if (c.mseed)
        p3zkc_lcgchain_kernel<<<(uint32_t)((Naug - Nr + 255) / 256), 256>>>(
            dm + Nr, c.mseed, Naug - Nr, p3zkc::lcg_ladder());
    else
        cudaMemsetAsync(dm + Nr, 0, (Naug - Nr) * 8, 0);
    c.sseed = p3zkc::next_seed();
    {
        p3zp::T zt(p3zp::g.commit_salt);
        c.root = p3zkc::salted_commit_root_dev(dm, Naug, R, c.sseed);
    }
    cudaFreeAsync(dm, 0);
    p3zkc::spill_packed(pk);
    c.pk = std::move(pk); c.ctot = Naug;
    return c;
}

// ---- public fixed table: c columns of length M=2^m, plus a binding hash ----
struct Table {
    std::vector<std::vector<gl_t>> cols;
    p3fri::Hash id;
};
static inline Table make_table(std::vector<std::vector<gl_t>> cols) {
    Table t; t.cols = std::move(cols);
    fs::Sha256Ctx cx; fs::sha256_init(cx);
    uint64_t nc = t.cols.size(), M = t.cols[0].size();
    fs::sha256_update(cx, &nc, 8); fs::sha256_update(cx, &M, 8);
    for (auto& col : t.cols) fs::sha256_update(cx, col.data(), col.size() * sizeof(gl_t));
    fs::sha256_final(cx, t.id.data());
    return t;
}

// witness-column spec of one lookup: committed column, VIRTUAL array (the
// caller binds virtual evals to a base commitment itself), or a GENERATED
// virtual column -- a closure that writes the gn values on demand at flush
// time, so big derived broadcasts are never held between registration and
// flush (design doc section 22)
struct LuColSpec { const Col* com; const std::vector<gl_t>* virt;
                   std::function<void(gl_t*, size_t)> gen; size_t gn = 0; };
static inline LuColSpec LC(const Col* c) { return {c, nullptr, {}, 0}; }
static inline LuColSpec LV(const std::vector<gl_t>* v) { return {nullptr, v, {}, 0}; }
static inline LuColSpec LG(std::function<void(gl_t*, size_t)> g, size_t n) {
    return {nullptr, nullptr, std::move(g), n};
}
static inline size_t luc_len(const LuColSpec& w) {
    return w.com ? col_len(*w.com) : w.virt ? w.virt->size() : w.gn;
}

// ---------------- deferred lookup obligations (R3b instance merging) ----------------
// Gadgets DEFER lookups into the shared context queue; the ledger owner (the
// standalone gadget or the composed layer) flushes ONCE: obligations grouped by
// (table id, log2 rows), each group proven as ONE logUp instance over the
// stacked member domain -- one cnt/hA/hT (+2x3 zk blinds) per GROUP instead of
// per lookup.  Merged layout u = i | (j<<n) | (ex<<(n+g)) (member row i, member
// j, zk ex slice): pad members j in [k,2^g) hold table row 0.  Every member's
// witness claims land at the shared point pm = (rA[0..n) || rA[n+g..)), whose
// SHAPE equals a standalone lookup's point in both modes, so per-gadget virtual
// binding code is drop-in (registered as a bind callback, run at flush).
struct XCtx;
struct VCtx;
// prover bind: absorb/ledger extra claims at pm; returned values ride the
// GroupProof (mem[j].extra) for the verifier-side bind to check.
using PBind = std::function<std::vector<gl_t>(fs::Transcript&, XCtx&,
                  const std::vector<gl_t>& pm, const std::vector<gl_t>& yv)>;
using VBind = std::function<bool(fs::Transcript&, VCtx&, const std::vector<gl_t>& pm,
                  const std::vector<gl_t>& yv, const std::vector<gl_t>& extra,
                  const char** why)>;
struct LuObl { std::vector<LuColSpec> W; const std::vector<uint32_t>* idx;
               const Table* tab; std::string label; PBind bind;
               // spilled index storage (P3_PK_SPILL): the flush's single
               // sequential read goes through the mapped file instead of
               // holding tokens x NLU x 4 B of anonymous memory to flush time
               std::shared_ptr<p3zkc::MagMap> imap; size_t ilen = 0;
               const uint32_t* iptr() const { return imap ? (const uint32_t*)imap->p : idx->data(); }
               size_t isize() const { return imap ? ilen : idx->size(); } };
struct VLuObl { std::vector<const p3fri::Hash*> roots; const Table* tab; uint32_t n;
                std::string label; VBind bind; };

// Shared-ledger composition context (design doc section 11.6 R3): a caller
// (e.g. the composed transformer layer) passes ONE XCtx through every gadget
// prover so all opening obligations land in a single PLedger and every
// committed column stays alive until the single end-of-layer batched-opening
// pass.  Gadgets given an XCtx skip their internal p3bo batches; the verifier
// mirror is VCtx (ledger + deferred lookup queue).
struct XCtx {
    p3bo::PLedger lg;                          // shared opening obligations
    std::deque<Col> keep;                      // lookup helper columns
    std::deque<std::vector<Col>> colvecs;      // gadget witness-column arenas
    std::vector<LuObl> luq;                    // deferred lookup obligations
    std::deque<std::vector<gl_t>> varena;      // deferred virtual-column arrays
    // compact-column registry (section 22): compacted committed columns keyed
    // by root; the batched-opening resolver rebuilds their values on demand
    std::map<p3fri::Hash, const Col*> creg;
    XCtx() {
        lg.resolve = [this](const p3fri::Hash& r, gl_t* out, size_t n) {
            auto it = creg.find(r);
            if (it == creg.end()) return false;
            mat_col_into(*it->second, out, n);
            return true;
        };
    }
    XCtx(const XCtx&) = delete;
    XCtx& operator=(const XCtx&) = delete;
    // compact a column and register it for opening-time materialization
    bool reg_compact(Col& c) {
        if (c.pk.on) { creg[c.root] = &c; return true; }   // pre-compacted commit
        if (!compact_col(c)) return false;
        p3zkc::spill_packed(c.pk);                         // no-op unless P3_PK_SPILL
        creg[c.root] = &c;
        return true;
    }
    std::vector<Col>& vec(size_t n) { colvecs.emplace_back(n); return colvecs.back(); }
    std::vector<gl_t>& varr(std::vector<gl_t> v) { varena.push_back(std::move(v)); return varena.back(); }
};
struct VCtx {
    p3bo::VLedger vlg;
    std::vector<VLuObl> luq;
};
static inline void defer_v(XCtx& xc, std::vector<LuColSpec> W, const std::vector<uint32_t>& idx,
                           const Table& T, std::string label, PBind bind = nullptr) {
    xc.luq.push_back({std::move(W), &idx, &T, std::move(label), std::move(bind)});
    // P3_PK_SPILL (g_free_idx callers only): big index arrays move to disk
    // until their single flush read -- the caller's vector is freed, the
    // same mutation the flush's g_free_idx path already performs
    if (!g_free_idx) return;
    LuObl& o = xc.luq.back();
    if (auto h = p3zkc::spill_bytes(idx.data(), idx.size() * 4)) {
        o.imap = std::move(h); o.ilen = idx.size(); o.idx = nullptr;
        auto* ix = const_cast<std::vector<uint32_t>*>(&idx);
        ix->clear(); ix->shrink_to_fit();
    }
}
// defer with an ALREADY-SPILLED index view (witness-build-time spill): the
// obligation reads through the mapped file exactly like defer_v's own spill
static inline void defer_v(XCtx& xc, std::vector<LuColSpec> W,
                           std::shared_ptr<p3zkc::MagMap> imap, size_t ilen,
                           const Table& T, std::string label, PBind bind = nullptr) {
    xc.luq.push_back({std::move(W), nullptr, &T, std::move(label), std::move(bind)});
    LuObl& o = xc.luq.back();
    o.imap = std::move(imap); o.ilen = ilen;
}
static inline void vdefer_v(VCtx& v, std::vector<const p3fri::Hash*> roots, const Table& T,
                            uint32_t n, std::string label, VBind bind = nullptr) {
    v.luq.push_back({std::move(roots), &T, n, std::move(label), std::move(bind)});
}

struct Msg4 { gl_t s0, s1, s2, s3; };      // cubic sumcheck round message
struct LookupProof {
    uint32_t n = 0, m = 0, c = 0;          // log2 witness rows, log2 table rows, #cols
    p3fri::Hash root_cnt, root_hA, root_hT;
    gl_t S = 0;                            // claimed common rational sum (zk: blinded S')
    p3zkc::Blind blA, blT;                 // zk: Libra blinds of the two chains
    std::vector<Msg4> msgsA, msgsT;
    p3bf::EvalProof open_hA;               // at rA
    std::vector<p3bf::EvalProof> open_W;   // each COMMITTED witness column at rA
    std::vector<gl_t> y_virt;              // claimed evals of VIRTUAL columns at rA
    p3bf::EvalProof open_hT, open_cnt;     // at rT
    // deferred-opening mode (prove_v with a p3bo ledger): claimed evals replace
    // the per-instance EvalProofs; the caller's batched opening backs them.
    std::vector<gl_t> yW;                  // committed witness cols at rA
    gl_t y_hA_c = 0, y_hT_c = 0, y_cnt_c = 0;
};

// one MERGED lookup instance.  v3 (GKR fractional-sum): a GroupProof is one
// SUPERGROUP = all obligations of one table, bundled as per-(log-rows) A-side
// subgroups that share ONE T side.  v3 replaces the v2 committed helper
// columns (hA per subgroup, hT) and their per-chain Libra blind columns with
// COMMIT-FREE GKR fraction trees (p3_gkr.cuh): each subgroup publishes its
// tree root (P_s, Q_s) with P_s/Q_s = sum of the subgroup's leaf fractions
// and a GKR layer chain reducing it to leaf claims; ONE T-side tree per
// table does the same for sum_j cnt_j/(Tc_j+beta).  Only cnt is committed
// (as in v2), plus in zk mode two uniform mask-leaf stream columns per tree
// (sibling-interleaved so every height-1 node is uniform -- the whole chain
// above the leaves becomes simulatable; the leaf claims land on the
// mechanism-1-blinded member/cnt augmented evals and on the mask MLEs).
// The multiset identity is checked on the published roots:
//   sum_s P_s/Q_s == P_T/Q_T
// with all mask fraction sums fixed BEFORE gamma/beta are drawn (the logUp
// Schwartz-Zippel argument needs the mask junk fixed pre-beta; the honest
// fixup makes it cancel exactly: SmT = sum_s SmA_s - Sm_cnt).
struct LuMember { std::vector<gl_t> yW, y_virt, extra; };
struct LuSubA {
    uint32_t n = 0, k = 0;                 // member log-rows, #members
    p3fri::Hash rt_pm = {}, rt_qm = {};    // zk: mask leaf stream commitments
    p3gkr::Proof gk;                       // root (P,Q) + layer chain
    std::vector<LuMember> mem;             // size k, member (registration) order
};
struct GroupProof {
    uint32_t m = 0, c = 0;                 // table log-rows, #cols
    p3fri::Hash root_cnt;
    p3fri::Hash rt_pmT = {}, rt_qmT = {};  // zk: T-side mask leaf commitments
    p3gkr::Proof gkT;
    std::vector<LuSubA> sub;               // per-(log-rows) subgroups, flush order
};
static inline size_t sz_group(const GroupProof& pf) {
    size_t s = 8 + 32 + p3gkr::sz_proof(pf.gkT);
    if (p3zkc::G.on) s += 2 * 32;
    for (auto& sb : pf.sub) {
        s += 8 + p3gkr::sz_proof(sb.gk);
        if (p3zkc::G.on) s += 2 * 32;
        for (auto& m : sb.mem)
            s += (m.yW.size() + m.y_virt.size() + m.extra.size()) * 8;
    }
    return s;
}

static inline gl_t chal(fs::Transcript& tr) {
    uint8_t b[32]; tr.challenge_bytes(b);
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v % GL_P;
}
static inline std::vector<gl_t> chal_vec(fs::Transcript& tr, uint32_t n) {
    std::vector<gl_t> v(n); for (auto& x : v) x = chal(tr); return v;
}
static inline gl_t gl_neg(gl_t x) { return gl_sub(0ULL, x); }
// Lagrange evaluation of the degree-3 poly through (0,s0)..(3,s3) at t.
static inline gl_t cubic_eval(gl_t s0, gl_t s1, gl_t s2, gl_t s3, gl_t t) {
    gl_t inv2 = gl_inv(2ULL), inv6 = gl_inv(6ULL);
    gl_t t1 = gl_sub(t, 1ULL), t2 = gl_sub(t, 2ULL), t3 = gl_sub(t, 3ULL);
    gl_t L0 = gl_neg(gl_mul(gl_mul(gl_mul(t1, t2), t3), inv6));
    gl_t L1 = gl_mul(gl_mul(gl_mul(t, t2), t3), inv2);
    gl_t L2 = gl_neg(gl_mul(gl_mul(gl_mul(t, t1), t3), inv2));
    gl_t L3 = gl_mul(gl_mul(gl_mul(t, t1), t2), inv6);
    return gl_add(gl_add(gl_mul(s0, L0), gl_mul(s1, L1)),
                  gl_add(gl_mul(s2, L2), gl_mul(s3, L3)));
}
static inline void bind_lsb(std::vector<gl_t>& f, gl_t a) {
    uint32_t h = (uint32_t)f.size() / 2; std::vector<gl_t> nf(h);
    #pragma omp parallel for schedule(static) if (h >= 65536) num_threads(p3bf::nthr(h))
    for (uint32_t i = 0; i < h; i++)
        nf[i] = gl_add(f[2*i], gl_mul(a, gl_sub(f[2*i+1], f[2*i])));
    f = nf;
}
static inline uint32_t ilog2(size_t n) { uint32_t l = 0; while ((1ull << l) < n) l++; return l; }

// out[i] = 1/(A[i]+beta) for all i -- Montgomery batch inversion: one gl_inv
// plus 3 muls/element instead of N Fermat inversions (~64 sqmuls each).
static inline std::vector<gl_t> inv_all_add(const std::vector<gl_t>& A, gl_t beta) {
    size_t N = A.size();
    std::vector<gl_t> out(N);
    // block-parallel Montgomery batch inversion: each block runs an independent
    // prefix/suffix pass with one gl_inv of its own block product -- out[i] is
    // the SAME field element 1/(A[i]+beta) regardless of blocking.
    const int P = N >= 65536 ? 256 : 1;
    std::atomic<bool> zero{false};
    #pragma omp parallel for schedule(static) if (P > 1) num_threads(p3bf::nthr(N))
    for (int p = 0; p < P; p++) {
        size_t lo = N * p / P, hi = N * (p + 1) / P;
        if (lo == hi) continue;
        std::vector<gl_t> d(hi - lo), pre(hi - lo);
        gl_t acc = 1ULL;
        for (size_t i = lo; i < hi; i++) {
            gl_t di = gl_add(A[i], beta);
            if (di == 0) { zero.store(true); di = 1ULL; }
            d[i - lo] = di; pre[i - lo] = acc; acc = gl_mul(acc, di);
        }
        gl_t inv = gl_inv(acc);
        for (size_t i = hi; i-- > lo;) { out[i] = gl_mul(pre[i - lo], inv); inv = gl_mul(inv, d[i - lo]); }
    }
    if (zero.load()) throw std::runtime_error("p3lu: zero denom (resample beta)");
    return out;
}

// gamma-combine c columns row-wise: out[i] = sum_k g^k cols[k][i]
static inline std::vector<gl_t> combine(const std::vector<const std::vector<gl_t>*>& cols, gl_t g) {
    size_t N = cols[0]->size();
    std::vector<gl_t> out(N, 0);
    gl_t gk = 1ULL;
    for (auto* col : cols) {
        for (size_t i = 0; i < N; i++) out[i] = gl_add(out[i], gl_mul(gk, (*col)[i]));
        gk = gl_mul(gk, g);
    }
    return out;
}

// One batched cubic sumcheck:  sum_b [ lam*H(b) + E(b)*( H(b)*(V(b)+beta) - D(b) ) ]
// (A side: D = all-ones;  T side: D = cnt).  Binds LSB-first; returns challenges.
// zk: B = 3 committed blind columns and rho add the degree-matched Libra term
// rho*( B0(b) + E(b)*B1(b) + E(b)^2*B2(b) ) -- round degrees 1,2,3, spanning
// every coefficient of the cubic message (p3_zkc mechanism 2).
static inline std::vector<gl_t> sc_prove(std::vector<gl_t> H, std::vector<gl_t> V,
                                         std::vector<gl_t> E, std::vector<gl_t> D,
                                         gl_t lam, gl_t beta,
                                         fs::Transcript& tr, const char* tag,
                                         std::vector<Msg4>& msgs,
                                         std::vector<gl_t>* B = nullptr, gl_t rho = 0) {
    uint32_t v = ilog2(H.size());
    std::vector<gl_t> r;
    for (uint32_t rd = 0; rd < v; rd++) {
        uint32_t half = (uint32_t)H.size() / 2;
        gl_t s[4] = {0, 0, 0, 0};
        const int P = half >= 4096 ? 128 : 1;
        std::vector<std::array<gl_t, 4>> part(P, {0, 0, 0, 0});
        #pragma omp parallel for schedule(static) if (P > 1) num_threads(p3bf::nthr((size_t)half * 8))
        for (int p = 0; p < P; p++) {
            size_t lo = (size_t)half * p / P, hi = (size_t)half * (p + 1) / P;
            std::array<gl_t, 4>& sp = part[p];
            for (size_t i = lo; i < hi; i++) {
                gl_t h = H[2*i], dh = gl_sub(H[2*i+1], H[2*i]);
                gl_t x = V[2*i], dx = gl_sub(V[2*i+1], V[2*i]);
                gl_t e = E[2*i], de = gl_sub(E[2*i+1], E[2*i]);
                gl_t d = D[2*i], dd = gl_sub(D[2*i+1], D[2*i]);
                gl_t b0 = 0, db0 = 0, b1 = 0, db1 = 0, b2 = 0, db2 = 0;
                if (B) {
                    b0 = B[0][2*i]; db0 = gl_sub(B[0][2*i+1], b0);
                    b1 = B[1][2*i]; db1 = gl_sub(B[1][2*i+1], b1);
                    b2 = B[2][2*i]; db2 = gl_sub(B[2][2*i+1], b2);
                }
                for (int t = 0; t < 4; t++) {
                    gl_t val = gl_add(gl_mul(lam, h),
                                      gl_mul(e, gl_sub(gl_mul(h, gl_add(x, beta)), d)));
                    if (B) val = gl_add(val, gl_mul(rho,
                                    gl_add(b0, gl_mul(e, gl_add(b1, gl_mul(e, b2))))));
                    sp[t] = gl_add(sp[t], val);
                    h = gl_add(h, dh); x = gl_add(x, dx); e = gl_add(e, de); d = gl_add(d, dd);
                    if (B) { b0 = gl_add(b0, db0); b1 = gl_add(b1, db1); b2 = gl_add(b2, db2); }
                }
            }
        }
        for (int p = 0; p < P; p++)
            for (int t = 0; t < 4; t++) s[t] = gl_add(s[t], part[p][t]);
        Msg4 msg{s[0], s[1], s[2], s[3]};
        msgs.push_back(msg); tr.absorb(tag, &msg, sizeof msg);
        gl_t a = chal(tr); r.push_back(a);
        bind_lsb(H, a); bind_lsb(V, a); bind_lsb(E, a); bind_lsb(D, a);
        if (B) { bind_lsb(B[0], a); bind_lsb(B[1], a); bind_lsb(B[2], a); }
    }
    return r;
}

// verifier side of one batched cubic sumcheck message chain
static inline bool sc_verify(const std::vector<Msg4>& msgs, uint32_t v, gl_t claim0,
                             fs::Transcript& tr, const char* tag,
                             std::vector<gl_t>& r_out, gl_t& claim_out) {
    if (msgs.size() != v) return false;
    gl_t claim = claim0;
    for (uint32_t rd = 0; rd < v; rd++) {
        const Msg4& m = msgs[rd];
        if (gl_add(m.s0, m.s1) != claim) return false;
        tr.absorb(tag, &m, sizeof m);
        gl_t a = chal(tr); r_out.push_back(a);
        claim = cubic_eval(m.s0, m.s1, m.s2, m.s3, a);
    }
    claim_out = claim;
    return true;
}

// ---------------- prover ----------------
// W: committed witness columns (all length N=2^n).  idx: prover hint, row i of
// the witness is table row idx[i] (only completeness depends on it).
// strict=false lets the selftest run the honest PROCEDURE on an inconsistent
// witness (the verifier must catch it).
// hA_tamper (selftest only): (index, delta) pairs added to hA AFTER computing the
// honest inverses -- a sum-preserving pair forgery must be caught by the
// eq-weighted zero-check, not the sum.
static inline LookupProof prove(fs::Transcript& tr, const std::vector<const Col*>& W,
                                const std::vector<uint32_t>& idx, const Table& T,
                                uint32_t R, uint32_t Q, const std::string& label,
                                bool gpu = true, bool strict = true,
                                const std::vector<std::pair<size_t, gl_t>>* hA_tamper = nullptr) {
    LookupProof pf;
    size_t N = W[0]->v.size(), M = T.cols[0].size();
    pf.n = ilog2(N); pf.m = ilog2(M); pf.c = (uint32_t)W.size();
    if ((1ull << pf.n) != N || (1ull << pf.m) != M) throw std::runtime_error("p3lu: pow2");
    if (T.cols.size() != W.size()) throw std::runtime_error("p3lu: col count");

    tr.absorb("lu-tab", T.id.data(), 32);
    for (auto* w : W) tr.absorb("lu-W", w->root.data(), 32);

    // multiplicities
    std::vector<gl_t> cnt(M, 0);
    for (size_t i = 0; i < N; i++) {
        if (idx[i] >= M) throw std::runtime_error("p3lu: idx range");
        cnt[idx[i]] = gl_add(cnt[idx[i]], 1ULL);
    }
    Col Ccnt = commit_col(cnt, R, gpu);
    pf.root_cnt = Ccnt.root; tr.absorb("lu-cnt", pf.root_cnt.data(), 32);

    gl_t gamma = chal(tr), beta = chal(tr);

    std::vector<const std::vector<gl_t>*> wp, tp;
    for (auto* w : W) wp.push_back(&w->v);
    for (auto& t : T.cols) tp.push_back(&t);
    std::vector<gl_t> A = combine(wp, gamma), Tc = combine(tp, gamma);

    std::vector<gl_t> hA(N), hT(M);
    for (size_t i = 0; i < N; i++) {
        gl_t d = gl_add(A[i], beta);
        if (d == 0) throw std::runtime_error("p3lu: zero denom (resample beta)");
        hA[i] = gl_inv(d);
    }
    for (size_t j = 0; j < M; j++) {
        gl_t d = gl_add(Tc[j], beta);
        if (d == 0) throw std::runtime_error("p3lu: zero denom (resample beta)");
        hT[j] = gl_mul(cnt[j], gl_inv(d));
    }
    if (hA_tamper) for (auto& [i, d] : *hA_tamper) hA[i] = gl_add(hA[i], d);
    Col ChA = commit_col(hA, R, gpu), ChT = commit_col(hT, R, gpu);
    pf.root_hA = ChA.root; pf.root_hT = ChT.root;
    tr.absorb("lu-hA", pf.root_hA.data(), 32); tr.absorb("lu-hT", pf.root_hT.data(), 32);

    gl_t SA = 0, ST = 0;
    for (auto x : hA) SA = gl_add(SA, x);
    for (auto x : hT) ST = gl_add(ST, x);
    if (strict && SA != ST) throw std::runtime_error("p3lu: witness not in table");
    pf.S = SA; tr.absorb("lu-S", &pf.S, sizeof pf.S);

    // A side
    std::vector<gl_t> zA = chal_vec(tr, pf.n); gl_t lamA = chal(tr);
    std::vector<gl_t> rA = sc_prove(hA, A, p3bf::build_eq(zA), std::vector<gl_t>(N, 1ULL),
                                    lamA, beta, tr, "lu-scA", pf.msgsA);
    gl_t y_hA = p3bf::eval_h(ChA.v, p3bf::build_eq(rA));
    pf.open_hA = p3bf::prove_eval(ChA.v, rA, y_hA, R, Q, ChA.cw, label + "-hA");
    for (uint32_t k = 0; k < pf.c; k++) {
        gl_t y = p3bf::eval_h(W[k]->v, p3bf::build_eq(rA));
        pf.open_W.push_back(p3bf::prove_eval(W[k]->v, rA, y, R, Q, W[k]->cw,
                                             label + "-W" + std::to_string(k)));
    }

    // T side
    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> rT = sc_prove(hT, Tc, p3bf::build_eq(zT), cnt,
                                    lamT, beta, tr, "lu-scT", pf.msgsT);
    gl_t y_hT = p3bf::eval_h(ChT.v, p3bf::build_eq(rT));
    gl_t y_cnt = p3bf::eval_h(Ccnt.v, p3bf::build_eq(rT));
    pf.open_hT = p3bf::prove_eval(ChT.v, rT, y_hT, R, Q, ChT.cw, label + "-hT");
    pf.open_cnt = p3bf::prove_eval(Ccnt.v, rT, y_cnt, R, Q, Ccnt.cw, label + "-cnt");

    // chain the opened terminals into the caller's transcript (composition binding)
    tr.absorb("lu-ends", &pf.open_hA.y, sizeof(gl_t));
    tr.absorb("lu-ends", &pf.open_hT.y, sizeof(gl_t));
    tr.absorb("lu-ends", &pf.open_cnt.y, sizeof(gl_t));
    return pf;
}

// ---------------- verifier ----------------
// W_roots: the caller-known commitments of the witness columns.  Q_pub/R_pub are
// PUBLIC FRI parameters (never read from the proof -- p3pfc red-team CRITICAL-1).
// On success, y_W_out (if non-null) receives the opened W_k(rA) values and rA_out
// the opening point, so callers can reuse the bound witness evaluations.
static inline bool verify(fs::Transcript& tr, const std::vector<p3fri::Hash>& W_roots,
                          const Table& T, const LookupProof& pf,
                          uint32_t Q_pub, uint32_t R_pub, const std::string& label,
                          const char** why = nullptr,
                          std::vector<gl_t>* y_W_out = nullptr,
                          std::vector<gl_t>* rA_out = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    size_t M = T.cols[0].size();
    if (pf.c != W_roots.size() || pf.c != T.cols.size()) return fail("col count");
    if ((1ull << pf.m) != M) return fail("table size");
    if (pf.open_W.size() != pf.c) return fail("open_W count");
    const uint32_t Q_MIN = 20;
    if (Q_pub < Q_MIN || R_pub < 1) return fail("insecure params");
    auto chkpar = [&](const p3bf::EvalProof& e, uint32_t logN_exp) {
        return e.Q == Q_pub && e.R == R_pub && e.logN == logN_exp;
    };

    tr.absorb("lu-tab", T.id.data(), 32);
    for (auto& r : W_roots) tr.absorb("lu-W", r.data(), 32);
    tr.absorb("lu-cnt", pf.root_cnt.data(), 32);
    gl_t gamma = chal(tr), beta = chal(tr);
    tr.absorb("lu-hA", pf.root_hA.data(), 32); tr.absorb("lu-hT", pf.root_hT.data(), 32);
    tr.absorb("lu-S", &pf.S, sizeof pf.S);

    // A side
    std::vector<gl_t> zA = chal_vec(tr, pf.n); gl_t lamA = chal(tr);
    std::vector<gl_t> rA; gl_t claimA;
    if (!sc_verify(pf.msgsA, pf.n, gl_mul(lamA, pf.S), tr, "lu-scA", rA, claimA))
        return fail("sumcheck claim A");
    if (pf.open_hA.roots.empty() || !(pf.open_hA.roots[0] == pf.root_hA) || pf.open_hA.z != rA)
        return fail("hA opening bind");
    if (!chkpar(pf.open_hA, pf.n)) return fail("hA opening params");
    if (!p3bf::verify_eval(pf.open_hA, label + "-hA", why)) return false;
    gl_t A_r = 0, gk = 1ULL;
    for (uint32_t k = 0; k < pf.c; k++) {
        const auto& op = pf.open_W[k];
        if (op.roots.empty() || !(op.roots[0] == W_roots[k]) || op.z != rA)
            return fail("W opening bind");
        if (!chkpar(op, pf.n)) return fail("W opening params");
        if (!p3bf::verify_eval(op, label + "-W" + std::to_string(k), why)) return false;
        A_r = gl_add(A_r, gl_mul(gk, op.y)); gk = gl_mul(gk, gamma);
    }
    gl_t eqA = p3bf::eq_point(rA, zA);
    gl_t y_hA = pf.open_hA.y;
    gl_t endA = gl_add(gl_mul(lamA, y_hA),
                       gl_mul(eqA, gl_sub(gl_mul(y_hA, gl_add(A_r, beta)), 1ULL)));
    if (claimA != endA) return fail("A terminal");

    // T side
    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> rT; gl_t claimT;
    if (!sc_verify(pf.msgsT, pf.m, gl_mul(lamT, pf.S), tr, "lu-scT", rT, claimT))
        return fail("sumcheck claim T");
    if (pf.open_hT.roots.empty() || !(pf.open_hT.roots[0] == pf.root_hT) || pf.open_hT.z != rT)
        return fail("hT opening bind");
    if (pf.open_cnt.roots.empty() || !(pf.open_cnt.roots[0] == pf.root_cnt) || pf.open_cnt.z != rT)
        return fail("cnt opening bind");
    if (!chkpar(pf.open_hT, pf.m) || !chkpar(pf.open_cnt, pf.m)) return fail("T opening params");
    if (!p3bf::verify_eval(pf.open_hT, label + "-hT", why)) return false;
    if (!p3bf::verify_eval(pf.open_cnt, label + "-cnt", why)) return false;
    // verifier evaluates the PUBLIC combined table at rT itself
    std::vector<gl_t> eqT = p3bf::build_eq(rT);
    gl_t Tc_r = 0; gk = 1ULL;
    for (auto& col : T.cols) {
        gl_t v = 0;
        for (size_t j = 0; j < M; j++) v = gl_add(v, gl_mul(col[j], eqT[j]));
        Tc_r = gl_add(Tc_r, gl_mul(gk, v)); gk = gl_mul(gk, gamma);
    }
    gl_t y_hT = pf.open_hT.y, y_cnt = pf.open_cnt.y;
    gl_t eqTr = p3bf::eq_point(rT, zT);
    gl_t endT = gl_add(gl_mul(lamT, y_hT),
                       gl_mul(eqTr, gl_sub(gl_mul(y_hT, gl_add(Tc_r, beta)), y_cnt)));
    if (claimT != endT) return fail("T terminal");

    tr.absorb("lu-ends", &pf.open_hA.y, sizeof(gl_t));
    tr.absorb("lu-ends", &pf.open_hT.y, sizeof(gl_t));
    tr.absorb("lu-ends", &pf.open_cnt.y, sizeof(gl_t));
    if (y_W_out) { y_W_out->clear(); for (auto& op : pf.open_W) y_W_out->push_back(op.y); }
    if (rA_out) *rA_out = rA;
    if (why) *why = "ok";
    return true;
}

// ---------------- virtual-column variant ----------------
// Some witness columns of a lookup can be VIRTUAL: not committed as their own
// Basefold column because they are broadcasts/bit-permutations of an already
// committed base polynomial (e.g. the per-product fp8 code a(p) = X(b,k) is
// constant in n, so a~(rA) = X~(rearranged rA)).  The prover supplies their
// value arrays; the proof carries their claimed MLE evaluations y_virt at the
// lookup point rA; the CALLER is responsible for binding each y_virt to the
// base commitment (a Basefold opening of the base at the rearranged point).
// verify_v returns rA and y_virt for exactly that purpose -- a caller that
// ignores them gets NO soundness for the virtual columns.
static inline LookupProof prove_v(fs::Transcript& tr, const std::vector<LuColSpec>& W,
                                  const std::vector<uint32_t>& idx, const Table& T,
                                  uint32_t R, uint32_t Q, const std::string& label,
                                  bool gpu = true, bool strict = true,
                                  std::vector<gl_t>* rA_out = nullptr,
                                  p3bo::PLedger* lg = nullptr, std::deque<Col>* keep = nullptr) {
    struct Tm { double t0; Tm() : t0(lu_now_ms()) {}
                ~Tm() { g_lustats.ms += lu_now_ms() - t0; g_lustats.calls++; } } tm_;
    LookupProof pf;
    const bool zk = p3zkc::G.on;
    if (zk && !lg) throw std::runtime_error("p3lu: zk requires the deferred ledger");
    for (auto& w : W)
        if (w.gen || (w.com && w.com->pk.on))
            throw std::runtime_error("p3lu: compact/generated columns need the "
                                     "merged-flush (v3) path");
    const std::vector<gl_t>* wv0 = W[0].com ? &W[0].com->v : W[0].virt;
    size_t N = wv0->size(), M = T.cols[0].size();       // zk: N = AUGMENTED length
    size_t Nreal = idx.size();
    pf.n = ilog2(Nreal); pf.m = ilog2(M); pf.c = (uint32_t)W.size();
    if ((1ull << pf.n) != Nreal || (1ull << pf.m) != M) throw std::runtime_error("p3lu: pow2");
    if (zk) { if (N != ((size_t)1 << p3zkc::vfull(pf.n))) throw std::runtime_error("p3lu: zk aug size"); }
    else if (N != Nreal) throw std::runtime_error("p3lu: idx size");
    if (T.cols.size() != W.size()) throw std::runtime_error("p3lu: col count");

    tr.absorb("lu-tab", T.id.data(), 32);
    for (auto& w : W) {
        if (w.com) tr.absorb("lu-W", w.com->root.data(), 32);
        else       tr.absorb("lu-Wv", "virt", 4);   // binding lives with the caller
    }

    std::vector<gl_t> cnt(M, 0);
    for (size_t i = 0; i < Nreal; i++) {
        if (idx[i] >= M) throw std::runtime_error("p3lu: idx range");
        cnt[idx[i]] = gl_add(cnt[idx[i]], 1ULL);
    }
    Col Ccnt = lg ? commit_col_nc(cnt, R) : commit_col(cnt, R, gpu);
    pf.root_cnt = Ccnt.root; tr.absorb("lu-cnt", pf.root_cnt.data(), 32);

    gl_t gamma = chal(tr), beta = chal(tr);

    std::vector<const std::vector<gl_t>*> wp, tp;
    for (auto& w : W) wp.push_back(w.com ? &w.com->v : w.virt);
    for (auto& t : T.cols) tp.push_back(&t);
    std::vector<gl_t> A = combine(wp, gamma), Tc = combine(tp, gamma);

    // helper inverses on the REAL region only; zk masks are free uniform (the
    // zero-check part is killed on mask rows by the eq((z||0),.) weight) except
    // hT's, whose sum is fixed so both chains share the blinded S'.
    double t_inv = lu_now_ms();
    std::vector<gl_t> Areal(A.begin(), A.begin() + Nreal);
    std::vector<gl_t> hA = inv_all_add(Areal, beta), hT = inv_all_add(Tc, beta);
    g_lustats.inv_ms += lu_now_ms() - t_inv;
    for (size_t j = 0; j < M; j++) hT[j] = gl_mul(cnt[j], hT[j]);
    gl_t SA = 0, ST = 0;
    for (auto x : hA) SA = gl_add(SA, x);
    for (auto x : hT) ST = gl_add(ST, x);
    if (strict && SA != ST) throw std::runtime_error("p3lu: witness not in table: " + label);

    Col ChA, ChT;
    gl_t Sp = SA;                                        // published sum (zk: blinded S')
    if (zk) {
        ChA = commit_col_nc(hA, R);                      // fresh uniform mask
        Sp = 0; for (auto x : ChA.v) Sp = gl_add(Sp, x); // S' = S_A + m,  m = mask sum
        gl_t m = gl_sub(Sp, SA);                         // the A-side mask sum
        // hT's mask sum must equal m (NOT absorb S_A-S_T): then the hT-aug sums
        // to S_T + m, which equals S' iff S_A == S_T -- so the sum-equality that
        // detects a witness-not-in-table tamper is PRESERVED, only its value hid.
        std::vector<gl_t> mT = p3zkc::fresh_mask(pf.m);
        gl_t ms = 0; for (auto x : mT) ms = gl_add(ms, x);
        mT.back() = gl_add(mT.back(), gl_sub(m, ms));
        ChT = commit_col_nc(hT, R, &mT);
    } else {
        ChA = lg ? commit_col_nc(hA, R) : commit_col(hA, R, gpu);
        ChT = lg ? commit_col_nc(hT, R) : commit_col(hT, R, gpu);
    }
    pf.root_hA = ChA.root; pf.root_hT = ChT.root;
    tr.absorb("lu-hA", pf.root_hA.data(), 32); tr.absorb("lu-hT", pf.root_hT.data(), 32);

    pf.S = Sp; tr.absorb("lu-S", &pf.S, sizeof pf.S);

    // zk: commit the Libra blind columns of both chains BEFORE the chains'
    // challenges.  H = sum_b Blind(b) is published AFTER z (E depends on z)
    // but BEFORE rho: a lie about H must satisfy lam*(S-S_true) = rho*(H-H*)
    // for a rho drawn afterwards -- Schwartz-Zippel kills it.
    std::vector<Col> BA(3), BT(3);
    gl_t rhoA = 0, rhoT = 0;
    if (zk) {
        for (int j = 0; j < 3; j++) {
            std::vector<gl_t> m;
            BA[j] = commit_col_nc(p3zkc::blind_col(pf.n, m), R, &m);
            pf.blA.rt[j] = BA[j].root; tr.absorb("lu-bA", BA[j].root.data(), 32);
        }
        pf.blA.nb = 3;
        for (int j = 0; j < 3; j++) {
            std::vector<gl_t> m;
            BT[j] = commit_col_nc(p3zkc::blind_col(pf.m, m), R, &m);
            pf.blT.rt[j] = BT[j].root; tr.absorb("lu-bT", BT[j].root.data(), 32);
        }
        pf.blT.nb = 3;
    }

    bool big = gpu && pf.n >= 16 && !zk;
    std::vector<gl_t> zA = chal_vec(tr, pf.n); gl_t lamA = chal(tr);
    std::vector<gl_t> zAf = p3zkc::zpt(zA);
    if (zk) {
        // full blind sum: sum_b [B0 + E*B1 + E^2*B2] with E public -- absorbed
        // before rho, after zA (a deterministic public function, not forgeable)
        std::vector<gl_t> E = p3bf::build_eq(zAf);
        gl_t HA = 0;
        for (size_t b = 0; b < N; b++)
            HA = gl_add(HA, gl_add(BA[0].v[b],
                     gl_mul(E[b], gl_add(BA[1].v[b], gl_mul(E[b], BA[2].v[b])))));
        pf.blA.H = HA;
        tr.absorb("lu-HA", &pf.blA.H, sizeof(gl_t));
        rhoA = chal(tr);
    }
    std::vector<gl_t> rA;
    double t_sc = lu_now_ms();
    if (big) {   // device-resident cubic sumcheck (identical messages)
        gl_t *dH, *dV, *dE, *dD, *dz;
        cudaMallocAsync(&dH, (size_t)N * 8, 0);
        cudaMemcpy(dH, hA.data(), (size_t)N * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dV, (size_t)N * 8, 0);
        cudaMemcpy(dV, A.data(), (size_t)N * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dz, (size_t)pf.n * 8, 0);
        cudaMemcpy(dz, zA.data(), (size_t)pf.n * 8, cudaMemcpyHostToDevice);
        cudaMallocAsync(&dE, (size_t)N * 8, 0);
        p3bf::p3bf_eq_kernel<<<((uint32_t)N + 255) / 256, 256>>>(dz, dE, pf.n, (uint32_t)N);
        cudaFreeAsync(dz, 0);
        cudaMallocAsync(&dD, (size_t)N * 8, 0);
        p3sg::p3sg_fill_kernel<<<((uint32_t)N + 255) / 256, 256>>>(dD, 1ULL, (uint32_t)N);
        std::vector<gl_t*> dc = {dH, dV, dE, dD};
        gl_t par2[2] = {lamA, beta};
        rA = p3sg::sc_prove_gpu<FLuGpu, Msg4, 4, 4>(tr, "lu-scA", dc, (uint32_t)N,
                                                    par2, 2, pf.msgsA);
        for (auto p : dc) cudaFreeAsync(p, 0);
    } else if (zk) {
        std::vector<gl_t> Bcols[3] = {BA[0].v, BA[1].v, BA[2].v};
        rA = sc_prove(ChA.v, A, p3bf::build_eq(zAf), std::vector<gl_t>(N, 1ULL),
                      lamA, beta, tr, "lu-scA", pf.msgsA, Bcols, rhoA);
    } else {
        rA = sc_prove(hA, A, p3bf::build_eq(zA), std::vector<gl_t>(N, 1ULL),
                      lamA, beta, tr, "lu-scA", pf.msgsA);
    }
    g_lustats.sc_ms += lu_now_ms() - t_sc;
    std::vector<gl_t> eqrA;
    if (!big) eqrA = p3bf::build_eq(rA);
    auto evA = [&](const std::vector<gl_t>& col) {
        return big ? p3bf::eval_h_gpu(col, rA) : p3bf::eval_h(col, eqrA);
    };
    gl_t y_hA = evA(ChA.v);
    if (lg) {
        pf.y_hA_c = y_hA;
        uint64_t ss = ChA.sseed;
        keep->push_back(std::move(ChA));
        lg->add(&keep->back().v, pf.root_hA, rA, y_hA, ss);
        if (zk) for (int j = 0; j < 3; j++) {
            pf.blA.yB[j] = evA(BA[j].v);
            tr.absorb("lu-ybA", &pf.blA.yB[j], sizeof(gl_t));
            uint64_t s2 = BA[j].sseed;
            keep->push_back(std::move(BA[j]));
            lg->add(&keep->back().v, pf.blA.rt[j], rA, pf.blA.yB[j], s2);
        }
    } else {
        pf.open_hA = p3bf::prove_eval(ChA.v, rA, y_hA, R, Q, ChA.cw, label + "-hA");
    }
    for (uint32_t k = 0; k < pf.c; k++) {
        gl_t y = evA(*wp[k]);
        if (W[k].com) {
            if (lg) {
                pf.yW.push_back(y);
                tr.absorb("lu-yW", &y, sizeof(gl_t));
                lg->add(&W[k].com->v, W[k].com->root, rA, y, W[k].com->sseed);
            } else {
                pf.open_W.push_back(p3bf::prove_eval(W[k].com->v, rA, y, R, Q, W[k].com->cw,
                                                     label + "-W" + std::to_string(k)));
            }
        } else {
            pf.y_virt.push_back(y);
            tr.absorb("lu-yv", &y, sizeof(gl_t));
        }
    }

    double t_scT = lu_now_ms();
    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> zTf = p3zkc::zpt(zT);
    std::vector<gl_t> rT;
    if (zk) {
        std::vector<gl_t> E = p3bf::build_eq(zTf);
        gl_t HT = 0;
        size_t Maug = ChT.v.size();
        for (size_t b = 0; b < Maug; b++)
            HT = gl_add(HT, gl_add(BT[0].v[b],
                     gl_mul(E[b], gl_add(BT[1].v[b], gl_mul(E[b], BT[2].v[b])))));
        pf.blT.H = HT;
        tr.absorb("lu-HT", &pf.blT.H, sizeof(gl_t));
        rhoT = chal(tr);
        std::vector<gl_t> TcA = Tc; TcA.resize(Maug, 0);       // zero-extended public table
        std::vector<gl_t> Bcols[3] = {BT[0].v, BT[1].v, BT[2].v};
        rT = sc_prove(ChT.v, TcA, E, Ccnt.v, lamT, beta, tr, "lu-scT", pf.msgsT,
                      Bcols, rhoT);
    } else {
        rT = sc_prove(hT, Tc, p3bf::build_eq(zT), cnt,
                      lamT, beta, tr, "lu-scT", pf.msgsT);
    }
    g_lustats.scg_ms += lu_now_ms() - t_scT;
    std::vector<gl_t> eqrT = p3bf::build_eq(rT);
    gl_t y_hT = p3bf::eval_h(ChT.v, eqrT);
    gl_t y_cnt = p3bf::eval_h(Ccnt.v, eqrT);
    if (lg) {
        pf.y_hT_c = y_hT; pf.y_cnt_c = y_cnt;
        uint64_t ssT = ChT.sseed, ssC = Ccnt.sseed;
        keep->push_back(std::move(ChT));
        lg->add(&keep->back().v, pf.root_hT, rT, y_hT, ssT);
        keep->push_back(std::move(Ccnt));
        lg->add(&keep->back().v, pf.root_cnt, rT, y_cnt, ssC);
        if (zk) for (int j = 0; j < 3; j++) {
            pf.blT.yB[j] = p3bf::eval_h(BT[j].v, eqrT);
            tr.absorb("lu-ybT", &pf.blT.yB[j], sizeof(gl_t));
            uint64_t s2 = BT[j].sseed;
            keep->push_back(std::move(BT[j]));
            lg->add(&keep->back().v, pf.blT.rt[j], rT, pf.blT.yB[j], s2);
        }
    } else {
        pf.open_hT = p3bf::prove_eval(ChT.v, rT, y_hT, R, Q, ChT.cw, label + "-hT");
        pf.open_cnt = p3bf::prove_eval(Ccnt.v, rT, y_cnt, R, Q, Ccnt.cw, label + "-cnt");
    }

    tr.absorb("lu-ends", &y_hA, sizeof(gl_t));
    tr.absorb("lu-ends", &y_hT, sizeof(gl_t));
    tr.absorb("lu-ends", &y_cnt, sizeof(gl_t));
    if (rA_out) *rA_out = rA;
    return pf;
}

// W_roots: entry k is the commitment of column k, or nullptr if column k is
// virtual.  On success *rA_out / *y_virt_out give the point and the claimed
// virtual-column evaluations THE CALLER MUST BIND to the base commitments.
static inline bool verify_v(fs::Transcript& tr, const std::vector<const p3fri::Hash*>& W_roots,
                            const Table& T, const LookupProof& pf,
                            uint32_t Q_pub, uint32_t R_pub, const std::string& label,
                            const char** why,
                            std::vector<gl_t>* rA_out, std::vector<gl_t>* y_virt_out,
                            std::vector<gl_t>* y_W_out = nullptr,
                            p3bo::VLedger* vlg = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    size_t M = T.cols[0].size();
    size_t n_virt = 0; for (auto* r : W_roots) if (!r) n_virt++;
    if (pf.c != W_roots.size() || pf.c != T.cols.size()) return fail("col count");
    if ((1ull << pf.m) != M) return fail("table size");
    if (vlg) { if (!pf.open_W.empty() || pf.yW.size() != pf.c - n_virt) return fail("yW count"); }
    else if (pf.open_W.size() != pf.c - n_virt) return fail("open_W count");
    if (pf.y_virt.size() != n_virt) return fail("y_virt count");
    const uint32_t Q_MIN = 20;
    if (Q_pub < Q_MIN || R_pub < 1) return fail("insecure params");
    auto chkpar = [&](const p3bf::EvalProof& e, uint32_t logN_exp) {
        return e.Q == Q_pub && e.R == R_pub && e.logN == logN_exp;
    };

    tr.absorb("lu-tab", T.id.data(), 32);
    for (auto* r : W_roots) {
        if (r) tr.absorb("lu-W", r->data(), 32);
        else   tr.absorb("lu-Wv", "virt", 4);
    }
    tr.absorb("lu-cnt", pf.root_cnt.data(), 32);
    gl_t gamma = chal(tr), beta = chal(tr);
    tr.absorb("lu-hA", pf.root_hA.data(), 32); tr.absorb("lu-hT", pf.root_hT.data(), 32);
    tr.absorb("lu-S", &pf.S, sizeof pf.S);

    const bool zk = p3zkc::G.on;
    if (zk && !vlg) return fail("zk requires deferred ledger");
    if (zk) {
        if (pf.blA.nb != 3 || pf.blT.nb != 3) return fail("zk blind count");
        for (int j = 0; j < 3; j++) tr.absorb("lu-bA", pf.blA.rt[j].data(), 32);
        for (int j = 0; j < 3; j++) tr.absorb("lu-bT", pf.blT.rt[j].data(), 32);
    }

    std::vector<gl_t> zA = chal_vec(tr, pf.n); gl_t lamA = chal(tr);
    std::vector<gl_t> zAf = p3zkc::zpt(zA);
    gl_t rhoA = 0;
    if (zk) { gl_t H = pf.blA.H; tr.absorb("lu-HA", &H, sizeof(gl_t)); rhoA = chal(tr); }
    std::vector<gl_t> rA; gl_t claimA;
    gl_t claimA0 = gl_add(gl_mul(lamA, pf.S), gl_mul(rhoA, pf.blA.H));
    if (!sc_verify(pf.msgsA, zk ? p3zkc::vfull(pf.n) : pf.n, claimA0, tr, "lu-scA", rA, claimA))
        return fail("sumcheck claim A");
    gl_t y_hA;
    if (vlg) {
        y_hA = pf.y_hA_c;
        vlg->add(pf.root_hA, rA, y_hA);
        if (zk) for (int j = 0; j < 3; j++) {
            gl_t yb = pf.blA.yB[j];
            tr.absorb("lu-ybA", &yb, sizeof(gl_t));
            vlg->add(pf.blA.rt[j], rA, yb);
        }
    } else {
        if (pf.open_hA.roots.empty() || !(pf.open_hA.roots[0] == pf.root_hA) || pf.open_hA.z != rA)
            return fail("hA opening bind");
        if (!chkpar(pf.open_hA, pf.n)) return fail("hA opening params");
        if (!p3bf::verify_eval(pf.open_hA, label + "-hA", why)) return false;
        y_hA = pf.open_hA.y;
    }
    gl_t A_r = 0, gk = 1ULL;
    size_t iw = 0, iv = 0;
    if (y_W_out) y_W_out->clear();
    for (uint32_t k = 0; k < pf.c; k++) {
        gl_t yk;
        if (W_roots[k]) {
            if (vlg) {
                yk = pf.yW[iw++];
                tr.absorb("lu-yW", &yk, sizeof(gl_t));
                vlg->add(*W_roots[k], rA, yk);
            } else {
                const auto& op = pf.open_W[iw++];
                if (op.roots.empty() || !(op.roots[0] == *W_roots[k]) || op.z != rA)
                    return fail("W opening bind");
                if (!chkpar(op, pf.n)) return fail("W opening params");
                if (!p3bf::verify_eval(op, label + "-W" + std::to_string(k), why)) return false;
                yk = op.y;
            }
        } else {
            yk = pf.y_virt[iv++];
            tr.absorb("lu-yv", &yk, sizeof(gl_t));
        }
        if (y_W_out) y_W_out->push_back(yk);
        A_r = gl_add(A_r, gl_mul(gk, yk)); gk = gl_mul(gk, gamma);
    }
    gl_t eqA = p3bf::eq_point(rA, zk ? zAf : zA);
    gl_t endA = gl_add(gl_mul(lamA, y_hA),
                       gl_mul(eqA, gl_sub(gl_mul(y_hA, gl_add(A_r, beta)), 1ULL)));
    if (zk) endA = gl_add(endA, gl_mul(rhoA,
                     gl_add(pf.blA.yB[0], gl_mul(eqA, gl_add(pf.blA.yB[1],
                            gl_mul(eqA, pf.blA.yB[2]))))));
    if (claimA != endA) return fail("A terminal");

    std::vector<gl_t> zT = chal_vec(tr, pf.m); gl_t lamT = chal(tr);
    std::vector<gl_t> zTf = p3zkc::zpt(zT);
    gl_t rhoT = 0;
    if (zk) { gl_t H = pf.blT.H; tr.absorb("lu-HT", &H, sizeof(gl_t)); rhoT = chal(tr); }
    std::vector<gl_t> rT; gl_t claimT;
    gl_t claimT0 = gl_add(gl_mul(lamT, pf.S), gl_mul(rhoT, pf.blT.H));
    if (!sc_verify(pf.msgsT, zk ? p3zkc::vfull(pf.m) : pf.m, claimT0, tr, "lu-scT", rT, claimT))
        return fail("sumcheck claim T");
    gl_t y_hT, y_cnt;
    if (vlg) {
        y_hT = pf.y_hT_c; y_cnt = pf.y_cnt_c;
        vlg->add(pf.root_hT, rT, y_hT);
        vlg->add(pf.root_cnt, rT, y_cnt);
        if (zk) for (int j = 0; j < 3; j++) {
            gl_t yb = pf.blT.yB[j];
            tr.absorb("lu-ybT", &yb, sizeof(gl_t));
            vlg->add(pf.blT.rt[j], rT, yb);
        }
    } else {
        if (pf.open_hT.roots.empty() || !(pf.open_hT.roots[0] == pf.root_hT) || pf.open_hT.z != rT)
            return fail("hT opening bind");
        if (pf.open_cnt.roots.empty() || !(pf.open_cnt.roots[0] == pf.root_cnt) || pf.open_cnt.z != rT)
            return fail("cnt opening bind");
        if (!chkpar(pf.open_hT, pf.m) || !chkpar(pf.open_cnt, pf.m)) return fail("T opening params");
        if (!p3bf::verify_eval(pf.open_hT, label + "-hT", why)) return false;
        if (!p3bf::verify_eval(pf.open_cnt, label + "-cnt", why)) return false;
        y_hT = pf.open_hT.y; y_cnt = pf.open_cnt.y;
    }
    // zk: rT spans the augmented domain; build_eq(rT)[j] for j < M already
    // carries the (1-r_ex) factors of the zero-extended public table
    std::vector<gl_t> eqT = p3bf::build_eq(rT);
    gl_t Tc_r = 0; gk = 1ULL;
    for (auto& col : T.cols) {
        gl_t v = 0;
        for (size_t j = 0; j < M; j++) v = gl_add(v, gl_mul(col[j], eqT[j]));
        Tc_r = gl_add(Tc_r, gl_mul(gk, v)); gk = gl_mul(gk, gamma);
    }
    gl_t eqTr = p3bf::eq_point(rT, zk ? zTf : zT);
    gl_t endT = gl_add(gl_mul(lamT, y_hT),
                       gl_mul(eqTr, gl_sub(gl_mul(y_hT, gl_add(Tc_r, beta)), y_cnt)));
    if (zk) endT = gl_add(endT, gl_mul(rhoT,
                     gl_add(pf.blT.yB[0], gl_mul(eqTr, gl_add(pf.blT.yB[1],
                            gl_mul(eqTr, pf.blT.yB[2]))))));
    if (claimT != endT) return fail("T terminal");

    tr.absorb("lu-ends", &y_hA, sizeof(gl_t));
    tr.absorb("lu-ends", &y_hT, sizeof(gl_t));
    tr.absorb("lu-ends", &y_cnt, sizeof(gl_t));
    if (rA_out) *rA_out = rA;
    if (y_virt_out) *y_virt_out = pf.y_virt;
    if (why) *why = "ok";
    return true;
}

// ==================== grouped (merged) lookup instances ====================
// One logUp instance for k same-(table, log-rows) lookups.  Merged index
// u = i | (j<<n) | (ex<<(n+g)), g = ceil log2 k, ex = the zk mask coordinates
// (width E = e_of(n), the MEMBER policy width -- so the shared claim point
// pm = (rA[0..n) || rA[n+g..)) has exactly a standalone lookup's shape).
// Pad members j in [k, 2^g) hold table row 0 on every row (their multiplicity
// is added to cnt[idx of row 0] -- the canonical row-0 index is 0).

static inline gl_t eq_bits(const std::vector<gl_t>& r, uint32_t off, uint32_t g, uint32_t j) {
    gl_t e = 1ULL;
    for (uint32_t t = 0; t < g; t++)
        e = gl_mul(e, (j >> t) & 1 ? r[off + t] : gl_sub(1ULL, r[off + t]));
    return e;
}

// ---- per-TABLE supergroup prover (v3 GKR: commit-free fraction chains) ----
// smi/sns: subgroup member-index lists and their member log-rows; all
// obligations share ONE table.  Each subgroup's A side and the shared T side
// are proven by GKR fractional-sum trees (p3_gkr.cuh) instead of committed
// helper columns + batched cubic sumchecks: no hA/hT commits, no Libra blind
// columns, no helper openings.  Only cnt is committed (augmented, mechanism
// 1), plus in zk mode two uniform mask-leaf stream columns per tree,
// interleaved as the odd sibling of every real leaf:
//   A side, x = i|(j<<n)|(ex<<(n+g)) over 2^(n+g+E):
//     leaf(2x) = ( ex==0 ? 1 : 0 ,  Am(x)+beta )   real / witness-mask rows
//     leaf(2x+1) = ( pmA(x), qmA(x) )              committed uniform masks
//   T side, x = j|(ex<<m) over 2^(m+e2):
//     leaf(2x) = ( cnt_aug(x), ex==0 ? Tc_j+beta : 1 ),  leaf(2x+1) = (pmT,qmT)
// Witness-mask rows contribute ZERO to the A sum (p=0); cnt's mask tail rides
// as beta-INDEPENDENT additive junk Sm_cnt; pmT's last entry is fixed so the
// total mask junk cancels (SmT = sum_s SmA_s - Sm_cnt).  All mask columns are
// committed BEFORE gamma/beta (the logUp Schwartz-Zippel argument needs the
// junk fixed pre-beta).  The multiset identity is checked on the published
// roots: sum_s P_s/Q_s == P_T/Q_T.  ZK: every height-1 tree node is uniform
// (each real leaf has a fresh uniform sibling), so the whole chain above the
// leaves is simulatable; the leaf claims land on chi[ex=0]~ (public), the
// mechanism-1-blinded member/cnt augmented evals, and the mask MLEs.
static inline GroupProof prove_super(fs::Transcript& tr, XCtx& xc,
                                     const std::vector<std::vector<size_t>>& smi,
                                     const std::vector<uint32_t>& sns,
                                     uint32_t R, uint32_t Q, bool strict, bool gpu) {
    struct Tm { double t0; Tm() : t0(lu_now_ms()) {}
                ~Tm() { g_lustats.ms += lu_now_ms() - t0; g_lustats.calls++; } } tm_;
    const bool zk = p3zkc::G.on;
    const Table& T = *xc.luq[smi[0][0]].tab;
    GroupProof pf;
    const uint32_t ns = (uint32_t)smi.size();
    const size_t M = T.cols[0].size();
    const uint32_t m = ilog2(M), c = (uint32_t)T.cols.size();
    pf.m = m; pf.c = c; pf.sub.resize(ns);

    tr.absorb("lug-tab", T.id.data(), 32);
    uint32_t hdr0[3] = {ns, m, c};
    tr.absorb("lug-sdims", hdr0, sizeof hdr0);
    for (uint32_t s = 0; s < ns; s++) {
        const uint32_t n = sns[s], k = (uint32_t)smi[s].size();
        uint32_t g = 0; while ((1u << g) < k) g++;
        const uint32_t E = zk ? p3zkc::e_of(n) : 0;
        const size_t NA = (size_t)1 << (n + E);
        uint32_t hdr[4] = {n, g, k, c};
        tr.absorb("lug-dims", hdr, sizeof hdr);
        for (size_t j : smi[s]) {
            auto& W = xc.luq[j].W;
            if ((uint32_t)W.size() != c) throw std::runtime_error("p3lu: group col count");
            for (auto& w : W) {
                if (luc_len(w) != NA)
                    throw std::runtime_error("p3lu: group member size: " + xc.luq[j].label +
                                             " col=" + std::to_string(&w - W.data()) +
                                             " len=" + std::to_string(luc_len(w)) +
                                             " NA=" + std::to_string(NA));
                if (w.com) tr.absorb("lug-W", w.com->root.data(), 32);
                else       tr.absorb("lug-Wv", "virt", 4);
            }
        }
    }
    // member-column materializer (section 22): committed COMPACT columns and
    // GENERATED virtual columns decode into one reusable scratch; live vectors
    // pass through untouched.  Values are bit-identical to the raw path.
    std::vector<gl_t> wscr;
    auto wcol = [&](const LuColSpec& w, size_t need) -> const std::vector<gl_t>* {
        if (w.com) {
            if (!w.com->pk.on) return &w.com->v;
            wscr.resize(need);
            mat_col_into(*w.com, wscr.data(), need);
            return &wscr;
        }
        if (w.virt) return w.virt;
        wscr.resize(need);
        w.gen(wscr.data(), need);
        return &wscr;
    };

    // summed multiplicities over the union of subgroup rows (pad rows = row 0)
    double zt_cnt0 = p3zp::nowms();
    std::vector<gl_t> cnt(M, 0);
    for (uint32_t s = 0; s < ns; s++) {
        const uint32_t n = sns[s], k = (uint32_t)smi[s].size();
        uint32_t g = 0; while ((1u << g) < k) g++;
        const size_t N = (size_t)1 << n;
        for (size_t j : smi[s]) {
            const uint32_t* I = xc.luq[j].iptr();
            if (xc.luq[j].isize() != N) throw std::runtime_error("p3lu: group idx size");
            for (size_t i = 0; i < N; i++) {
                if (I[i] >= M) throw std::runtime_error("p3lu: idx range");
                cnt[I[i]] = gl_add(cnt[I[i]], 1ULL);
            }
        }
        cnt[0] = gl_add(cnt[0], ((uint64_t)(((size_t)1 << g) - k) << n) % GL_P);
    }
    Col Ccnt = commit_col_nc(cnt, R);
    pf.root_cnt = Ccnt.root; tr.absorb("lug-cnt", pf.root_cnt.data(), 32);
    if (g_free_idx)
        for (auto& mi : smi) for (size_t j : mi) {
            if (xc.luq[j].imap) { xc.luq[j].imap.reset(); continue; }   // spilled
            auto* ix = const_cast<std::vector<uint32_t>*>(xc.luq[j].idx);
            ix->clear(); ix->shrink_to_fit();
        }
    if (p3zp::on()) { p3zp::g.lug_cnt.ms += p3zp::nowms() - zt_cnt0; p3zp::g.lug_cnt.n++; }

    // ---- zk: commit ALL mask-leaf stream columns BEFORE gamma/beta ----
    // (their fraction sums are the blinding junk; committing them pre-beta is
    // what keeps the mask-cancelled logUp identity sound under Schwartz-Zippel)
    // per-subgroup mask streams: only seeds/roots are retained -- values are
    // regenerated on the fly at leaf build and by the opening ledger (at
    // d=1024 the resident pm/qm host columns of all subgroups would be tens
    // of GB)
    struct AMask { uint64_t pseed = 0, qseed = 0, sP = 0, sQ = 0;
                   p3fri::Hash rtP, rtQ; gl_t Sm = 0; };
    std::vector<AMask> am(ns);
    Col PMT, QMT;
    if (zk) {
        double t_mk = lu_now_ms();
        const bool mo = p3zkc::G.mask_on;
        gl_t SmA = 0;
        for (uint32_t s = 0; s < ns; s++) {
            const uint32_t n = sns[s], k = (uint32_t)smi[s].size();
            uint32_t g = 0; while ((1u << g) < k) g++;
            const uint32_t E = p3zkc::e_of(n);
            const size_t NM = (size_t)1 << (n + g + E);
            AMask& a = am[s];
            // multiplicative masks: pm = sm*qm with independent uniform streams
            // sm, qm -- pm/qm = sm, so the mask fraction sum is a PLAIN SUM of
            // the sm stream (no inversions), and (pm,qm) stays a uniform pair
            a.pseed = p3zkc::next_seed(); a.qseed = p3zkc::next_seed();
            // GPU: sm-stream sum, then generate + commit pm and qm entirely on
            // device (identical values/seed order -> identical roots + salts)
            gl_t sm = 0;
            {
                const uint32_t NB = 256;
                gl_t* dblk = p3bf::dmalloc(NB, "lug:smsum");
                p3lu_smsum_kernel<<<NB, 256>>>(a.pseed, mo ? 1 : 0, dblk, NM);
                std::vector<gl_t> hb(NB);
                cudaMemcpy(hb.data(), dblk, (size_t)NB * 8, cudaMemcpyDeviceToHost);
                cudaFreeAsync(dblk, 0);
                for (auto x : hb) sm = gl_add(sm, x);
            }
            a.sP = p3zkc::next_seed();
            double t_c0 = p3zp::nowms();
            gl_t* dm = p3bf::dmalloc(NM, "lug:pmqm");
            p3lu_pmgen_kernel<<<(uint32_t)((NM + 255) / 256), 256>>>(dm, a.pseed, a.qseed, NM, mo ? 1 : 0);
            a.rtP = p3zkc::salted_commit_root_dev(dm, NM, R, a.sP);
            a.sQ = p3zkc::next_seed();
            p3lu_qmgen_kernel<<<(uint32_t)((NM + 255) / 256), 256>>>(dm, a.qseed, NM, mo ? 1 : 0);
            a.rtQ = p3zkc::salted_commit_root_dev(dm, NM, R, a.sQ);
            cudaFreeAsync(dm, 0);
            if (p3zp::on()) { p3zp::g.lug_inv.ms += p3zp::nowms() - t_c0; p3zp::g.lug_inv.n++; }
            pf.sub[s].rt_pm = a.rtP; pf.sub[s].rt_qm = a.rtQ;
            tr.absorb("lug-pm", a.rtP.data(), 32);
            tr.absorb("lug-qm", a.rtQ.data(), 32);
            a.Sm = sm; SmA = gl_add(SmA, sm);
        }
        // T side: qmT stream; pmT stream with the LAST entry fixed so the total
        // mask junk cancels: SmT = sum_s SmA_s - Sm_cnt
        const uint32_t e2 = p3zkc::e_of(m);
        const size_t MT = (size_t)1 << (m + e2);
        uint64_t pseedT = p3zkc::next_seed(), qseedT = p3zkc::next_seed();
        PMT.v.resize(MT); QMT.v.resize(MT);
        gl_t Smcnt = 0;
        for (size_t i = M; i < Ccnt.v.size(); i++) Smcnt = gl_add(Smcnt, Ccnt.v[i]);
        gl_t part = 0;
        for (size_t i = 0; i < MT; i++) {
            gl_t smt = mo ? p3zkc::zprng_at(pseedT, i) : 0ULL;
            QMT.v[i] = mo ? p3zkc::zprng_at(qseedT, i) : 1ULL;
            if (i + 1 < MT) { PMT.v[i] = gl_mul(smt, QMT.v[i]); part = gl_add(part, smt); }
        }
        // last mask fraction fixed so the junk cancels: smT_last = target - part
        PMT.v[MT - 1] = gl_mul(gl_sub(gl_sub(SmA, Smcnt), part), QMT.v[MT - 1]);
        PMT.vreal = m + e2; PMT.sseed = p3zkc::next_seed();
        PMT.root = p3zkc::salted_commit_root(PMT.v, R, PMT.sseed);
        QMT.vreal = m + e2; QMT.sseed = p3zkc::next_seed();
        QMT.root = p3zkc::salted_commit_root(QMT.v, R, QMT.sseed);
        pf.rt_pmT = PMT.root; pf.rt_qmT = QMT.root;
        tr.absorb("lug-pmT", PMT.root.data(), 32);
        tr.absorb("lug-qmT", QMT.root.data(), 32);
        if (p3zp::on()) { p3zp::g.lug_hcom.ms += lu_now_ms() - t_mk; p3zp::g.lug_hcom.n++; }
    }

    gl_t gamma = chal(tr), beta = chal(tr);
    std::vector<gl_t> gp(c); { gl_t w = 1ULL; for (uint32_t t = 0; t < c; t++) { gp[t] = w; w = gl_mul(w, gamma); } }
    gl_t a0 = 0; for (uint32_t t = 0; t < c; t++) a0 = gl_add(a0, gl_mul(gp[t], T.cols[t][0]));

    // T-side combined table -- ONCE per table (no helper inverses in v3)
    std::vector<const std::vector<gl_t>*> tp;
    for (auto& t : T.cols) tp.push_back(&t);
    std::vector<gl_t> Tc = combine(tp, gamma);

    gl_t accP = 0, accQ = 1ULL;            // exact running fraction sum_s P_s/Q_s

    // ---- per-subgroup A-sides (each fully proven, then its working set drops) --
    for (uint32_t s = 0; s < ns; s++) {
        const std::vector<size_t>& mi = smi[s];
        LuSubA& sp = pf.sub[s];
        const uint32_t n = sns[s], k = (uint32_t)mi.size();
        uint32_t g = 0; while ((1u << g) < k) g++;
        const uint32_t kp = 1u << g;
        const size_t N = (size_t)1 << n;
        const uint32_t E = zk ? p3zkc::e_of(n) : 0;
        const size_t NR = (size_t)1 << (n + g);
        const size_t NM = (size_t)1 << (n + g + E);
        sp.n = n; sp.k = k; sp.mem.resize(k);
        if (p3bf::memlog())
            fprintf(stderr, "# lu group %s: n=%u g=%u k=%u E=%u NM=2^%u (%.2f GB host Am)\n",
                    xc.luq[mi[0]].label.c_str(), n, g, k, E, n + g + E,
                    (double)NM * 8 / 1073741824.0);

        // merged combined witness A over the subgroup's full merged domain.
        // COLUMN-major accumulation: each member column is materialized (or
        // borrowed) exactly ONCE and gamma-axpy'd into its member rows across
        // all ex slices -- only the per-element ADD ORDER differs from the
        // row-major build (exact field adds: identical values, transcript
        // unchanged), and compact/generated columns never coexist.
        // Device path (devleaf groups, section 23): Am is built ON DEVICE --
        // compacted member columns rematerialize via mat_col_range_dev (packed
        // upload + device unpack + device mask PRNG), raw/virtual/generated
        // columns upload once -- and the leaves consume dAm in place.  The
        // host Am array and its NM-sized upload disappear.
        double t_am = lu_now_ms();
        const size_t NAf = (size_t)1 << (n + E);       // full member column length
        const bool devleaf = zk && gpu && 2 * NM >= p3gkr::GKR_FULLGPU_MIN;
        static const bool devam_on = [] {
            const char* e = getenv("P3_LUG_DEVAM"); return !e || atoi(e) != 0; }();
        const bool devam = devleaf && devam_on;
        std::vector<gl_t> Am;
        gl_t* dAm = nullptr;
        if (devam) {
            dAm = p3bf::dmalloc(NM, "lug:dAm");
            p3lu_amfill_kernel<<<(uint32_t)((NM + 255) / 256), 256>>>(
                dAm, a0, n, g, k, NM);
            gl_t* dcol = p3bf::dmalloc(NAf, "lug:amcol");
            for (uint32_t j = 0; j < k; j++) {
                auto& W = xc.luq[mi[j]].W;
                for (uint32_t t = 0; t < c; t++) {
                    if (W[t].com && W[t].com->pk.on) {
                        mat_col_range_dev(*W[t].com, 0, dcol, NAf);
                    } else {
                        const std::vector<gl_t>& v = *wcol(W[t], NAf);
                        cudaMemcpy(dcol, v.data(), NAf * 8, cudaMemcpyHostToDevice);
                    }
                    p3lu_amaxpy_kernel<<<(uint32_t)((NAf + 255) / 256), 256>>>(
                        dAm, dcol, gp[t], n, g, j, NAf);
                }
            }
            cudaFreeAsync(dcol, 0);
            p3bf::ckcuda("lug:amdev");
        } else {
            Am.assign(NM, 0);
            for (uint32_t ex = 0; ex < (1u << E); ex++)
                for (uint32_t j = k; j < kp; j++) {
                    gl_t* dst = Am.data() + (((size_t)ex << (n + g)) | ((size_t)j << n));
                    #pragma omp parallel for schedule(static) if (N >= 65536) num_threads(p3bf::nthr(N))
                    for (size_t i = 0; i < N; i++) dst[i] = a0;
                }
            for (uint32_t j = 0; j < k; j++) {
                auto& W = xc.luq[mi[j]].W;
                for (uint32_t t = 0; t < c; t++) {
                    const std::vector<gl_t>& v = *wcol(W[t], NAf);
                    for (uint32_t ex = 0; ex < (1u << E); ex++) {
                        gl_t* dst = Am.data() + (((size_t)ex << (n + g)) | ((size_t)j << n));
                        const gl_t* src = v.data() + ((size_t)ex << n);
                        const gl_t gpt = gp[t];
                        #pragma omp parallel for schedule(static) if (N >= 65536) num_threads(p3bf::nthr(N))
                        for (size_t i = 0; i < N; i++)
                            dst[i] = gl_add(dst[i], gl_mul(gpt, src[i]));
                    }
                }
            }
        }
        double t_inv = lu_now_ms();
        const double am_ms = t_inv - t_am;
        if (p3zp::on()) { p3zp::g.lug_am.ms += am_ms; p3zp::g.lug_am.n++; }

        // ---- GKR leaves: even = real/witness-mask rows, odd = zk mask siblings
        double t_sc = lu_now_ms();
        double t_leaf0 = p3zp::nowms();
        std::vector<gl_t> LP, LQ;
        gl_t *dLP = nullptr, *dLQ = nullptr;
        if (devleaf) {
            // leaves built directly on device (bit-identical values); Am
            // either was built on device (devam) or uploads once -- never
            // materialized as host LP/LQ
            AMask& a = am[s];
            const bool mo = p3zkc::G.mask_on;
            if (!dAm) {
                dAm = p3bf::dmalloc(NM, "lug:dAm");
                cudaMemcpy(dAm, Am.data(), NM * 8, cudaMemcpyHostToDevice);
            }
            dLP = p3bf::dmalloc(2 * NM, "lug:dLP");
            dLQ = p3bf::dmalloc(2 * NM, "lug:dLQ");
            p3lu_zkleaf_kernel<<<(uint32_t)((NM + 255) / 256), 256>>>(
                dAm, beta, a.pseed, a.qseed, mo ? 1 : 0, NR, dLP, dLQ, NM);
            cudaFreeAsync(dAm, 0); dAm = nullptr;
        } else if (zk) {
            AMask& a = am[s];
            const bool mo = p3zkc::G.mask_on;
            LP.resize(2 * NM); LQ.resize(2 * NM);
            #pragma omp parallel for schedule(static) if (NM >= 65536) num_threads(p3bf::nthr(NM))
            for (size_t x = 0; x < NM; x++) {
                LP[2*x]   = x < NR ? 1ULL : 0ULL;
                LQ[2*x]   = gl_add(Am[x], beta);
                gl_t qm = mo ? p3zkc::zprng_at(a.qseed, x) : 1ULL;
                LP[2*x+1] = mo ? gl_mul(p3zkc::zprng_at(a.pseed, x), qm) : 0ULL;
                LQ[2*x+1] = qm;
            }
        } else {
            LP.assign(NR, 1ULL); LQ.resize(NR);
            #pragma omp parallel for schedule(static) if (NR >= 65536) num_threads(p3bf::nthr(NR))
            for (size_t x = 0; x < NR; x++) LQ[x] = gl_add(Am[x], beta);
        }
        std::vector<gl_t>().swap(Am);
        if (p3zp::on()) { p3zp::g.lug_blA.ms += p3zp::nowms() - t_leaf0; p3zp::g.lug_blA.n++; }
        std::vector<gl_t> rA; gl_t cpA = 0, cqA = 0;
        sp.gk = devleaf
              ? p3gkr::prove_dev(tr, "lug-gA", dLP, dLQ, 2 * NM, !zk, rA, &cpA, &cqA)
              : p3gkr::prove(tr, "lug-gA", std::move(LP), std::move(LQ), !zk,
                             rA, &cpA, &cqA, gpu);
        accP = gl_add(gl_mul(accP, sp.gk.Q), gl_mul(sp.gk.P, accQ));
        accQ = gl_mul(accQ, sp.gk.Q);
        const double sc_ms = lu_now_ms() - t_sc;
        g_lustats.sc_ms += sc_ms;
        if (p3zp::on()) { p3zp::g.lug_scA.ms += sc_ms; p3zp::g.lug_scA.n++; }
        double t_cl = lu_now_ms();

        // shared member point pm = (rA[0..n) || rA[n+g..))
        std::vector<gl_t> pm(rA.begin(), rA.begin() + n);
        pm.insert(pm.end(), rA.begin() + n + g, rA.end());
        const size_t NAc = (size_t)1 << pm.size();
        const bool deveq = gpu && NAc >= ((size_t)1 << 15);
        double t_eq0 = p3zp::nowms();
        std::vector<gl_t> eqpm;
        if (!deveq) eqpm = p3bf::build_eq(pm);   // host fallback path only
        if (p3zp::on()) { p3zp::g.lug_blT.ms += p3zp::nowms() - t_eq0; p3zp::g.lug_blT.n++; }

        // zk: ledger-open the two mask columns at rA (claims = the published
        // last-layer terminals); pure zprng streams -> drop + regenerate
        if (zk) {
            AMask& a = am[s];
            const p3gkr::Lay& lyA = sp.gk.lay.back();
            size_t NM3 = NM; bool mo2 = p3zkc::G.mask_on;
            {
                Col pc; pc.root = a.rtP; pc.vreal = n + g + E; pc.sseed = a.sP;
                uint64_t sd = a.pseed, sq = a.qseed;
                xc.keep.push_back(std::move(pc));
                xc.lg.add(&xc.keep.back().v, sp.rt_pm, rA, lyA.p1, a.sP,
                          [NM3, sd, sq, mo2](gl_t* b, size_t n) {
                    (void)NM3;
                    #pragma omp parallel for schedule(static) if (n >= 65536) num_threads(p3bf::nthr(n))
                    for (size_t i = 0; i < n; i++)
                        b[i] = mo2 ? gl_mul(p3zkc::zprng_at(sd, i), p3zkc::zprng_at(sq, i)) : 0ULL;
                }, [sd, sq, mo2](gl_t* b, size_t n) {
                    p3lu_pmgen_kernel<<<(uint32_t)((n + 255) / 256), 256>>>(b, sd, sq, n, mo2 ? 1 : 0);
                });
            }
            {
                Col qc; qc.root = a.rtQ; qc.vreal = n + g + E; qc.sseed = a.sQ;
                uint64_t sd = a.qseed;
                xc.keep.push_back(std::move(qc));
                xc.lg.add(&xc.keep.back().v, sp.rt_qm, rA, lyA.q1, a.sQ,
                          [NM3, sd, mo2](gl_t* b, size_t n) {
                    (void)NM3;
                    #pragma omp parallel for schedule(static) if (n >= 65536) num_threads(p3bf::nthr(n))
                    for (size_t i = 0; i < n; i++) b[i] = mo2 ? p3zkc::zprng_at(sd, i) : 1ULL;
                }, [sd, mo2](gl_t* b, size_t n) {
                    p3lu_qmgen_kernel<<<(uint32_t)((n + 255) / 256), 256>>>(b, sd, n, mo2 ? 1 : 0);
                });
            }
        }
        // per-member witness claims at pm (+ the member's bind callback).
        // values first (independent exact dot products -- identical field
        // elements in any order and on either device), absorbed in order
        std::vector<gl_t> ymem((size_t)k * c);
        // device dots from 2^15 up, against a DEVICE-BUILT eq(pm) column (the
        // host build_eq of a 2^26 eq vector was ~20 s of the d=1024 proof)
        if (deveq) {
            const uint32_t NB = 256;
            gl_t* deq = p3bf::dmalloc(NAc, "lug:eqpm");
            {
                gl_t* dz = p3bf::dmalloc(pm.size(), "lug:pmz");
                cudaMemcpy(dz, pm.data(), pm.size() * 8, cudaMemcpyHostToDevice);
                p3bf::p3bf_eq_kernel<<<(uint32_t)((NAc + 255) / 256), 256>>>(
                    dz, deq, (uint32_t)pm.size(), (uint32_t)NAc);
                cudaFreeAsync(dz, 0);
            }
            gl_t* dcol = p3bf::dmalloc(NAc, "lug:ycol");
            gl_t* dblk = p3bf::dmalloc(NB, "lug:yblk");
            std::vector<gl_t> hb(NB);
            for (size_t jt = 0; jt < (size_t)k * c; jt++) {
                auto& W = xc.luq[mi[jt / c]].W;
                uint32_t t = (uint32_t)(jt % c);
                if (W[t].com && W[t].com->pk.on) {
                    // packed upload + device unpack/PRNG: identical values,
                    // 2-8x less PCIe than staging raw elements via the host
                    mat_col_range_dev(*W[t].com, 0, dcol, NAc);
                } else {
                    const std::vector<gl_t>& v = *wcol(W[t], NAc);
                    cudaMemcpy(dcol, v.data(), NAc * 8, cudaMemcpyHostToDevice);
                }
                p3bf::p3bf_dot_kernel<<<NB, 256>>>(dcol, deq, dblk, (uint32_t)NAc);
                cudaMemcpy(hb.data(), dblk, (size_t)NB * 8, cudaMemcpyDeviceToHost);
                gl_t sacc = 0; for (auto x : hb) sacc = gl_add(sacc, x);
                ymem[jt] = sacc;
            }
            cudaFreeAsync(deq, 0); cudaFreeAsync(dcol, 0); cudaFreeAsync(dblk, 0);
            p3bf::ckcuda("lug:claims");
        } else {
            #pragma omp parallel for schedule(dynamic) if ((size_t)k * c > 4) num_threads(p3bf::nthr((size_t)k * c * NAc))
            for (size_t jt = 0; jt < (size_t)k * c; jt++) {
                auto& W = xc.luq[mi[jt / c]].W;
                uint32_t t = (uint32_t)(jt % c);
                // thread-local materialization (small columns; the shared
                // wscr scratch is not safe under this parallel loop)
                std::vector<gl_t> loc;
                const std::vector<gl_t>* v;
                if (W[t].com && !W[t].com->pk.on) v = &W[t].com->v;
                else if (!W[t].com && W[t].virt) v = W[t].virt;
                else {
                    loc.resize(NAc);
                    if (W[t].com) mat_col_into(*W[t].com, loc.data(), NAc);
                    else W[t].gen(loc.data(), NAc);
                    v = &loc;
                }
                ymem[jt] = p3bf::eval_h(*v, eqpm);
            }
        }
        for (uint32_t j = 0; j < k; j++) {
            auto& W = xc.luq[mi[j]].W;
            for (uint32_t t = 0; t < c; t++) {
                gl_t y = ymem[(size_t)j * c + t];
                if (W[t].com) {
                    sp.mem[j].yW.push_back(y);
                    tr.absorb("lug-yW", &y, sizeof(gl_t));
                    xc.lg.add(&W[t].com->v, W[t].com->root, pm, y, W[t].com->sseed);
                } else {
                    sp.mem[j].y_virt.push_back(y);
                    tr.absorb("lug-yv", &y, sizeof(gl_t));
                }
            }
            if (xc.luq[mi[j]].bind) {
                double t_b0 = p3zp::nowms();
                sp.mem[j].extra = xc.luq[mi[j]].bind(tr, xc, pm, sp.mem[j].y_virt);
                if (p3zp::on()) { p3zp::g.lug_blH.ms += p3zp::nowms() - t_b0; p3zp::g.lug_blH.n++; }
            }
        }
        const double cl_ms = lu_now_ms() - t_cl;
        if (p3zp::on()) { p3zp::g.lug_claims.ms += cl_ms; p3zp::g.lug_claims.n++; }
        if (p3bf::memlog() && NM >= ((size_t)1 << 22))
            fprintf(stderr, "# lu group %s timing: am=%.0f gkr=%.0f claims=%.0f ms\n",
                    xc.luq[mi[0]].label.c_str(), am_ms, sc_ms, cl_ms);
        if (NM >= ((size_t)1 << 24)) p3bf::trim_heap();
    }

    // ---- ONE T-side per table ----
    double t_scT = lu_now_ms();
    const size_t MTf = zk ? Ccnt.v.size() : M;   // zk: augmented cnt domain 2^(m+e2)
    std::vector<gl_t> LPT, LQT;
    if (zk) {
        LPT.resize(2 * MTf); LQT.resize(2 * MTf);
        for (size_t x = 0; x < MTf; x++) {
            LPT[2*x]   = Ccnt.v[x];
            LQT[2*x]   = x < M ? gl_add(Tc[x], beta) : 1ULL;
            LPT[2*x+1] = PMT.v[x];
            LQT[2*x+1] = QMT.v[x];
        }
    } else {
        LPT = cnt; LQT.resize(M);
        for (size_t j = 0; j < M; j++) LQT[j] = gl_add(Tc[j], beta);
    }
    std::vector<gl_t> rT; gl_t cpT = 0, cqT = 0;
    pf.gkT = p3gkr::prove(tr, "lug-gT", std::move(LPT), std::move(LQT), !zk,
                          rT, &cpT, &cqT, gpu);
    g_lustats.scg_ms += lu_now_ms() - t_scT;
    if (p3zp::on()) { p3zp::g.lug_scT.ms += lu_now_ms() - t_scT; p3zp::g.lug_scT.n++; }

    // multiset identity on the exact tree roots (masks cancel by construction)
    if (strict && gl_mul(accP, pf.gkT.Q) != gl_mul(pf.gkT.P, accQ))
        throw std::runtime_error("p3lu: witness not in table: " + xc.luq[smi[0][0]].label);

    double zt_tev0 = p3zp::nowms();
    // cnt claim: zk = the even-leaf terminal at rT; non-zk = the combined p claim
    gl_t y_cnt = zk ? pf.gkT.lay.back().p0 : cpT;
    {
        uint64_t ssC = Ccnt.sseed;
        xc.keep.push_back(std::move(Ccnt));
        xc.lg.add(&xc.keep.back().v, pf.root_cnt, rT, y_cnt, ssC);
    }
    if (zk) {
        const p3gkr::Lay& lyT = pf.gkT.lay.back();
        uint64_t ssP = PMT.sseed, ssQ = QMT.sseed;
        xc.keep.push_back(std::move(PMT));
        xc.lg.add(&xc.keep.back().v, pf.rt_pmT, rT, lyT.p1, ssP);
        xc.keep.push_back(std::move(QMT));
        xc.lg.add(&xc.keep.back().v, pf.rt_qmT, rT, lyT.q1, ssQ);
    }
    if (p3zp::on()) { p3zp::g.lug_tev.ms += p3zp::nowms() - zt_tev0; p3zp::g.lug_tev.n++; }
    return pf;
}

// group the queued obligations by (table id, log-rows), first-appearance order.
// LU_GCAP caps the STACKED group domain: a k-member group over 2^n rows is
// proven on 2^(n+g) merged rows (g = ceil log2 k) and its helper columns live
// there; at llama-68m dims an uncapped stack exceeds the card, so groups are
// split deterministically (first-appearance order, chunks of 2^(LU_GCAP-n))
// with n >= LU_GCAP members proving as singletons.  Public protocol constant:
// prover and verifier derive the same split.  Inactive at the tiny-layer dims
// (all n + g <= 26 there), so historical transcripts are unchanged.
static const uint32_t LU_GCAP = 25;   // v3: caps NM=2^(GCAP+E) so the mask-
// stream commits stay within the proven 2^28-leaf codeword territory
template <typename OBL>
static inline void lu_groups(const std::vector<OBL>& q,
                             std::function<uint32_t(const OBL&)> n_of,
                             std::vector<std::vector<size_t>>& members,
                             std::vector<uint32_t>& gn) {
    std::vector<std::vector<size_t>> mem0;
    std::vector<uint32_t> gn0;
    std::vector<p3fri::Hash> gid;
    for (size_t i = 0; i < q.size(); i++) {
        uint32_t n = n_of(q[i]);
        size_t gi = 0;
        for (; gi < gid.size(); gi++) if (gid[gi] == q[i].tab->id && gn0[gi] == n) break;
        if (gi == gid.size()) { gid.push_back(q[i].tab->id); gn0.push_back(n); mem0.push_back({}); }
        mem0[gi].push_back(i);
    }
    for (size_t gi = 0; gi < mem0.size(); gi++) {
        uint32_t n = gn0[gi];
        size_t cap = n >= LU_GCAP ? 1 : (size_t)1 << (LU_GCAP - n);
        for (size_t off = 0; off < mem0[gi].size(); off += cap) {
            size_t end = std::min(off + cap, mem0[gi].size());
            members.emplace_back(mem0[gi].begin() + off, mem0[gi].begin() + end);
            gn.push_back(n);
        }
    }
}

// bundle the (table, log-rows) subgroups by TABLE, first-appearance order
// (over the subgroup order lu_groups already fixed) -- public deterministic
// rule, mirrored by prover and verifier.
template <typename OBL>
static inline void lu_supers(const std::vector<OBL>& q,
                             std::function<uint32_t(const OBL&)> n_of,
                             std::vector<std::vector<std::vector<size_t>>>& smem,
                             std::vector<std::vector<uint32_t>>& sgn) {
    std::vector<std::vector<size_t>> members; std::vector<uint32_t> gn;
    lu_groups<OBL>(q, std::move(n_of), members, gn);
    std::vector<p3fri::Hash> tid;
    for (size_t gi = 0; gi < members.size(); gi++) {
        const p3fri::Hash& id = q[members[gi][0]].tab->id;
        size_t si = 0;
        for (; si < tid.size(); si++) if (tid[si] == id) break;
        if (si == tid.size()) { tid.push_back(id); smem.push_back({}); sgn.push_back({}); }
        smem[si].push_back(std::move(members[gi]));
        sgn[si].push_back(gn[gi]);
    }
}

static inline std::vector<GroupProof> lu_flush(fs::Transcript& tr, XCtx& xc,
                                               uint32_t R, uint32_t Q, bool strict,
                                               bool gpu = true) {
    std::vector<std::vector<std::vector<size_t>>> smem;
    std::vector<std::vector<uint32_t>> sgn;
    lu_supers<LuObl>(xc.luq, [](const LuObl& o) { return ilog2(o.isize()); }, smem, sgn);
    std::vector<GroupProof> out;
    for (size_t si = 0; si < smem.size(); si++)
        out.push_back(prove_super(tr, xc, smem[si], sgn[si], R, Q, strict, gpu));
    xc.luq.clear();
    // the deferred virtual-column arrays exist ONLY to be read by the flush
    // (their claims are bound to base commitments; they are never in the
    // opening ledger) -- release them here instead of holding gigabytes of
    // broadcast columns until the end of the proof (section 22)
    for (auto& a : xc.varena) { a.clear(); a.shrink_to_fit(); }
    return out;
}

static inline bool verify_super(fs::Transcript& tr, VCtx& V,
                                const std::vector<std::vector<size_t>>& smi,
                                const std::vector<uint32_t>& sns,
                                const GroupProof& pf,
                                uint32_t Q_pub, uint32_t R_pub, const char** why) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    const bool zk = p3zkc::G.on;
    const Table& T = *V.luq[smi[0][0]].tab;
    const size_t M = T.cols[0].size();
    const uint32_t m = ilog2(M), c = (uint32_t)T.cols.size();
    const uint32_t ns = (uint32_t)smi.size();
    const uint32_t Q_MIN = 20;
    if (Q_pub < Q_MIN || R_pub < 1) return fail("insecure params");
    if (pf.m != m || pf.c != c || pf.sub.size() != ns) return fail("group dims");

    tr.absorb("lug-tab", T.id.data(), 32);
    uint32_t hdr0[3] = {ns, m, c};
    tr.absorb("lug-sdims", hdr0, sizeof hdr0);
    for (uint32_t s = 0; s < ns; s++) {
        const uint32_t n = sns[s], k = (uint32_t)smi[s].size();
        uint32_t g = 0; while ((1u << g) < k) g++;
        if (pf.sub[s].n != n || pf.sub[s].k != k) return fail("group dims");
        if (pf.sub[s].mem.size() != k) return fail("group member count");
        uint32_t hdr[4] = {n, g, k, c};
        tr.absorb("lug-dims", hdr, sizeof hdr);
        for (size_t j : smi[s]) {
            auto& roots = V.luq[j].roots;
            if ((uint32_t)roots.size() != c) return fail("group root count");
            for (auto* r : roots) {
                if (r) tr.absorb("lug-W", r->data(), 32);
                else   tr.absorb("lug-Wv", "virt", 4);
            }
        }
    }
    tr.absorb("lug-cnt", pf.root_cnt.data(), 32);
    if (zk) {
        for (uint32_t s = 0; s < ns; s++) {
            tr.absorb("lug-pm", pf.sub[s].rt_pm.data(), 32);
            tr.absorb("lug-qm", pf.sub[s].rt_qm.data(), 32);
        }
        tr.absorb("lug-pmT", pf.rt_pmT.data(), 32);
        tr.absorb("lug-qmT", pf.rt_qmT.data(), 32);
    }
    gl_t gamma = chal(tr), beta = chal(tr);
    std::vector<gl_t> gp(c); { gl_t w = 1ULL; for (uint32_t t = 0; t < c; t++) { gp[t] = w; w = gl_mul(w, gamma); } }
    gl_t a0 = 0; for (uint32_t t = 0; t < c; t++) a0 = gl_add(a0, gl_mul(gp[t], T.cols[t][0]));

    gl_t accP = 0, accQ = 1ULL;
    for (uint32_t s = 0; s < ns; s++) {
        const std::vector<size_t>& mi = smi[s];
        const LuSubA& sb = pf.sub[s];
        const uint32_t n = sns[s], k = (uint32_t)mi.size();
        uint32_t g = 0; while ((1u << g) < k) g++;
        const uint32_t kp = 1u << g;
        const uint32_t E = zk ? p3zkc::e_of(n) : 0;

        // GKR fraction chain: root (P,Q) -> leaf claims at rA
        const uint32_t L = n + g + E + (zk ? 1 : 0);
        std::vector<gl_t> rA; gl_t o4[4];
        if (!p3gkr::verify(tr, "lug-gA", L, sb.gk, !zk, rA, o4, why)) return false;
        accP = gl_add(gl_mul(accP, sb.gk.Q), gl_mul(sb.gk.P, accQ));
        accQ = gl_mul(accQ, sb.gk.Q);
        if (zk) {                       // ledger order matches the prover: masks
            V.vlg.add(sb.rt_pm, rA, o4[1]);   // first, then the member claims
            V.vlg.add(sb.rt_qm, rA, o4[3]);
        }

        std::vector<gl_t> pm(rA.begin(), rA.begin() + n);
        pm.insert(pm.end(), rA.begin() + n + g, rA.end());
        gl_t A_r = 0;
        for (uint32_t j = 0; j < k; j++) {
            auto& roots = V.luq[mi[j]].roots;
            size_t iw = 0, iv = 0;
            gl_t yj = 0;
            for (uint32_t t = 0; t < c; t++) {
                gl_t y;
                if (roots[t]) {
                    if (iw >= sb.mem[j].yW.size()) return fail("group yW count");
                    y = sb.mem[j].yW[iw++];
                    tr.absorb("lug-yW", &y, sizeof(gl_t));
                    V.vlg.add(*roots[t], pm, y);
                } else {
                    if (iv >= sb.mem[j].y_virt.size()) return fail("group y_virt count");
                    y = sb.mem[j].y_virt[iv++];
                    tr.absorb("lug-yv", &y, sizeof(gl_t));
                }
                yj = gl_add(yj, gl_mul(gp[t], y));
            }
            if (iw != sb.mem[j].yW.size() || iv != sb.mem[j].y_virt.size())
                return fail("group claim count");
            A_r = gl_add(A_r, gl_mul(eq_bits(rA, n, g, j), yj));
            if (V.luq[mi[j]].bind &&
                !V.luq[mi[j]].bind(tr, V, pm, sb.mem[j].y_virt, sb.mem[j].extra, why))
                return false;
        }
        for (uint32_t j = k; j < kp; j++)
            A_r = gl_add(A_r, gl_mul(eq_bits(rA, n, g, j), a0));
        // leaf-claim binding: even-p = chi[ex=0]~ (public), even-q = Am~ + beta
        // (the gamma-combined member claims), odd = the committed mask MLEs
        if (zk) {
            gl_t eq0 = 1ULL;
            for (uint32_t t = n + g; t < n + g + E; t++)
                eq0 = gl_mul(eq0, gl_sub(1ULL, rA[t]));
            if (o4[0] != eq0) return fail("group A p-claim");
            if (o4[2] != gl_add(A_r, beta)) return fail("group A terminal");
        } else {
            if (o4[0] != 1ULL) return fail("group A p-claim");
            if (o4[2] != gl_add(A_r, beta)) return fail("group A terminal");
        }
    }

    // ---- ONE T-side per table ----
    const uint32_t e2 = zk ? p3zkc::e_of(m) : 0;
    const uint32_t LT = m + e2 + (zk ? 1 : 0);
    std::vector<gl_t> rT; gl_t t4[4];
    if (!p3gkr::verify(tr, "lug-gT", LT, pf.gkT, !zk, rT, t4, why)) return false;
    // public combined-table eval at the first m coordinates
    std::vector<gl_t> rTm(rT.begin(), rT.begin() + m);
    std::vector<gl_t> eqT = p3bf::build_eq(rTm);
    gl_t Tc_r = 0;
    for (uint32_t t = 0; t < c; t++) {
        gl_t v = 0;
        for (size_t j = 0; j < M; j++) v = gl_add(v, gl_mul(T.cols[t][j], eqT[j]));
        Tc_r = gl_add(Tc_r, gl_mul(gp[t], v));
    }
    if (zk) {
        gl_t eq0 = 1ULL;
        for (uint32_t t = m; t < m + e2; t++) eq0 = gl_mul(eq0, gl_sub(1ULL, rT[t]));
        gl_t expect = gl_add(gl_mul(eq0, gl_add(Tc_r, beta)), gl_sub(1ULL, eq0));
        if (t4[2] != expect) return fail("group T terminal");
        V.vlg.add(pf.root_cnt, rT, t4[0]);      // cnt_aug~(rT), mechanism-1 blinded
        V.vlg.add(pf.rt_pmT, rT, t4[1]);
        V.vlg.add(pf.rt_qmT, rT, t4[3]);
    } else {
        if (t4[2] != gl_add(Tc_r, beta)) return fail("group T terminal");
        V.vlg.add(pf.root_cnt, rT, t4[0]);
    }

    // multiset identity on the published tree roots
    if (gl_mul(accP, pf.gkT.Q) != gl_mul(pf.gkT.P, accQ)) return fail("group multiset");
    return true;
}

static inline bool lu_verify_flush(fs::Transcript& tr, VCtx& V,
                                   const std::vector<GroupProof>& gps,
                                   uint32_t Q_pub, uint32_t R_pub, const char** why) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    std::vector<std::vector<std::vector<size_t>>> smem;
    std::vector<std::vector<uint32_t>> sgn;
    lu_supers<VLuObl>(V.luq, [](const VLuObl& o) { return o.n; }, smem, sgn);
    if (gps.size() != smem.size()) return fail("lookup group count");
    for (size_t si = 0; si < smem.size(); si++)
        if (!verify_super(tr, V, smem[si], sgn[si], gps[si], Q_pub, R_pub, why))
            return false;
    V.luq.clear();
    return true;
}

} // namespace p3lu
