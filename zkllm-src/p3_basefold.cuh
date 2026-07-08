// P3.4 Basefold multilinear-evaluation opening over Goldilocks.
//
// Commit: RS-encode the length-N=2^v coefficient vector c (root = roots[0]); the
//   committed polynomial is h = multilinear extension of the array c on the cube.
// Open at z in F^v with claimed value y = h(z) = sum_b c[b]*eq(b,z):
//   v rounds, each binding variable r.  Per round we (a) send the degree-2
//   sumcheck message of  sum_b c[b]*eq(b,z), get challenge alpha_r, then (b)
//   MLE-fold the codeword with the SAME alpha_r.  The MLE fold tracks the
//   alpha-binding of c, so the final folded constant C = c~(alpha) and the
//   running sumcheck claim must equal C * eq(alpha,z).
//   Q FRI queries authenticate the folds against roots[0..v-1] (the commitment).
//
// Soundness note: challenges are drawn from the 64-bit base field, giving ~2^-58
// for our round counts; a degree-2 extension (mechanical) lifts this to ~2^-116
// and is the one production change flagged for this module.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if defined(__GLIBC__) || defined(__linux__)
#include <malloc.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "p3_goldilocks.cuh"
#include "p3_fri.cuh"
#include "p3_ntt.cuh"
#include "fs_transcript.hpp"

namespace p3bf {
using p3fri::Hash; using p3fri::Merkle; using p3fri::RoundOpen; using p3fri::QueryProof;
using p3fri::leaf_hash; using p3fri::verify_path; using p3fri::fold; using p3fri::idx_from;

// Loud CUDA failure surface.  The p3 stack cannot recover from a failed device
// allocation; an UNCHECKED failure silently corrupts every downstream proof
// message (the prover completes and the verifier rejects an honest proof).
// Throw at the failure point instead.
static inline void ckcuda(const char* tag) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("p3: CUDA error at ") + tag + ": " +
                                 cudaGetErrorString(e));
}
static inline gl_t* dmalloc(size_t nelem, const char* tag) {
    void* p = nullptr;
    cudaError_t e = cudaMallocAsync(&p, nelem * sizeof(gl_t), 0);
    if (e != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error(std::string("p3: device alloc failed at ") + tag +
                                 " (" + std::to_string(nelem * sizeof(gl_t)) + " bytes): " +
                                 cudaGetErrorString(e));
    }
    return (gl_t*)p;
}
// memory-shape diagnostics for the scale work (stderr, off unless P3_MEMLOG=1)
static inline bool memlog() { static int v = -1;
    if (v < 0) { const char* e = getenv("P3_MEMLOG"); v = e && *e == '1' ? 1 : 0; }
    return v == 1; }
// return freed glibc-arena memory to the OS at phase boundaries -- with 128
// OpenMP threads the retained arenas otherwise hold GBs against the 41 GB
// container cap.  No-op cost when there is nothing to trim.
static inline void trim_heap() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}
static inline void rsslog(const char* tag) {
    if (!memlog()) return;
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[128]; size_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "VmRSS: %zu kB", &kb) == 1) break;
    fclose(f);
    fprintf(stderr, "# rss %.1f GB at %s\n", kb / 1048576.0, tag);
}

// adaptive OpenMP team size: one thread per ~2^14 elements of work, capped at
// the machine.  Tiny loops stay serial (a 128-thread fork/join costs ~10 ms on
// this box -- it dominated the whole prover at small dims); big loops keep the
// full machine.  Team size never changes the computed values: every parallel
// loop here partitions work by a FIXED block count, so this is pure scheduling.
static inline int nthr(size_t work) {
#ifdef _OPENMP
    size_t t = work >> 14; if (t < 1) t = 1;
    int mx = omp_get_max_threads();
    return (int)(t > (size_t)mx ? (size_t)mx : t);
#else
    (void)work; return 1;
#endif
}

struct SumMsg { gl_t s0, s1, s2; };           // sumcheck univariate at t=0,1,2
struct EvalProof {
    uint32_t logN, R, Q;                       // v = logN variables
    std::vector<Hash>     roots;               // v round roots (roots[0] = commitment)
    std::vector<SumMsg>   msgs;                 // v sumcheck messages
    std::vector<gl_t>     final_word;           // length 2^R, in clear
    std::vector<QueryProof> queries;
    std::vector<gl_t>     z;                     // opening point (v values)
    gl_t                  y;                     // claimed g(z)
};

// eq weights over the hypercube: w[b] = eq(b,z) = prod_i (b_i?z_i:(1-z_i)), length 2^v.
// OpenMP-parallel when compiled with -Xcompiler -fopenmp (bit-identical values --
// exact field arithmetic, embarrassingly parallel); serial otherwise.
static inline std::vector<gl_t> build_eq(const std::vector<gl_t>& z) {
    uint32_t v = (uint32_t)z.size(), N = 1u << v;
    std::vector<gl_t> w(N, 1ULL);
    #pragma omp parallel for schedule(static) if (N >= 65536) num_threads(nthr(N))
    for (uint32_t b = 0; b < N; b++) {
        gl_t prod = 1ULL;
        for (uint32_t i = 0; i < v; i++)
            prod = gl_mul(prod, (b & (1u << i)) ? z[i] : gl_sub(1ULL, z[i]));
        w[b] = prod;
    }
    return w;
}
// committed-poly value h(z) = c~(z) = sum_b c[b]*eq(b,z).  Parallel partial sums
// combine to the SAME field element (exact addition, any order).
static inline gl_t eval_h(const std::vector<gl_t>& c, const std::vector<gl_t>& eq) {
    const size_t n = c.size();
    if (n < 65536) {
        gl_t acc = 0; for (size_t b = 0; b < n; b++) acc = gl_add(acc, gl_mul(c[b], eq[b])); return acc;
    }
    const int P = 256;
    gl_t part[P];
    #pragma omp parallel for schedule(static) num_threads(nthr(n))
    for (int p = 0; p < P; p++) {
        size_t lo = n * p / P, hi = n * (p + 1) / P;
        gl_t acc = 0;
        for (size_t b = lo; b < hi; b++) acc = gl_add(acc, gl_mul(c[b], eq[b]));
        part[p] = acc;
    }
    gl_t acc = 0; for (int p = 0; p < P; p++) acc = gl_add(acc, part[p]);
    return acc;
}

// log2 of a power-of-two.
static inline uint32_t ilog2(uint32_t n) { uint32_t l = 0; while ((1u << l) < n) l++; return l; }

// RS-encode a length-N=2^v coefficient array onto the size-2^(v+R) subgroup (Horner).
// O(N*2^(v+R)); fine for tests, replaced by GPU NTT in the perf pass.
static inline std::vector<gl_t> rs_encode(const std::vector<gl_t>& c, uint32_t R) {
    uint32_t v = ilog2((uint32_t)c.size()), logM0 = v + R, M0 = 1u << logM0;
    gl_t w = gl_root_of_unity(logM0);
    std::vector<gl_t> cw(M0);
    for (uint32_t jx = 0; jx < M0; jx++) {
        gl_t xj = gl_pow(w, jx), p = 0;
        for (int i = (int)c.size() - 1; i >= 0; i--) p = gl_add(gl_mul(p, xj), c[i]);
        cw[jx] = p;
    }
    return cw;
}
// Commit to coefficient array c: returns Merkle root, fills cw with the codeword.
static inline Hash commit(const std::vector<gl_t>& c, uint32_t R, std::vector<gl_t>& cw) {
    cw = rs_encode(c, R); Merkle mk; mk.build(cw); return mk.root();
}

// GPU Reed-Solomon encode returning the DEVICE codeword (caller frees with cudaFree).
// Lets commit build the Merkle tree directly on the device codeword (no D2H/H2D
// round trip of the M0-length codeword that the old path paid on every commit).
// cached NTT plans (twiddle tables live on device for the process lifetime --
// rebuilding them per commit dominated small-column commit latency)
static inline const P3Ntt& ntt_plan(uint32_t logM0) {
    static P3Ntt* plans[32] = {};
    if (!plans[logM0]) plans[logM0] = new P3Ntt(logM0);
    return *plans[logM0];
}
static inline gl_t* rs_encode_gpu_dev(const std::vector<gl_t>& c, uint32_t R, uint32_t& M0_out) {
    uint32_t v = ilog2((uint32_t)c.size()), logM0 = v + R, M0 = 1u << logM0; M0_out = M0;
    gl_t* d_in = dmalloc(M0, "rs_encode:in");
    gl_t* d_out = dmalloc(M0, "rs_encode:out");
    cudaMemsetAsync(d_in, 0, (size_t)M0 * sizeof(gl_t), 0);
    cudaMemcpyAsync(d_in, c.data(), c.size() * sizeof(gl_t), cudaMemcpyHostToDevice, 0);
    ntt_plan(logM0).run(d_in, d_out, true);
    cudaFreeAsync(d_in, 0);
    return d_out;                    // caller frees with cudaFreeAsync(d_out, 0)
}
// GPU Reed-Solomon encode (forward NTT of the zero-padded coeff vector) -- same
// codeword as rs_encode but in ms instead of the O(N*M) host Horner.
static inline std::vector<gl_t> rs_encode_gpu(const std::vector<gl_t>& c, uint32_t R) {
    uint32_t M0; gl_t* d_out = rs_encode_gpu_dev(c, R, M0);
    std::vector<gl_t> cw(M0);
    cudaMemcpy(cw.data(), d_out, (size_t)M0 * sizeof(gl_t), cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_out, 0);
    return cw;
}
// commit returning ONLY the root (no host codeword materialization) -- for
// callers that defer openings to the batched-opening module (p3_batchopen),
// which re-encodes coefficient vectors on device when it needs codewords.
static inline Hash commit_gpu_rootonly(const std::vector<gl_t>& c, uint32_t R) {
    uint32_t M0; gl_t* d_out = rs_encode_gpu_dev(c, R, M0);
    Hash r;
    if (M0 >= (1u << 24)) {          // stream: the full tree would hold 64*M0 bytes
        p3fri::RetTree rt;
        r = p3fri::stream_tree_build(M0, [&](size_t off, uint32_t len, uint8_t* out) {
                p3_merkle_leaf_kernel<<<(len + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS,
                                        P3_MERKLE_THREADS>>>(d_out + off, out, len);
            }, &rt).root();
        p3fri::rettree_reg()[r] = std::move(rt);
    } else {
        p3fri::DeviceMerkle mk; mk.build_dev(d_out, M0);
        r = mk.root(); mk.free_();
    }
    cudaFreeAsync(d_out, 0);
    ckcuda("commit_gpu_rootonly");
    return r;
}
static inline Hash commit_gpu(const std::vector<gl_t>& c, uint32_t R, std::vector<gl_t>& cw) {
    uint32_t M0; gl_t* d_out = rs_encode_gpu_dev(c, R, M0);
    p3fri::DeviceMerkle mk; mk.build_dev(d_out, M0);
    Hash r = mk.root(); mk.free_();                  // tree from device codeword; root only
    cw.resize(M0); cudaMemcpy(cw.data(), d_out, (size_t)M0 * sizeof(gl_t), cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_out, 0);
    return r;
}

// GPU MLE fold of a codeword (openings use mle=true): out[c] = (1-beta)E + beta*O,
// E=(f[c]+f[c+half])/2, O=(f[c]-f[c+half])/(2*x), x=w_M^c.  Replaces the O(M) host fold.
__global__ void p3bf_fold_kernel(const gl_t* f, gl_t* out, uint32_t half, gl_t winv, gl_t beta, gl_t inv2) {
    uint32_t c = blockIdx.x*blockDim.x + threadIdx.x; if (c >= half) return;
    gl_t invx = gl_pow(winv, c), a = f[c], b = f[c+half];
    gl_t E = gl_mul(gl_add(a,b), inv2);
    gl_t O = gl_mul(gl_mul(gl_sub(a,b), inv2), invx);
    out[c] = gl_add(gl_mul(gl_sub(1ULL,beta), E), gl_mul(beta, O));
}
// GPU build_eq: out[b] = prod_i (b_i ? z[i] : 1-z[i]).  Replaces the O(N*v) host loop.
__global__ void p3bf_eq_kernel(const gl_t* z, gl_t* out, uint32_t v, uint32_t N) {
    uint32_t b = blockIdx.x*blockDim.x + threadIdx.x; if (b >= N) return;
    gl_t p = 1ULL;
    for (uint32_t i = 0; i < v; i++) p = gl_mul(p, (b & (1u<<i)) ? z[i] : gl_sub(1ULL, z[i]));
    out[b] = p;
}
// GPU dot product sum_i a[i]*b[i] (block-reduced)
__global__ void p3bf_dot_kernel(const gl_t* a, const gl_t* b, gl_t* blk, uint32_t n) {
    __shared__ gl_t sh[256]; uint32_t t=threadIdx.x; gl_t l=0;
    for (uint32_t i=blockIdx.x*blockDim.x+t; i<n; i+=gridDim.x*blockDim.x) l=gl_add(l,gl_mul(a[i],b[i]));
    sh[t]=l; __syncthreads();
    for (uint32_t s=blockDim.x/2;s>0;s>>=1){ if(t<s) sh[t]=gl_add(sh[t],sh[t+s]); __syncthreads(); }
    if(t==0) blk[blockIdx.x]=sh[0];
}
// GPU evaluation  c~(z) = sum_b c[b]*eq(b,z)  (replaces host build_eq+eval_h)
static inline gl_t eval_h_gpu(const std::vector<gl_t>& c, const std::vector<gl_t>& z) {
    uint32_t v=(uint32_t)z.size(), N=1u<<v; const uint32_t NB=256;
    gl_t *dz,*deq,*dc,*dblk;
    cudaMallocAsync(&dz,(size_t)v*8,0); cudaMemcpy(dz,z.data(),(size_t)v*8,cudaMemcpyHostToDevice);
    cudaMallocAsync(&deq,(size_t)N*8,0); p3bf_eq_kernel<<<(N+255)/256,256>>>(dz,deq,v,N);
    cudaMallocAsync(&dc,(size_t)N*8,0); cudaMemcpy(dc,c.data(),(size_t)N*8,cudaMemcpyHostToDevice);
    cudaMallocAsync(&dblk,(size_t)NB*8,0); p3bf_dot_kernel<<<NB,256>>>(dc,deq,dblk,N);
    std::vector<gl_t> hb(NB); cudaMemcpy(hb.data(),dblk,(size_t)NB*8,cudaMemcpyDeviceToHost);
    gl_t s=0; for(auto x:hb) s=gl_add(s,x);
    cudaFreeAsync(dz,0);cudaFreeAsync(deq,0);cudaFreeAsync(dc,0);cudaFreeAsync(dblk,0);
    return s;
}

static inline std::vector<gl_t> build_eq_gpu(const std::vector<gl_t>& z) {
    uint32_t v=(uint32_t)z.size(), N=1u<<v;
    gl_t *dz,*d; cudaMalloc(&dz,(size_t)v*sizeof(gl_t)); cudaMalloc(&d,(size_t)N*sizeof(gl_t));
    cudaMemcpy(dz,z.data(),(size_t)v*sizeof(gl_t),cudaMemcpyHostToDevice);
    p3bf_eq_kernel<<<(N+255)/256,256>>>(dz,d,v,N);
    std::vector<gl_t> out(N); cudaMemcpy(out.data(),d,(size_t)N*sizeof(gl_t),cudaMemcpyDeviceToHost);
    cudaFree(dz); cudaFree(d); return out;
}

static inline std::vector<gl_t> fold_gpu(const std::vector<gl_t>& f, gl_t w_M, gl_t beta) {
    uint32_t M=(uint32_t)f.size(), half=M/2;
    gl_t *d_in,*d_out; cudaMalloc(&d_in,(size_t)M*sizeof(gl_t)); cudaMalloc(&d_out,(size_t)half*sizeof(gl_t));
    cudaMemcpy(d_in,f.data(),(size_t)M*sizeof(gl_t),cudaMemcpyHostToDevice);
    gl_t winv=gl_inv(w_M), inv2=gl_inv(2ULL);
    p3bf_fold_kernel<<<(half+255)/256,256>>>(d_in,d_out,half,winv,beta,inv2);
    std::vector<gl_t> out(half); cudaMemcpy(out.data(),d_out,(size_t)half*sizeof(gl_t),cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out); return out;
}

static inline gl_t alpha_from(fs::Transcript& tr) {
    uint8_t b[32]; tr.challenge_bytes(b);
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v % GL_P;
}
// evaluate the degree-2 poly through (0,s0),(1,s1),(2,s2) at t.
static inline gl_t quad_eval(gl_t s0, gl_t s1, gl_t s2, gl_t t) {
    gl_t inv2 = gl_inv(2ULL);
    gl_t t1 = gl_sub(t, 1ULL), t2 = gl_sub(t, 2ULL);
    gl_t L0 = gl_mul(gl_mul(t1, t2), inv2);                 // (t-1)(t-2)/2
    gl_t L1 = gl_sub(0ULL, gl_mul(t, t2));                  // -t(t-2)
    gl_t L2 = gl_mul(gl_mul(t, t1), inv2);                  // t(t-1)/2
    return gl_add(gl_add(gl_mul(s0, L0), gl_mul(s1, L1)), gl_mul(s2, L2));
}
// eq(alpha,z) = prod_i (a_i z_i + (1-a_i)(1-z_i))
static inline gl_t eq_point(const std::vector<gl_t>& alpha, const std::vector<gl_t>& z) {
    gl_t prod = 1ULL;
    for (size_t i = 0; i < alpha.size(); i++) {
        gl_t t = gl_add(gl_mul(alpha[i], z[i]), gl_mul(gl_sub(1ULL, alpha[i]), gl_sub(1ULL, z[i])));
        prod = gl_mul(prod, t);
    }
    return prod;
}

// device sumcheck message: per-block partial sums of (cl*wl, ch*wh, (2ch-cl)(2wh-wl))
__global__ void p3bf_scmsg_kernel(const gl_t* c, const gl_t* w, gl_t* b0, gl_t* b1, gl_t* b2, uint32_t half) {
    __shared__ gl_t s0[256], s1[256], s2[256];
    uint32_t t=threadIdx.x; gl_t l0=0,l1=0,l2=0;
    for (uint32_t i=blockIdx.x*blockDim.x+t; i<half; i+=gridDim.x*blockDim.x) {
        gl_t cl=c[2*i],ch=c[2*i+1],wl=w[2*i],wh=w[2*i+1];
        l0=gl_add(l0,gl_mul(cl,wl)); l1=gl_add(l1,gl_mul(ch,wh));
        gl_t c2=gl_sub(gl_add(ch,ch),cl), w2=gl_sub(gl_add(wh,wh),wl); l2=gl_add(l2,gl_mul(c2,w2));
    }
    s0[t]=l0; s1[t]=l1; s2[t]=l2; __syncthreads();
    for (uint32_t s=blockDim.x/2; s>0; s>>=1){ if(t<s){ s0[t]=gl_add(s0[t],s0[t+s]); s1[t]=gl_add(s1[t],s1[t+s]); s2[t]=gl_add(s2[t],s2[t+s]); } __syncthreads(); }
    if(t==0){ b0[blockIdx.x]=s0[0]; b1[blockIdx.x]=s1[0]; b2[blockIdx.x]=s2[0]; }
}
// gather: out[t] = src[idx[t]]
__global__ void p3bf_gather_kernel(const gl_t* src, const uint32_t* idx, gl_t* out, uint32_t n) {
    uint32_t t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return; out[t]=src[idx[t]];
}
// device MLE bind: out[i] = in[2i] + a*(in[2i+1]-in[2i])
__global__ void p3bf_bind_kernel(const gl_t* in, gl_t* out, uint32_t half, gl_t a) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=half) return;
    out[i]=gl_add(in[2*i], gl_mul(a, gl_sub(in[2*i+1], in[2*i])));
}

// Retain freed device memory in the async pool (don't return it to the driver) so
// repeated allocations across operands/rounds are recycled. Call once at startup.
static inline void p3_enable_mempool() {
    cudaMemPool_t pool; cudaDeviceGetDefaultMemPool(&pool, 0);
    uint64_t thr = UINT64_MAX; cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr);
}

// opening phase profiling accumulators (ms); reset/read by callers
static double g_t_merkle=0, g_t_fold=0, g_t_query=0, g_t_eq=0, g_t_sc=0, g_t_setup=0, g_t_teardown=0;
static inline double bf_now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }

// fully device-resident opening (same protocol/transcript as prove_eval): codeword,
// coeffs, and eq stay on the GPU across all rounds. Only the 3 scalar sumcheck values
// per round and the Q query spot-checks/paths return to host. Used when g_gpu_merkle.
static inline EvalProof prove_eval_dev(const std::vector<gl_t>& c, const std::vector<gl_t>& z, gl_t y,
                                       uint32_t R, uint32_t Q, const std::vector<gl_t>& cw0, const std::string& seed) {
    uint32_t v=(uint32_t)z.size(), logM0=v+R, M0=1u<<logM0, N=1u<<v;
    EvalProof pf; pf.logN=v; pf.R=R; pf.Q=Q; pf.z=z; pf.y=y;
    fs::Transcript tr(seed); tr.absorb("z",z.data(),z.size()*sizeof(gl_t)); tr.absorb("y",&y,sizeof(gl_t));
    double tsetup=bf_now_ms();
    gl_t *d_c,*d_w; cudaMallocAsync(&d_c,(size_t)N*sizeof(gl_t),0); cudaMemcpy(d_c,c.data(),(size_t)N*sizeof(gl_t),cudaMemcpyHostToDevice);
    { gl_t* dz; cudaMallocAsync(&dz,(size_t)v*sizeof(gl_t),0); cudaMemcpy(dz,z.data(),(size_t)v*sizeof(gl_t),cudaMemcpyHostToDevice);
      cudaMallocAsync(&d_w,(size_t)N*sizeof(gl_t),0); p3bf_eq_kernel<<<(N+255)/256,256>>>(dz,d_w,v,N); cudaFreeAsync(dz,0); }
    std::vector<gl_t*> d_cw(v+1,nullptr);
    cudaMallocAsync(&d_cw[0],(size_t)M0*sizeof(gl_t),0); cudaMemcpy(d_cw[0],cw0.data(),(size_t)M0*sizeof(gl_t),cudaMemcpyHostToDevice);
    std::vector<p3fri::DeviceMerkle> dtrees(v);
    const uint32_t NB=256; gl_t *db0,*db1,*db2; cudaMallocAsync(&db0,NB*8,0);cudaMallocAsync(&db1,NB*8,0);cudaMallocAsync(&db2,NB*8,0);
    std::vector<gl_t> hb0(NB),hb1(NB),hb2(NB);
    cudaDeviceSynchronize(); g_t_setup += bf_now_ms()-tsetup;
    for (uint32_t r=0;r<v;r++){
        uint32_t Mr=M0>>r, n=N>>r, half=n/2;
        double tm=bf_now_ms();
        dtrees[r].build_dev(d_cw[r],Mr); cudaDeviceSynchronize(); Hash root=dtrees[r].root();
        g_t_merkle+=bf_now_ms()-tm;
        pf.roots.push_back(root); tr.absorb("root",root.data(),32);
        double tsc=bf_now_ms();
        p3bf_scmsg_kernel<<<NB,256>>>(d_c,d_w,db0,db1,db2,half);
        cudaMemcpy(hb0.data(),db0,NB*8,cudaMemcpyDeviceToHost); cudaMemcpy(hb1.data(),db1,NB*8,cudaMemcpyDeviceToHost); cudaMemcpy(hb2.data(),db2,NB*8,cudaMemcpyDeviceToHost);
        gl_t s0=0,s1=0,s2=0; for(uint32_t i=0;i<NB;i++){s0=gl_add(s0,hb0[i]);s1=gl_add(s1,hb1[i]);s2=gl_add(s2,hb2[i]);}
        SumMsg msg{s0,s1,s2}; pf.msgs.push_back(msg); tr.absorb("sc",&msg,sizeof(SumMsg));
        gl_t a=alpha_from(tr);
        gl_t *nc,*nw; cudaMallocAsync(&nc,(size_t)half*sizeof(gl_t),0); cudaMallocAsync(&nw,(size_t)half*sizeof(gl_t),0);
        p3bf_bind_kernel<<<(half+255)/256,256>>>(d_c,nc,half,a); p3bf_bind_kernel<<<(half+255)/256,256>>>(d_w,nw,half,a);
        cudaFreeAsync(d_c,0); cudaFreeAsync(d_w,0); d_c=nc; d_w=nw;
        g_t_sc+=bf_now_ms()-tsc;
        double tf=bf_now_ms();
        gl_t w=gl_root_of_unity(logM0-r), winv=gl_inv(w), inv2=gl_inv(2ULL); uint32_t fh=Mr/2;
        cudaMallocAsync(&d_cw[r+1],(size_t)fh*sizeof(gl_t),0);
        p3bf_fold_kernel<<<(fh+255)/256,256>>>(d_cw[r],d_cw[r+1],fh,winv,a,inv2);
        cudaDeviceSynchronize(); g_t_fold+=bf_now_ms()-tf;
    }
    pf.final_word.resize(1u<<R); cudaMemcpy(pf.final_word.data(),d_cw[v],(size_t)(1u<<R)*sizeof(gl_t),cudaMemcpyDeviceToHost);
    tr.absorb("final",pf.final_word.data(),pf.final_word.size()*sizeof(gl_t));
    double tq=bf_now_ms();
    // derive all query coset indices per (query, round) from the transcript
    std::vector<std::vector<uint32_t>> ccq(Q, std::vector<uint32_t>(v));
    for (uint32_t q=0;q<Q;q++){ uint32_t c0=(uint32_t)idx_from(tr,M0/2), p=c0;
        for (uint32_t r=0;r<v;r++){ uint32_t h=(M0>>r)/2; uint32_t cc=p%h; ccq[q][r]=cc; p=cc; } }
    std::vector<QueryProof> qps(Q); for(auto& x:qps) x.rounds.resize(v);
    for (uint32_t r=0;r<v;r++){
        uint32_t h=(M0>>r)/2; std::vector<uint32_t> idxs(2*Q);
        for (uint32_t q=0;q<Q;q++){ idxs[2*q]=ccq[q][r]; idxs[2*q+1]=ccq[q][r]+h; }
        uint32_t* d_idx; cudaMallocAsync(&d_idx,(size_t)2*Q*4,0); cudaMemcpy(d_idx,idxs.data(),(size_t)2*Q*4,cudaMemcpyHostToDevice);
        gl_t* d_val; cudaMallocAsync(&d_val,(size_t)2*Q*sizeof(gl_t),0);
        p3bf_gather_kernel<<<(2*Q+255)/256,256>>>(d_cw[r],d_idx,d_val,2*Q);
        std::vector<gl_t> vals(2*Q); cudaMemcpy(vals.data(),d_val,(size_t)2*Q*sizeof(gl_t),cudaMemcpyDeviceToHost);
        cudaFreeAsync(d_idx,0); cudaFreeAsync(d_val,0);
        auto paths = dtrees[r].paths_batch(idxs);
        for (uint32_t q=0;q<Q;q++){ qps[q].rounds[r].a=vals[2*q]; qps[q].rounds[r].b=vals[2*q+1];
            qps[q].rounds[r].pa=paths[2*q]; qps[q].rounds[r].pb=paths[2*q+1]; }
    }
    pf.queries=qps;
    g_t_query+=bf_now_ms()-tq;
    double ttd=bf_now_ms();
    for (uint32_t r=0;r<=v;r++) if(d_cw[r]) cudaFreeAsync(d_cw[r],0);
    for (auto& t:dtrees) t.free_();
    cudaFreeAsync(d_c,0);cudaFreeAsync(d_w,0);cudaFreeAsync(db0,0);cudaFreeAsync(db1,0);cudaFreeAsync(db2,0);
    cudaDeviceSynchronize(); g_t_teardown += bf_now_ms()-ttd;
    return pf;
}

// ---------------- prover ----------------
static inline EvalProof prove_eval(std::vector<gl_t> c, const std::vector<gl_t>& z, gl_t y,
                                   uint32_t R, uint32_t Q,
                                   const std::vector<gl_t>& cw0, const std::string& seed = "p3-bf") {
    if (p3fri::g_gpu_merkle) return prove_eval_dev(c, z, y, R, Q, cw0, seed);
    uint32_t v = (uint32_t)z.size(), logM0 = v + R, M0 = 1u << logM0;
    EvalProof pf; pf.logN = v; pf.R = R; pf.Q = Q; pf.z = z; pf.y = y;
    fs::Transcript tr(seed);
    tr.absorb("z", z.data(), z.size() * sizeof(gl_t));
    tr.absorb("y", &y, sizeof(gl_t));

    std::vector<std::vector<gl_t>> words; words.push_back(cw0);
    bool GM = p3fri::g_gpu_merkle;                 // device-resident Merkle for large openings
    std::vector<Merkle> trees;
    std::vector<p3fri::DeviceMerkle> dtrees;
    double teq=bf_now_ms();
    std::vector<gl_t> cur_c = c, cur_w = (GM && z.size()>=10) ? build_eq_gpu(z) : build_eq(z), alphas;
    g_t_eq += bf_now_ms()-teq;

    for (uint32_t r = 0; r < v; r++) {
        Hash root;
        double tm=bf_now_ms();
        if (GM) { p3fri::DeviceMerkle mk; mk.build(words[r]); cudaDeviceSynchronize(); root=mk.root(); dtrees.push_back(mk); }
        else    { Merkle mk; mk.build(words[r]); root=mk.root(); trees.push_back(mk); }
        g_t_merkle += bf_now_ms()-tm;
        pf.roots.push_back(root);
        tr.absorb("root", root.data(), 32);
        // sumcheck message over the LSB split of cur_c, cur_w
        double tsc=bf_now_ms();
        uint32_t n = (uint32_t)cur_c.size(), half = n / 2;
        gl_t s0 = 0, s1 = 0, s2 = 0;
        for (uint32_t b = 0; b < half; b++) {
            gl_t cl = cur_c[2*b], ch = cur_c[2*b+1], wl = cur_w[2*b], wh = cur_w[2*b+1];
            s0 = gl_add(s0, gl_mul(cl, wl));
            s1 = gl_add(s1, gl_mul(ch, wh));
            gl_t c2 = gl_sub(gl_add(ch, ch), cl), w2 = gl_sub(gl_add(wh, wh), wl);  // val at t=2
            s2 = gl_add(s2, gl_mul(c2, w2));
        }
        SumMsg msg{s0, s1, s2}; pf.msgs.push_back(msg);
        tr.absorb("sc", &msg, sizeof(SumMsg));
        gl_t a = alpha_from(tr); alphas.push_back(a);
        // bind cur_c, cur_w to alpha (LSB) ; fold codeword with same alpha
        std::vector<gl_t> nc(half), nw(half);
        for (uint32_t b = 0; b < half; b++) {
            nc[b] = gl_add(cur_c[2*b], gl_mul(a, gl_sub(cur_c[2*b+1], cur_c[2*b])));
            nw[b] = gl_add(cur_w[2*b], gl_mul(a, gl_sub(cur_w[2*b+1], cur_w[2*b])));
        }
        cur_c = nc; cur_w = nw;
        g_t_sc += bf_now_ms()-tsc;
        gl_t w = gl_root_of_unity(logM0 - r);
        double tf=bf_now_ms();
        words.push_back(GM ? fold_gpu(words[r], w, a) : fold(words[r], w, a, /*mle=*/true));
        if(GM) cudaDeviceSynchronize();
        g_t_fold += bf_now_ms()-tf;
    }
    pf.final_word = words[v];
    tr.absorb("final", pf.final_word.data(), pf.final_word.size() * sizeof(gl_t));

    double tq=bf_now_ms();
    for (uint32_t q = 0; q < Q; q++) {
        uint32_t c0 = (uint32_t)idx_from(tr, M0 / 2);
        QueryProof qp; uint32_t p = c0;
        for (uint32_t r = 0; r < v; r++) {
            uint32_t Mr = M0 >> r, h = Mr / 2, cc = p % h;
            RoundOpen ro; ro.a = words[r][cc]; ro.b = words[r][cc + h];
            if (GM) { ro.pa = dtrees[r].path(cc); ro.pb = dtrees[r].path(cc + h); }
            else    { ro.pa = trees[r].path(cc);  ro.pb = trees[r].path(cc + h); }
            qp.rounds.push_back(ro); p = cc;
        }
        pf.queries.push_back(qp);
    }
    g_t_query += bf_now_ms()-tq;
    if (GM) for (auto& t : dtrees) t.free_();
    return pf;
}

// ---------------- verifier ----------------
static inline bool verify_eval(const EvalProof& pf, const std::string& seed = "p3-bf", const char** why = nullptr) {
    auto fail = [&](const char* m){ if (why) *why = m; return false; };
    uint32_t v = pf.logN, R = pf.R, M0 = 1u << (v + R);
    if (pf.roots.size() != v || pf.msgs.size() != v) return fail("size");
    if (pf.final_word.size() != (1u << R)) return fail("final size");
    if (pf.queries.size() != pf.Q) return fail("query count");
    if (pf.z.size() != v) return fail("z size");

    fs::Transcript tr(seed);
    tr.absorb("z", pf.z.data(), pf.z.size() * sizeof(gl_t));
    tr.absorb("y", &pf.y, sizeof(gl_t));

    std::vector<gl_t> alphas(v);
    gl_t claim = pf.y;
    for (uint32_t r = 0; r < v; r++) {
        tr.absorb("root", pf.roots[r].data(), 32);
        const SumMsg& m = pf.msgs[r];
        if (gl_add(m.s0, m.s1) != claim) return fail("sumcheck claim");   // s(0)+s(1)==H
        tr.absorb("sc", &m, sizeof(SumMsg));
        gl_t a = alpha_from(tr); alphas[r] = a;
        claim = quad_eval(m.s0, m.s1, m.s2, a);
    }
    tr.absorb("final", pf.final_word.data(), pf.final_word.size() * sizeof(gl_t));

    for (size_t i = 1; i < pf.final_word.size(); i++)
        if (pf.final_word[i] != pf.final_word[0]) return fail("final not constant");

    // tie: final sumcheck claim == c~(alpha) * eq(alpha,z), with c~(alpha)=final constant C
    gl_t C = pf.final_word[0];
    if (claim != gl_mul(C, eq_point(alphas, pf.z))) return fail("eval tie");

    std::vector<uint32_t> c0s(pf.Q);
    for (uint32_t q = 0; q < pf.Q; q++) c0s[q] = (uint32_t)idx_from(tr, M0 / 2);
    if (!p3fri::check_queries(pf.roots, pf.final_word, alphas, pf.queries, v, R, c0s, why, /*mle=*/true)) return false;
    if (why) *why = "ok";
    return true;
}

} // namespace p3bf
