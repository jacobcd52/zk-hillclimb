// p3_zkc.cuh -- ZERO-KNOWLEDGE core for the composed transformer-layer prover
// (design doc section 12).  Applies the p3_zkopen mechanisms (section 10) to the
// WHOLE p3 stack through three global hooks, so every committed column, every
// sumcheck message chain and every opened evaluation in the composed proof is
// hiding:
//
//  (1) MASK-SLICE AUGMENTATION (p3_zkopen mechanism 1, generalized to 2^e
//      slices).  A committed column of length N=2^v is committed as the
//      augmented column [real | mask] of length N*2^e, mask fresh uniform,
//      with e = e_of(v) chosen so the mask dimension N*(2^e-1) exceeds the
//      per-proof revealed-functional budget (2Q round-0 query values + the
//      terminal/binding claims, with 4x headroom for seam-linked column
//      groups).  Constraint zero-checks run over the augmented domain with
//      eq((z||0),.) weights -- the (1-e_j) factors KILL the mask slices, so
//      masks are unconstrained there; every opened evaluation is taken at a
//      point with random ex-coordinates and is uniform over the mask.
//      LINEAR cross-column identities that surface as claim algebra (row-sum
//      bindings, slice bindings, composition seams) are extended to slice 1
//      by construction: the prover derives the dependent columns' slice-1
//      masks by the SAME row formula / seam transform (see the gadget
//      provers), and those claims use points (z || zex || 0...) with one
//      fresh ex challenge -- the claim value is then (1-zex)*real + zex*mask,
//      uniform, while the claim ALGEBRA still holds slice-by-slice.
//
//  (2) DEGREE-MATCHED LIBRA BLINDS (p3_zkopen mechanism 2, fixed for the
//      quartic message class).  A multilinear blind alone leaves the t^2..t^4
//      coefficients of a degree-4 round message unblinded (the finite-
//      difference leak: Delta^2 s_t of a blinded message would still be a pure
//      witness functional).  Every sumcheck therefore adds
//          rho * ( B1(b) + E(b)*B2(b) + E(b)^2*B3(b) + E(b)^3*B4(b) )
//      with B1..B4 fresh uniform committed columns and E the sumcheck's own
//      (public) weight column: the four terms have round degree 1,2,3,4, so
//      the blind spans ALL message coefficients.  The prover publishes
//      H = sum_b Blind(b) BEFORE rho is drawn (sound by Schwartz-Zippel in
//      rho); the verifier starts the chain at claim0 + rho*H and finishes at
//      F(opened) + rho*(yB1 + w*yB2 + w^2*yB3 + w^3*yB4), w = the terminal
//      weight value; yB1..yB4 are ordinary ledger claims (the blind columns
//      are committed and batch-opened like everything else).  Cubic (logUp)
//      chains use the 3-column variant.
//
//  (3) SALTED HIDING COMMITMENTS (p3_zkopen mechanism 3).  Every commitment
//      root in zk mode is a Merkle root over salted leaves
//      SHA256(value || salt_i), salt_i = SHA256(sseed || i); an opened
//      round-0 position discloses (value, salt, path) only, and the values
//      are already uniform by (1).  Internal nodes keep the p3fri
//      single-compress node hash so p3fri::verify_path works unchanged.
//
//  (4) BATCH BLINDER COLUMN (p3_batchopen).  The per-class batched opening
//      additionally RLCs in one fresh full-domain uniform committed column,
//      which one-time-pads the combined word U: the opening sumcheck
//      messages, the fold-round openings and the final word become uniform
//      (without it, 2Q*(v+e) fold openings over-determine U and reveal the
//      rho-combination of the real columns).  The blinder also blinds the
//      reduction sumcheck messages through its own mu-weighted term.
//
// The global context G switches the stack between the bit-identical legacy
// path (G.on = false; default) and the zk path; mask_on/blind_on/salt_on are
// the hiding battery's negative-control toggles.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <stdexcept>
#include "p3_goldilocks.cuh"
#include "p3_merkle.cuh"
#include "p3_fri.cuh"
#include "p3_ntt.cuh"
#include "p3_basefold.cuh"
#include "fs_transcript.hpp"

namespace p3zkc {

using p3fri::Hash;

struct Ctx {
    bool on = false;            // master zk switch (default: legacy bit-identical)
    bool mask_on = true;        // negative control: mask-slice augmentation
    bool blind_on = true;       // negative control: Libra message blinds
    bool salt_on = true;        // negative control: salted Merkle leaves
    bool sim = false;           // HVZK simulator mode (tape foreknowledge H-fixups)
    bool nolink = false;        // DEBUG: force fresh (unlinked) seam masks
    uint32_t Q = 24;            // query budget the e_of policy is sized for
    uint64_t seed = 1;          // master randomness (masks, blinds, salts)
    uint64_t ctr = 0;           // per-draw counter (fresh randomness per commit)
};
static Ctx G;

// ---- randomness ----
static inline gl_t zprng(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    uint64_t z = s; z ^= z >> 31; return z % GL_P;
}
// random-access stream (splitmix64 mixing): the SAME value for index i on host
// and device -- lets big blind columns be generated on the GPU and regenerated
// on the host by the opening ledger, bit-identically, with no sequential chain.
static inline __host__ __device__ gl_t zprng_at(uint64_t seed, uint64_t i) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (i + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z % GL_P;
}
__global__ void p3zkc_fill_at_kernel(gl_t* out, uint64_t seed, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = zprng_at(seed, i);
}
static inline uint64_t next_seed() {
    uint64_t z = (G.seed += 0x9E3779B97F4A7C15ULL * (++G.ctr));
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ---- augmentation policy ----
// mask dimension N*(2^e - 1) >= ZKB = 4*(2Q + 64): 2Q round-0 openings + the
// terminal/binding claims per column per proof, with 4x headroom for
// seam-linked mask groups (max group size 4 in the composed layer).
static inline uint32_t zkb() { return 4 * (2 * G.Q + 64); }
static inline uint32_t e_of(uint32_t v) {
    if (!G.on) return 0;
    uint64_t N = 1ull << v, need = zkb();
    uint32_t e = 1;
    while (N * ((1ull << e) - 1) < need) e++;
    return e;
}
static inline uint32_t vfull(uint32_t v) { return v + e_of(v); }

// fresh uniform mask region for a length-2^v column (zeros under the negative control)
static inline std::vector<gl_t> fresh_mask(uint32_t v) {
    uint32_t e = e_of(v);
    std::vector<gl_t> m(((size_t)1 << (v + e)) - ((size_t)1 << v), 0);
    if (G.mask_on) { uint64_t s = next_seed(); for (auto& x : m) x = zprng(s); }
    return m;
}
// [real | mask]: mask slices at the HIGH addresses (ex = high variables)
static inline std::vector<gl_t> augment(const std::vector<gl_t>& vals, std::vector<gl_t> mask) {
    std::vector<gl_t> a(vals.size() + mask.size());
    memcpy(a.data(), vals.data(), vals.size() * 8);
    memcpy(a.data() + vals.size(), mask.data(), mask.size() * 8);
    return a;
}

// ---- point helpers ----
// zero-extend a point to the column's augmented variable count (zero ex-coords
// weight ONLY the real slice: used for zero-check eq weights and public
// input/output bindings, where the revealed value must be the real evaluation)
static inline std::vector<gl_t> zpt(const std::vector<gl_t>& z) {
    if (!G.on) return z;
    std::vector<gl_t> p = z;
    p.resize(z.size() + e_of((uint32_t)z.size()), 0);
    return p;
}
// hiding claim point (z || zex || 0...): touches the real slice and mask slice 1
static inline std::vector<gl_t> xpt(const std::vector<gl_t>& z, gl_t zex) {
    if (!G.on) return z;
    std::vector<gl_t> p = z;
    p.push_back(zex);
    p.resize(z.size() + e_of((uint32_t)z.size()), 0);
    return p;
}
// map the ex-part of a full sumcheck point onto a (possibly different-e) target
// class: truncate extra coords (the source broadcast is constant in them) or
// zero-pad missing ones (the broadcast reads base slices with high ex bits 0).
static inline std::vector<gl_t> expt(const std::vector<gl_t>& zlow,
                                     const std::vector<gl_t>& rfull, uint32_t src_v) {
    if (!G.on) return zlow;
    uint32_t e_t = e_of((uint32_t)zlow.size());
    std::vector<gl_t> p = zlow;
    for (uint32_t i = 0; i < e_t; i++)
        p.push_back(src_v + i < rfull.size() ? rfull[src_v + i] : 0);
    return p;
}

// ---- broadcast on the augmented domain ----
// out[(idx), ex] = base_aug[map(idx), exmap(ex)] with exmap = truncate/zero-pad:
// exact by construction, so the binding claim at expt(...) needs no linkage.
template <typename MapFn>
static inline std::vector<gl_t> bc_aug(const std::vector<gl_t>& base_aug, uint32_t base_v,
                                       uint32_t out_v, size_t out_n, MapFn map) {
    uint32_t eo = e_of(out_v), eb = e_of(base_v);
    size_t No = (size_t)1 << out_v, Nb = (size_t)1 << base_v;
    std::vector<gl_t> out(No << eo);
    (void)out_n; (void)Nb;
    for (uint32_t ex = 0; ex < (1u << eo); ex++) {
        // eb < eo: constant in the extra high ex coords (truncate);
        // eb > eo: read base slices with high ex bits fixed to 0 (zero-pad).
        uint32_t exb = (eb >= eo) ? ex : (ex & ((1u << eb) - 1));
        for (size_t i = 0; i < No; i++)
            out[((size_t)ex << out_v) | i] = base_aug[((size_t)exb << base_v) | map(i)];
    }
    return out;
}

// ---- salted device Merkle commitment ----
// leaf_i = SHA256(value_le8 || salt_i), salt_i = SHA256(sseed_le8 || i_le8);
// internal nodes = p3_sha256_compress64 (identical to p3fri), so
// p3fri::verify_path authenticates salted openings unchanged.
typedef std::array<uint8_t, 32> Salt;
static inline Salt salt_of(uint64_t sseed, uint64_t i) {
    Salt s{};
    uint8_t b[16];
    for (int k = 0; k < 8; k++) { b[k] = (uint8_t)(sseed >> (8 * k)); b[8 + k] = (uint8_t)(i >> (8 * k)); }
    fs::sha256(b, 16, s.data());
    return s;
}
static inline Hash salted_leaf(gl_t v, const Salt& s) {
    uint8_t b[40];
    for (int k = 0; k < 8; k++) b[k] = (uint8_t)(v >> (8 * k));
    memcpy(b + 8, s.data(), 32);
    Hash h; fs::sha256(b, 40, h.data());
    return h;
}

__global__ void p3zkc_salted_leaf_kernel(const gl_t* cw, uint64_t sseed, uint8_t* out, uint32_t M) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    uint8_t sb[16], salt[32], buf[40];
    for (int k = 0; k < 8; k++) { sb[k] = (uint8_t)(sseed >> (8 * k)); sb[8 + k] = (uint8_t)((uint64_t)i >> (8 * k)); }
    p3_sha256(sb, 16, salt);
    gl_t v = cw[i];
    for (int k = 0; k < 8; k++) buf[k] = (uint8_t)(v >> (8 * k));
    for (int k = 0; k < 32; k++) buf[8 + k] = salt[k];
    p3_sha256(buf, 40, out + (size_t)i * 32);
}
// chunk variant for streamed trees: leaves [off, off+len) of the full codeword
// (the salt index is the GLOBAL leaf index off+i)
__global__ void p3zkc_salted_leaf_off_kernel(const gl_t* cw_chunk, uint64_t sseed, uint8_t* out,
                                             uint64_t off, uint32_t len) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    uint64_t gi = off + i;
    uint8_t sb[16], salt[32], buf[40];
    for (int k = 0; k < 8; k++) { sb[k] = (uint8_t)(sseed >> (8 * k)); sb[8 + k] = (uint8_t)(gi >> (8 * k)); }
    p3_sha256(sb, 16, salt);
    gl_t v = cw_chunk[i];
    for (int k = 0; k < 8; k++) buf[k] = (uint8_t)(v >> (8 * k));
    for (int k = 0; k < 32; k++) buf[8 + k] = salt[k];
    p3_sha256(buf, 40, out + (size_t)i * 32);
}

// device Merkle tree with salted leaves (structure mirrors p3fri::DeviceMerkle)
struct SaltedDevMerkle {
    std::vector<uint8_t*> dlev; std::vector<uint32_t> sz;
    void build_dev(const gl_t* d_cw, uint32_t M, uint64_t sseed) {
        uint8_t* d0; cudaMallocAsync(&d0, (size_t)M * 32, 0);
        p3zkc_salted_leaf_kernel<<<(M + 255) / 256, 256>>>(d_cw, sseed, d0, M);
        dlev.assign(1, d0); sz.assign(1, M);
        for (uint32_t cnt = M; cnt > 1;) {
            uint32_t half = cnt / 2; uint8_t* dn; cudaMallocAsync(&dn, (size_t)half * 32, 0);
            p3_merkle_internal_kernel<<<(half + 255) / 256, 256>>>(dlev.back(), dn, half);
            dlev.push_back(dn); sz.push_back(half); cnt = half;
        }
    }
    Hash root() const { Hash h; cudaMemcpy(h.data(), dlev.back(), 32, cudaMemcpyDeviceToHost); return h; }
    std::vector<std::vector<Hash>> paths_batch(const std::vector<uint32_t>& idxs) const {
        // reuse the p3fri path gather (level layout identical)
        p3fri::DeviceMerkle shim; shim.dlev = dlev; shim.sz = sz;
        auto r = shim.paths_batch(idxs);
        shim.dlev.clear(); shim.sz.clear();      // do NOT free our levels
        return r;
    }
    void free_() { for (auto p : dlev) cudaFreeAsync(p, 0); dlev.clear(); sz.clear(); }
};

// salted commitment root of a coefficient vector (GPU NTT encode + salted tree)
static inline Hash salted_commit_root(const std::vector<gl_t>& c, uint32_t R, uint64_t sseed) {
    uint32_t M0; gl_t* d_cw = p3bf::rs_encode_gpu_dev(c, R, M0);
    Hash r;
    const bool big = M0 >= (1u << 24);       // stream: full tree = 64*M0 bytes
    if (G.salt_on) {
        if (big) {
            r = p3fri::stream_tree_build(M0, [&](size_t off, uint32_t len, uint8_t* out) {
                    p3zkc_salted_leaf_off_kernel<<<(len + 255) / 256, 256>>>(
                        d_cw + off, sseed, out, off, len);
                }).root();
        } else {
            SaltedDevMerkle mk; mk.build_dev(d_cw, M0, sseed); cudaDeviceSynchronize();
            r = mk.root(); mk.free_();
        }
    } else {
        if (big) {
            r = p3fri::stream_tree_build(M0, [&](size_t off, uint32_t len, uint8_t* out) {
                    p3_merkle_leaf_kernel<<<(len + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS,
                                            P3_MERKLE_THREADS>>>(d_cw + off, out, len);
                }).root();
        } else {
            p3fri::DeviceMerkle mk; mk.build_dev(d_cw, M0); cudaDeviceSynchronize();
            r = mk.root(); mk.free_();
        }
    }
    cudaFreeAsync(d_cw, 0);
    p3bf::ckcuda("salted_commit_root");
    return r;
}

// commit a DEVICE-resident value vector (n = 2^v values): NTT encode + salted
// tree, root only -- for blind columns generated on the GPU (their host copy
// never exists; the opening ledger regenerates them via zprng_at).
static inline Hash salted_commit_root_dev(const gl_t* d_vals, size_t n, uint32_t R,
                                          uint64_t sseed) {
    uint32_t v = 0; while (((size_t)1 << v) < n) v++;
    uint32_t logM0 = v + R, M0 = 1u << logM0;
    gl_t* d_in = p3bf::dmalloc(M0, "sccd:in");
    cudaMemsetAsync(d_in, 0, (size_t)M0 * 8, 0);
    cudaMemcpyAsync(d_in, d_vals, n * 8, cudaMemcpyDeviceToDevice, 0);
    gl_t* d_out = p3bf::dmalloc(M0, "sccd:out");
    p3bf::ntt_plan(logM0).run(d_in, d_out, true);
    cudaFreeAsync(d_in, 0);
    Hash r;
    const bool big = M0 >= (1u << 24);
    if (G.salt_on) {
        if (big) {
            r = p3fri::stream_tree_build(M0, [&](size_t off, uint32_t len, uint8_t* out) {
                    p3zkc_salted_leaf_off_kernel<<<(len + 255) / 256, 256>>>(
                        d_out + off, sseed, out, off, len);
                }).root();
        } else {
            SaltedDevMerkle mk; mk.build_dev(d_out, M0, sseed); cudaDeviceSynchronize();
            r = mk.root(); mk.free_();
        }
    } else {
        if (big) {
            r = p3fri::stream_tree_build(M0, [&](size_t off, uint32_t len, uint8_t* out) {
                    p3_merkle_leaf_kernel<<<(len + P3_MERKLE_THREADS - 1) / P3_MERKLE_THREADS,
                                            P3_MERKLE_THREADS>>>(d_out + off, out, len);
                }).root();
        } else {
            p3fri::DeviceMerkle mk; mk.build_dev(d_out, M0); cudaDeviceSynchronize();
            r = mk.root(); mk.free_();
        }
    }
    cudaFreeAsync(d_out, 0);
    p3bf::ckcuda("salted_commit_root_dev");
    return r;
}

// ---- seam mask linkage (composed layer, design doc section 12) ----
// A cross-commitment seam checks producer~(zp) == consumer~(zc) at correlated
// points, opened at a SHARED ex-coordinate zex via xpt() (which touches only the
// real slice, weight 1-zex, and mask slice 1, weight zex).  For the augmented
// evals to agree the consumer's mask SLICE 1 must be the same seam transform of
// the producer's mask slice 1 that relates their real values.  slice1() reads a
// committed column's slice-1 (the first Nreal mask entries); mk_linked() builds
// a consumer mask whose slice 1 is map(producer slice 1) and whose higher slices
// (the independent hiding headroom for that column's own openings) are fresh.
static inline std::vector<gl_t> slice1(const std::vector<gl_t>& aug, uint32_t vreal) {
    size_t N = (size_t)1 << vreal;
    if (aug.size() < 2 * N) return std::vector<gl_t>(N, 0);   // e==0 guard (zk off)
    return std::vector<gl_t>(aug.begin() + N, aug.begin() + 2 * N);
}
// build consumer mask (length Nc*(2^ec - 1)); slice 1 = cons1, rest fresh
static inline std::vector<gl_t> mk_linked(uint32_t vc, const std::vector<gl_t>& cons1) {
    uint32_t ec = e_of(vc);
    size_t Nc = (size_t)1 << vc;
    std::vector<gl_t> m(Nc * ((1ull << ec) - 1), 0);
    for (size_t i = 0; i < Nc && i < cons1.size(); i++) m[i] = cons1[i];   // slice 1
    if (G.mask_on) { uint64_t s = next_seed();
        for (size_t i = Nc; i < m.size(); i++) m[i] = zprng(s); }          // slices 2..
    return m;
}

// a Libra blind column over a v-variable REAL domain: uniform real region and
// uniform mask region (both zero under the blind_on negative control); pass
// the mask to commit_col_nc so the whole augmented column is the blind.
static inline std::vector<gl_t> blind_col_seeded(uint32_t v, uint64_t s, std::vector<gl_t>& mask_out) {
    uint32_t e = e_of(v);
    std::vector<gl_t> c((size_t)1 << v, 0);
    mask_out.assign(((size_t)1 << (v + e)) - c.size(), 0);
    if (G.blind_on) {
        for (auto& x : c) x = zprng(s);
        for (auto& x : mask_out) x = zprng(s);
    }
    return c;
}
static inline std::vector<gl_t> blind_col(uint32_t v, std::vector<gl_t>& mask_out) {
    // seed drawn ONLY under blind_on (preserves the global seed sequence of the
    // blind_on=false negative controls)
    return blind_col_seeded(v, G.blind_on ? next_seed() : 0, mask_out);
}
// regenerate the AUGMENTED blind column [c | mask] committed from seed s --
// blind columns are pure PRNG streams, so the batched-opening ledger can DROP
// them after their claims and rebuild them transiently at opening time instead
// of holding tens of GB of random columns for the whole proof.
static inline std::vector<gl_t> blind_col_aug(uint32_t v, uint64_t s) {
    std::vector<gl_t> m;
    std::vector<gl_t> c = blind_col_seeded(v, s, m);
    return augment(c, std::move(m));
}

// ---- Libra blind material for one sumcheck instance ----
// nb committed blind columns (4 for quartic Msg5 chains, 3 for cubic Msg4);
// serialized in the proof as (roots, H); yB claims ride the shared ledger.
struct Blind {
    Hash rt[4] = {};
    gl_t H = 0;
    gl_t yB[4] = {};
    uint32_t nb = 0;
};
// blind terminal term rho*(yB1 + w*yB2 + ... ) with w = the terminal weight value
static inline gl_t blind_term(const Blind& bl, gl_t rho, gl_t w) {
    gl_t acc = 0, wp = 1ULL;
    for (uint32_t j = 0; j < bl.nb; j++) { acc = gl_add(acc, gl_mul(wp, bl.yB[j])); wp = gl_mul(wp, w); }
    return gl_mul(rho, acc);
}

} // namespace p3zkc
