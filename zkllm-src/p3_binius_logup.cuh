// Binius migration step (design doc section 21.9): logUp over the binary
// tower -- the lookup argument for the tower-field prover.
//
// THE CHAR-2 TRAP (why this is NOT a transliteration of p3_logup.cuh):
// additive logUp proves  sum_i 1/(alpha+v_i) == sum_j m_j/(alpha+t_j), which
// is sound over a prime field because the FORMAL rational identity forces
// integer multiset equality.  In characteristic 2 the formal identity only
// sees multiplicities MOD 2: a value outside the table inserted an EVEN
// number of times XOR-cancels out of the fractional sum and the additive
// argument ACCEPTS it (demonstrated by a tooth in p3_binius_logup_test.cu).
// The sound tower port is the MULTIPLICATIVE form
//
//     prod_i (alpha + v_i)  ==  prod_j (alpha + t_j)^{m_j}
//
// (unique factorization in F[alpha] forces integer multiset equality in any
// characteristic), with the committed multiplicities in BINARY and the 2^b
// exponents absorbed by FROBENIUS: squaring is linear in char 2, so
// (alpha+t_j)^{2^b} = alpha^{2^b} + sum_k beta_k^{2^b} * Tbit_k(j) stays
// DEGREE 1 in the public table bits, and the table-side product becomes one
// grand product over the (multiplicity-bit, table-row) cube of leaves
//     L(b,j) = 1 + m_{j,b} * u(b,j),   u(b,j) = (alpha+t_j)^{2^b} + 1,
// which is degree 1 in the committed m bits with u fully public.
//
// Fingerprints are F_2-LINEAR in the committed bits: v_i = sum_k beta_k *
// wbit_k(i) with beta_k = beta^{k+1}, so the witness-side leaf MLE at the
// GKR endpoint is  alpha + sum_k beta_k * wbitcol_k~(rfin)  -- a linear
// combination of column evals of the CALLER's existing stacked commitment
// (bound by a second point of a bfpcs multi-open; no extra witness columns).
//
// Protocol (one shared transcript; caller has already committed the witness):
//   1. commit multiplicity bits (MBP=2^lMB slices of 2^lT bits, one bfpcs)
//   2. beta, alpha <- transcript   (AFTER both commitments: a prover choosing
//      m after seeing alpha could solve a subset-product for the forged root
//      -- GF(2^128) discrete logs are not a soundness assumption we make)
//   3. witness grand product: GKR chain root W -> leaf claim cw at rfin_w
//      (caller binds cw = alpha + sum beta_k * colk~(rfin_w) to its opening)
//   4. table grand product over 2^(lT+lMB) leaves: root T -> claim cl at rfin_t
//   5. verifier checks W == T (and W != 0)
//   6. binding sumcheck:  cl + 1 = sum_x eq(rfin_t,x) * m(x) * u(x)  (deg 3);
//      endpoint finals: eq recomputed, u~ computed from PUBLIC table bits via
//      the Frobenius factorization, m~ checked by the multiplicity opening.
//
// The grand-product GKR (bfgkr_*) reduces a published root through layer
// sumchecks  claim = sum_y eq(z,y) * lo(y) * hi(y)  reusing the generic
// bf_sumcheck host/GPU provers (byte-identical proofs, teeth in the selftest).
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "p3_binius_sumcheck.cuh"

namespace bflu {

static inline int bflu_ilog2(size_t n) { int l = 0; while (((size_t)1 << l) < n) l++; return l; }
static inline bf128_t bflu_lerp(bf128_t a, bf128_t b, bf128_t t) {
    return bf128_add(a, bf128_mul(t, bf128_add(a, b)));
}
// eq(a,b) at points: prod_t (1 + a_t + b_t)   (char-2 form of ab+(1-a)(1-b))
static inline bf128_t bflu_eq_point(const std::vector<bf128_t>& a, const std::vector<bf128_t>& b) {
    bf128_t e = bf128_one();
    for (size_t t = 0; t < a.size(); t++)
        e = bf128_mul(e, bf128_add(bf128_one(), bf128_add(a[t], b[t])));
    return e;
}

// the triple-product constraint w[0]*w[1]*w[2]: every sumcheck in this file
// (GKR layers: eq*lo*hi; binding: eq*m*u) is this one functor
struct Bf3Prod {
    static constexpr int K = 3, D = 3;
    __host__ __device__ bf128_t operator()(const bf128_t* w) const {
        return bf128_mul(w[0], bf128_mul(w[1], w[2]));
    }
};

// one triple-product sumcheck over 2^l rows: columns [eq(z,.), A, B], both
// paths byte-identical (XOR round reductions + exact per-element folds).
// A/B are consumed.  GPU threshold: below it kernel-launch overhead loses.
static const size_t BFLU_GPU_MIN = (size_t)1 << 13;
static inline void bflu_sc3(const std::vector<bf128_t>& z,
                            std::vector<bf128_t>& A, std::vector<bf128_t>& B,
                            fs::Transcript& tr, BfScProof& pf,
                            std::vector<bf128_t>& zeta, bool gpu) {
    int l = (int)z.size();
    size_t n = (size_t)1 << l;
    if (gpu && n >= BFLU_GPU_MIN) {
        BfScDev dv; bfsc_dev_alloc(dv, 3, l);
        bfsc_dev_eq(dv.a, z.data(), l);
        cudaMemcpy(dv.a + dv.n,     A.data(), n * sizeof(bf128_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dv.a + 2 * dv.n, B.data(), n * sizeof(bf128_t), cudaMemcpyHostToDevice);
        Bf3Prod cf;
        bf_sumcheck_prove_gpu(dv, cf, tr, pf, zeta);
        bfsc_dev_free(dv);
    } else {
        std::vector<std::vector<bf128_t>> cols(3);
        bf_eq_table(z.data(), l, cols[0]);
        cols[1] = std::move(A); cols[2] = std::move(B);
        Bf3Prod cf;
        auto fn = [](const bf128_t* w, const void* ctx) {
            return (*(const Bf3Prod*)ctx)(w);
        };
        bf_sumcheck_prove(l, 3, cols, 3, fn, &cf, tr, pf, zeta);
    }
    std::vector<bf128_t>().swap(A); std::vector<bf128_t>().swap(B);
}

// ---- grand-product GKR over T_128 ----
struct BfGkrProof {
    bf128_t root = bf128_zero(), g0 = bf128_zero(), g1 = bf128_zero();
    std::vector<BfScProof> lay;               // L-1 layer sumchecks
    size_t bytes() const {
        size_t s = 3 * sizeof(bf128_t);
        for (auto& p : lay) s += (p.rounds.size() + p.finals.size()) * sizeof(bf128_t);
        return s;
    }
};

// device product-tree kernels (exact field ops: comb/split values are
// byte-identical to the host tree, so GPU and host chains emit the same proof)
static __global__ void bflu_comb_kernel(const bf128_t* c, bf128_t* o, size_t n) {
    size_t y = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (y < n) o[y] = bf128_mul(c[2 * y], c[2 * y + 1]);
}
static __global__ void bflu_split_kernel(const bf128_t* c, bf128_t* A, bf128_t* B, size_t n) {
    size_t y = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (y < n) { A[y] = c[2 * y]; B[y] = c[2 * y + 1]; }
}

// prove: leaves (size 2^L, L >= 1) consumed; absorbs root + top children;
// outputs the final leaf point rfin (LSB-first) and the leaf-MLE claim there.
// Throws on a zero root (some alpha+v hit zero -- resample upstream).
// GPU path: the whole tree is DEVICE-RESIDENT (one upload; comb/split/eq/
// round kernels on device, small top layers downloaded for the host loop).
static inline void bfgkr_prove(std::vector<bf128_t> leaves, const char* tag,
                               fs::Transcript& tr, BfGkrProof& pf,
                               std::vector<bf128_t>& rfin, bf128_t& claim,
                               bool gpu = true) {
    int L = bflu_ilog2(leaves.size());
    size_t NL = (size_t)1 << L;
    pf.lay.resize(L >= 1 ? L - 1 : 0);
    if (gpu && NL >= 2 * BFLU_GPU_MIN) {
        // ---- device-resident tree ----
        std::vector<bf128_t*> dlv(L + 1, nullptr);
        cudaMalloc(&dlv[0], NL * sizeof(bf128_t));
        cudaMemcpy(dlv[0], leaves.data(), NL * sizeof(bf128_t), cudaMemcpyHostToDevice);
        std::vector<bf128_t>().swap(leaves);
        for (int h = 1; h <= L; h++) {
            size_t n = NL >> h;
            cudaMalloc(&dlv[h], n * sizeof(bf128_t));
            bflu_comb_kernel<<<(uint32_t)((n + 255) / 256), 256>>>(dlv[h - 1], dlv[h], n);
        }
        bf128_t top[2];
        cudaMemcpy(&pf.root, dlv[L], sizeof(bf128_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(top, dlv[L - 1], 2 * sizeof(bf128_t), cudaMemcpyDeviceToHost);
        if (bf128_is0(pf.root)) {
            for (auto d : dlv) if (d) cudaFree(d);
            throw std::runtime_error("bfgkr: zero product root");
        }
        pf.g0 = top[0]; pf.g1 = top[1];
        bf128_t hdr[3] = {pf.root, pf.g0, pf.g1};
        tr.absorb(tag, hdr, sizeof hdr);
        bf128_t mu = bf_chal128(tr);
        claim = bflu_lerp(pf.g0, pf.g1, mu);
        std::vector<bf128_t> z{mu};
        cudaFree(dlv[L]); dlv[L] = nullptr;
        BfScDev dv; bfsc_dev_alloc(dv, 3, L - 1);   // reused across all GPU layers
        size_t stride = dv.n;
        for (int h = L - 1; h >= 1; h--) {
            size_t n = (size_t)1 << (L - h);
            std::vector<bf128_t> zeta;
            if (n >= BFLU_GPU_MIN) {
                dv.l = L - h; dv.n = stride;        // stride stays the alloc size
                bfsc_dev_eq(dv.a, z.data(), L - h);
                bflu_split_kernel<<<(uint32_t)((n + 255) / 256), 256>>>(
                    dlv[h - 1], dv.a + stride, dv.a + 2 * stride, n);
                Bf3Prod cf;
                bf_sumcheck_prove_gpu(dv, cf, tr, pf.lay[L - 1 - h], zeta);
            } else {
                std::vector<bf128_t> c(2 * n), A(n), B(n);
                cudaMemcpy(c.data(), dlv[h - 1], 2 * n * sizeof(bf128_t),
                           cudaMemcpyDeviceToHost);
                for (size_t y = 0; y < n; y++) { A[y] = c[2 * y]; B[y] = c[2 * y + 1]; }
                bflu_sc3(z, A, B, tr, pf.lay[L - 1 - h], zeta, false);
            }
            cudaFree(dlv[h - 1]); dlv[h - 1] = nullptr;
            bf128_t f0 = pf.lay[L - 1 - h].finals[1], f1 = pf.lay[L - 1 - h].finals[2];
            bf128_t mu2 = bf_chal128(tr);
            claim = bflu_lerp(f0, f1, mu2);
            z.assign(1, mu2);
            z.insert(z.end(), zeta.begin(), zeta.end());
        }
        bfsc_dev_free(dv);
        for (auto d : dlv) if (d) cudaFree(d);
        rfin = z;
        return;
    }
    std::vector<std::vector<bf128_t>> lv(L + 1);
    lv[0] = std::move(leaves);
    for (int h = 1; h <= L; h++) {
        size_t n = (size_t)1 << (L - h);
        lv[h].resize(n);
        const std::vector<bf128_t>& c = lv[h - 1];
        std::vector<bf128_t>& o = lv[h];
        #pragma omp parallel for schedule(static) if (n >= 16384)
        for (int64_t y = 0; y < (int64_t)n; y++)
            o[y] = bf128_mul(c[2 * y], c[2 * y + 1]);
    }
    pf.root = lv[L][0];
    if (bf128_is0(pf.root)) throw std::runtime_error("bfgkr: zero product root");
    pf.g0 = lv[L - 1][0]; pf.g1 = lv[L - 1][1];
    bf128_t hdr[3] = {pf.root, pf.g0, pf.g1};
    tr.absorb(tag, hdr, sizeof hdr);
    bf128_t mu = bf_chal128(tr);
    claim = bflu_lerp(pf.g0, pf.g1, mu);
    std::vector<bf128_t> z{mu};
    for (int h = L - 1; h >= 1; h--) {
        size_t n = (size_t)1 << (L - h);          // domain: level-h positions
        std::vector<bf128_t> A(n), B(n);
        const std::vector<bf128_t>& c = lv[h - 1];
        #pragma omp parallel for schedule(static) if (n >= 16384)
        for (int64_t y = 0; y < (int64_t)n; y++) { A[y] = c[2 * y]; B[y] = c[2 * y + 1]; }
        lv[h - 1].clear(); lv[h - 1].shrink_to_fit();
        std::vector<bf128_t> zeta;
        bflu_sc3(z, A, B, tr, pf.lay[L - 1 - h], zeta, gpu);
        bf128_t f0 = pf.lay[L - 1 - h].finals[1], f1 = pf.lay[L - 1 - h].finals[2];
        bf128_t mu2 = bf_chal128(tr);
        claim = bflu_lerp(f0, f1, mu2);
        z.assign(1, mu2);
        z.insert(z.end(), zeta.begin(), zeta.end());
    }
    rfin = z;
}

// verify: replays the chain; on success rfin/claim are the leaf-MLE point and
// value the CALLER must bind to committed (or public) leaf data.
static inline bool bfgkr_verify(const BfGkrProof& pf, int L, const char* tag,
                                fs::Transcript& tr, std::vector<bf128_t>& rfin,
                                bf128_t& claim) {
    if (L < 1 || (int)pf.lay.size() != L - 1) return false;
    if (bf128_is0(pf.root)) return false;
    if (!bf128_eq(bf128_mul(pf.g0, pf.g1), pf.root)) return false;
    bf128_t hdr[3] = {pf.root, pf.g0, pf.g1};
    tr.absorb(tag, hdr, sizeof hdr);
    bf128_t mu = bf_chal128(tr);
    claim = bflu_lerp(pf.g0, pf.g1, mu);
    std::vector<bf128_t> z{mu};
    for (int h = L - 1; h >= 1; h--) {
        const BfScProof& sp = pf.lay[L - 1 - h];
        if (sp.l != L - h || sp.K != 3 || sp.D != 3 || (int)sp.finals.size() != 3)
            return false;
        std::vector<bf128_t> zeta; bf128_t E;
        if (!bf_sumcheck_verify(sp, claim, tr, zeta, &E)) return false;
        if (!bf128_eq(E, bf128_mul(sp.finals[0], bf128_mul(sp.finals[1], sp.finals[2]))))
            return false;
        if (!bf128_eq(sp.finals[0], bflu_eq_point(z, zeta))) return false;
        bf128_t mu2 = bf_chal128(tr);
        claim = bflu_lerp(sp.finals[1], sp.finals[2], mu2);
        z.assign(1, mu2);
        z.insert(z.end(), zeta.begin(), zeta.end());
    }
    rfin = z;
    return true;
}

// ---- the lookup argument ----
struct BfLuProof {
    int lN = 0, lT = 0, lMB = 0;      // log witness rows / table rows / mult-bit slots
    uint8_t mroot[32] = {0};
    BfGkrProof gw, gt;                // witness / table grand products
    BfScProof bind;                   // table-leaf binding sumcheck (eq*m*u)
    BfPcsProof mopen;                 // multiplicity opening at the bind endpoint
    size_t bytes() const {
        return 32 + gw.bytes() + gt.bytes() +
               (bind.rounds.size() + bind.finals.size()) * sizeof(bf128_t) + mopen.bytes();
    }
};

static inline int bflu_lmb(int lN) { return bflu_ilog2((size_t)lN + 1); }
static inline BfPcsParams bflu_mparams(int lT, int lMB) {
    BfPcsParams p;
    p.l = lT + lMB; p.lcol = p.l / 2; p.lrow = p.l - p.lcol; p.Q = 100;
    return p;
}
// beta powers: coef_k = beta^{k+1}
static inline void bflu_betas(bf128_t beta, int J, std::vector<bf128_t>& bk) {
    bk.resize(J);
    bf128_t b = beta;
    for (int k = 0; k < J; k++) { bk[k] = b; b = bf128_mul(b, beta); }
}
// x^(2^b) by repeated squaring
static inline bf128_t bflu_frob(bf128_t x, int b) {
    for (int i = 0; i < b; i++) x = bf128_mul(x, x);
    return x;
}

// prove.  wcols: J bit-column pointers, each 2^lN bytes (0/1) -- the caller's
// committed columns.  idx: table key per row (multiplicity counting only; the
// PROOF binds the tuple bits, not idx).  tbits: J public table bit-columns,
// col-major, each 2^lT.  Outputs rfin_w for the caller's column-eval binding.
// mbits_commit_override / mbits_leaf_override are TEST HOOKS (cheating-prover
// teeth); production passes nullptr.
static inline void bflu_prove(fs::Transcript& tr, int lN, int lT, int J,
                              const uint8_t* const* wcols, const uint32_t* idx,
                              const uint8_t* tbits,
                              BfLuProof& pf, std::vector<bf128_t>& rfin_w,
                              bool gpu = true,
                              const uint8_t* mbits_commit_override = nullptr,
                              const uint8_t* mbits_leaf_override = nullptr,
                              size_t* mcommitted_out = nullptr) {
    size_t N = (size_t)1 << lN, TN = (size_t)1 << lT;
    pf.lN = lN; pf.lT = lT; pf.lMB = bflu_lmb(lN);
    int MBP = 1 << pf.lMB;
    // 1. multiplicities (from idx) and their bit slices
    std::vector<uint32_t> m(TN, 0);
    for (size_t i = 0; i < N; i++) m[idx[i]]++;
    std::vector<uint8_t> mbits((size_t)MBP << lT, 0);
    for (size_t j = 0; j < TN; j++)
        for (int b = 0; b < MBP; b++)
            mbits[((size_t)b << lT) + j] = (m[j] >> b) & 1;
    BfPcsParams pm = bflu_mparams(lT, pf.lMB);
    BfPcsCommit mC;
    bfpcs_commit(pm, mbits_commit_override ? mbits_commit_override : mbits.data(), tr, mC);
    memcpy(pf.mroot, mC.root, 32);
    if (mcommitted_out) *mcommitted_out = mC.committed_bytes;
    if (mbits_leaf_override) mbits.assign(mbits_leaf_override, mbits_leaf_override + ((size_t)MBP << lT));
    // 2. challenges (after both commitments)
    bf128_t beta = bf_chal128(tr), alpha = bf_chal128(tr);
    std::vector<bf128_t> bk;
    bflu_betas(beta, J, bk);
    // 3. witness grand product
    std::vector<bf128_t> wl(N);
    #pragma omp parallel for schedule(static) if (N >= 16384)
    for (int64_t i = 0; i < (int64_t)N; i++) {
        bf128_t v = alpha;
        for (int k = 0; k < J; k++)
            if (wcols[k][i] & 1) v = bf128_add(v, bk[k]);
        wl[i] = v;
    }
    bf128_t cw;
    bfgkr_prove(std::move(wl), "bflu-gw", tr, pf.gw, rfin_w, cw, gpu);
    // 4. table grand product: leaves 1 + m*u over the (b,j) cube, j = low vars
    std::vector<bf128_t> tf(TN);
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < (int64_t)TN; j++) {
        bf128_t v = alpha;
        for (int k = 0; k < J; k++)
            if (tbits[(size_t)k << lT | j] & 1) v = bf128_add(v, bk[k]);
        tf[j] = v;                                 // (alpha + t_j)^(2^0)
    }
    std::vector<bf128_t> u((size_t)MBP << lT), tl((size_t)MBP << lT);
    const bf128_t one = bf128_one();
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < (int64_t)TN; j++) {
        bf128_t v = tf[j];
        for (int b = 0; b < MBP; b++) {
            if (b) v = bf128_mul(v, v);            // Frobenius step
            size_t x = ((size_t)b << lT) + j;
            u[x] = bf128_add(v, one);
            tl[x] = mbits[x] ? v : one;            // 1 + m*u
        }
    }
    std::vector<bf128_t> rfin_t; bf128_t cl;
    bfgkr_prove(std::move(tl), "bflu-gt", tr, pf.gt, rfin_t, cl, gpu);
    // 5. binding sumcheck: cl + 1 == sum eq(rfin_t,.) * m * u
    std::vector<bf128_t> mv((size_t)MBP << lT);
    #pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < (int64_t)((size_t)MBP << lT); x++)
        mv[x] = mbits[x] ? one : bf128_zero();
    std::vector<bf128_t> zb;
    bflu_sc3(rfin_t, mv, u, tr, pf.bind, zb, gpu);
    // 6. multiplicity opening at the bind endpoint (full-domain point)
    bfpcs_open(mC, zb.data(), pf.bind.finals[1], tr, pf.mopen);
}

// verify.  Outputs rfin_w / cw / alpha / beta powers: the caller MUST check
//   cw == alpha + sum_k bk[k] * colk~(rfin_w)
// against its own commitment opening at rfin_w.
static inline bool bflu_verify(fs::Transcript& tr, int lN, int lT, int J,
                               const uint8_t* tbits, const BfLuProof& pf,
                               std::vector<bf128_t>& rfin_w, bf128_t& cw,
                               bf128_t& alpha, std::vector<bf128_t>& bk,
                               const char** why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (pf.lN != lN || pf.lT != lT || pf.lMB != bflu_lmb(lN)) return fail("lu dims");
    size_t TN = (size_t)1 << lT;
    int MBP = 1 << pf.lMB;
    tr.absorb("bfpcs-root", pf.mroot, 32);
    bf128_t beta = bf_chal128(tr); alpha = bf_chal128(tr);
    bflu_betas(beta, J, bk);
    if (!bfgkr_verify(pf.gw, lN, "bflu-gw", tr, rfin_w, cw)) return fail("lu gw chain");
    std::vector<bf128_t> rfin_t; bf128_t cl;
    if (!bfgkr_verify(pf.gt, lT + pf.lMB, "bflu-gt", tr, rfin_t, cl)) return fail("lu gt chain");
    if (!bf128_eq(pf.gw.root, pf.gt.root)) return fail("lu product roots differ");
    // binding sumcheck
    const BfScProof& b = pf.bind;
    if (b.l != lT + pf.lMB || b.K != 3 || b.D != 3 || (int)b.finals.size() != 3)
        return fail("lu bind shape");
    std::vector<bf128_t> zb; bf128_t E;
    if (!bf_sumcheck_verify(b, bf128_add(cl, bf128_one()), tr, zb, &E))
        return fail("lu bind chain");
    if (!bf128_eq(E, bf128_mul(b.finals[0], bf128_mul(b.finals[1], b.finals[2]))))
        return fail("lu bind terminal");
    if (!bf128_eq(b.finals[0], bflu_eq_point(rfin_t, zb))) return fail("lu bind eq");
    // u~(zb) from PUBLIC data via the Frobenius factorization:
    //   u~(zb_b, zb_j) = sum_b eqb[b] * (alpha^(2^b) + 1 + sum_k bk[k]^(2^b) * Tk~)
    {
        std::vector<bf128_t> zj(zb.begin(), zb.begin() + lT);
        std::vector<bf128_t> zbb(zb.begin() + lT, zb.end());
        std::vector<bf128_t> eqj, eqb;
        bf_eq_table(zj.data(), lT, eqj);
        bf_eq_table(zbb.data(), pf.lMB, eqb);
        std::vector<bf128_t> Tk(J, bf128_zero());
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < J; k++) {
            bf128_t acc = bf128_zero();
            for (size_t j = 0; j < TN; j++)
                if (tbits[(size_t)k << lT | j] & 1) acc = bf128_add(acc, eqj[j]);
            Tk[k] = acc;
        }
        bf128_t uev = bf128_zero();
        bf128_t af = alpha;
        std::vector<bf128_t> bf = bk;
        const bf128_t one = bf128_one();
        for (int bb = 0; bb < MBP; bb++) {
            if (bb) {
                af = bf128_mul(af, af);
                for (int k = 0; k < J; k++) bf[k] = bf128_mul(bf[k], bf[k]);
            }
            bf128_t row = bf128_add(af, one);
            for (int k = 0; k < J; k++)
                row = bf128_add(row, bf128_mul(bf[k], Tk[k]));
            uev = bf128_add(uev, bf128_mul(eqb[bb], row));
        }
        if (!bf128_eq(b.finals[2], uev)) return fail("lu bind u eval");
    }
    // m~(zb) against the multiplicity commitment
    BfPcsParams pm = bflu_mparams(lT, pf.lMB);
    if (!bfpcs_verify(pm, pf.mroot, zb.data(), b.finals[1], tr, pf.mopen))
        return fail("lu mult opening");
    if (why) *why = "ok";
    return true;
}

} // namespace bflu
