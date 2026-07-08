// Binius migration step (design doc section 21.10): the ACCUMULATION gadget --
// the 21.8 carry-save decision, built.  Proves the per-group Hawkeye dot-
// product sums  contribution_g = sum_{kk<32} al_{g,kk}  over the binary tower,
// as POS/NEG-SPLIT BINARY ADDER TREES on the committed sign-magnitude al
// columns (almag[15], alsg) of the per-product gadget's stacked commitment.
//
// Integer addition does not exist in char 2, so every tree node is a ripple-
// carry adder with COMMITTED sum and carry bit-slices ("carry-save columns"):
//   level l (l = 1..5) halves the domain; node y sums nodes 2y and 2y+1 of
//   level l-1.  Per side (P = positive part, N = negative part) and input
//   width win = 14+l the committed columns are s[0..win-1], c[0..win-1]; the
//   (win+1)-bit output value is s | (c[win-1] << win).  Constraints (2 per
//   bit per side, gamma-batched into ONE eq-weighted zerocheck per level):
//     s_j + a_j + b_j + c_{j-1} = 0
//     c_j + a_j*b_j + c_{j-1}*(a_j + b_j) = 0
//   where (a, b) are the even/odd RESTRICTIONS of the level-(l-1) outputs --
//   the level-l zerocheck runs on the (lN-l)-cube and its input columns are
//   the parent columns with the pairing coordinate (row-index bit 0) fixed.
//   At level 1 the inputs are the sign-magnitude mux of the level-0 columns,
//   a_j = almag_j * (1 xor alsg) on the P side and almag_j * alsg on the N
//   side, which raises the carry constraint to degree 4 (D=5 zerocheck at
//   level 1 only; levels 2..5 are degree 2, D=3).
//
// Commitment layout: levels 1..4 are PACKED into a second 64-slot stack
// (2^l columns of length N/2^l per slot, packing bits = TOP l bits of the
// slot-local index); level 5 (the final group sums P[20]/N[20], 4 slots)
// lives in the MAIN stack's free slots so the outputs sit next to the
// per-product witness.  Sum ("out") and carry ("int") columns of one level
// never share a slot: each binding point must cover every slot it derives
// COMPLETELY from sumcheck finals, or not at all.
//
// Binding (per level): point A_l = (zeta_l, tau_l) binds the level's
// committed-column finals; point B_l = (sigma_l, zeta_l, taup_l) binds the
// level's INPUT finals to the level-(l-1) output slots (B_1 lands on the
// main stack's almag/alsg columns -- this is what welds the accumulation to
// the per-product proof).  For slots not derivable from finals at a point the
// prover supplies their evals (transcript-absorbed); lying there is caught by
// the multi-point PCS opening, which checks the TRUE stacked eval.
//
// REQUIRES group-contiguous row order: row index = group*32 + kk (kk is the
// low 5 bits), i.e. the hawkeye_ref.py group_witness_rows layout.  The
// per-product zerocheck and DM lookup are row-order-agnostic, so this is a
// pure relabeling of the same witness.  Teeth: p3_binius_acc_test.cu.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_binius_sumcheck.cuh"

namespace bacc {

static const int ACCL = 5;                       // 2^5 = 32 products per group
// per level l: input width win, acc-stack slot bases for out/int column groups
static const int A_WIN[6]  = {0, 15, 16, 17, 18, 19};
static const int A_OUTB[6] = {0,  0, 30, 47, 56,  0};   // level 5 -> main stack
static const int A_INTB[6] = {0, 16, 39, 52, 59,  0};
static const int LOGSTK2 = 6, NSTK2 = 64;        // second stack: 62 used slots

static inline int acc_nc(int l)  { return 4 * A_WIN[l]; }          // constraints
static inline int acc_ni2(int l) { return l == 1 ? 16 : 2 * A_WIN[l]; } // inputs/parity

// slot + packed sub-index of (level l, grp: 0 out / 1 int, col); level 5 maps
// into the main stack at mainbase (out) / mainbase+2 (int)
static inline void acc_slot(int l, int grp, int col, int mainbase, int& slot, int& t) {
    int base = (l == ACCL) ? mainbase + (grp ? 2 : 0)
                           : (grp ? A_INTB[l] : A_OUTB[l]);
    slot = base + (col >> l);
    t = col & ((1 << l) - 1);
}
// canonical committed-column enumeration k in [0, 4*win):
// [sP_0..sP_{win-1}, cP_0..cP_{win-1}, sN_..., cN_...] -> (grp, col-in-group).
// out-group col order is [P: s_0..s_{win-1}, c_{win-1} | N: same] so that the
// level-(l+1) input bit b of a side is exactly out col side*(win+1)+b.
static inline void acc_cmap(int win, int k, int& grp, int& col) {
    int side = k / (2 * win), r = k % (2 * win);
    if (r < win) { grp = 0; col = side * (win + 1) + r; }
    else {
        int j = r - win;
        if (j < win - 1) { grp = 1; col = side * (win - 1) + j; }
        else             { grp = 0; col = side * (win + 1) + win; }
    }
}

// ---- witness ----
struct Wit {
    int lN = 0;
    std::vector<uint8_t> bits;                   // second stack, NSTK2 * N bytes
    std::vector<uint32_t> sumP, sumN;            // level-5 values (teeth aid)
};

// build the adder-tree bit witness from the committed almag/alsg columns of
// the main bit array; writes levels 1..4 into w.bits and level 5 into the
// main array's slots [mainbase, mainbase+4)
static inline void build(int lN, std::vector<uint8_t>& mb, int colALM, int colALS,
                         int mainbase, Wit& w) {
    size_t N = (size_t)1 << lN;
    w.lN = lN;
    w.bits.assign((size_t)NSTK2 * N, 0);
    std::vector<uint32_t> P(N), Q(N);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)N; i++) {
        uint32_t m = 0;
        for (int j = 0; j < 15; j++)
            m |= (uint32_t)(mb[(size_t)(colALM + j) * N + i] & 1) << j;
        uint32_t s = mb[(size_t)colALS * N + i] & 1;
        P[i] = s ? 0 : m;
        Q[i] = s ? m : 0;
    }
    for (int l = 1; l <= ACCL; l++) {
        int win = A_WIN[l];
        size_t half = N >> l;
        std::vector<uint32_t> P2(half), Q2(half);
        uint8_t* dst = (l == ACCL) ? mb.data() : w.bits.data();
        #pragma omp parallel for schedule(static)
        for (int64_t y = 0; y < (int64_t)half; y++) {
            for (int side = 0; side < 2; side++) {
                const std::vector<uint32_t>& V = side ? Q : P;
                uint32_t a = V[2 * y], b = V[2 * y + 1], c = 0;
                for (int j = 0; j < win; j++) {
                    uint32_t aj = (a >> j) & 1, bj = (b >> j) & 1;
                    uint32_t sj = aj ^ bj ^ c;
                    uint32_t cj = (aj & bj) | (c & (aj ^ bj));
                    int slot, t;
                    acc_slot(l, 0, side * (win + 1) + j, mainbase, slot, t);
                    dst[((size_t)slot << lN) + ((size_t)t << (lN - l)) + y] = (uint8_t)sj;
                    if (j < win - 1) acc_slot(l, 1, side * (win - 1) + j, mainbase, slot, t);
                    else             acc_slot(l, 0, side * (win + 1) + win, mainbase, slot, t);
                    dst[((size_t)slot << lN) + ((size_t)t << (lN - l)) + y] = (uint8_t)cj;
                    c = cj;
                }
                (side ? Q2 : P2)[y] = a + b;
            }
        }
        P.swap(P2); Q.swap(Q2);
    }
    w.sumP = P; w.sumN = Q;
}

// ---- the per-level constraint functor.  w[0] = eq, then the even and odd
// input restrictions (acc_ni2 each), then the 4*win committed columns in
// canonical order.  g = the 4*win gamma powers. ----
template <int LVL> struct AccF {
    static constexpr int WIN = 14 + LVL;
    static constexpr int NI  = (LVL == 1) ? 32 : 4 * WIN;
    static constexpr int NCM = 4 * WIN;
    static constexpr int K = 1 + NI + NCM;
    static constexpr int D = (LVL == 1) ? 5 : 3;
    const bf128_t* g;
    __host__ __device__ bf128_t operator()(const bf128_t* w) const {
        const bf128_t* E = w + 1;
        const bf128_t* O = w + 1 + NI / 2;
        const bf128_t* CM = w + 1 + NI;
        bf128_t acc = bf128_zero();
        int c = 0;
        for (int side = 0; side < 2; side++) {
            const bf128_t* s  = CM + 2 * side * WIN;
            const bf128_t* cc = CM + (2 * side + 1) * WIN;
            bf128_t prev = bf128_zero();
            for (int j = 0; j < WIN; j++) {
                bf128_t a, b;
                if (LVL == 1) {          // sign-mux of the level-0 inputs
                    bf128_t se = E[15], so = O[15];
                    a = side ? bf128_mul(E[j], se)
                             : bf128_add(E[j], bf128_mul(E[j], se));
                    b = side ? bf128_mul(O[j], so)
                             : bf128_add(O[j], bf128_mul(O[j], so));
                } else {
                    a = E[side * WIN + j];
                    b = O[side * WIN + j];
                }
                bf128_t axb = bf128_add(a, b);
                bf128_t x = bf128_add(bf128_add(s[j], axb), prev);
                acc = bf128_add(acc, bf128_mul(g[c++], x));
                x = bf128_add(cc[j], bf128_add(bf128_mul(a, b), bf128_mul(prev, axb)));
                acc = bf128_add(acc, bf128_mul(g[c++], x));
                prev = cc[j];
            }
        }
        return bf128_mul(w[0], acc);
    }
};

// ---- extract the level-l sumcheck byte columns (inputs even, inputs odd,
// committed; col-major, K-1 columns of length N>>l).  ab = acc-stack bits,
// mb = main bits. ----
static inline void acc_sc_bytes(int lN, int l, const uint8_t* mb, int colALM, int colALS,
                                int mainbase, const uint8_t* ab, std::vector<uint8_t>& out) {
    size_t N = (size_t)1 << lN, half = N >> l;
    int win = A_WIN[l], ni2 = acc_ni2(l), ncm = acc_nc(l);
    int KM1 = 2 * ni2 + ncm;
    out.assign((size_t)KM1 * half, 0);
    #pragma omp parallel for schedule(static)
    for (int64_t y = 0; y < (int64_t)half; y++) {
        for (int par = 0; par < 2; par++)
            for (int k = 0; k < ni2; k++) {
                uint8_t v;
                if (l == 1) {
                    int col = (k < 15) ? colALM + k : colALS;
                    v = mb[((size_t)col << lN) + 2 * y + par];
                } else {                 // parent out col k at node 2y+par
                    int slot, t;
                    acc_slot(l - 1, 0, k, mainbase, slot, t);
                    v = ab[((size_t)slot << lN) + ((size_t)t << (lN - (l - 1))) + 2 * y + par];
                }
                out[(size_t)(par * ni2 + k) * half + y] = v & 1;
            }
        for (int k = 0; k < ncm; k++) {
            int grp, col, slot, t;
            acc_cmap(win, k, grp, col);
            acc_slot(l, grp, col, mainbase, slot, t);
            const uint8_t* src = (l == ACCL) ? mb : ab;
            out[(size_t)(2 * ni2 + k) * half + y] =
                src[((size_t)slot << lN) + ((size_t)t << (lN - l)) + y] & 1;
        }
    }
}

// eval every slot bit-column of a stack at an lN-coordinate row point
static inline void acc_stack_evals(const uint8_t* bits, int nslot, int lN,
                                   const bf128_t* r, bf128_t* out) {
    size_t N = (size_t)1 << lN;
    std::vector<bf128_t> eq;
    bf_eq_table(r, lN, eq);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int s = 0; s < nslot; s++) {
        bf128_t a = bf128_zero();
        const uint8_t* c = bits + ((size_t)s << lN);
        for (size_t i = 0; i < N; i++)
            if (c[i] & 1) a = bf128_add(a, eq[i]);
        out[s] = a;
    }
}

// ---- verifier-side derived slot evals (the actual binding) ----
// A_l: every level-l slot from the level's committed finals (fin + 1 + NI)
static inline void acc_derive_A(int l, int mainbase, const bf128_t* fin, int ni,
                                const bf128_t* eqtau, bf128_t* der, uint8_t* have) {
    int win = A_WIN[l];
    for (int k = 0; k < 4 * win; k++) {
        int grp, col, slot, t;
        acc_cmap(win, k, grp, col);
        acc_slot(l, grp, col, mainbase, slot, t);
        if (!have[slot]) { have[slot] = 1; der[slot] = bf128_zero(); }
        der[slot] = bf128_add(der[slot], bf128_mul(eqtau[t], fin[1 + ni + k]));
    }
}
// B_l (l >= 2): the level-(l-1) OUT slots from the level-l input finals
static inline void acc_derive_B(int l, int mainbase, const bf128_t* fin,
                                bf128_t sig, const bf128_t* eqtaup,
                                bf128_t* der, uint8_t* have) {
    int ni2 = acc_ni2(l);
    for (int k = 0; k < ni2; k++) {
        int slot, t;
        acc_slot(l - 1, 0, k, mainbase, slot, t);
        bf128_t fe = fin[1 + k], fo = fin[1 + ni2 + k];
        bf128_t v = bf128_add(fe, bf128_mul(sig, bf128_add(fe, fo)));  // (1+s)fe+s*fo
        if (!have[slot]) { have[slot] = 1; der[slot] = bf128_zero(); }
        der[slot] = bf128_add(der[slot], bf128_mul(eqtaup[t], v));
    }
}

// ---- proof piece (embedded in the Hawkeye proof) ----
struct AccProof {
    int on = 0;
    uint8_t root2[32];
    BfScProof sc[5];
    std::vector<bf128_t> evB1, evA5;             // supplied main-stack slot evals
    std::vector<bf128_t> evA[4], evB[4];         // supplied acc-stack slot evals
    BfPcsProofM pcs2;
    size_t bytes() const {
        if (!on) return sizeof(int);
        size_t s = sizeof(int) + 32 + pcs2.bytes() +
                   (evB1.size() + evA5.size()) * sizeof(bf128_t);
        for (int i = 0; i < 5; i++)
            s += (sc[i].rounds.size() + sc[i].finals.size()) * sizeof(bf128_t);
        for (int i = 0; i < 4; i++)
            s += (evA[i].size() + evB[i].size()) * sizeof(bf128_t);
        return s;
    }
};

static inline BfPcsParams acc_params(int lN) {
    BfPcsParams p;
    p.l = lN + LOGSTK2; p.lcol = p.l / 2; p.lrow = p.l - p.lcol; p.Q = 100;
    return p;
}

// gamma powers for one level from one transcript challenge
static inline void acc_gammas(fs::Transcript& tr, int l, std::vector<bf128_t>& g) {
    bf128_t gamma = bf_chal128(tr);
    g.resize(acc_nc(l));
    g[0] = bf128_one();
    for (int c = 1; c < (int)g.size(); c++) g[c] = bf128_mul(g[c - 1], gamma);
}

// one level's zerocheck, GPU or host, byte-identical either way.  ws is a
// PRE-RESERVED device workspace (bfsc_dev_reserve for the level-1 shape, the
// largest) shared across all five levels -- per-level cudaMalloc/cudaFree of
// hundreds of MB costs ~200 ms/level otherwise (measured).
template <int L>
static inline void acc_zc(int lN, const std::vector<uint8_t>& sb, fs::Transcript& tr,
                          BfScProof& pf, std::vector<bf128_t>& zeta,
                          const std::vector<bf128_t>& g, bool gpu,
                          BfScDev* ws = nullptr, uint8_t* d_scratch = nullptr) {
    using CF = AccF<L>;
    int lz = lN - L;
    size_t half = (size_t)1 << lz;
    std::vector<bf128_t> rz(lz);
    for (auto& x : rz) x = bf_chal128(tr);
    if (gpu) {
        BfScDev own; BfScDev& dv = ws ? *ws : own;
        if (!ws) bfsc_dev_alloc(dv, CF::K, lz);
        else     bfsc_dev_shape(dv, CF::K, lz);
        bfsc_dev_eq(dv.a, rz.data(), lz);
        uint8_t* d_b = d_scratch;
        if (!d_scratch) cudaMalloc(&d_b, sb.size());
        cudaMemcpy(d_b, sb.data(), sb.size(), cudaMemcpyHostToDevice);
        bfsc_bits_kernel<<<(uint32_t)((sb.size() + 255) / 256), 256>>>(
            d_b, dv.a + dv.n, sb.size());
        if (!d_scratch) cudaFree(d_b);
        bf128_t* d_g;
        cudaMalloc(&d_g, g.size() * sizeof(bf128_t));
        cudaMemcpy(d_g, g.data(), g.size() * sizeof(bf128_t), cudaMemcpyHostToDevice);
        CF cf; cf.g = d_g;
        bf_sumcheck_prove_gpu(dv, cf, tr, pf, zeta);
        cudaFree(d_g);
        if (!ws) bfsc_dev_free(dv);
    } else {
        std::vector<std::vector<bf128_t>> cols(CF::K);
        bf_eq_table(rz.data(), lz, cols[0]);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int k = 1; k < CF::K; k++) {
            cols[k].resize(half);
            for (size_t y = 0; y < half; y++)
                cols[k][y] = sb[(size_t)(k - 1) * half + y] ? bf128_one() : bf128_zero();
        }
        CF cf; cf.g = g.data();
        auto fn = [](const bf128_t* w, const void* ctx) {
            return (*(const CF*)ctx)(w);
        };
        bf_sumcheck_prove(lz, CF::K, cols, CF::D, fn, &cf, tr, pf, zeta);
    }
}

// verifier half of one level: replay + integrand + eq-final checks
template <int L>
static inline bool acc_zc_verify(int lN, const BfScProof& pf, fs::Transcript& tr,
                                 std::vector<bf128_t>& zeta,
                                 const std::vector<bf128_t>& g) {
    using CF = AccF<L>;
    int lz = lN - L;
    std::vector<bf128_t> rz(lz);
    for (auto& x : rz) x = bf_chal128(tr);
    if ((int)pf.finals.size() != CF::K || pf.l != lz || pf.D != CF::D) return false;
    bf128_t E;
    if (!bf_sumcheck_verify(pf, bf128_zero(), tr, zeta, &E)) return false;
    CF cf; cf.g = g.data();
    if (!bf128_eq(E, cf(pf.finals.data()))) return false;
    bf128_t eqv = bf128_one();
    for (int t = 0; t < lz; t++)
        eqv = bf128_mul(eqv, bf128_add(bf128_one(), bf128_add(rz[t], zeta[t])));
    return bf128_eq(pf.finals[0], eqv);
}

} // namespace bacc
