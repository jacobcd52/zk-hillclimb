// Binius migration step (design doc section 21.8): the REAL Hawkeye
// per-product alignment gadget (q / r / align -- the p3_hawkeye_prod.cuh
// semantics) proven over the binary tower at TRUE bit width.
//
// Per product row the prover commits 110 F_2 bit-slices, stacked (column id =
// top 7 index bits, 128 slots, 18 zero pads) into ONE Binius PCS commitment:
//     mag[15] sg pr sh[6] h[16] t1 t2 m3 o1 q[15] r[15] almag[15] alsg   (89
//     constrained)  +  a[8] b[8] eb[5]  (committed-only, DM lookup pending)
// vs the Goldilocks gadget's 11 x 64-bit columns.  Booleanity is STRUCTURAL
// (the packed commitment can only contain bits).
//
// The Goldilocks gadget's semantics map to 401 gamma-batched degree-2
// constraints (x eq -> ONE degree-3 zerocheck):
//   C1  (q*pw + r = mag, pw = 2^min(sh,15)) becomes the shift-mux
//        mag_j = sum_s h_s * (j >= s ? q_{j-s} : r_j)             [15]
//        with h one-hot (pairwise products + XOR-parity)          [121]
//        linked to the committed sh bits via helper bits t1 = sh0*sh1,
//        t2 = t1*sh2, m3 = t2*sh3, o1 = sh4 OR sh5 (h_15 = o1 OR m3 covers
//        min(.,15); min-bits force the selected s = sh low bits)  [9]
//   REM lookup (r < pw)   -> structural bits:  h_s * r_j = 0, j >= s   [120]
//   RANGE15 lookup (q<2^15)-> structural bits: h_s * q_j = 0, j >= 15-s [120]
//   SH lookup (sh -> pw)  -> the h/sh linkage above (in-circuit)
//   C2  (al = pr*(1-2sg)*q) in sign-magnitude: almag_j = pr * q_j [15],
//        alsg = sg * pr                                           [1]
//   DM lookup (a,b -> eb,mag,sg,pr; the 65536-row fused decode-multiply-scale
//        table) via logUp over towers (p3_binius_logup.cuh, section 21.9):
//        the 38 tuple bit-slices are fingerprinted per row and proven a
//        multiset of table-row fingerprints by the multiplicative grand-
//        product argument (char-2-sound; binary multiplicities, Frobenius
//        exponents).  The lookup's leaf claim binds to the SAME stacked
//        commitment through a second point of one multi-point opening.
// With the DM lookup landed the gadget covers the FULL p3_hawkeye_prod.cuh
// per-product semantics.
//
// Teeth + measured A/B vs the Goldilocks p3hw gadget: p3_binius_hawkeye_test.cu.
#pragma once
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include "p3_binius_logup.cuh"
#include "p3_binius_acc.cuh"
#include "p3_binius_trans.cuh"

namespace bhw {

static inline double bhw_now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- stacked column layout ----
enum {
    LMAG = 0,               // 15
    LSG = 15, LPR = 16,
    LSH = 17,               // 6
    LH = 23,                // 16
    LT1 = 39, LT2 = 40, LM3 = 41, LO1 = 42,
    LQ = 43,                // 15
    LR = 58,                // 15
    LALM = 73,              // 15
    LALS = 88,
    NCON = 89,              // constrained columns (zerocheck)
    LA = 89,                // 8   committed-only below
    LB = 97,                // 8
    LEB = 105,              // 5
    NCOLB = 110,
    L5BASE = 110,           // 4   accumulation level-5 group sums (acc mode)
    LPC = 114,              // 6   link carries eb + sh = ME12 (trans mode)
    LTSEL = 120,            // 1   tightness selector t (trans mode)
    LORT = 121, LORP = 122, // 2   OR-tree levels 1..4 over t / pr (heap rows)
    NCOLA = 123,            // slots covered by the xev/xev2 bindings
    LOGSTK = 7, NSTK = 128  // stacked slots
};
static const int NC = 401;  // batched constraint count

// ---- DM lookup wiring: tuple column order a[8] b[8] eb[5] mag[15] sg pr ----
static const int DMJ = 38;
static inline void bhw_dm_cols(int* out) {
    int c = 0;
    for (int j = 0; j < 8; j++) out[c++] = LA + j;
    for (int j = 0; j < 8; j++) out[c++] = LB + j;
    for (int j = 0; j < 5; j++) out[c++] = LEB + j;
    for (int j = 0; j < 15; j++) out[c++] = LMAG + j;
    out[c++] = LSG; out[c++] = LPR;
}
static inline void bhw_decode_e4m3(uint32_t raw, int& exp_eff, int& sig_abs, int& sign) {
    sign = (raw >> 7) & 1;
    int eb = (raw >> 3) & 15, mant = raw & 7;
    sig_abs = eb != 0 ? (mant | 8) : mant;
    exp_eff = eb != 0 ? eb : 1;
}
// the 65536-row DM table as DMJ public bit-columns (col-major, 2^16 each);
// bitwise-identical content to p3hw::build_tables().DM (cross-checked by a
// tooth in the selftest)
static inline const std::vector<uint8_t>& bhw_dm_table_bits() {
    static std::vector<uint8_t> tb = [] {
        std::vector<uint8_t> t((size_t)DMJ * 65536, 0);
        auto B = [&](int col, uint32_t j) -> uint8_t& { return t[(size_t)col * 65536 + j]; };
        for (uint32_t j = 0; j < 65536; j++) {
            uint32_t ca = j >> 8, cb = j & 255;
            int ea, siga, sna, ebx, sigb, snb;
            bhw_decode_e4m3(ca, ea, siga, sna); bhw_decode_e4m3(cb, ebx, sigb, snb);
            uint32_t eb = (uint32_t)(ea + ebx - 2);               // in [0,28]
            uint32_t mag = (uint32_t)(siga * sigb) << 7;          // < 2^15
            uint32_t pr = (siga != 0 && sigb != 0) ? 1 : 0;
            uint32_t sg = (pr && (sna ^ snb)) ? 1 : 0;
            int c = 0;
            for (int k = 0; k < 8; k++) B(c++, j) = (ca >> k) & 1;
            for (int k = 0; k < 8; k++) B(c++, j) = (cb >> k) & 1;
            for (int k = 0; k < 5; k++) B(c++, j) = (eb >> k) & 1;
            for (int k = 0; k < 15; k++) B(c++, j) = (mag >> k) & 1;
            B(c++, j) = (uint8_t)sg; B(c++, j) = (uint8_t)pr;
        }
        return t;
    }();
    return tb;
}

// ---- witness bit build from the raw integer vectors (numpy ref layout:
// a,b,eb,mag,sg,pr,sh,q,r,al).  Pads to 2^lN with the all-zero product row
// (h_0 = 1), a valid row of every constraint. ----
static inline int bhw_build_bits(const std::vector<int64_t>* raw /*10 arrays*/,
                                 std::vector<uint8_t>& bits, size_t* n_real = nullptr) {
    size_t n = raw[0].size(), N = 1; int lN = 0;
    while (N < n) { N <<= 1; lN++; }
    if (n_real) *n_real = n;
    bits.assign((size_t)NSTK * N, 0);
    auto B = [&](int col, size_t i) -> uint8_t& { return bits[(size_t)col * N + i]; };
    for (size_t i = 0; i < N; i++) {
        int64_t a = 0, b = 0, eb = 0, mag = 0, sg = 0, pr = 0, sh = 0, q = 0, r = 0;
        if (i < n) { a = raw[0][i]; b = raw[1][i]; eb = raw[2][i]; mag = raw[3][i];
                     sg = raw[4][i]; pr = raw[5][i]; sh = raw[6][i]; q = raw[7][i];
                     r = raw[8][i]; }
        int s = sh < 15 ? (int)sh : 15;
        for (int j = 0; j < 15; j++) B(LMAG + j, i) = (mag >> j) & 1;
        B(LSG, i) = (uint8_t)sg; B(LPR, i) = (uint8_t)pr;
        for (int j = 0; j < 6; j++) B(LSH + j, i) = (sh >> j) & 1;
        B(LH + s, i) = 1;
        uint8_t s0 = (sh >> 0) & 1, s1 = (sh >> 1) & 1, s2 = (sh >> 2) & 1,
                s3 = (sh >> 3) & 1, s4 = (sh >> 4) & 1, s5 = (sh >> 5) & 1;
        uint8_t t1 = s0 & s1, t2 = t1 & s2, m3 = t2 & s3, o1 = s4 | s5;
        B(LT1, i) = t1; B(LT2, i) = t2; B(LM3, i) = m3; B(LO1, i) = o1;
        for (int j = 0; j < 15; j++) B(LQ + j, i) = (q >> j) & 1;
        for (int j = 0; j < 15; j++) B(LR + j, i) = (r >> j) & 1;
        int64_t alm = pr ? q : 0;
        for (int j = 0; j < 15; j++) B(LALM + j, i) = (alm >> j) & 1;
        B(LALS, i) = (uint8_t)(sg & pr);
        for (int j = 0; j < 8; j++) B(LA + j, i) = (a >> j) & 1;
        for (int j = 0; j < 8; j++) B(LB + j, i) = (b >> j) & 1;
        for (int j = 0; j < 5; j++) B(LEB + j, i) = (eb >> j) & 1;
    }
    return lN;
}

// ---- the batched constraint.  w[0] = eq, w[1 + c] = constrained column c.
// g points to the NC gamma powers (host or device).  The factored sums below
// are the SAME linear combination sum_c g[c] * C_c, term for term; the gamma
// index c advances in the fixed order documented in the header comment. ----
struct HwF {
    static constexpr int K = 1 + NCON, D = 3;
    const bf128_t* g;
    __host__ __device__ bf128_t operator()(const bf128_t* w) const {
        const bf128_t *MAG = w + 1 + LMAG, *SH = w + 1 + LSH, *H = w + 1 + LH,
                      *Qb = w + 1 + LQ, *Rb = w + 1 + LR, *ALM = w + 1 + LALM;
        const bf128_t sg = w[1 + LSG], pr = w[1 + LPR], t1 = w[1 + LT1],
                      t2 = w[1 + LT2], m3 = w[1 + LM3], o1 = w[1 + LO1],
                      als = w[1 + LALS];
        const bf128_t one = bf128_one();
        bf128_t acc = bf128_zero();
        int c = 0;
        // 1. one-hot pairwise: sum_{s<t} g[c] h_s h_t   (factored per s)
        for (int s = 0; s < 16; s++) {
            bf128_t inner = bf128_zero();
            for (int t = s + 1; t < 16; t++)
                inner = bf128_add(inner, bf128_mul(g[c++], H[t]));
            acc = bf128_add(acc, bf128_mul(H[s], inner));
        }
        // 2. parity: XOR_s h_s + 1
        {
            bf128_t x = one;
            for (int s = 0; s < 16; s++) x = bf128_add(x, H[s]);
            acc = bf128_add(acc, bf128_mul(g[c++], x));
        }
        // 3. helper definitions
        acc = bf128_add(acc, bf128_mul(g[c++], bf128_add(t1, bf128_mul(SH[0], SH[1]))));
        acc = bf128_add(acc, bf128_mul(g[c++], bf128_add(t2, bf128_mul(t1, SH[2]))));
        acc = bf128_add(acc, bf128_mul(g[c++], bf128_add(m3, bf128_mul(t2, SH[3]))));
        acc = bf128_add(acc, bf128_mul(g[c++],
              bf128_add(bf128_add(o1, bf128_add(SH[4], SH[5])), bf128_mul(SH[4], SH[5]))));
        // 4. g-link: h_15 + o1 + m3 + o1*m3
        acc = bf128_add(acc, bf128_mul(g[c++],
              bf128_add(bf128_add(H[15], bf128_add(o1, m3)), bf128_mul(o1, m3))));
        // 5. min-bits i=0..3: (XOR_{s<15, bit_i(s)=1} h_s) + sh_i + h_15*sh_i
        //    (the s=15 term of the selected-bits XOR cancels the standalone
        //    h_15 of  u_i + h_15 + sh_i + h_15*sh_i, so the loop excludes it)
        for (int i = 0; i < 4; i++) {
            bf128_t x = SH[i];
            for (int s = 0; s < 15; s++)
                if ((s >> i) & 1) x = bf128_add(x, H[s]);
            x = bf128_add(x, bf128_mul(H[15], SH[i]));
            acc = bf128_add(acc, bf128_mul(g[c++], x));
        }
        // 6. shift-mux j=0..14: g[c+j]*(mag_j + sum_s h_s v(j,s)), factored:
        //    sum_j g mag_j  +  sum_s h_s * (sum_j g v(j,s))
        {
            const bf128_t* gm = g + c;
            bf128_t x = bf128_zero();
            for (int j = 0; j < 15; j++) x = bf128_add(x, bf128_mul(gm[j], MAG[j]));
            for (int s = 0; s < 16; s++) {
                bf128_t inner = bf128_zero();
                for (int j = 0; j < 15; j++)
                    inner = bf128_add(inner, bf128_mul(gm[j], j >= s ? Qb[j - s] : Rb[j]));
                x = bf128_add(x, bf128_mul(H[s], inner));
            }
            acc = bf128_add(acc, x);
            c += 15;
        }
        // 7. r range: h_s * r_j = 0 for j >= s (s=0..15, j<=14), factored per s
        for (int s = 0; s < 16; s++) {
            bf128_t inner = bf128_zero();
            for (int j = s; j < 15; j++)
                inner = bf128_add(inner, bf128_mul(g[c++], Rb[j]));
            if (s < 15) acc = bf128_add(acc, bf128_mul(H[s], inner));
        }
        // 8. q range: h_s * q_j = 0 for j >= 15-s, factored per s
        for (int s = 0; s < 16; s++) {
            bf128_t inner = bf128_zero();
            for (int j = 15 - s; j < 15; j++)
                inner = bf128_add(inner, bf128_mul(g[c++], Qb[j]));
            if (s > 0) acc = bf128_add(acc, bf128_mul(H[s], inner));
        }
        // 9. C2 magnitude: almag_j + pr*q_j, factored
        {
            const bf128_t* gm = g + c;
            bf128_t x = bf128_zero(), xq = bf128_zero();
            for (int j = 0; j < 15; j++) {
                x = bf128_add(x, bf128_mul(gm[j], ALM[j]));
                xq = bf128_add(xq, bf128_mul(gm[j], Qb[j]));
            }
            acc = bf128_add(acc, bf128_add(x, bf128_mul(pr, xq)));
            c += 15;
        }
        // 10. C2 sign: alsg + sg*pr
        acc = bf128_add(acc, bf128_mul(g[c++], bf128_add(als, bf128_mul(sg, pr))));
        // c == NC by construction (asserted host-side in the selftest)
        return bf128_mul(w[0], acc);
    }
};

// host-side per-row witness validator (debug/teeth aid): returns the number
// of rows on which some raw constraint is violated
static inline size_t bhw_validate(const std::vector<uint8_t>& bits, int lN) {
    size_t N = (size_t)1 << lN, bad = 0;
    auto B = [&](int col, size_t i) -> int { return bits[(size_t)col * N + i] & 1; };
    for (size_t i = 0; i < N; i++) {
        bool ok = true;
        int hs = -1, nh = 0;
        for (int s = 0; s < 16; s++) if (B(LH + s, i)) { hs = s; nh++; }
        if (nh != 1) ok = false;
        int sh = 0; for (int j = 0; j < 6; j++) sh |= B(LSH + j, i) << j;
        if (ok && hs != (sh < 15 ? sh : 15)) ok = false;
        int t1 = B(LSH, i) & B(LSH + 1, i), t2 = t1 & B(LSH + 2, i), m3 = t2 & B(LSH + 3, i);
        int o1 = B(LSH + 4, i) | B(LSH + 5, i);
        if (B(LT1, i) != t1 || B(LT2, i) != t2 || B(LM3, i) != m3 || B(LO1, i) != o1) ok = false;
        int mag = 0, q = 0, r = 0, alm = 0;
        for (int j = 0; j < 15; j++) {
            mag |= B(LMAG + j, i) << j; q |= B(LQ + j, i) << j;
            r |= B(LR + j, i) << j; alm |= B(LALM + j, i) << j;
        }
        if (ok) {
            int s = hs;
            if (r >= (1 << s) || q >= (1 << (15 - s))) ok = false;
            if (mag != (q << s) + r) ok = false;
        }
        int pr = B(LPR, i), sg = B(LSG, i);
        if (alm != (pr ? q : 0) || B(LALS, i) != (sg & pr)) ok = false;
        if (!ok) bad++;
    }
    return bad;
}

// ---- proof ----
struct BhwProof {
    int lN = 0;
    uint8_t root[32];
    bflu::BfLuProof lu;          // DM lookup (multiplicative logUp over towers)
    BfScProof sc;
    bf128_t xev[NCOLA - NCON];   // committed-only col evals at zeta (opening-bound)
    bf128_t xev2[NCOLA];         // ALL column evals at the lookup point rfin_w
    BfPcsProofM pcs;             // ONE multi-point main-stack opening
    bacc::AccProof acc;          // accumulation gadget (section 21.10), if on
    btr::TrProof tr;             // group transition gadget (21.11), if on
    size_t bytes() const {
        return 32 + lu.bytes() + sc.rounds.size() * 16 + sc.finals.size() * 16 +
               sizeof(xev) + sizeof(xev2) + pcs.bytes() + acc.bytes() + tr.bytes();
    }
};
struct BhwStats { double commit_ms = 0, lu_ms = 0, sc_ms = 0, open_ms = 0,
                  verify_ms = 0, acc_ms = 0, tr_ms = 0; size_t committed = 0; };

static inline BfPcsParams bhw_params(int lN) {
    BfPcsParams p;
    p.l = lN + LOGSTK; p.lcol = p.l / 2; p.lrow = p.l - p.lcol; p.Q = 100;
    return p;
}

// gamma powers from one transcript challenge
static inline void bhw_gammas(fs::Transcript& tr, std::vector<bf128_t>& g) {
    bf128_t gamma = bf_chal128(tr);
    g.resize(NC);
    g[0] = bf128_one();
    for (int c = 1; c < NC; c++) g[c] = bf128_mul(g[c - 1], gamma);
}

// acc_in (test/attack hook): alternate main-bit array the accumulation
// sumchecks READ from (default: bits, the committed array).  An attacker
// proving a consistent adder tree over inputs that differ from the committed
// almag/alsg columns is exactly what the B_1 restriction binding must catch.
static inline void bhw_prove(int lN, const std::vector<uint8_t>& bits,
                             BhwProof& pf, BhwStats& st, bool gpu = true,
                             const bacc::Wit* aw = nullptr,
                             const uint8_t* acc_in = nullptr,
                             const btr::Wit* tw = nullptr) {
    size_t N = (size_t)1 << lN;
    pf.lN = lN;
    pf.acc.on = aw ? 1 : 0;
    pf.tr.on = tw ? 1 : 0;
    pf.tr.CH = tw ? tw->CH : 1;
    BfPcsParams p = bhw_params(lN);
    fs::Transcript tr("binius-hawkeye");
    tr.absorb("stmt-bhw", &lN, sizeof lN);
    tr.absorb("stmt-acc", &pf.acc.on, sizeof pf.acc.on);
    tr.absorb("stmt-tr", &pf.tr.on, sizeof pf.tr.on);
    if (tw) tr.absorb("stmt-tr-ch", &pf.tr.CH, sizeof pf.tr.CH);
    BfPcsCommit C;
    bfpcs_commit(p, bits.data(), tr, C);
    st.commit_ms = C.commit_ms; st.committed = C.committed_bytes;
    memcpy(pf.root, C.root, 32);
    // accumulation stack committed up front (levels 1..4; level 5 is in `bits`)
    BfPcsCommit C2;
    if (aw) {
        double ta = bhw_now_ms();
        bfpcs_commit(bacc::acc_params(lN), aw->bits.data(), tr, C2);
        memcpy(pf.acc.root2, C2.root, 32);
        st.committed += C2.committed_bytes;
        st.acc_ms += bhw_now_ms() - ta;
    }
    // transition stack (21.11) committed up front as well
    BfPcsCommit C3;
    if (tw) {
        double ta = bhw_now_ms();
        bfpcs_commit(btr::tr_params(lN), tw->bits.data(), tr, C3);
        memcpy(pf.tr.root3, C3.root, 32);
        st.committed += C3.committed_bytes;
        st.tr_ms += bhw_now_ms() - ta;
    }
    // DM lookup: the 38 tuple columns against the public decode-multiply table
    double t0 = bhw_now_ms();
    std::vector<bf128_t> rfin_w;
    {
        int dmc[DMJ]; bhw_dm_cols(dmc);
        const uint8_t* wc[DMJ];
        for (int k = 0; k < DMJ; k++) wc[k] = bits.data() + (size_t)dmc[k] * N;
        std::vector<uint32_t> idx(N);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)N; i++) {
            uint32_t a = 0, b = 0;
            for (int j = 0; j < 8; j++) {
                a |= (uint32_t)(bits[(size_t)(LA + j) * N + i] & 1) << j;
                b |= (uint32_t)(bits[(size_t)(LB + j) * N + i] & 1) << j;
            }
            idx[i] = (a << 8) | b;
        }
        size_t mbytes = 0;
        bflu::bflu_prove(tr, lN, 16, DMJ, wc, idx.data(), bhw_dm_table_bits().data(),
                         pf.lu, rfin_w, gpu, nullptr, nullptr, &mbytes);
        st.committed += mbytes;
    }
    st.lu_ms = bhw_now_ms() - t0;
    std::vector<bf128_t> rz(lN);
    for (auto& x : rz) x = bf_chal128(tr);
    std::vector<bf128_t> g;
    bhw_gammas(tr, g);
    std::vector<bf128_t> zeta;
    t0 = bhw_now_ms();
    if (gpu) {
        BfScDev dv; bfsc_dev_alloc(dv, HwF::K, lN);
        bfsc_dev_eq(dv.a, rz.data(), lN);
        uint8_t* d_bits;
        cudaMalloc(&d_bits, (size_t)NCON * N);
        cudaMemcpy(d_bits, bits.data(), (size_t)NCON * N, cudaMemcpyHostToDevice);
        bfsc_bits_kernel<<<(uint32_t)(((size_t)NCON * N + 255) / 256), 256>>>(
            d_bits, dv.a + dv.n, (size_t)NCON * N);
        cudaFree(d_bits);
        bf128_t* d_g;
        cudaMalloc(&d_g, NC * sizeof(bf128_t));
        cudaMemcpy(d_g, g.data(), NC * sizeof(bf128_t), cudaMemcpyHostToDevice);
        HwF cf; cf.g = d_g;
        bf_sumcheck_prove_gpu(dv, cf, tr, pf.sc, zeta);
        cudaFree(d_g);
        bfsc_dev_free(dv);
    } else {
        std::vector<std::vector<bf128_t>> cols(HwF::K);
        bf_eq_table(rz.data(), lN, cols[0]);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < NCON; j++) {
            cols[1 + j].resize(N);
            for (size_t x = 0; x < N; x++)
                cols[1 + j][x] = bits[(size_t)j * N + x] ? bf128_one() : bf128_zero();
        }
        HwF cf; cf.g = g.data();
        auto fn = [](const bf128_t* w, const void* ctx) {
            return (*(const HwF*)ctx)(w);
        };
        bf_sumcheck_prove(lN, HwF::K, cols, HwF::D, fn, &cf, tr, pf.sc, zeta);
    }
    st.sc_ms = bhw_now_ms() - t0;
    // committed-only column evals at zeta (XOR-select over the eq table)
    std::vector<bf128_t> eqz;
    bf_eq_table(zeta.data(), lN, eqz);
    #pragma omp parallel for schedule(static)
    for (int j = NCON; j < NCOLA; j++) {
        bf128_t acc = bf128_zero();
        const uint8_t* col = bits.data() + (size_t)j * N;
        for (size_t x = 0; x < N; x++)
            if (col[x] & 1) acc = bf128_add(acc, eqz[x]);
        pf.xev[j - NCON] = acc;
    }
    tr.absorb("bhw-xev", pf.xev, sizeof pf.xev);
    // ALL column evals at the lookup point rfin_w (binds the DM leaf claim)
    std::vector<bf128_t> eqw;
    bf_eq_table(rfin_w.data(), lN, eqw);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < NCOLA; j++) {
        bf128_t acc = bf128_zero();
        const uint8_t* col = bits.data() + (size_t)j * N;
        for (size_t x = 0; x < N; x++)
            if (col[x] & 1) acc = bf128_add(acc, eqw[x]);
        pf.xev2[j] = acc;
    }
    tr.absorb("bhw-xev2", pf.xev2, sizeof pf.xev2);
    // ---- accumulation gadget (section 21.10): one zerocheck per tree level,
    // then the binding challenges and the supplied slot evals ----
    std::vector<bf128_t> az[5];
    bf128_t asig[5];
    std::vector<bf128_t> atau[5], ataup[5];
    if (aw) {
        double ta = bhw_now_ms();
        const uint8_t* asrc = acc_in ? acc_in : bits.data();
        // one shared device workspace for all five levels (level 1 is largest)
        BfScDev ws; ws.a = nullptr;
        uint8_t* d_sb = nullptr;
        if (gpu) {
            bfsc_dev_reserve(ws, (size_t)bacc::AccF<1>::K << (lN - 1));
            cudaMalloc(&d_sb, (size_t)(bacc::AccF<1>::K - 1) << (lN - 1));
        }
        for (int l = 1; l <= bacc::ACCL; l++) {
            std::vector<bf128_t> g2;
            bacc::acc_gammas(tr, l, g2);
            std::vector<uint8_t> sb;
            bacc::acc_sc_bytes(lN, l, asrc, LALM, LALS, L5BASE,
                               aw->bits.data(), sb);
            BfScDev* w = gpu ? &ws : nullptr;
            switch (l) {
            case 1: bacc::acc_zc<1>(lN, sb, tr, pf.acc.sc[0], az[0], g2, gpu, w, d_sb); break;
            case 2: bacc::acc_zc<2>(lN, sb, tr, pf.acc.sc[1], az[1], g2, gpu, w, d_sb); break;
            case 3: bacc::acc_zc<3>(lN, sb, tr, pf.acc.sc[2], az[2], g2, gpu, w, d_sb); break;
            case 4: bacc::acc_zc<4>(lN, sb, tr, pf.acc.sc[3], az[3], g2, gpu, w, d_sb); break;
            case 5: bacc::acc_zc<5>(lN, sb, tr, pf.acc.sc[4], az[4], g2, gpu, w, d_sb); break;
            }
        }
        if (gpu) { bfsc_dev_free(ws); cudaFree(d_sb); }
        for (int l = 1; l <= bacc::ACCL; l++) {
            asig[l - 1] = bf_chal128(tr);
            ataup[l - 1].resize(l - 1);
            for (auto& x : ataup[l - 1]) x = bf_chal128(tr);
            atau[l - 1].resize(l);
            for (auto& x : atau[l - 1]) x = bf_chal128(tr);
        }
        // supplied slot evals at every binding point
        std::vector<bf128_t> rB1(lN), rA5(lN);
        rB1[0] = asig[0];
        for (int t = 0; t < lN - 1; t++) rB1[1 + t] = az[0][t];
        for (int t = 0; t < lN - 5; t++) rA5[t] = az[4][t];
        for (int t = 0; t < 5; t++) rA5[lN - 5 + t] = atau[4][t];
        pf.acc.evB1.resize(NCOLA);
        bacc::acc_stack_evals(bits.data(), NCOLA, lN, rB1.data(), pf.acc.evB1.data());
        pf.acc.evA5.resize(NCOLA);
        bacc::acc_stack_evals(bits.data(), NCOLA, lN, rA5.data(), pf.acc.evA5.data());
        tr.absorb("bacc-evB1", pf.acc.evB1.data(), NCOLA * sizeof(bf128_t));
        tr.absorb("bacc-evA5", pf.acc.evA5.data(), NCOLA * sizeof(bf128_t));
        for (int l = 1; l <= 4; l++) {
            std::vector<bf128_t> rA(lN);
            for (int t = 0; t < lN - l; t++) rA[t] = az[l - 1][t];
            for (int t = 0; t < l; t++) rA[lN - l + t] = atau[l - 1][t];
            pf.acc.evA[l - 1].resize(bacc::NSTK2);
            bacc::acc_stack_evals(aw->bits.data(), bacc::NSTK2, lN, rA.data(),
                                  pf.acc.evA[l - 1].data());
            tr.absorb("bacc-evA", pf.acc.evA[l - 1].data(),
                      bacc::NSTK2 * sizeof(bf128_t));
        }
        for (int l = 2; l <= 5; l++) {
            std::vector<bf128_t> rB(lN);
            rB[0] = asig[l - 1];
            for (int t = 0; t < lN - l; t++) rB[1 + t] = az[l - 1][t];
            for (int t = 0; t < l - 1; t++) rB[lN - l + 1 + t] = ataup[l - 1][t];
            pf.acc.evB[l - 2].resize(bacc::NSTK2);
            bacc::acc_stack_evals(aw->bits.data(), bacc::NSTK2, lN, rB.data(),
                                  pf.acc.evB[l - 2].data());
            tr.absorb("bacc-evB", pf.acc.evB[l - 2].data(),
                      bacc::NSTK2 * sizeof(bf128_t));
        }
        st.acc_ms += bhw_now_ms() - ta;
    }
    // ---- transition gadget (section 21.11): link + OR-tree + A/B/C
    // zerochecks, then the chain-binding points and supplied evals ----
    int lG = lN - 5, LCH = tw ? tw->LCH : 0;
    std::vector<bf128_t> zlk, zor[5], zA, zB, zC;
    std::vector<bf128_t> tau_tr(5), sig_or(5), zetaH, zsh[16];
    if (tw) {
        double ta = bhw_now_ms();
        BfScDev ws; ws.a = nullptr;
        uint8_t* d_sb = nullptr;
        if (gpu) {
            bfsc_dev_reserve(ws, (size_t)btr::LkF::K << lN);
            cudaMalloc(&d_sb, (size_t)25 << lN);
        }
        std::vector<uint8_t> sb;
        std::vector<bf128_t> g2;
        // link zerocheck (product cube)
        btr::tr_gammas(tr, btr::LkF::NC, g2);
        btr::sc_bytes_link(lN, bits.data(), LEB, LSH, LPR, LTSEL, LPC,
                           tw->bits.data(), sb);
        btr::tr_zc<btr::LkF>(lN, sb, tr, pf.tr.lk, zlk, g2, gpu,
                             gpu ? &ws : nullptr, d_sb);
        // OR-tree levels
        for (int l = 1; l <= 5; l++) {
            btr::tr_gammas(tr, btr::OrF::NC, g2);
            btr::sc_bytes_or(lN, l, bits.data(), LTSEL, LPR, LORT, LORP,
                             tw->bits.data(), sb);
            btr::tr_zc<btr::OrF>(lN - l, sb, tr, pf.tr.orsc[l - 1], zor[l - 1],
                                 g2, gpu, gpu ? &ws : nullptr, d_sb);
        }
        // group-cube zerochecks A, B, C: host-only in BOTH prover modes (the
        // wide functors would blow up ptxas, and the group cube is tiny)
        {
            int ca[btr::NA]; btr::cols_A(ca);
            btr::tr_gammas(tr, btr::AF::NC, g2);
            btr::sc_bytes_cols(lG, ca, btr::NA, tw->bits.data(), sb);
            btr::tr_zc<btr::AF, false>(lG, sb, tr, pf.tr.A, zA, g2, false);
        }
        btr::tr_gammas(tr, btr::BF::NC, g2);
        btr::sc_bytes_B(lN, bits.data(), L5BASE, tw->bits.data(), sb);
        btr::tr_zc<btr::BF, false>(lG, sb, tr, pf.tr.B, zB, g2, false);
        {
            int cc[btr::NCC]; btr::cols_C(cc);
            btr::tr_gammas(tr, btr::CF3::NC, g2);
            btr::sc_bytes_cols(lG, cc, btr::NCC, tw->bits.data(), sb);
            btr::tr_zc<btr::CF3, false>(lG, sb, tr, pf.tr.C, zC, g2, false);
        }
        if (gpu) { bfsc_dev_free(ws); cudaFree(d_sb); }
        // binding challenges
        for (auto& x : tau_tr) x = bf_chal128(tr);
        for (auto& x : sig_or) x = bf_chal128(tr);
        zetaH.resize(lG - LCH);
        for (auto& x : zetaH) x = bf_chal128(tr);
        for (int j = 0; j < LCH; j++) {
            zsh[j].resize(lG - 1 - j);
            for (auto& x : zsh[j]) x = bf_chal128(tr);
        }
        // supplied main-stack evals at the 11 new points
        {
            std::vector<bf128_t> R(lN);
            auto ev = [&](int i) {
                pf.tr.evMain[i].resize(NCOLA);
                bacc::acc_stack_evals(bits.data(), NCOLA, lN, R.data(),
                                      pf.tr.evMain[i].data());
                tr.absorb("btr-evM", pf.tr.evMain[i].data(),
                          NCOLA * sizeof(bf128_t));
            };
            for (int t = 0; t < lG; t++) R[t] = zB[t];
            for (int t = 0; t < 5; t++) R[lG + t] = tau_tr[t];
            ev(0);                                        // RTR
            for (int t = 0; t < lN; t++) R[t] = zlk[t];
            ev(1);                                        // RLK
            for (int l = 1; l <= 4; l++) {                // orA_l
                for (int t = 0; t < lN - l; t++) R[t] = zor[l - 1][t];
                R[lN - l] = bf128_one();
                for (int t = lN - l + 1; t < lN; t++) R[t] = bf128_zero();
                ev(1 + l);
            }
            R[0] = sig_or[0];                             // orB_1
            for (int t = 0; t < lN - 1; t++) R[1 + t] = zor[0][t];
            ev(6);
            for (int l = 2; l <= 5; l++) {                // orB_l
                R[0] = sig_or[l - 1];
                for (int t = 0; t < lN - l; t++) R[1 + t] = zor[l - 1][t];
                R[1 + lN - l] = bf128_one();
                for (int t = lN - l + 2; t < lN; t++) R[t] = bf128_zero();
                ev(5 + l);
            }
        }
        // supplied transition-stack evals
        {
            std::vector<bf128_t> R(lG);
            auto ev = [&](std::vector<bf128_t>& dst) {
                dst.resize(btr::NCOLT);
                bacc::acc_stack_evals(tw->bits.data(), btr::NCOLT, lG, R.data(),
                                      dst.data());
                tr.absorb("btr-evT", dst.data(), btr::NCOLT * sizeof(bf128_t));
            };
            R.assign(zA.begin(), zA.end()); ev(pf.tr.evT[0]);      // TA
            R.assign(zB.begin(), zB.end()); ev(pf.tr.evT[1]);      // TB
            R.assign(zC.begin(), zC.end()); ev(pf.tr.evT[2]);      // TC
            for (int t = 0; t < lG; t++) R[t] = zlk[5 + t];
            ev(pf.tr.evT[3]);                                      // TLK
            R.assign(zor[4].begin(), zor[4].end()); ev(pf.tr.evT[4]); // TOR5
            for (int t = 0; t < LCH; t++) R[t] = bf128_zero();     // THEAD
            for (int t = LCH; t < lG; t++) R[t] = zetaH[t - LCH];
            ev(pf.tr.evT[5]);
            pf.tr.evTS.resize(2 * LCH);
            for (int j = 0; j < LCH; j++) {                        // TS pairs
                for (int t = 0; t < j; t++) R[t] = bf128_zero();
                R[j] = bf128_one();
                for (int t = j + 1; t < lG; t++) R[t] = zsh[j][t - j - 1];
                ev(pf.tr.evTS[2 * j]);
                for (int t = 0; t < j; t++) R[t] = bf128_one();
                R[j] = bf128_zero();
                ev(pf.tr.evTS[2 * j + 1]);
            }
        }
        st.tr_ms += bhw_now_ms() - ta;
    }
    // one stacked MULTI-POINT opening per commitment
    std::vector<bf128_t> rho(LOGSTK), eqsel, R1(p.l), R2(p.l);
    for (auto& x : rho) x = bf_chal128(tr);
    bf_eq_table(rho.data(), LOGSTK, eqsel);
    bf128_t V1 = bf128_zero(), V2 = bf128_zero();
    for (int j = 0; j < NCON; j++)
        V1 = bf128_add(V1, bf128_mul(eqsel[j], pf.sc.finals[1 + j]));
    for (int j = NCON; j < NCOLA; j++)
        V1 = bf128_add(V1, bf128_mul(eqsel[j], pf.xev[j - NCON]));
    for (int j = 0; j < NCOLA; j++)
        V2 = bf128_add(V2, bf128_mul(eqsel[j], pf.xev2[j]));
    for (int t = 0; t < lN; t++) { R1[t] = zeta[t]; R2[t] = rfin_w[t]; }
    for (int t = 0; t < LOGSTK; t++) R1[lN + t] = R2[lN + t] = rho[t];
    t0 = bhw_now_ms();
    if (!aw) {
        bfpcs_open_multi(C, {R1.data(), R2.data()}, {V1, V2}, tr, pf.pcs);
    } else {
        std::vector<bf128_t> RB1(p.l), RA5(p.l);
        RB1[0] = asig[0];
        for (int t = 0; t < lN - 1; t++) RB1[1 + t] = az[0][t];
        for (int t = 0; t < lN - 5; t++) RA5[t] = az[4][t];
        for (int t = 0; t < 5; t++) RA5[lN - 5 + t] = atau[4][t];
        for (int t = 0; t < LOGSTK; t++) RB1[lN + t] = RA5[lN + t] = rho[t];
        bf128_t VB1 = bf128_zero(), VA5 = bf128_zero();
        for (int j = 0; j < NCOLA; j++) {
            VB1 = bf128_add(VB1, bf128_mul(eqsel[j], pf.acc.evB1[j]));
            VA5 = bf128_add(VA5, bf128_mul(eqsel[j], pf.acc.evA5[j]));
        }
        std::vector<std::vector<bf128_t>> RM;
        std::vector<bf128_t> VM;
        RM.push_back(R1); RM.push_back(R2); RM.push_back(RB1); RM.push_back(RA5);
        VM.push_back(V1); VM.push_back(V2); VM.push_back(VB1); VM.push_back(VA5);
        if (tw) {           // the 11 transition points on the main stack
            std::vector<bf128_t> R(p.l);
            for (int t = 0; t < LOGSTK; t++) R[lN + t] = rho[t];
            auto pushm = [&](int i) {
                bf128_t V = bf128_zero();
                for (int j = 0; j < NCOLA; j++)
                    V = bf128_add(V, bf128_mul(eqsel[j], pf.tr.evMain[i][j]));
                RM.push_back(R); VM.push_back(V);
            };
            for (int t = 0; t < lG; t++) R[t] = zB[t];
            for (int t = 0; t < 5; t++) R[lG + t] = tau_tr[t];
            pushm(0);
            for (int t = 0; t < lN; t++) R[t] = zlk[t];
            pushm(1);
            for (int l = 1; l <= 4; l++) {
                for (int t = 0; t < lN - l; t++) R[t] = zor[l - 1][t];
                R[lN - l] = bf128_one();
                for (int t = lN - l + 1; t < lN; t++) R[t] = bf128_zero();
                pushm(1 + l);
            }
            R[0] = sig_or[0];
            for (int t = 0; t < lN - 1; t++) R[1 + t] = zor[0][t];
            pushm(6);
            for (int l = 2; l <= 5; l++) {
                R[0] = sig_or[l - 1];
                for (int t = 0; t < lN - l; t++) R[1 + t] = zor[l - 1][t];
                R[1 + lN - l] = bf128_one();
                for (int t = lN - l + 2; t < lN; t++) R[t] = bf128_zero();
                pushm(5 + l);
            }
        }
        std::vector<const bf128_t*> rmp;
        for (auto& R : RM) rmp.push_back(R.data());
        bfpcs_open_multi(C, rmp, VM, tr, pf.pcs);
        // acc-stack opening: points [A_1, A_2, A_3, A_4, B_2, B_3, B_4, B_5]
        BfPcsParams p2 = bacc::acc_params(lN);
        std::vector<bf128_t> rho2(bacc::LOGSTK2), eqsel2;
        for (auto& x : rho2) x = bf_chal128(tr);
        bf_eq_table(rho2.data(), bacc::LOGSTK2, eqsel2);
        std::vector<std::vector<bf128_t>> RR;
        std::vector<bf128_t> VV;
        for (int l = 1; l <= 4; l++) {
            std::vector<bf128_t> R(p2.l);
            for (int t = 0; t < lN - l; t++) R[t] = az[l - 1][t];
            for (int t = 0; t < l; t++) R[lN - l + t] = atau[l - 1][t];
            for (int t = 0; t < bacc::LOGSTK2; t++) R[lN + t] = rho2[t];
            bf128_t V = bf128_zero();
            for (int s = 0; s < bacc::NSTK2; s++)
                V = bf128_add(V, bf128_mul(eqsel2[s], pf.acc.evA[l - 1][s]));
            RR.push_back(std::move(R)); VV.push_back(V);
        }
        for (int l = 2; l <= 5; l++) {
            std::vector<bf128_t> R(p2.l);
            R[0] = asig[l - 1];
            for (int t = 0; t < lN - l; t++) R[1 + t] = az[l - 1][t];
            for (int t = 0; t < l - 1; t++) R[lN - l + 1 + t] = ataup[l - 1][t];
            for (int t = 0; t < bacc::LOGSTK2; t++) R[lN + t] = rho2[t];
            bf128_t V = bf128_zero();
            for (int s = 0; s < bacc::NSTK2; s++)
                V = bf128_add(V, bf128_mul(eqsel2[s], pf.acc.evB[l - 2][s]));
            RR.push_back(std::move(R)); VV.push_back(V);
        }
        std::vector<const bf128_t*> rp;
        for (auto& R : RR) rp.push_back(R.data());
        bfpcs_open_multi(C2, rp, VV, tr, pf.acc.pcs2);
        // transition-stack opening: [TA TB TC TLK TOR5 THEAD TS_I/O pairs]
        if (tw) {
            BfPcsParams p3 = btr::tr_params(lN);
            std::vector<bf128_t> rho3(btr::LOGSTK3), eqsel3;
            for (auto& x : rho3) x = bf_chal128(tr);
            bf_eq_table(rho3.data(), btr::LOGSTK3, eqsel3);
            std::vector<std::vector<bf128_t>> RT;
            std::vector<bf128_t> VT;
            std::vector<bf128_t> R(p3.l);
            for (int t = 0; t < btr::LOGSTK3; t++) R[lG + t] = rho3[t];
            auto pusht = [&](const std::vector<bf128_t>& ev) {
                bf128_t V = bf128_zero();
                for (int j = 0; j < btr::NCOLT; j++)
                    V = bf128_add(V, bf128_mul(eqsel3[j], ev[j]));
                RT.push_back(R); VT.push_back(V);
            };
            for (int t = 0; t < lG; t++) R[t] = zA[t];
            pusht(pf.tr.evT[0]);
            for (int t = 0; t < lG; t++) R[t] = zB[t];
            pusht(pf.tr.evT[1]);
            for (int t = 0; t < lG; t++) R[t] = zC[t];
            pusht(pf.tr.evT[2]);
            for (int t = 0; t < lG; t++) R[t] = zlk[5 + t];
            pusht(pf.tr.evT[3]);
            for (int t = 0; t < lG; t++) R[t] = zor[4][t];
            pusht(pf.tr.evT[4]);
            for (int t = 0; t < LCH; t++) R[t] = bf128_zero();
            for (int t = LCH; t < lG; t++) R[t] = zetaH[t - LCH];
            pusht(pf.tr.evT[5]);
            for (int j = 0; j < LCH; j++) {
                for (int t = 0; t < j; t++) R[t] = bf128_zero();
                R[j] = bf128_one();
                for (int t = j + 1; t < lG; t++) R[t] = zsh[j][t - j - 1];
                pusht(pf.tr.evTS[2 * j]);
                for (int t = 0; t < j; t++) R[t] = bf128_one();
                R[j] = bf128_zero();
                pusht(pf.tr.evTS[2 * j + 1]);
            }
            std::vector<const bf128_t*> rtp;
            for (auto& RR2 : RT) rtp.push_back(RR2.data());
            bfpcs_open_multi(C3, rtp, VT, tr, pf.tr.pcs3);
        }
    }
    st.open_ms = bhw_now_ms() - t0;
}

static inline bool bhw_verify(const BhwProof& pf, BhwStats* st = nullptr) {
    double t0 = bhw_now_ms();
    int lN = pf.lN;
    BfPcsParams p = bhw_params(lN);
    fs::Transcript tr("binius-hawkeye");
    tr.absorb("stmt-bhw", &lN, sizeof lN);
    tr.absorb("stmt-acc", &pf.acc.on, sizeof pf.acc.on);
    tr.absorb("stmt-tr", &pf.tr.on, sizeof pf.tr.on);
    if (pf.tr.on) tr.absorb("stmt-tr-ch", &pf.tr.CH, sizeof pf.tr.CH);
    if (pf.tr.on && !pf.acc.on) return false;   // transition requires the trees
    tr.absorb("bfpcs-root", pf.root, 32);
    if (pf.acc.on) tr.absorb("bfpcs-root", pf.acc.root2, 32);
    if (pf.tr.on) tr.absorb("bfpcs-root", pf.tr.root3, 32);
    // DM lookup chain; its leaf claim is bound against xev2 below
    std::vector<bf128_t> rfin_w, bk; bf128_t cw, alpha;
    if (!bflu::bflu_verify(tr, lN, 16, DMJ, bhw_dm_table_bits().data(), pf.lu,
                           rfin_w, cw, alpha, bk)) return false;
    std::vector<bf128_t> rz(lN);
    for (auto& x : rz) x = bf_chal128(tr);
    std::vector<bf128_t> g;
    bhw_gammas(tr, g);
    if ((int)pf.sc.finals.size() != HwF::K) return false;
    std::vector<bf128_t> zeta; bf128_t E;
    if (!bf_sumcheck_verify(pf.sc, bf128_zero(), tr, zeta, &E)) return false;
    HwF cf; cf.g = g.data();
    if (!bf128_eq(E, cf(pf.sc.finals.data()))) return false;
    bf128_t eqv = bf128_one();
    for (int t = 0; t < lN; t++)
        eqv = bf128_mul(eqv, bf128_add(bf128_one(), bf128_add(rz[t], zeta[t])));
    if (!bf128_eq(pf.sc.finals[0], eqv)) return false;
    tr.absorb("bhw-xev", pf.xev, sizeof pf.xev);
    tr.absorb("bhw-xev2", pf.xev2, sizeof pf.xev2);
    // lookup leaf binding: cw == alpha + sum_k bk[k] * tuple-col-eval at rfin_w
    {
        int dmc[DMJ]; bhw_dm_cols(dmc);
        bf128_t exp = alpha;
        for (int k = 0; k < DMJ; k++)
            exp = bf128_add(exp, bf128_mul(bk[k], pf.xev2[dmc[k]]));
        if (!bf128_eq(cw, exp)) return false;
    }
    // ---- accumulation levels: replay + finals-vs-integrand, then bindings ----
    std::vector<bf128_t> az[5];
    bf128_t asig[5];
    std::vector<bf128_t> atau[5], ataup[5];
    if (pf.acc.on) {
        if (lN < bacc::ACCL + 1) return false;
        for (int l = 1; l <= bacc::ACCL; l++) {
            std::vector<bf128_t> g2;
            bacc::acc_gammas(tr, l, g2);
            bool ok = false;
            switch (l) {
            case 1: ok = bacc::acc_zc_verify<1>(lN, pf.acc.sc[0], tr, az[0], g2); break;
            case 2: ok = bacc::acc_zc_verify<2>(lN, pf.acc.sc[1], tr, az[1], g2); break;
            case 3: ok = bacc::acc_zc_verify<3>(lN, pf.acc.sc[2], tr, az[2], g2); break;
            case 4: ok = bacc::acc_zc_verify<4>(lN, pf.acc.sc[3], tr, az[3], g2); break;
            case 5: ok = bacc::acc_zc_verify<5>(lN, pf.acc.sc[4], tr, az[4], g2); break;
            }
            if (!ok) return false;
        }
        if ((int)pf.acc.evB1.size() != NCOLA || (int)pf.acc.evA5.size() != NCOLA)
            return false;
        for (int i = 0; i < 4; i++)
            if ((int)pf.acc.evA[i].size() != bacc::NSTK2 ||
                (int)pf.acc.evB[i].size() != bacc::NSTK2) return false;
        for (int l = 1; l <= bacc::ACCL; l++) {
            asig[l - 1] = bf_chal128(tr);
            ataup[l - 1].resize(l - 1);
            for (auto& x : ataup[l - 1]) x = bf_chal128(tr);
            atau[l - 1].resize(l);
            for (auto& x : atau[l - 1]) x = bf_chal128(tr);
        }
        tr.absorb("bacc-evB1", pf.acc.evB1.data(), NCOLA * sizeof(bf128_t));
        tr.absorb("bacc-evA5", pf.acc.evA5.data(), NCOLA * sizeof(bf128_t));
        for (int l = 1; l <= 4; l++)
            tr.absorb("bacc-evA", pf.acc.evA[l - 1].data(),
                      bacc::NSTK2 * sizeof(bf128_t));
        for (int l = 2; l <= 5; l++)
            tr.absorb("bacc-evB", pf.acc.evB[l - 2].data(),
                      bacc::NSTK2 * sizeof(bf128_t));
    }
    // ---- transition gadget: replay the 9 zerochecks, draw the binding
    // challenges, absorb the supplied evals, check the chain equalities ----
    int lG = lN - 5, LCH = 0;
    std::vector<bf128_t> zlk, zor[5], zA, zB, zC;
    std::vector<bf128_t> tau_tr(5), sig_or(5), zetaH, zsh[16];
    if (pf.tr.on) {
        int CH = pf.tr.CH;
        if (CH < 1 || (CH & (CH - 1))) return false;
        while ((1 << LCH) < CH) LCH++;
        if (lG < LCH) return false;
        std::vector<bf128_t> g2;
        btr::tr_gammas(tr, btr::LkF::NC, g2);
        if (!btr::tr_zc_verify<btr::LkF>(lN, pf.tr.lk, tr, zlk, g2)) return false;
        for (int l = 1; l <= 5; l++) {
            btr::tr_gammas(tr, btr::OrF::NC, g2);
            if (!btr::tr_zc_verify<btr::OrF>(lN - l, pf.tr.orsc[l - 1], tr,
                                             zor[l - 1], g2)) return false;
        }
        btr::tr_gammas(tr, btr::AF::NC, g2);
        if (!btr::tr_zc_verify<btr::AF>(lG, pf.tr.A, tr, zA, g2)) return false;
        btr::tr_gammas(tr, btr::BF::NC, g2);
        if (!btr::tr_zc_verify<btr::BF>(lG, pf.tr.B, tr, zB, g2)) return false;
        btr::tr_gammas(tr, btr::CF3::NC, g2);
        if (!btr::tr_zc_verify<btr::CF3>(lG, pf.tr.C, tr, zC, g2)) return false;
        for (auto& x : tau_tr) x = bf_chal128(tr);
        for (auto& x : sig_or) x = bf_chal128(tr);
        zetaH.resize(lG - LCH);
        for (auto& x : zetaH) x = bf_chal128(tr);
        for (int j = 0; j < LCH; j++) {
            zsh[j].resize(lG - 1 - j);
            for (auto& x : zsh[j]) x = bf_chal128(tr);
        }
        for (int i = 0; i < 11; i++)
            if ((int)pf.tr.evMain[i].size() != NCOLA) return false;
        for (int i = 0; i < 6; i++)
            if ((int)pf.tr.evT[i].size() != btr::NCOLT) return false;
        if ((int)pf.tr.evTS.size() != 2 * LCH) return false;
        for (auto& v : pf.tr.evTS)
            if ((int)v.size() != btr::NCOLT) return false;
        for (int i = 0; i < 11; i++)
            tr.absorb("btr-evM", pf.tr.evMain[i].data(), NCOLA * sizeof(bf128_t));
        for (int i = 0; i < 6; i++)
            tr.absorb("btr-evT", pf.tr.evT[i].data(), btr::NCOLT * sizeof(bf128_t));
        for (auto& v : pf.tr.evTS)
            tr.absorb("btr-evT", v.data(), btr::NCOLT * sizeof(bf128_t));
        // the CHAIN WELD: in-state at (10^j, zeta) == out-state at (01^j, zeta)
        for (int j = 0; j < LCH; j++)
            for (int c = 0; c < btr::NST; c++)
                if (!bf128_eq(pf.tr.evTS[2 * j][c],
                              pf.tr.evTS[2 * j + 1][btr::NST + c])) return false;
    }
    std::vector<bf128_t> rho(LOGSTK), eqsel, R1(p.l), R2(p.l);
    for (auto& x : rho) x = bf_chal128(tr);
    bf_eq_table(rho.data(), LOGSTK, eqsel);
    bf128_t V1 = bf128_zero(), V2 = bf128_zero();
    for (int j = 0; j < NCON; j++)
        V1 = bf128_add(V1, bf128_mul(eqsel[j], pf.sc.finals[1 + j]));
    for (int j = NCON; j < NCOLA; j++)
        V1 = bf128_add(V1, bf128_mul(eqsel[j], pf.xev[j - NCON]));
    for (int j = 0; j < NCOLA; j++)
        V2 = bf128_add(V2, bf128_mul(eqsel[j], pf.xev2[j]));
    for (int t = 0; t < lN; t++) { R1[t] = zeta[t]; R2[t] = rfin_w[t]; }
    for (int t = 0; t < LOGSTK; t++) R1[lN + t] = R2[lN + t] = rho[t];
    bool ok;
    if (!pf.acc.on) {
        ok = bfpcs_verify_multi(p, pf.root, {R1.data(), R2.data()}, {V1, V2},
                                tr, pf.pcs);
    } else {
        // derived slot evals OVERRIDE the supplied ones wherever a binding
        // point's slots are fully determined by sumcheck finals -- this is
        // what forces the finals to be true column evals of the commitments
        std::vector<bf128_t> RB1(p.l), RA5(p.l);
        RB1[0] = asig[0];
        for (int t = 0; t < lN - 1; t++) RB1[1 + t] = az[0][t];
        for (int t = 0; t < lN - 5; t++) RA5[t] = az[4][t];
        for (int t = 0; t < 5; t++) RA5[lN - 5 + t] = atau[4][t];
        for (int t = 0; t < LOGSTK; t++) RB1[lN + t] = RA5[lN + t] = rho[t];
        bf128_t derB1[NCOLA], derA5[NCOLA];
        uint8_t hvB1[NCOLA] = {0}, hvA5[NCOLA] = {0};
        {   // B_1: the level-1 input finals against the almag/alsg columns
            const bf128_t* f1 = pf.acc.sc[0].finals.data();
            for (int k = 0; k < 16; k++) {
                int slot = (k < 15) ? LALM + k : LALS;
                bf128_t fe = f1[1 + k], fo = f1[1 + 16 + k];
                derB1[slot] = bf128_add(fe, bf128_mul(asig[0], bf128_add(fe, fo)));
                hvB1[slot] = 1;
            }
        }
        {   // A_5: the level-5 committed finals against main slots 110..113
            std::vector<bf128_t> eqt;
            bf_eq_table(atau[4].data(), 5, eqt);
            bacc::acc_derive_A(5, L5BASE, pf.acc.sc[4].finals.data(),
                               bacc::AccF<5>::NI, eqt.data(), derA5, hvA5);
        }
        bf128_t VB1 = bf128_zero(), VA5 = bf128_zero();
        for (int j = 0; j < NCOLA; j++) {
            VB1 = bf128_add(VB1, bf128_mul(eqsel[j], hvB1[j] ? derB1[j] : pf.acc.evB1[j]));
            VA5 = bf128_add(VA5, bf128_mul(eqsel[j], hvA5[j] ? derA5[j] : pf.acc.evA5[j]));
        }
        std::vector<std::vector<bf128_t>> RM;
        std::vector<bf128_t> VM;
        RM.push_back(R1); RM.push_back(R2); RM.push_back(RB1); RM.push_back(RA5);
        VM.push_back(V1); VM.push_back(V2); VM.push_back(VB1); VM.push_back(VA5);
        if (pf.tr.on) {
            std::vector<bf128_t> R(p.l);
            for (int t = 0; t < LOGSTK; t++) R[lN + t] = rho[t];
            bf128_t der[NCOLA];
            uint8_t hv[NCOLA];
            auto pushm = [&](int i) {
                bf128_t V = bf128_zero();
                for (int j = 0; j < NCOLA; j++)
                    V = bf128_add(V, bf128_mul(eqsel[j],
                                   hv[j] ? der[j] : pf.tr.evMain[i][j]));
                RM.push_back(R); VM.push_back(V);
            };
            // RTR: level-5 tree sums derive from ZC-B's P/N input finals
            memset(hv, 0, sizeof hv);
            for (int t = 0; t < lG; t++) R[t] = zB[t];
            for (int t = 0; t < 5; t++) R[lG + t] = tau_tr[t];
            {
                std::vector<bf128_t> eqt;
                bf_eq_table(tau_tr.data(), 5, eqt);
                for (int col = 0; col < 40; col++) {
                    int slot, t;
                    bacc::acc_slot(5, 0, col, L5BASE, slot, t);
                    if (!hv[slot]) { hv[slot] = 1; der[slot] = bf128_zero(); }
                    der[slot] = bf128_add(der[slot],
                                bf128_mul(eqt[t], pf.tr.B.finals[1 + col]));
                }
            }
            pushm(0);
            // RLK: the link zerocheck's main columns derive from its finals
            memset(hv, 0, sizeof hv);
            for (int t = 0; t < lN; t++) R[t] = zlk[t];
            for (int j = 0; j < 5; j++) { hv[LEB + j] = 1; der[LEB + j] = pf.tr.lk.finals[1 + j]; }
            for (int j = 0; j < 6; j++) { hv[LSH + j] = 1; der[LSH + j] = pf.tr.lk.finals[1 + 5 + j]; }
            hv[LPR] = 1; der[LPR] = pf.tr.lk.finals[1 + 11];
            hv[LTSEL] = 1; der[LTSEL] = pf.tr.lk.finals[1 + 12];
            for (int j = 0; j < 6; j++) { hv[LPC + j] = 1; der[LPC + j] = pf.tr.lk.finals[1 + 13 + j]; }
            pushm(1);
            // orA_l: the level's committed OR columns
            for (int l = 1; l <= 4; l++) {
                memset(hv, 0, sizeof hv);
                for (int t = 0; t < lN - l; t++) R[t] = zor[l - 1][t];
                R[lN - l] = bf128_one();
                for (int t = lN - l + 1; t < lN; t++) R[t] = bf128_zero();
                hv[LORT] = 1; der[LORT] = pf.tr.orsc[l - 1].finals[1 + 4];
                hv[LORP] = 1; der[LORP] = pf.tr.orsc[l - 1].finals[1 + 5];
                pushm(1 + l);
            }
            // orB_l: the level's INPUT finals against level l-1 (l=1: t / pr)
            for (int l = 1; l <= 5; l++) {
                memset(hv, 0, sizeof hv);
                R[0] = sig_or[l - 1];
                for (int t = 0; t < lN - l; t++) R[1 + t] = zor[l - 1][t];
                if (l >= 2) {
                    R[1 + lN - l] = bf128_one();
                    for (int t = lN - l + 2; t < lN; t++) R[t] = bf128_zero();
                }
                const bf128_t* f = pf.tr.orsc[l - 1].finals.data();
                auto comb = [&](bf128_t fe, bf128_t fo) {
                    return bf128_add(fe, bf128_mul(sig_or[l - 1], bf128_add(fe, fo)));
                };
                int st_ = (l == 1) ? LTSEL : LORT, sp_ = (l == 1) ? LPR : LORP;
                hv[st_] = 1; der[st_] = comb(f[1 + 0], f[1 + 1]);
                hv[sp_] = 1; der[sp_] = comb(f[1 + 2], f[1 + 3]);
                pushm(l == 1 ? 6 : 5 + l);
            }
        }
        std::vector<const bf128_t*> rmp;
        for (auto& RX : RM) rmp.push_back(RX.data());
        ok = bfpcs_verify_multi(p, pf.root, rmp, VM, tr, pf.pcs);
        if (!ok) return false;
        BfPcsParams p2 = bacc::acc_params(lN);
        std::vector<bf128_t> rho2(bacc::LOGSTK2), eqsel2;
        for (auto& x : rho2) x = bf_chal128(tr);
        bf_eq_table(rho2.data(), bacc::LOGSTK2, eqsel2);
        std::vector<std::vector<bf128_t>> RR;
        std::vector<bf128_t> VV;
        for (int l = 1; l <= 4; l++) {       // A_l: all level-l slots derived
            std::vector<bf128_t> R(p2.l);
            for (int t = 0; t < lN - l; t++) R[t] = az[l - 1][t];
            for (int t = 0; t < l; t++) R[lN - l + t] = atau[l - 1][t];
            for (int t = 0; t < bacc::LOGSTK2; t++) R[lN + t] = rho2[t];
            bf128_t der[bacc::NSTK2];
            uint8_t hv[bacc::NSTK2] = {0};
            std::vector<bf128_t> eqt;
            bf_eq_table(atau[l - 1].data(), l, eqt);
            int ni = (l == 1) ? bacc::AccF<1>::NI : 4 * bacc::A_WIN[l];
            bacc::acc_derive_A(l, L5BASE, pf.acc.sc[l - 1].finals.data(), ni,
                               eqt.data(), der, hv);
            bf128_t V = bf128_zero();
            for (int s = 0; s < bacc::NSTK2; s++)
                V = bf128_add(V, bf128_mul(eqsel2[s],
                                           hv[s] ? der[s] : pf.acc.evA[l - 1][s]));
            RR.push_back(std::move(R)); VV.push_back(V);
        }
        for (int l = 2; l <= 5; l++) {       // B_l: level-(l-1) out slots derived
            std::vector<bf128_t> R(p2.l);
            R[0] = asig[l - 1];
            for (int t = 0; t < lN - l; t++) R[1 + t] = az[l - 1][t];
            for (int t = 0; t < l - 1; t++) R[lN - l + 1 + t] = ataup[l - 1][t];
            for (int t = 0; t < bacc::LOGSTK2; t++) R[lN + t] = rho2[t];
            bf128_t der[bacc::NSTK2];
            uint8_t hv[bacc::NSTK2] = {0};
            std::vector<bf128_t> eqt;
            bf_eq_table(ataup[l - 1].data(), l - 1, eqt);
            bacc::acc_derive_B(l, L5BASE, pf.acc.sc[l - 1].finals.data(),
                               asig[l - 1], eqt.data(), der, hv);
            bf128_t V = bf128_zero();
            for (int s = 0; s < bacc::NSTK2; s++)
                V = bf128_add(V, bf128_mul(eqsel2[s],
                                           hv[s] ? der[s] : pf.acc.evB[l - 2][s]));
            RR.push_back(std::move(R)); VV.push_back(V);
        }
        std::vector<const bf128_t*> rp;
        for (auto& R : RR) rp.push_back(R.data());
        ok = bfpcs_verify_multi(p2, pf.acc.root2, rp, VV, tr, pf.acc.pcs2);
        if (!ok) return false;
        // transition-stack opening with derived overrides per point
        if (pf.tr.on) {
            BfPcsParams p3 = btr::tr_params(lN);
            std::vector<bf128_t> rho3(btr::LOGSTK3), eqsel3;
            for (auto& x : rho3) x = bf_chal128(tr);
            bf_eq_table(rho3.data(), btr::LOGSTK3, eqsel3);
            std::vector<std::vector<bf128_t>> RT;
            std::vector<bf128_t> VT;
            std::vector<bf128_t> R(p3.l);
            for (int t = 0; t < btr::LOGSTK3; t++) R[lG + t] = rho3[t];
            bf128_t der[btr::NCOLT];
            uint8_t hv[btr::NCOLT];
            auto pusht = [&](const std::vector<bf128_t>& ev) {
                bf128_t V = bf128_zero();
                for (int j = 0; j < btr::NCOLT; j++)
                    V = bf128_add(V, bf128_mul(eqsel3[j], hv[j] ? der[j] : ev[j]));
                RT.push_back(R); VT.push_back(V);
            };
            {   // TA: every ZC-A column derives from its finals
                memset(hv, 0, sizeof hv);
                int ca[btr::NA]; btr::cols_A(ca);
                for (int i = 0; i < btr::NA; i++) {
                    hv[ca[i]] = 1; der[ca[i]] = pf.tr.A.finals[1 + i];
                }
                for (int t = 0; t < lG; t++) R[t] = zA[t];
                pusht(pf.tr.evT[0]);
            }
            {   // TB: ZC-B's transition-committed columns
                memset(hv, 0, sizeof hv);
                int cb[btr::NBT]; btr::cols_B(cb);
                for (int i = 0; i < btr::NBT; i++) {
                    hv[cb[i]] = 1; der[cb[i]] = pf.tr.B.finals[1 + 40 + i];
                }
                for (int t = 0; t < lG; t++) R[t] = zB[t];
                pusht(pf.tr.evT[1]);
            }
            {   // TC: every ZC-C column
                memset(hv, 0, sizeof hv);
                int cc[btr::NCC]; btr::cols_C(cc);
                for (int i = 0; i < btr::NCC; i++) {
                    hv[cc[i]] = 1; der[cc[i]] = pf.tr.C.finals[1 + i];
                }
                for (int t = 0; t < lG; t++) R[t] = zC[t];
                pusht(pf.tr.evT[2]);
            }
            {   // TLK: the lifted ME12 input of the link zerocheck
                memset(hv, 0, sizeof hv);
                for (int j = 0; j < 6; j++) {
                    hv[btr::TME12 + j] = 1;
                    der[btr::TME12 + j] = pf.tr.lk.finals[1 + 19 + j];
                }
                for (int t = 0; t < lG; t++) R[t] = zlk[5 + t];
                pusht(pf.tr.evT[3]);
            }
            {   // TOR5: og / pg from the level-5 OR zerocheck
                memset(hv, 0, sizeof hv);
                hv[btr::TOG] = 1; der[btr::TOG] = pf.tr.orsc[4].finals[1 + 4];
                hv[btr::TPG] = 1; der[btr::TPG] = pf.tr.orsc[4].finals[1 + 5];
                for (int t = 0; t < lG; t++) R[t] = zor[4][t];
                pusht(pf.tr.evT[4]);
            }
            {   // THEAD: chain heads carry the zero in-state
                memset(hv, 0, sizeof hv);
                for (int c = 0; c < btr::NST; c++) {
                    hv[c] = 1; der[c] = bf128_zero();
                }
                for (int t = 0; t < LCH; t++) R[t] = bf128_zero();
                for (int t = LCH; t < lG; t++) R[t] = zetaH[t - LCH];
                pusht(pf.tr.evT[5]);
            }
            memset(hv, 0, sizeof hv);   // TS pairs: equality checked above
            for (int j = 0; j < LCH; j++) {
                for (int t = 0; t < j; t++) R[t] = bf128_zero();
                R[j] = bf128_one();
                for (int t = j + 1; t < lG; t++) R[t] = zsh[j][t - j - 1];
                pusht(pf.tr.evTS[2 * j]);
                for (int t = 0; t < j; t++) R[t] = bf128_one();
                R[j] = bf128_zero();
                pusht(pf.tr.evTS[2 * j + 1]);
            }
            std::vector<const bf128_t*> rtp;
            for (auto& RX : RT) rtp.push_back(RX.data());
            ok = bfpcs_verify_multi(p3, pf.tr.root3, rtp, VT, tr, pf.tr.pcs3);
        }
    }
    if (st) st->verify_ms = bhw_now_ms() - t0;
    return ok;
}

} // namespace bhw
