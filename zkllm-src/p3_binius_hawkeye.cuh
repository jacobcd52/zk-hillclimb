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
    NCOLA = 114,            // slots covered by the xev/xev2 bindings
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
    size_t bytes() const {
        return 32 + lu.bytes() + sc.rounds.size() * 16 + sc.finals.size() * 16 +
               sizeof(xev) + sizeof(xev2) + pcs.bytes() + acc.bytes();
    }
};
struct BhwStats { double commit_ms = 0, lu_ms = 0, sc_ms = 0, open_ms = 0,
                  verify_ms = 0, acc_ms = 0; size_t committed = 0; };

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
                             const uint8_t* acc_in = nullptr) {
    size_t N = (size_t)1 << lN;
    pf.lN = lN;
    pf.acc.on = aw ? 1 : 0;
    BfPcsParams p = bhw_params(lN);
    fs::Transcript tr("binius-hawkeye");
    tr.absorb("stmt-bhw", &lN, sizeof lN);
    tr.absorb("stmt-acc", &pf.acc.on, sizeof pf.acc.on);
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
        bfpcs_open_multi(C, {R1.data(), R2.data(), RB1.data(), RA5.data()},
                         {V1, V2, VB1, VA5}, tr, pf.pcs);
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
    tr.absorb("bfpcs-root", pf.root, 32);
    if (pf.acc.on) tr.absorb("bfpcs-root", pf.acc.root2, 32);
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
        ok = bfpcs_verify_multi(p, pf.root,
                                {R1.data(), R2.data(), RB1.data(), RA5.data()},
                                {V1, V2, VB1, VA5}, tr, pf.pcs);
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
    }
    if (st) st->verify_ms = bhw_now_ms() - t0;
    return ok;
}

} // namespace bhw
