// Binius migration step (design doc section 21.11): the GROUP TRANSITION
// gadget -- the rest of the Hawkeye accumulator loop, proven over the binary
// tower on top of the per-product gadget (21.8), the DM lookup (21.9) and the
// carry-save adder trees (21.10).  With this gadget the Binius side proves
// the COMPLETE Hawkeye matmul semantics per group:
//
//   max_exp   MEZ = max(acc_exp_eff, prod_exp over present) - ZE, welded to
//             the committed per-product shifts by the LINK zerocheck
//             (pr * (eb + sh - ME12) = 0 as a bit-adder, ME12 = MEZ - 127)
//             plus TIGHTNESS (some present product achieves the max, or the
//             accumulator does): per-product selector t with t->(pr, sh=0),
//             OR-trees over t (og) and pr (pg), and og OR tacc = 1 with
//             tacc -> d = 0.
//   realign   d = MEZ - aeI (bit-adder, d >= 0 structural = acc dominance),
//             one-hot ha over min(d,14), aligned acc am = nsI >> d (shift-
//             select), sign-split amP/amN by the state sign sgI.
//   reconcile SP = P + amP, SN = N + amN (bit-adders on the committed level-5
//             tree sums), then SIGNED RECONCILIATION cmag = |SP - SN| with
//             sign csg as a mux-operand bit-adder (lo + cmag = hi).
//   normalize width one-hot w with the pow2 sandwich (selected top bit of
//             cmag = 1, all bits >= width = 0), nsO = cmag shifted to 14
//             bits (truncating, shift-select), out exponent
//             aeO = MEZ + width - 14 (two bit-adders), zero path via w_0.
//   chain     state I=(sgI,aeI,nsI) / O=(sgO,aeO,nsO) committed per group;
//             groups are CHAIN-CONTIGUOUS (group = chain*CH + t, CH a power
//             of two, all-absent padding groups are the identity transition)
//             so I_g = O_{g-1} needs NO sumcheck: the borrow decomposition
//             of "t-1" gives LCH restriction-point PAIRS (I at low bits
//             10^j vs O at low bits 01^j, shared random tail) checked as
//             supplied-eval equalities in the transition-stack multi-point
//             opening, plus ONE head point where the I slots derive to zero.
//
// All new witness lives in a THIRD stack (rows = the group cube, NSTK3=512
// slots, ~317 used) except the per-product link carries pc[6], the tightness
// selector t and the two OR-trees (heap-packed levels 1..4), which go in the
// main stack's free slots.  Three group-cube zerochecks (A: align/max_exp,
// B: SP/SN adders, C: reconcile+normalize) + the link zerocheck (product
// cube) + 5 tiny OR-tree zerochecks; every consumed column is welded to a
// commitment through derived finals at the existing-style binding points.
// Teeth: p3_binius_trans_test.cu.
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_binius_acc.cuh"

namespace btr {

static const int ZE = -139;                      // the Hawkeye zero exponent

// ---- transition-stack layout (rows = group cube) ----
enum {
    TSGI = 0,  TAEI = 1,  TNSI = 9,              // in-state:  sg, ae[8], ns[14]
    TSGO = 23, TAEO = 24, TNSO = 32,             // out-state: sg, ae[8], ns[14]
    TMEZ = 46,                                   // 8   max_exp - ZE
    TME12 = 54,                                  // 6   max_exp + 12 (iff pg)
    TMC = 60,                                    // 8   carries ME12 + 127 = MEZ
    TD = 68,                                     // 8   d = MEZ - aeI
    TDC = 76,                                    // 8   carries aeI + d = MEZ
    THI = 84, TT3 = 85,                          // d>=16 OR-helper, d3*d2*d1
    THA = 86,                                    // 15  one-hot min(d,14)
    TAM = 101,                                   // 14  am = nsI >> d
    TAMP = 115, TAMN = 129,                      // 14+14 sign-split am
    TSP = 143, TSPC = 164,                       // 21+20 SP = P + amP
    TSN = 184, TSNC = 205,                       // 21+20 SN = N + amN
    TCSG = 225,                                  // sign of SP - SN
    TCM = 226, TCC = 247,                        // 21+21 cmag = |SP-SN|
    TW = 268,                                    // 22  width one-hot 0..21
    TX = 290, TN1 = 298, TN2 = 306,              // X = MEZ+width, its carries,
                                                 // carries aeO + 14 = X
    TTAC = 314, TOG = 315, TPG = 316,            // tacc, OR(t), OR(pr)
    NCOLT = 317,
    LOGSTK3 = 9, NSTK3 = 512
};
static const int NST = 23;                       // state columns per direction

// ---- functor column lists ----
static const int NA = 123;                       // ZC-A committed columns
static inline void cols_A(int* c) {
    int n = 0;
    c[n++] = TSGI;
    for (int j = 0; j < 8; j++)  c[n++] = TAEI + j;
    for (int j = 0; j < 14; j++) c[n++] = TNSI + j;
    for (int j = 0; j < 8; j++)  c[n++] = TMEZ + j;
    for (int j = 0; j < 6; j++)  c[n++] = TME12 + j;
    for (int j = 0; j < 8; j++)  c[n++] = TMC + j;
    for (int j = 0; j < 8; j++)  c[n++] = TD + j;
    for (int j = 0; j < 8; j++)  c[n++] = TDC + j;
    c[n++] = THI; c[n++] = TT3;
    for (int j = 0; j < 15; j++) c[n++] = THA + j;
    for (int j = 0; j < 14; j++) c[n++] = TAM + j;
    for (int j = 0; j < 14; j++) c[n++] = TAMP + j;
    for (int j = 0; j < 14; j++) c[n++] = TAMN + j;
    c[n++] = TTAC; c[n++] = TOG; c[n++] = TPG;   // n == NA
}
static const int NBT = 110;                      // ZC-B trans-committed columns
static inline void cols_B(int* c) {              // (P/N inputs come from main)
    int n = 0;
    for (int j = 0; j < 14; j++) c[n++] = TAMP + j;
    for (int j = 0; j < 14; j++) c[n++] = TAMN + j;
    for (int j = 0; j < 21; j++) c[n++] = TSP + j;
    for (int j = 0; j < 20; j++) c[n++] = TSPC + j;
    for (int j = 0; j < 21; j++) c[n++] = TSN + j;
    for (int j = 0; j < 20; j++) c[n++] = TSNC + j;
}
static const int NCC = 162;                      // ZC-C committed columns
static inline void cols_C(int* c) {
    int n = 0;
    for (int j = 0; j < 21; j++) c[n++] = TSP + j;
    for (int j = 0; j < 21; j++) c[n++] = TSN + j;
    c[n++] = TCSG;
    for (int j = 0; j < 21; j++) c[n++] = TCM + j;
    for (int j = 0; j < 21; j++) c[n++] = TCC + j;
    for (int j = 0; j < 22; j++) c[n++] = TW + j;
    for (int j = 0; j < 8; j++)  c[n++] = TX + j;
    for (int j = 0; j < 8; j++)  c[n++] = TN1 + j;
    for (int j = 0; j < 8; j++)  c[n++] = TN2 + j;
    for (int j = 0; j < 8; j++)  c[n++] = TMEZ + j;
    c[n++] = TSGO;
    for (int j = 0; j < 8; j++)  c[n++] = TAEO + j;
    for (int j = 0; j < 14; j++) c[n++] = TNSO + j;
}

// ---- ZC-A: align/max_exp.  w = [eq | cols_A] ----
struct AF {
    static constexpr int K = 1 + NA, D = 5;
    static constexpr int NC = 305;
    const bf128_t* g;
    __host__ __device__ bf128_t operator()(const bf128_t* w) const {
        const bf128_t one = bf128_one();
        const bf128_t sgi = w[1 + 0];
        const bf128_t *AEI = w + 1 + 1, *NSI = w + 1 + 9, *MEZ = w + 1 + 23,
                      *ME = w + 1 + 31, *MC = w + 1 + 37, *Db = w + 1 + 45,
                      *DC = w + 1 + 53, *HA = w + 1 + 63, *AM = w + 1 + 78,
                      *AP = w + 1 + 92, *AN = w + 1 + 106;
        const bf128_t hi = w[1 + 61], t3 = w[1 + 62];
        const bf128_t tac = w[1 + 120], og = w[1 + 121], pg = w[1 + 122];
        bf128_t acc = bf128_zero();
        int c = 0;
        // 1. pg-gated adder ME12 + 127 = MEZ (sum & carry per bit)
        {
            bf128_t prev = bf128_zero(), inner = bf128_zero();
            for (int j = 0; j < 8; j++) {
                bf128_t a = (j < 6) ? ME[j] : bf128_zero();
                bf128_t b = (j < 7) ? one : bf128_zero();
                bf128_t s = bf128_add(bf128_add(MEZ[j], bf128_add(a, b)), prev);
                inner = bf128_add(inner, bf128_mul(g[c++], s));
                bf128_t cj = bf128_add(MC[j],
                             bf128_add(bf128_mul(a, b), bf128_mul(prev, bf128_add(a, b))));
                inner = bf128_add(inner, bf128_mul(g[c++], cj));
                prev = MC[j];
            }
            acc = bf128_add(acc, bf128_mul(pg, inner));
        }
        // 2. adder aeI + d = MEZ (+ overflow)
        {
            bf128_t prev = bf128_zero();
            for (int j = 0; j < 8; j++) {
                bf128_t axb = bf128_add(AEI[j], Db[j]);
                bf128_t s = bf128_add(bf128_add(MEZ[j], axb), prev);
                acc = bf128_add(acc, bf128_mul(g[c++], s));
                bf128_t cj = bf128_add(DC[j],
                             bf128_add(bf128_mul(AEI[j], Db[j]), bf128_mul(prev, axb)));
                acc = bf128_add(acc, bf128_mul(g[c++], cj));
                prev = DC[j];
            }
            acc = bf128_add(acc, bf128_mul(g[c++], DC[7]));
        }
        // 3. hi = OR(d4..d7); t3 = d1*d2*d3; ha_14 = hi OR t3
        {
            bf128_t p = one;
            for (int j = 4; j < 8; j++) p = bf128_mul(p, bf128_add(one, Db[j]));
            acc = bf128_add(acc, bf128_mul(g[c++], bf128_add(bf128_add(hi, one), p)));
            bf128_t q = bf128_mul(Db[1], bf128_mul(Db[2], Db[3]));
            acc = bf128_add(acc, bf128_mul(g[c++], bf128_add(t3, q)));
            bf128_t x = bf128_add(bf128_add(HA[14], bf128_add(hi, t3)),
                                  bf128_mul(hi, t3));
            acc = bf128_add(acc, bf128_mul(g[c++], x));
        }
        // 4. ha one-hot: pairwise + parity
        for (int s = 0; s < 15; s++) {
            bf128_t inner = bf128_zero();
            for (int t = s + 1; t < 15; t++)
                inner = bf128_add(inner, bf128_mul(g[c++], HA[t]));
            acc = bf128_add(acc, bf128_mul(HA[s], inner));
        }
        {
            bf128_t x = one;
            for (int s = 0; s < 15; s++) x = bf128_add(x, HA[s]);
            acc = bf128_add(acc, bf128_mul(g[c++], x));
        }
        // 5. bucket bits: ha_i * (d_b + bit_b(i)), i<14
        for (int i = 0; i < 14; i++) {
            bf128_t inner = bf128_zero();
            for (int b = 0; b < 8; b++) {
                bf128_t x = ((i >> b) & 1) ? bf128_add(Db[b], one) : Db[b];
                inner = bf128_add(inner, bf128_mul(g[c++], x));
            }
            acc = bf128_add(acc, bf128_mul(HA[i], inner));
        }
        // 6. am shift-select: am_j = sum_i ha_i * nsI_{j+i}
        for (int j = 0; j < 14; j++) {
            bf128_t x = AM[j];
            for (int i = 0; i + j <= 13; i++)
                x = bf128_add(x, bf128_mul(HA[i], NSI[j + i]));
            acc = bf128_add(acc, bf128_mul(g[c++], x));
        }
        // 7. sign split: amP = (1+sgI)*am, amN = sgI*am
        for (int j = 0; j < 14; j++) {
            bf128_t sm = bf128_mul(sgi, AM[j]);
            acc = bf128_add(acc, bf128_mul(g[c++],
                  bf128_add(AP[j], bf128_add(AM[j], sm))));
            acc = bf128_add(acc, bf128_mul(g[c++], bf128_add(AN[j], sm)));
        }
        // 8. tightness: og OR tacc = 1;  tacc -> d = 0
        acc = bf128_add(acc, bf128_mul(g[c++],
              bf128_add(bf128_add(one, bf128_add(og, tac)), bf128_mul(og, tac))));
        {
            bf128_t inner = bf128_zero();
            for (int j = 0; j < 8; j++)
                inner = bf128_add(inner, bf128_mul(g[c++], Db[j]));
            acc = bf128_add(acc, bf128_mul(tac, inner));
        }
        return bf128_mul(w[0], acc);
    }
};

// ---- ZC-B: SP/SN adders.  w = [eq | P[20] N[20] (main) | cols_B] ----
struct BF {
    static constexpr int K = 1 + 40 + NBT, D = 3;
    static constexpr int NC = 82;
    const bf128_t* g;
    __host__ __device__ bf128_t operator()(const bf128_t* w) const {
        const bf128_t *P = w + 1, *N = w + 1 + 20, *AP = w + 1 + 40,
                      *AN = w + 1 + 54, *SP = w + 1 + 68, *SPC = w + 1 + 89,
                      *SN = w + 1 + 109, *SNC = w + 1 + 130;
        bf128_t acc = bf128_zero();
        int c = 0;
        for (int side = 0; side < 2; side++) {
            const bf128_t *V = side ? N : P, *A = side ? AN : AP,
                          *S = side ? SN : SP, *C = side ? SNC : SPC;
            bf128_t prev = bf128_zero();
            for (int j = 0; j < 20; j++) {
                bf128_t a = V[j], b = (j < 14) ? A[j] : bf128_zero();
                bf128_t axb = bf128_add(a, b);
                bf128_t s = bf128_add(bf128_add(S[j], axb), prev);
                acc = bf128_add(acc, bf128_mul(g[c++], s));
                bf128_t cj = bf128_add(C[j],
                             bf128_add(bf128_mul(a, b), bf128_mul(prev, axb)));
                acc = bf128_add(acc, bf128_mul(g[c++], cj));
                prev = C[j];
            }
            acc = bf128_add(acc, bf128_mul(g[c++], bf128_add(S[20], C[19])));
        }
        return bf128_mul(w[0], acc);
    }
};

// ---- ZC-C: reconcile + normalize.  w = [eq | cols_C] ----
struct CF3 {
    static constexpr int K = 1 + NCC, D = 4;
    static constexpr int NC = 355;
    const bf128_t* g;
    __host__ __device__ bf128_t operator()(const bf128_t* w) const {
        const bf128_t one = bf128_one();
        const bf128_t *SP = w + 1, *SN = w + 1 + 21, *CM = w + 1 + 43,
                      *CC = w + 1 + 64, *W = w + 1 + 85, *X = w + 1 + 107,
                      *N1 = w + 1 + 115, *N2 = w + 1 + 123, *MEZ = w + 1 + 131,
                      *AEO = w + 1 + 140, *NSO = w + 1 + 148;
        const bf128_t csg = w[1 + 42], sgo = w[1 + 139];
        bf128_t acc = bf128_zero();
        int c = 0;
        // 1. reconciliation adder lo + cmag = hi (lo/hi = csg-mux of SN/SP)
        {
            bf128_t prev = bf128_zero();
            for (int j = 0; j < 21; j++) {
                bf128_t s = bf128_add(bf128_add(SP[j], SN[j]),
                                      bf128_add(CM[j], prev));
                acc = bf128_add(acc, bf128_mul(g[c++], s));
                bf128_t lo = bf128_add(SN[j], bf128_mul(csg, bf128_add(SN[j], SP[j])));
                bf128_t cj = bf128_add(CC[j],
                             bf128_add(bf128_mul(lo, CM[j]),
                                       bf128_mul(prev, bf128_add(lo, CM[j]))));
                acc = bf128_add(acc, bf128_mul(g[c++], cj));
                prev = CC[j];
            }
            acc = bf128_add(acc, bf128_mul(g[c++], CC[20]));
        }
        // 2. csg forced 0 on cmag = 0
        acc = bf128_add(acc, bf128_mul(g[c++], bf128_mul(csg, W[0])));
        // 3. width one-hot: pairwise + parity
        for (int s = 0; s < 22; s++) {
            bf128_t inner = bf128_zero();
            for (int t = s + 1; t < 22; t++)
                inner = bf128_add(inner, bf128_mul(g[c++], W[t]));
            acc = bf128_add(acc, bf128_mul(W[s], inner));
        }
        {
            bf128_t x = one;
            for (int s = 0; s < 22; s++) x = bf128_add(x, W[s]);
            acc = bf128_add(acc, bf128_mul(g[c++], x));
        }
        // 4. pow2 sandwich: selected top bit = 1; bits >= width = 0
        {
            bf128_t x = bf128_zero();
            for (int i = 1; i < 22; i++)
                x = bf128_add(x, bf128_mul(W[i], bf128_add(one, CM[i - 1])));
            acc = bf128_add(acc, bf128_mul(g[c++], x));
        }
        {
            bf128_t U = bf128_zero();
            for (int j = 0; j < 21; j++) {
                U = bf128_add(U, W[j]);          // U_j = sum_{i<=j} W_i
                acc = bf128_add(acc, bf128_mul(g[c++], bf128_mul(CM[j], U)));
            }
        }
        // 5. nsO shift-select: nsO_j = sum_i W_i * cmag_{j+i-14}
        for (int j = 0; j < 14; j++) {
            bf128_t x = NSO[j];
            for (int i = 1; i < 22; i++) {
                int k = j + i - 14;
                if (k >= 0 && k <= 20) x = bf128_add(x, bf128_mul(W[i], CM[k]));
            }
            acc = bf128_add(acc, bf128_mul(g[c++], x));
        }
        // 6. adder MEZ + width = X (+ overflow); wenc = binary encode of W
        {
            bf128_t wenc[5];
            for (int b = 0; b < 5; b++) {
                wenc[b] = bf128_zero();
                for (int i = 1; i < 22; i++)
                    if ((i >> b) & 1) wenc[b] = bf128_add(wenc[b], W[i]);
            }
            bf128_t prev = bf128_zero();
            for (int j = 0; j < 8; j++) {
                bf128_t a = MEZ[j], b = (j < 5) ? wenc[j] : bf128_zero();
                bf128_t axb = bf128_add(a, b);
                bf128_t s = bf128_add(bf128_add(X[j], axb), prev);
                acc = bf128_add(acc, bf128_mul(g[c++], s));
                bf128_t cj = bf128_add(N1[j],
                             bf128_add(bf128_mul(a, b), bf128_mul(prev, axb)));
                acc = bf128_add(acc, bf128_mul(g[c++], cj));
                prev = N1[j];
            }
            acc = bf128_add(acc, bf128_mul(g[c++], N1[7]));
        }
        // 7. gated adder aeO + 14 = X; zero path aeO = 0, sgO = csg (kept)
        {
            bf128_t gate = bf128_add(one, W[0]), inner = bf128_zero();
            bf128_t prev = bf128_zero();
            for (int j = 0; j < 8; j++) {
                bf128_t a = AEO[j], b = ((14 >> j) & 1) ? one : bf128_zero();
                bf128_t axb = bf128_add(a, b);
                bf128_t s = bf128_add(bf128_add(X[j], axb), prev);
                inner = bf128_add(inner, bf128_mul(g[c++], s));
                bf128_t cj = bf128_add(N2[j],
                             bf128_add(bf128_mul(a, b), bf128_mul(prev, axb)));
                inner = bf128_add(inner, bf128_mul(g[c++], cj));
                prev = N2[j];
            }
            inner = bf128_add(inner, bf128_mul(g[c++], N2[7]));
            acc = bf128_add(acc, bf128_mul(gate, inner));
        }
        {
            bf128_t inner = bf128_zero();
            for (int j = 0; j < 8; j++)
                inner = bf128_add(inner, bf128_mul(g[c++], AEO[j]));
            acc = bf128_add(acc, bf128_mul(W[0], inner));
        }
        acc = bf128_add(acc, bf128_mul(g[c++],
              bf128_add(bf128_add(sgo, csg), bf128_mul(W[0], csg))));
        return bf128_mul(w[0], acc);
    }
};

// ---- link zerocheck (product cube): pr*(eb + sh = ME12) + t constraints.
// w = [eq | eb[5] sh[6] pr t pc[6] (main) | ME12 lifted (trans)] ----
struct LkF {
    static constexpr int K = 26, D = 4;
    static constexpr int NC = 20;
    const bf128_t* g;
    __host__ __device__ bf128_t operator()(const bf128_t* w) const {
        const bf128_t one = bf128_one();
        const bf128_t *EB = w + 1, *SH = w + 1 + 5, *PC = w + 1 + 13,
                      *ME = w + 1 + 19;
        const bf128_t pr = w[1 + 11], t = w[1 + 12];
        bf128_t acc = bf128_zero();
        int c = 0;
        {
            bf128_t prev = bf128_zero(), inner = bf128_zero();
            for (int j = 0; j < 6; j++) {
                bf128_t a = (j < 5) ? EB[j] : bf128_zero(), b = SH[j];
                bf128_t axb = bf128_add(a, b);
                bf128_t s = bf128_add(bf128_add(ME[j], axb), prev);
                inner = bf128_add(inner, bf128_mul(g[c++], s));
                bf128_t cj = bf128_add(PC[j],
                             bf128_add(bf128_mul(a, b), bf128_mul(prev, axb)));
                inner = bf128_add(inner, bf128_mul(g[c++], cj));
                prev = PC[j];
            }
            inner = bf128_add(inner, bf128_mul(g[c++], PC[5]));
            acc = bf128_add(acc, bf128_mul(pr, inner));
        }
        {
            bf128_t inner = bf128_zero();
            for (int j = 0; j < 6; j++)
                inner = bf128_add(inner, bf128_mul(g[c++], SH[j]));
            acc = bf128_add(acc, bf128_mul(t, inner));
        }
        acc = bf128_add(acc, bf128_mul(g[c++], bf128_mul(t, bf128_add(one, pr))));
        return bf128_mul(w[0], acc);
    }
};

// ---- OR-tree level: two trees (t -> og, pr -> pg) share one zerocheck.
// w = [eq | tE tO pE pO | ot op] ----
struct OrF {
    static constexpr int K = 7, D = 3;
    static constexpr int NC = 2;
    const bf128_t* g;
    __host__ __device__ bf128_t operator()(const bf128_t* w) const {
        bf128_t c0 = bf128_add(bf128_add(w[5], bf128_add(w[1], w[2])),
                               bf128_mul(w[1], w[2]));
        bf128_t c1 = bf128_add(bf128_add(w[6], bf128_add(w[3], w[4])),
                               bf128_mul(w[3], w[4]));
        return bf128_mul(w[0], bf128_add(bf128_mul(g[0], c0), bf128_mul(g[1], c1)));
    }
};

// OR-tree heap layout: level l (1..4) lives in the main OR slot at rows
// [2^(lN-l), 2^(lN-l+1)) -- top-l row bits = 0..01, prefix-free across levels
static inline size_t or_off(int lN, int l) { return (size_t)1 << (lN - l); }

// ---- witness ----
struct Wit {
    int lN = 0, CH = 1, LCH = 0;
    std::vector<uint8_t> bits;                   // NSTK3 << lG bytes
};

static inline uint32_t tr_carries(uint32_t a, uint32_t b, int nb) {
    uint32_t r = 0, c = 0;
    for (int j = 0; j < nb; j++) {
        uint32_t aj = (a >> j) & 1, bj = (b >> j) & 1;
        c = (aj & bj) | (c & (aj ^ bj));
        r |= c << j;
    }
    return r;
}

// build the transition witness from the main bit array (which must already
// contain the level-5 tree sums from bacc::build).  Also fills the main
// stack's link carries (colPC), tightness selector (colT) and OR trees.
// Attack hooks (teeth): bump_g -- build a FULLY CONSISTENT witness that used
// max_exp+1 in that group (the caller must have bumped the group's committed
// shifts; the tightness selector lands on a non-achiever); inject_g -- replace
// the chain in-state at that group with (inj_sg, inj_ae, inj_ns) and stay
// consistent downstream (a chain-weld / head attack).
static inline void build(int lN, int CH, std::vector<uint8_t>& mb,
                         int colEB, int colPR, int colSH, int colT, int colPC,
                         int colORT, int colORP, int mainbase, Wit& w,
                         int64_t bump_g = -1, int64_t inject_g = -1,
                         uint32_t inj_sg = 0, uint32_t inj_ae = 0,
                         uint32_t inj_ns = 0) {
    size_t N = (size_t)1 << lN;
    int lG = lN - 5;
    size_t nG = N >> 5;
    int LCH = 0;
    while ((1 << LCH) < CH) LCH++;
    assert((1 << LCH) == CH && nG % CH == 0 && lG >= LCH);
    w.lN = lN; w.CH = CH; w.LCH = LCH;
    w.bits.assign((size_t)NSTK3 << lG, 0);
    auto MB = [&](int col, size_t i) -> uint8_t& { return mb[((size_t)col << lN) + i]; };
    auto TB = [&](int col, size_t g) -> uint8_t& { return w.bits[((size_t)col << lG) + g]; };
    // level-5 tree sums out of the main slots
    std::vector<uint32_t> P(nG, 0), Q(nG, 0);
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < (int64_t)nG; g++)
        for (int j = 0; j < 20; j++) {
            int slot, t;
            bacc::acc_slot(5, 0, j, mainbase, slot, t);
            P[g] |= (uint32_t)(mb[((size_t)slot << lN) + ((size_t)t << lG) + g] & 1) << j;
            bacc::acc_slot(5, 0, 20 + j, mainbase, slot, t);
            Q[g] |= (uint32_t)(mb[((size_t)slot << lN) + ((size_t)t << lG) + g] & 1) << j;
        }
    size_t nchain = nG / CH;
    #pragma omp parallel for schedule(static)
    for (int64_t ch = 0; ch < (int64_t)nchain; ch++) {
        uint32_t sg = 0, ae = 0, ns = 0;
        for (int tt = 0; tt < CH; tt++) {
            size_t g = (size_t)ch * CH + tt;
            if ((int64_t)g == inject_g) { sg = inj_sg; ae = inj_ae; ns = inj_ns; }
            auto put = [&](int base, uint32_t v, int nb) {
                for (int j = 0; j < nb; j++) TB(base + j, g) = (uint8_t)((v >> j) & 1);
            };
            uint32_t MEZ = ae, pgb = 0;
            int tsel = -1;
            uint32_t ebv[32], prv[32], shv[32];
            for (int kk = 0; kk < 32; kk++) {
                size_t i = g * 32 + kk;
                uint32_t eb = 0, sh = 0;
                for (int j = 0; j < 5; j++) eb |= (uint32_t)(MB(colEB + j, i) & 1) << j;
                for (int j = 0; j < 6; j++) sh |= (uint32_t)(MB(colSH + j, i) & 1) << j;
                ebv[kk] = eb; shv[kk] = sh; prv[kk] = MB(colPR, i) & 1;
                if (prv[kk]) { pgb = 1; if (eb + 127 > MEZ) MEZ = eb + 127; }
            }
            bool bump = (int64_t)g == bump_g;
            if (bump) { assert(pgb && MEZ + 1 < 256); MEZ += 1; }
            for (int kk = 0; kk < 32 && tsel < 0; kk++)
                if (prv[kk] && (bump || ebv[kk] + 127 == MEZ)) tsel = kk;
            uint32_t tacc = (tsel < 0) ? 1u : 0u, ogb = 1u - tacc;
            assert(!tacc || MEZ == ae);
            uint32_t d = MEZ - ae;
            put(TSGI, sg, 1); put(TAEI, ae, 8); put(TNSI, ns, 14);
            put(TMEZ, MEZ, 8);
            uint32_t me12 = pgb ? MEZ - 127 : 0;
            assert(!pgb || (MEZ >= 127 && me12 < 64));
            put(TME12, me12, 6);
            put(TMC, pgb ? tr_carries(me12, 127, 8) : 0, 8);
            put(TD, d, 8); put(TDC, tr_carries(ae, d, 8), 8);
            assert(ae + d == MEZ && MEZ < 256);
            TB(THI, g) = ((d >> 4) & 15) != 0;
            TB(TT3, g) = ((d >> 1) & 7) == 7;
            TB(THA + (d < 14 ? (int)d : 14), g) = 1;
            uint32_t am = (d <= 13) ? (ns >> d) : 0;
            put(TAM, am, 14);
            uint32_t amp = sg ? 0 : am, amn = sg ? am : 0;
            put(TAMP, amp, 14); put(TAMN, amn, 14);
            uint32_t SP = P[g] + amp, SN = Q[g] + amn;
            assert(SP < (1u << 21) && SN < (1u << 21));
            put(TSP, SP, 21); put(TSPC, tr_carries(P[g], amp, 20), 20);
            put(TSN, SN, 21); put(TSNC, tr_carries(Q[g], amn, 20), 20);
            uint32_t csg = SN > SP ? 1u : 0u;
            uint32_t cm = csg ? SN - SP : SP - SN;
            uint32_t lo = csg ? SP : SN;
            put(TCSG, csg, 1); put(TCM, cm, 21);
            put(TCC, tr_carries(lo, cm, 21), 21);
            int width = 0;
            while (cm >> width) width++;
            TB(TW + width, g) = 1;
            uint32_t nso = width > 14 ? (cm >> (width - 14)) : (cm << (14 - width));
            assert(nso < (1u << 14) && (!cm || (nso >> 13) == 1));
            uint32_t X = MEZ + width;
            assert(X < 256);
            put(TX, X, 8); put(TN1, tr_carries(MEZ, (uint32_t)width, 8), 8);
            uint32_t aeo = 0, sgo = 0, nc2 = 0;
            if (width) {
                assert(X >= 14);
                aeo = X - 14; nc2 = tr_carries(aeo, 14, 8); sgo = csg;
            } else nso = 0;
            put(TN2, nc2, 8);
            put(TSGO, sgo, 1); put(TAEO, aeo, 8); put(TNSO, nso, 14);
            TB(TTAC, g) = (uint8_t)tacc; TB(TOG, g) = (uint8_t)ogb;
            TB(TPG, g) = (uint8_t)pgb;
            if (tsel >= 0) {
                MB(colT, g * 32 + tsel) = 1;
                assert(bump || shv[tsel] == 0);
            }
            for (int kk = 0; kk < 32; kk++)
                if (prv[kk]) {
                    size_t i = g * 32 + kk;
                    assert(ebv[kk] + shv[kk] == me12);
                    uint32_t pc = tr_carries(ebv[kk], shv[kk], 6);
                    for (int j = 0; j < 6; j++)
                        MB(colPC + j, i) = (uint8_t)((pc >> j) & 1);
                }
            sg = sgo; ae = aeo; ns = nso;
        }
    }
    // OR trees over t and pr (heap-packed levels 1..4; level 5 = og/pg)
    std::vector<uint8_t> curT(N), curP(N);
    for (size_t i = 0; i < N; i++) {
        curT[i] = MB(colT, i) & 1;
        curP[i] = MB(colPR, i) & 1;
    }
    for (int l = 1; l <= 5; l++) {
        size_t half = N >> l, off = or_off(lN, l);
        std::vector<uint8_t> nxT(half), nxP(half);
        #pragma omp parallel for schedule(static)
        for (int64_t y = 0; y < (int64_t)half; y++) {
            nxT[y] = curT[2 * y] | curT[2 * y + 1];
            nxP[y] = curP[2 * y] | curP[2 * y + 1];
            if (l < 5) {
                MB(colORT, off + y) = nxT[y];
                MB(colORP, off + y) = nxP[y];
            } else {
                assert(nxT[y] == TB(TOG, y) && nxP[y] == TB(TPG, y));
            }
        }
        curT.swap(nxT); curP.swap(nxP);
    }
}

// ---- host-side per-group witness validator (teeth aid): number of groups
// violating some raw transition constraint (excluding chain binding) ----
static inline size_t validate(const Wit& w, const std::vector<uint8_t>& mb,
                              int colEB, int colPR, int colSH, int colT,
                              int colPC, int mainbase) {
    int lN = w.lN, lG = lN - 5;
    size_t nG = (size_t)1 << lG, bad = 0;
    auto MB = [&](int col, size_t i) -> int { return mb[((size_t)col << lN) + i] & 1; };
    auto gt = [&](int base, size_t g, int nb) {
        uint32_t v = 0;
        for (int j = 0; j < nb; j++)
            v |= (uint32_t)(w.bits[((size_t)(base + j) << lG) + g] & 1) << j;
        return v;
    };
    for (size_t g = 0; g < nG; g++) {
        bool ok = true;
        uint32_t sgi = gt(TSGI, g, 1), aei = gt(TAEI, g, 8), nsi = gt(TNSI, g, 14);
        uint32_t sgo = gt(TSGO, g, 1), aeo = gt(TAEO, g, 8), nso = gt(TNSO, g, 14);
        uint32_t MEZ = gt(TMEZ, g, 8), me12 = gt(TME12, g, 6);
        uint32_t d = gt(TD, g, 8);
        uint32_t ha = gt(THA, g, 15), am = gt(TAM, g, 14);
        uint32_t amp = gt(TAMP, g, 14), amn = gt(TAMN, g, 14);
        uint32_t SP = gt(TSP, g, 21), SN = gt(TSN, g, 21);
        uint32_t csg = gt(TCSG, g, 1), cm = gt(TCM, g, 21), woh = gt(TW, g, 22);
        uint32_t X = gt(TX, g, 8);
        uint32_t tac = gt(TTAC, g, 1), og = gt(TOG, g, 1), pg = gt(TPG, g, 1);
        // recompute from the per-product columns
        uint32_t pgr = 0, ogr = 0, mez_r = aei;
        for (int kk = 0; kk < 32; kk++) {
            size_t i = g * 32 + kk;
            uint32_t eb = 0, sh = 0;
            for (int j = 0; j < 5; j++) eb |= (uint32_t)MB(colEB + j, i) << j;
            for (int j = 0; j < 6; j++) sh |= (uint32_t)MB(colSH + j, i) << j;
            int pr = MB(colPR, i), t = MB(colT, i);
            if (pr) {
                pgr = 1;
                if (eb + 127 > mez_r) mez_r = eb + 127;
                if (eb + sh != me12) ok = false;          // link
            }
            if (t && (!pr || sh != 0)) ok = false;        // selector
            if (t) ogr = 1;
        }
        if (pg != pgr || og != ogr) ok = false;
        if (MEZ != mez_r) ok = false;                     // dominance+tightness
        if (!(og || tac)) ok = false;
        if (tac && d != 0) ok = false;
        if (aei + d != MEZ) ok = false;
        if (pg && me12 + 127 != MEZ) ok = false;
        uint32_t bucket = d < 14 ? d : 14;
        if (ha != (1u << bucket)) ok = false;
        if (am != ((d <= 13) ? (nsi >> d) : 0)) ok = false;
        if (amp != (sgi ? 0 : am) || amn != (sgi ? am : 0)) ok = false;
        if (csg ? (SP + cm != SN) : (SN + cm != SP)) ok = false;
        if (csg && cm == 0) ok = false;
        int width = 0;
        while (cm >> width) width++;
        if (woh != (1u << width)) ok = false;
        uint32_t nso_r = width > 14 ? (cm >> (width - 14))
                                    : (cm << (14 - width));
        if (nso != nso_r) ok = false;
        if (X != MEZ + (uint32_t)width) ok = false;
        if (width) { if (aeo != X - 14 || sgo != csg) ok = false; }
        else       { if (aeo != 0 || sgo != 0 || nso != 0) ok = false; }
        // chain binding
        int CH = w.CH;
        if ((int)(g % CH) == 0) {
            if (sgi || aei || nsi) ok = false;
        } else {
            if (sgi != gt(TSGO, g - 1, 1) || aei != gt(TAEO, g - 1, 8) ||
                nsi != gt(TNSO, g - 1, 14)) ok = false;
        }
        // SP/SN vs the actual tree sums are checked by ZC-B against main;
        // here trust the committed L5 (validated by the acc gadget's teeth)
        if (!ok) bad++;
    }
    return bad;
}

// ---- sumcheck byte-column extractors ----
static inline void sc_bytes_cols(int lG, const int* cols, int ncols,
                                 const uint8_t* tb, std::vector<uint8_t>& out) {
    size_t nG = (size_t)1 << lG;
    out.assign((size_t)ncols * nG, 0);
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < ncols; k++)
        for (size_t y = 0; y < nG; y++)
            out[(size_t)k * nG + y] = tb[((size_t)cols[k] << lG) + y] & 1;
}
static inline void sc_bytes_B(int lN, const uint8_t* mb, int mainbase,
                              const uint8_t* tb, std::vector<uint8_t>& out) {
    int lG = lN - 5;
    size_t nG = (size_t)1 << lG;
    out.assign((size_t)(40 + NBT) * nG, 0);
    int cb[NBT]; cols_B(cb);
    #pragma omp parallel for schedule(static)
    for (int64_t y = 0; y < (int64_t)nG; y++) {
        for (int j = 0; j < 40; j++) {
            int slot, t;
            bacc::acc_slot(5, 0, j, mainbase, slot, t);
            out[(size_t)j * nG + y] =
                mb[((size_t)slot << lN) + ((size_t)t << lG) + y] & 1;
        }
        for (int k = 0; k < NBT; k++)
            out[(size_t)(40 + k) * nG + y] = tb[((size_t)cb[k] << lG) + y] & 1;
    }
}
static inline void sc_bytes_link(int lN, const uint8_t* mb, int colEB, int colSH,
                                 int colPR, int colT, int colPC,
                                 const uint8_t* tb, std::vector<uint8_t>& out) {
    size_t N = (size_t)1 << lN;
    int lG = lN - 5;
    out.assign((size_t)25 * N, 0);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)N; i++) {
        int c = 0;
        for (int j = 0; j < 5; j++)
            out[(size_t)(c++) * N + i] = mb[((size_t)(colEB + j) << lN) + i] & 1;
        for (int j = 0; j < 6; j++)
            out[(size_t)(c++) * N + i] = mb[((size_t)(colSH + j) << lN) + i] & 1;
        out[(size_t)(c++) * N + i] = mb[((size_t)colPR << lN) + i] & 1;
        out[(size_t)(c++) * N + i] = mb[((size_t)colT << lN) + i] & 1;
        for (int j = 0; j < 6; j++)
            out[(size_t)(c++) * N + i] = mb[((size_t)(colPC + j) << lN) + i] & 1;
        size_t g = (size_t)i >> 5;
        for (int j = 0; j < 6; j++)
            out[(size_t)(c++) * N + i] = tb[((size_t)(TME12 + j) << lG) + g] & 1;
    }
}
// OR-tree level l: inputs = even/odd of level l-1 (level 0 = t/pr in main),
// committed = level l (main heap region, or og/pg in the trans stack at l=5)
static inline void sc_bytes_or(int lN, int l, const uint8_t* mb, int colT,
                               int colPR, int colORT, int colORP,
                               const uint8_t* tb, std::vector<uint8_t>& out) {
    size_t half = (size_t)1 << (lN - l);
    int lG = lN - 5;
    out.assign((size_t)6 * half, 0);
    size_t offp = (l >= 2) ? or_off(lN, l - 1) : 0, offl = or_off(lN, l);
    #pragma omp parallel for schedule(static)
    for (int64_t y = 0; y < (int64_t)half; y++) {
        for (int par = 0; par < 2; par++) {
            size_t i = 2 * y + par;
            uint8_t tv = (l == 1) ? mb[((size_t)colT << lN) + i]
                                  : mb[((size_t)colORT << lN) + offp + i];
            uint8_t pv = (l == 1) ? mb[((size_t)colPR << lN) + i]
                                  : mb[((size_t)colORP << lN) + offp + i];
            out[(size_t)(0 + par) * half + y] = tv & 1;
            out[(size_t)(2 + par) * half + y] = pv & 1;
        }
        if (l < 5) {
            out[(size_t)4 * half + y] = mb[((size_t)colORT << lN) + offl + y] & 1;
            out[(size_t)5 * half + y] = mb[((size_t)colORP << lN) + offl + y] & 1;
        } else {
            out[(size_t)4 * half + y] = tb[((size_t)TOG << lG) + y] & 1;
            out[(size_t)5 * half + y] = tb[((size_t)TPG << lG) + y] & 1;
        }
    }
}

// ---- proof piece (embedded in the Hawkeye proof) ----
struct TrProof {
    int on = 0, CH = 1;
    uint8_t root3[32];
    BfScProof lk, orsc[5], A, B, C;
    std::vector<bf128_t> evMain[11];     // RTR RLK orA1..4 orB1..4 orB5
    std::vector<bf128_t> evT[6];         // TA TB TC TLK TOR5 THEAD
    std::vector<std::vector<bf128_t>> evTS;   // 2*LCH point evals (I,O pairs)
    BfPcsProofM pcs3;
    size_t bytes() const {
        if (!on) return sizeof(int);
        size_t s = 2 * sizeof(int) + 32 + pcs3.bytes();
        auto sc = [&](const BfScProof& p) {
            return (p.rounds.size() + p.finals.size()) * sizeof(bf128_t);
        };
        s += sc(lk) + sc(A) + sc(B) + sc(C);
        for (int i = 0; i < 5; i++) s += sc(orsc[i]);
        for (int i = 0; i < 11; i++) s += evMain[i].size() * sizeof(bf128_t);
        for (int i = 0; i < 6; i++) s += evT[i].size() * sizeof(bf128_t);
        for (auto& v : evTS) s += v.size() * sizeof(bf128_t);
        return s;
    }
};

static inline BfPcsParams tr_params(int lN) {
    BfPcsParams p;
    p.l = (lN - 5) + LOGSTK3; p.lcol = p.l / 2; p.lrow = p.l - p.lcol; p.Q = 100;
    return p;
}
static inline void tr_gammas(fs::Transcript& tr, int nc, std::vector<bf128_t>& g) {
    bf128_t gamma = bf_chal128(tr);
    g.resize(nc);
    g[0] = bf128_one();
    for (int c = 1; c < nc; c++) g[c] = bf128_mul(g[c - 1], gamma);
}

// one zerocheck, GPU or host, byte-identical either way (bacc::acc_zc shape).
// GPUOK=false compiles the host path ONLY: the wide group-cube functors
// (K = 124..163) make ptxas spend tens of minutes register-allocating the
// spill-bound generic round kernel, and their domain is the tiny group cube
// where the OpenMP host prover is faster anyway -- so they never touch the
// device in either prover mode (proofs stay byte-identical by construction).
template <class CF, bool GPUOK = true>
static inline void tr_zc(int lz, const std::vector<uint8_t>& sb, fs::Transcript& tr,
                         BfScProof& pf, std::vector<bf128_t>& zeta,
                         const std::vector<bf128_t>& g, bool gpu,
                         BfScDev* ws = nullptr, uint8_t* d_scratch = nullptr) {
    size_t half = (size_t)1 << lz;
    std::vector<bf128_t> rz(lz);
    for (auto& x : rz) x = bf_chal128(tr);
    if constexpr (GPUOK) {
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
            cudaMemcpy(d_g, g.data(), g.size() * sizeof(bf128_t),
                       cudaMemcpyHostToDevice);
            CF cf; cf.g = d_g;
            bf_sumcheck_prove_gpu(dv, cf, tr, pf, zeta);
            cudaFree(d_g);
            if (!ws) bfsc_dev_free(dv);
            return;
        }
    }
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
template <class CF>
static inline bool tr_zc_verify(int lz, const BfScProof& pf, fs::Transcript& tr,
                                std::vector<bf128_t>& zeta,
                                const std::vector<bf128_t>& g) {
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

} // namespace btr
