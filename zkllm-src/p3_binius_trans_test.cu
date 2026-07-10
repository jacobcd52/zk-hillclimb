// Selftest + measured A/B for the Binius GROUP TRANSITION gadget
// (p3_binius_trans.cuh + the bhw integration, design doc section 21.11):
// max_exp (dominance + tightness), acc realign, signed reconciliation
// cmag = |P - N + acc| and normalize, chained across groups -- the COMPLETE
// Hawkeye matmul semantics over the binary tower.
//
// Teeth run on REAL chain-ordered golden vectors (hawkeye_trans.bin from
// hawkeye_ref.py --dumptrans, Triton-cross-checked, with golden per-group
// max_exp and out-states): the built witness reproduces every golden
// MEZ/sgO/aeO/nsO bitwise (so the last group of every chain IS the layer's
// final accumulator state); honest accept; GPU/host byte-identical proofs;
// targeted witness flips at every stage; THREE weld attacks with fully
// consistent downstream witnesses (overstated max_exp caught by the
// tightness selector, a broken chain caught by the shift-restriction pair,
// a nonzero head state caught by the head point); and proof-object tampers.
//
// Measurement: hawkeye_trans_big.bin (262144 real products = 8192 groups =
// 256 chains of CH=32, regenerable: python3 hawkeye_ref.py --dumptransbig
// hawkeye_trans_big.bin), A/B'd against the Goldilocks per-product gadget
// (R=2, Q=24).  Stated plainly: the GL gadget proves NEITHER accumulation
// NOR the transition; the Binius side proves strictly more here.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>
#include "p3_binius_hawkeye.cuh"
#include "p3_hawkeye_prod.cuh"

using std::vector;

static int np_ = 0, nf_ = 0;
static void ck(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) np_++; else nf_++;
}
static uint64_t rs_ = 0x7274616e73747231ULL;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }

struct TransGold {
    int64_t CH = 1, ng = 0;
    vector<int64_t> P, N, S, MEZ, SGO, AEO, NSO;
};
static bool load_trans(const char* path, vector<int64_t> raw[10], TransGold& G) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t n = 0;
    if (fread(&n, 8, 1, f) != 1) { fclose(f); return false; }
    for (int c = 0; c < 10; c++) {
        raw[c].resize(n);
        if (fread(raw[c].data(), 8, n, f) != (size_t)n) { fclose(f); return false; }
    }
    int64_t magic = 0;
    if (fread(&magic, 8, 1, f) != 1 || magic != 0x54524E31 ||
        fread(&G.CH, 8, 1, f) != 1 || fread(&G.ng, 8, 1, f) != 1) {
        fclose(f); return false;
    }
    vector<int64_t>* v[7] = {&G.P, &G.N, &G.S, &G.MEZ, &G.SGO, &G.AEO, &G.NSO};
    for (int i = 0; i < 7; i++) {
        v[i]->resize(G.ng);
        if (fread(v[i]->data(), 8, G.ng, f) != (size_t)G.ng) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

static bool sc_equal(const BfScProof& a, const BfScProof& b) {
    return a.rounds.size() == b.rounds.size() && a.finals.size() == b.finals.size() &&
           !memcmp(a.rounds.data(), b.rounds.data(), a.rounds.size() * sizeof(bf128_t)) &&
           !memcmp(a.finals.data(), b.finals.data(), a.finals.size() * sizeof(bf128_t));
}
static bool vec_equal(const vector<bf128_t>& a, const vector<bf128_t>& b) {
    return a.size() == b.size() &&
           !memcmp(a.data(), b.data(), a.size() * sizeof(bf128_t));
}
static bool pcsm_equal(const BfPcsProofM& x, const BfPcsProofM& y) {
    if (x.t.size() != y.t.size()) return false;
    for (size_t m = 0; m < x.t.size(); m++)
        if (!vec_equal(x.t[m], y.t[m])) return false;
    return vec_equal(x.u, y.u) && x.cols == y.cols && x.paths == y.paths;
}
static bool proofs_equal(const bhw::BhwProof& x, const bhw::BhwProof& y) {
    if (memcmp(x.root, y.root, 32) || memcmp(x.xev, y.xev, sizeof x.xev) ||
        memcmp(x.xev2, y.xev2, sizeof x.xev2) || !sc_equal(x.sc, y.sc) ||
        !pcsm_equal(x.pcs, y.pcs)) return false;
    if (x.acc.on != y.acc.on || x.tr.on != y.tr.on || x.tr.CH != y.tr.CH)
        return false;
    if (x.acc.on) {
        if (memcmp(x.acc.root2, y.acc.root2, 32)) return false;
        for (int i = 0; i < 5; i++)
            if (!sc_equal(x.acc.sc[i], y.acc.sc[i])) return false;
        if (!vec_equal(x.acc.evB1, y.acc.evB1) || !vec_equal(x.acc.evA5, y.acc.evA5))
            return false;
        for (int i = 0; i < 4; i++)
            if (!vec_equal(x.acc.evA[i], y.acc.evA[i]) ||
                !vec_equal(x.acc.evB[i], y.acc.evB[i])) return false;
        if (!pcsm_equal(x.acc.pcs2, y.acc.pcs2)) return false;
    }
    if (x.tr.on) {
        if (memcmp(x.tr.root3, y.tr.root3, 32)) return false;
        if (!sc_equal(x.tr.lk, y.tr.lk) || !sc_equal(x.tr.A, y.tr.A) ||
            !sc_equal(x.tr.B, y.tr.B) || !sc_equal(x.tr.C, y.tr.C)) return false;
        for (int i = 0; i < 5; i++)
            if (!sc_equal(x.tr.orsc[i], y.tr.orsc[i])) return false;
        for (int i = 0; i < 11; i++)
            if (!vec_equal(x.tr.evMain[i], y.tr.evMain[i])) return false;
        for (int i = 0; i < 6; i++)
            if (!vec_equal(x.tr.evT[i], y.tr.evT[i])) return false;
        if (x.tr.evTS.size() != y.tr.evTS.size()) return false;
        for (size_t i = 0; i < x.tr.evTS.size(); i++)
            if (!vec_equal(x.tr.evTS[i], y.tr.evTS[i])) return false;
        if (!pcsm_equal(x.tr.pcs3, y.tr.pcs3)) return false;
    }
    return true;
}

// decode an nb-bit value from transition-stack columns
static uint32_t tval(const btr::Wit& w, int base, size_t g, int nb) {
    int lG = w.lN - 5;
    uint32_t v = 0;
    for (int j = 0; j < nb; j++)
        v |= (uint32_t)(w.bits[((size_t)(base + j) << lG) + g] & 1) << j;
    return v;
}

int main() {
    printf("=== Binius group transition gadget: max_exp + realign + signed "
           "reconciliation + normalize + chain (section 21.11) ===\n");
#ifndef __CUDA_ARCH__
    bf16_tab();
#endif
    #pragma omp parallel
    { volatile int wa = omp_get_thread_num(); (void)wa; }

    vector<int64_t> raw[10];
    TransGold G;
    if (!load_trans("hawkeye_trans.bin", raw, G)) {
        printf("FATAL: hawkeye_trans.bin missing (python3 hawkeye_ref.py "
               "--dumptrans hawkeye_trans.bin)\n");
        return 1;
    }
    size_t n_real = 0;
    vector<uint8_t> bits;
    int lN = bhw::bhw_build_bits(raw, bits, &n_real);
    size_t N = (size_t)1 << lN;
    int CH = (int)G.CH;
    printf("golden vectors: %zu products (%lld groups, CH=%d) -> N=%zu (lN=%d)\n",
           n_real, (long long)G.ng, CH, N, lN);

    ck("per-product witness still validates in chain order (0 violated rows)",
       bhw::bhw_validate(bits, lN) == 0);

    bacc::Wit aw;
    bacc::build(lN, bits, bhw::LALM, bhw::LALS, bhw::L5BASE, aw);
    {
        bool ok = true;
        for (size_t g = 0; g < (size_t)G.ng; g++)
            if ((int64_t)aw.sumP[g] != G.P[g] || (int64_t)aw.sumN[g] != G.N[g] ||
                G.P[g] - G.N[g] != G.S[g]) ok = false;
        ck("adder trees reproduce the golden P/N/S sums in chain order", ok);
    }

    btr::Wit tw;
    btr::build(lN, CH, bits, bhw::LEB, bhw::LPR, bhw::LSH, bhw::LTSEL,
               bhw::LPC, bhw::LORT, bhw::LORP, bhw::L5BASE, tw);
    {   // THE golden tooth: every per-group max_exp and out-state bitwise
        bool okM = true, okS = true, okA = true, okN = true;
        for (size_t g = 0; g < (size_t)G.ng; g++) {
            if (tval(tw, btr::TMEZ, g, 8) != (uint32_t)G.MEZ[g]) okM = false;
            if (tval(tw, btr::TSGO, g, 1) != (uint32_t)G.SGO[g]) okS = false;
            if (tval(tw, btr::TAEO, g, 8) != (uint32_t)G.AEO[g]) okA = false;
            if (tval(tw, btr::TNSO, g, 14) != (uint32_t)G.NSO[g]) okN = false;
        }
        ck("per-group max_exp reproduces the golden MEZ bitwise (all groups)", okM);
        ck("out-state sign == golden (all groups; last group = layer output)", okS);
        ck("out-state exponent == golden aeO bitwise (all groups)", okA);
        ck("out-state significand == golden nsO bitwise (all groups)", okN);
    }
    ck("transition witness validates (0 violated groups)",
       btr::validate(tw, bits, bhw::LEB, bhw::LPR, bhw::LSH, bhw::LTSEL,
                     bhw::LPC, bhw::L5BASE) == 0);
    {   // validator teeth: flips at every stage are seen
        bool ok = true;
        int cols[] = {btr::TMEZ + 2, btr::TME12, btr::TD + 1, btr::THA + 3,
                      btr::TAM + 4, btr::TAMP + 2, btr::TSP + 7, btr::TCSG,
                      btr::TCM + 5, btr::TW + 9, btr::TX + 1, btr::TAEO + 2,
                      btr::TNSO + 6, btr::TOG, btr::TPG,
                      btr::TSGI, btr::TAEI + 1, btr::TNSI + 3};
        int lG = lN - 5;
        for (int c : cols) {
            size_t g = rnd() % G.ng;
            tw.bits[((size_t)c << lG) + g] ^= 1;
            if (btr::validate(tw, bits, bhw::LEB, bhw::LPR, bhw::LSH,
                              bhw::LTSEL, bhw::LPC, bhw::L5BASE) == 0) ok = false;
            tw.bits[((size_t)c << lG) + g] ^= 1;
        }
        ck("validator catches a flipped bit at every transition stage (18 cols)", ok);
    }

    {   // functor spot checks: zero on honest rows, nonzero after a flip
        int lG = lN - 5;
        size_t half = (size_t)1 << lG;
        bool ok = true;
        auto probe = [&](auto cf, int KM1, vector<uint8_t>& sb, int ncg,
                         size_t dom, int flipcol, long fliprow = -1) {
            vector<bf128_t> g(ncg), w(1 + KM1);
            for (auto& x : g) { x.lo = rnd(); x.hi = rnd(); }
            cf.g = g.data();
            auto evalrow = [&](size_t y) {
                w[0] = bf128_one();
                for (int k = 0; k < KM1; k++)
                    w[1 + k] = sb[(size_t)k * dom + y] ? bf128_one() : bf128_zero();
                return cf(w.data());
            };
            for (int t = 0; t < 300; t++)
                if (!bf128_eq(evalrow(rnd() % dom), bf128_zero())) ok = false;
            size_t y = fliprow >= 0 ? (size_t)fliprow : rnd() % dom;
            sb[(size_t)flipcol * dom + y] ^= 1;
            if (bf128_eq(evalrow(y), bf128_zero())) ok = false;
            sb[(size_t)flipcol * dom + y] ^= 1;
        };
        vector<uint8_t> sb;
        int ca[btr::NA]; btr::cols_A(ca);
        btr::sc_bytes_cols(lG, ca, btr::NA, tw.bits.data(), sb);
        probe(btr::AF{}, btr::NA, sb, btr::AF::NC, half, 61);        // HI
        btr::sc_bytes_B(lN, bits.data(), bhw::L5BASE, tw.bits.data(), sb);
        probe(btr::BF{}, 40 + btr::NBT, sb, btr::BF::NC, half, 75);  // SP bit
        int cc[btr::NCC]; btr::cols_C(cc);
        btr::sc_bytes_cols(lG, cc, btr::NCC, tw.bits.data(), sb);
        probe(btr::CF3{}, btr::NCC, sb, btr::CF3::NC, half, 81);     // CC bit
        btr::sc_bytes_link(lN, bits.data(), bhw::LEB, bhw::LSH, bhw::LPR,
                           bhw::LTSEL, bhw::LPC, tw.bits.data(), sb);
        {   // flip a link CARRY on a present row (a t flip is caught by the
            // OR tree, not by the link functor)
            long pres = 0;
            while (!sb[(size_t)11 * N + pres]) pres++;
            probe(btr::LkF{}, 25, sb, btr::LkF::NC, N, 13, pres);    // pc_0
        }
        for (int l = 1; l <= 5; l++) {
            btr::sc_bytes_or(lN, l, bits.data(), bhw::LTSEL, bhw::LPR,
                             bhw::LORT, bhw::LORP, tw.bits.data(), sb);
            probe(btr::OrF{}, 6, sb, btr::OrF::NC, N >> l, 4);       // ot node
        }
        ck("A/B/C, link and OR functors == 0 on honest rows, != 0 after a "
           "flipped bit (9 zerochecks)", ok);
    }

    // ---- honest proof + GPU/host identity + regressions ----
    bhw::BhwProof pf; bhw::BhwStats st;
    bhw::bhw_prove(lN, bits, pf, st, true, &aw, nullptr, &tw);
    ck("honest composed proof (products + acc + transition) accepts (GPU)",
       bhw::bhw_verify(pf));
    {
        bhw::BhwProof ph; bhw::BhwStats sh;
        bhw::bhw_prove(lN, bits, ph, sh, false, &aw, nullptr, &tw);
        ck("GPU proof byte-identical to host proof (all 9 new zerochecks + "
           "both new openings)", proofs_equal(pf, ph));
    }
    {
        bhw::BhwProof p0; bhw::BhwStats s0;
        bhw::bhw_prove(lN, bits, p0, s0, true, &aw);
        ck("trans=off regression: products+acc proof on the chain-ordered "
           "witness accepts", bhw::bhw_verify(p0));
    }

    // ---- COMPOSED OUTPUT BINDING (21.12): the committed output tensor Yout at
    // every chain-final group must equal the proven Hawkeye out-state.  A prover
    // that commits ANY other output (flip one Yout bit at a chain-final group,
    // fully consistently) is rejected by the chain-final binding point. ----
    {
        int lG = lN - 5;
        // Yout is a per-group copy of the out-state; the chain-final groups are
        // g = CH-1, 2*CH-1, ...  Rebuild the transition witness flipping Yout at
        // the first chain-final group and re-prove the whole composed statement.
        struct { const char* what; int bit; } yf[] = {
            {"flipped Yout SIGN at chain-final rejects (output binding)", 0},
            {"flipped Yout EXPONENT bit at chain-final rejects (output binding)", 3},
            {"flipped Yout SIGNIFICAND bit at chain-final rejects (output binding)", 11},
        };
        for (auto& f : yf) {
            btr::Wit ty;
            btr::build(lN, CH, bits, bhw::LEB, bhw::LPR, bhw::LSH, bhw::LTSEL,
                       bhw::LPC, bhw::LORT, bhw::LORP, bhw::L5BASE, ty,
                       -1, -1, 0, 0, 0, /*yflip_g=*/CH - 1, /*yflip_bit=*/f.bit);
            bhw::BhwProof q; bhw::BhwStats sq;
            bhw::bhw_prove(lN, bits, q, sq, true, &aw, nullptr, &ty);
            ck(f.what, !bhw::bhw_verify(q));
        }
        // and the honest Yout (no flip) still accepts through the same path
        {
            btr::Wit ty;
            btr::build(lN, CH, bits, bhw::LEB, bhw::LPR, bhw::LSH, bhw::LTSEL,
                       bhw::LPC, bhw::LORT, bhw::LORP, bhw::L5BASE, ty);
            bhw::BhwProof q; bhw::BhwStats sq;
            bhw::bhw_prove(lN, bits, q, sq, true, &aw, nullptr, &ty);
            ck("honest Yout == out-state accepts (output binding, all chains)",
               bhw::bhw_verify(q));
        }
    }

    // ---- witness attacks ----
    auto attack = [&](const char* what, const vector<uint8_t>& mb2,
                      const bacc::Wit& a2, const btr::Wit& t2) {
        bhw::BhwProof q; bhw::BhwStats s2;
        bhw::bhw_prove(lN, mb2, q, s2, true, &a2, nullptr, &t2);
        ck(what, !bhw::bhw_verify(q));
    };
    {   // simple flips (each caught by its zerocheck / opening)
        int lG = lN - 5;
        struct { const char* what; int col; } flips[] = {
            {"flipped MEZ bit rejects", btr::TMEZ + 1},
            {"flipped aligned-acc bit rejects", btr::TAM + 3},
            {"flipped SP carry bit rejects", btr::TSPC + 5},
            {"flipped cmag bit rejects", btr::TCM + 2},
            {"flipped width one-hot bit rejects", btr::TW + 7},
            {"flipped out-exponent bit rejects", btr::TAEO + 1},
            {"flipped out-significand bit rejects", btr::TNSO + 5},
        };
        for (auto& fl : flips) {
            btr::Wit t2 = tw;
            t2.bits[((size_t)fl.col << lG) + (rnd() % G.ng)] ^= 1;
            attack(fl.what, bits, aw, t2);
        }
        {   // tacc flip at a group where d != 0 (tacc -> d = 0 must fire)
            size_t g = 0;
            while (tval(tw, btr::TD, g, 8) == 0) g++;
            btr::Wit t2 = tw;
            t2.bits[((size_t)btr::TTAC << lG) + g] ^= 1;
            attack("flipped tacc bit rejects (tacc -> d=0 weld)", bits, aw, t2);
        }
        {   // main-stack flips: link carry, tightness selector, OR-tree node
            vector<uint8_t> b2 = bits;
            size_t i = 0;
            while (!(b2[((size_t)bhw::LPR << lN) + i] & 1)) i++;
            b2[((size_t)(bhw::LPC + 1) << lN) + i] ^= 1;
            attack("flipped link carry bit (main stack) rejects", b2, aw, tw);
            b2 = bits;
            size_t j = 0;   // a present NON-achiever row: t there violates t*sh
            while (!(b2[((size_t)bhw::LPR << lN) + j] & 1) ||
                   !(b2[((size_t)bhw::LSH << lN) + j] & 1)) j++;
            b2[((size_t)bhw::LTSEL << lN) + j] ^= 1;
            attack("tightness selector on a non-achiever (main stack) rejects", b2, aw, tw);
            b2 = bits;
            b2[((size_t)bhw::LORT << lN) + btr::or_off(lN, 2) + 5] ^= 1;
            attack("flipped OR-tree level-2 node (main stack) rejects", b2, aw, tw);
        }
    }
    {   // WELD ATTACK 1: overstated max_exp with a FULLY CONSISTENT downstream
        // witness -- one group's shifts all bumped by 1, q/r/al recomputed,
        // trees and transition rebuilt from the tampered bits (link adder,
        // d-adder, both trees, reconciliation and normalize ALL consistent).
        // Only the tightness weld (t lands on a non-achiever, t*sh != 0) or
        // og/tacc can catch it.
        // bump the LAST group with products of some chain, so the (changed)
        // out-state only flows into all-absent padding groups -- the cascade
        // stays consistent without touching any other group's shifts
        long bg = -1;
        for (size_t c = 0; c * CH < (size_t)G.ng && bg < 0; c++) {
            long last = -1;
            bool room = true;
            for (int t = 0; t < CH; t++) {
                size_t g = c * CH + t;
                for (int kk = 0; kk < 32; kk++) {
                    size_t i = g * 32 + kk;
                    if (raw[5][i]) {
                        last = (long)g;
                        if (raw[6][i] + 1 > 62) room = false;
                    }
                }
            }
            if (last >= 0 && room && (uint32_t)G.MEZ[last] + 1 < 256) bg = last;
        }
        assert(bg >= 0);
        vector<int64_t> r2[10];
        for (int c = 0; c < 10; c++) r2[c] = raw[c];
        for (int kk = 0; kk < 32; kk++) {
            size_t i = (size_t)bg * 32 + kk;
            int64_t sh = r2[6][i] + 1;
            int64_t shc = sh < 15 ? sh : 15;
            int64_t q = r2[3][i] >> shc;
            r2[6][i] = sh; r2[7][i] = q;
            r2[8][i] = r2[3][i] - (q << shc);
            r2[9][i] = r2[5][i] ? (1 - 2 * r2[4][i]) * q : 0;
        }
        vector<uint8_t> b2;
        size_t nr2 = 0;
        int lN2 = bhw::bhw_build_bits(r2, b2, &nr2);
        assert(lN2 == lN);
        bacc::Wit a2;
        bacc::build(lN, b2, bhw::LALM, bhw::LALS, bhw::L5BASE, a2);
        btr::Wit t2;
        btr::build(lN, CH, b2, bhw::LEB, bhw::LPR, bhw::LSH, bhw::LTSEL,
                   bhw::LPC, bhw::LORT, bhw::LORP, bhw::L5BASE, t2, bg);
        attack("WELD 1: overstated max_exp, fully consistent downstream, "
               "rejects (tightness selector weld)", b2, a2, t2);
    }
    {   // WELD ATTACK 2: broken chain -- a non-head group's in-state tampered,
        // its whole row and every downstream group recomputed CONSISTENTLY
        // (all 9 zerochecks pass; only the shift-restriction pair sees it).
        // pick a non-head group with d >= 1: flipping nsI bit 0 then leaves
        // am = nsI >> d (and thus the whole row and out-state) unchanged --
        // zero cascade, every zerocheck still satisfied
        long ig = -1;
        for (size_t g = 0; g < (size_t)G.ng && ig < 0; g++)
            if (g % CH != 0 && tval(tw, btr::TNSI, g, 14) != 0 &&
                tval(tw, btr::TD, g, 8) >= 1) ig = (long)g;
        assert(ig >= 0);
        uint32_t isg = tval(tw, btr::TSGI, ig, 1), iae = tval(tw, btr::TAEI, ig, 8),
                 ins = tval(tw, btr::TNSI, ig, 14);
        btr::Wit t2;
        btr::build(lN, CH, bits, bhw::LEB, bhw::LPR, bhw::LSH, bhw::LTSEL,
                   bhw::LPC, bhw::LORT, bhw::LORP, bhw::L5BASE, t2, -1,
                   ig, isg, iae, ins ^ 1);
        attack("WELD 2: consistent transition over a BROKEN chain rejects "
               "(shift-restriction binding alone)", bits, aw, t2);
    }
    {   // WELD ATTACK 3: nonzero in-state at a chain HEAD.  ae=1 keeps the
        // group's max_exp product-dominated (MEZ >= 127) and d = MEZ-1 >= 14
        // zeroes the aligned acc -- the head row and everything downstream
        // stay bit-identical to honest, so ONLY the head point sees it.
        long hg = -1;                      // a head with present products
        for (size_t g = 0; g < (size_t)G.ng && hg < 0; g += CH)
            if (tval(tw, btr::TPG, g, 1)) hg = (long)g;
        assert(hg >= 0);
        btr::Wit t2;
        btr::build(lN, CH, bits, bhw::LEB, bhw::LPR, bhw::LSH, bhw::LTSEL,
                   bhw::LPC, bhw::LORT, bhw::LORP, bhw::L5BASE, t2, -1,
                   hg, 0, /*ae*/ 1, /*ns*/ 1u << 13);
        attack("WELD 3: nonzero head in-state, consistent downstream, rejects "
               "(head point binding alone)", bits, aw, t2);
    }

    {   // proof-object tampers
        auto q = pf; q.tr.root3[3] ^= 1;
        ck("tampered transition-stack root rejects", !bhw::bhw_verify(q));
        q = pf; q.tr.lk.finals[7].lo ^= 2;
        ck("tampered link finals reject", !bhw::bhw_verify(q));
        q = pf; q.tr.A.rounds[4].hi ^= 1;
        ck("tampered ZC-A round polynomial rejects", !bhw::bhw_verify(q));
        q = pf; q.tr.B.finals[1 + 3].lo ^= 4;
        ck("tampered ZC-B P-input final rejects (RTR derive)", !bhw::bhw_verify(q));
        q = pf; q.tr.C.finals[1 + 50].hi ^= 8;
        ck("tampered ZC-C finals reject (TC derive)", !bhw::bhw_verify(q));
        q = pf; q.tr.orsc[2].rounds[2].lo ^= 1;
        ck("tampered OR-tree round rejects", !bhw::bhw_verify(q));
        q = pf; q.tr.evMain[0][bhw::LMAG].lo ^= 1;
        ck("tampered supplied main eval at RTR rejects (opening)", !bhw::bhw_verify(q));
        q = pf; q.tr.evT[5][btr::TSGO].hi ^= 2;
        ck("tampered supplied eval at the head point rejects", !bhw::bhw_verify(q));
        q = pf; q.tr.evTS[0][btr::TSGI].lo ^= 1;
        ck("tampered chain-pair I eval rejects (equality or opening)",
           !bhw::bhw_verify(q));
        q = pf; q.tr.evTS[1][btr::NST + btr::TSGI].lo ^= 1;
        ck("tampered chain-pair O eval rejects (equality or opening)",
           !bhw::bhw_verify(q));
        q = pf;
        if (!q.tr.pcs3.t.empty() && !q.tr.pcs3.t[0].empty()) q.tr.pcs3.t[0][0].lo ^= 1;
        ck("tampered transition PCS opening rejects", !bhw::bhw_verify(q));
        q = pf; q.tr.on = 0;
        ck("stripping the transition from the proof rejects (stmt binding)",
           !bhw::bhw_verify(q));
        q = pf; q.tr.CH = CH * 2;
        ck("tampered chain length CH rejects (stmt binding)", !bhw::bhw_verify(q));
    }

    // ---- measured A/B, 262144 real products = 8192 groups = 256 chains ----
    vector<int64_t> rawb[10];
    TransGold GB;
    if (!load_trans("hawkeye_trans_big.bin", rawb, GB)) {
        printf("FATAL: hawkeye_trans_big.bin missing (see header comment)\n");
        return 1;
    }
    {
        size_t nb = 0;
        vector<uint8_t> bb;
        int lB = bhw::bhw_build_bits(rawb, bb, &nb);
        bacc::Wit ab;
        bacc::build(lB, bb, bhw::LALM, bhw::LALS, bhw::L5BASE, ab);
        btr::Wit tb;
        btr::build(lB, (int)GB.CH, bb, bhw::LEB, bhw::LPR, bhw::LSH, bhw::LTSEL,
                   bhw::LPC, bhw::LORT, bhw::LORP, bhw::L5BASE, tb);
        bool gold = true;
        for (size_t g = 0; g < (size_t)GB.ng; g++)
            if (tval(tb, btr::TMEZ, g, 8) != (uint32_t)GB.MEZ[g] ||
                tval(tb, btr::TSGO, g, 1) != (uint32_t)GB.SGO[g] ||
                tval(tb, btr::TAEO, g, 8) != (uint32_t)GB.AEO[g] ||
                tval(tb, btr::TNSO, g, 14) != (uint32_t)GB.NSO[g]) gold = false;
        ck("big witness: all 8192 golden max_exp + out-states bitwise", gold);
        bhw::BhwProof pb; bhw::BhwStats sb;
        bhw::bhw_prove(lB, bb, pb, sb, true, &ab, nullptr, &tb);
        bhw::BhwStats sv;
        ck("big composed proof (products + acc + transition) accepts",
           bhw::bhw_verify(pb, &sv));
        bhw::BhwProof pbh; bhw::BhwStats sbh;
        bhw::bhw_prove(lB, bb, pbh, sbh, false, &ab, nullptr, &tb);
        ck("big-witness GPU proof byte-identical to host proof", proofs_equal(pb, pbh));
        double bin_prove = sb.commit_ms + sb.lu_ms + sb.sc_ms + sb.open_ms +
                           sb.acc_ms + sb.tr_ms;
        printf("  [meas] BINIUS FULL lN=%d: committed %.2f MB | commit %.0f ms, "
               "lu %.0f ms, sc %.0f ms, acc %.0f ms, tr %.0f ms, open %.0f ms"
               " -> prove %.2f s | proof %.2f MB | verify %.0f ms\n",
               lB, sb.committed / 1048576.0, sb.commit_ms, sb.lu_ms, sb.sc_ms,
               sb.acc_ms, sb.tr_ms, sb.open_ms, bin_prove / 1000.0,
               pb.bytes() / 1048576.0, sv.verify_ms);

        // Goldilocks side: the per-product gadget on the same rows (proves
        // neither accumulation nor the transition)
        const uint32_t R = 2, Q = 24;
        p3hw::Tables T = p3hw::build_tables();
        double t0 = bhw::bhw_now_ms();
        p3hw::ProdWitness W = p3hw::build_witness(rawb);
        vector<p3lu::Col> C(p3hw::NCOL);
        for (int c = 0; c < p3hw::NCOL; c++) C[c] = p3lu::commit_col(W.col[c], R);
        double gl_commit = bhw::bhw_now_ms() - t0;
        size_t gl_bytes = 0;
        for (int c = 0; c < p3hw::NCOL; c++)
            gl_bytes += C[c].cw.size() * 8 + (2 * C[c].cw.size() - 1) * 32;
        t0 = bhw::bhw_now_ms();
        fs::Transcript tp("hw-prod");
        p3hw::ProdProof gpf = p3hw::prove(tp, W, C, T, R, Q);
        double gl_prove = bhw::bhw_now_ms() - t0;
        const char* why = "";
        fs::Transcript tv("hw-prod");
        ck("Goldilocks gadget accepts the same rows (A/B sanity)",
           p3hw::verify(tv, T, gpf, Q, R, &why));
        printf("  [meas] GL          lN=%d: committed %.2f MB | commit %.0f ms "
               "-> prove %.2f s (NO acc, NO transition)\n",
               lB, gl_bytes / 1048576.0, gl_commit, gl_prove / 1000.0);
        printf("  [meas] ratios: committed %.1fx | prove %.1fx (Binius proves "
               "strictly more)\n", gl_bytes / (double)sb.committed,
               gl_prove / bin_prove);
    }

    printf("\nBINIUS-TRANS: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
