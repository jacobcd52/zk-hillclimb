// Selftest + measured A/B for the Binius ACCUMULATION gadget (p3_binius_acc.cuh
// + the bhw integration, design doc section 21.10): pos/neg-split carry-save
// adder trees proving the per-group Hawkeye sums  contribution_g = sum al  on
// top of the full per-product gadget, all in one composed proof.
//
// Teeth run on REAL group-ordered golden vectors (hawkeye_acc_vectors.bin from
// hawkeye_ref.py --dumpacc, Triton-cross-checked, with golden per-group P/N/S
// sums): the tree witness reproduces every golden sum bitwise; honest accept;
// GPU/host byte-identical proofs; targeted witness attacks at several tree
// levels; and THE weld tooth -- a fully CONSISTENT adder tree computed over
// inputs that differ from the committed almag/alsg (every zerocheck passes,
// only the B_1 restriction binding can catch it) must reject.
//
// Measurement: hawkeye_acc_big.bin (262144 real products = 8192 groups,
// regenerable:  python3 -c "import numpy as np, hawkeye_ref as H; rng =
// np.random.default_rng(20260708); H.dump_acc_witness('hawkeye_acc_big.bin',
// [(rng.integers(0,256,(32,1024)).astype(np.uint8),
// rng.integers(0,256,(8,1024)).astype(np.uint8))])" ), A/B'd against the
// Goldilocks per-product gadget (R=2, Q=24).  Stated plainly: the GL gadget
// does NOT prove accumulation (the composed GL prover gets integer addition
// natively from the field), so the Binius side proves strictly MORE here.
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
static uint64_t rs_ = 0xacc0acc0acc0acc0ULL;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }

static bool load_acc(const char* path, vector<int64_t> raw[10],
                     vector<int64_t>& P, vector<int64_t>& Nn, vector<int64_t>& S) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t n = 0;
    if (fread(&n, 8, 1, f) != 1) { fclose(f); return false; }
    for (int c = 0; c < 10; c++) {
        raw[c].resize(n);
        if (fread(raw[c].data(), 8, n, f) != (size_t)n) { fclose(f); return false; }
    }
    int64_t magic = 0, ng = 0;
    if (fread(&magic, 8, 1, f) != 1 || magic != 0x41434332 ||
        fread(&ng, 8, 1, f) != 1) { fclose(f); return false; }
    P.resize(ng); Nn.resize(ng); S.resize(ng);
    bool ok = fread(P.data(), 8, ng, f) == (size_t)ng &&
              fread(Nn.data(), 8, ng, f) == (size_t)ng &&
              fread(S.data(), 8, ng, f) == (size_t)ng;
    fclose(f);
    return ok;
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
    if (x.acc.on != y.acc.on) return false;
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
    return true;
}

// decode the level-l committed value of (side, node y) from the bit arrays
static int64_t decode_val(int lN, int l, int side, size_t y,
                          const vector<uint8_t>& mb, const vector<uint8_t>& ab) {
    int win = bacc::A_WIN[l];
    const uint8_t* src = (l == bacc::ACCL) ? mb.data() : ab.data();
    int64_t v = 0;
    for (int b = 0; b <= win; b++) {           // out col side*(win+1)+b = bit b
        int slot, t;
        bacc::acc_slot(l, 0, side * (win + 1) + b, bhw::L5BASE, slot, t);
        v |= (int64_t)(src[((size_t)slot << lN) + ((size_t)t << (lN - l)) + y] & 1) << b;
    }
    return v;
}

int main() {
    printf("=== Binius accumulation gadget: pos/neg carry-save adder trees "
           "(section 21.10) ===\n");
#ifndef __CUDA_ARCH__
    bf16_tab();
#endif
    #pragma omp parallel
    { volatile int wa = omp_get_thread_num(); (void)wa; }

    vector<int64_t> raw[10], gP, gN, gS;
    if (!load_acc("hawkeye_acc_vectors.bin", raw, gP, gN, gS)) {
        printf("FATAL: hawkeye_acc_vectors.bin missing (python3 hawkeye_ref.py "
               "--dumpacc hawkeye_acc_vectors.bin)\n");
        return 1;
    }
    size_t n_real = 0;
    vector<uint8_t> bits;
    int lN = bhw::bhw_build_bits(raw, bits, &n_real);
    size_t N = (size_t)1 << lN;
    printf("golden vectors: %zu products (%zu groups) -> N=%zu (lN=%d)\n",
           n_real, gP.size(), N, lN);

    ck("per-product witness still validates in group order (0 violated rows)",
       bhw::bhw_validate(bits, lN) == 0);

    bacc::Wit aw;
    bacc::build(lN, bits, bhw::LALM, bhw::LALS, bhw::L5BASE, aw);

    { // the tree reproduces every golden group sum bitwise
        bool okP = true, okN = true, okS = true, okD = true;
        for (size_t g = 0; g < N >> 5; g++) {
            int64_t p = g < gP.size() ? gP[g] : 0;
            int64_t nn = g < gP.size() ? gN[g] : 0;
            if ((int64_t)aw.sumP[g] != p) okP = false;
            if ((int64_t)aw.sumN[g] != nn) okN = false;
            if (g < gP.size() && p - nn != gS[g]) okS = false;
            if (decode_val(lN, 5, 0, g, bits, aw.bits) != p ||
                decode_val(lN, 5, 1, g, bits, aw.bits) != nn) okD = false;
        }
        ck("level-5 P sums == golden positive sums (all groups, bitwise)", okP);
        ck("level-5 N sums == golden negative sums (all groups, bitwise)", okN);
        ck("golden P - N == golden contribution S (dump self-consistency)", okS);
        ck("committed level-5 bit-slices decode to the sums (packing check)", okD);
    }

    { // level functors: zero on honest rows at every level, nonzero after a flip
        bool allz = true;
        for (int l = 1; l <= 5 && allz; l++) {
            std::vector<uint8_t> sb;
            bacc::acc_sc_bytes(lN, l, bits.data(), bhw::LALM, bhw::LALS,
                               bhw::L5BASE, aw.bits.data(), sb);
            int win = bacc::A_WIN[l], ni2 = bacc::acc_ni2(l);
            int KM1 = 2 * ni2 + 4 * win;
            size_t half = N >> l;
            vector<bf128_t> g(4 * win), w(1 + KM1);
            for (auto& x : g) { x.lo = rnd(); x.hi = rnd(); }
            auto evalrow = [&](size_t y) {
                w[0] = bf128_one();
                for (int k = 0; k < KM1; k++)
                    w[1 + k] = sb[(size_t)k * half + y] ? bf128_one() : bf128_zero();
                switch (l) {
                case 1: { bacc::AccF<1> c; c.g = g.data(); return c(w.data()); }
                case 2: { bacc::AccF<2> c; c.g = g.data(); return c(w.data()); }
                case 3: { bacc::AccF<3> c; c.g = g.data(); return c(w.data()); }
                case 4: { bacc::AccF<4> c; c.g = g.data(); return c(w.data()); }
                default: { bacc::AccF<5> c; c.g = g.data(); return c(w.data()); }
                }
            };
            for (int t = 0; t < 400; t++)
                if (!bf128_eq(evalrow(rnd() % half), bf128_zero())) allz = false;
            size_t y = rnd() % half;
            sb[(size_t)(2 * ni2 + win / 2) * half + y] ^= 1;   // flip a sum bit
            if (bf128_eq(evalrow(y), bf128_zero())) allz = false;
        }
        ck("level constraints == 0 on honest rows, != 0 after a flipped bit "
           "(all 5 levels)", allz);
    }

    // ---- honest proof + GPU/host identity + non-acc regression ----
    bhw::BhwProof pf; bhw::BhwStats st;
    bhw::bhw_prove(lN, bits, pf, st, true, &aw);
    ck("honest composed proof (products + accumulation) accepts (GPU)",
       bhw::bhw_verify(pf));
    {
        bhw::BhwProof ph; bhw::BhwStats sh;
        bhw::bhw_prove(lN, bits, ph, sh, false, &aw);
        ck("GPU proof byte-identical to host proof (incl. all 5 acc levels + "
           "acc opening)", proofs_equal(pf, ph));
    }
    {
        bhw::BhwProof p0; bhw::BhwStats s0;
        bhw::bhw_prove(lN, bits, p0, s0, true);
        ck("acc=off regression: per-product proof on the group-ordered witness "
           "accepts", bhw::bhw_verify(p0));
    }

    // ---- witness attacks ----
    auto attack = [&](const char* what, const vector<uint8_t>& mb2, const bacc::Wit& a2,
                      const uint8_t* acc_in = nullptr) {
        bhw::BhwProof q; bhw::BhwStats s2;
        bhw::bhw_prove(lN, mb2, q, s2, true, &a2, acc_in);
        ck(what, !bhw::bhw_verify(q));
    };
    { // flipped level-1 sum bit (acc stack)
        bacc::Wit a2 = aw;
        int slot, t;
        bacc::acc_slot(1, 0, 3, bhw::L5BASE, slot, t);
        a2.bits[((size_t)slot << lN) + ((size_t)t << (lN - 1)) + 17] ^= 1;
        attack("flipped level-1 sum bit rejects", bits, a2);
    }
    { // flipped level-1 INTERNAL carry bit (acc stack)
        bacc::Wit a2 = aw;
        int slot, t;
        bacc::acc_slot(1, 1, 2, bhw::L5BASE, slot, t);
        a2.bits[((size_t)slot << lN) + ((size_t)t << (lN - 1)) + 40] ^= 1;
        attack("flipped level-1 internal carry bit rejects", bits, a2);
    }
    { // flipped level-3 sum bit
        bacc::Wit a2 = aw;
        int slot, t;
        bacc::acc_slot(3, 0, 7, bhw::L5BASE, slot, t);
        a2.bits[((size_t)slot << lN) + ((size_t)t << (lN - 3)) + 9] ^= 1;
        attack("flipped level-3 sum bit rejects", bits, a2);
    }
    { // flipped level-5 group-sum bit (MAIN stack)
        vector<uint8_t> b2 = bits;
        int slot, t;
        bacc::acc_slot(5, 0, 4, bhw::L5BASE, slot, t);
        b2[((size_t)slot << lN) + ((size_t)t << (lN - 5)) + 3] ^= 1;
        attack("flipped level-5 group-sum bit (main stack) rejects", b2, aw);
    }
    { // THE WELD TOOTH: a fully consistent adder tree over TAMPERED inputs --
      // flip one committed almag bit only in the tree's view, rebuild every
      // level honestly from it (all 5 zerochecks pass; level-5 slots grafted
      // into the committed main stack so A_5 binds too).  Only the B_1
      // restriction binding against the real almag/alsg columns catches it.
        long i = -1;
        for (size_t k = 0; k < n_real; k++)
            if (raw[5][k]) { i = (long)k; break; }
        vector<uint8_t> src = bits;
        src[(size_t)(bhw::LALM + 1) * N + i] ^= 1;
        bacc::Wit a2;
        bacc::build(lN, src, bhw::LALM, bhw::LALS, bhw::L5BASE, a2);
        vector<uint8_t> mb2 = bits;               // committed: REAL products...
        for (int s = bhw::L5BASE; s < bhw::L5BASE + 4; s++)   // ...tampered sums
            memcpy(mb2.data() + ((size_t)s << lN), src.data() + ((size_t)s << lN), N);
        attack("CONSISTENT tree over tampered inputs rejects (B_1 weld binding)",
               mb2, a2, src.data());
    }
    { // proof-object tampers
        auto q = pf; q.acc.root2[7] ^= 1;
        ck("tampered acc-stack root rejects", !bhw::bhw_verify(q));
        q = pf; q.acc.sc[1].finals[5].lo ^= 2;
        ck("tampered level-2 sumcheck finals reject", !bhw::bhw_verify(q));
        q = pf; q.acc.sc[4].rounds[3].hi ^= 1;
        ck("tampered level-5 round polynomial rejects", !bhw::bhw_verify(q));
        q = pf; q.acc.evB1[bhw::LSG].lo ^= 1;     // an UNCOVERED supplied eval
        ck("tampered supplied eval at B_1 rejects (opening)", !bhw::bhw_verify(q));
        q = pf; q.acc.evA[2][50].hi ^= 4;
        ck("tampered supplied eval at A_3 rejects (opening)", !bhw::bhw_verify(q));
        q = pf; q.acc.evB[3][10].lo ^= 8;
        ck("tampered supplied eval at B_5 rejects (opening)", !bhw::bhw_verify(q));
        q = pf; q.acc.pcs2.t[2][9].lo ^= 1;
        ck("tampered acc PCS opening rejects", !bhw::bhw_verify(q));
        q = pf; q.acc.on = 0;
        ck("stripping the accumulation from the proof rejects (stmt binding)",
           !bhw::bhw_verify(q));
    }

    // ---- measured A/B, 262144 real products (8192 groups) ----
    vector<int64_t> rawb[10], bP, bN, bS;
    if (!load_acc("hawkeye_acc_big.bin", rawb, bP, bN, bS)) {
        printf("FATAL: hawkeye_acc_big.bin missing (see header comment)\n");
        return 1;
    }
    {
        size_t nb = 0;
        vector<uint8_t> bb;
        int lB = bhw::bhw_build_bits(rawb, bb, &nb);
        bacc::Wit ab;
        bacc::build(lB, bb, bhw::LALM, bhw::LALS, bhw::L5BASE, ab);
        bool sums = true;
        for (size_t g = 0; g < bP.size(); g++)
            if ((int64_t)ab.sumP[g] != bP[g] || (int64_t)ab.sumN[g] != bN[g])
                sums = false;
        ck("big witness: all 8192 golden group sums reproduced bitwise", sums);
        bhw::BhwProof pb; bhw::BhwStats sb;
        bhw::bhw_prove(lB, bb, pb, sb, true, &ab);
        bhw::BhwStats sv;
        ck("big composed proof accepts (GPU prover)", bhw::bhw_verify(pb, &sv));
        bhw::BhwProof pbh; bhw::BhwStats sbh;
        bhw::bhw_prove(lB, bb, pbh, sbh, false, &ab);
        ck("big-witness GPU proof byte-identical to host proof", proofs_equal(pb, pbh));
        double bin_prove = sb.commit_ms + sb.lu_ms + sb.sc_ms + sb.open_ms + sb.acc_ms;
        printf("  [meas] BINIUS+ACC lN=%d: committed %.2f MB | commit %.0f ms, "
               "lu %.0f ms, sc %.0f ms, acc %.0f ms (host A/B %.0f ms), open %.0f ms"
               " -> prove %.0f ms | proof %.2f MB | verify %.0f ms\n",
               lB, sb.committed / 1048576.0, sb.commit_ms, sb.lu_ms, sb.sc_ms,
               sb.acc_ms, sbh.acc_ms, sb.open_ms, bin_prove,
               pb.bytes() / 1048576.0, sv.verify_ms);

        // Goldilocks side: the per-product gadget on the same rows (it proves
        // NO accumulation -- the Binius side proves strictly more here)
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
        printf("  [meas] GL         lN=%d: committed %.2f MB | commit %.0f ms, "
               "prove %.0f ms (accumulation NOT proven on this side)\n",
               lB, gl_bytes / 1048576.0, gl_commit, gl_prove);
        printf("  [meas] ratios (Binius = per-product + DM + ACCUMULATION vs GL "
               "per-product + DM): committed-data %.1fx | prove %.0f/%.0f ms = %.1fx\n",
               (double)gl_bytes / sb.committed, gl_prove, bin_prove,
               gl_prove / bin_prove);
    }

    printf("\nBINIUS-ACC: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
