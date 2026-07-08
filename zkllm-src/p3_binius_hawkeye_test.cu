// Selftest + measured A/B for the Binius Hawkeye per-product alignment gadget
// (p3_binius_hawkeye.cuh, design doc section 21.8).
//
// Teeth run on the REAL golden vectors (hawkeye_prod_vectors.bin, from
// hawkeye_ref.py --dump, Triton-cross-checked): honest accept, GPU/host
// byte-identical proofs, and one targeted attack per constraint family --
// including the two classic Goldilocks-gadget attacks (q-1 with r absorbed,
// doubled shift with halved q), which are caught here by the STRUCTURAL bit
// ranges and the sh<->h linkage instead of the REM/SH lookups.
//
// Measurement runs on hawkeye_prod_big.bin (262144 real products, random fp8
// codes through the same numpy replay) and A/Bs against the REAL Goldilocks
// gadget: p3hw::build_witness + commit_col x11 + p3hw::prove (R=2, Q=24, the
// prod-test parameters), reporting committed bytes and prove time, plus the
// DM-lookup-only share (the one piece the Binius side has NOT yet ported).
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
static uint64_t rs_ = 0xb1a5b1a5b1a5b1a5ULL;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }

static bool load_vectors(const char* path, vector<int64_t> raw[10]) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t n = 0;
    if (fread(&n, 8, 1, f) != 1) { fclose(f); return false; }
    for (int c = 0; c < 10; c++) {
        raw[c].resize(n);
        if (fread(raw[c].data(), 8, n, f) != (size_t)n) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

// replicate the HwF gamma-index walk to pin the batched-constraint count
static int count_gammas() {
    int c = 0;
    for (int s = 0; s < 16; s++) for (int t = s + 1; t < 16; t++) c++;   // one-hot pairs
    c += 1 + 4 + 1 + 4;                    // parity, helpers, g-link, min-bits
    c += 15;                               // shift-mux
    for (int s = 0; s < 16; s++) for (int j = s; j < 15; j++) c++;       // r range
    for (int s = 0; s < 16; s++) for (int j = 15 - s; j < 15; j++) c++;  // q range
    c += 15 + 1;                           // C2 magnitude + sign
    return c;
}

// row of the bit witness -> T_128 vector (w[0]=1 stands in for eq)
static bf128_t row_eval(const vector<uint8_t>& bits, size_t N, size_t i,
                        const vector<bf128_t>& g) {
    vector<bf128_t> w(bhw::HwF::K, bf128_zero());
    w[0] = bf128_one();
    for (int c = 0; c < bhw::NCON; c++)
        w[1 + c] = bits[(size_t)c * N + i] ? bf128_one() : bf128_zero();
    bhw::HwF cf; cf.g = g.data();
    return cf(w.data());
}

static bool proofs_equal(const bhw::BhwProof& x, const bhw::BhwProof& y) {
    return !memcmp(x.root, y.root, 32) &&
           x.sc.rounds.size() == y.sc.rounds.size() &&
           !memcmp(x.sc.rounds.data(), y.sc.rounds.data(),
                   x.sc.rounds.size() * sizeof(bf128_t)) &&
           !memcmp(x.sc.finals.data(), y.sc.finals.data(),
                   x.sc.finals.size() * sizeof(bf128_t)) &&
           !memcmp(x.xev, y.xev, sizeof x.xev) &&
           x.pcs.t.size() == y.pcs.t.size() &&
           !memcmp(x.pcs.t.data(), y.pcs.t.data(), x.pcs.t.size() * sizeof(bf128_t)) &&
           !memcmp(x.pcs.u.data(), y.pcs.u.data(), x.pcs.u.size() * sizeof(bf128_t)) &&
           x.pcs.cols == y.pcs.cols && x.pcs.paths == y.pcs.paths;
}

// rewrite the 15 bits of a value column on one row
static void set_val(vector<uint8_t>& bits, size_t N, int base, size_t i, int64_t v) {
    for (int j = 0; j < 15; j++) bits[(size_t)(base + j) * N + i] = (v >> j) & 1;
}

int main() {
    printf("=== Binius Hawkeye per-product alignment gadget (section 21.8) ===\n");
#ifndef __CUDA_ARCH__
    bf16_tab();
#endif
    #pragma omp parallel
    { volatile int wa = omp_get_thread_num(); (void)wa; }

    ck("gamma-batched constraint count == NC", count_gammas() == bhw::NC);

    vector<int64_t> raw[10];
    if (!load_vectors("hawkeye_prod_vectors.bin", raw)) {
        printf("FATAL: hawkeye_prod_vectors.bin missing (python3 hawkeye_ref.py --dump)\n");
        return 1;
    }
    size_t n_real = 0;
    vector<uint8_t> bits;
    int lN = bhw::bhw_build_bits(raw, bits, &n_real);
    size_t N = (size_t)1 << lN;
    printf("golden vectors: %zu products -> N=%zu (lN=%d), %d bit-columns\n",
           n_real, N, lN, (int)bhw::NCOLB);

    ck("bit-witness validator: 0 violated rows on the real vectors",
       bhw::bhw_validate(bits, lN) == 0);

    { // per-row functor: zero on honest rows, nonzero after a bit flip
        vector<bf128_t> g(bhw::NC);
        for (auto& x : g) { x.lo = rnd(); x.hi = rnd(); }
        bool allz = true;
        for (int t = 0; t < 2000; t++)
            if (!bf128_eq(row_eval(bits, N, rnd() % N, g), bf128_zero())) allz = false;
        ck("constraint functor == 0 on 2000 random real rows", allz);
        auto b2 = bits;
        size_t i = 0; while (!raw[5][i]) i++;                    // a present row
        b2[(size_t)(bhw::LMAG + 4) * N + i] ^= 1;
        ck("constraint functor != 0 after one flipped mag bit",
           !bf128_eq(row_eval(b2, N, i, g), bf128_zero()));
    }

    // ---- honest proof + GPU/host identity ----
    bhw::BhwProof pf; bhw::BhwStats st;
    bhw::bhw_prove(lN, bits, pf, st);
    ck("honest real-vector witness accepts (GPU prover)", bhw::bhw_verify(pf));
    {
        bhw::BhwProof ph; bhw::BhwStats sh;
        bhw::bhw_prove(lN, bits, ph, sh, false);
        ck("GPU proof byte-identical to host proof (root+rounds+finals+xev+opening)",
           proofs_equal(pf, ph));
    }

    // ---- targeted attacks (one per constraint family) ----
    auto attack = [&](const char* what, vector<uint8_t>& b2) {
        bhw::BhwProof q; bhw::BhwStats s2;
        bhw::bhw_prove(lN, b2, q, s2);
        ck(what, !bhw::bhw_verify(q));
    };
    auto find_row = [&](auto pred) -> long {
        for (size_t i = 0; i < n_real; i++) if (pred(i)) return (long)i;
        return -1;
    };
    { // classic attack 1: q-1 with r absorbed (q*pw+r=mag still true as integers)
        long i = find_row([&](size_t i) {
            return raw[6][i] >= 1 && raw[6][i] <= 14 && raw[7][i] >= 1; });
        auto b2 = bits;
        int s = (int)raw[6][i];
        set_val(b2, N, bhw::LQ, i, raw[7][i] - 1);
        set_val(b2, N, bhw::LR, i, raw[8][i] + (1ll << s));
        set_val(b2, N, bhw::LALM, i, raw[5][i] ? raw[7][i] - 1 : 0);
        attack("q-1 / r+pw attack rejects (structural r-range, no REM lookup)", b2);
    }
    { // classic attack 2: doubled shift, halved q (even q so the mux still holds)
        long i = find_row([&](size_t i) {
            return raw[6][i] >= 1 && raw[6][i] <= 13 && raw[7][i] >= 2 && raw[7][i] % 2 == 0; });
        auto b2 = bits;
        int s = (int)raw[6][i];
        b2[(size_t)(bhw::LH + s) * N + i] = 0;
        b2[(size_t)(bhw::LH + s + 1) * N + i] = 1;
        set_val(b2, N, bhw::LQ, i, raw[7][i] / 2);
        set_val(b2, N, bhw::LALM, i, raw[5][i] ? raw[7][i] / 2 : 0);
        attack("doubled-shift/halved-q attack rejects (sh<->h linkage, no SH lookup)", b2);
    }
    { // one-hot: a second h bit
        auto b2 = bits;
        size_t i = 5;
        int s = (int)(raw[6][i] < 15 ? raw[6][i] : 15);
        b2[(size_t)(bhw::LH + ((s + 3) & 15)) * N + i] = 1;
        attack("two h bits set on one row rejects (one-hot pairwise)", b2);
    }
    { // one-hot: no h bit
        auto b2 = bits;
        size_t i = 7;
        int s = (int)(raw[6][i] < 15 ? raw[6][i] : 15);
        b2[(size_t)(bhw::LH + s) * N + i] = 0;
        attack("all h bits clear on one row rejects (parity)", b2);
    }
    { // mux: single flipped mag bit
        auto b2 = bits;
        b2[(size_t)(bhw::LMAG + 6) * N + 123] ^= 1;
        attack("one flipped mag bit rejects (shift-mux)", b2);
    }
    { // C2: almag != 0 on an absent (pr=0) product
        long i = find_row([&](size_t i) { return raw[5][i] == 0; });
        auto b2 = bits;
        b2[(size_t)(bhw::LALM + 2) * N + i] ^= 1;
        attack("nonzero almag on a pr=0 row rejects (C2 magnitude)", b2);
    }
    { // C2: almag != 0 on a big-shift row (sh>=15 -> q=0 -> al=0)
        long i = find_row([&](size_t i) { return raw[6][i] >= 15 && raw[5][i]; });
        auto b2 = bits;
        b2[(size_t)(bhw::LALM + 1) * N + i] ^= 1;
        attack("nonzero almag on a sh>=15 row rejects (C2, al=0 branch)", b2);
    }
    { // C2 sign
        long i = find_row([&](size_t i) { return raw[5][i] != 0; });
        auto b2 = bits;
        b2[(size_t)bhw::LALS * N + i] ^= 1;
        attack("flipped al sign bit rejects (C2 sign)", b2);
    }
    { // proof-object tampers
        auto q = pf; q.sc.finals[9].lo ^= 4;
        ck("tampered sumcheck finals reject", !bhw::bhw_verify(q));
        q = pf; q.xev[3].hi ^= 1;
        ck("tampered committed-only column eval (a/b/eb binding) rejects",
           !bhw::bhw_verify(q));
        q = pf; q.root[11] ^= 1;
        ck("tampered commitment root rejects", !bhw::bhw_verify(q));
        q = pf; q.pcs.t[5].lo ^= 2;
        ck("tampered PCS opening rejects", !bhw::bhw_verify(q));
    }

    // ---- measured A/B vs the REAL Goldilocks gadget, 262144 real products ----
    vector<int64_t> rawb[10];
    if (!load_vectors("hawkeye_prod_big.bin", rawb)) {
        printf("FATAL: hawkeye_prod_big.bin missing (see design doc section 21.8)\n");
        return 1;
    }
    {
        size_t nb = 0;
        vector<uint8_t> bb;
        int lB = bhw::bhw_build_bits(rawb, bb, &nb);
        ck("big witness (262144 real products) validates", bhw::bhw_validate(bb, lB) == 0);
        bhw::BhwProof pb; bhw::BhwStats sb;
        bhw::bhw_prove(lB, bb, pb, sb);
        bhw::BhwStats sv;
        ck("big witness accepts (GPU prover)", bhw::bhw_verify(pb, &sv));
        bhw::BhwProof pbh; bhw::BhwStats sbh;
        bhw::bhw_prove(lB, bb, pbh, sbh, false);
        ck("big-witness GPU proof byte-identical to host proof", proofs_equal(pb, pbh));
        double bin_prove = sb.commit_ms + sb.sc_ms + sb.open_ms;
        printf("  [meas] BINIUS  lN=%d: committed %.2f MB | commit %.0f ms, sc %.0f ms "
               "(host A/B %.0f ms), open %.0f ms -> prove %.0f ms | proof %.2f MB | "
               "verify %.0f ms\n",
               lB, sb.committed / 1048576.0, sb.commit_ms, sb.sc_ms, sbh.sc_ms,
               sb.open_ms, bin_prove, pb.bytes() / 1048576.0, sv.verify_ms);

        // Goldilocks side: the production gadget on the same vectors
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
        ck("Goldilocks gadget accepts the same vectors (A/B sanity)",
           p3hw::verify(tv, T, gpf, Q, R, &why));
        // DM-lookup-only share (the piece the Binius gadget has NOT ported yet)
        t0 = bhw::bhw_now_ms();
        {
            fs::Transcript td("hw-dm-only");
            p3lu::LookupProof dm = p3lu::prove(td,
                {&C[p3hw::CA], &C[p3hw::CB], &C[p3hw::CEB], &C[p3hw::CMAG],
                 &C[p3hw::CSG], &C[p3hw::CPR]},
                W.idxDM, T.DM, R, Q, "hwDM", true, true);
            (void)dm;
        }
        double gl_dm = bhw::bhw_now_ms() - t0;
        printf("  [meas] GL      lN=%d: committed %.2f MB (11 witness cols only, "
               "lookup aux excluded) | commit %.0f ms, prove %.0f ms (DM lookup alone "
               "%.0f ms; Binius covers everything but DM)\n",
               lB, gl_bytes / 1048576.0, gl_commit, gl_prove, gl_dm);
        printf("  [meas] ratios: committed-data %.1fx  commit-time %.1fx  "
               "prove(GL total)/%.0fms=%.1fx  prove(GL excl DM)=%.1fx\n",
               (double)gl_bytes / sb.committed, gl_commit / sb.commit_ms,
               bin_prove, gl_prove / bin_prove, (gl_prove - gl_dm) / bin_prove);
    }

    printf("\nBINIUS-HAWKEYE: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
