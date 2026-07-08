// Selftest with teeth for logUp over the binary tower (p3_binius_logup.cuh,
// design doc section 21.9).
//
// The star tooth is the CHAR-2 SOUNDNESS DEMONSTRATION: a witness containing
// the same out-of-table tuple TWICE satisfies the additive logUp identity
// sum 1/(alpha+v) == sum m/(alpha+t) over GF(2^128) EXACTLY (even
// multiplicities XOR-cancel) -- asserted here by direct field computation --
// while the multiplicative argument this file tests rejects it.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>
#include "p3_binius_logup.cuh"

using std::vector;

static int np_ = 0, nf_ = 0;
static void ck(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) np_++; else nf_++;
}
static uint64_t rs_ = 0x1009e15c0de5eedULL;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }

// ---- toy lookup instance: J bit-columns, table 2^LT rows, witness 2^LN ----
// (LN=14 / LT=10 put both grand-product chains past the device-tree
// threshold, so the GPU/host byte-identity tooth exercises the GPU path)
static const int LN = 14, LT = 10, J = 10, LOGJS = 4, JS = 16;
static const size_t N = (size_t)1 << LN, TN = (size_t)1 << LT;

struct Inst {
    vector<uint8_t> tbits;        // J x TN, col-major (public)
    vector<uint8_t> wstk;         // JS x N stacked witness bits (cols 0..J-1 used)
    vector<uint32_t> idx;         // table key per row
};
static Inst make_inst() {
    Inst I;
    I.tbits.assign((size_t)J * TN, 0);
    for (size_t x = 0; x < I.tbits.size(); x++) I.tbits[x] = rnd() & 1;
    I.idx.resize(N);
    I.wstk.assign((size_t)JS * N, 0);
    for (size_t i = 0; i < N; i++) {
        uint32_t j = (uint32_t)(rnd() & (TN - 1));
        I.idx[i] = j;
        for (int k = 0; k < J; k++)
            I.wstk[(size_t)k * N + i] = I.tbits[(size_t)k * TN + j];
    }
    return I;
}

// ---- full protocol around the lookup: stacked witness commitment + the
// caller-side column-eval binding via one opening at rfin_w ----
struct Proof {
    uint8_t wroot[32];
    bflu::BfLuProof lu;
    bf128_t xev[J];
    BfPcsProof open;
};
static BfPcsParams wparams() {
    BfPcsParams p;
    p.l = LN + LOGJS; p.lcol = p.l / 2; p.lrow = p.l - p.lcol; p.Q = 100;
    return p;
}
static void prove(const Inst& I, Proof& pf, bool gpu,
                  const uint32_t* idx_override = nullptr,
                  const uint8_t* mcommit_override = nullptr,
                  const uint8_t* mleaf_override = nullptr) {
    fs::Transcript tr("bflu-test");
    int stmt[3] = {LN, LT, J};
    tr.absorb("stmt", stmt, sizeof stmt);
    BfPcsParams p = wparams();
    BfPcsCommit C;
    bfpcs_commit(p, I.wstk.data(), tr, C);
    memcpy(pf.wroot, C.root, 32);
    const uint8_t* wc[J];
    for (int k = 0; k < J; k++) wc[k] = I.wstk.data() + (size_t)k * N;
    vector<bf128_t> rfin_w;
    bflu::bflu_prove(tr, LN, LT, J, wc, idx_override ? idx_override : I.idx.data(),
                     I.tbits.data(), pf.lu, rfin_w, gpu,
                     mcommit_override, mleaf_override);
    // column evals at rfin_w + one opening binding them to the commitment
    std::vector<bf128_t> eqz;
    bf_eq_table(rfin_w.data(), LN, eqz);
    for (int k = 0; k < J; k++) {
        bf128_t acc = bf128_zero();
        for (size_t i = 0; i < N; i++)
            if (wc[k][i] & 1) acc = bf128_add(acc, eqz[i]);
        pf.xev[k] = acc;
    }
    tr.absorb("lu-xev", pf.xev, sizeof pf.xev);
    std::vector<bf128_t> rho(LOGJS), eqsel, R(p.l);
    for (auto& x : rho) x = bf_chal128(tr);
    bf_eq_table(rho.data(), LOGJS, eqsel);
    bf128_t V = bf128_zero();
    for (int k = 0; k < J; k++) V = bf128_add(V, bf128_mul(eqsel[k], pf.xev[k]));
    for (int t = 0; t < LN; t++) R[t] = rfin_w[t];
    for (int t = 0; t < LOGJS; t++) R[LN + t] = rho[t];
    bfpcs_open(C, R.data(), V, tr, pf.open);
}
static bool verify(const Inst& I, const Proof& pf, const char** why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    fs::Transcript tr("bflu-test");
    int stmt[3] = {LN, LT, J};
    tr.absorb("stmt", stmt, sizeof stmt);
    tr.absorb("bfpcs-root", pf.wroot, 32);
    vector<bf128_t> rfin_w, bk; bf128_t cw, alpha;
    if (!bflu::bflu_verify(tr, LN, LT, J, I.tbits.data(), pf.lu, rfin_w, cw,
                           alpha, bk, why)) return false;
    // the leaf-claim binding: cw == alpha + sum bk[k]*colk~(rfin_w)
    bf128_t exp = alpha;
    for (int k = 0; k < J; k++) exp = bf128_add(exp, bf128_mul(bk[k], pf.xev[k]));
    if (!bf128_eq(cw, exp)) return fail("witness leaf binding");
    tr.absorb("lu-xev", pf.xev, sizeof pf.xev);
    BfPcsParams p = wparams();
    std::vector<bf128_t> rho(LOGJS), eqsel, R(p.l);
    for (auto& x : rho) x = bf_chal128(tr);
    bf_eq_table(rho.data(), LOGJS, eqsel);
    bf128_t V = bf128_zero();
    for (int k = 0; k < J; k++) V = bf128_add(V, bf128_mul(eqsel[k], pf.xev[k]));
    for (int t = 0; t < LN; t++) R[t] = rfin_w[t];
    for (int t = 0; t < LOGJS; t++) R[LN + t] = rho[t];
    if (!bfpcs_verify(p, pf.wroot, R.data(), V, tr, pf.open))
        return fail("witness opening");
    if (why) *why = "ok";
    return true;
}

static bool sc_equal(const BfScProof& a, const BfScProof& b) {
    return a.rounds.size() == b.rounds.size() && a.finals.size() == b.finals.size() &&
           !memcmp(a.rounds.data(), b.rounds.data(), a.rounds.size() * sizeof(bf128_t)) &&
           !memcmp(a.finals.data(), b.finals.data(), a.finals.size() * sizeof(bf128_t));
}
static bool gkr_equal(const bflu::BfGkrProof& a, const bflu::BfGkrProof& b) {
    if (!bf128_eq(a.root, b.root) || !bf128_eq(a.g0, b.g0) || !bf128_eq(a.g1, b.g1) ||
        a.lay.size() != b.lay.size()) return false;
    for (size_t i = 0; i < a.lay.size(); i++)
        if (!sc_equal(a.lay[i], b.lay[i])) return false;
    return true;
}
static bool proofs_equal(const Proof& a, const Proof& b) {
    return !memcmp(a.wroot, b.wroot, 32) && !memcmp(a.lu.mroot, b.lu.mroot, 32) &&
           gkr_equal(a.lu.gw, b.lu.gw) && gkr_equal(a.lu.gt, b.lu.gt) &&
           sc_equal(a.lu.bind, b.lu.bind) &&
           !memcmp(a.xev, b.xev, sizeof a.xev) &&
           a.open.cols == b.open.cols && a.open.paths == b.open.paths &&
           a.lu.mopen.cols == b.lu.mopen.cols && a.lu.mopen.paths == b.lu.mopen.paths &&
           a.open.t.size() == b.open.t.size() &&
           !memcmp(a.open.t.data(), b.open.t.data(), a.open.t.size() * sizeof(bf128_t)) &&
           !memcmp(a.open.u.data(), b.open.u.data(), a.open.u.size() * sizeof(bf128_t)) &&
           a.lu.mopen.t.size() == b.lu.mopen.t.size() &&
           !memcmp(a.lu.mopen.t.data(), b.lu.mopen.t.data(),
                   a.lu.mopen.t.size() * sizeof(bf128_t)) &&
           !memcmp(a.lu.mopen.u.data(), b.lu.mopen.u.data(),
                   a.lu.mopen.u.size() * sizeof(bf128_t));
}

int main() {
    printf("=== Binius logUp over towers (section 21.9) ===\n");
#ifndef __CUDA_ARCH__
    bf16_tab();
#endif
    #pragma omp parallel
    { volatile int wa = omp_get_thread_num(); (void)wa; }

    Inst I = make_inst();
    printf("instance: witness 2^%d rows, table 2^%d rows, %d tuple bit-columns, "
           "MBP=%d mult-bit slices\n", LN, LT, J, 1 << bflu::bflu_lmb(LN));

    // ---- grand-product GKR unit checks ----
    {
        vector<bf128_t> lv(256);
        bf128_t prod = bf128_one();
        for (auto& x : lv) { x.lo = rnd() | 1; x.hi = rnd(); prod = bf128_mul(prod, x); }
        fs::Transcript tp("gkr-unit");
        bflu::BfGkrProof g; vector<bf128_t> rf; bf128_t cl;
        bflu::bfgkr_prove(lv, "u", tp, g, rf, cl, true);
        ck("gkr root == direct product of leaves", bf128_eq(g.root, prod));
        fs::Transcript tv("gkr-unit");
        vector<bf128_t> rf2; bf128_t cl2;
        bool ok = bflu::bfgkr_verify(g, 8, "u", tv, rf2, cl2);
        // leaf binding: claim must equal the true leaf MLE at rfin
        bf128_t direct = bf128_zero();
        std::vector<bf128_t> eq;
        bf_eq_table(rf2.data(), 8, eq);
        for (size_t i = 0; i < 256; i++) direct = bf128_add(direct, bf128_mul(eq[i], lv[i]));
        bool rf_same = rf2.size() == rf.size() &&
            !memcmp(rf2.data(), rf.data(), rf.size() * sizeof(bf128_t));
        ck("gkr chain verifies and binds the true leaf MLE",
           ok && rf_same && bf128_eq(cl2, cl) && bf128_eq(direct, cl));
        auto g2 = g; g2.lay[3].rounds[2].lo ^= 1;
        fs::Transcript tt("gkr-unit");
        ck("gkr tampered layer round rejects", !bflu::bfgkr_verify(g2, 8, "u", tt, rf2, cl2));
        auto g3 = g; g3.g0.lo ^= 2;
        fs::Transcript t3("gkr-unit");
        ck("gkr tampered root child rejects", !bflu::bfgkr_verify(g3, 8, "u", t3, rf2, cl2));
    }

    // ---- honest protocol ----
    Proof pf;
    prove(I, pf, true);
    const char* why = "";
    ck("honest lookup accepts (GPU sumchecks)", verify(I, pf, &why));
    {
        Proof ph;
        prove(I, ph, false);
        ck("GPU proof byte-identical to host proof (gkr chains + bind + openings)",
           proofs_equal(pf, ph));
        ck("host proof verifies", verify(I, ph));
    }
    printf("  [size] lu proof %.1f KB (+ caller opening %.1f KB)\n",
           pf.lu.bytes() / 1024.0, (pf.open.bytes() + sizeof pf.xev) / 1024.0);

    // ---- single out-of-table row ----
    {
        Inst B = I;
        B.wstk[(size_t)3 * N + 77] ^= 1;              // tuple no longer any table row
        Proof pb;
        prove(B, pb, true);
        ck("one out-of-table row rejects (product roots differ)", !verify(B, pb, &why));
    }

    // ---- THE char-2 tooth: even-count out-of-table tuple ----
    {
        // craft two rows with the same key, overwrite both with the SAME
        // garbage tuple (not in the table)
        Inst B = I;
        size_t i1 = 100, i2 = 0;
        for (size_t i = 101; i < N; i++) if (B.idx[i] == B.idx[i1]) { i2 = i; break; }
        vector<uint8_t> bad(J);
        bool intab = true;
        while (intab) {
            for (int k = 0; k < J; k++) bad[k] = rnd() & 1;
            intab = false;
            for (size_t j = 0; j < TN && !intab; j++) {
                bool eq = true;
                for (int k = 0; k < J; k++)
                    if (bad[k] != B.tbits[(size_t)k * TN + j]) { eq = false; break; }
                intab = eq;
            }
        }
        for (int k = 0; k < J; k++) {
            B.wstk[(size_t)k * N + i1] = bad[k];
            B.wstk[(size_t)k * N + i2] = bad[k];
        }
        // 1. demonstrate the vulnerability: the ADDITIVE identity holds exactly
        bf128_t alpha, beta;
        alpha.lo = rnd(); alpha.hi = rnd(); beta.lo = rnd(); beta.hi = rnd();
        vector<bf128_t> bk;
        bflu::bflu_betas(beta, J, bk);
        auto fp = [&](const uint8_t* bits, size_t stride, size_t i) {
            bf128_t v = alpha;
            for (int k = 0; k < J; k++)
                if (bits[(size_t)k * stride + i] & 1) v = bf128_add(v, bk[k]);
            return v;
        };
        bf128_t lhs = bf128_zero();
        for (size_t i = 0; i < N; i++)
            lhs = bf128_add(lhs, bf128_inv(fp(B.wstk.data(), N, i)));
        vector<uint32_t> m(TN, 0);
        for (size_t i = 0; i < N; i++) m[B.idx[i]]++;   // multiplicities as if honest
        bf128_t rhs = bf128_zero();
        for (size_t j = 0; j < TN; j++)
            if (m[j] & 1) rhs = bf128_add(rhs, bf128_inv(fp(B.tbits.data(), TN, j)));
        ck("VULN DEMO: additive char-2 logUp identity HOLDS for the even-count "
           "out-of-table witness", bf128_eq(lhs, rhs));
        // 2. the multiplicative argument rejects it
        Proof pb;
        prove(B, pb, true);
        ck("even-count out-of-table witness REJECTS under the product argument",
           !verify(B, pb, &why));
    }

    // ---- wrong multiplicities ----
    {
        vector<uint32_t> idx2 = I.idx;                 // move one count j1 -> j2
        uint32_t j1 = idx2[42];
        for (size_t i = 0; i < N; i++)
            if (idx2[i] == j1) { idx2[i] = (j1 + 1) & (TN - 1); break; }
        Proof pb;
        prove(I, pb, true, idx2.data());
        ck("wrong multiplicity counts reject", !verify(I, pb, &why));
    }

    // ---- m commit/leaf inconsistencies (cheating-prover hooks) ----
    {
        int MBP = 1 << bflu::bflu_lmb(LN);
        vector<uint32_t> m(TN, 0);
        for (size_t i = 0; i < N; i++) m[I.idx[i]]++;
        vector<uint8_t> mb((size_t)MBP << LT, 0);
        for (size_t j = 0; j < TN; j++)
            for (int b = 0; b < MBP; b++) mb[((size_t)b << LT) + j] = (m[j] >> b) & 1;
        auto mc = mb; mc[((size_t)0 << LT) + I.idx[5]] ^= 1;
        Proof pb;
        prove(I, pb, true, nullptr, mc.data(), nullptr);
        ck("committed m != tree m rejects (binding sumcheck / opening)",
           !verify(I, pb, &why));
        auto ml = mb; ml[((size_t)1 << LT) + I.idx[5]] ^= 1;
        Proof pc;
        prove(I, pc, true, nullptr, nullptr, ml.data());
        ck("tree built from tampered m rejects (root equality)", !verify(I, pc, &why));
    }

    // ---- proof-object tampers ----
    {
        auto q = pf; q.lu.gw.root.lo ^= 1;
        ck("tampered witness product root rejects", !verify(I, q));
        q = pf; q.lu.gw.lay[2].rounds[1].hi ^= 4;
        ck("tampered gw layer message rejects", !verify(I, q));
        q = pf; q.lu.gt.g1.lo ^= 8;
        ck("tampered gt root child rejects", !verify(I, q));
        q = pf; q.lu.bind.rounds[5].lo ^= 2;
        ck("tampered binding-sumcheck message rejects", !verify(I, q));
        q = pf; q.lu.bind.finals[1].lo ^= 1;
        ck("tampered m final (multiplicity eval) rejects", !verify(I, q));
        q = pf; q.lu.bind.finals[2].hi ^= 1;
        ck("tampered u final (public-table eval) rejects", !verify(I, q));
        q = pf; q.lu.mroot[7] ^= 1;
        ck("tampered multiplicity commitment root rejects", !verify(I, q));
        q = pf; q.lu.mopen.t[3].lo ^= 1;
        ck("tampered multiplicity opening rejects", !verify(I, q));
        q = pf; q.xev[4].lo ^= 1;
        ck("tampered witness column eval rejects (leaf binding)", !verify(I, q));
        q = pf; q.open.t[2].hi ^= 1;
        ck("tampered witness opening rejects", !verify(I, q));
    }

    printf("\nBINIUS-LOGUP: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
