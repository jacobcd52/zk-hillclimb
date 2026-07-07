// bf16 ADD gadget test suite (p3_bfadd.cuh) -- teeth included.
//
//   python3 transformer_ref.py --dump-goldens-bfadd p3_bfadd_golden.bin
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_bfadd_test.cu -o /root/p3_bfadd_test
//   cd /root/zkllm && /root/p3_bfadd_test
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_bfadd.cuh"
using namespace p3bfa;
using std::vector;

static int np_ = 0, nf_ = 0;
static void ck(const char* n, bool c, const char* why = nullptr) {
    printf("  [%s] %s%s%s%s\n", c ? "PASS" : "FAIL", n,
           why ? "  (why=" : "", why ? why : "", why ? ")" : "");
    if (c) np_++; else nf_++;
}
static const uint32_t R = 2, Q = 24;
static Tables* TT;

static bool prove_verify(const Wit& wt, const Operands& ops, const char** why,
                         bool strict) {
    BfaProof pf;
    try {
        fs::Transcript tp("bfa");
        pf = prove(tp, wt, *TT, ops, R, Q, strict);
    } catch (const std::exception& e) {
        *why = "prover threw (out of supported domain)";
        return false;
    }
    fs::Transcript tv("bfa");
    return verify(tv, *TT, pf, ops.X1.root, ops.X2.root, ops.OUT.root, wt.n, Q, R, why);
}

int main() {
    printf("=== bf16-add gadget selftest (canonical spec = transformer_ref.py) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    vector<Golden> Gs;
    if (!load_goldens("p3_bfadd_golden.bin", Gs)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-goldens-bfadd p3_bfadd_golden.bin\n");
        return 1;
    }
    Tables T = p3bfa::build_tables(); TT = &T;
    printf("%zu golden cases\n", Gs.size());

    printf("-- honest accept (in-domain cases; incl. layer residuals bitwise) --\n");
    double tot_p = 0, tot_v = 0; size_t tot_sz = 0; int nhon = 0;
    for (size_t ci = 0; ci < Gs.size(); ci++) {
        Golden& L = Gs[ci];
        if (L.flags != 0) continue;
        Wit wt = gen_witness(L);
        bool obit = wt.O.size() == L.o.size() &&
                    memcmp(wt.O.data(), L.o.data(), 2 * L.o.size()) == 0;
        Operands ops = commit_operands(wt, R);
        double t0 = p3hwl::now_ms();
        fs::Transcript tp("bfa");
        BfaProof pf = prove(tp, wt, T, ops, R, Q);
        double t1 = p3hwl::now_ms();
        fs::Transcript tv("bfa");
        bool ok = verify(tv, T, pf, ops.X1.root, ops.X2.root, ops.OUT.root,
                         L.n, Q, R, &why);
        double t2 = p3hwl::now_ms();
        tot_p += t1 - t0; tot_v += t2 - t1; tot_sz += proof_size(pf); nhon++;
        char name[128];
        snprintf(name, sizeof name,
                 "case %zu (n=%u): witness==golden bitwise AND proof accepts", ci, L.n);
        ck(name, obit && ok, ok ? (obit ? nullptr : "bitwise mismatch") : why);
    }
    printf("  totals: prove %.2f s, verify %.2f s, avg proof %.2f MB\n",
           tot_p / 1e3, tot_v / 1e3, tot_sz / (double)nhon / 1048576.0);

    // out-of-domain case: reference flushes/saturates; gadget v1 prover throws
    for (auto& L : Gs) {
        if (L.flags != 1) continue;
        bool threw = false;
        try { gen_witness(L); } catch (const std::exception& e) { threw = true; }
        ck("flush/saturate rows are rejected by the prover (v1 domain)", threw);
    }

    Golden& L0 = Gs[0];                        // residual1, n=256
    Wit w0 = gen_witness(L0);
    Operands ops0 = commit_operands(w0, R);

    // pick useful tamper targets
    uint32_t j_near = 0, j_tie = 0;
    bool have_tie = false;
    for (uint32_t j = 0; j < L0.n; j++) {
        if (w0.ws[BA_NN][j] == 1) { j_near = j; break; }
    }
    for (uint32_t j = 0; j < L0.n; j++) {
        if (w0.ws[BA_NN][j] == 1 && w0.ws[BA_RR][j] != 0) { j_tie = j; have_tie = true; break; }
    }

    printf("-- must-reject (each forgery caught by the intended sub-argument) --\n");
    // (1) RNE round bit flipped, honest downstream (1-ulp output forgery)
    {
        BaTamper tm{BAT_RUP};
        Wit wt = gen_witness(L0, &tm, have_tie ? j_tie : j_near);
        bool odiff = memcmp(wt.O.data(), w0.O.data(), 2 * wt.O.size()) != 0;
        printf("      (round-bit forgery changes the bf16 output: %s)\n", odiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("1-ulp round-direction forgery rejects via the RNE constraints", rej, why);
    }
    // (2) false cancellation claim (output forced to +0)
    {
        BaTamper tm{BAT_CZ};
        Wit wt = gen_witness(L0, &tm, j_near);
        bool odiff = memcmp(wt.O.data(), w0.O.data(), 2 * wt.O.size()) != 0;
        printf("      (false-cancel forgery zeroes the output: %s)\n", odiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("false cancellation-to-zero claim rejects via CZ*AV", rej, why);
    }
    // (3) near row claimed FAR (output snaps to the hi operand)
    {
        BaTamper tm{BAT_FAR};
        Wit wt = gen_witness(L0, &tm, j_near);
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("near-row-claimed-FAR forgery rejects via the DR range", rej, why);
    }
    // (4) swapped hi/lo on an equal-exponent row (alignment forgery); search
    // all golden cases for a D=0, MH!=ML row (the edges case has several)
    {
        Golden* Ld = nullptr; uint32_t jd = 0;
        for (auto& L : Gs) {
            if (L.flags != 0) continue;
            Wit wh = gen_witness(L);
            for (uint32_t j = 0; j < L.n; j++)
                if (wh.ws[BA_NN][j] == 1 && wh.ws[BA_DZ][j] == 1 &&
                    wh.ws[BA_MH][j] != wh.ws[BA_ML][j]) { Ld = &L; jd = j; break; }
            if (Ld) break;
        }
        if (Ld) {
            BaTamper tm{BAT_SWAP};
            Wit wt = gen_witness(*Ld, &tm, jd);
            Operands ops = commit_operands(wt, R);
            bool rej = !prove_verify(wt, ops, &why, false);
            ck("swapped-magnitude alignment forgery rejects via the DM range", rej, why);
        } else {
            ck("swapped-magnitude alignment forgery: NO D=0 row found (fix goldens)", false);
        }
    }
    // (5) result exponent forged one binade up, honest downstream
    {
        BaTamper tm{BAT_EO};
        Wit wt = gen_witness(L0, &tm, j_near);
        bool odiff = memcmp(wt.O.data(), w0.O.data(), 2 * wt.O.size()) != 0;
        printf("      (exponent forgery changes the bf16 output: %s)\n", odiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("one-binade exponent forgery rejects via the EO constraint", rej, why);
    }
    // (6) committed output off by 1 ulp, witness otherwise honest
    {
        Wit wt = w0;
        wt.opat[j_near] = gl_add(wt.opat[j_near], 1ULL);
        Operands ops = ops0;
        ops.OUT = p3lu::commit_col_nc(wt.opat, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("committed output off by 1 ulp rejects via the zero-check", rej, why);
    }
    // (7) wrong X2 operand proven against the REAL X2 commitment
    {
        Golden L2 = L0;
        L2.b[j_near] ^= 0x0100;
        Wit wt = gen_witness(L2);
        Operands ops;
        ops.X1 = ops0.X1; ops.X2 = ops0.X2;    // REAL commitments
        ops.OUT = p3lu::commit_col_nc(wt.opat, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("wrong operand rejects via the X2 commitment binding", rej, why);
    }
    // (8) params + proof-object tampers
    {
        fs::Transcript tp("bfa");
        BfaProof pf = prove(tp, w0, T, ops0, R, Q);
        auto vfy = [&](const BfaProof& p2, uint32_t n, uint32_t q) {
            fs::Transcript tv("bfa");
            return verify(tv, T, p2, ops0.X1.root, ops0.X2.root, ops0.OUT.root,
                          n, q, R, &why);
        };
        { bool rj = !vfy(pf, L0.n, 0);
          ck("Q=0 params forgery rejects", rj, why); }
        { bool rj = !vfy(pf, L0.n * 2, Q);
          ck("wrong public n rejects", rj, why); }
        { fs::Transcript tv("bfa");
          bool ok = verify(tv, T, pf, ops0.X2.root, ops0.X1.root, ops0.OUT.root,
                           L0.n, Q, R, &why);
          ck("swapped X1/X2 roots rejects", !ok, why); }
        { auto p2 = pf; p2.mE[0].s3 = gl_add(p2.mE[0].s3, 1ULL);
          bool rj = !vfy(p2, L0.n, Q);
          ck("tampered zero-check message rejects", rj, why); }
        { auto p2 = pf; p2.yOUT = gl_add(p2.yOUT, 1ULL);
          bool rj = !vfy(p2, L0.n, Q);
          ck("tampered claimed OUT evaluation rejects", rj, why); }
        { auto p2 = pf; p2.lug[0].sub[0].S = gl_add(p2.lug[0].sub[0].S, 1ULL);
          bool rj = !vfy(p2, L0.n, Q);
          ck("tampered RNE-half lookup sum rejects", rj, why); }
        { auto p2 = pf; p2.yE[BA_RUP] = gl_add(p2.yE[BA_RUP], 1ULL);
          bool rj = !vfy(p2, L0.n, Q);
          ck("tampered round-bit column claim rejects", rj, why); }
    }

    printf("\nBFADD-GADGET: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
