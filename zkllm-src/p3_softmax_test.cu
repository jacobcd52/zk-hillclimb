// Softmax gadget test suite (p3_softmax.cuh) -- teeth included.
//
//   python3 transformer_ref.py --dump-tables          p3_rmsnorm_tables.bin
//   python3 transformer_ref.py --dump-goldens-softmax p3_softmax_golden.bin
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_softmax_test.cu -o /root/p3_softmax_test
//   cd /root/zkllm && /root/p3_softmax_test
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_rmsnorm.cuh"
#include "p3_bfadd.cuh"
#include "p3_softmax.cuh"
using namespace p3smx;
using std::vector;

static int np_ = 0, nf_ = 0;
static void ck(const char* n, bool c, const char* why = nullptr) {
    printf("  [%s] %s%s%s%s\n", c ? "PASS" : "FAIL", n,
           why ? "  (why=" : "", why ? why : "", why ? ")" : "");
    if (c) np_++; else nf_++;
}
static const uint32_t R = 2, Q = 24;
static Art A;
static Tables* TT;

static bool prove_verify(const Wit& wt, const Golden& L, const Operands& ops,
                         const char** why, bool strict) {
    SmxProof pf;
    try {
        fs::Transcript tp("smx");
        pf = prove(tp, wt, *TT, A, ops, R, Q, strict);
    } catch (const std::exception& e) {
        *why = "prover threw (out of supported domain)";
        return false;
    }
    fs::Transcript tv("smx");
    return verify(tv, *TT, A, pf, L.msk, ops.S.root, ops.P.root,
                  L.B, L.n, Q, R, why);
}

int main() {
    printf("=== softmax gadget selftest (canonical spec = transformer_ref.py) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    if (!p3rms::load_art("p3_rmsnorm_tables.bin", A)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-tables p3_rmsnorm_tables.bin\n");
        return 1;
    }
    vector<Golden> Gs;
    if (!load_goldens("p3_softmax_golden.bin", Gs)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-goldens-softmax p3_softmax_golden.bin\n");
        return 1;
    }
    Tables T = p3smx::build_tables(A); TT = &T;
    printf("%zu golden cases (incl. both layer heads, causal)\n", Gs.size());

    printf("-- honest accept --\n");
    double tot_p = 0, tot_v = 0; size_t tot_sz = 0;
    for (size_t ci = 0; ci < Gs.size(); ci++) {
        Golden& L = Gs[ci];
        Wit wt = gen_witness(L, A);
        bool pbit = wt.P.size() == L.p.size() &&
                    memcmp(wt.P.data(), L.p.data(), 2 * L.p.size()) == 0;
        Operands ops = commit_operands(wt, R);
        double t0 = p3hwl::now_ms();
        fs::Transcript tp("smx");
        SmxProof pf = prove(tp, wt, T, A, ops, R, Q);
        double t1 = p3hwl::now_ms();
        fs::Transcript tv("smx");
        bool ok = verify(tv, T, A, pf, L.msk, ops.S.root, ops.P.root,
                         L.B, L.n, Q, R, &why);
        double t2 = p3hwl::now_ms();
        tot_p += t1 - t0; tot_v += t2 - t1; tot_sz += proof_size(pf);
        char name[160];
        snprintf(name, sizeof name,
                 "case %zu (B=%u n=%u): probs==golden bitwise AND proof accepts",
                 ci, L.B, L.n);
        ck(name, pbit && ok, ok ? (pbit ? nullptr : "bitwise mismatch") : why);
    }
    printf("  totals: prove %.2f s, verify %.2f s, avg proof %.2f MB\n",
           tot_p / 1e3, tot_v / 1e3, tot_sz / (double)Gs.size() / 1048576.0);

    Golden& L0 = Gs[2];                        // random 8x16 full mask
    Wit w0 = gen_witness(L0, A);
    Operands ops0 = commit_operands(w0, R);

    printf("-- must-reject --\n");
    // (1) misplaced rowmax attainment (wrong participating lane claimed max)
    {
        SmTamper tm{SMT_MAX, 1, 0};
        Wit wt = gen_witness(L0, A, &tm);
        bool pdiff = memcmp(wt.P.data(), w0.P.data(), 2 * wt.P.size()) != 0;
        printf("      (forged rowmax changes the probs: %s)\n", pdiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, L0, ops, &why, false);
        ck("misplaced rowmax attainment rejects via the DK range", rej, why);
    }
    // (2) forged exp table output (EXV+1, not an EXPT row), honest downstream
    {
        SmTamper tm{SMT_EXP, 0, 1};
        Wit wt = gen_witness(L0, A, &tm);
        bool pdiff = memcmp(wt.P.data(), w0.P.data(), 2 * wt.P.size()) != 0;
        printf("      (forged exp row changes the probs: %s)\n", pdiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, L0, ops, &why, false);
        ck("forged exp row rejects via the EXPT logUp", rej, why);
    }
    // (3) forged denominator (row sum S+1), honest downstream
    {
        SmTamper tm{SMT_DENOM, 2, 0};
        Wit wt = gen_witness(L0, A, &tm);
        bool pdiff = memcmp(wt.P.data(), w0.P.data(), 2 * wt.P.size()) != 0;
        printf("      (forged denominator changes the probs: %s)\n", pdiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, L0, ops, &why, false);
        ck("forged denominator rejects via the row-binding sumcheck", rej, why);
    }
    // (4) forged reciprocal mantissa (MRC+1, not an RCPT row), honest downstream
    {
        SmTamper tm{SMT_RCP, 3, 0};
        Wit wt = gen_witness(L0, A, &tm);
        bool pdiff = memcmp(wt.P.data(), w0.P.data(), 2 * wt.P.size()) != 0;
        printf("      (forged reciprocal changes the probs: %s)\n", pdiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, L0, ops, &why, false);
        ck("forged reciprocal rejects via the RCPT logUp", rej, why);
    }
    // (5) masked lane leaks into the denominator (causal-mask violation)
    {
        Golden& Lc = Gs[3];                    // random 8x8 causal
        Wit wc = gen_witness(Lc, A);
        uint32_t bb = 2, jj = 3;               // row 2, lane 3 is masked (j > i)
        SmTamper tm{SMT_MASKLEAK, bb, jj};
        Wit wt = gen_witness(Lc, A, &tm);
        bool sdiff = wt.db[B_S][bb] != wc.db[B_S][bb];
        printf("      (mask leak changes the row denominator: %s)\n", sdiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, Lc, ops, &why, false);
        ck("masked-lane denominator leak rejects via the lane gating", rej, why);
    }
    // (6) committed prob off by 1 ulp, witness otherwise honest
    {
        Wit wt = w0;
        wt.ppat[3] = gl_add(wt.ppat[3], 1ULL);
        Operands ops = ops0;
        ops.P = p3lu::commit_col_nc(wt.ppat, R);
        bool rej = !prove_verify(wt, L0, ops, &why, false);
        ck("committed prob off by 1 ulp rejects via the zero-check", rej, why);
    }
    // (7) wrong scores proven against the REAL S commitment
    {
        Golden L2 = L0;
        L2.s[5] ^= 0x0100;
        Wit wt = gen_witness(L2, A);
        Operands ops;
        ops.S = ops0.S;                        // REAL S commitment
        ops.P = p3lu::commit_col_nc(wt.ppat, R);
        bool rej = !prove_verify(wt, L0, ops, &why, false);
        ck("wrong score operand rejects via the S commitment binding", rej, why);
    }
    // (8) verifier given a DIFFERENT public mask than the prover used
    {
        fs::Transcript tp("smx");
        SmxProof pf = prove(tp, w0, T, A, ops0, R, Q);
        Golden L2 = L0;
        L2.msk[1] = 0;                         // drop one lane from the mask
        fs::Transcript tv("smx");
        bool ok = verify(tv, T, A, pf, L2.msk, ops0.S.root, ops0.P.root,
                         L0.B, L0.n, Q, R, &why);
        ck("tampered public mask rejects", !ok, why);
    }
    // (9) params + proof tampers
    {
        fs::Transcript tp("smx");
        SmxProof pf = prove(tp, w0, T, A, ops0, R, Q);
        auto vfy = [&](const SmxProof& p2, uint32_t q) {
            fs::Transcript tv("smx");
            return verify(tv, T, A, p2, L0.msk, ops0.S.root, ops0.P.root,
                          L0.B, L0.n, q, R, &why);
        };
        { bool rj = !vfy(pf, 0);
          ck("Q=0 params forgery rejects", rj, why); }
        { auto p2 = pf; p2.mDe[0].s2 = gl_add(p2.mDe[0].s2, 1ULL);
          bool rj = !vfy(p2, Q);
          ck("tampered De zero-check message rejects", rj, why); }
        { auto p2 = pf; p2.yDeMX = gl_add(p2.yDeMX, 1ULL);
          bool rj = !vfy(p2, Q);
          ck("tampered max-pattern broadcast claim rejects", rj, why); }
        { auto p2 = pf; p2.lug[0].S = gl_add(p2.lug[0].S, 1ULL);
          bool rj = !vfy(p2, Q);
          ck("tampered EXP lookup sum rejects", rj, why); }
        { auto p2 = pf; p2.yBS = gl_add(p2.yBS, 1ULL);
          bool rj = !vfy(p2, Q);
          ck("tampered denominator sum claim rejects", rj, why); }
        { auto p2 = pf;
          for (auto& g : p2.lug) { for (auto& m : g.mem) if (m.extra.size() > 1) { m.extra[1] = gl_add(m.extra[1], 1ULL); goto smxdone; } } smxdone:;
          bool rj = !vfy(p2, Q);
          ck("tampered reciprocal key binding rejects", rj, why); }
    }

    printf("\nSOFTMAX-GADGET: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
