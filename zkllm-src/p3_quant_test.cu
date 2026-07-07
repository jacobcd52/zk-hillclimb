// Quantize gadget test suite (p3_quant.cuh) -- teeth included.
//
//   python3 transformer_ref.py --dump-tables        p3_rmsnorm_tables.bin
//   python3 transformer_ref.py --dump-goldens-quant p3_quant_golden.bin
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_quant_test.cu -o /root/p3_quant_test
//   cd /root/zkllm && /root/p3_quant_test
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_rmsnorm.cuh"
#include "p3_quant.cuh"
using namespace p3qnt;
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

static bool prove_verify(const Wit& wt, const Operands& ops, const char** why,
                         bool strict) {
    QntProof pf;
    try {
        fs::Transcript tp("qnt");
        pf = prove(tp, wt, *TT, ops, R, Q, strict);
    } catch (const std::exception& e) {
        *why = "prover threw (out of supported domain)";
        return false;
    }
    fs::Transcript tv("qnt");
    return verify(tv, *TT, pf, ops.X.root, ops.CODES.root, ops.SCALES.root,
                  wt.dm.B, wt.dm.ld, Q, R, why);
}

int main() {
    printf("=== quantize gadget selftest (canonical spec = transformer_ref.py) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    if (!p3rms::load_art("p3_rmsnorm_tables.bin", A)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-tables p3_rmsnorm_tables.bin\n");
        return 1;
    }
    vector<Golden> Gs;
    if (!load_goldens("p3_quant_golden.bin", Gs)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-goldens-quant p3_quant_golden.bin\n");
        return 1;
    }
    Tables T = p3qnt::build_tables(A); TT = &T;
    printf("%zu golden cases (incl. all 7 layer call-site shapes)\n", Gs.size());

    printf("-- honest accept: %zu golden cases --\n", Gs.size());
    double tot_p = 0, tot_v = 0; size_t tot_sz = 0;
    for (size_t ci = 0; ci < Gs.size(); ci++) {
        Golden& L = Gs[ci];
        Wit wt = gen_witness(L, A);
        bool cbit = wt.C.size() == L.codes.size() &&
                    memcmp(wt.C.data(), L.codes.data(), L.codes.size()) == 0;
        bool sbit = wt.S.size() == L.scales.size() &&
                    memcmp(wt.S.data(), L.scales.data(), 4 * L.scales.size()) == 0;
        Operands ops = commit_operands(wt, R);
        double t0 = p3hwl::now_ms();
        fs::Transcript tp("qnt");
        QntProof pf = prove(tp, wt, T, ops, R, Q);
        double t1 = p3hwl::now_ms();
        fs::Transcript tv("qnt");
        bool ok = verify(tv, T, pf, ops.X.root, ops.CODES.root, ops.SCALES.root,
                         L.B, wt.dm.ld, Q, R, &why);
        double t2 = p3hwl::now_ms();
        tot_p += t1 - t0; tot_v += t2 - t1; tot_sz += proof_size(pf);
        char name[160];
        snprintf(name, sizeof name,
                 "case %zu (B=%u d=%u): codes+scales==golden bitwise AND proof accepts",
                 ci, L.B, L.d);
        ck(name, cbit && sbit && ok,
           ok ? (cbit && sbit ? nullptr : "bitwise mismatch") : why);
    }
    printf("  totals: prove %.2f s, verify %.2f s, avg proof %.2f MB\n",
           tot_p / 1e3, tot_v / 1e3, tot_sz / (double)Gs.size() / 1048576.0);

    Golden& L0 = Gs[0];                        // rms1 call site, B=4 d=64
    Wit w0 = gen_witness(L0, A);
    Operands ops0 = commit_operands(w0, R);

    printf("-- must-reject --\n");
    // (1) forged row emax (E+1, no attainer -> fsel forced), honest downstream
    {
        QTamper tm{QT_MAXEXP, 1, 0};
        Wit wt = gen_witness(L0, A, &tm);
        bool sdiff = memcmp(wt.S.data(), w0.S.data(), 4 * wt.S.size()) != 0;
        bool cdiff = memcmp(wt.C.data(), w0.C.data(), wt.C.size()) != 0;
        printf("      (forged emax changes scale/codes: %s/%s)\n",
               sdiff ? "YES" : "no", cdiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("forged row emax rejects via the attainment argument", rej, why);
    }
    // (2) forged magnitude code (MAG+1 -> not a QEXT row), honest downstream
    {
        uint32_t i = 0;
        while (i < L0.d && w0.de[D_XZ][i]) i++;
        QTamper tm{QT_MAG, 0, i};
        Wit wt = gen_witness(L0, A, &tm);
        bool cdiff = memcmp(wt.C.data(), w0.C.data(), wt.C.size()) != 0;
        printf("      (forged code changes the committed codes: %s)\n", cdiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("1-ulp code forgery rejects via the QEXT logUp", rej, why);
    }
    // (3) off-by-one-binade scale (E honest, scale forged)
    {
        QTamper tm{QT_SCALE, 2, 0};
        Wit wt = gen_witness(L0, A, &tm);
        bool sdiff = memcmp(wt.S.data(), w0.S.data(), 4 * wt.S.size()) != 0;
        printf("      (forged scale changes the committed scales: %s)\n", sdiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("off-by-one-binade scale rejects via the Db zero-check", rej, why);
    }
    // (4) committed code byte tampered, witness otherwise honest
    {
        Wit wt = w0;
        wt.cpat[5] = gl_add(wt.cpat[5], 1ULL);
        Operands ops = ops0;
        ops.CODES = p3lu::commit_col_nc(wt.cpat, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("tampered committed code byte rejects via the De zero-check", rej, why);
    }
    // (5) wrong X proven against the REAL X commitment
    {
        Golden L2 = L0;
        L2.x[3] ^= 0x0100;
        Wit wt = gen_witness(L2, A);
        Operands ops;
        ops.X = ops0.X;                        // REAL X commitment
        ops.CODES = p3lu::commit_col_nc(wt.cpat, R);
        ops.SCALES = p3lu::commit_col_nc(wt.spat, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("wrong input operand rejects via the X commitment binding", rej, why);
    }
    // (6) params + proof-object tampers
    {
        fs::Transcript tp("qnt");
        QntProof pf = prove(tp, w0, T, ops0, R, Q);
        auto vfy = [&](const QntProof& p2, uint32_t B, uint32_t ld, uint32_t q) {
            fs::Transcript tv("qnt");
            return verify(tv, T, p2, ops0.X.root, ops0.CODES.root, ops0.SCALES.root,
                          B, ld, q, R, &why);
        };
        { bool rj = !vfy(pf, L0.B, w0.dm.ld, 0);
          ck("Q=0 params forgery rejects", rj, why); }
        { bool rj = !vfy(pf, L0.B + 1, w0.dm.ld, Q);
          ck("wrong public B rejects", rj, why); }
        { auto p2 = pf; p2.mDe[0].s2 = gl_add(p2.mDe[0].s2, 1ULL);
          bool rj = !vfy(p2, L0.B, w0.dm.ld, Q);
          ck("tampered De zero-check message rejects", rj, why); }
        { auto p2 = pf; p2.mDb[0].s1 = gl_add(p2.mDb[0].s1, 1ULL);
          bool rj = !vfy(p2, L0.B, w0.dm.ld, Q);
          ck("tampered Db zero-check message rejects", rj, why); }
        { auto p2 = pf; p2.yDbS = gl_add(p2.yDbS, 1ULL);
          bool rj = !vfy(p2, L0.B, w0.dm.ld, Q);
          ck("tampered claimed SCALES evaluation rejects", rj, why); }
        { auto p2 = pf; p2.lug[0].sub[0].gk.P = gl_add(p2.lug[0].sub[0].gk.P, 1ULL);
          bool rj = !vfy(p2, L0.B, w0.dm.ld, Q);
          ck("tampered QEXT lookup sum rejects", rj, why); }
        { auto p2 = pf; p2.yBF = gl_add(p2.yBF, 1ULL);
          bool rj = !vfy(p2, L0.B, w0.dm.ld, Q);
          ck("tampered attainment binding claim rejects", rj, why); }
    }
    // (7) swapped CODES/SCALES roots
    {
        fs::Transcript tp("qnt");
        QntProof pf = prove(tp, w0, T, ops0, R, Q);
        fs::Transcript tv("qnt");
        bool ok = verify(tv, T, pf, ops0.X.root, ops0.SCALES.root, ops0.CODES.root,
                         L0.B, w0.dm.ld, Q, R, &why);
        ck("swapped CODES/SCALES roots rejects", !ok, why);
    }

    printf("\nQUANT-GADGET: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
