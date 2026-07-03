// SwiGLU gadget test suite (p3_swiglu.cuh) -- teeth included.
//
//   python3 transformer_ref.py --dump-tables         p3_rmsnorm_tables.bin
//   python3 transformer_ref.py --dump-goldens-swiglu p3_swiglu_golden.bin
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_swiglu_test.cu -o /root/p3_swiglu_test
//   cd /root/zkllm && /root/p3_swiglu_test
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_rmsnorm.cuh"
#include "p3_swiglu.cuh"
using namespace p3swg;
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
    SwgProof pf;
    try {
        fs::Transcript tp("swg");
        pf = prove(tp, wt, *TT, ops, R, Q, strict);
    } catch (const std::exception& e) {
        *why = "prover threw (out of supported domain)";
        return false;
    }
    fs::Transcript tv("swg");
    return verify(tv, *TT, pf, ops.GATE.root, ops.UP.root, ops.M.root, wt.n, Q, R, why);
}

int main() {
    printf("=== SwiGLU gadget selftest (canonical spec = transformer_ref.py) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    if (!p3rms::load_art("p3_rmsnorm_tables.bin", A)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-tables p3_rmsnorm_tables.bin\n");
        return 1;
    }
    vector<Golden> Gs;
    if (!load_goldens("p3_swiglu_golden.bin", Gs)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-goldens-swiglu p3_swiglu_golden.bin\n");
        return 1;
    }
    Tables T = p3swg::build_tables(A); TT = &T;
    printf("%zu golden cases\n", Gs.size());

    printf("-- honest accept: %zu golden cases --\n", Gs.size());
    double tot_p = 0, tot_v = 0; size_t tot_sz = 0;
    for (size_t ci = 0; ci < Gs.size(); ci++) {
        Golden& L = Gs[ci];
        Wit wt = gen_witness(L, A);
        bool mbit = wt.M.size() == L.m.size() &&
                    memcmp(wt.M.data(), L.m.data(), 2 * L.m.size()) == 0;
        Operands ops = commit_operands(wt, R);
        double t0 = p3hwl::now_ms();
        fs::Transcript tp("swg");
        SwgProof pf = prove(tp, wt, T, ops, R, Q);
        double t1 = p3hwl::now_ms();
        fs::Transcript tv("swg");
        bool ok = verify(tv, T, pf, ops.GATE.root, ops.UP.root, ops.M.root,
                         L.n, Q, R, &why);
        double t2 = p3hwl::now_ms();
        tot_p += t1 - t0; tot_v += t2 - t1; tot_sz += proof_size(pf);
        char name[128];
        snprintf(name, sizeof name,
                 "case %zu (n=%u): witness==golden bitwise AND proof accepts", ci, L.n);
        ck(name, mbit && ok, ok ? (mbit ? nullptr : "bitwise mismatch") : why);
    }
    printf("  totals: prove %.2f s, verify %.2f s, avg proof %.2f MB\n",
           tot_p / 1e3, tot_v / 1e3, tot_sz / (double)Gs.size() / 1048576.0);

    Golden& L0 = Gs[0];
    Wit w0 = gen_witness(L0, A);
    Operands ops0 = commit_operands(w0, R);

    printf("-- must-reject --\n");
    // (1) wrong silu value (SG+1), honest downstream
    {
        STamper tm{ST_SILU, 5};
        Wit wt = gen_witness(L0, A, &tm);
        bool mdiff = memcmp(wt.M.data(), w0.M.data(), 2 * wt.M.size()) != 0;
        printf("      (forged silu changes the bf16 output: %s)\n", mdiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("wrong silu table row rejects via the SILU logUp", rej, why);
    }
    // (2) product mantissa rounds the wrong way (MO+1), honest downstream
    {
        uint32_t j = 0;
        while (j < L0.n && (w0.ws[S_SGZ][j] || w0.ws[S_UZ][j])) j++;
        STamper tm{ST_MULUP, j};
        Wit wt = gen_witness(L0, A, &tm);
        bool mdiff = memcmp(wt.M.data(), w0.M.data(), 2 * wt.M.size()) != 0;
        printf("      (mul round-up forgery changes the bf16 output: %s)\n",
               mdiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("1-ulp multiply forgery rejects via the MUL7 logUp", rej, why);
    }
    // (3) committed output off by 1 ulp, witness otherwise honest
    {
        Wit wt = w0;
        wt.mpat[7] = gl_add(wt.mpat[7], 1ULL);
        Operands ops = ops0;
        ops.M = p3lu::commit_col_nc(wt.mpat, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("committed output off by 1 ulp rejects via the zero-check", rej, why);
    }
    // (4) wrong up operand proven against the REAL up commitment
    {
        Golden L2 = L0;
        L2.up[9] ^= 0x0100;
        Wit wt = gen_witness(L2, A);
        Operands ops;
        ops.GATE = ops0.GATE; ops.UP = ops0.UP;    // REAL UP commitment
        ops.M = p3lu::commit_col_nc(wt.mpat, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("wrong up operand rejects via the UP commitment binding", rej, why);
    }
    // (5) wrong gate operand proven against the REAL gate commitment
    {
        Golden L2 = L0;
        L2.gate[11] ^= 0x0040;
        Wit wt = gen_witness(L2, A);
        Operands ops;
        ops.GATE = ops0.GATE;                       // REAL GATE commitment
        ops.UP = ops0.UP;
        ops.M = p3lu::commit_col_nc(wt.mpat, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("wrong gate operand rejects via the GATE commitment binding", rej, why);
    }
    // (6) params + proof tampers
    {
        fs::Transcript tp("swg");
        SwgProof pf = prove(tp, w0, T, ops0, R, Q);
        { fs::Transcript tv("swg");
          bool ok = verify(tv, T, pf, ops0.GATE.root, ops0.UP.root, ops0.M.root,
                           L0.n, /*Q=*/0, R, &why);
          ck("Q=0 params forgery rejects", !ok, why); }
        { fs::Transcript tv("swg");
          bool ok = verify(tv, T, pf, ops0.GATE.root, ops0.UP.root, ops0.M.root,
                           L0.n * 2, Q, R, &why);
          ck("wrong public n rejects", !ok, why); }
        { fs::Transcript tv("swg");
          bool ok = verify(tv, T, pf, ops0.UP.root, ops0.GATE.root, ops0.M.root,
                           L0.n, Q, R, &why);
          ck("swapped GATE/UP roots rejects", !ok, why); }
        { auto p2 = pf; p2.mE[0].s2 = gl_add(p2.mE[0].s2, 1ULL);
          fs::Transcript tv("swg");
          bool ok = verify(tv, T, p2, ops0.GATE.root, ops0.UP.root, ops0.M.root,
                           L0.n, Q, R, &why);
          ck("tampered zero-check message rejects", !ok, why); }
        { auto p2 = pf; p2.yM = gl_add(p2.yM, 1ULL);
          fs::Transcript tv("swg");
          bool ok = verify(tv, T, p2, ops0.GATE.root, ops0.UP.root, ops0.M.root,
                           L0.n, Q, R, &why);
          ck("tampered claimed M evaluation rejects", !ok, why); }
        { auto p2 = pf; p2.lu[LUS_SILU].S = gl_add(p2.lu[LUS_SILU].S, 1ULL);
          fs::Transcript tv("swg");
          bool ok = verify(tv, T, p2, ops0.GATE.root, ops0.UP.root, ops0.M.root,
                           L0.n, Q, R, &why);
          ck("tampered SILU lookup sum rejects", !ok, why); }
        { auto p2 = pf; p2.yMumb = gl_add(p2.yMumb, 1ULL);
          fs::Transcript tv("swg");
          bool ok = verify(tv, T, p2, ops0.GATE.root, ops0.UP.root, ops0.M.root,
                           L0.n, Q, R, &why);
          ck("tampered virtual-key binding claim rejects", !ok, why); }
    }

    printf("\nSWIGLU-GADGET: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
