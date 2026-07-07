// RMSNorm gadget test suite (p3_rmsnorm.cuh) -- teeth included.
//
//   python3 transformer_ref.py --dump-tables  p3_rmsnorm_tables.bin
//   python3 transformer_ref.py --dump-goldens p3_rmsnorm_golden.bin
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_rmsnorm_test.cu -o /root/p3_rmsnorm_test
//   cd /root/zkllm && /root/p3_rmsnorm_test
//
// HONEST ACCEPT: every golden case (random rows, wildly different row norms,
// zero row / one-hot / tiny / signed-zero rows, +-0 gains, eps-dominated rows
// at the EMIN floor) has witness output BITWISE equal to the canonical Python
// reference AND the proof verifies against the committed X/G/Y roots.
//
// MUST-REJECT: semantic forgeries replayed honestly downstream from one forged
// step (wrong sum-of-squares, wrong rsqrt table row, wrong eps row, overstated
// row max, round-up alignment, round-up output multiply), a 1-ulp output flip,
// a wrong gain vector against the real gain commitment, wrong public params,
// and proof-object tampers.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
#include "p3_rmsnorm.cuh"
using namespace p3rms;
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

static bool prove_verify(const Wit& wt, const Operands& ops, uint32_t B, uint32_t ld,
                         const char** why, bool strict, RmsProof* out = nullptr) {
    RmsProof pf;
    try {
        fs::Transcript tp("rms");
        pf = prove(tp, wt, *TT, A, ops, R, Q, strict);
    } catch (const std::exception& e) {
        *why = "prover threw (out of supported domain)";
        return false;
    }
    if (out) *out = pf;
    fs::Transcript tv("rms");
    return verify(tv, *TT, A, pf, ops.X.root, ops.G.root, ops.Y.root, B, ld, Q, R, why);
}

int main(int argc, char** argv) {
    // optional: argv[1] tables.bin, argv[2] goldens.bin (large-d runs, e.g.
    // ld=8/ld=10; default = the historical ld=6 pair)
    const char* tab_path = argc > 1 ? argv[1] : "p3_rmsnorm_tables.bin";
    const char* gld_path = argc > 2 ? argv[2] : "p3_rmsnorm_golden.bin";
    printf("=== RMSNorm gadget selftest (canonical spec = transformer_ref.py) ===\n");
    printf("tables=%s goldens=%s\n", tab_path, gld_path);
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    if (!load_art(tab_path, A)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-tables %s\n", tab_path);
        return 1;
    }
    uint32_t ld = 0;
    vector<Golden> Gs;
    if (!load_goldens(gld_path, Gs, ld)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-goldens %s\n", gld_path);
        return 1;
    }
    printf("artifact: ld=%u EMIN=%ld eps_bits=0x%08lx; %zu golden cases\n",
           ld, (long)A.EMIN, (unsigned long)A.eps_bits, Gs.size());
    Tables T = build_tables(A); TT = &T;

    // ---------------- HONEST ACCEPT ----------------
    printf("-- honest accept: %zu golden cases --\n", Gs.size());
    double tot_p = 0, tot_v = 0; size_t tot_sz = 0;
    for (size_t ci = 0; ci < Gs.size(); ci++) {
        Golden& L = Gs[ci];
        Wit wt = gen_witness(L, A);
        bool ybit = wt.Y.size() == L.y.size() &&
                    memcmp(wt.Y.data(), L.y.data(), 2 * L.y.size()) == 0;
        Operands ops = commit_operands(wt, R);
        double t0 = p3hwl::now_ms();
        fs::Transcript tp("rms");
        RmsProof pf = prove(tp, wt, T, A, ops, R, Q);
        double t1 = p3hwl::now_ms();
        fs::Transcript tv("rms");
        bool ok = verify(tv, T, A, pf, ops.X.root, ops.G.root, ops.Y.root,
                         L.B, ld, Q, R, &why);
        double t2 = p3hwl::now_ms();
        tot_p += t1 - t0; tot_v += t2 - t1; tot_sz += proof_size(pf);
        char name[128];
        snprintf(name, sizeof name,
                 "case %zu (B=%u d=%u): witness==golden bitwise AND proof accepts",
                 ci, L.B, L.d);
        ck(name, ybit && ok, ok ? (ybit ? nullptr : "bitwise mismatch") : why);
    }
    printf("  totals: prove %.2f s, verify %.2f s, avg proof %.2f MB\n",
           tot_p / 1e3, tot_v / 1e3, tot_sz / (double)Gs.size() / 1048576.0);

    Golden& L0 = Gs[0];
    Wit w0 = gen_witness(L0, A);
    Operands ops0 = commit_operands(w0, R);

    // ---------------- MUST-REJECT: semantic forgeries ----------------
    printf("-- must-reject: semantic forgeries (honest downstream replay) --\n");
    // (1) wrong sum of squares (S+1), everything downstream recomputed honestly
    {
        RTamper tm{RT_SUM, 1, 0};
        Wit wt = gen_witness(L0, A, &tm);
        bool ydiff = memcmp(wt.Y.data(), w0.Y.data(), 2 * wt.Y.size()) != 0;
        printf("      (forged sum changes the bf16 output: %s)\n", ydiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, L0.B, ld, &why, false);

        ck("wrong sum-of-squares rejects via the row-binding sumcheck", rej, why);
    }
    // (2) wrong rsqrt table row (mr+1), downstream honest
    {
        RTamper tm{RT_RSQ, 0, 0};
        Wit wt = gen_witness(L0, A, &tm);
        bool ydiff = memcmp(wt.Y.data(), w0.Y.data(), 2 * wt.Y.size()) != 0;
        printf("      (forged rsqrt changes the bf16 output: %s)\n", ydiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, L0.B, ld, &why, false);

        ck("wrong rsqrt lookup rejects via the RSQ logUp", rej, why);
    }
    // (3) wrong eps row (EPSA+1), downstream honest
    {
        RTamper tm{RT_EPSA, 2, 0};
        Wit wt = gen_witness(L0, A, &tm);
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, L0.B, ld, &why, false);

        ck("forged EPSA value rejects via the EPSA logUp", rej, why);
    }
    // (4) overstated row max exponent (E+1), downstream honest
    {
        RTamper tm{RT_MAXEXP, 1, 0};
        Wit wt = gen_witness(L0, A, &tm);
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, L0.B, ld, &why, false);

        ck("overstated row max rejects via the attainment constraints", rej, why);
    }
    // (5) per-element alignment rounds up (q+1, r-=pw): only REM can catch
    {
        long ei = -1; uint32_t bb = 0, ii = 0;
        for (uint32_t b = 0; b < L0.B && ei < 0; b++)
            for (uint32_t i = 0; i < L0.d; i++) {
                size_t e = ((size_t)b << ld) | i;
                if (w0.de[D_XZ][e] == 0 && (int64_t)w0.de[D_SH][e] >= 1 &&
                    (int64_t)w0.de[D_RR][e] > 0) { ei = (long)e; bb = b; ii = i; break; }
            }
        ck("found element for round-up forgery", ei >= 0);
        if (ei >= 0) {
            RTamper tm{RT_ROUND, bb, ii};
            Wit wt = gen_witness(L0, A, &tm);
            bool ydiff = memcmp(wt.Y.data(), w0.Y.data(), 2 * wt.Y.size()) != 0;
            printf("      (round-up forgery changes the bf16 output: %s)\n",
                   ydiff ? "YES" : "no");
            Operands ops = commit_operands(wt, R);
            bool rej = !prove_verify(wt, ops, L0.B, ld, &why, false);

            ck("round-up (q+1, r-=pw) alignment rejects via the REM logUp", rej, why);
        }
    }
    // (6) output multiply rounds the wrong way (MO2+1): 1-ulp mul forgery
    {
        RTamper tm{RT_MULUP, 0, 3};
        Wit wt = gen_witness(L0, A, &tm);
        bool ydiff = memcmp(wt.Y.data(), w0.Y.data(), 2 * wt.Y.size()) != 0;
        printf("      (mul round-up forgery changes the bf16 output: %s)\n",
               ydiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, L0.B, ld, &why, false);

        ck("1-ulp output-multiply forgery rejects via the MUL7 logUp", rej, why);
    }

    printf("-- must-reject: output / operand / param forgeries --\n");
    // (7) public output off by 1 ulp (witness honest, committed Y forged)
    {
        Wit wt = w0;
        wt.ypat[3] = gl_add(wt.ypat[3], 1ULL) ;
        Operands ops = ops0;
        ops.Y = p3lu::commit_col_nc(wt.ypat, R);
        bool rej = !prove_verify(wt, ops, L0.B, ld, &why, false);

        ck("committed output off by 1 ulp rejects via the De zero-check", rej, why);
    }
    // (8) wrong gain vector proven against the REAL gain commitment
    {
        Golden L2 = L0;
        L2.g[3] ^= 0x0100;                    // change one gain's exponent
        Wit wt = gen_witness(L2, A);
        Operands ops;
        ops.X = ops0.X; ops.G = ops0.G;       // REAL G commitment
        ops.Y = p3lu::commit_col_nc(wt.ypat, R);
        bool rej = !prove_verify(wt, ops, L0.B, ld, &why, false);

        ck("wrong gain vector rejects via the G commitment binding", rej, why);
    }
    // (9) wrong input proven against the REAL input commitment
    {
        Golden L2 = L0;
        L2.x[5] ^= 0x0008;
        Wit wt = gen_witness(L2, A);
        Operands ops;
        ops.X = ops0.X;                       // REAL X commitment
        ops.G = ops0.G;
        ops.Y = p3lu::commit_col_nc(wt.ypat, R);
        bool rej = !prove_verify(wt, ops, L0.B, ld, &why, false);

        ck("wrong input vector rejects via the X commitment binding", rej, why);
    }
    // (10) parameter forgeries: verifier pins its OWN params
    {
        fs::Transcript tp("rms");
        RmsProof pf = prove(tp, w0, T, A, ops0, R, Q);
        { fs::Transcript tv("rms");
          bool ok = verify(tv, T, A, pf, ops0.X.root, ops0.G.root, ops0.Y.root,
                           L0.B, ld, /*Q=*/0, R, &why);
          ck("Q=0 params forgery rejects", !ok, why); }
        { fs::Transcript tv("rms");
          bool ok = verify(tv, T, A, pf, ops0.X.root, ops0.G.root, ops0.Y.root,
                           L0.B + 1, ld, Q, R, &why);
          ck("wrong public B rejects", !ok, why); }
        { fs::Transcript tv("rms");
          bool ok = verify(tv, T, A, pf, ops0.G.root, ops0.X.root, ops0.Y.root,
                           L0.B, ld, Q, R, &why);
          ck("swapped X/G commitment roots rejects", !ok, why); }
        { Art A2 = A; A2.EMIN += 1;
          fs::Transcript tv("rms");
          bool ok = verify(tv, T, A2, pf, ops0.X.root, ops0.G.root, ops0.Y.root,
                           L0.B, ld, Q, R, &why);
          ck("wrong public EMIN rejects", !ok, why); }
        // (11) proof-object tampers
        { auto p2 = pf; p2.mDe[0].s1 = gl_add(p2.mDe[0].s1, 1ULL);
          fs::Transcript tv("rms");
          bool ok = verify(tv, T, A, p2, ops0.X.root, ops0.G.root, ops0.Y.root,
                           L0.B, ld, Q, R, &why);
          ck("tampered De sumcheck message rejects", !ok, why); }
        { auto p2 = pf; p2.yDeY = gl_add(p2.yDeY, 1ULL);
          fs::Transcript tv("rms");
          bool ok = verify(tv, T, A, p2, ops0.X.root, ops0.G.root, ops0.Y.root,
                           L0.B, ld, Q, R, &why);
          ck("tampered claimed Y evaluation rejects", !ok, why); }
        { auto p2 = pf; p2.lug[0].sub[0].S = gl_add(p2.lug[0].sub[0].S, 1ULL);
          fs::Transcript tv("rms");
          bool ok = verify(tv, T, A, p2, ops0.X.root, ops0.G.root, ops0.Y.root,
                           L0.B, ld, Q, R, &why);
          ck("tampered RSQ lookup sum rejects", !ok, why); }
        { auto p2 = pf; p2.rde[D_Q] = p2.rde[D_RR];
          fs::Transcript tv("rms");
          bool ok = verify(tv, T, A, p2, ops0.X.root, ops0.G.root, ops0.Y.root,
                           L0.B, ld, Q, R, &why);
          ck("swapped witness column root rejects", !ok, why); }
        { auto p2 = pf;
          for (auto& g : p2.lug) { for (auto& sb : g.sub) for (auto& m : sb.mem) if (!m.extra.empty()) { m.extra.back() = gl_add(m.extra.back(), 1ULL); goto rmsdone; } } rmsdone:;
          fs::Transcript tv("rms");
          bool ok = verify(tv, T, A, p2, ops0.X.root, ops0.G.root, ops0.Y.root,
                           L0.B, ld, Q, R, &why);
          ck("tampered virtual-key binding claim rejects", !ok, why); }
        { auto p2 = pf;
          if (!p2.batches.empty() && !p2.batches[0].ystar.empty())
              p2.batches[0].ystar[0] = gl_add(p2.batches[0].ystar[0], 1ULL);
          fs::Transcript tv("rms");
          bool ok = verify(tv, T, A, p2, ops0.X.root, ops0.G.root, ops0.Y.root,
                           L0.B, ld, Q, R, &why);
          ck("tampered batch-opening terminal rejects", !ok, why); }
    }

    printf("\nRMSNORM-GADGET: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
