// RoPE gadget test suite (p3_rope.cuh) -- teeth included.
//
//   python3 transformer_ref.py --dump-tables        p3_rmsnorm_tables.bin
//   python3 transformer_ref.py --dump-goldens-rope  p3_rope_golden.bin
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_rope_test.cu -o /root/p3_rope_test
//   cd /root/zkllm && /root/p3_rope_test
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
#include "p3_rope.cuh"
using namespace p3rope;
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
static GoldenSet GS;

static bool prove_verify(const Wit& wt, const Operands& ops, const char** why,
                         bool strict, const vector<uint16_t>* vcos = nullptr) {
    RopeProof pf;
    try {
        fs::Transcript tp("rop");
        pf = prove(tp, wt, *TT, ops, R, Q, strict);
    } catch (const std::exception& e) {
        *why = "prover threw (out of supported domain)";
        return false;
    }
    fs::Transcript tv("rop");
    return verify(tv, *TT, pf, vcos ? *vcos : GS.cos, GS.sin,
                  ops.Q.root, ops.OUT.root, GS.seq, GS.dh, Q, R, why);
}

int main() {
    printf("=== RoPE gadget selftest (canonical spec = transformer_ref.py) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    if (!p3rms::load_art("p3_rmsnorm_tables.bin", A)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-tables p3_rmsnorm_tables.bin\n");
        return 1;
    }
    if (!load_goldens("p3_rope_golden.bin", GS)) {
        printf("FATAL: run  python3 transformer_ref.py --dump-goldens-rope p3_rope_golden.bin\n");
        return 1;
    }
    Tables T = p3rope::build_tables(A); TT = &T;
    printf("%zu golden cases (seq=%u dh=%u)\n", GS.cases.size(), GS.seq, GS.dh);

    printf("-- honest accept (in-domain cases; incl. layer q/k bitwise) --\n");
    double tot_p = 0, tot_v = 0; size_t tot_sz = 0; int nhon = 0;
    for (size_t ci = 0; ci < GS.cases.size(); ci++) {
        auto& L = GS.cases[ci];
        if (L.flags != 0) continue;
        Wit wt = gen_witness(GS, L, A);
        bool obit = wt.OUT.size() == L.out.size() &&
                    memcmp(wt.OUT.data(), L.out.data(), 2 * L.out.size()) == 0;
        Operands ops = commit_operands(wt, R);
        double t0 = p3hwl::now_ms();
        fs::Transcript tp("rop");
        RopeProof pf = prove(tp, wt, T, ops, R, Q);
        double t1 = p3hwl::now_ms();
        fs::Transcript tv("rop");
        bool ok = verify(tv, T, pf, GS.cos, GS.sin, ops.Q.root, ops.OUT.root,
                         GS.seq, GS.dh, Q, R, &why);
        double t2 = p3hwl::now_ms();
        tot_p += t1 - t0; tot_v += t2 - t1; tot_sz += proof_size(pf); nhon++;
        char name[128];
        snprintf(name, sizeof name,
                 "case %zu: witness==golden bitwise AND proof accepts", ci);
        ck(name, obit && ok, ok ? (obit ? nullptr : "bitwise mismatch") : why);
    }
    printf("  totals: prove %.2f s, verify %.2f s, avg proof %.2f MB\n",
           tot_p / 1e3, tot_v / 1e3, tot_sz / (double)nhon / 1048576.0);

    for (auto& L : GS.cases) {
        if (L.flags != 1) continue;
        bool threw = false;
        try { gen_witness(GS, L, A); } catch (const std::exception& e) { threw = true; }
        ck("underflowing products are rejected by the prover (v1 domain)", threw);
    }

    auto& L0 = GS.cases[0];                          // layer q head 0
    Wit w0 = gen_witness(GS, L0, A);
    Operands ops0 = commit_operands(w0, R);

    // an element where both products and the combine are nonzero
    uint32_t je = 0;
    for (uint32_t e = 0; e < w0.Ne; e++)
        if (w0.ws[RP_A1 + p3bfa::BA_NN][e] == 1) { je = e; break; }

    printf("-- must-reject --\n");
    // (1) forged product mantissa (a*cos MO+1), honest downstream
    {
        RpTamper tm{RPT_MUL, je};
        Wit wt = gen_witness(GS, L0, A, &tm);
        bool odiff = memcmp(wt.OUT.data(), w0.OUT.data(), 2 * wt.OUT.size()) != 0;
        printf("      (mul forgery changes the rotated output: %s)\n", odiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("1-ulp product forgery rejects via the MUL7 logUp", rej, why);
    }
    // (2) combine RNE round bit flipped, honest downstream
    {
        RpTamper tm{RPT_ADDRUP, je};
        Wit wt = gen_witness(GS, L0, A, &tm);
        bool odiff = memcmp(wt.OUT.data(), w0.OUT.data(), 2 * wt.OUT.size()) != 0;
        printf("      (combine round forgery changes the output: %s)\n", odiff ? "YES" : "no");
        Operands ops = commit_operands(wt, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("combine round-direction forgery rejects via the add block", rej, why);
    }
    // (3) rotated-pair swap: witness built from Q with a/b halves swapped,
    // proven against the REAL Q commitment
    {
        GoldenSet::Case L2 = L0;
        uint32_t half = GS.dh / 2;
        for (uint32_t p = 0; p < GS.seq; p++)
            for (uint32_t j = 0; j < half; j++)
                std::swap(L2.q[(size_t)p * GS.dh + j],
                          L2.q[(size_t)p * GS.dh + half + j]);
        Wit wt = gen_witness(GS, L2, A);
        Operands ops;
        ops.Q = ops0.Q;                               // REAL Q commitment
        ops.OUT = p3lu::commit_col_nc(wt.opat, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("rotated-pair swap rejects via the half-slice openings", rej, why);
    }
    // (4) wrong cos table given to the verifier (public-input binding)
    {
        vector<uint16_t> badcos = GS.cos;
        badcos[3] ^= 0x0001;
        fs::Transcript tp("rop");
        RopeProof pf = prove(tp, w0, T, ops0, R, Q);
        fs::Transcript tv("rop");
        bool ok = verify(tv, T, pf, badcos, GS.sin, ops0.Q.root, ops0.OUT.root,
                         GS.seq, GS.dh, Q, R, &why);
        ck("tampered public cos table rejects", !ok, why);
    }
    // (5) committed output off by 1 ulp
    {
        Wit wt = w0;
        wt.opat[7] = gl_add(wt.opat[7], 1ULL);
        Operands ops = ops0;
        ops.OUT = p3lu::commit_col_nc(wt.opat, R);
        bool rej = !prove_verify(wt, ops, &why, false);
        ck("committed output off by 1 ulp rejects via the zero-check", rej, why);
    }
    // (6) params + proof tampers
    {
        fs::Transcript tp("rop");
        RopeProof pf = prove(tp, w0, T, ops0, R, Q);
        auto vfy = [&](const RopeProof& p2, uint32_t q) {
            fs::Transcript tv("rop");
            return verify(tv, T, p2, GS.cos, GS.sin, ops0.Q.root, ops0.OUT.root,
                          GS.seq, GS.dh, q, R, &why);
        };
        { bool rj = !vfy(pf, 0);
          ck("Q=0 params forgery rejects", rj, why); }
        { auto p2 = pf; p2.mE[0].s1 = gl_add(p2.mE[0].s1, 1ULL);
          bool rj = !vfy(p2, Q);
          ck("tampered zero-check message rejects", rj, why); }
        { auto p2 = pf; p2.yQA = gl_add(p2.yQA, 1ULL);
          bool rj = !vfy(p2, Q);
          ck("tampered Q half-slice claim rejects", rj, why); }
        { auto p2 = pf;
          for (auto& g : p2.lug) { for (auto& m : g.mem) if (!m.extra.empty()) { m.extra[0] = gl_add(m.extra[0], 1ULL); goto ropedone; } } ropedone:;
          bool rj = !vfy(p2, Q);
          ck("tampered MUL7 key binding rejects", rj, why); }
        { auto p2 = pf; p2.lug[0].S = gl_add(p2.lug[0].S, 1ULL);
          bool rj = !vfy(p2, Q);
          ck("tampered combine RNE lookup sum rejects", rj, why); }
    }

    printf("\nROPE-GADGET: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
