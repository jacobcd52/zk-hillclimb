// Selftest for the Hawkeye per-product gadget (p3_hawkeye_prod.cuh), driven by
// GOLDEN vectors from the numpy reference (hawkeye_ref.py --dump), which is
// bitwise-validated against the Triton hawkeye_fp8_sum kernel on this GPU.
//   python3 hawkeye_ref.py --dump hawkeye_prod_vectors.bin
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_hawkeye_prod_test.cu -o p3_hawkeye_prod_test
// Honest accept + SEMANTIC forgeries, each crafted to be consistent everywhere
// except the one sub-argument that must catch it:
//   - claim aligned value q-1 with r absorbed into the remainder -> REM lookup
//   - lie about the decode-multiply output mag -> DM lookup
//   - use pw != 2^min(sh,15) (halved q) -> SHIFT lookup
//   - flip the sign of one aligned output -> constraint C2
// plus transcript/opening tampers.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye_prod.cuh"
using namespace p3hw;
using std::vector;

static int np_=0, nf_=0;
static void ck(const char* n, bool c){ printf("  [%s] %s\n", c?"PASS":"FAIL", n); if(c)np_++; else nf_++; }
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); }

static const uint32_t R = 2, Q = 24;

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

// commit all columns of a witness
static vector<Col> commit_all(const ProdWitness& W) {
    vector<Col> C(NCOL);
    for (int c = 0; c < NCOL; c++) C[c] = commit_col(W.col[c], R);
    return C;
}
static bool run_verify(const Tables& T, const ProdProof& pf, const char** why) {
    fs::Transcript tv("hw-prod");
    return verify(tv, T, pf, Q, R, why);
}
static ProdProof run_prove(const ProdWitness& W, const vector<Col>& C, const Tables& T, bool strict=true) {
    fs::Transcript tp("hw-prod");
    return prove(tp, W, C, T, R, Q, true, strict);
}

int main() {
    printf("=== Hawkeye per-product gadget selftest (golden vectors from hawkeye_ref.py) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    vector<int64_t> raw[10];
    if (!load_vectors("hawkeye_prod_vectors.bin", raw)) {
        printf("FATAL: run  python3 hawkeye_ref.py --dump hawkeye_prod_vectors.bin  first\n");
        return 1;
    }
    size_t n = raw[0].size();
    // edge coverage + host-side semantic sanity of the golden vectors
    size_t c_sh0=0, c_shbig=0, c_neg=0, c_zero=0, c_nan=0;
    for (size_t i = 0; i < n; i++) {
        if (raw[6][i] == 0 && raw[5][i]) c_sh0++;
        if (raw[6][i] >= 15) { c_shbig++; if (raw[9][i] != 0) { printf("BAD vector: sh>=15 but al!=0\n"); return 1; } }
        if (raw[9][i] < 0) c_neg++;
        if (raw[5][i] == 0) c_zero++;
        if (raw[0][i] == 0x7F || raw[0][i] == 0xFF || raw[1][i] == 0x7F || raw[1][i] == 0xFF) c_nan++;
    }
    printf("  %zu product rows: shift0(present)=%zu shift>=15=%zu negative-al=%zu absent=%zu nan-code=%zu\n",
           n, c_sh0, c_shbig, c_neg, c_zero, c_nan);
    ck("edge classes all populated", c_sh0>0 && c_shbig>0 && c_neg>0 && c_zero>0 && c_nan>0);

    double t0 = now_ms();
    Tables T = build_tables();
    printf("  tables built: DM=65536x6 SHIFT=64x2 REM=65536x2 RANGE15=32768x1 (%.0f ms)\n", now_ms()-t0);

    ProdWitness W = build_witness(raw);
    printf("  witness: %zu real rows padded to %zu (11 columns)\n", W.n_real, W.col[0].size());

    t0 = now_ms();
    vector<Col> C = commit_all(W);
    double t_commit = now_ms() - t0;
    t0 = now_ms();
    ProdProof pf = run_prove(W, C, T);
    double t_prove = now_ms() - t0;
    t0 = now_ms();
    bool ok = run_verify(T, pf, &why);
    double t_verify = now_ms() - t0;
    ck("honest per-product proof accepts", ok);
    if (why && std::string(why) != "ok") printf("      why=%s\n", why);
    printf("  commit %.0f ms | prove %.0f ms | verify %.0f ms\n", t_commit, t_prove, t_verify);

    // ---- semantic forgery battery (each consistent except ONE sub-argument) ----
    auto find_row = [&](auto pred) -> long {
        for (size_t i = 0; i < n; i++) if (pred(i)) return (long)i;
        return -1;
    };

    // (1) wrong truncation: claim q-1, push the difference into r (r+=pw).
    //     C1 still holds, al adjusted consistently -> ONLY the REM (r<pw) lookup catches.
    { long i = find_row([&](size_t i){ return raw[6][i] >= 1 && raw[6][i] <= 14 && raw[7][i] >= 1; });
      ck("found row for REM forgery", i >= 0);
      if (i >= 0) {
      vector<int64_t> t[10]; for (int c=0;c<10;c++) t[c]=raw[c];
      int64_t pw = 1ll << (t[6][i] < 15 ? t[6][i] : 15);
      t[7][i] -= 1; t[8][i] += pw;
      t[9][i] = t[5][i] ? (1 - 2*t[4][i]) * t[7][i] : 0;
      ProdWitness W2 = build_witness(t); auto C2 = commit_all(W2);
      auto pf2 = run_prove(W2, C2, T, /*strict=*/false);
      ck("wrong truncation (q-1, r+=pw) rejects via REM", !run_verify(T, pf2, &why));
      printf("      why=%s\n", why); } }

    // (2) lie about the decode-multiply output: mag+1 (r+1 keeps C1, al unchanged)
    //     -> ONLY the DM lookup catches.
    { long i = find_row([&](size_t i){ int64_t pw = 1ll << (raw[6][i]<15?raw[6][i]:15);
                                       return raw[5][i] && raw[8][i] + 1 < pw; });
      ck("found row for DM forgery", i >= 0);
      if (i >= 0) {
      vector<int64_t> t[10]; for (int c=0;c<10;c++) t[c]=raw[c];
      t[3][i] += 1; t[8][i] += 1;
      ProdWitness W2 = build_witness(t); auto C2 = commit_all(W2);
      auto pf2 = run_prove(W2, C2, T, false);
      ck("forged decode-multiply output rejects via DM", !run_verify(T, pf2, &why));
      printf("      why=%s\n", why); } }

    // (3) wrong power-of-two for the shift: pw*2 with q/2 (C1, REM, RANGE15, C2 all
    //     still consistent) -> ONLY the SHIFT lookup catches. Proves sh->pw binding.
    { long i = find_row([&](size_t i){ return raw[6][i] >= 1 && raw[6][i] <= 14
                                          && raw[7][i] >= 2 && raw[7][i] % 2 == 0; });
      ck("found row for SHIFT forgery", i >= 0);
      if (i >= 0) {
      vector<int64_t> t[10]; for (int c=0;c<10;c++) t[c]=raw[c];
      t[7][i] /= 2;
      t[9][i] = t[5][i] ? (1 - 2*t[4][i]) * t[7][i] : 0;
      ProdWitness W2 = build_witness(t); auto C2 = commit_all(W2);
      // build_witness derives pw from sh; force the doubled pw + halved q manually
      W2.col[CPW][i] = gl_add(W2.col[CPW][i], W2.col[CPW][i]);
      W2.idxRM[i] = (uint32_t)((2*(1ll << t[6][i])) - 1 + t[8][i]);
      C2[CPW] = commit_col(W2.col[CPW], R);
      auto pf2 = run_prove(W2, C2, T, false);
      ck("wrong pw=2^(sh+1) rejects via SHIFT", !run_verify(T, pf2, &why));
      printf("      why=%s\n", why); } }

    // (4) flip the sign of one aligned output (all lookups untouched) -> C2 catches.
    { long i = find_row([&](size_t i){ return raw[5][i] && raw[7][i] >= 1; });
      vector<int64_t> t[10]; for (int c=0;c<10;c++) t[c]=raw[c];
      t[9][i] = -t[9][i];
      ProdWitness W2 = build_witness(t); auto C2 = commit_all(W2);
      auto pf2 = run_prove(W2, C2, T, false);
      ck("sign-flipped aligned value rejects via C2", !run_verify(T, pf2, &why));
      printf("      why=%s\n", why); }

    // (5) absent product must contribute 0 (pr=0 rows have mag=0 -> q=0; forge a
    //     nonzero contribution al on one) -> C2 catches.
    { long i = find_row([&](size_t i){ return raw[5][i] == 0; });
      ck("found absent row", i >= 0);
      if (i >= 0) {
      vector<int64_t> t[10]; for (int c=0;c<10;c++) t[c]=raw[c];
      t[9][i] = 3;
      ProdWitness W2 = build_witness(t); auto C2 = commit_all(W2);
      auto pf2 = run_prove(W2, C2, T, false);
      ck("absent product with nonzero al rejects via C2", !run_verify(T, pf2, &why));
      printf("      why=%s\n", why); } }

    // ---- proof-object tampers ----
    { auto pf2 = pf; pf2.msgs[0].s1 = gl_add(pf2.msgs[0].s1, 1ULL);
      ck("tampered constraint sumcheck msg rejects", !run_verify(T, pf2, &why));
      printf("      why=%s\n", why); }
    { auto pf2 = pf; pf2.open_al.y = gl_add(pf2.open_al.y, 1ULL);
      ck("tampered opened al value rejects", !run_verify(T, pf2, &why));
      printf("      why=%s\n", why); }
    { auto pf2 = pf; pf2.L_DM.S = gl_add(pf2.L_DM.S, 1ULL);
      ck("tampered DM lookup sum rejects", !run_verify(T, pf2, &why));
      printf("      why=%s\n", why); }
    { auto pf2 = pf; pf2.root[CAL] = pf2.root[CQ];
      ck("swapped column root rejects", !run_verify(T, pf2, &why));
      printf("      why=%s\n", why); }

    printf("\nHAWKEYE-PROD: %d passed, %d failed -> %s\n", np_, nf_, nf_==0?"ALL PASS":"FAIL");
    return nf_==0 ? 0 : 1;
}
