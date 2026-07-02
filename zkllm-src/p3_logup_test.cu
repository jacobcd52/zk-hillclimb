// Selftest for p3_logup.cuh: p3-native logUp over Goldilocks/Basefold.
//   nvcc -arch=sm_89 -std=c++17 -O2 p3_logup_test.cu -o p3_logup_test && ./p3_logup_test
// Honest accept (single-col range table, tuple table, multiplicities>1) +
// negative controls: out-of-table witness, tampered cnt/S, tampered sumcheck
// message, tampered opened value, tampered opening codeword, forged hA.
#include <cstdio>
#include <vector>
#include <string>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
using namespace p3lu;
using std::vector;

static uint64_t S_ = 0xC0FFEE;
static uint64_t rng() { S_ = S_*6364136223846793005ULL+1442695040888963407ULL; uint64_t z=S_; z^=z>>31; return z; }
static int np_=0, nf_=0;
static void ck(const char* n, bool c){ printf("  [%s] %s\n", c?"PASS":"FAIL", n); if(c)np_++; else nf_++; }

static const uint32_t R = 2, Q = 24;

// range table [0,256)
static Table range_table() {
    vector<gl_t> t(256); for (uint32_t j = 0; j < 256; j++) t[j] = j;
    return make_table({t});
}
// tuple table (x, x^2+3, 7x+1) for x in [0,64)
static Table tuple_table() {
    vector<gl_t> t0(64), t1(64), t2(64);
    for (uint32_t j = 0; j < 64; j++) {
        t0[j] = j; t1[j] = gl_add(gl_mul(j, j), 3ULL); t2[j] = gl_add(gl_mul(7ULL, j), 1ULL);
    }
    return make_table({t0, t1, t2});
}

int main() {
    printf("=== p3_logup selftest (logUp over Goldilocks/Basefold) ===\n");
    p3fri::g_gpu_merkle = true; p3bf::p3_enable_mempool();
    const char* why = nullptr;

    // ---- single-column range lookup, N=1024 rows, heavy multiplicities ----
    {
        Table T = range_table();
        uint32_t N = 1024;
        vector<gl_t> w(N); vector<uint32_t> idx(N);
        for (uint32_t i = 0; i < N; i++) { idx[i] = rng() % 256; w[i] = idx[i]; }
        Col W = commit_col(w, R);

        { fs::Transcript tp("lu-t1");
          auto pf = prove(tp, {&W}, idx, T, R, Q, "t1");
          fs::Transcript tv("lu-t1");
          ck("honest range lookup accepts", verify(tv, {W.root}, T, pf, Q, R, "t1", &why));
          if (why && std::string(why) != "ok") printf("      why=%s\n", why); }

        // out-of-table witness value (dishonest prover runs honest procedure, strict=false)
        { vector<gl_t> w2 = w; w2[17] = 300ULL;   // not in [0,256)
          Col W2 = commit_col(w2, R);
          fs::Transcript tp("lu-t1");
          auto pf = prove(tp, {&W2}, idx, T, R, Q, "t1", true, /*strict=*/false);
          fs::Transcript tv("lu-t1");
          ck("out-of-table witness rejects", !verify(tv, {W2.root}, T, pf, Q, R, "t1", &why));
          printf("      why=%s\n", why); }

        // proof tampers
        { fs::Transcript tp("lu-t1"); auto pf = prove(tp, {&W}, idx, T, R, Q, "t1");
          pf.S = gl_add(pf.S, 1ULL);
          fs::Transcript tv("lu-t1");
          ck("tampered S rejects", !verify(tv, {W.root}, T, pf, Q, R, "t1", &why));
          printf("      why=%s\n", why); }
        { fs::Transcript tp("lu-t1"); auto pf = prove(tp, {&W}, idx, T, R, Q, "t1");
          pf.msgsA[0].s0 = gl_add(pf.msgsA[0].s0, 1ULL);
          fs::Transcript tv("lu-t1");
          ck("tampered A-side sumcheck msg rejects", !verify(tv, {W.root}, T, pf, Q, R, "t1", &why));
          printf("      why=%s\n", why); }
        { fs::Transcript tp("lu-t1"); auto pf = prove(tp, {&W}, idx, T, R, Q, "t1");
          pf.msgsT[1].s2 = gl_add(pf.msgsT[1].s2, 1ULL);
          fs::Transcript tv("lu-t1");
          ck("tampered T-side sumcheck msg rejects", !verify(tv, {W.root}, T, pf, Q, R, "t1", &why));
          printf("      why=%s\n", why); }
        { fs::Transcript tp("lu-t1"); auto pf = prove(tp, {&W}, idx, T, R, Q, "t1");
          pf.open_hA.y = gl_add(pf.open_hA.y, 1ULL);
          fs::Transcript tv("lu-t1");
          ck("tampered opened hA value rejects", !verify(tv, {W.root}, T, pf, Q, R, "t1", &why));
          printf("      why=%s\n", why); }
        { fs::Transcript tp("lu-t1"); auto pf = prove(tp, {&W}, idx, T, R, Q, "t1");
          pf.open_cnt.queries[0].rounds[0].a = gl_add(pf.open_cnt.queries[0].rounds[0].a, 1ULL);
          fs::Transcript tv("lu-t1");
          ck("tampered opening codeword rejects", !verify(tv, {W.root}, T, pf, Q, R, "t1", &why));
          printf("      why=%s\n", why); }
        // sum-preserving hA forgery: +d at row 3, -d at row 4 (defeats sum-only
        // checks; MUST be caught by the eq-weighted zero-check)
        { std::vector<std::pair<size_t, gl_t>> tam = {{3, 5ULL}, {4, gl_sub(0ULL, 5ULL)}};
          fs::Transcript tp("lu-t1");
          auto pf = prove(tp, {&W}, idx, T, R, Q, "t1", true, false, &tam);
          fs::Transcript tv("lu-t1");
          ck("sum-preserving forged hA rejects", !verify(tv, {W.root}, T, pf, Q, R, "t1", &why));
          printf("      why=%s\n", why); }
        { fs::Transcript tp("lu-t1"); auto pf = prove(tp, {&W}, idx, T, R, Q, "t1");
          pf.open_hA.Q = 0; pf.open_hA.queries.clear();
          fs::Transcript tv("lu-t1");
          ck("Q=0 forgery rejects (params pinned)", !verify(tv, {W.root}, T, pf, Q, R, "t1", &why));
          printf("      why=%s\n", why); }
    }

    // ---- tuple lookup c=3, N=512 (witness rows must match table ROWS jointly) ----
    {
        Table T = tuple_table();
        uint32_t N = 512;
        vector<gl_t> w0(N), w1(N), w2(N); vector<uint32_t> idx(N);
        for (uint32_t i = 0; i < N; i++) {
            uint32_t j = rng() % 64; idx[i] = j;
            w0[i] = T.cols[0][j]; w1[i] = T.cols[1][j]; w2[i] = T.cols[2][j];
        }
        Col W0 = commit_col(w0, R), W1 = commit_col(w1, R), W2 = commit_col(w2, R);

        { fs::Transcript tp("lu-t2");
          auto pf = prove(tp, {&W0, &W1, &W2}, idx, T, R, Q, "t2");
          fs::Transcript tv("lu-t2");
          vector<gl_t> yW, rA;
          ck("honest tuple lookup accepts",
             verify(tv, {W0.root, W1.root, W2.root}, T, pf, Q, R, "t2", &why, &yW, &rA));
          if (why && std::string(why) != "ok") printf("      why=%s\n", why);
          ck("returned witness evals bound (3 cols, n-dim point)", yW.size()==3 && rA.size()==9); }

        // per-column values individually IN table columns, but row tuple NOT a table row
        { vector<gl_t> w2b = w2; w2b[5] = T.cols[2][(idx[5] + 1) % 64];
          Col W2b = commit_col(w2b, R);
          fs::Transcript tp("lu-t2");
          auto pf = prove(tp, {&W0, &W1, &W2b}, idx, T, R, Q, "t2", true, /*strict=*/false);
          fs::Transcript tv("lu-t2");
          ck("tuple row-mismatch rejects", !verify(tv, {W0.root, W1.root, W2b.root}, T, pf, Q, R, "t2", &why));
          printf("      why=%s\n", why); }

        // forged multiplicities: prover lies about cnt (procedure otherwise honest)
        { vector<uint32_t> idx2 = idx; idx2[0] = (idx[0] + 1) % 64;  // cnt now wrong for 2 rows
          fs::Transcript tp("lu-t2");
          auto pf = prove(tp, {&W0, &W1, &W2}, idx2, T, R, Q, "t2", true, /*strict=*/false);
          fs::Transcript tv("lu-t2");
          ck("wrong multiplicities reject", !verify(tv, {W0.root, W1.root, W2.root}, T, pf, Q, R, "t2", &why));
          printf("      why=%s\n", why); }
    }

    printf("\nP3-LOGUP: %d passed, %d failed -> %s\n", np_, nf_, nf_==0?"ALL PASS":"FAIL");
    return nf_==0 ? 0 : 1;
}
