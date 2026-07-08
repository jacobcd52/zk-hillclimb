// quick profile of the accumulation prove section (not part of the battery)
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>
#include "p3_binius_hawkeye.cuh"

using std::vector;

static bool load_acc(const char* path, vector<int64_t> raw[10]) {
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

int main() {
#ifndef __CUDA_ARCH__
    bf16_tab();
#endif
    #pragma omp parallel
    { volatile int wa = omp_get_thread_num(); (void)wa; }
    vector<int64_t> raw[10];
    if (!load_acc("hawkeye_acc_big.bin", raw)) { printf("no big file\n"); return 1; }
    size_t nr = 0;
    vector<uint8_t> bits;
    int lN = bhw::bhw_build_bits(raw, bits, &nr);
    size_t N = (size_t)1 << lN;
    bacc::Wit aw;
    double t0 = bhw::bhw_now_ms();
    bacc::build(lN, bits, bhw::LALM, bhw::LALS, bhw::L5BASE, aw);
    printf("build       %7.1f ms\n", bhw::bhw_now_ms() - t0);
    // commit
    fs::Transcript tr("prof");
    BfPcsCommit C2;
    t0 = bhw::bhw_now_ms();
    bfpcs_commit(bacc::acc_params(lN), aw.bits.data(), tr, C2);
    printf("commit2     %7.1f ms\n", bhw::bhw_now_ms() - t0);
    // per-level: bytes + zerocheck (shared workspace, as the prover does it)
    BfScDev ws;
    bfsc_dev_reserve(ws, (size_t)bacc::AccF<1>::K << (lN - 1));
    uint8_t* d_sb;
    cudaMalloc(&d_sb, (size_t)(bacc::AccF<1>::K - 1) << (lN - 1));
    for (int l = 1; l <= 5; l++) {
        std::vector<bf128_t> g2;
        bacc::acc_gammas(tr, l, g2);
        t0 = bhw::bhw_now_ms();
        std::vector<uint8_t> sb;
        bacc::acc_sc_bytes(lN, l, bits.data(), bhw::LALM, bhw::LALS, bhw::L5BASE,
                           aw.bits.data(), sb);
        double tb = bhw::bhw_now_ms() - t0;
        t0 = bhw::bhw_now_ms();
        BfScProof pf;
        std::vector<bf128_t> zeta;
        switch (l) {
        case 1: bacc::acc_zc<1>(lN, sb, tr, pf, zeta, g2, true, &ws, d_sb); break;
        case 2: bacc::acc_zc<2>(lN, sb, tr, pf, zeta, g2, true, &ws, d_sb); break;
        case 3: bacc::acc_zc<3>(lN, sb, tr, pf, zeta, g2, true, &ws, d_sb); break;
        case 4: bacc::acc_zc<4>(lN, sb, tr, pf, zeta, g2, true, &ws, d_sb); break;
        case 5: bacc::acc_zc<5>(lN, sb, tr, pf, zeta, g2, true, &ws, d_sb); break;
        }
        printf("level %d     bytes %7.1f ms  zerocheck %7.1f ms\n", l, tb,
               bhw::bhw_now_ms() - t0);
    }
    // supplied evals: one main-stack point and one acc-stack point, timed
    std::vector<bf128_t> r(lN);
    for (auto& x : r) x = bf_chal128(tr);
    std::vector<bf128_t> ev(bhw::NCOLA);
    t0 = bhw::bhw_now_ms();
    bacc::acc_stack_evals(bits.data(), bhw::NCOLA, lN, r.data(), ev.data());
    printf("evals main  %7.1f ms (x2 points)\n", bhw::bhw_now_ms() - t0);
    std::vector<bf128_t> ev2(bacc::NSTK2);
    t0 = bhw::bhw_now_ms();
    bacc::acc_stack_evals(aw.bits.data(), bacc::NSTK2, lN, r.data(), ev2.data());
    printf("evals acc   %7.1f ms (x8 points)\n", bhw::bhw_now_ms() - t0);
    (void)N;
    return 0;
}
