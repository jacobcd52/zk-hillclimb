// scratch profiler for bflu_prove internals on the big Hawkeye witness
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <omp.h>
#include "p3_binius_hawkeye.cuh"
using std::vector;
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static bool load_vectors(const char* path, vector<int64_t> raw[10]) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int64_t n = 0;
    if (fread(&n, 8, 1, f) != 1) { fclose(f); return false; }
    for (int c = 0; c < 10; c++) {
        raw[c].resize(n);
        if (fread(raw[c].data(), 8, n, f) != (size_t)n) { fclose(f); return false; }
    }
    fclose(f); return true;
}
int main() {
#ifndef __CUDA_ARCH__
    bf16_tab();
#endif
    #pragma omp parallel
    { volatile int wa = omp_get_thread_num(); (void)wa; }
    vector<int64_t> raw[10];
    if (!load_vectors("hawkeye_prod_big.bin", raw)) { printf("no big file\n"); return 1; }
    vector<uint8_t> bits;
    size_t nr = 0;
    int lN = bhw::bhw_build_bits(raw, bits, &nr);
    size_t N = (size_t)1 << lN;
    int lT = 16, J = bhw::DMJ;
    int dmc[bhw::DMJ]; bhw::bhw_dm_cols(dmc);
    const uint8_t* wc[bhw::DMJ];
    for (int k = 0; k < J; k++) wc[k] = bits.data() + (size_t)dmc[k] * N;
    vector<uint32_t> idx(N);
    for (size_t i = 0; i < N; i++) {
        uint32_t a = 0, b = 0;
        for (int j = 0; j < 8; j++) {
            a |= (uint32_t)(bits[(size_t)(bhw::LA + j) * N + i] & 1) << j;
            b |= (uint32_t)(bits[(size_t)(bhw::LB + j) * N + i] & 1) << j;
        }
        idx[i] = (a << 8) | b;
    }
    const uint8_t* tbits = bhw::bhw_dm_table_bits().data();
    // warm gpu
    { int* d; cudaMalloc(&d, 4); cudaFree(d); }
    for (int it = 0; it < 2; it++) {
    printf("--- iteration %d ---\n", it);
    fs::Transcript tr("prof");
    double t0 = now_ms();
    int lMB = bflu::bflu_lmb(lN); int MBP = 1 << lMB;
    size_t TN = (size_t)1 << lT;
    vector<uint32_t> m(TN, 0);
    for (size_t i = 0; i < N; i++) m[idx[i]]++;
    vector<uint8_t> mbits((size_t)MBP << lT, 0);
    for (size_t j = 0; j < TN; j++)
        for (int b = 0; b < MBP; b++) mbits[((size_t)b << lT) + j] = (m[j] >> b) & 1;
    BfPcsParams pm = bflu::bflu_mparams(lT, lMB);
    BfPcsCommit mC;
    bfpcs_commit(pm, mbits.data(), tr, mC);
    printf("m commit: %.0f ms\n", now_ms() - t0);
    t0 = now_ms();
    bf128_t beta = bf_chal128(tr), alpha = bf_chal128(tr);
    vector<bf128_t> bk; bflu::bflu_betas(beta, J, bk);
    vector<bf128_t> wl(N);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)N; i++) {
        bf128_t v = alpha;
        for (int k = 0; k < J; k++) if (wc[k][i] & 1) v = bf128_add(v, bk[k]);
        wl[i] = v;
    }
    printf("witness leaves: %.0f ms\n", now_ms() - t0);
    t0 = now_ms();
    bflu::BfGkrProof gw; vector<bf128_t> rfw; bf128_t cw;
    bflu::bfgkr_prove(std::move(wl), "bflu-gw", tr, gw, rfw, cw, true);
    printf("witness gkr (2^%d): %.0f ms\n", lN, now_ms() - t0);
    t0 = now_ms();
    vector<bf128_t> tf(TN);
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < (int64_t)TN; j++) {
        bf128_t v = alpha;
        for (int k = 0; k < J; k++) if (tbits[(size_t)k << lT | j] & 1) v = bf128_add(v, bk[k]);
        tf[j] = v;
    }
    vector<bf128_t> u((size_t)MBP << lT), tl((size_t)MBP << lT);
    const bf128_t one = bf128_one();
    for (int b = 0; b < MBP; b++) {
        #pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < (int64_t)TN; j++) {
            if (b) tf[j] = bf128_mul(tf[j], tf[j]);
            size_t x = ((size_t)b << lT) + j;
            u[x] = bf128_add(tf[j], one);
            tl[x] = mbits[x] ? tf[j] : one;
        }
    }
    printf("table leaves (2^%d): %.0f ms\n", lT + lMB, now_ms() - t0);
    t0 = now_ms();
    bflu::BfGkrProof gt; vector<bf128_t> rft; bf128_t cl;
    bflu::bfgkr_prove(std::move(tl), "bflu-gt", tr, gt, rft, cl, true);
    printf("table gkr (2^%d): %.0f ms\n", lT + lMB, now_ms() - t0);
    t0 = now_ms();
    vector<bf128_t> mv((size_t)MBP << lT);
    #pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < (int64_t)((size_t)MBP << lT); x++)
        mv[x] = mbits[x] ? one : bf128_zero();
    vector<bf128_t> zb; BfScProof bind;
    bflu::bflu_sc3(rft, mv, u, tr, bind, zb, true);
    printf("binding sumcheck: %.0f ms\n", now_ms() - t0);
    t0 = now_ms();
    BfPcsProof mopen;
    bfpcs_open(mC, zb.data(), bind.finals[1], tr, mopen);
    printf("m open: %.0f ms\n", now_ms() - t0);
    }
    return 0;
}
