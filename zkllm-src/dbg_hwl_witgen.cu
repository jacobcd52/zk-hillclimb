// Quick check: C++ full-layer witness replay == golden (numpy/Triton) Y, bitwise.
#include <cstdio>
#include "p3_goldilocks.cuh"
#include "p3_basefold.cuh"
#include "p3_logup.cuh"
#include "p3_hawkeye.cuh"
using namespace p3hwl;

int main() {
    std::vector<Golden> Ls;
    if (!load_layers("hawkeye_layers.bin", Ls)) { printf("FATAL: no hawkeye_layers.bin\n"); return 1; }
    int nf = 0;
    for (size_t i = 0; i < Ls.size(); i++) {
        auto& L = Ls[i];
        try {
            LayerWit wt = gen_witness(L);
            bool ok = wt.Y.size() == L.y.size();
            size_t nmis = 0;
            for (size_t j = 0; j < L.y.size() && ok; j++) if (wt.Y[j] != L.y[j]) nmis++;
            ok = ok && nmis == 0;
            printf("  [%s] layer %zu B=%u K=%u N=%u (P=%zu G=%zu Opad=%u) mismatches=%zu/%zu\n",
                   ok ? "PASS" : "FAIL", i, L.B, L.K, L.N, wt.d.P, wt.d.G, wt.d.Opad,
                   nmis, L.y.size());
            if (!ok) nf++;
        } catch (std::exception& e) {
            printf("  [FAIL] layer %zu: exception: %s\n", i, e.what()); nf++;
        }
    }
    printf("WITGEN: %s\n", nf == 0 ? "ALL PASS" : "FAILURES");
    return nf;
}
