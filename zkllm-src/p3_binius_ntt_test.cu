// Selftest for p3_binius_ntt.cuh (design doc section 21.2).
// Teeth: the iterative butterfly network is checked BITWISE against direct
// evaluation of the novel basis polynomials Xhat_k, where the subspace
// polynomials W_i are computed BY DEFINITION (product of (x+v) over the whole
// subspace span(beta_0..beta_{i-1})) -- fully independent of the recurrence
// and twiddle machinery.  Plus Reed-Solomon distance teeth (a rate-1/4
// zero-padded encode of distinct messages must differ in >= 3n+1 positions,
// which any indexing/degree bug breaks) and GPU == host bitwise.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include "p3_binius_ntt.cuh"

static int np_ = 0, nf_ = 0;
static void ck(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) np_++; else nf_++;
}
static uint64_t rs_ = 0xdeadbeefcafe1234ULL;
static uint64_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return rs_; }

// brute-force What_i(x) by definition: W_i(x) = prod_{v in span(beta_0..beta_{i-1})}(x+v)
static uint32_t W_bydef(const std::vector<bf16_t>& beta, int i, uint32_t x) {
    uint32_t p = 1;
    for (uint32_t v = 0; v < (1u << i); v++) {
        uint32_t pt = 0;
        for (int b = 0; b < i; b++) if ((v >> b) & 1) pt ^= beta[b];
        p = bf16_mul(p, x ^ pt);
    }
    return p;
}
static uint32_t What_bydef(const std::vector<bf16_t>& beta, int i, uint32_t x) {
    return bf16_mul(W_bydef(beta, i, x), bf16_inv(W_bydef(beta, i, beta[i])));
}

int main() {
    // ---- transform == brute-force novel-basis evaluation ----
    for (int m : {1, 2, 4, 8, 10}) {
        uint32_t n = 1u << m;
        BfNtt nt; bfntt_init(nt, m);
        std::vector<bf16_t> d(n), c(n);
        for (auto& x : d) x = (bf16_t)(rnd() & 0xffff);
        c = d;
        bfntt_fwd_host(nt, c.data());
        uint32_t bad = 0;
        for (uint32_t u = 0; u < n; u++) {
            uint32_t om = 0;
            for (int i = 0; i < m; i++) if ((u >> i) & 1) om ^= nt.beta[i];
            uint32_t acc = 0;
            for (uint32_t k = 0; k < n; k++) {
                uint32_t xk = 1;
                for (int i = 0; i < m; i++)
                    if ((k >> i) & 1) xk = bf16_mul(xk, What_bydef(nt.beta, i, om));
                acc ^= bf16_mul(d[k], xk);
            }
            if (acc != c[u]) bad++;
        }
        char msg[80];
        snprintf(msg, sizeof msg, "NTT == brute-force Xhat evaluation, n=2^%d (bad=%u)", m, bad);
        ck(msg, bad == 0);
    }
    // ---- F_2-linearity over the message (scalar action by T_16) ----
    {
        int m = 8; uint32_t n = 1u << m;
        BfNtt nt; bfntt_init(nt, m);
        std::vector<bf16_t> f(n), g(n), h(n);
        for (uint32_t i = 0; i < n; i++) { f[i] = (bf16_t)(rnd() & 0xffff); g[i] = (bf16_t)(rnd() & 0xffff); }
        uint32_t a = (uint32_t)(rnd() & 0xffff), b = (uint32_t)(rnd() & 0xffff);
        for (uint32_t i = 0; i < n; i++)
            h[i] = (bf16_t)(bf16_mul(a, f[i]) ^ bf16_mul(b, g[i]));
        bfntt_fwd_host(nt, f.data()); bfntt_fwd_host(nt, g.data()); bfntt_fwd_host(nt, h.data());
        uint32_t bad = 0;
        for (uint32_t i = 0; i < n; i++)
            if (h[i] != (bf16_t)(bf16_mul(a, f[i]) ^ bf16_mul(b, g[i]))) bad++;
        ck("NTT(a*f + b*g) == a*NTT(f) + b*NTT(g)", bad == 0);
    }
    // ---- RS distance teeth: rate-1/4 encode of distinct messages ----
    {
        int mm = 8; uint32_t n = 1u << mm, N = 4 * n;   // message n, code N
        BfNtt nt; bfntt_init(nt, mm + 2);
        int worst = (int)N;
        for (int it = 0; it < 20; it++) {
            std::vector<bf16_t> f(N, 0), g(N, 0);
            for (uint32_t i = 0; i < n; i++) { f[i] = (bf16_t)(rnd() & 0xffff); g[i] = (bf16_t)(rnd() & 0xffff); }
            g[rnd() % n] ^= 1;                          // ensure distinct
            bfntt_fwd_host(nt, f.data()); bfntt_fwd_host(nt, g.data());
            int diff = 0;
            for (uint32_t i = 0; i < N; i++) diff += (f[i] != g[i]);
            if (diff < worst) worst = diff;
        }
        char msg[96];
        snprintf(msg, sizeof msg,
                 "RS rate-1/4 distance: min diff %d/%u >= 3n+1 = %u (20 random pairs)",
                 worst, N, 3 * n + 1);
        ck(msg, worst >= (int)(3 * n + 1));
    }
    // ---- GPU batch == host, bitwise ----
    {
        int m = 12; uint32_t n = 1u << m, R = 513;      // odd row count on purpose
        BfNtt nt; bfntt_init(nt, m);
        BfNttDev dv; bfntt_to_device(nt, dv);
        std::vector<bf16_t> rows((size_t)R * n), ref;
        for (auto& x : rows) x = (bf16_t)(rnd() & 0xffff);
        ref = rows;
        for (uint32_t r = 0; r < R; r++) bfntt_fwd_host(nt, ref.data() + (size_t)r * n);
        bf16_t* d_rows;
        cudaMalloc(&d_rows, rows.size() * sizeof(bf16_t));
        cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(bf16_t), cudaMemcpyHostToDevice);
        bfntt_fwd_gpu(dv, d_rows, R);
        cudaMemcpy(rows.data(), d_rows, rows.size() * sizeof(bf16_t), cudaMemcpyDeviceToHost);
        bool ok = (cudaGetLastError() == cudaSuccess) &&
                  !memcmp(rows.data(), ref.data(), rows.size() * sizeof(bf16_t));
        ck("GPU batch NTT == host, 513 rows x 2^12, bitwise", ok);
        // throughput number for the doc (re-run on fresh data, timed)
        cudaMemcpy(d_rows, ref.data(), rows.size() * sizeof(bf16_t), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < 20; it++) bfntt_fwd_gpu(dv, d_rows, R);
        cudaDeviceSynchronize();
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count() / 20;
        printf("  [info] GPU batch NTT 513x4096 GF(2^16): %.3f ms (%.1f Melem/s)\n",
               ms, (double)R * n / ms / 1e3);
        cudaFree(d_rows);
    }

    printf("\nBINIUS-NTT: %d passed, %d failed -> %s\n", np_, nf_,
           nf_ == 0 ? "ALL PASS" : "FAIL");
    return nf_ == 0 ? 0 : 1;
}
