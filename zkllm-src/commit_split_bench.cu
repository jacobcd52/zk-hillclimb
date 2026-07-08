// salted_commit_root cost split at big dims: upload+NTT vs salted leaves vs tree
#include <cstdio>
#include <chrono>
#include "p3_zkc.cuh"

static double ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    p3zkc::G.on = true;
    for (uint32_t v : {25u, 26u}) {
        size_t N = (size_t)1 << v;
        std::vector<gl_t> c(N);
        uint64_t s = 7;
        p3zkc::zprng_fill(s, c.data(), N);
        uint32_t R = 2, logM0 = v + R; uint32_t M0 = 1u << logM0;
        // warmup NTT plan
        p3bf::ntt_plan(logM0);
        cudaDeviceSynchronize();

        double t0 = ms();
        uint32_t M0o; gl_t* d_cw = p3bf::rs_encode_gpu_dev(c, R, M0o);
        cudaDeviceSynchronize();
        double t1 = ms();
        // salted leaves only
        uint8_t* dl; cudaMalloc(&dl, ((size_t)1 << 22) * 32);
        double t2 = ms();
        for (size_t off = 0; off < M0; off += ((size_t)1 << 22))
            p3zkc::p3zkc_salted_leaf_off_kernel<<<((1u << 22) + 255) / 256, 256>>>(
                d_cw + off, 12345, dl, off, 1u << 22);
        cudaDeviceSynchronize();
        double t3 = ms();
        cudaFree(dl);
        // full streamed salted tree
        double t4 = ms();
        p3fri::Hash r = p3fri::stream_tree_build(M0, [&](size_t off, uint32_t len, uint8_t* out) {
            p3zkc::p3zkc_salted_leaf_off_kernel<<<(len + 255) / 256, 256>>>(
                d_cw + off, 12345, out, off, len);
        }).root();
        cudaDeviceSynchronize();
        double t5 = ms();
        cudaFree(d_cw);
        printf("v=%u M0=2^%u: upload+NTT=%.0fms  salted-leaves-only=%.0fms  full-tree(leaves+internal)=%.0fms  root=%02x%02x\n",
               v, logM0, t1 - t0, t3 - t2, t5 - t4, r[0], r[1]);
    }
    return 0;
}
