// Benchmark the proof components a CODEBOOK integerization would add to /
// swap into zkLLM, vs the native Rescaling lookup:
//   1. codebook membership lookup: 256-entry table over D rows
//      (proves every activation/weight entry is a valid FP8 codebook value)
//   2. 16-bit range lookup over D rows (absmax-gadget range check |x_i| <= M)
//   3. native Rescaling(2^16) prove over D rows (what zkLLM pays today)
// One-hot selector inner product is timed separately via ./bench_matmul 1024 768 1.
// Usage: ./bench_codebook <log2_D> [reps]
#include "tlookup.cuh"
#include "rescaling.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "timer.hpp"
#include <iostream>
using namespace std;

double time_lookup(tLookupRange& tl, FrTensor& vals, uint D, int reps) {
    auto u = random_vec(ceilLog2(D));
    auto v = random_vec(ceilLog2(D));
    auto rnd = random_vec(2);
    // warmup
    {
        auto m = tl.prep(vals);
        vector<Polynomial> proof;
        tl.prove(vals, m, rnd[0], rnd[1], u, v, proof);
    }
    cudaDeviceSynchronize();
    Timer t;
    t.start();
    for (int i = 0; i < reps; i++) {
        auto m = tl.prep(vals);
        vector<Polynomial> proof;
        tl.prove(vals, m, rnd[0], rnd[1], u, v, proof);
    }
    cudaDeviceSynchronize();
    t.stop();
    return t.getTotalTime() * 1000.0 / reps;
}

int main(int argc, char** argv) {
    uint logD = stoi(argv[1]);
    int reps = argc > 2 ? stoi(argv[2]) : 3;
    uint D = 1u << logD;

    // 1. codebook membership: 256-entry table (FP8 e4m3 has <= 256 codes)
    tLookupRange tl_code(-(1 << 7), 1 << 8);
    FrTensor vals8 = FrTensor::random_int(D, 8);
    double ms_code = time_lookup(tl_code, vals8, D, reps);
    cout << "CODEBOOK_LOOKUP_256 logD=" << logD << " ms=" << ms_code << endl;

    // 2. absmax-gadget range check: 2^16-entry range table
    tLookupRange tl_range(-(1 << 15), 1 << 16);
    FrTensor vals16 = FrTensor::random_int(D, 16);
    double ms_range = time_lookup(tl_range, vals16, D, reps);
    cout << "RANGE_LOOKUP_2^16 logD=" << logD << " ms=" << ms_range << endl;

    // 3. what zkLLM-native pays: Rescaling(2^16) prove over the same D
    Rescaling rs(1 << 16);
    FrTensor X = FrTensor::random_int(D, 30);
    auto X_ = rs(X);
    rs.prove(X, X_);  // warmup
    cudaDeviceSynchronize();
    Timer t;
    t.start();
    for (int i = 0; i < reps; i++) rs.prove(X, X_);
    cudaDeviceSynchronize();
    t.stop();
    cout << "NATIVE_RESCALING_2^16 logD=" << logD
         << " ms=" << t.getTotalTime() * 1000.0 / reps << endl;
    return 0;
}
