// Benchmark: time zkFC.prove (one matmul ZK proof) at given dimensions.
// Usage: ./bench_matmul <seq_len> <in_dim> <out_dim> [reps]
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "timer.hpp"
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char **argv) {
    uint seq = std::stoi(argv[1]);
    uint in_dim = std::stoi(argv[2]);
    uint out_dim = std::stoi(argv[3]);
    int reps = (argc > 4) ? std::stoi(argv[4]) : 3;

    FrTensor weight = FrTensor::random(in_dim * out_dim);
    FrTensor X = FrTensor::random(seq * in_dim);
    zkFC layer(in_dim, out_dim, weight);
    FrTensor Y = layer(X);
    cudaDeviceSynchronize();

    // warmup
    layer.prove(X, Y);
    cudaDeviceSynchronize();

    Timer t;
    t.start();
    for (int i = 0; i < reps; i++) {
        auto claims = layer.prove(X, Y);
    }
    cudaDeviceSynchronize();
    t.stop();
    double ms = t.getTotalTime() * 1000.0 / reps;
    cout << "MATMUL_PROVE seq=" << seq << " in=" << in_dim << " out=" << out_dim
         << " ms_per_prove=" << ms << endl;
    return 0;
}
