// Time RMSNorm ZK proof, per rmsnorm.cu. Usage: ./bench_rmsnorm <seq_len> <embed> [reps]
#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include "timer.hpp"
#include <iostream>
using namespace std;
int main(int argc, char** argv){
  uint seq=stoi(argv[1]); uint embed=stoi(argv[2]); int reps=argc>3?stoi(argv[3]):3;
  FrTensor weight=FrTensor::random_int(embed,16);
  FrTensor X=FrTensor::random_int(seq*embed,16);
  FrTensor rms_inv=FrTensor::random_int(seq,16);
  zkFC g(1, embed, weight);
  Rescaling rs1(1<<16), rs2(1<<16);
  auto run=[&](){
    auto gir=g(rms_inv); auto gir_=rs1(gir);
    auto Y=gir_*X; auto Y_=rs2(Y);
    rs2.prove(Y,Y_);
    hadamard_product_sumcheck(gir_, X, random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)));
    rs1.prove(gir, gir_);
    g.prove(rms_inv, gir);
  };
  run(); cudaDeviceSynchronize();
  Timer t; t.start(); for(int i=0;i<reps;i++) run(); cudaDeviceSynchronize(); t.stop();
  cout<<"RMSNORM_PROVE seq="<<seq<<" embed="<<embed<<" ms="<<t.getTotalTime()*1000/reps<<endl;
}
