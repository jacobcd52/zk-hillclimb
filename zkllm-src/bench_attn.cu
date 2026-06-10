// Time one attention head's ZK proof (QK^T + softmax + A*V), per self-attn.cu attn mode.
// Usage: ./bench_attn <seq_len> <head_dim> [reps]
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
  uint seq=stoi(argv[1]); uint d=stoi(argv[2]); int reps=argc>3?stoi(argv[3]):3;
  auto run=[&](){
    FrTensor Q=FrTensor::random_int(seq*d,6), K=FrTensor::random_int(seq*d,6), V=FrTensor::random_int(seq*d,6);
    auto X = FrTensor::matmul(Q, K.transpose(seq,d), seq, d, seq);
    zkSoftmax softmax({1<<8,1<<20,1<<20},1,0,1UL<<32,{1<<18,1<<22},seq,seq,d,1);
    Rescaling rs1(1<<20), rs2(1<<20);
    FrTensor shift(seq), X_shifted(seq*seq);
    vector<FrTensor> Xs,Ys,ms;
    FrTensor Y = softmax.compute(X, shift, X_shifted, Xs, Ys, ms);
    auto out = FrTensor::matmul(Y, V, seq, seq, d);
    auto out_=rs2(out); auto out__=rs1(out_);
    rs1.prove(out_,out__); rs2.prove(out,out_);
    auto tr=random_vec(3); vector<Polynomial> proof;
    auto u1=random_vec(ceilLog2(seq)), u2=random_vec(ceilLog2(d)), ud=random_vec(ceilLog2(seq));
    auto claim=out.multi_dim_me({u1,u2},{seq,d});
    auto fc=zkip(claim, Y.partial_me(u1,seq,seq), V.partial_me(u2,d,1), ud, proof);
    softmax.prove(Y,X,shift,X_shifted,Xs,Ys,ms, random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)), tr[0],tr[1],tr[2], proof);
    auto u1_=random_vec(ceilLog2(seq)),u2_=random_vec(ceilLog2(seq)),ud_=random_vec(ceilLog2(d));
    auto claim_=X.multi_dim_me({u1_,u2_},{seq,seq});
    auto fc_=zkip(claim_, Q.partial_me(u1_,seq,d), K.partial_me(u2_,seq,d), ud_, proof);
  };
  run(); cudaDeviceSynchronize();
  Timer t; t.start(); for(int i=0;i<reps;i++) run(); cudaDeviceSynchronize(); t.stop();
  cout<<"ATTN_PROVE seq="<<seq<<" d="<<d<<" ms="<<t.getTotalTime()*1000/reps<<endl;
}
