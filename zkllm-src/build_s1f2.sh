#!/bin/bash
set -e
cd /root/zkllm
echo "=== rebuild upstream objects touched by S1 ==="
nvcc -arch=sm_89 -std=c++17 -I/usr/local/cuda/include -dc -dlto g1-tensor.cu -o g1-tensor.o
echo "OK g1-tensor.o"
nvcc -arch=sm_89 -std=c++17 -I/usr/local/cuda/include -dc -dlto commitment.cu -o commitment.o
echo "OK commitment.o"
bash build_c2.sh
echo "=== relink ppgen against rebuilt objects ==="
LINKOBJS="bls12-381.o ioutils.o commitment.o fr-tensor.o g1-tensor.o proof.o zkrelu.o zkfc.o tlookup.o polynomial.o zksoftmax.o rescaling.o timer.o"
nvcc -arch=sm_89 -std=c++17 -dlto ppgen.o $LINKOBJS -o ppgen
echo "OK ppgen"
echo "S1F2 BUILD ALL DONE"
