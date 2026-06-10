#!/usr/bin/env bash
# Build the baseline-native prove-dump + verify drivers on LOCAL disk (/root/zkllm).
# Reuses the prebuilt upstream object files. Run from anywhere.
set -euo pipefail
ZK=/root/zkllm
cd "$ZK"

ARCH=sm_89
NVCC="nvcc -arch=$ARCH -std=c++17 -I/usr/local/cuda/include"
OBJS="bls12-381.o ioutils.o commitment.o fr-tensor.o g1-tensor.o proof.o zkrelu.o zkfc.o tlookup.o polynomial.o zksoftmax.o rescaling.o timer.o"

# compile our new units
for f in zkhelpers zkprove_dump zkverify; do
  $NVCC -dc -dlto "$f.cu" -o "$f.o"
done

# link
$NVCC -dlto zkprove_dump.o zkhelpers.o $OBJS -o zkprove_dump -L/usr/local/cuda/lib64
$NVCC -dlto zkverify.o    zkhelpers.o $OBJS -o zkverify    -L/usr/local/cuda/lib64

echo "built: $ZK/zkprove_dump  $ZK/zkverify"
