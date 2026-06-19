#!/usr/bin/env bash
# Rebuild zkob_* single-layer targets on local disk. Usage: build_zkob.sh [target ...]
set -euo pipefail
ZK=/root/zkllm; cd "$ZK"
ARCH=sm_89
NVCC="nvcc -arch=$ARCH -std=c++17 -I/usr/local/cuda/include"
SHARED_CU="bls12-381 ioutils commitment fr-tensor g1-tensor proof zkrelu zkfc tlookup polynomial zksoftmax rescaling"
# compile shared objects once (skip if .o newer than .cu)
for f in $SHARED_CU; do
  if [ ! -f "$f.o" ] || [ "$f.cu" -nt "$f.o" ]; then
    echo "CC $f.cu"; $NVCC -dc -dlto "$f.cu" -o "$f.o"
  fi
done
if [ ! -f timer.o ] || [ timer.cpp -nt timer.o ]; then echo "CC timer.cpp"; $NVCC -x cu -dc -dlto timer.cpp -o timer.o; fi
SHARED_O="$(for f in $SHARED_CU; do echo -n "$f.o "; done) timer.o"
for tgt in "$@"; do
  echo "CC $tgt.cu"; $NVCC -dc -dlto "$tgt.cu" -o "$tgt.o"
  echo "LINK $tgt"; $NVCC -dlto "$tgt.o" $SHARED_O -o "$tgt" -L/usr/local/cuda/lib64
  echo "built: $ZK/$tgt"
done
