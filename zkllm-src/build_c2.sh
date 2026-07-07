#!/bin/bash
# Stage C2 rebuild: every TU including zkob_claims.cuh (header-edit rule).
# Pinned build line (STAGE_A/RMSNORM reports): sm_89 -dc -dlto + -dlto link
# against the standard upstream object set.
set -e
cd /root/zkllm
LINKOBJS="bls12-381.o ioutils.o commitment.o fr-tensor.o g1-tensor.o proof.o zkrelu.o zkfc.o tlookup.o polynomial.o zksoftmax.o rescaling.o timer.o"
TUS="vrf_toy_batchopen zkob_batchopen zkob_fc zkob_rescale zkob_glu zkob_rope zkob_headmerge zkob_headslice zkob_rmsnorm zkob_rowmax zkob_softmax zkob_softmax8 zkob_skip"
for f in $TUS; do
  echo "=== build $f ==="
  nvcc -arch=sm_89 -std=c++17 -I/usr/local/cuda/include -dc -dlto $f.cu -o $f.o
  nvcc -arch=sm_89 -std=c++17 -dlto $f.o $LINKOBJS -o $f
done
echo "BUILD_C2 ALL DONE"
