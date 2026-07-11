#!/bin/bash
# Benchmark sweep: exact-fp8 layer proof (p3_transformer_bench, zk 0/1) vs
# native fp8 forward (tf_fwd_bench.py) vs integerized GEMM proof
# (p3_matmul_bench2), varying model size / seq / batch.
cd /root/zkllm
log2(){ python3 -c "import math;print(int(math.log2($1)))"; }

run_cfg(){
  local d=$1 seq=$2 batch=$3
  local dh=16 nh=$((d/16)) dff=$((4*d)) ld=$(log2 $d) T=$((batch*seq))
  local tb=tables_ld${ld}.bin
  echo "### CFG d=$d seq=$seq batch=$batch nh=$nh dh=$dh dff=$dff tokens=$T ld=$ld"
  # exact-fp8 layer proof, non-zk then zk
  timeout 1200 /root/p3_transformer_bench $seq $d $nh $dh $dff $batch 0 $tb 2>&1 | grep -E "^BENCH|^STAGES|FATAL" | sed 's/^/FP8ZK0 /'
  timeout 1800 /root/p3_transformer_bench $seq $d $nh $dh $dff $batch 1 $tb 2>&1 | grep -E "^BENCH|^STAGES|FATAL" | sed 's/^/FP8ZK1 /'
  # native fp8 forward (per-layer ms)
  timeout 300 python3 tf_fwd_bench.py $seq $d $nh $dh $dff $batch 30 2>&1 | grep -E "^FWD|Error|error" | sed 's/^/NATIVE /'
  # integerized GEMM proofs at the layer's dominant shapes (proj d->d, mlp d->4d)
  local lT=$(log2 $T) ld=$(log2 $d) l4d=$(log2 $dff)
  echo -n "INT_PROJ "; timeout 300 /root/p3_matmul_bench2 $lT $ld $ld 2>&1 | grep IMM
  echo -n "INT_MLP  "; timeout 300 /root/p3_matmul_bench2 $lT $ld $l4d 2>&1 | grep IMM
  echo
}

echo "===== MODEL-SIZE SWEEP (seq=64, batch=1) ====="
for d in 64 128 256 512; do run_cfg $d 64 1; done
echo "===== SEQ-LEN SWEEP (d=256, batch=1) ====="
for seq in 16 64 256; do run_cfg 256 $seq 1; done
echo "===== BATCH SWEEP (d=256, seq=16) ====="
for batch in 1 4 16; do run_cfg 256 16 $batch; done
echo BENCH_SWEEP_DONE
