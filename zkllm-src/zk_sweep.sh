#!/bin/bash
cd /root/zkllm
B=/root/p3_transformer_bench; T=p3_rmsnorm_tables.bin
run(){ seq=$1; bat=$2
  bench=$($B $seq 64 2 32 256 $bat 1 $T 2>/dev/null | grep "^BENCH")
  fwd=$(/root/int-model-env/bin/python tf_fwd_bench.py $seq 64 2 32 256 $bat 30 2>/dev/null | grep "^FWD")
  pr=$(echo "$bench" | grep -oE "prove=[0-9.]+" | cut -d= -f2)
  pm=$(echo "$bench" | grep -oE "proof_mb=[0-9.]+" | cut -d= -f2)
  fms=$(echo "$fwd" | grep -oE "fwd_ms=[0-9.]+" | cut -d= -f2)
  ovh=$(/root/int-model-env/bin/python -c "print(f'{$pr*1000/$fms:.0f}')" 2>/dev/null)
  echo "seq=$seq batch=$bat tokens=$((seq*bat))  ZKprove=${pr}s proof=${pm}MB fwd=${fms}ms  ZK_OVERHEAD=${ovh}x"
}
echo "### ZK TOKENS SWEEP (batch=1) ###"
for s in 8 16 32 64; do run $s 1; done
echo "### ZK SEQ-vs-BATCH at fixed tokens=64 ###"
run 8 8; run 16 4; run 32 2; run 64 1
echo "ZKSWEEP_DONE"
