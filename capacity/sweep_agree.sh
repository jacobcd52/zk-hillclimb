#!/bin/bash
cd /workspace/projects/zk-hillclimb/capacity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMA_TEACHER_KERNEL=fp8_scaled_mm
PY=/root/int-model-env/bin/python
SUM=agree_runs/SUMMARY.txt; : > $SUM
run(){ tag=$1; shift; echo ">>> $tag $@" | tee -a $SUM
  $PY train_agree.py --tag "$tag" --neval 200 "$@" > agree_runs/$tag.log 2>&1
  grep -E "RESULT|Traceback|out of memory" agree_runs/$tag.log | tail -2 | tee -a $SUM; echo "" >> $SUM; }
run fp8_frozen        --mode frozen   --base fp8
run int8_frozen       --mode frozen   --base int8
run fp8_qat_kl_1e4    --mode full_qat --base fp8 --loss topk_kl --lr 1e-4 --bs 4
run fp8_qat_ce_1e4    --mode full_qat --base fp8 --loss hard_ce --lr 1e-4 --bs 4
run fp8_qat_kl_1e4_nt --mode full_qat --base fp8 --loss topk_kl --lr 1e-4 --neartie --bs 4
run fp8_qat_kl_3e4    --mode full_qat --base fp8 --loss topk_kl --lr 3e-4 --bs 4
run int8_qat_kl_1e4   --mode full_qat --base int8 --loss topk_kl --lr 1e-4 --bs 4
run int8_qat_ce_1e4   --mode full_qat --base int8 --loss hard_ce --lr 1e-4 --bs 4
run int8_qat_kl_3e4   --mode full_qat --base int8 --loss topk_kl --lr 3e-4 --bs 4
run int8_qat_gumbel_1e4 --mode full_qat --base int8 --loss gumbel --lr 1e-4 --bs 4
run int8_bias_kl_1e3  --mode bias --base int8 --loss topk_kl --lr 1e-3 --bs 8
echo "ALLDONE" | tee -a $SUM
