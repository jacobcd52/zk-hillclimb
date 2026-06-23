#!/bin/bash
cd /workspace/projects/zk-hillclimb/capacity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IMA_TEACHER_KERNEL=fp8_scaled_mm
PY=/root/int-model-env/bin/python
SUM=agree_runs/MSE16_SUMMARY.txt; : > $SUM
R(){ tag=$1; shift; echo ">>> $tag $@" | tee -a $SUM; $PY train_agree.py --tag $tag --neval 200 --bs 4 --loss mse5 --dtype fp32 --topm 16 "$@" > agree_runs/$tag.log 2>&1; grep -E "RESULT|out of memory|Traceback" agree_runs/$tag.log|tail -2|tee -a $SUM; echo "">>$SUM; }
R mse16_int8_1e6 --mode full_qat --base int8 --lr 1e-6 --max_steps 600 --eval_every 100
R mse16_int8_3e7 --mode full_qat --base int8 --lr 3e-7 --max_steps 800 --eval_every 100
R mse16_int8_3e6 --mode full_qat --base int8 --lr 3e-6 --max_steps 400 --eval_every 100
R mse16_fp8_1e6  --mode full_qat --base fp8  --lr 1e-6 --max_steps 600 --eval_every 100
R mse16_fp8_3e7  --mode full_qat --base fp8  --lr 3e-7 --max_steps 800 --eval_every 100
echo ALLDONE | tee -a $SUM
