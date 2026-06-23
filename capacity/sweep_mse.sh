#!/bin/bash
cd /workspace/projects/zk-hillclimb/capacity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IMA_TEACHER_KERNEL=fp8_scaled_mm
PY=/root/int-model-env/bin/python
SUM=agree_runs/MSE_SUMMARY.txt; : > $SUM
R(){ tag=$1; shift; echo ">>> $tag $@" | tee -a $SUM; $PY train_agree.py --tag $tag --neval 150 --bs 4 --loss mse5 --dtype fp32 "$@" > agree_runs/$tag.log 2>&1; grep -E "^\[init|^\[300|^\[250|RESULT|out of memory|Traceback" agree_runs/$tag.log | tail -3 | tee -a $SUM; echo "">>$SUM; }
R mse5_qat_1e6 --mode full_qat --base int8 --lr 1e-6 --max_steps 300 --eval_every 50
R mse5_qat_3e6 --mode full_qat --base int8 --lr 3e-6 --max_steps 300 --eval_every 50
R mse5_qat_1e5 --mode full_qat --base int8 --lr 1e-5 --max_steps 300 --eval_every 50
R mse5_qat_3e5 --mode full_qat --base int8 --lr 3e-5 --max_steps 300 --eval_every 50
R mse5_head_1e3 --mode head --base int8 --lr 1e-3 --max_steps 600 --eval_every 150
echo ALLDONE | tee -a $SUM
