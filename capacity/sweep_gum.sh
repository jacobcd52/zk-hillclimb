#!/bin/bash
cd /workspace/projects/zk-hillclimb/capacity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IMA_TEACHER_KERNEL=fp8_scaled_mm
PY=/root/int-model-env/bin/python
SUM=agree_runs/GUM_SUMMARY.txt; : > $SUM
R(){ tag=$1; shift; echo ">>> $tag $@" | tee -a $SUM; $PY train_agree.py --tag $tag --neval 200 --bs 4 "$@" > agree_runs/$tag.log 2>&1; grep -E "RESULT|out of memory|Traceback" agree_runs/$tag.log|tail -2|tee -a $SUM; echo "">>$SUM; }
R int8_head_gum_1e3 --mode head --base int8 --loss gumbel --lr 1e-3 --max_steps 800 --eval_every 200
R fp8_head_gum_1e3  --mode head --base fp8  --loss gumbel --lr 1e-3 --max_steps 800 --eval_every 200
R int8_qat_gum_3e6  --mode full_qat --base int8 --loss gumbel --lr 3e-6 --max_steps 400 --eval_every 100
echo ALLDONE|tee -a $SUM
