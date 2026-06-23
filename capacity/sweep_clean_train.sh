#!/bin/bash
cd /workspace/projects/zk-hillclimb/capacity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IMA_TEACHER_KERNEL=fp8_scaled_mm
PY=/root/int-model-env/bin/python
SUM=agree_runs/CLEAN_SUMMARY.txt; : > $SUM
R(){ tag=$1; shift; echo ">>> $tag $@"|tee -a $SUM; $PY train_agree.py --tag $tag --neval 250 --loss mse5 --dtype fp32 --mode full_qat "$@" > agree_runs/$tag.log 2>&1; grep -E "RESULT|out of memory|Traceback" agree_runs/$tag.log|tail -1|tee -a $SUM; echo "">>$SUM; }
R clean_int8_t16_3e7 --base int8 --topm 16 --lr 3e-7 --bs 2 --max_steps 1200 --eval_every 200
R clean_int8_t16_1e6 --base int8 --topm 16 --lr 1e-6 --bs 2 --max_steps 1200 --eval_every 200
R clean_int8_t64_3e7 --base int8 --topm 64 --lr 3e-7 --bs 2 --max_steps 1200 --eval_every 200
R clean_fp8_t16_3e7  --base fp8  --topm 16 --lr 3e-7 --bs 4 --max_steps 1000 --eval_every 200
R clean_fp8_t64_3e7  --base fp8  --topm 64 --lr 3e-7 --bs 4 --max_steps 1000 --eval_every 200
echo ALLDONE|tee -a $SUM
