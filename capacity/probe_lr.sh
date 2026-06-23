#!/bin/bash
cd /workspace/projects/zk-hillclimb/capacity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IMA_TEACHER_KERNEL=fp8_scaled_mm
PY=/root/int-model-env/bin/python
P(){ tag=$1; shift; echo ">>> $tag $@"; $PY train_agree.py --tag $tag --neval 40 --eval_every 25 --max_steps 150 "$@" > agree_runs/$tag.log 2>&1; grep -E "^\[|RESULT|out of memory|Traceback" agree_runs/$tag.log | tail -9; echo; }
P pr_int8_kl_1e5 --mode full_qat --base int8 --loss topk_kl --lr 1e-5 --bs 4
P pr_int8_kl_1e6 --mode full_qat --base int8 --loss topk_kl --lr 1e-6 --bs 4
P pr_int8_ce_1e5 --mode full_qat --base int8 --loss hard_ce --lr 1e-5 --bs 4
P pr_int8_bias_kl_1e4 --mode bias --base int8 --loss topk_kl --lr 1e-4 --bs 8 --max_steps 300 --eval_every 50
echo PROBEDONE
