#!/bin/bash
cd /workspace/projects/zk-hillclimb/capacity
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IMA_TEACHER_KERNEL=fp8_scaled_mm
PY=/root/int-model-env/bin/python
for cfg in "int8 bf16" "fp8 fp32"; do set -- $cfg; b=$1; dt=$2
  echo ">>> frozen $b"; $PY train_agree.py --tag clean_frozen_$b --mode frozen --base $b --dtype $dt --neval 300 > agree_runs/clean_frozen_$b.log 2>&1
  grep -E "^\[init\]" agree_runs/clean_frozen_$b.log; done
echo CLEANFROZENDONE
