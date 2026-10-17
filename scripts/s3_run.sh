#!/bin/bash
# [S.3] SGWA on the paper's six non-RTE RoBERTa columns, both [S.2] configs, seeds 42-46.
# Cheapest task first, so every stopping point is a set of COMPLETE columns. Resumable.
#   nohup scripts/s3_run.sh > scratchpad/s3/run.log 2>&1 &
set -u
cd /workspace/SGWA || exit 1
export CUDA_VISIBLE_DEVICES=0
env/bin/python src/verify_sgwa.py > scratchpad/s3/verify.log 2>&1 || { echo "[s3] verify_sgwa FAILED"; exit 1; }
for t in cb cola mrpc stsb boolq sst2; do
  echo "[s3] ---- task $t begins $(date +%F' '%T) ----"
  scripts/s3_drive.sh "$t"
done
echo "[s3] ===== ALL TASKS DONE $(date +%F' '%T) ====="
