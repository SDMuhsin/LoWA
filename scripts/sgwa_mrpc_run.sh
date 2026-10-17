#!/bin/bash
# [S.1] SGWA on MRPC, the [R.310] protocol verbatim, seeds 42-46, 3 workers on cuda:0.
# Prereg: llmdocs/S1_sgwa_mrpc_prereg.md (sha256 in results/sgwa/PREREG_SHA256).
# Resumable: a seed with a CSV is skipped.
set -u
cd /workspace/SGWA || exit 1
D=results/sgwa/mrpc; mkdir -p "$D/csv" "$D/logs"
export CUDA_VISIBLE_DEVICES=0
env/bin/python src/verify_sgwa.py > "$D/verify.log" 2>&1 || { echo "GATES FAILED"; exit 1; }
ARGS="--model_name_or_path roberta-base --task_name mrpc --dtype float32 --adapter_target_modules query,value --per_device_train_batch_size 32 --num_train_epochs 30 --num_warmup_steps 206 --optimizer adamw-sgwa --sgwa_k 256 --sgwa_K 3 --sgwa_seed 777 --sgwa_materialise 1 --sgwa_target_modules query,value --learning_rate 0.16666667 --classifier_lr 5e-3 --weight_decay 0.01"
run() {
  s=$1; lab="mrpc-sgwa-seed$s"
  [ -s "$D/csv/$lab.csv" ] && { echo "skip $lab"; return; }
  echo "START $lab $(date +%T)"
  GLUE_SEEDS=$s GLUE_RESULTS_FILE="$D/csv/$lab.csv" env/bin/python -u src/train_glue.py $ARGS --name "$lab" > "$D/logs/$lab.log" 2>&1
  echo "END $lab rc=$? $(date +%T)"
}
export -f run; export D ARGS
printf "%s\n" 42 43 44 45 46 | xargs -P 3 -I{} bash -c 'run {}'
echo "ALL DONE $(date +%T)"
