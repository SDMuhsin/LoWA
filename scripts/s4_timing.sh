#!/bin/bash
# [S.4] quiet-GPU timing for S4-P3/S4-P4.  Run ONLY when nothing else is on cuda:0.
#   2 epochs each: population merged (R=10), population stencil (R=10),
#   one train_glue RTE cell alone, three concurrent train_glue RTE cells.
set -u; cd /workspace/SGWA || exit 1
export CUDA_VISIBLE_DEVICES=0
D=scratchpad/s4/timing; mkdir -p $D
CELL="--model_name_or_path roberta-base --task_name rte --dtype float32 --adapter_target_modules query,value --per_device_train_batch_size 32 --num_train_epochs 2 --num_warmup_steps 140 --optimizer adamw-sgwa --sgwa_k 256 --sgwa_K 3 --sgwa_seed 777 --sgwa_materialise 1 --sgwa_target_modules query,value --sgwa_scaling 0.552427 --learning_rate 0.166667 --classifier_lr 5e-3 --weight_decay 0.01"
for M in merged stencil; do
  t0=$(date +%s)
  env/bin/python -u src/train_glue_pop.py --task_name rte --runs_file scratchpad/s4/runs_rte_R10.tsv --mode $M --num_train_epochs 2 --num_warmup_steps 140 --out_dir $D/csv_$M > $D/pop_$M.log 2>&1
  echo "pop_$M R=10 2ep wall $(( $(date +%s) - t0 )) s" | tee -a $D/summary.txt
done
t0=$(date +%s)
GLUE_SEEDS=42 GLUE_RESULTS_FILE=$D/single.csv env/bin/python -u src/train_glue.py $CELL --name s4t-single > $D/single.log 2>&1
echo "single cell 2ep wall $(( $(date +%s) - t0 )) s" | tee -a $D/summary.txt
t0=$(date +%s)
for s in 42 43 44; do GLUE_SEEDS=$s GLUE_RESULTS_FILE=$D/conc$s.csv env/bin/python -u src/train_glue.py $CELL --name s4t-conc$s > $D/conc$s.log 2>&1 & done; wait
echo "3 concurrent cells 2ep wall $(( $(date +%s) - t0 )) s" | tee -a $D/summary.txt
echo "S4 TIMING DONE" | tee -a $D/summary.txt
