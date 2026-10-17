#!/usr/bin/env bash
# [S.5] the S.4c arm picked by the prereg rule (SYNC = --btm_sync_every) on the six remaining RoBERTa columns,
# HPs unchanged.  Per task and seed one torchrun pair: rank g on cuda:g owns atoms [128g,128(g+1)) and data
# half g; blocks exchanged exactly, heads averaged.  PAIRS seed-pairs run concurrently (PAIRS procs per GPU).
set -u
cd "$(dirname "$0")/.."
OUT=scratchpad/s5
PAIRS=${PAIRS:-2}
SYNC=${SYNC:?set SYNC (steps between exchanges)}
mkdir -p $OUT/csv $OUT/log
declare -A WARM=( [cb]=14 [mrpc]=206 [stsb]=323 [cola]=481 [boolq]=529 [sst2]=3778 )   # [R.310] job files
TASKS=${TASKS:-"cb mrpc stsb cola boolq sst2"}
port=29700
for task in $TASKS; do
  seeds=(42 43 44 45 46)
  for ((i=0; i<${#seeds[@]}; i+=PAIRS)); do
    for seed in "${seeds[@]:i:PAIRS}"; do
      lab="${task}-sync-seed${seed}"
      printf '%s\t%s\t0.166667\t0.552427\n' "$lab" "$seed" > $OUT/runs_${lab}.tsv
      port=$((port+1))
      env/bin/torchrun --nproc_per_node 2 --master_port $port src/train_glue_pop.py --task_name $task \
        --runs_file $OUT/runs_${lab}.tsv --num_warmup_steps ${WARM[$task]} --num_train_epochs 30 \
        --btm_G 2 --btm_arm disjoint --btm_sync_every $SYNC --out_dir $OUT/csv > $OUT/log/${lab}.log 2>&1 &
    done
    wait
    echo "$(date -u +%FT%TZ) done $task seeds ${seeds[*]:i:PAIRS}"
  done
done
echo "$(date -u +%FT%TZ) ALL DONE"
