#!/usr/bin/env bash
# [S.4b] branch-train-merge driver: for each arm and seed, branch g runs on cuda:g as its own process.
# Zero communication between branches; the merge is done afterwards by scripts/s4b_read.py.
set -u
cd "$(dirname "$0")/.."
OUT=scratchpad/s4b
mkdir -p $OUT/csv $OUT/state $OUT/log
for arm in disjoint shared; do
  for seed in 42 43 44 45 46; do
    for g in 0 1; do
      lab="rte-${arm}-b${g}-seed${seed}"
      printf '%s\t%s\t0.166667\t0.552427\n' "$lab" "$seed" > $OUT/runs_${lab}.tsv
      env/bin/python src/train_glue_pop.py --task_name rte --runs_file $OUT/runs_${lab}.tsv \
        --num_warmup_steps 140 --num_train_epochs 30 --btm_G 2 --btm_branch $g --btm_arm $arm \
        --state_dir $OUT/state --out_dir $OUT/csv --device cuda:$g > $OUT/log/${lab}.log 2>&1 &
    done
    wait
    echo "$(date -u +%FT%TZ) done $arm seed $seed"
  done
done
echo "$(date -u +%FT%TZ) ALL DONE"
