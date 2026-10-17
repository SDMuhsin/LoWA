#!/usr/bin/env bash
# [S.4c] block-owned branches with periodic exact exchange on RTE: arm E (epoch end) and arm S (every 10 steps).
# One torchrun pair per arm, both arms concurrently (2 processes per GPU).
set -u
cd "$(dirname "$0")/.."
OUT=scratchpad/s4c; mkdir -p $OUT/csv $OUT/log
run_arm() {   # $1 arm  $2 sync_every  $3 port
  for seed in 42 43 44 45 46; do
    lab="rte-sync$1-seed${seed}"
    printf '%s\t%s\t0.166667\t0.552427\n' "$lab" "$seed" > $OUT/runs_${lab}.tsv
    env/bin/torchrun --nproc_per_node 2 --master_port $3 src/train_glue_pop.py --task_name rte \
      --runs_file $OUT/runs_${lab}.tsv --num_warmup_steps 140 --num_train_epochs 30 \
      --btm_G 2 --btm_arm disjoint --btm_sync_every $2 --out_dir $OUT/csv > $OUT/log/${lab}.log 2>&1
    echo "$(date -u +%FT%TZ) done arm $1 seed $seed"
  done
}
run_arm E 100000 29601 &
run_arm S 10 29602 &
wait
echo "$(date -u +%FT%TZ) ALL DONE"
