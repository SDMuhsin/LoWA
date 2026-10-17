#!/usr/bin/env bash
# [S.5] computational metrics on a QUIET cuda:0: per task, 2 epochs of (a) the full single-process cell
# and (b) one branch (half data, 128 atoms), both through train_glue_pop.py so the code path is identical.
# Reads the "[pop] epoch 1 ..." line (train s, eval s, peak MiB).  Writes scratchpad/s5/timing/summary.tsv.
set -u
cd "$(dirname "$0")/.."
OUT=scratchpad/s5/timing; mkdir -p $OUT/csv $OUT/state
declare -A WARM=( [rte]=140 [cb]=14 [mrpc]=206 [stsb]=323 [cola]=481 [boolq]=529 [sst2]=3778 )
printf 'task\tmode\ttrain_s\teval_s\tpeak_mib\tsteps_per_epoch\n' > $OUT/summary.tsv
for task in rte cb mrpc stsb cola boolq sst2; do
  for mode in full branch; do
    if [ $mode = full ]; then G=1; else G=2; fi
    lab="t-${task}-${mode}"
    printf '%s\t42\t0.166667\t0.552427\n' "$lab" > $OUT/runs_${lab}.tsv
    env/bin/python src/train_glue_pop.py --task_name $task --runs_file $OUT/runs_${lab}.tsv \
      --num_warmup_steps ${WARM[$task]} --num_train_epochs 2 --btm_G $G --btm_branch 0 --btm_arm disjoint \
      --out_dir $OUT/csv --device cuda:0 > $OUT/${lab}.log 2>&1
    line=$(grep "\[pop\] epoch  1" $OUT/${lab}.log)
    tr_s=$(echo "$line" | sed -E 's/.*train +([0-9.]+)s.*/\1/'); ev_s=$(echo "$line" | sed -E 's/.*eval +([0-9.]+)s.*/\1/')
    pk=$(echo "$line" | sed -E 's/.*peak ([0-9,]+) MiB.*/\1/' | tr -d ,)
    f=$OUT/csv/${lab}.csv; [ -f $OUT/csv/${lab}-b0.csv ] && f=$OUT/csv/${lab}-b0.csv   # branches carry a -b<g> suffix
    n=$(env/bin/python -c "import csv;r=list(csv.DictReader(open('$f')))[-1];print(r.get('steps_per_epoch','?'))" 2>/dev/null || echo "?")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$mode" "$tr_s" "$ev_s" "$pk" "$n" >> $OUT/summary.tsv
    echo "$(date -u +%FT%TZ) $task $mode train $tr_s s eval $ev_s s peak $pk MiB"
  done
done
cat $OUT/summary.tsv
