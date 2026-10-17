#!/bin/bash
# [G.1] SGWA on gemma-2b, Table II protocol, RTE/MRPC/STS-B/CoLA x seeds 42-46 (llmdocs/G1_sgwa_gemma_prereg.md).
# One cell per GPU (28.9 GiB peak).  Worker 0 on cuda:0; worker 1 on cuda:1 claims a cell only when cuda:1
# has >= G1_MIN_FREE MiB free (another tenant).  Claims are mkdir-atomic; a failed cell is released and
# re-run by the next pass; completion = rc 0 AND a non-empty per-cell CSV.  Resumable.
#   nohup scripts/g1_run.sh > scratchpad/g1/run.log 2>&1 &
set -u
cd /workspace/SGWA || exit 1
D=scratchpad/g1
mkdir -p $D/{csv,logs,done,claim,failed}
MIN_FREE=${G1_MIN_FREE:-31000}
COMMON="--model_name_or_path google/gemma-2b --dtype float32 --adapter_target_modules q_proj,o_proj --per_device_train_batch_size 32 --optimizer adamw-sgwa --sgwa_k 256 --sgwa_K 3 --sgwa_seed 777 --sgwa_materialise 1 --sgwa_target_modules q_proj,o_proj --sgwa_scaling 0.552427 --learning_rate 0.166667 --classifier_lr 5e-3 --weight_decay 0.01"
JOBS=$D/jobs.tsv
: > $JOBS
for spec in rte:20:93 mrpc:20:138 stsb:15:162 cola:12:192; do        # stage-05 epochs / warmup
  IFS=: read -r t e w <<< "$spec"
  for s in 42 43 44 45 46; do
    printf '%s\t%s\t%s\n' "g1-$t-sgwa-seed$s" "$s" "$COMMON --task_name $t --num_train_epochs $e --num_warmup_steps $w" >> $JOBS
  done
done

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

worker() {   # $1 = gpu
  local gpu=$1
  while IFS=$'\t' read -r label seed args; do
    [ -f "$D/done/$label" ] && continue
    if [ "$gpu" = 1 ]; then
      [ "$(free_mib 1)" -ge "$MIN_FREE" ] || continue
    fi
    mkdir "$D/claim/$label" 2>/dev/null || continue
    local t0=$(date +%s)
    echo "[g1][gpu$gpu] START $label $(date -u +%FT%TZ)"
    CUDA_VISIBLE_DEVICES=$gpu GLUE_SEEDS=$seed GLUE_RESULTS_FILE="$D/csv/$label.csv" \
      env/bin/python -u src/train_glue.py $args --name "$label" --output_dir "$D/out" > "$D/logs/$label.log" 2>&1
    local rc=$? dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ] && [ -s "$D/csv/$label.csv" ]; then
      touch "$D/done/$label"; echo "[g1][gpu$gpu] OK   $label ${dt}s"
    else
      echo "rc=$rc dt=${dt}s gpu=$gpu $(date -u +%FT%TZ)" >> "$D/failed/$label"
      rmdir "$D/claim/$label" 2>/dev/null
      echo "[g1][gpu$gpu] FAIL $label rc=$rc ${dt}s" >&2
    fi
  done < $JOBS
}

gpu1_loop() {   # cuda:1 is opportunistic: re-scan every 10 min while cuda:0's worker is still working
  while [ ! -f $D/.pass_over ]; do worker 1; sleep 600; done
}
for pass in 1 2 3; do              # later passes pick up failed / skipped cells
  rm -f $D/.pass_over
  gpu1_loop & g1pid=$!
  worker 0
  touch $D/.pass_over; wait $g1pid
  n=$(ls $D/done | wc -l)
  echo "[g1] pass $pass: $n/20 done $(date -u +%FT%TZ)"
  [ "$n" -ge 20 ] && break
  rm -rf $D/claim/*                # nothing is running between passes
done
echo "[g1] EXIT $(ls $D/done | wc -l)/20 $(date -u +%FT%TZ)"
