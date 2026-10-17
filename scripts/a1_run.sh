#!/bin/bash
# [A.1] LoWA (SGWA) ablations, llmdocs/A1_lowa_ablation_prereg.md.  STS-B + CoLA, RoBERTa-base, [S.3] protocol.
#   K ladder  {0,1,2,4,8} at gamma = 0.552427     (K = 3 reused from [S.3])
#   gamma ladder /8../x8 about 0.552427, lr fixed (ref reused from [S.3])
# Phase 1 = seeds 42-44 (+1 reproduction cell), phase 2 = seeds 45-46; the job file is in that order.
# 3 workers on cuda:0; 3 opportunistic workers on cuda:1 that claim only when >= A1_MIN_FREE MiB is free.
# Claims are mkdir-atomic; completion = rc 0 AND a non-empty per-cell CSV; failures are released and re-run by
# the next pass.  Resumable:  nohup scripts/a1_run.sh > scratchpad/a1/run.log 2>&1 &
set -u
cd /workspace/SGWA || exit 1
D=scratchpad/a1
mkdir -p $D/{csv,logs,done,claim,failed}
MIN_FREE=${A1_MIN_FREE:-6000}
WPG=${A1_WORKERS_PER_GPU:-3}
COMMON="--model_name_or_path roberta-base --dtype float32 --adapter_target_modules query,value --per_device_train_batch_size 32 --num_train_epochs 30 --optimizer adamw-sgwa --sgwa_k 256 --sgwa_seed 777 --sgwa_materialise 1 --sgwa_target_modules query,value --learning_rate 0.166667 --classifier_lr 5e-3 --weight_decay 0.01"
G=0.552427
declare -A WARM=([stsb]=323 [cola]=481)
declare -A GAMMA=([d8]=0.0690534 [d4]=0.138107 [d2]=0.276214 [x2]=1.104854 [x4]=2.209708 [x8]=4.419416)
JOBS=$D/jobs.tsv
: > $JOBS
cell() {  # label seed task K gamma
  printf '%s\t%s\t%s\n' "$1" "$2" "$COMMON --task_name $3 --num_warmup_steps ${WARM[$3]} --sgwa_K $4 --sgwa_scaling $5" >> $JOBS
}
cell a1-cola-ref-seed42 42 cola 3 $G                      # PROCESS 1.2 reproduction of [S.3] cola-sgwa-seed42
for seeds in "42 43 44" "45 46"; do
  for s in $seeds; do
    for t in stsb cola; do
      for K in 0 1 2 4 8; do cell a1-$t-K$K-seed$s $s $t $K $G; done
      for g in d8 d4 d2 x2 x4 x8; do cell a1-$t-g$g-seed$s $s $t 3 ${GAMMA[$g]}; done
    done
  done
done
N=$(wc -l < $JOBS)

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

worker() {   # $1 = gpu, $2 = worker id
  local gpu=$1 wid=$2
  while IFS=$'\t' read -r label seed args; do
    [ -f "$D/done/$label" ] && continue
    if [ "$gpu" = 1 ]; then
      [ "$(free_mib 1)" -ge "$MIN_FREE" ] || continue
    fi
    mkdir "$D/claim/$label" 2>/dev/null || continue
    local t0=$(date +%s)
    echo "[a1][gpu$gpu.w$wid] START $label $(date -u +%FT%TZ)"
    CUDA_VISIBLE_DEVICES=$gpu GLUE_SEEDS=$seed GLUE_RESULTS_FILE="$D/csv/$label.csv" \
      env/bin/python -u src/train_glue.py $args --name "$label" > "$D/logs/$label.log" 2>&1
    local rc=$? dt=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ] && [ -s "$D/csv/$label.csv" ]; then
      touch "$D/done/$label"; echo "$label $dt $gpu" >> $D/walls.txt
      echo "[a1][gpu$gpu.w$wid] OK   $label ${dt}s  ($(ls $D/done | wc -l)/$N)"
    else
      echo "rc=$rc dt=${dt}s gpu=$gpu $(date -u +%FT%TZ)" >> "$D/failed/$label"
      rmdir "$D/claim/$label" 2>/dev/null
      echo "[a1][gpu$gpu.w$wid] FAIL $label rc=$rc ${dt}s" >&2
    fi
  done < $JOBS
}

gpu1_loop() {   # opportunistic: re-scan every 5 min while cuda:0's workers are still working
  while [ ! -f $D/.pass_over ]; do worker 1 "$1"; sleep 300; done
}

CUDA_VISIBLE_DEVICES=0 env/bin/python src/verify_sgwa.py > $D/verify.log 2>&1 || { echo "[a1] verify_sgwa FAILED"; exit 1; }
echo "[a1] $N cells, $WPG workers/GPU, start $(date -u +%FT%TZ)"
for pass in 1 2 3; do
  rm -f $D/.pass_over
  p1=()
  if [ "${A1_USE_GPU1:-1}" = 1 ]; then   # A1_USE_GPU1=0: cuda:0 only (cuda:1 reserved by the user)
    for w in $(seq 1 $WPG); do gpu1_loop $w & p1+=($!); sleep 20; done
  fi
  p0=()
  for w in $(seq 1 $WPG); do worker 0 $w & p0+=($!); sleep 20; done
  wait "${p0[@]}"
  touch $D/.pass_over; [ ${#p1[@]} -gt 0 ] && wait "${p1[@]}"
  n=$(ls $D/done | wc -l)
  echo "[a1] pass $pass: $n/$N done $(date -u +%FT%TZ)"
  [ "$n" -ge "$N" ] && break
  rm -rf $D/claim/*                # nothing is running between passes
done
echo "[a1] EXIT $(ls $D/done | wc -l)/$N $(date -u +%FT%TZ)"
