#!/bin/bash
# [S.2] ORCHESTRATOR -- SGWA's equal-effort RTE search. [R.306]'s r306_run.sh,
# re-pointed: stage A, stage B up to 2 rounds (stops when it emits nothing),
# stage C, stage D.  Resumable: `done` markers + the planner's manifest.
#   nohup scripts/s2_run.sh > scratchpad/s2/run.log 2>&1 &
set -u
cd /workspace/SGWA || exit 1
D=scratchpad/s2
PLAN="env/bin/python scripts/s2_plan.py"
mkdir -p "$D"
export CUDA_VISIBLE_DEVICES=0
env/bin/python src/verify_sgwa.py > "$D/verify.log" 2>&1 || { echo "[s2] verify_sgwa FAILED -- refusing to spend GPU"; exit 1; }
$PLAN --selftest || { echo "[s2] SELFTEST FAILED -- refusing to spend GPU"; exit 1; }
run_stage() {
  local st=$1
  $PLAN --generate "$st" || exit 1
  [ -s "$D/jobs_${st}.tsv" ] || { echo "[s2] stage $st empty, skipping"; return 0; }
  scripts/s2_drive.sh "$st"
}
echo "[s2] ===== START $(date +%F' '%T) ====="
run_stage A
for round in 1 2; do
  before=$(wc -l < "$D/jobs_B.tsv" 2>/dev/null || echo 0)
  run_stage B
  after=$(wc -l < "$D/jobs_B.tsv" 2>/dev/null || echo 0)
  [ "$after" -eq "$before" ] && { echo "[s2] stage B converged after $((round-1)) round(s)"; break; }
done
run_stage C
run_stage D
echo "[s2] ===== ALL STAGES DONE $(date +%F' '%T) ====="
env/bin/python scripts/s2_read.py || true
