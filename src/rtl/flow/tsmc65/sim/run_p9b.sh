#!/usr/bin/env bash
# P9b batch: gate-level SAIF windows of the reduced step (1 layer, B 2, L 32), one simulation per window of
# tb/power/win/p9b_plan.txt ("name stimulus first end"), at most GLS_PAR at a time.  Stimuli in build/tsmc65/stim/<dir>
# (l1 = the whole program, c<k> = tb/power/mk_cut_stim.py --cut <k>), copied up beforehand.  Each window's SAIF is
# appended to the PrimePower window list GLS_WINLIST (default <dir>/p9b_windows.txt; sta/run_pt_gls.sh reads it while
# running).  GLS_DIR: simulation directory under build/tsmc65 (default gls), passed on to sim/run_gls.sh.
#   flow/tsmc65/sim/run_p9b.sh [names]        default: every window of the plan (GLS_PLAN); GLS_NOBUILD=1 skips the compile
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || exit 1
PAR=${GLS_PAR:-2}
G=$ROOT/build/tsmc65/${GLS_DIR:-gls}
WL=${GLS_WINLIST:-$G/p9b_windows.txt}
[ "${GLS_NOBUILD:-0}" = 1 ] || flow/tsmc65/sim/run_gls.sh build || exit 1
mkdir -p "$G/win"
sel=${1:-}
while read -r n st a b _; do
    case $n in ''|\#*) continue ;; esac
    [ -n "$sel" ] && ! echo " $sel " | grep -q " $n " && continue
    # at most PAR gate-level simulations of this user on the host (any batch, RTL references included)
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ] || [ "$(pgrep -u "$USER" -c -x simv)" -ge "$PAR" ]; do sleep 20; done
    echo "$n $a $b" > "$G/win/$n.txt"
    echo "$n $G/w_$n/$n.saif" >> "$WL"
    echo "GLS_WIN_START $n $st $a $b $(date -u +%H:%M)"
    ( flow/tsmc65/sim/run_gls.sh run "w_$n" "build/tsmc65/stim/$st" "$G/win/$n.txt" +hb=10000 +maxcyc=$((b + 100000)) ) &
    sleep 5
done < "${GLS_PLAN:-tb/power/win/p9b_plan.txt}"
wait
echo "GLS_ALL_DONE $(date -u +%H:%M)"
