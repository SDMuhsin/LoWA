#!/usr/bin/env bash
# P8d exit rule (PLAN R14 / R15): vu16 at 5.0 ns, at 5.25 ns if 5.0 misses; then gm16 at the closed period.
# Each run gets its own output directory hier/<block>_p<period> (syn/run_hier.sh T65_TAG); the chip links them
# with T65_KID_TAG.  Log lines: EXIT_RUN, EXIT_WNS, EXIT_DONE period <p>.
#   cd <tree>/src/rtl || exit 1; nohup flow/tsmc65/syn/run_exit.sh > build/tsmc65/hier_exit.log 2>&1 < /dev/null &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
wns() { awk '$1 == "setup_wns" { print $2 }' "$ROOT/build/tsmc65/summary/hier_$1.txt" 2>/dev/null; }
met() { [ -n "$1" ] && awk -v w="$1" 'BEGIN { exit !(w >= 0) }'; }
closed=""
for p in ${EXIT_PERIODS:-5.0 5.25}; do
    echo "EXIT_RUN vu16 $p $(date +%H:%M:%S)"
    T65_PERIOD=$p T65_TAG=_p$p FORCE=1 "$HERE/run_hier.sh" vu16
    w=$(wns vu16_p$p); echo "EXIT_WNS vu16 $p ${w:-none}"
    if met "$w"; then closed=$p; break; fi
done
[ -n "$closed" ] || { echo "EXIT_DONE period none (vu16 misses every candidate)"; exit 1; }
echo "EXIT_RUN gm16 $closed $(date +%H:%M:%S)"
T65_PERIOD=$closed T65_TAG=_p$closed FORCE=1 "$HERE/run_hier.sh" gm16
w=$(wns gm16_p$closed); echo "EXIT_WNS gm16 $closed ${w:-none}"
echo "EXIT_DONE period $closed gm16_met $(met "$w" && echo yes || echo no)"
