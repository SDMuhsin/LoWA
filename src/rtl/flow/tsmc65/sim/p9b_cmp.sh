#!/usr/bin/env bash
# P9b: gate-level window runs vs their RTL references (sim/run_p9b_ref.sh): every heartbeat line both runs printed
# (cycle, unit busy flags, DRAM write beats, write-stream signature) and every op-log event up to the shorter run.
#   flow/tsmc65/sim/p9b_cmp.sh [gls_dir [plan]]       (default gls2, tb/power/win/p9b_plan.txt)
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
B=$ROOT/build/tsmc65
G=$B/${1:-gls2}
while read -r n st a b _; do
    case $n in ''|\#*) continue ;; esac
    t=$G/w_$n/sim.out; R=$B/gls_rtl; [ -d "$B/gls_rtl25/ref_$st" ] && R=$B/gls_rtl25; r=$R/ref_$st/sim.out
    [ -f "$t" ] && [ -f "$r" ] || { echo "CMP $n missing"; continue; }
    grep '^HB cyc' "$t" | sort > /tmp/lwta_p9b/cmp_t.txt; grep '^HB cyc' "$r" | sort > /tmp/lwta_p9b/cmp_r.txt
    nt=$(wc -l < /tmp/lwta_p9b/cmp_t.txt); nc=$(comm -12 /tmp/lwta_p9b/cmp_t.txt /tmp/lwta_p9b/cmp_r.txt | wc -l)
    last=$(tail -n 1 /tmp/lwta_p9b/cmp_t.txt | cut -d' ' -f3)
    ot=$G/w_$n/oplog.txt; orf=$R/ref_$st/oplog.txt
    no=$(wc -l < "$ot"); nos=$(head -n "$no" "$orf" | diff -q - "$ot" > /dev/null && echo same || echo DIFF)
    echo "CMP $n stim $st heartbeats $nc/$nt identical (to cycle $last; last: $(tail -n 1 /tmp/lwta_p9b/cmp_t.txt | cut -d' ' -f5-)) oplog $no events $nos"
done < "${2:-$ROOT/tb/power/win/p9b_plan.txt}"
