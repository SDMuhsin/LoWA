#!/usr/bin/env bash
# Results-summary extractor: one line per DC summary (block, period, area, setup / flop WNS + TNS, hold WNS,
# worst-path start -> end, DC wall time, peak memory).  Prints summary numbers only.
#   flow/tsmc65/tools/summarize.sh [build/tsmc65/summary/syn_*.txt]   (-v: also the top-5 paths)
V=0; [ "${1:-}" = -v ] && { V=1; shift; }
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
[ $# -eq 0 ] && set -- "$ROOT"/build/tsmc65/summary/syn_*.txt
printf "%-9s %5s %11s %11s %9s %10s %9s %8s %7s %7s  %s\n" block P_ns area_um2 dut_um2 WNS TNS flopWNS holdWNS wall_s memMB "worst path"
for f in "$@"; do
    [ -f "$f" ] || continue
    awk -v V=$V '
        $1=="block" {b=$2} $1=="period_ns" {p=$2}
        $1=="area_um2" {a=$2} $1=="dut_area_um2" {d=$2}
        $1=="setup_wns" {w=$2; t=$4} $1=="flop_wns" {fw=$2} $1=="hold_wns" {hw=$2}
        $1=="setup_path" {sp=$2" -> "$4}
        $1=="dc_wall_s" {ws=$2; mk=$6}
        /^time_Maximum resident/ {rss=$NF}
        $1=="top_path" {tp[$2]=$3" -> "$5"  "$7}
        END {
            mem = (rss != "") ? rss/1024 : mk/1024
            printf "%-9s %5s %11s %11s %9s %10s %9s %8s %7s %7.0f  %s\n", b, p, a, d, w, t, fw, hw, ws, mem, sp
            if (V) for (i = 1; i <= 5; i++) if (i in tp) printf "          top%d %s\n", i, tp[i]
        }' "$f"
done
