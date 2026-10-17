#!/usr/bin/env bash
# One line per P8b block summary (build/tsmc65/summary/hier_*.txt): area, memory area, flops / latches, setup WNS /
# TNS (all endpoints and into flops), ff hold WNS / violators, DC wall time, peak RSS, worst flop path.
#   flow/tsmc65/tools/summarize_hier.sh [-v] [files]      (-v: also the integrity lines and child instance counts)
# area marked * = own cells + instances x child summaries (children linked as block abstractions)
V=0; [ "${1:-}" = -v ] && { V=1; shift; }
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
[ $# -eq 0 ] && set -- "$ROOT"/build/tsmc65/summary/hier_*.txt
printf "%-8s %4s %12s %11s %7s %7s %8s %9s %8s %8s %6s %6s %6s  %s\n" block P area_um2 mem_um2 flops latches WNS TNS flopWNS holdWNS holdN wall_m memGB "worst flop path"
for f in "$@"; do
    [ -f "$f" ] || continue
    awk -v V=$V '
        $1=="block" {b=$2} $1=="period_ns" {p=$2}
        $1=="area_um2" {a=$2} $1=="mem_area_um2" {m=$2} $1=="area_flops" {fl=$4} $1=="area_latches" {la=$4}
        $1=="unit_area_um2" {ua=$2; um=$4; ul=$6; uf=$8}
        $1=="setup_wns" {w=$2; t=$4} $1=="flop_wns" {fw=$2} $1=="hold_wns" {hw=$2; hn=$6}
        $1=="flop_path" {fp=$2" -> "$4"  arr "$8}
        $1=="dc_wall_s" {ws=$2; mk=$6}
        /^time_Maximum resident/ {rss=$NF}
        /^integ_|^child|^noarea|^own_area|^T65_(ERROR|NOTE|MATCH)/ {ex[++ne]=$0}
        END {
            if (ua != "") { a = ua "*"; m = um; la = ul; fl = uf }
            mem = (rss != "") ? rss/1048576 : mk/1048576
            printf "%-8s %4s %12s %11s %7s %7s %8s %9s %8s %8s %6s %6.0f %6.1f  %s\n", b, p, a, m, fl, la, w, t, fw, hw, hn, ws/60, mem, fp
            if (V) for (i = 1; i <= ne; i++) print "         " ex[i]
        }' "$f"
done
