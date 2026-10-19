#!/usr/bin/env bash
# P8e (PLAN R16): bottom-up DC of blocks of common/hier.txt in the SRAM-macro variant (or the latch-RF baseline
# through the same script), one DC at a time, on the server:
#   T65_VARIANT=sram|latch T65_PERIOD=5.0 flow/tsmc65/sram/run_sram.sh <unit|block> ...
# Same block table, ordering and child reuse as syn/run_hier.sh; runs syn/dc_unit.tcl through sram/dc_sram.tcl.
# Outputs (stay on the server): build/tsmc65/hier_<variant>_p<period>/<block>/; summaries (numbers only) in
# build/tsmc65/summary/sram/<variant>_p<period>_<block>.txt.  Markers: START / DC_DONE per block, SRAM_HIER_DONE.
#   cd <tree>/src/rtl || exit 1; nohup flow/tsmc65/sram/run_sram.sh bank256 > build/sram/run.log 2>&1 < /dev/null &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
TAB=$HERE/../common/hier.txt
. "$HERE/../site/sram_site.sh"
export T65_MC_OUT T65_MC_CORNERS T65_MC_SHAPES
V=${T65_VARIANT:-sram}; P=${T65_PERIOD:-5.0}; C=${T65_CORNER:-ss}
export T65_VARIANT=$V
HR=$ROOT/build/tsmc65/hier_${V}_p$P
SUM=$ROOT/build/tsmc65/summary/sram
mkdir -p "$HR" "$SUM"
kids()  { awk -v b="$1" '$1 == b && $1 !~ /^[#@]/ { if ($4 != "-") { gsub(",", " ", $4); print $4 } }' "$TAB"; }
units() { awk -v u="$1" '$1 == "@unit" && $2 == u { for (i = 3; i <= NF; i++) print $i }' "$TAB"; }
known() { awk -v b="$1" '$1 == b && $1 !~ /^[#@]/ { f = 1 } END { exit !f }' "$TAB"; }
ORDER=""
visit() { local k; for k in $(kids "$1"); do visit "$k"; done; case " $ORDER " in *" $1 "*) ;; *) ORDER="$ORDER $1" ;; esac; }
for a in "$@"; do
    u=$(units "$a")
    if [ -n "$u" ]; then for b in $u; do visit "$b"; done
    elif known "$a"; then visit "$a"
    else echo "unknown unit / block $a"; exit 1; fi
done
echo "ORDER$ORDER variant $V period $P corner $C"
for b in $ORDER; do
    out=$HR/$b
    if [ -f "$out/design.txt" ] && [ "${FORCE:-0}" != 1 ]; then echo "REUSE $b"; continue; fi
    # T65_WAIT_IDLE=1 (unit / top runs on a shared server): start a block only when no other synthesis, equivalence,
    # timing or simulation job of this account is running (this script runs one block at a time itself)
    if [ "${T65_WAIT_IDLE:-0}" = 1 ]; then
        w=0
        while ps -u "$(id -un)" -o args= | grep -qE '^[^ ]*(dc_shell|fm_shell|pt_shell|vcs|simv)'; do
            [ $w = 0 ] && echo "WAIT_IDLE $b $(date +%H:%M:%S)"; w=1; sleep 300
        done
    fi
    # T65_RSS_CAP=<GB> (P8e under the PI's shared-server rule): start a block only when the box's total resident memory
    # (used memory, free -g) plus the block's expected need (T65_NEED_<block> GB, default 4; gm16 13) stays within the cap
    if [ -n "${T65_RSS_CAP:-}" ]; then
        nv=T65_NEED_$b; need=${!nv:-4}; [ "$b" = gm16 ] && need=${!nv:-13}
        w=0
        while :; do
            used=$(free -g | awk '/^Mem/ { print $3 }')   # used memory (a sum of RSS counts shared pages repeatedly)
            [ $((used + need)) -le "$T65_RSS_CAP" ] && break
            [ $w = 0 ] && echo "WAIT_RSS $b used ${used} GB need ${need} GB cap $T65_RSS_CAP $(date +%H:%M:%S)"; w=1; sleep 120
        done
    fi
    echo "START $b $(date +%H:%M:%S)"
    mkdir -p "$out"; rm -f "$out/design.txt" "$out/summary.txt"
    (
        export T65_ROOT=$ROOT T65_BLOCK=$b T65_PERIOD=$P T65_CORNER=$C T65_TOPO=${T65_TOPO:-1} T65_OUT=$out
        cv=T65_CORES_$b; export T65_CORES=${!cv:-${T65_CORES:-4}} T65_FP=0
        mode=""; [ "$T65_TOPO" = 1 ] && mode=-topographical_mode
        cd "$out" || exit 1
        /usr/bin/time -v -o "$out/time.txt" tcsh -f -c \
            "source $ROOT/flow/tsmc65/common/env.csh; dc_shell $mode -f $ROOT/flow/tsmc65/sram/dc_sram.tcl" > dc.log 2>&1
    )
    rm -rf "$out/work"     # elaboration library: not read again (parents link the .ddc); keeps the quota lean
    s=$SUM/${V}_p${P}_$b.txt
    if [ -f "$out/summary.txt" ]; then
        cp -f "$out/summary.txt" "$s"
        grep -E "Maximum resident|Elapsed" "$out/time.txt" | sed 's/^\s*/time_/' >> "$s"
        "$ROOT/flow/tsmc65/tools/dclog_check.sh" "$out" >> "$s" 2>/dev/null
        grep -E '^T65_(SRAM|DESIGN|ERROR|NOTE)' "$out/dc.log" >> "$s"
    fi
    r="DC_DONE $b errors $(grep '^Error' "$out/dc.log" | grep -vc PROC-3) t65err $(grep -c '^T65_ERROR' "$out/dc.log") sram_done $(grep -c '^T65_SRAM_DONE' "$out/dc.log") summary $([ -f "$out/summary.txt" ] && echo yes || echo no)"
    echo "$r"
    case "$r" in *"summary yes"*) ;; *) echo "SRAM_HIER_FAILED $b"; echo "SRAM_HIER_DONE $(date +%H:%M:%S)"; exit 1 ;; esac
done
echo "SRAM_HIER_DONE $(date +%H:%M:%S)"
