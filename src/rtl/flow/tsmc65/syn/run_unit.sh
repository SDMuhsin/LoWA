#!/usr/bin/env bash
# One block of the P8b hierarchical flow on the server:  flow/tsmc65/syn/run_unit.sh <block> [period=4.0] [corner=ss]
# Output (stays on the server): build/tsmc65/hier/<block><T65_TAG>/ (dc.log, ddc, netlist, design.txt, rpt/).  The summary
# (numbers + RTL instance names, plus the integrity scan and GNU time) goes to build/tsmc65/summary/hier_<block><T65_TAG>.txt.
# T65_FP=1: floorplanned run (syn/floorplan.tcl); T65_TAG (default empty): output-name suffix, e.g. _fp or _fp_p4.5, so
# that variants of a block do not overwrite its reference run (parents always link the untagged children).
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
block=$1; period=${2:-4.0}; corner=${3:-ss}
tag=${T65_TAG:-}
out=$ROOT/build/tsmc65/hier/$block$tag
sum=$ROOT/build/tsmc65/summary/hier_$block$tag.txt
mkdir -p "$out" "$ROOT/build/tsmc65/summary"
rm -f "$out/design.txt" "$out/summary.txt" "$out/groups.txt"
export T65_ROOT=$ROOT T65_BLOCK=$block T65_PERIOD=$period T65_CORNER=$corner T65_TOPO=${T65_TOPO:-1} T65_OUT=$out
export T65_CORES=${T65_CORES:-8} T65_FP=${T65_FP:-0}
mode=""; [ "$T65_TOPO" = 1 ] && mode=-topographical_mode
cd "$out" || exit 1
TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v -o $out/time.txt"
$TIMEV tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; dc_shell $mode -f $ROOT/flow/tsmc65/syn/dc_unit.tcl" > dc.log 2>&1
rc=$?
if [ -f summary.txt ]; then
    cp -f summary.txt "$sum"
    [ -f time.txt ] && grep -E "Maximum resident|Elapsed" time.txt | sed 's/^\s*/time_/' >> "$sum"
    "$ROOT/flow/tsmc65/tools/dclog_check.sh" "$out" >> "$sum"
    grep '^T65_' dc.log >> "$sum"
fi
[ -f groups.txt ] && cp -f groups.txt "$ROOT/build/tsmc65/summary/groups_$block$tag.txt"
echo "DC_DONE $block$tag rc $rc errors $(grep '^Error' dc.log | grep -vc PROC-3) t65err $(grep -c '^T65_ERROR' dc.log) summary $([ -f summary.txt ] && echo yes || echo no)"
