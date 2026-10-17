#!/usr/bin/env bash
# One DC block synthesis on the server:  flow/tsmc65/syn/run_block.sh <block> <period_ns> [corner=ss] [topo=1]
# Output (stays on the server): build/tsmc65/syn/<block>_<corner>_p<period>/ (dc.log, netlist, ddc, rpt/);
# summary.txt (numbers + RTL instance names) is copied to build/tsmc65/summary/ and is the only thing that comes back.
# Peak resident memory of the DC process tree comes from GNU time when available (time.txt).
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
block=$1; period=$2; corner=${3:-ss}; topo=${4:-1}
tag=${block}_${corner}_p${period}
out=$ROOT/build/tsmc65/syn/$tag
mkdir -p "$out" "$ROOT/build/tsmc65/summary"
export T65_ROOT=$ROOT T65_BLOCK=$block T65_PERIOD=$period T65_CORNER=$corner T65_TOPO=$topo T65_OUT=$out
export T65_CORES=${T65_CORES:-8}
mode=""; [ "$topo" = 1 ] && mode=-topographical_mode
cd "$out" || exit 1
TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v -o $out/time.txt"
$TIMEV tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; dc_shell $mode -f $ROOT/flow/tsmc65/syn/dc_block.tcl" > dc.log 2>&1
rc=$?
if [ -f summary.txt ]; then
    cp -f summary.txt "$ROOT/build/tsmc65/summary/syn_$tag.txt"
    [ -f time.txt ] && grep -E "Maximum resident|Elapsed" time.txt | sed 's/^\s*/time_/' >> "$ROOT/build/tsmc65/summary/syn_$tag.txt"
fi
# PROC-3 (DC helper child cannot connect back on this host) is harmless, counted out
echo "DC_DONE $tag rc $rc errors $(grep '^Error' dc.log | grep -vc PROC-3) summary $([ -f summary.txt ] && echo yes || echo no)"
