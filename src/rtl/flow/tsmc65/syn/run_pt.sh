#!/usr/bin/env bash
# PrimeTime corner timing of finished P8b blocks:  flow/tsmc65/syn/run_pt.sh <corner> <block> ...   (one at a time)
# Summary (numbers + RTL instance names) -> build/tsmc65/summary/pt_<block>_<corner>.txt; end marker PT_DONE.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
corner=$1; shift
for b in "$@"; do
    out=$ROOT/build/tsmc65/hier/$b
    [ -f "$out/design.txt" ] || { echo "PT_SKIP $b (not built)"; continue; }
    export T65_ROOT=$ROOT T65_BLOCK=$b T65_PERIOD=${T65_PERIOD:-4.0} T65_CORNER=$corner T65_OUT=$out
    cd "$out" || exit 1
    TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v -o $out/pt_time_$corner.txt"
    $TIMEV tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; setenv LD_PRELOAD /usr/lib64/libz.so.1; pt_shell -f $ROOT/flow/tsmc65/sta/pt_corner.tcl" > pt_$corner.log 2>&1
    s=$ROOT/build/tsmc65/summary/pt_${b}_$corner.txt
    if [ -f "pt_$corner.summary.txt" ]; then
        cp -f "pt_$corner.summary.txt" "$s"
        [ -f "pt_time_$corner.txt" ] && grep -E "Maximum resident|Elapsed" "pt_time_$corner.txt" | sed 's/^\s*/time_/' >> "$s"
    fi
    echo "PT_DONE1 $b $corner errors $(grep -c '^Error' pt_$corner.log) summary $([ -f "$s" ] && echo yes || echo no)"
done
echo "PT_DONE"
