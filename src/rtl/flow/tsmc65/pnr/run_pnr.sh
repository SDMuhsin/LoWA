#!/usr/bin/env bash
# ICC2 floorplan / placement / clock tree / global route of one block (P9c):
#   flow/tsmc65/pnr/run_pnr.sh <block> [first_stage] [last_stage]      stages: ndm floorplan place cts groute
# Environment: T65_PERIOD (required), T65_UTIL (0.65), T65_CORES (8), T65_KID_TAG, T65_HIER (default
#   build/tsmc65/hier of this tree), T65_PNR_TAG (run-dir suffix).
# Output build/tsmc65/pnr/<block><tag>/ (stays on the server); per-stage summary_<stage>.txt, time_<stage>.txt
# (wall clock, peak memory) -> build/tsmc65/summary/pnr_<block><tag>_<stage>.txt.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
b=$1; first=${2:-ndm}; last=${3:-groute}
export T65_ROOT=$ROOT T65_BLOCK=$b T65_PERIOD=${T65_PERIOD:?clock period}
export T65_HIER=${T65_HIER:-$ROOT/build/tsmc65/hier}
export T65_NDM=$ROOT/build/tsmc65/pnr/ndm
export T65_PNR=$ROOT/build/tsmc65/pnr/$b${T65_PNR_TAG:-}
mkdir -p "$T65_NDM" "$T65_PNR" "$ROOT/build/tsmc65/summary"
BIN=$(sed -n 's/^set ::T65_ICC2_BIN *//p' "$ROOT/flow/tsmc65/site/site_pnr.tcl")
stages="ndm floorplan place cts groute"; run=0
for s in $stages; do
    [ "$s" = "$first" ] && run=1
    [ $run = 1 ] || continue
    echo "STAGE_START $s $(date +%H:%M:%S)"
    if [ "$s" = ndm ]; then
        [ -d "$T65_NDM/t65_std.ndm" ] || { cd "$T65_NDM" || exit 1
            tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; setenv LD_PRELOAD /usr/lib64/libz.so.1; $BIN/icc2_lm_shell -f $ROOT/flow/tsmc65/pnr/ndm.tcl" > lm.log 2>&1
            grep -q NDM_OK lm.log || { echo "PNR_FAIL ndm $T65_NDM/lm.log"; exit 1; }; }
    else
        cd "$T65_PNR" || exit 1
        /usr/bin/time -v -o time_$s.txt tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; setenv LD_PRELOAD /usr/lib64/libz.so.1; $BIN/icc2_shell -f $ROOT/flow/tsmc65/pnr/$s.tcl" > $s.log 2>&1
        sm=$ROOT/build/tsmc65/summary/pnr_$b${T65_PNR_TAG:-}_$s.txt
        { [ -f summary_$s.txt ] && cat summary_$s.txt; grep -E "Maximum resident|Elapsed" time_$s.txt | sed 's/^\s*/time_/'
          echo "icc2_errors $(grep -c '^Error' $s.log)"; } > "$sm"
        grep -q "STAGE_OK $s" $s.log || { echo "PNR_FAIL $s ($(grep -c '^Error' $s.log) errors, $T65_PNR/$s.log)"; exit 1; }
    fi
    echo "STAGE_DONE $s $(date +%H:%M:%S)"
    [ "$s" = "$last" ] && break
done
echo "PNR_DONE $b"
