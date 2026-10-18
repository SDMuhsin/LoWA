#!/usr/bin/env bash
# PrimeTime (optionally PrimePower) of one block's complete netlist, or the chip on its units' extracted models:
#   flow/tsmc65/sta/run_pt_unit.sh <block> <corner> [none|vectorless|saif]
# Environment: T65_PERIOD (chip clock), T65_KID_TAG (unit runs the chip links), T65_PT_AREA, T65_ETM, T65_USE_ETM,
#   T65_SAIF, T65_TR, T65_SP, T65_PT_SUFFIX (summary-name suffix).
# Output build/tsmc65/pt/<block>/ (stays on the server); summary -> build/tsmc65/summary/ptu_<block>_<corner>[_<power>].txt.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
b=$1; corner=$2; power=${3:-none}
out=$ROOT/build/tsmc65/pt/$b
mkdir -p "$out" "$ROOT/build/tsmc65/summary"
export T65_ROOT=$ROOT T65_BLOCK=$b T65_CORNER=$corner T65_POWER=$power T65_PT_OUT=$out T65_PERIOD=${T65_PERIOD:?chip clock}
cd "$out" || exit 1
TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v -o $out/time_$corner.txt"
$TIMEV tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; setenv LD_PRELOAD /usr/lib64/libz.so.1; pt_shell -f $ROOT/flow/tsmc65/sta/pt_unit.tcl" > pt_$corner.log 2>&1
tagp=""; [ "$power" != none ] && tagp=_$power
s=$ROOT/build/tsmc65/summary/ptu_${b}_$corner$tagp${T65_PT_SUFFIX:-}.txt
if [ -f "summary_$corner.txt" ]; then
    cp -f "summary_$corner.txt" "$s"
    [ -f "time_$corner.txt" ] && grep -E "Maximum resident|Elapsed" "time_$corner.txt" | sed 's/^\s*/time_/' >> "$s"
    echo "pt_errors $(grep -c '^Error' pt_$corner.log)" >> "$s"
fi
echo "PTU_DONE $b $corner $power errors $(grep -c '^Error' pt_$corner.log) summary $([ -f "$s" ] && echo yes || echo no)"
