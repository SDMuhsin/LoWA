#!/usr/bin/env bash
# PrimeTime (and optionally PrimePower) of the flat chip netlist, one corner per call:
#   flow/tsmc65/sta/run_pt_chip.sh <corner> [none|vectorless|saif]
# Environment: T65_PERIOD (chip clock), T65_KID_TAG (the unit runs the chip linked), T65_PT_AREA, T65_SAIF, T65_TR, T65_SP.
# Output build/tsmc65/pt_chip/ (stays on the server); summary -> build/tsmc65/summary/ptchip_<corner>[_<power>].txt.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
corner=$1; power=${2:-none}
out=$ROOT/build/tsmc65/pt_chip
mkdir -p "$out" "$ROOT/build/tsmc65/summary"
export T65_ROOT=$ROOT T65_CORNER=$corner T65_POWER=$power T65_PT_OUT=$out T65_PERIOD=${T65_PERIOD:?chip clock}
cd "$out" || exit 1
TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v -o $out/time_$corner.txt"
$TIMEV tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; setenv LD_PRELOAD /usr/lib64/libz.so.1; pt_shell -f $ROOT/flow/tsmc65/sta/pt_chip.tcl" > pt_$corner.log 2>&1
tagp=""; [ "$power" != none ] && tagp=_$power
s=$ROOT/build/tsmc65/summary/ptchip_$corner$tagp.txt
if [ -f "summary_$corner.txt" ]; then
    cp -f "summary_$corner.txt" "$s"
    [ -f "time_$corner.txt" ] && grep -E "Maximum resident|Elapsed" "time_$corner.txt" | sed 's/^\s*/time_/' >> "$s"
    echo "pt_errors $(grep -c '^Error' pt_$corner.log)" >> "$s"
fi
echo "PTCHIP_DONE $corner $power errors $(grep -c '^Error' pt_$corner.log) summary $([ -f "$s" ] && echo yes || echo no)"
