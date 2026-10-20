#!/usr/bin/env bash
# P9b: PrimePower of the flat chip with the gate-level SAIF windows of sim/run_gls.sh (sta/pt_gls_power.tcl).
#   flow/tsmc65/sta/run_pt_gls.sh <run_tag> <corner> <window_list>   ("name saif" lines, "END" closes; re-read while running)
# Environment: T65_PERIOD (default 5.5), T65_KID_TAG (default: chip of record), T65_HIER (netlist root).
# Output build/tsmc65/pt_gls/<run_tag>/ (stays on the server); summary -> build/tsmc65/summary/ptgls_<run_tag>_<corner>.txt
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
tag=$1; corner=$2; lst=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
out=$ROOT/build/tsmc65/pt_gls/$tag
mkdir -p "$out" "$ROOT/build/tsmc65/summary"
export T65_ROOT=$ROOT T65_CORNER=$corner T65_PT_OUT=$out T65_PERIOD=${T65_PERIOD:-5.5}
export T65_KID_TAG=${T65_KID_TAG:-gm16=_p5.25,vu16=_p5.25,mm=_p8f,rng9=_p8f,rng3=_p8f}
export T65_GLS_LIST=$lst
cd "$out" || exit 1
/usr/bin/time -v -o time_$corner.txt tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; setenv LD_PRELOAD /usr/lib64/libz.so.1; pt_shell -f $ROOT/flow/tsmc65/sta/pt_gls_power.tcl" > pt_$corner.log 2>&1
s=$ROOT/build/tsmc65/summary/ptgls_${tag}_$corner.txt
if [ -f "summary_gls_$corner.txt" ]; then
    cp -f "summary_gls_$corner.txt" "$s"
    grep -E "Maximum resident|Elapsed" "time_$corner.txt" | sed 's/^\s*/time_/' >> "$s"
    echo "pt_errors $(grep -c '^Error' pt_$corner.log)" >> "$s"
fi
echo "PTGLS_DONE $tag $corner errors $(grep -c '^Error' pt_$corner.log) summary $([ -f "$s" ] && echo yes || echo no)"
