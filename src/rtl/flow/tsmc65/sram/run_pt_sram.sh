#!/usr/bin/env bash
# P8e: PrimeTime / PrimePower (sta/pt_unit.tcl via sram/pt_sram.tcl) of a block of a sram/run_sram.sh run:
#   T65_VARIANT=sram|latch T65_PERIOD=5.0 flow/tsmc65/sram/run_pt_sram.sh <block> <corner> [none|vectorless]
# Output build/tsmc65/pt_<variant>_p<P>/<block>/ (stays on the server); summary (numbers only) ->
# build/tsmc65/summary/sram/pt_<variant>_p<P>_<block>_<corner>[_<power>].txt.  Marker PTS_DONE.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
. "$HERE/../site/sram_site.sh"
export T65_MC_OUT T65_MC_CORNERS T65_MC_SHAPES
b=$1; corner=$2; power=${3:-none}
V=${T65_VARIANT:-sram}; P=${T65_PERIOD:-5.0}
out=$ROOT/build/tsmc65/pt_${V}_p$P/$b; mkdir -p "$out" "$ROOT/build/tsmc65/summary/sram"
export T65_ROOT=$ROOT T65_BLOCK=$b T65_CORNER=$corner T65_POWER=$power T65_PT_OUT=$out T65_PERIOD=$P T65_VARIANT=$V
export T65_HROOT=$ROOT/build/tsmc65/hier_${V}_p$P T65_CORES=${T65_CORES:-4}
cd "$out" || exit 1
/usr/bin/time -v -o "$out/time_$corner.txt" tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; setenv LD_PRELOAD /usr/lib64/libz.so.1; pt_shell -f $HERE/pt_sram.tcl" > pt_$corner.log 2>&1
tagp=""; [ "$power" != none ] && tagp=_$power
s=$ROOT/build/tsmc65/summary/sram/pt_${V}_p${P}_${b}_$corner$tagp.txt
if [ -f "summary_$corner.txt" ]; then
    cp -f "summary_$corner.txt" "$s"
    grep -E "Maximum resident|Elapsed" "time_$corner.txt" | sed 's/^\s*/time_/' >> "$s"
    echo "pt_errors $(grep -c '^Error' pt_$corner.log)" >> "$s"
fi
echo "PTS_DONE $b $corner $power errors $(grep -c '^Error' pt_$corner.log) summary $([ -f "$s" ] && echo yes || echo no)"
