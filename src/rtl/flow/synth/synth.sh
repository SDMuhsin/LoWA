#!/usr/bin/env bash
# Synthesis + STA of one hw/ unit inside its input-register wrapper (flow/synth/wrap_<unit>.v).
#   flow/synth/synth.sh <unit> <out_dir> [PARAM=value ...]
# Yosys 0.62 (synth -flatten -booth, the unit kept as one hierarchy level so its own area is
# reported), gf180mcu_as_sc_mcu7t3v3 tt lib for mapping/area (llmdocs/asic_flow_tutorial.md
# section 4), then OpenSTA at ss_125C_3v00 and tt_025C_3v30 with flow/sta/unit.sdc (40 ns).
# Writes <out_dir>/{synth.log,stat.txt,nl.v,sta_<corner>.rpt,summary.txt}, and gls.v = the bare
# unit synthesised flat with the same parameters (for gate-level simulation).  Needs env.sh sourced.
set -euo pipefail
UNIT=$1; OUT=$2; shift 2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LIBDIR=$(ls -d "$PDK_ROOT"/ciel/gf180mcu/versions/*/gf180mcuD/libs.ref/gf180mcu_as_sc_mcu7t3v3/lib | head -1)
LIB=$LIBDIR/gf180mcu_as_sc_mcu7t3v3__tt_025C_3v30.lib
PERIOD=${PERIOD:-40}
mkdir -p "$OUT"
HW=$(find "$ROOT/hw" -name '*.v' | sort | tr '\n' ' ')
CHP=""; CHU=""
for p in "$@"; do
    CHP="$CHP chparam -set ${p%%=*} ${p#*=} wrap_$UNIT;"; CHU="$CHU chparam -set ${p%%=*} ${p#*=} $UNIT;"
done
yosys -q -l "$OUT/synth.log" -p "
    read_verilog -DSYNTHESIS $HW $ROOT/flow/synth/wrap_$UNIT.v
    $CHP
    hierarchy -check -top wrap_$UNIT
    synth -flatten -booth -top wrap_$UNIT
    select -assert-none t:\$_DLATCH* t:\$dlatch* t:\$sr t:\$_SR*
    check -assert
    dfflibmap -liberty $LIB
    abc -D $((PERIOD * 1000)) -liberty $LIB
    delete t:\$scopeinfo
    opt_clean
    tee -q -o $OUT/stat.txt stat -liberty $LIB
    write_verilog -noattr $OUT/nl.v" >/dev/null
yosys -q -l "$OUT/gls.log" -p "
    read_verilog -DSYNTHESIS $HW
    $CHU
    hierarchy -check -top $UNIT
    synth -flatten -booth -top $UNIT
    dfflibmap -liberty $LIB
    abc -D $((PERIOD * 1000)) -liberty $LIB
    delete t:\$scopeinfo
    opt_clean
    write_verilog -noattr $OUT/gls.v" >/dev/null
# the unit's own area: the stat block of the kept (non-wrapper) module
UAREA=$(awk '/^=== /{m=$2} /Chip area for module/{ if (m !~ /wrap_/) {print $NF; exit} }' "$OUT/stat.txt")
UFLOP=$(awk '/^=== /{m=$2} m !~ /wrap_/ && /gf180mcu_as_sc_mcu7t3v3__(df|sdf)/ {s+=$1} /^=== /{if (m ~ /wrap_/ && s) exit} END{print s+0}' "$OUT/stat.txt")
: > "$OUT/summary.txt"
for c in ss_125C_3v00 tt_025C_3v30; do
    NETLIST=$OUT/nl.v TOP=wrap_$UNIT CORNER=$c PERIOD=$PERIOD SDC=$ROOT/flow/sta/unit.sdc \
        sta -no_init -exit "$ROOT/flow/sta/sta.tcl" > "$OUT/sta_$c.rpt" 2>&1
    grep '^STA ' "$OUT/sta_$c.rpt" >> "$OUT/summary.txt"
done
echo "AREA $UAREA FLOPS $UFLOP" >> "$OUT/summary.txt"
cat "$OUT/summary.txt"
