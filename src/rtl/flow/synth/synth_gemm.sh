#!/usr/bin/env bash
# Synthesis + STA of a hw/gemm unit inside its input-register wrapper, keeping the hierarchy of
# the repeated leaf units (gm_pe, fp_sfu, mm_sram_bank / the SRAM macro blackbox) so that Yosys
# synthesises each leaf once: `synth -flatten` honours (* keep_hierarchy *).
#   flow/synth/synth_gemm.sh <unit> <out_dir> [PARAM=value ...]
# Same libraries, SDC and outputs as flow/synth/synth.sh (docs/p2_report.md); area = sum over the
# hierarchy (`stat -top`), summary.txt gets AREA / FLOPS / CELLS and the STA lines.  The SRAM macro
# is a blackbox (-DSYNTHESIS): its area is NOT in the Yosys number (counted separately, 0.0678 mm2
# each).  Needs env.sh sourced.
set -euo pipefail
UNIT=$1; OUT=$2; shift 2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LIBDIR=$(ls -d "$PDK_ROOT"/ciel/gf180mcu/versions/*/gf180mcuD/libs.ref/gf180mcu_as_sc_mcu7t3v3/lib | head -1)
LIB=$LIBDIR/gf180mcu_as_sc_mcu7t3v3__tt_025C_3v30.lib
PERIOD=${PERIOD:-40}
mkdir -p "$OUT"
HW=$(find "$ROOT/hw" -name '*.v' | sort | tr '\n' ' ')
CHP=""
for p in "$@"; do CHP="$CHP chparam -set ${p%%=*} ${p#*=} wrap_$UNIT;"; done
yosys -q -l "$OUT/synth.log" -p "
    read_verilog -DSYNTHESIS $HW $ROOT/flow/synth/wrap_$UNIT.v
    $CHP
    hierarchy -check -top wrap_$UNIT
    setattr -set keep_hierarchy 1 t:*gm_pe* t:*fp_sfu t:*fp_sfu_ucode t:*mm_sram_bank* t:cm_rstbuf
    synth -flatten -booth -top wrap_$UNIT
    select -assert-none t:\$_DLATCH* t:\$dlatch* t:\$sr t:\$_SR*
    check -assert
    dfflibmap -liberty $LIB
    abc -D $((PERIOD * 1000)) -liberty $LIB
    delete t:\$scopeinfo
    opt_clean
    tee -q -o $OUT/stat.txt stat -top wrap_$UNIT -liberty $LIB
    write_verilog -noattr $OUT/nl.v" >/dev/null
# hierarchical totals (the design = everything but the wrapper's own input flops)
UAREA=$(awk '/design hierarchy/{h=1} h && /Chip area for top module/{print $NF; exit}' "$OUT/stat.txt")
UCELLS=$(awk -v t="wrap_$UNIT" '/design hierarchy/{h=1} h && $3 == t {print $1; exit}' "$OUT/stat.txt")
: > "$OUT/summary.txt"
for c in ss_125C_3v00 tt_025C_3v30; do
    NETLIST=$OUT/nl.v TOP=wrap_$UNIT CORNER=$c PERIOD=$PERIOD SDC=$ROOT/flow/sta/unit.sdc \
        sta -no_init -exit "$ROOT/flow/sta/sta.tcl" > "$OUT/sta_$c.rpt" 2>&1
    grep '^STA ' "$OUT/sta_$c.rpt" >> "$OUT/summary.txt"
done
# OpenROAD: buffer the high-fanout nets (register-file read pointers, lane selects: Yosys inserts
# no buffer trees) and re-time at ss; see flow/sta/repair_sta.tcl
NETLIST=$OUT/nl.v TOP=wrap_$UNIT CORNER=ss_125C_3v00 PERIOD=$PERIOD SDC=$ROOT/flow/sta/unit.sdc \
    openroad -exit "$ROOT/flow/sta/repair_sta.tcl" > "$OUT/repair_ss.rpt" 2>&1 || true
grep '^STA ' "$OUT/repair_ss.rpt" >> "$OUT/summary.txt" || true
echo "AREA $UAREA CELLS $UCELLS" >> "$OUT/summary.txt"
cat "$OUT/summary.txt"
