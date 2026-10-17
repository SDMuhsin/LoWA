#!/usr/bin/env bash
# make lint: every file under hw/ must be accepted by all three tools (PLAN fixed decision 6).
#   1. Verilator 5.044 --lint-only -Wall: every module as top with default parameters, plus the
#      parameter sweeps below (waivers live in the source, each with a one-line reason)
#   2. Yosys 0.62: read all files, then per module hierarchy + proc + check -assert, no latches
#   3. Icarus 13: iverilog -g2005, every module as root
# Needs env.sh sourced.  Logs in build/lint/.  Exit 1 on any failure.
set -u
cd "$(dirname "$0")/.."
OUT=build/lint; mkdir -p $OUT
HW=$(find hw -name '*.v' | sort | tr '\n' ' ')
MODS=$(for f in $HW; do basename "$f" .v; done)
fail=0; n=0
vl() {  # top, -G params...
    local top=$1; shift
    local log=$OUT/vl_${top}$(echo "$@" | tr ' =' '__').log
    n=$((n+1))
    if ! verilator --lint-only -Wall --top-module "$top" "$@" $HW > "$log" 2>&1; then
        echo "LINT FAIL verilator $top $*"; grep -E '%(Warning|Error)' "$log" | head -5; fail=1
    fi
}
# the vendor SRAM macro (name with "__", which Verilator cannot select as a top) is linted through mm_sram_bank
for m in $MODS; do [ "$m" = gf180mcu_ocd_ip_sram__sram256x8m8wm1 ] || vl "$m"; done
for cfg in "16 16" "8 8" "4 4"; do set -- $cfg; vl gm_top -GROWS=$1 -GCOLS=$2 -GNSFU=2; done
for cfg in "16 4" "8 3" "4 2"; do set -- $cfg; vl vu_top -GV=$1 -GVW=$2; done
vl vu_lane -GNS=4 -GSW=2 -GNOB=4
for h in 0 1 3; do vl vu_loader -GHBL=$h; done; vl vu_top -GNOB=4; vl vu_writer -GWK=4
for c in "16 9 3" "16 3 3" "16 1 1" "4 3 3" "2 1 14"; do set -- $c; vl rn_mkgen -GMKW=$1 -GNCORE=$2 -GSTAGES=$3; done
for s in 1 2 7 14; do vl rn_tf -GSTAGES=$s; done
vl mm_sram_bank -GWORDS=512 -GAW=9; vl mm_sram_bank -GWORDS=1024 -GAW=10
for s in 1 2 3 4 5 6; do vl fp_fma -GSTAGES=$s; vl fp_add -GSTAGES=$s; done
for s in 1 2 3; do vl fp_mul -GSTAGES=$s; vl fp_i2f -GSTAGES=$s; vl fp_cmp -GSTAGES=$s; done
for lf in 1 2 3 4 6; do for nt in 1 2 $((lf + 2)); do [ $nt -le $((lf + 2)) ] && vl fp_sfu -GLF=$lf -GNTHR=$nt; done; done
for nn in 0 1 2 3; do for r in 0 1; do vl cm_delay -GN=$nn -GRST=$r -GW=3; done; done
for w in "32 16 1" "201 64 4" "10 3 4" "4224 4 64"; do set -- $w; vl cm_ohmux -GW=$1 -GN=$2 -GREP=$3; done
for w in "8 3" "29 5" "32 5" "78 7"; do set -- $w; vl fp_norm -GW=$1 -GCW=$2; done
for w in 29 32 48 78; do vl fp_round -GMW=$w; done
# Yosys
YS=""
for m in $MODS; do
    YS="$YS design -reset; read_verilog -DSYNTHESIS $HW; hierarchy -check -top $m; proc;
        check -assert; select -assert-none t:\$dlatch* t:\$sr;"
done
if ! yosys -q -l $OUT/yosys.log -p "$YS" > /dev/null 2>&1; then
    echo "LINT FAIL yosys (see $OUT/yosys.log)"; tail -5 $OUT/yosys.log; fail=1
fi
# Icarus
for m in $MODS; do
    n=$((n+1))
    if ! iverilog -g2005 -s "$m" -o $OUT/iv.out $HW > $OUT/iv_$m.log 2>&1; then
        echo "LINT FAIL icarus $m"; head -5 $OUT/iv_$m.log; fail=1
    fi
done
rm -f $OUT/iv.out
nf=$(echo $MODS | wc -w)
if [ $fail = 0 ]; then echo "LINT PASS: $(echo $HW | wc -w) files, $nf modules; Verilator -Wall $((n - nf)) runs (defaults + sweeps), Yosys $nf tops, Icarus $nf tops"
else echo "LINT FAIL"; fi
exit $fail
