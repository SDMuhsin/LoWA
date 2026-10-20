#!/usr/bin/env bash
# P9b: the RTL reference of the gate-level power runs in Verilator (the simulator of the bit-exact e2e tests):
# tb/power/tb_top_gls.sv (SAIF code compiled out) on the RTL, then every stimulus dir given, to a cycle limit.
#   tb/power/run_ref_verilator.sh <stopcyc> <stim_dir>...      (EDA environment as for make test-e2e)
# Output <stim_dir>/vref.log (heartbeats with the DRAM write signature, STOP line) and <stim_dir>/vref_oplog.txt.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OBJ=$ROOT/build/p9b/obj_gls_vref
stop=$1; shift
if [ ! -x "$OBJ/Vtb_top_gls" ]; then
    verilator --binary --timing -O3 -j 16 --x-assign fast --x-initial fast -Wno-fatal -Wno-lint -Wno-style -CFLAGS -O2 \
        --top-module tb_top_gls +define+MM_DMEMW=25 -I"$ROOT/tb/mem" -Mdir "$OBJ" "$ROOT/tb/power/tb_top_gls.sv" \
        "$ROOT/tb/mem/tb_dram.sv" $(find "$ROOT/hw" -name '*.v' | sort) > "$OBJ.build.log" 2>&1 || { echo "VREF build failed"; exit 1; }
fi
for d in "$@"; do
    "$OBJ/Vtb_top_gls" +reg="$d/dram.reg" +img="$d/dram.bin" +host="$d/host.txt" +oplog="$d/vref_oplog.txt" \
        +stopcyc="$stop" +hb=10000 > "$d/vref.log" 2>&1
    echo "VREF $(basename "$d") $(grep -E '^STOP' "$d/vref.log")"
done
