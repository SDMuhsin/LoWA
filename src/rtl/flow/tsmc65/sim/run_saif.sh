#!/usr/bin/env bash
# P8d switching activity for PrimePower: RTL simulation (VCS) of an existing unit testbench on a real case,
# tsmc65 memory backend (MM_TECH=2, latch RF, behavioural clock gate), toggles of the DUT recorded for a bounded
# number of cycles by tb/power/saif_ctl.sv.
#   flow/tsmc65/sim/run_saif.sh <gm|vu> <case_dir> <case> <cycles> <chip_period_ns>
# <case_dir>/<case>.desc/.mem.hex (+ .mask.hex) are the generator's files (tb/gemm, tb/vec), copied up by the user.
# The testbench clock is 40 ns; the SAIF time unit is rescaled so that one testbench cycle = one chip cycle.
# Output build/tsmc65/saif/<unit>_<case>/dut.saif (stays on the server); marker SAIF_DONE.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
u=$1; d=$(cd "$2" && pwd) || exit 1; c=$3; n=$4; p=$5
out=$ROOT/build/tsmc65/saif/${u}_$c
mkdir -p "$out"; cd "$out" || exit 1
case $u in
    gm) tb=tb/gemm/tb_gm_top.sv; def=""; pv="-pvalue+tb_gm_top.MEMW=25" ;;
    vu) tb=tb/vec/tb_vu_top.sv; def="+define+SAIF_VU"; pv="-pvalue+tb_vu_top.MEMW=25" ;;
    *) echo "unit gm | vu"; exit 1 ;;
esac
find "$ROOT/hw" -name '*.v' | sort > files.f
echo 0 > dummy.exp.hex
mask=""; [ -s "$d/$c.mask.hex" ] && mask="+mask=$d/$c.mask.hex"
tcsh -f -c "source $ROOT/flow/tsmc65/site/site.csh >& /dev/null; source \$T65_VCS_CSH >& /dev/null; \
  vcs -full64 -sverilog -timescale=1ns/1ps +define+MM_TECH=2 $def $pv -o simv -l vcs.log \
  $ROOT/$tb $ROOT/tb/power/saif_ctl.sv -f files.f >& /dev/null; \
  ./simv +desc=$d/$c.desc +mem=$d/$c.mem.hex +exp=dummy.exp.hex $mask +saifcyc=$n -l sim.log >& /dev/null"
if [ -f dut.saif ]; then
    fs=$(awk -v p="$p" 'BEGIN { printf "%d", p * 1000000 / 40 }')
    sed -i -E "0,/\(TIMESCALE[^)]*\)/s//(TIMESCALE $fs fs)/" dut.saif
    grep -m1 TIMESCALE dut.saif
    # generate-block scopes as DC names them (g_lane[0].u_lane -> g_lane_0__u_lane; PT 2019 read_saif has no name
    # mapping): dut_g.saif is the file PrimePower reads
    sed -E 's/\\\[([0-9]+)\\\]\\\./_\1__/g' dut.saif > dut_g.saif
fi
echo "SAIF_DONE $u $c vcs_errors $(grep -c '^Error' vcs.log 2>/dev/null) $(grep -m1 SAIF_WRITTEN sim.log 2>/dev/null) saif $([ -f dut.saif ] && du -h dut.saif | cut -f1 || echo none)"
