#!/usr/bin/env bash
# P8e (PLAN R16): mm_sram_bank port contract (tb/mem/tb_mm_sram_bank.sv, hold check on) on the tsmc65_sram backend
# with the vendor Verilog models of the generated macros (site/mm_sram_macro_site.v), VCS on the server.  The models
# stay on the server; only the PASS / FAIL lines come back.  One run per bank shape of T65_MC_SHAPES, N ops each,
# full model (timing checks on: a setup / hold violation corrupts the macro output and fails the check).
#   flow/tsmc65/sram/sim_banks.sh [N=100000]      -> build/sram/sim/<words>x<bits>/, marker SIM_BANKS_DONE
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
. "$HERE/../site/sram_site.sh"
N=${1:-100000}
out=$ROOT/build/sram/sim; mkdir -p "$out"
find "$ROOT/hw" -name '*.v' | sort | sed "s#.*/mm_sram_macro\.v\$#$ROOT/flow/tsmc65/site/mm_sram_macro_site.v#" > "$out/files.f"
models=""   # every macro model (the site wrapper's generate branches all name a macro)
for spec in $T65_MC_SHAPES; do i=${spec##*:}; models="$models $T65_MC_OUT/$i/$i.v"; done
for spec in $T65_MC_SHAPES; do
    IFS=: read -r w b c m i <<< "$spec"
    aw=0; while [ $((1 << aw)) -lt "$w" ]; do aw=$((aw + 1)); done
    d=$out/${w}x$b; mkdir -p "$d"
    ( cd "$d" || exit 1
      timeout 1800 tcsh -f -c "source $ROOT/flow/tsmc65/site/site.csh >& /dev/null; source \$T65_VCS_CSH >& /dev/null; \
        vcs -full64 -sverilog -timescale=1ns/1ps +define+MM_TECH=3 \
          -pvalue+tb_mm_sram_bank.WORDS=$w -pvalue+tb_mm_sram_bank.WIDTH=$b -pvalue+tb_mm_sram_bank.AW=$aw \
          -pvalue+tb_mm_sram_bank.N=$N -o simv -l vcs.log \
          $ROOT/tb/mem/tb_mm_sram_bank.sv -f ../files.f $models >& /dev/null; \
        ./simv -l sim.log >& /dev/null" )
    echo "SIM ${w}x$b AW$aw vcs_errors $(grep -c '^Error' "$d/vcs.log" 2>/dev/null) timing_viol $(grep -ci 'timing violation\|setup\b.*violation\|hold\b.*violation' "$d/sim.log" 2>/dev/null) $(grep -h -m1 -E '^(PASS|FAIL) mm_sram_bank' "$d/sim.log" 2>/dev/null || echo 'no result line')"
    rm -rf "$d/csrc" "$d/simv.daidir"
done
echo "SIM_BANKS_DONE"
