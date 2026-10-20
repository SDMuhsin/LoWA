#!/usr/bin/env bash
# P9b gate-level power simulation: VCS, zero-delay (no SDF, +nospecify), of the synthesized chip netlists (the 17
# block netlists of the flat-chip PrimeTime run, docs/p8_report.md section 4.2) running a compiled program through
# tb/power/tb_top_gls.sv; toggles of u_chip recorded per cycle window (+saifwin) into SAIF files for PrimePower.
#   flow/tsmc65/sim/run_gls.sh build                       compile build/tsmc65/gls/simv
#   flow/tsmc65/sim/run_gls.sh run <tag> <stim_dir> <win_file> [extra plusargs]
#       <stim_dir>: dram.reg, dram.bin, host.txt of tb/top/run_e2e.py --prep-only (copied up; design data)
#       <win_file>: "name first_cycle end_cycle" lines; output build/tsmc65/gls/<tag>/<name>.saif, sim.log, oplog.txt
# Environment: T65_HIER (netlist root, default build/tsmc65/hier), T65_KID_TAG (as for DC / PT; default = the
#   chip of record at 5.5 ns: gm16=_p5.25,vu16=_p5.25,mm=_p8f,rng9=_p8f,rng3=_p8f), GLS_RTL=1: the RTL instead of the
#   netlists (same testbench, behavioural clock gate, MM_TECH=2 latch RF) for the cycle-by-cycle comparison;
#   GLS_DIR: build / run directory under build/tsmc65 (default gls, gls_rtl); GLS_MEMW: DRAM model words log2 (24);
#   GLS_RTL_DEF: defines of the RTL build (default +define+MM_TECH=2; empty = the generic memory backend).
# GLS_SAIF_DIR (or the file build/tsmc65/gls/saif_dir): where the SAIF files go (see run).
# Cell models: the library Verilog ($T65_CELL_V from site.csh) for the combinational cells, site/gls_cells.v for
# the sequential ones (they take precedence: the library is read with -v).
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HIER=${T65_HIER:-$ROOT/build/tsmc65/hier}
TAGS=${T65_KID_TAG:-gm16=_p5.25,vu16=_p5.25,mm=_p8f,rng9=_p8f,rng3=_p8f}
BLOCKS="axil bank256_gm bank256_lw bank256_vu bank512_gm bank512_lw bank768_lw chip gm16 lane lw mm pe rng3 rng9 seq vu16"
G=$ROOT/build/tsmc65/${GLS_DIR:-gls${GLS_RTL:+_rtl}}
mkdir -p "$G"
tagof() { local t=""; for kv in ${TAGS//,/ }; do [ "${kv%%=*}" = "$1" ] && t=${kv#*=}; done; echo "$t"; }
case ${1:-} in
build)
    cd "$G" || exit 1
    : > files.f
    if [ -n "${GLS_RTL:-}" ]; then
        find "$ROOT/hw" -name '*.v' | sort > files.f
        def="${GLS_RTL_DEF-+define+MM_TECH=2}"
        cells=""
    else
        for b in $BLOCKS; do d=$HIER/$b$(tagof "$b"); [ -f "$d/$b.v" ] || { echo "GLS_ERROR missing $d/$b.v"; exit 1; }; echo "$d/$b.v" >> files.f; done
        def=""
        cells="$ROOT/flow/tsmc65/site/gls_cells.v"
    fi
    /usr/bin/time -v -o time_build.txt tcsh -f -c "source $ROOT/flow/tsmc65/site/site.csh >& /dev/null; source \$T65_VCS_CSH >& /dev/null; \
      vcs -full64 -sverilog -timescale=1ns/10ps +nospecify +notimingcheck +lint=none -j8 ${GLS_VCS_OPTS:-} $def +define+MM_DMEMW=${GLS_MEMW:-24} \
      -o simv -Mdir=csrc -l vcs.log $ROOT/tb/power/tb_top_gls.sv $ROOT/tb/mem/tb_dram.sv $cells -f files.f \
      ${cells:+-v \$T65_CELL_V} >& vcs.out"
    echo "GLS_BUILD_DONE $([ -x simv ] && echo ok || echo FAILED) errors $(grep -c '^Error' vcs.log 2>/dev/null) $(grep -E 'Elapsed|Maximum resident' time_build.txt | tr -s ' ' | tr '\n' ' ')" ;;
run)
    tag=$2; s=$(cd "$3" && pwd) || exit 1; w=$(cd "$(dirname "$4")" && pwd)/$(basename "$4"); shift 4
    out=$G/$tag; mkdir -p "$out"; cd "$out" || exit 1
    cp -f "$w" windows.txt
    # bulky SAIFs outside the NFS home (quota): GLS_SAIF_DIR, or the directory named in $G/saif_dir; each window's
    # <name>.saif here is a symlink to <dir>/<tag>/<name>.saif (the testbench writes through it, PrimePower reads it)
    sd=${GLS_SAIF_DIR:-$( [ -f "$G/saif_dir" ] && cat "$G/saif_dir")}
    if [ -n "$sd" ]; then
        mkdir -p "$sd/$tag"
        while read -r n _; do [ -n "$n" ] && ln -sfn "$sd/$tag/$n.saif" "$n.saif"; done < windows.txt
    fi
    /usr/bin/time -v -o time_run.txt tcsh -f -c "source $ROOT/flow/tsmc65/site/site.csh >& /dev/null; source \$T65_VCS_CSH >& /dev/null; \
      $G/simv +reg=$s/dram.reg +img=$s/dram.bin +host=$s/host.txt +saifwin=$out/windows.txt +oplog=$out/oplog.txt $* -l sim.log >& sim.out"
    echo "GLS_RUN_DONE $tag $(grep -E '^(SAIF_ALL_DONE|STOP|PASS|FAIL)' sim.log | head -3 | tr '\n' ' ') saifs $(ls *.saif.ok 2>/dev/null | wc -l) $(grep -E 'Elapsed|Maximum resident' time_run.txt | tr -s ' ' | tr '\n' ' ')" ;;
*) echo "usage: $0 build | run <tag> <stim_dir> <win_file> [plusargs]"; exit 1 ;;
esac
