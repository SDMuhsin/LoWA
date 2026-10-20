#!/usr/bin/env bash
# P9b: RTL references (VCS, the same testbench, GLS_RTL=1) of every window stimulus, without SAIF, for the
# cycle-by-cycle (op log) and DRAM-write-signature (HB wsig) comparison with the gate-level runs.
#   flow/tsmc65/sim/run_p9b_ref.sh [stopcyc [stimuli]]     output build/tsmc65/gls_rtl/ref_<stim>/
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || exit 1
export GLS_RTL=1
[ "${GLS_NOBUILD:-0}" = 1 ] || flow/tsmc65/sim/run_gls.sh build || exit 1
STOP=${1:-80000}; shift 2>/dev/null
for s in ${*:-l1 c3 c11 c32 c7 c10 c30 c40 c48 c44 c53}; do
    flow/tsmc65/sim/run_gls.sh run "ref_$s" "build/tsmc65/stim/$s" build/tsmc65/win/nowin.txt +stopcyc=$STOP +hb=10000
done
echo "REF_ALL_DONE"
