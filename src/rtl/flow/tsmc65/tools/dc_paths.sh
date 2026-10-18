#!/usr/bin/env bash
# flow/tsmc65/tools/dc_paths.sh <block> '<endpoint pin glob> ...'   -> build/tsmc65/summary/paths_<block>.txt
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
out=$ROOT/build/tsmc65/hier/$1
export T65_ROOT=$ROOT T65_BLOCK=$1 T65_PERIOD=${T65_PERIOD:-4.0} T65_OUT=$out T65_PATHS="$2"
cd "$out" || exit 1
rm -rf "$out/paths_mw"      # a stale Milkyway library from an earlier run cannot be re-created (MWUI-290)
tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; dc_shell -topographical_mode -f $ROOT/flow/tsmc65/tools/dc_paths.tcl" > paths.log 2>&1
[ -f paths.txt ] && cp -f paths.txt "$ROOT/build/tsmc65/summary/paths_$1.txt"
echo "PATHS_DONE $1 $([ -f paths.txt ] && echo yes || echo no)"
