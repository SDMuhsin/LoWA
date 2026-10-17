#!/usr/bin/env bash
# Sequential batch of block syntheses (one DC at a time):  flow/tsmc65/syn/run_scout.sh <block>:<period> ...
# Prints START / DC_DONE per run and SCOUT_DONE at the end.  Launch detached on the server, e.g.
#   cd <tree>/src/rtl || exit 1; nohup flow/tsmc65/syn/run_scout.sh fma:5.0 pe:5.0 > build/tsmc65/scout.log 2>&1 < /dev/null &
HERE=$(cd "$(dirname "$0")" && pwd)
for job in "$@"; do
    echo "START $job $(date +%H:%M:%S)"
    "$HERE/run_block.sh" "${job%%:*}" "${job#*:}" "${T65_CORNER:-ss}" "${T65_TOPO:-1}"
done
echo "SCOUT_DONE $(date +%H:%M:%S)"
