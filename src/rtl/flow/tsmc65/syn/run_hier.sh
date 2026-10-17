#!/usr/bin/env bash
# Sequential bottom-up batch of the P8b flow (one DC at a time):  flow/tsmc65/syn/run_hier.sh <unit|block> ...
# Units and blocks come from common/hier.txt; every child is built before its parent, and a block that already
# has a design.txt (a finished run) is reused, not recompiled (FORCE=1 rebuilds the named blocks themselves, FORCE=all every block in the order).
# Prints START / DC_DONE per run and HIER_DONE at the end; stops at the first failed block.
#   cd <tree>/src/rtl || exit 1; nohup flow/tsmc65/syn/run_hier.sh vec > build/tsmc65/hier_vec.log 2>&1 < /dev/null &
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
TAB=$HERE/../common/hier.txt
P=${T65_PERIOD:-4.0}
kids()  { awk -v b="$1" '$1 == b && $1 !~ /^[#@]/ { if ($4 != "-") { gsub(",", " ", $4); print $4 } }' "$TAB"; }
units() { awk -v u="$1" '$1 == "@unit" && $2 == u { for (i = 3; i <= NF; i++) print $i }' "$TAB"; }
known() { awk -v b="$1" '$1 == b && $1 !~ /^[#@]/ { f = 1 } END { exit !f }' "$TAB"; }
ORDER=""
visit() { local k; for k in $(kids "$1"); do visit "$k"; done; case " $ORDER " in *" $1 "*) ;; *) ORDER="$ORDER $1" ;; esac; }
NAMED=""
for a in "$@"; do
    u=$(units "$a")
    if [ -n "$u" ]; then for b in $u; do visit "$b"; NAMED="$NAMED $b"; done
    elif known "$a"; then visit "$a"; NAMED="$NAMED $a"
    else echo "unknown unit / block $a"; exit 1; fi
done
echo "ORDER$ORDER"
for b in $ORDER; do
    if [ -f "$ROOT/build/tsmc65/hier/$b/design.txt" ] && [ "${FORCE:-0}" != all ]; then
        case " $NAMED " in *" $b "*) [ "${FORCE:-0}" = 1 ] || { echo "REUSE $b"; continue; } ;; *) echo "REUSE $b"; continue ;; esac
    fi
    echo "START $b $(date +%H:%M:%S)"
    r=$("$HERE/run_unit.sh" "$b" "$P" "${T65_CORNER:-ss}")
    echo "$r"
    case "$r" in *"summary yes"*) ;; *) echo "HIER_FAILED $b"; echo "HIER_DONE $(date +%H:%M:%S)"; exit 1 ;; esac
done
echo "HIER_DONE $(date +%H:%M:%S)"
