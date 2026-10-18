#!/usr/bin/env bash
# Formality of finished blocks, one at a time:  flow/tsmc65/lec/run_fm.sh <block>[:<tag>] ...
#   <tag> selects the DC run build/tsmc65/hier/<block><tag> (e.g. vu16:_p5.0).  Output build/tsmc65/fm/<block><tag>/
#   (stays on the server); summary -> build/tsmc65/summary/fm_<block><tag>.txt.  Markers FM_DONE1 / FM_DONE.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
for a in "$@"; do
    b=${a%%:*}; tag=""; [ "$a" != "$b" ] && tag=${a#*:}
    out=$ROOT/build/tsmc65/hier/$b$tag
    [ -f "$out/design.txt" ] || { echo "FM_SKIP $b$tag (not built)"; continue; }
    fo=$ROOT/build/tsmc65/fm/$b$tag
    mkdir -p "$fo" "$ROOT/build/tsmc65/summary"
    export T65_ROOT=$ROOT T65_BLOCK=$b T65_OUT=$out T65_FM_OUT=$fo T65_CORNER=${T65_CORNER:-ss}
    cd "$fo" || exit 1
    TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v -o $fo/time.txt"
    $TIMEV tcsh -f -c "source $ROOT/flow/tsmc65/common/env.csh; fm_shell -f $ROOT/flow/tsmc65/lec/fm_unit.tcl" > fm.log 2>&1
    s=$ROOT/build/tsmc65/summary/fm_$b$tag.txt
    if [ -f summary.txt ]; then
        cp -f summary.txt "$s"
        [ -f time.txt ] && grep -E "Maximum resident|Elapsed" time.txt | sed 's/^\s*/time_/' >> "$s"
        echo "fm_errors $(grep -c '^Error' fm.log)" >> "$s"
    fi
    echo "FM_DONE1 $b$tag errors $(grep -c '^Error' fm.log) $(grep -m1 -oE 'Verification (SUCCEEDED|FAILED|INCONCLUSIVE)' fm.log) summary $([ -f "$s" ] && echo yes || echo no)"
done
echo "FM_DONE"
