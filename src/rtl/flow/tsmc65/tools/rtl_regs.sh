#!/usr/bin/env bash
# RTL register expectation of a block of common/hier.txt (local, Yosys; tsmc65 memory backend, behavioural mm_icg):
#   flow/tsmc65/tools/rtl_regs.sh <block> ...
# Prints flop bits and latch bits of the flattened block twice: raw (after proc) and after Yosys' own removal of
# unloaded / constant registers (opt_clean, opt_dff), the counterpart of DC's OPT-1206 / OPT-1207.  Latch bits
# include one latch per mm_icg (DC maps those to library ICG cells, which the DC summary counts separately).
set -u
T65=$(cd "$(dirname "$0")/.." && pwd)
RTL=$(cd "$T65/../.." && pwd)
TAB=$T65/common/hier.txt
OUT=$RTL/build/tsmc65/rtl_regs; mkdir -p "$OUT"
FILES=$(find "$RTL/hw" -name '*.v' | sort | tr '\n' ' ')
count() {   # yosys stat -width log -> "flops N latches M icg K"
    awk '/Number of cells/ {sec++} {
            if ($2 ~ /^\$(a|s)?dffc?e?_[0-9]+$/) { n = split($2, t, "_"); f[sec] += t[n] * $1 }
            if ($2 ~ /^\$a?dlatch_[0-9]+$/)     { n = split($2, t, "_"); l[sec] += t[n] * $1 }
         } END { printf "flops %d latches %d", f[sec], l[sec] }' "$1"
}
for b in "$@"; do
    read -r mod params <<< "$(awk -v b="$b" '$1 == b && $1 !~ /^[#@]/ { print $2, $3 }' "$TAB")"
    [ -n "$mod" ] || { echo "unknown block $b"; continue; }
    chp=""
    [ "$params" != "-" ] && for kv in ${params//,/ }; do chp="$chp -chparam ${kv%%=>*} ${kv##*=>}"; done
    base="read_verilog -DSYNTHESIS -DMM_TECH=2 $FILES; hierarchy -check -top $mod $chp; proc; flatten"
    yosys -q -l "$OUT/$b.raw.log" -p "$base; tee -o $OUT/$b.raw.stat stat -width" > /dev/null 2>&1 || { echo "$b yosys failed ($OUT/$b.raw.log)"; continue; }
    yosys -q -l "$OUT/$b.opt.log" -p "$base; opt_clean -purge; opt_dff; opt_clean -purge; tee -o $OUT/$b.opt.stat stat -width" > /dev/null 2>&1
    echo "rtl_regs $b $mod raw: $(count "$OUT/$b.raw.stat")  opt: $(count "$OUT/$b.opt.stat")"
done
