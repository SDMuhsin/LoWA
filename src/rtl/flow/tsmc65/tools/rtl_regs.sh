#!/usr/bin/env bash
# RTL register expectation of a block of common/hier.txt (local, Yosys; tsmc65 memory backend, behavioural mm_icg):
#   flow/tsmc65/tools/rtl_regs.sh <block> ...
# Prints flop bits (plain registers + register-array bits, SIZE x WIDTH of each collected memory) and latch bits
# of the flattened block twice: raw (after proc) and after Yosys' own removal of
# unloaded / constant registers (opt_clean, opt_dff), the counterpart of DC's OPT-1206 / OPT-1207.  Latch bits
# include one latch per mm_icg (DC maps those to library ICG cells, which the DC summary counts separately).
set -u
T65=$(cd "$(dirname "$0")/.." && pwd)
RTL=$(cd "$T65/../.." && pwd)
TAB=$T65/common/hier.txt
OUT=$RTL/build/tsmc65/rtl_regs; mkdir -p "$OUT"
FILES=$(find "$RTL/hw" -name '*.v' | sort | tr '\n' ' ')
count() {   # yosys stat -width + memory dump -> "flops N latches M" (array bits counted as flops)
    awk '/^===/ {sec = $2} {
            if ($2 ~ /^\$(a|s)?dffc?e?_[0-9]+$/) { n = split($2, t, "_"); f[sec] += t[n] * $1 }
            if ($2 ~ /^\$a?dlatch_[0-9]+$/)     { n = split($2, t, "_"); l[sec] += t[n] * $1 }
         } END { k = ("design" in f || "design" in l) ? "design" : sec; printf "%d %d", f[k], l[k] }' "$1" | {
        read -r ff ll
        mb=$(awk '/parameter .WIDTH /{w=$3} /parameter .SIZE /{s=$3} /^ *end$/ && w != "" {t += w * s; w = ""} END {print t + 0}' "${1%.stat}.mem")
        printf "flops %d (arrays %d) latches %d" $((ff + mb)) "$mb" "$ll"
    }
}
for b in "$@"; do
    read -r mod params <<< "$(awk -v b="$b" '$1 == b && $1 !~ /^[#@]/ { print $2, $3 }' "$TAB")"
    [ -n "$mod" ] || { echo "unknown block $b"; continue; }
    chp=""
    [ "$params" != "-" ] && for kv in ${params//,/ }; do chp="$chp -chparam ${kv%%=>*} ${kv##*=>}"; done
    base="read_verilog -DSYNTHESIS -DMM_TECH=2 $FILES; hierarchy -check -top $mod $chp; proc; flatten; hierarchy -top $mod; memory_collect"
    # Yosys asserts on some -chparam sets that equal the module defaults (vu_lane); those run with the defaults
    if ! yosys -q -l "$OUT/$b.raw.log" -p "$base; tee -o $OUT/$b.raw.stat stat -width; tee -q -o $OUT/$b.raw.mem dump t:\$mem_v2" > /dev/null 2>&1; then
        base="read_verilog -DSYNTHESIS -DMM_TECH=2 $FILES; hierarchy -check -top $mod; proc; flatten; hierarchy -top $mod; memory_collect"
        echo "$b: -chparam failed, using the module defaults (check they equal: $params)"
        yosys -q -l "$OUT/$b.raw.log" -p "$base; tee -o $OUT/$b.raw.stat stat -width; tee -q -o $OUT/$b.raw.mem dump t:\$mem_v2" > /dev/null 2>&1 || { echo "$b yosys failed ($OUT/$b.raw.log)"; continue; }
    fi
    yosys -q -l "$OUT/$b.opt.log" -p "$base; opt_clean -purge; opt_dff; opt_clean -purge; tee -o $OUT/$b.opt.stat stat -width; tee -q -o $OUT/$b.opt.mem dump t:\$mem_v2" > /dev/null 2>&1
    echo "rtl_regs $b $mod raw: $(count "$OUT/$b.raw.stat")  opt: $(count "$OUT/$b.opt.stat")"
done
