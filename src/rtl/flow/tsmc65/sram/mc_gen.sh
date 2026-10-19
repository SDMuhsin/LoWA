#!/usr/bin/env bash
# P8e (PLAN R16): front-end views of the compiled SRAM / register-file macros, on the server.
#   flow/tsmc65/sram/mc_gen.sh sweep     datatable only for every candidate in T65_MC_CANDIDATES (area, access, cycle)
#   flow/tsmc65/sram/mc_gen.sh gen       liberty (all corners), Verilog model, footprint and datatable for every shape
#                                        in T65_MC_SHAPES, then the .lib files compiled to .db (sram/lib2db.tcl)
#   flow/tsmc65/sram/mc_gen.sh dbonly    only the .lib -> .db step (after a library rename)
#   flow/tsmc65/sram/mc_gen.sh clean     remove the staged compilers / runtime from the local scratch
# Everything kit-specific (compiler dir and names, corners, runtime, instance names) is in the git-ignored
# site/sram_site.sh (template site/sram_site.example.sh).  Outputs stay on the server: T65_MC_OUT/<inst>/.
#
# Why the staging (P8e attempts, docs/p8e_report.md section 2):
#   - the compilers' bundled 32-bit runtime does not start on this host, so a system Java 8 runtime runs the
#     generator's main class directly, headless, with the compiler dir as its base dir;
#   - the compilers' 32-bit helper programs stat() their own files, which fails on the NFS kit tree (inode
#     numbers above 2^32); the compiler dir is copied to a local scratch file system with small inode numbers;
#   - the footprint generator calls <java.home>/../jre_tiling/bin/java, so the runtime is copied to the scratch
#     and given that link.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SITE=$HERE/../site/sram_site.sh
[ -f "$SITE" ] || { echo "missing $SITE (template site/sram_site.example.sh)"; exit 1; }
. "$SITE"
S=$T65_MC_SCRATCH
mkdir -p "$S/kit" "$T65_MC_OUT"; chmod 700 "$S"

stage() {   # stage <compiler>: local copy of the compiler and the runtime
    [ -d "$S/kit/$1" ] || cp -r "$T65_MC_KIT/$1" "$S/kit/" || { echo "MC_ERROR staging $1"; exit 1; }
    if [ ! -x "$S/jvm/jre/bin/java" ]; then
        mkdir -p "$S/jvm/jre_tiling/bin"
        cp -rL "$T65_MC_JRE" "$S/jvm/jre" || { echo "MC_ERROR staging runtime"; exit 1; }
        ln -sf "$S/jvm/jre/bin/java" "$S/jvm/jre_tiling/bin/java"
    fi
}
run_mc() {  # run_mc <compiler> <dir> <views...> -- <options...>
    local c=$1 d=$2; shift 2
    stage "$c"; mkdir -p "$d"
    ( cd "$d" || exit 1
      timeout 600 env LANG=C LC_ALL=C LD_LIBRARY_PATH= "$S/jvm/jre/bin/java" -Djava.awt.headless=true -Xmx1g \
          -Dbasedir="$S/kit/$c" -cp "$S/kit/$c/lib/gui.zip" Main "$@" -corners "$T65_MC_CORNERS" < /dev/null > gen.log 2>&1 )
}
dat_line() {  # dat_line <inst> <datfile>: summary numbers of the first corner (ss) from the datatable
    awk -v i="$1" -v e="$T65_MC_EMA_IDX" '
        $1 == "geomx" { x = $2 } $1 == "geomy" { y = $2 } $1 == "ring_size" { r = $2 }
        $1 == "ta" e { a = $2 } $1 == "tcyc" e { c = $2 } $1 == "ta7" { a7 = $2 } $1 == "tcyc7" { c7 = $2 }
        END { printf "%s core_um2 %.0f macro_um2 %.0f ta_ns %s tcyc_ns %s ta_ema7_ns %s tcyc_ema7_ns %s\n",
              i, x * y, (x + 2 * r) * (y + 2 * r), a, c, a7, c7 }' "$2"
}

case "${1:-}" in
sweep)
    for spec in $T65_MC_CANDIDATES; do
        IFS=: read -r w b c m <<< "$spec"
        i=sw_${c%%_*}${w}x${b}m$m
        run_mc "$c" "$S/sweep/$i" ascii -words "$w" -bits "$b" -mux "$m" -instname "$i"
        if [ -f "$S/sweep/$i/$i.dat" ]; then echo "SWEEP $w x $b $c mux $m: $(dat_line "$i" "$S/sweep/$i/$i.dat" | cut -d' ' -f2-)"
        else echo "SWEEP $w x $b $c mux $m: FAIL $(grep -m1 -i error "$S/sweep/$i/gen.log")"; fi
    done
    rm -rf "$S/sweep" ;;
gen)
    for spec in $T65_MC_SHAPES; do
        IFS=: read -r w b c m i <<< "$spec"
        d=$S/gen/$i; rm -rf "$d"
        run_mc "$c" "$d" synopsys verilog vclef-fp ascii -words "$w" -bits "$b" -mux "$m" -instname "$i"
        o=$T65_MC_OUT/$i; mkdir -p "$o"
        n=0
        for f in "$d/$i.v" "$d/$i.vclef" "$d/$i.dat" "$d"/${i}_*_syn.lib; do [ -f "$f" ] && cp -f "$f" "$o/" && n=$((n + 1)); done
        cp -f "$d/gen.log" "$o/gen.log"
        # every generated liberty names its library USERLIB: name it after the file (<inst>_<corner>_syn) so that
        # several macros and corners can be linked in one DC / PT session
        for f in "$o"/${i}_*_syn.lib; do lb=$(basename "$f" .lib); sed -i "0,/^library *( *[A-Za-z0-9_]* *)/s//library($lb)/" "$f"; done
        echo "GEN $w x $b $c mux $m files $n $(grep -c 'generator succeeded' "$d/gen.log") ok $(grep -c 'generator failed' "$d/gen.log") failed"
        [ -f "$o/$i.dat" ] && echo "MACRO $w x $b $(dat_line "$i" "$o/$i.dat")"
    done
    rm -rf "$S/gen"
    ;;&
dbonly|gen)
    # liberty -> db (Library Compiler), one run for every .lib
    ( cd "$T65_MC_OUT" || exit 1
      tcsh -f -c "source $T65_MC_LC_SETUP >& /dev/null; lc_shell -f $HERE/lib2db.tcl" > lib2db.log 2>&1 )
    echo "LIB2DB $(grep -c '^DB_OK' "$T65_MC_OUT/lib2db.log") ok $(grep -c '^DB_FAIL' "$T65_MC_OUT/lib2db.log") failed, errors $(grep -c '^Error' "$T65_MC_OUT/lib2db.log")"
    du -sh "$T65_MC_OUT" | sed 's/^/MC_OUT_SIZE /' ;;
clean)
    rm -rf "$S/kit" "$S/jvm"; echo "scratch cleaned" ;;
*)  echo "usage: $0 sweep|gen|dbonly|clean"; exit 1 ;;
esac
echo "MC_GEN_DONE"
