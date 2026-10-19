#!/usr/bin/env bash
# P8f (PLAN R17) on the server, one job at a time:  flow/tsmc65/run_p8f.sh <p8d_rtl_dir> [stage ...]
# <p8d_rtl_dir>: src/rtl of the P8d checkout, whose DC runs of the unchanged blocks are linked READ ONLY.
# Stages (default: link mm pt):
#   link  symlink build/tsmc65/hier/<block> -> <p8d_rtl_dir>/build/tsmc65/hier/<block> for every block the flat chip
#         reads except mm and its generators (chip, gm16_p<P>, vu16_p<P>, lw, seq, axil and their children)
#   mm    DC of rng9, rng3 (rn_mkgen: registered mk_req_ready) and mm (skid buffers) at <P> from this tree's RTL
#   pt    flat-chip PrimeTime at ss and <P> (as P8d: unit SPEFs, top SPEF skipped), with T65_PT_SWEEP periods
#         re-timed in the same session if <P> misses
# T65_PERIOD (default 5.25), T65_CORES (default 6).
#   cd <tree>/src/rtl || exit 1; nohup flow/tsmc65/run_p8f.sh <p8d_rtl_dir> > build/tsmc65/p8f.log 2>&1 < /dev/null &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
REF=$1; shift
P=${T65_PERIOD:-5.25}
ST=${*:-link mm pt}
H=$ROOT/build/tsmc65/hier
export T65_CORES=${T65_CORES:-6}
for s in $ST; do
    echo "STAGE $s $(date +%H:%M:%S)"
    case $s in
    link) mkdir -p "$H"
          for b in chip gm16_p$P vu16_p$P lw seq axil pe bank512_gm bank256_gm lane bank256 bank256_vu bank256_lw \
                   bank512_lw bank768_lw; do
              [ -f "$REF/build/tsmc65/hier/$b/design.txt" ] || { echo "P8F_FAILED missing $b in $REF"; exit 1; }
              [ -e "$H/$b" ] || ln -s "$REF/build/tsmc65/hier/$b" "$H/$b"
          done ;;
    mm)   T65_PERIOD=$P "$HERE/syn/run_hier.sh" rng mem ;;
    pt)   T65_PERIOD=$P T65_KID_TAG="gm16=_p$P,vu16=_p$P" T65_PT_TOP_SPEF=0 T65_PT_SUFFIX=_flat T65_PT_AREA=1 \
          T65_PT_SWEEP="${T65_PT_SWEEP:-5.5 5.75 6.0 6.25 6.5 6.75 7.0 7.25 7.5 7.75}" "$HERE/sta/run_pt_unit.sh" chip ss ;;
    *)    echo "unknown stage $s"; exit 1 ;;
    esac
done
echo "P8F_DONE $(date +%H:%M:%S)"
