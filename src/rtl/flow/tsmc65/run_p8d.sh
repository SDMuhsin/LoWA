#!/usr/bin/env bash
# P8d batch at the chip clock, one job at a time:  flow/tsmc65/run_p8d.sh <period> [stage ...]
# Stages (default: stale chip fm pt):
#   stale    lw (with its three bank copies) and mm rebuilt at 4.0 ns from the current RTL: the server checkout still
#            held their P8b runs (RTL before the P8c / P8c2 repairs of lw_merge, mm_top, mm_rdport; the P8c runs were
#            made in a work tree that no longer exists).  Every other block's run postdates its last RTL change.
#   chip     top_chip DC at <period>, linking gm16 / vu16 from their exit-rule runs (hier/<unit>_p<period>)
#   fm       Formality of every block of the chip netlist (lec/run_fm.sh)
#   pt       PrimeTime of every unit at ss / tt / ff (complete netlists, extracted models), PrimePower vectorless at
#            tt, then the chip on the models at each corner
#   ptsaif   PrimePower of gm16 / vu16 at tt with the RTL SAIF of sim/run_saif.sh (T65_SAIF_GM, T65_SAIF_VU)
#   cd <tree>/src/rtl || exit 1; nohup flow/tsmc65/run_p8d.sh 5.0 > build/tsmc65/p8d.log 2>&1 < /dev/null &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
P=$1; shift
ST=${*:-stale chip fm pt}
export T65_KID_TAG="gm16=_p$P,vu16=_p$P"
UNITS="gm16 vu16 lw mm seq axil"
for s in $ST; do
    echo "STAGE $s $(date +%H:%M:%S)"
    case $s in
    stale)   T65_PERIOD=4.0 FORCE=1 "$HERE/syn/run_hier.sh" lw mm ;;
    chip)    T65_PERIOD=$P "$HERE/syn/run_hier.sh" chip ;;
    fm)      "$HERE/lec/run_fm.sh" pe bank512_gm bank256_gm gm16:_p$P lane bank256_vu vu16:_p$P \
                 bank256_lw bank512_lw bank768_lw lw rng9 rng3 mm seq axil chip ;;
    pt)      for c in ss tt ff; do
                 pw=none; [ $c = tt ] && pw=vectorless
                 for u in $UNITS; do T65_PERIOD=$P T65_ETM=1 T65_PT_AREA=$([ $c = ss ] && echo 1 || echo 0) "$HERE/sta/run_pt_unit.sh" $u $c $pw; done
                 T65_PERIOD=$P T65_USE_ETM=1 "$HERE/sta/run_pt_unit.sh" chip $c
             done ;;
    ptsaif)  T65_PERIOD=$P T65_SAIF="${T65_SAIF_GM:?} tb_gm_top/dut {}" T65_PT_SUFFIX=_rtlsaif "$HERE/sta/run_pt_unit.sh" gm16 tt saif
             T65_PERIOD=$P T65_SAIF="${T65_SAIF_VU:?} tb_vu_top/dut {}" T65_PT_SUFFIX=_rtlsaif "$HERE/sta/run_pt_unit.sh" vu16 tt saif ;;
    *) echo "unknown stage $s"; exit 1 ;;
    esac
done
echo "P8D_DONE $(date +%H:%M:%S)"
