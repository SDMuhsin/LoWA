#!/usr/bin/env bash
# Integrity scan of one DC run directory (dc.log + rpt/check_design.rpt); prints counts and RTL module names only.
#   tools/dclog_check.sh <run dir>
#   integ_latch_rf / integ_latch_other   latch inferences (Presto) inside mm_bank_t65 / anywhere else (+ modules)
#   integ_unresolved                      LINK-5 unresolved references + check_design unresolved / black-box lines
#   integ_opt_const / unloaded / merged   registers DC removed as constant / unloaded / merged (OPT-1206/1207/1215)
#   integ_errors                          ^Error lines without the harmless PROC-3
#   integ_warn                            the most frequent warning codes
d=$1; log=$d/dc.log; cd_rpt=$d/rpt/check_design.rpt
[ -f "$log" ] || { echo "integ_missing $log"; exit 0; }
awk '
    /in routine/ { for (i = 1; i <= NF; i++) if ($i == "routine") r = $(i+1) }
    /^\|/ && /\| *Latch *\|/ { n = split($0, f, "|"); w = f[4] + 0; if (r ~ /^mm_bank_t65/) rf += w; else { ot += w; om[r] += w } }
    END {
        printf "integ_latch_rf %d\n", rf
        s = ""; for (m in om) s = s " " m "(" om[m] ")"
        printf "integ_latch_other %d%s\n", ot, s
    }' "$log"
echo "integ_unresolved $(grep -c 'LINK-5' "$log") check_design $( [ -f "$cd_rpt" ] && grep -ciE 'unresolved|black.?box' "$cd_rpt" || echo na)"
echo "integ_opt_const $(grep -c 'OPT-1206' "$log") unloaded $(grep -c 'OPT-1207' "$log") merged $(grep -c 'OPT-1215' "$log")"
echo "integ_errors $(grep '^Error' "$log" | grep -vc PROC-3)"
echo "integ_warn $(grep -o '^Warning: .*(\([A-Z]*-[0-9]*\))$' "$log" | grep -o '([A-Z]*-[0-9]*)$' | sort | uniq -c | sort -rn | head -8 | awk '{printf "%s%s ", $2, $1}')"
