#!/usr/bin/env bash
# P9b progress summary on the server: segment simulations (last heartbeat, SAIF windows written) and the PrimePower run.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
B=$ROOT/build/tsmc65
date -u +"now %H:%M:%S"
grep -E "GLS_" "$B/p9b_gls.log" 2>/dev/null | tail -8
for g in A B C D E; do
    f=$B/gls/seg$g/sim.out
    [ -f "$f" ] || continue
    echo "seg$g $(grep -E '^(T0|HB cyc|SAIF_WRITTEN|SAIF_ALL|STOP|FAIL|GMERR|MMERR)|Error' "$f" | tail -2 | tr '\n' ' ') last_wall $(grep HBWALL "$f" | tail -1 | cut -c8-)"
done
ls -la "$B"/gls/seg*/*.saif 2>/dev/null | awk '{print "saif", $5, $9}' | sed "s#$B/gls/##"
for f in "$B"/pt_gls/*/summary_gls_tt.txt; do grep -E "^(leaf|window |power .*Total|window_s|windows_not|pt_wall)" "$f"; done
grep -c "^Error" "$B/pt_gls/gls1/pt_tt.log" 2>/dev/null | sed 's/^/pt_errors /'
ps -u "$USER" -o pid,etime,rss,args | grep -E "[s]imv|[p]t_shell" | cut -c1-60
free -g | sed -n 2p
