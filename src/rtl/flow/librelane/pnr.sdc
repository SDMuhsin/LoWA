# PnR SDC for mb_chip (PNR_SDC_FILE): base.sdc over-constrained to max_transition 0.9 ns /
# max_capacitance 0.12 pF so resizer repair leaves headroom for routed (SPEF) parasitics and
# antenna diodes (same as librelane/fp16_pnr.sdc).
set ::mb_max_tran 0.9
set ::mb_max_cap 0.12
source [file join [file dirname [file normalize [info script]]] base.sdc]
