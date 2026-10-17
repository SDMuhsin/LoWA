# OpenSTA on a Yosys netlist: env NETLIST, TOP, CORNER, PERIOD, SDC.  Prints one summary line
#   STA <corner> wns_setup <ns> wns_hold <ns> r2r_arrival <ns>
set libdir [glob $::env(PDK_ROOT)/ciel/gf180mcu/versions/*/gf180mcuD/libs.ref/gf180mcu_as_sc_mcu7t3v3/lib]
read_liberty $libdir/gf180mcu_as_sc_mcu7t3v3__$::env(CORNER).lib
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
report_checks -path_delay max -group_path_count 1 -digits 2
report_checks -path_delay min -digits 2
set ws [worst_slack -max]
set wh [worst_slack -min]
puts [format "STA %s wns_setup %.2f wns_hold %.2f" $::env(CORNER) $ws $wh]
