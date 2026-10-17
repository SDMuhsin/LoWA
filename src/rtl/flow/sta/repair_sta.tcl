# OpenROAD pre-layout STA with high-fanout repair (P3): Yosys inserts no buffer trees, so nets
# such as a register-file read pointer driving thousands of mux inputs dominate a raw OpenSTA
# report.  This script buffers them (repair_design, no placement: pin loads only, wire RC of
# Metal2 as a nominal estimate) and reports the resulting worst setup path, which is what a
# placed design would see before timing-driven optimisation.  Env: NETLIST, TOP, CORNER, PERIOD, SDC.
set pdk $::env(PDK_ROOT)/ciel/gf180mcu/versions
set scl [glob $pdk/*/gf180mcuD/libs.ref/gf180mcu_as_sc_mcu7t3v3]
read_lef $scl/techlef/gf180mcu_as_sc_mcu7t3v3__nom.tlef
read_lef $scl/lef/gf180mcu_as_sc_mcu7t3v3.lef
read_liberty $scl/lib/gf180mcu_as_sc_mcu7t3v3__$::env(CORNER).lib
# the SRAM macro (blackbox in the Yosys netlist): the signed-off GanAccel copy of the GF180 ocd IP
set sram /workspace/GanAccelAsic/gf180mcu_ocd_ip_sram__sram256x8m8wm1
if {[file exists $sram/gf180mcu_ocd_ip_sram__sram256x8m8wm1.lef]} {
    read_lef $sram/gf180mcu_ocd_ip_sram__sram256x8m8wm1.lef
    read_liberty $sram/gf180mcu_ocd_ip_sram__sram256x8m8wm1__$::env(CORNER).lib
}
read_verilog $::env(NETLIST)
link_design $::env(TOP)
read_sdc $::env(SDC)
set_wire_rc -layer Metal2
set_dont_use {*_1 *_20}
set_max_fanout 24 [current_design]
set_max_transition 3.0 [current_design]
puts "RAW [format %.2f [worst_slack -max]]"
repair_design -max_wire_length 2000
estimate_parasitics -placement
report_checks -path_delay max -group_path_count 1 -digits 2
puts [format "STA %s wns_setup %.2f wns_hold %.2f (after repair_design)" $::env(CORNER) [worst_slack -max] [worst_slack -min]]
