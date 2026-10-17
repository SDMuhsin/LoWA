# Per-block setup slack of the LoWA unit after OpenROAD repair_design (same flow as repair_sta.tcl:
# flow/synth/synth_lowa.sh keeps lw_sg / lw_adamw / lw_merge / lw_scal as hierarchy, so paths can be
# grouped by the instance their endpoint lies in).  Prints "BLOCK <name> wns_setup <ns>" lines.
set pdk $::env(PDK_ROOT)/ciel/gf180mcu/versions
set scl [glob $pdk/*/gf180mcuD/libs.ref/gf180mcu_as_sc_mcu7t3v3]
read_lef $scl/techlef/gf180mcu_as_sc_mcu7t3v3__nom.tlef
read_lef $scl/lef/gf180mcu_as_sc_mcu7t3v3.lef
read_liberty $scl/lib/gf180mcu_as_sc_mcu7t3v3__$::env(CORNER).lib
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
repair_design -max_wire_length 2000
estimate_parasitics -placement
puts [format "STA %s wns_setup %.2f wns_hold %.2f (after repair_design)" $::env(CORNER) [worst_slack -max] [worst_slack -min]]
foreach blk {u.u_sg u.u_ad u.u_mg u.u_sc} {
    set pins [get_pins -hierarchical $blk/*/D]
    if {[llength $pins] == 0} { set pins [get_pins -hierarchical $blk/*/*/D] }
    set paths [find_timing_paths -path_delay max -to $pins -group_path_count 1]
    if {[llength $paths] == 0} { puts "BLOCK $blk wns_setup none"; continue }
    set p [lindex $paths 0]
    puts [format "BLOCK %s wns_setup %.2f endpoint %s" $blk [get_property $p slack] [get_full_name [get_property $p endpoint]]]
}
report_checks -path_delay max -group_path_count 1 -digits 2
