# Shared ICC2 helpers of flow/tsmc65/pnr (P9c): floorplan, placement and clock tree of one block of common/hier.txt
# (a unit with all the netlists below it, or the chip), global routing for parasitics.  No detail routing.
# Environment (pnr/run_pnr.sh):
#   T65_ROOT   src/rtl of the tree holding the scripts     T65_HIER   build/tsmc65/hier with the DC netlists
#   T65_BLOCK  block (common/hier.txt)                    T65_PERIOD clock period, ns
#   T65_PNR    run directory (t65.nlib, rpt/, out/)        T65_NDM    reference library (pnr/ndm.tcl)
#   T65_UTIL   target core utilisation (0.65)             T65_CORES  threads (8)
#   T65_KID_TAG  child run tags, as for DC / PT           T65_PNR_SETUP_MARGIN  extra ICC2 setup uncertainty (0.10 ns)
# Each stage starts from the previous stage's snapshot <design>_<stage> and saves its own.
set ::t0 [clock seconds]
set ::flow $::env(T65_ROOT)/flow/tsmc65
source $::flow/common/site.tcl
source $::flow/site/site_pnr.tcl
source $::flow/common/hier.tcl
t65_hier_load
proc envd {k d} { expr {[info exists ::env($k)] ? $::env($k) : $d} }
set ::top    $::env(T65_BLOCK)
set ::period $::env(T65_PERIOD)

proc t65p_lib {} { return $::env(T65_PNR)/t65.nlib }
proc t65p_rpt {name} { file mkdir $::env(T65_PNR)/rpt; return $::env(T65_PNR)/rpt/$name }
proc t65p_host {} { set_host_options -max_cores [envd T65_CORES 8] }
proc t65p_log {msg} { puts "T65P [clock format [clock seconds] -format %H:%M:%S] [expr {[clock seconds] - $::t0}]s mem_mb [mem] $msg"; flush stdout }

# netlists: the block and, recursively, every child its parent links as an abstraction (as sta/pt_unit.tcl)
proc t65p_tree {b} {
    set l [list $b]
    if {[t65_hier $b link] eq "abs"} { foreach k [t65_hier $b children] { set l [concat $l [t65p_tree $k]] } }
    return $l
}
proc t65p_netlists {} {
    set files {}
    foreach b [lsort -unique [t65p_tree $::top]] {
        set d [t65_kid_dir $::env(T65_HIER) $b]
        if {$b eq $::top} { set fh [open $d/design.txt]; set ::design [string trim [read $fh]]; close $fh }
        lappend files $d/$b.v
    }
    return $files
}
proc t65p_design {} {
    set fh [open [t65_kid_dir $::env(T65_HIER) $::top]/design.txt]; set d [string trim [read $fh]]; close $fh
    return $d
}

proc t65p_open_stage {from} {
    t65p_host
    open_lib [t65p_lib]
    set d [t65p_design]
    copy_block -from_block ${d}_$from -to_block $d -force
    current_block $d
    link_block
    t65p_margins
    t65p_log "opened from $from"
}
proc t65p_save_stage {stage} {
    set d [t65p_design]
    save_block
    copy_block -from_block $d -to_block ${d}_$stage -force
    save_lib -all
    t65p_log "saved $stage"
}

proc t65p_layers {} {
    set_ignored_layers -min_routing_layer $::T65_LAYER_MIN -max_routing_layer $::T65_LAYER_MAX
    foreach l [get_object_name [get_layers -filter "layer_type==interconnect"]] {
        if {![regexp {^M([0-9]+)$} $l -> n]} { continue }
        set_attribute [get_layers $l] routing_direction [expr {$n % 2 ? "horizontal" : "vertical"}]
    }
}
proc t65p_cell_usage {} {
    foreach p $::T65_PNR_DONT_USE {
        set c [get_lib_cells -quiet */$p]
        if {[sizeof_collection $c]} { set_dont_use $c }
    }
    foreach p $::T65_TIE {
        set c [get_lib_cells -quiet */$p]
        if {[sizeof_collection $c]} { set_lib_cell_purpose -include optimization $c }
    }
    set cts {}
    foreach p $::T65_CTS_CELLS { append_to_collection cts [get_lib_cells -quiet */$p] }
    set_lib_cell_purpose -exclude cts [get_lib_cells */*]
    set_lib_cell_purpose -include cts $cts
}

# MCMM: one mode, ss (setup, worst RC) and ff (hold, best RC); tt is analysed in PrimeTime / PrimePower afterwards.
# Constraints are the synthesis SDC (sdc/gen_sdc.tcl) of each corner, with the library name of the reference library.
if {[info commands load_of] eq ""} {
    proc load_of {pin} { return [lindex [get_attribute [get_lib_pins */[join [lrange [split $pin /] end-1 end] /]] pin_capacitance] 0] }
}
proc t65p_scenarios {} {
    foreach c {ss ff} {
        read_parasitic_tech -tlup $::T65_TLU/$::T65_CORNER_TLU($c) -layermap $::T65_TLU_MAP -name rc_$c
    }
    create_mode func
    foreach c {ss ff} {
        create_corner $c
        set_parasitic_parameters -corner $c -late_spec rc_$c -early_spec rc_$c
        lassign $::T65_CORNER_PVT($c) v t
        set_voltage $v -corner $c -object_list [get_supply_nets [lindex $::T65_PG 0]]
        set_voltage 0.0 -corner $c -object_list [get_supply_nets [lindex $::T65_PG 1]]
        set_temperature $t -corner $c
        set_process_number 1.0 -corner $c
        create_scenario -mode func -corner $c -name func_$c
    }
    foreach c {ss ff} {
        current_scenario func_$c
        set sdc $::env(T65_PNR)/icc2_$c.sdc
        set argv [list $::period $c $sdc.raw [t65_hier $::top io]]   ;# gen_sdc.tcl runs in this proc scope
        source $::flow/sdc/gen_sdc.tcl
        set fh [open $sdc.raw]; set txt [read $fh]; close $fh
        set txt [string map [list "-library [t65_lib $c] " "-library t65_std " "load_of [t65_lib $c]/" "load_of t65_std/"] $txt]
        set fh [open $sdc w]; puts -nonewline $fh $txt; close $fh
        source $sdc
    }
    t65p_margins
    set_scenario_status func_ss -active true -setup true  -hold false -leakage_power false -dynamic_power false -max_transition true -max_capacitance true
    set_scenario_status func_ff -active true -setup false -hold true  -leakage_power false -dynamic_power false -max_transition false -max_capacitance false
    current_scenario func_ss
}
# ICC2-only extra setup uncertainty (its delay calculation reads paths faster than PrimeTime); PrimeTime signs off with the
# plain SDC.  Clock-path transition target for CTS.
proc t65p_margins {} {
    set cur [current_scenario]
    foreach_in_collection sc [get_scenarios] {
        current_scenario $sc
        set_clock_uncertainty -setup [expr {0.25 + [envd T65_PNR_SETUP_MARGIN 0.10]}] [get_clocks clk]
        set_clock_uncertainty -hold  [expr {0.05 + [envd T65_PNR_HOLD_MARGIN 0.02]}] [get_clocks clk]
        set_max_transition [envd T65_PNR_CLK_TRAN 0.25] -clock_path [get_clocks clk]
    }
    current_scenario $cur
    set_app_options -name opt.common.max_fanout -value 16
    set_app_options -name cts.common.max_fanout -value 24
}
proc t65p_stage_reports {stage} {
    redirect [t65p_rpt ${stage}_qor.rpt]      { report_qor -summary }
    redirect [t65p_rpt ${stage}_setup_ss.rpt] { report_timing -scenarios func_ss -max_paths 3 -nosplit }
    redirect [t65p_rpt ${stage}_hold_ff.rpt]  { report_timing -scenarios func_ff -delay_type min -max_paths 3 -nosplit }
    redirect [t65p_rpt ${stage}_util.rpt]     { report_utilization }
    set fh [open $::env(T65_PNR)/summary_$stage.txt w]
    puts $fh "block $::top design [t65p_design] stage $stage period_ns $::period util_target [envd T65_UTIL 0.65] cores [envd T65_CORES 8]"
    set leaf [get_cells -hierarchical -quiet -filter "is_hierarchical==false"]
    set a 0.0; foreach v [get_attribute -quiet $leaf area] { set a [expr {$a + $v}] }
    set cb [get_attribute [get_core_area] bbox]
    lassign $cb ll ur
    set cw [expr {[lindex $ur 0] - [lindex $ll 0]}]; set ch [expr {[lindex $ur 1] - [lindex $ll 1]}]
    puts $fh [format "core_um %.1f x %.1f core_area_mm2 %.4f leaf_cells %d cell_area_mm2 %.4f utilisation %.3f" \
        $cw $ch [expr {$cw*$ch/1e6}] [sizeof_collection $leaf] [expr {$a/1e6}] [expr {$a/($cw*$ch)}]]
    foreach {sc t} {func_ss max func_ff min} {
        set p [get_timing_paths -scenarios $sc -delay_type $t -max_paths 1]
        set nv [sizeof_collection [get_timing_paths -scenarios $sc -delay_type $t -max_paths 100000 -nworst 1 -slack_lesser_than -0.0005]]
        puts $fh [format "icc2 %s %s wns %s endpoints_viol %d" $sc [expr {$t eq "max" ? "setup" : "hold"}] [get_attribute $p slack] $nv]
    }
    puts $fh "wall_s [expr {[clock seconds] - $::t0}] mem_mb [mem]"
    close $fh
}
