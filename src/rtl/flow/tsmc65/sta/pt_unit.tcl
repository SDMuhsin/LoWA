# PrimeTime of one unit's complete netlist (pre-layout, ideal clocks), optionally with PrimePower average power,
# and the chip level on extracted timing models of its units.
# Unit mode: a parent run (syn/dc_unit.tcl) links its children as block abstractions, so its netlist does not
# define them: this script reads the unit netlist plus every block netlist below it (a child linked as `abs` is
# a separate netlist; a child linked as a full netlist is already inside its parent's), links the flat unit,
# annotates each block's DC topographical parasitics (ss RC estimate) on every instance of that block, and with
# T65_ETM=1 writes the unit's extracted timing model (extract_model) for the chip level.
# Chip mode (T65_USE_ETM=1): only the top netlist and its own parasitics; each child unit is its extracted model
# from the unit run at the same corner (a flat chip netlist, ~5 M cells, does not fit the 30 GB server in PT).
# Environment (sta/run_pt_unit.sh):
#   T65_ROOT, T65_PERIOD, T65_CORNER (ss | tt | ff), T65_PT_OUT (output dir), T65_KID_TAG (as for DC)
#   T65_BLOCK block (common/hier.txt)   T65_PT_AREA 1: per-instance cell / area / register counts
#   T65_ETM 1: write the extracted model   T65_USE_ETM 1: chip mode   T65_ETM_DIR dir of the unit runs (build/tsmc65/pt)
#   T65_POWER  none | vectorless | saif     T65_SAIF  "file strip_path instance_path ..." (saif mode)
#   T65_TR / T65_SP  vectorless default toggle rate (per clock cycle) / static probability (0.1 / 0.5)
set t0 [clock seconds]
set flow $::env(T65_ROOT)/flow/tsmc65
source $flow/common/site.tcl
source $flow/common/hier.tcl
source $flow/common/summary.tcl
t65_hier_load
proc envd {k d} { expr {[info exists ::env($k)] ? $::env($k) : $d} }
set top    [envd T65_BLOCK chip]
set period $::env(T65_PERIOD)
set corner $::env(T65_CORNER)
set out    $::env(T65_PT_OUT)
set power  [envd T65_POWER none]
set hroot  $::env(T65_ROOT)/build/tsmc65/hier
file mkdir $out
set_host_options -max_cores [envd T65_CORES 8]
if {$power ne "none"} {
    set_app_var power_enable_analysis true
    set_app_var power_analysis_mode averaged
}
# blocks whose netlist is read: the top and, recursively, every child its parent links as an abstraction
proc t65_tree {b} {
    set l [list $b]
    if {[t65_hier $b link] eq "abs"} { foreach k [t65_hier $b children] { set l [concat $l [t65_tree $k]] } }
    return $l
}
set useetm [envd T65_USE_ETM 0]
set blocks [expr {$useetm ? [list $top] : [lsort -unique [t65_tree $top]]}]
set etms {}
if {$useetm} {
    foreach k [t65_hier $top children] {
        set g [glob -nocomplain [envd T65_ETM_DIR $::env(T65_ROOT)/build/tsmc65/pt]/$k/etm_${corner}*.db]
        if {[llength $g] != 1} { puts "T65_ERROR no extracted model of $k at $corner"; exit 1 }
        lappend etms [lindex $g 0]
    }
}
set files {}; set bdesign [dict create]; set bdir [dict create]
foreach b $blocks {
    set d [t65_kid_dir $hroot $b]
    set fh [open $d/design.txt]; dict set bdesign $b [string trim [read $fh]]; close $fh
    dict set bdir $b $d
    lappend files $d/$b.v
}
set_app_var search_path [concat . $::T65_LIBDIR]
set_app_var link_path [concat * [t65_db $corner] $etms]
read_verilog $files
current_design [dict get $bdesign $top]
link_design
set ::argv [list $period $corner $out/pt_$corner.sdc [t65_hier $top io]]
source $flow/sdc/gen_sdc.tcl
source $out/pt_$corner.sdc

# parasitics: each block's own SPEF on each of its instances
set npar 0; set nparf 0
foreach b $blocks {
    set f [dict get $bdir $b]/$b.spef
    if {![file exists $f]} { puts "T65_NOTE no spef for $b"; continue }
    if {$b eq $top} {
        # T65_PT_TOP_SPEF=0: skip the top's own SPEF.  The chip run's topographical parasitics come from unplaced
        # unit abstractions (OPT-1442) and also cover nets inside the units, where they would override the units'
        # own SPEF; top_chip has no cells, so nothing of its own is lost.
        if {![envd T65_PT_TOP_SPEF 1]} { puts "T65_NOTE top spef skipped"; continue }
        if {[catch {read_parasitics -format spef $f} e]} { incr nparf; puts "T65_NOTE spef $b: $e" } else { incr npar }
        continue
    }
    foreach_in_collection c [get_cells -quiet -hierarchical -filter "ref_name==[dict get $bdesign $b]"] {
        if {[catch {read_parasitics -format spef -path [get_object_name $c] $f} e]} { incr nparf } else { incr npar }
    }
}
update_timing
redirect $out/rpt_annot_$corner.rpt { report_annotated_parasitics -list_not_annotated -max_nets 20 }

set fh [open $out/summary_$corner.txt w]
puts $fh "block $top corner $corner period_ns $period netlists [llength $files] ([join $blocks ,]) etm_children [llength $etms] clocks ideal (pre-CTS)"
puts $fh "parasitics dc_topo_ss_estimate spef_reads $npar failed $nparf"
set r [open $out/rpt_annot_$corner.rpt]
foreach line [split [read $r] "\n"] { if {[regexp {^\s*(Total|.*\|\s*[0-9]+\s*\|)} $line]} { puts $fh "annot [string trim $line]" } }
close $r
puts $fh "leaf_cells [sizeof_collection [get_cells -quiet -hierarchical -filter is_hierarchical==false]]"
t65_put_timing $fh
# the worst paths split by top-level instance (arrival spent in each instance along the path, ns)
proc t65_path_top {p} {
    set s ""; set prev 0.0; set cur ""; set acc 0.0
    foreach_in_collection pt [get_attribute $p points] {
        set a [get_attribute $pt arrival]; set c [split [get_object_name [get_attribute $pt object]] /]
        set k [expr {[llength $c] > 1 ? [lindex $c 0] : "top"}]
        if {$k ne $cur && $cur ne ""} { append s [format " %s %.2f >" $cur $acc]; set acc 0.0 }
        set cur $k; set acc [expr {$acc + $a - $prev}]; set prev $a
    }
    if {$cur ne ""} { append s [format " %s %.2f" $cur $acc] }
    return $s
}
set i 0
foreach_in_collection q [get_timing_paths -delay_type max -max_paths 3 -nworst 1] {
    incr i; puts $fh "worst_split $i [t65_path_line $q] |[t65_path_top $q]"
}
# worst setup / hold slack per unit instance (endpoints inside the instance) and at the top level
set regd [all_registers -data_pins]
set units [get_cells -quiet -filter is_hierarchical==true]
foreach_in_collection u $units {
    set un [get_object_name $u]
    set e [filter_collection $regd "full_name=~$un/*"]
    if {[sizeof_collection $e] == 0} continue
    set ps [get_timing_paths -delay_type max -max_paths 1 -to $e]
    set ph [get_timing_paths -delay_type min -max_paths 1 -to $e]
    set nv [sizeof_collection [get_timing_paths -delay_type max -max_paths 100000 -nworst 1 -slack_lesser_than -0.0005 -to $e]]
    set nh [sizeof_collection [get_timing_paths -delay_type min -max_paths 100000 -nworst 1 -slack_lesser_than -0.0005 -to $e]]
    puts $fh [format "unit %s ref %s endpoints %d setup_wns %s setup_nviol %d hold_wns %s hold_nviol %d" $un \
        [get_attribute $u ref_name] [sizeof_collection $e] [t65_n [get_attribute $ps slack]] $nv [t65_n [get_attribute $ph slack]] $nh]
}
set e [filter_collection $regd "full_name!~*/*"]
if {[sizeof_collection $e]} {
    puts $fh [format "unit top_glue endpoints %d setup_wns %s hold_wns %s" [sizeof_collection $e] \
        [t65_n [get_attribute [get_timing_paths -delay_type max -max_paths 1 -to $e] slack]] \
        [t65_n [get_attribute [get_timing_paths -delay_type min -max_paths 1 -to $e] slack]]]
}
foreach {lbl io} [list in_to_reg [all_inputs] reg_to_out [all_outputs]] {
    if {$lbl eq "in_to_reg"} { set p [get_timing_paths -delay_type max -max_paths 1 -from $io] } else { set p [get_timing_paths -delay_type max -max_paths 1 -to $io] }
    puts $fh "io $lbl [t65_path_line $p]"
}
# T65_PT_SWEEP "p1 p2 ...": after the reports at T65_PERIOD, re-time the same session at each further period (SDC
# regenerated by sdc/gen_sdc.tcl: clock and IO delays scale with it) until setup closes; one summary line per period
if {[envd T65_PT_SWEEP ""] ne "" && [get_attribute [get_timing_paths -delay_type max -max_paths 1] slack] < 0} {
    flush $fh
    set sfh $fh
    foreach sp [envd T65_PT_SWEEP ""] {
        set ::argv [list $sp $corner $out/pt_${corner}_p$sp.sdc [t65_hier $top io]]
        source $flow/sdc/gen_sdc.tcl
        set fh $sfh                         ;# gen_sdc.tcl uses (and closes) its own global fh
        source $out/pt_${corner}_p$sp.sdc
        update_timing
        set q [get_timing_paths -delay_type max -max_paths 1]
        set ws [get_attribute $q slack]
        set nv [sizeof_collection [get_timing_paths -delay_type max -max_paths 100000 -nworst 1 -slack_lesser_than -0.0005]]
        puts $fh "sweep period_ns $sp setup_wns [t65_n $ws] setup_nviol $nv path [t65_path_line $q] |[t65_path_top $q]"
        flush $fh
        if {$ws >= 0} break
    }
}
if {[envd T65_PT_AREA 0]} {
    # per unit instance (and the top-level glue): leaf cells, cell area, flops and latches (ICG latches excluded)
    proc lsum {l} { set s 0.0; foreach v $l { if {$v ne ""} { set s [expr {$s + $v}] } }; return $s }
    set icg [get_cells -quiet -hierarchical -filter "is_hierarchical==false && ref_name=~$::T65_ICG_PATTERN"]
    set allff [remove_from_collection [all_registers -edge_triggered] $icg]
    set alllat [remove_from_collection [all_registers -level_sensitive] $icg]
    set rows {}
    foreach_in_collection u $units { lappend rows [get_object_name $u] "full_name=~[get_object_name $u]/*" }
    lappend rows top_glue "full_name!~*/*"
    foreach {un flt} $rows {
        set c [get_cells -quiet -hierarchical -filter "is_hierarchical==false && $flt"]
        puts $fh [format "area_unit %s cells %d area_um2 %.1f flops %d latches %d icg %d" $un [sizeof_collection $c] \
            [lsum [get_attribute -quiet $c area]] [sizeof_collection [filter_collection $allff $flt]] \
            [sizeof_collection [filter_collection $alllat $flt]] [sizeof_collection [filter_collection $icg $flt]]]
    }
}
if {$power ne "none"} {
    if {$power eq "saif"} {
        foreach {f strip inst} $::env(T65_SAIF) {
            if {$inst eq ""} { redirect -append $out/rpt_saif.rpt { read_saif -strip_path $strip $f } } else { redirect -append $out/rpt_saif.rpt { read_saif -strip_path $strip -path $inst $f } }
        }
        redirect $out/rpt_switching.rpt { report_switching_activity -list_not_annotated -max_objects 20 }
    } else {
        set_app_var power_default_toggle_rate [envd T65_TR 0.1]
        set_app_var power_default_static_probability [envd T65_SP 0.5]
        redirect $out/rpt_switching.rpt { report_switching_activity }
    }
    update_power
    redirect $out/rpt_power_$corner.rpt { report_power -nosplit }
    redirect $out/rpt_power_hier_$corner.rpt { report_power -nosplit -hierarchy -levels 1 }
    puts $fh "power_mode $power tr [envd T65_TR 0.1] sp [envd T65_SP 0.5] saif {[envd T65_SAIF {}]}"
    set r [open $out/rpt_power_$corner.rpt]
    foreach line [split [read $r] "\n"] {
        if {[regexp {^\s*(Net Switching Power|Cell Internal Power|Cell Leakage Power|Total Power|clock_network|register|combinational|sequential|memory|black_box|io_pad|Total)\s} $line]} { puts $fh "power [string trim $line]" }
    }
    close $r
    set r [open $out/rpt_power_hier_$corner.rpt]
    foreach line [split [read $r] "\n"] { if {[regexp {^\s*(u_\S+|top_chip\S*)\s+\(} $line]} { puts $fh "power_hier [regsub -all {\s+} [string trim $line] { }]" } }
    close $r
    set r [open $out/rpt_switching.rpt]
    foreach line [split [read $r] "\n"] { if {[regexp {^\s*(Nets|Pins|Registers|Combinational|Sequential|Clock|Primary|Black|Total|Memory)} $line]} { puts $fh "switching [regsub -all {\s+} [string trim $line] { }]" } }
    close $r
}
if {[envd T65_ETM 0]} {
    if {[catch {extract_model -output $out/etm_$corner -format {db}} e]} { puts $fh "etm failed: [string range $e 0 200]" } else {
        puts $fh "etm [file tail [lindex [glob -nocomplain $out/etm_${corner}*.db] 0]]"
    }
}
puts $fh [format "pt_wall_s %d pt_mem_mb %s" [expr {[clock seconds] - $t0}] [mem]]
close $fh
exit
