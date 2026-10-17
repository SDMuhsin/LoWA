# Compact summary writer of the TSMC-65 flow: numbers and RTL instance names only (no library data),
# safe to copy off the server.  Used by syn/dc_block.tcl (DC); the timing part also runs in PrimeTime.

proc t65_leaf {pattern} {
    return [get_cells -quiet -hierarchical -filter "is_hierarchical==false && full_name=~$pattern"]
}
# cells without an area attribute (none expected) are counted in ::T65_NOAREA and the first one is kept
set ::T65_NOAREA 0; set ::T65_NOAREA_CELL ""
proc t65_area {cells} {
    set a 0.0
    foreach_in_collection c $cells {
        set v [get_attribute -quiet $c area]
        if {$v eq ""} {
            incr ::T65_NOAREA
            if {$::T65_NOAREA_CELL eq ""} { set ::T65_NOAREA_CELL [get_object_name $c] }
            continue
        }
        set a [expr {$a + $v}]
    }
    return $a
}
# instance (not pin) name of a path start / end point
proc t65_pname {obj} {
    if {[get_attribute -quiet $obj object_class] eq "pin"} {
        return [get_object_name [get_cells -of_objects $obj]]
    }
    return [get_object_name $obj]
}
proc t65_path_line {p} {
    set n 0
    foreach_in_collection pt [get_attribute $p points] { incr n }
    return [format "%s -> %s  slack %.4f  arrival %.3f  points %d" [t65_pname [get_attribute $p startpoint]] \
        [t65_pname [get_attribute $p endpoint]] [get_attribute $p slack] [get_attribute $p arrival] $n]
}
proc t65_slack {type {to ""}} {
    if {$to eq ""} {
        set ps [get_timing_paths -delay_type $type -max_paths 100000 -nworst 1 -slack_lesser_than -0.0005]
        set p  [get_timing_paths -delay_type $type -max_paths 1]
    } else {
        set ps [get_timing_paths -delay_type $type -max_paths 100000 -nworst 1 -slack_lesser_than -0.0005 -to $to]
        set p  [get_timing_paths -delay_type $type -max_paths 1 -to $to]
    }
    set tns 0.0; set nv 0
    foreach_in_collection q $ps { set tns [expr {$tns + [get_attribute $q slack]}]; incr nv }
    return [list [get_attribute $p slack] $tns $nv $p]
}

proc t65_put_timing {fh} {
    lassign [t65_slack max] wns tns nv p
    puts $fh [format "setup_wns %.4f setup_tns %.3f setup_nviol %d" $wns $tns $nv]
    puts $fh "setup_path [t65_path_line $p]"
    # latch endpoints that borrow sit at slack 0: also the worst path into flops / outputs
    set ends [add_to_collection [all_registers -edge_triggered -data_pins] [all_outputs]]
    lassign [t65_slack max $ends] fw ft fn fp
    puts $fh [format "flop_wns %.4f flop_tns %.3f flop_nviol %d" $fw $ft $fn]
    puts $fh "flop_path [t65_path_line $fp]"
    lassign [t65_slack min] hw ht hn hp
    puts $fh [format "hold_wns %.4f hold_tns %.3f hold_nviol %d" $hw $ht $hn]
    # the five worst flop / output endpoints (distinct; borrowing latch endpoints all sit at 0), to see which
    # stages limit the clock
    set i 0
    foreach_in_collection q [get_timing_paths -delay_type max -max_paths 5 -nworst 1 -to $ends] {
        incr i; puts $fh "top_path $i [t65_path_line $q]"
    }
    set lat [all_registers -level_sensitive -data_pins]
    if {[sizeof_collection $lat] > 0} {
        puts $fh "latch_wr_path [t65_path_line [get_timing_paths -delay_type max -max_paths 1 -to $lat]]"
    }
}

proc t65_put_area {fh dut} {
    set nand2 [lindex [get_attribute [get_lib_cells */$::T65_NAND2] area] 0]
    set all  [t65_leaf *]
    set icg  [get_cells -quiet -hierarchical -filter "is_hierarchical==false && ref_name=~$::T65_ICG_PATTERN"]
    set flop [remove_from_collection [all_registers -edge_triggered] $icg]
    set lat  [remove_from_collection [all_registers -level_sensitive] $icg]
    set comb [remove_from_collection [remove_from_collection [remove_from_collection $all $flop] $lat] $icg]
    set a [t65_area $all]
    puts $fh [format "area_um2 %.1f ge %.0f cells %d" $a [expr {$a / $nand2}] [sizeof_collection $all]]
    foreach {lbl c} [list comb $comb flops $flop latches $lat icg $icg] {
        puts $fh [format "area_%s %.1f cells %d" $lbl [t65_area $c] [sizeof_collection $c]]
    }
    set d [t65_leaf $dut/*]
    puts $fh [format "dut_area_um2 %.1f cells %d (without the wrapper's input flops)" [t65_area $d] [sizeof_collection $d]]
    redirect -variable ra {report_area -nosplit}
    foreach k {{Net Interconnect} {Total cell}} {
        if {[regexp -line "^$k area:\\s+(\[0-9.\]+)" $ra -> v]} { puts $fh [format "dc_area_%s %s" [string map {" " _} $k] $v] }
    }
}

proc t65_summary {path block top params period corner topo dut t0} {
    set fh [open $path w]
    puts $fh "block $block top $top params {$params}"
    puts $fh "period_ns $period corner $corner mode [expr {$topo ? {topographical} : {wire-load}}]"
    t65_put_area $fh $dut
    t65_put_timing $fh
    set cpu -1; set mem -1
    catch {set cpu [cputime]}
    catch {set mem [mem]}
    puts $fh [format "dc_wall_s %d dc_cpu_s %s dc_mem_kb %s" [expr {[clock seconds] - $t0}] $cpu $mem]
    close $fh
}

# ---------------------------------------------------------------- P8b hierarchical units (syn/dc_unit.tcl)
# memory = cells inside mm_sram_bank instances (latch words, write / read logic, word ICGs), from the
# hierarchical area report (instance name, absolute total area, ..., design name)
proc t65_mem_area {rpt} {
    set a 0.0; set n 0
    set fh [open $rpt]
    while {[gets $fh line] >= 0} {
        set t [regexp -all -inline {\S+} $line]
        if {[llength $t] >= 4 && [string match "*mm_sram_bank*" [lindex $t end]] && [string is double -strict [lindex $t 1]]} {
            set a [expr {$a + [lindex $t 1]}]; incr n
        }
    }
    close $fh
    return [list $a $n]
}
# numbers of a finished block summary (dict: area_um2 mem_area_um2 flops latches setup_wns flop_wns hold_nviol);
# for a block with abstracted children the unit totals are used
proc t65_read_summary {f} {
    set d [dict create area_um2 0 mem_area_um2 0 flops 0 latches 0 setup_wns NA flop_wns NA hold_nviol NA]
    set fh [open $f]
    while {[gets $fh l] >= 0} {
        set t [regexp -all -inline {\S+} $l]
        switch -- [lindex $t 0] {
            area_um2      { dict set d area_um2 [lindex $t 1] }
            mem_area_um2  { dict set d mem_area_um2 [lindex $t 1] }
            area_flops    { dict set d flops [lindex $t 3] }
            area_latches  { dict set d latches [lindex $t 3] }
            setup_wns     { dict set d setup_wns [lindex $t 1] }
            flop_wns      { dict set d flop_wns [lindex $t 1] }
            hold_wns      { dict set d hold_nviol [lindex $t 5] }
            unit_area_um2 { dict set d area_um2 [lindex $t 1]; dict set d mem_area_um2 [lindex $t 3]; dict set d latches [lindex $t 5]; dict set d flops [lindex $t 7] }
        }
    }
    close $fh
    return $d
}
proc t65_unit_summary {path block design params period corner topo io kid_design rpt t0 {abs 0} {hroot ""}} {
    set fh [open $path w]
    puts $fh "block $block top $design params {$params}"
    puts $fh "period_ns $period corner $corner mode [expr {$topo ? {topographical} : {wire-load}}] io_pct $io hold_corner ff"
    set nand2 [lindex [get_attribute [get_lib_cells */$::T65_NAND2] area] 0]
    set all  [t65_leaf *]
    set icg  [get_cells -quiet -hierarchical -filter "is_hierarchical==false && ref_name=~$::T65_ICG_PATTERN"]
    set flop [remove_from_collection [all_registers -edge_triggered] $icg]
    set lat  [remove_from_collection [all_registers -level_sensitive] $icg]
    set comb [remove_from_collection [remove_from_collection [remove_from_collection $all $flop] $lat] $icg]
    set a [t65_area $all]
    puts $fh [format "area_um2 %.1f ge %.0f cells %d" $a [expr {$a / $nand2}] [sizeof_collection $all]]
    foreach {lbl c} [list comb $comb flops $flop latches $lat icg $icg] {
        puts $fh [format "area_%s %.1f cells %d" $lbl [t65_area $c] [sizeof_collection $c]]
    }
    lassign [t65_mem_area $rpt] ma mn
    puts $fh [format "mem_area_um2 %.1f banks %d" $ma $mn]
    puts $fh "noarea_cells $::T65_NOAREA $::T65_NOAREA_CELL"
    if {$abs} {
        # children linked as block abstractions: the leaf cells above hold only their interface logic, so the
        # unit totals are this block's own cells plus instances x each child's own summary
        set own $all
        dict for {k kn} $kid_design {
            foreach_in_collection ci [get_cells -quiet -hierarchical -filter ref_name==$kn] {
                set own [remove_from_collection $own [t65_leaf [get_object_name $ci]/*]]
            }
        }
        set ta [t65_area $own]; set tm 0.0; set tl 0
        set tf [sizeof_collection [remove_from_collection $flop [remove_from_collection $all $own]]]
        puts $fh [format "own_area_um2 %.1f cells %d (outside the abstracted children)" $ta [sizeof_collection $own]]
        dict for {k kn} $kid_design {
            set n [sizeof_collection [get_cells -quiet -hierarchical -filter ref_name==$kn]]
            set cs [t65_read_summary $hroot/$k/summary.txt]
            set ta [expr {$ta + $n * [dict get $cs area_um2]}]
            set tm [expr {$tm + $n * [dict get $cs mem_area_um2]}]
            set tl [expr {$tl + $n * [dict get $cs latches]}]
            set tf [expr {$tf + $n * [dict get $cs flops]}]
            puts $fh [format "child_summary %s x%d area %.1f mem %.1f flops %d latches %d setup_wns %s flop_wns %s hold_nviol %s" \
                $k $n [dict get $cs area_um2] [dict get $cs mem_area_um2] [dict get $cs flops] [dict get $cs latches] \
                [dict get $cs setup_wns] [dict get $cs flop_wns] [dict get $cs hold_nviol]]
        }
        puts $fh [format "unit_area_um2 %.1f unit_mem_area_um2 %.1f unit_latches %d unit_flops %d (own + children)" $ta $tm $tl $tf]
    }
    redirect -variable ra {report_area -nosplit}
    foreach k {{Net Interconnect} {Total cell}} {
        if {[regexp -line "^$k area:\\s+(\[0-9.\]+)" $ra -> v]} { puts $fh [format "dc_area_%s %s" [string map {" " _} $k] $v] }
    }
    dict for {k kn} $kid_design {
        puts $fh "child $k $kn [sizeof_collection [get_cells -quiet -hierarchical -filter ref_name==$kn]]"
    }
    t65_put_timing $fh
    set cpu -1; set mem -1
    catch {set cpu [cputime]}
    catch {set mem [mem]}
    puts $fh [format "dc_wall_s %d dc_cpu_s %s dc_mem_kb %s" [expr {[clock seconds] - $t0}] $cpu $mem]
    close $fh
}
