# P9b: PrimePower (averaged) of the complete chip netlist with gate-level SAIF windows.  The design is loaded once
# exactly as the flat-chip run of sta/pt_unit.tcl (all block netlists, each block's DC topographical SPEF on every
# instance, the DC SDC regenerated at the corner, ideal clocks); then, per SAIF file, the switching activity is
# reset, the SAIF read (strip path tb_top_gls/u_chip) and the power reported.
# Environment (sta/run_pt_gls.sh): T65_ROOT, T65_PERIOD, T65_CORNER, T65_PT_OUT, T65_KID_TAG, T65_HIER (netlist root),
#   T65_GLS_LIST (window list file), T65_GLS_WAIT_H, T65_GLS_STRIP (default tb_top_gls/u_chip), T65_CORES.
set t0 [clock seconds]
set flow $::env(T65_ROOT)/flow/tsmc65
source $flow/common/site.tcl
source $flow/common/hier.tcl
t65_hier_load
proc envd {k d} { expr {[info exists ::env($k)] ? $::env($k) : $d} }
set top chip
set period $::env(T65_PERIOD)
set corner $::env(T65_CORNER)
set out    $::env(T65_PT_OUT)
set hroot  [envd T65_HIER $::env(T65_ROOT)/build/tsmc65/hier]
file mkdir $out
set_host_options -max_cores [envd T65_CORES 8]
set_app_var power_enable_analysis true
set_app_var power_analysis_mode averaged
proc t65_tree {b} {
    set l [list $b]
    if {[t65_hier $b link] eq "abs"} { foreach k [t65_hier $b children] { set l [concat $l [t65_tree $k]] } }
    return $l
}
set blocks [lsort -unique [t65_tree $top]]
set files {}; set bdesign [dict create]; set bdir [dict create]
foreach b $blocks {
    set d [t65_kid_dir $hroot $b]
    set fh [open $d/design.txt]; dict set bdesign $b [string trim [read $fh]]; close $fh
    dict set bdir $b $d
    lappend files $d/$b.v
}
set_app_var search_path [concat . $::T65_LIBDIR]
set_app_var link_path [concat * [t65_db $corner]]
read_verilog $files
current_design [dict get $bdesign $top]
link_design
set ::argv [list $period $corner $out/pt_$corner.sdc [t65_hier $top io]]
source $flow/sdc/gen_sdc.tcl
source $out/pt_$corner.sdc
set npar 0; set nparf 0
foreach b $blocks {
    set f [dict get $bdir $b]/$b.spef
    if {$b eq $top || ![file exists $f]} continue
    foreach_in_collection c [get_cells -quiet -hierarchical -filter "ref_name==[dict get $bdesign $b]"] {
        if {[catch {read_parasitics -format spef -path [get_object_name $c] $f} e]} { incr nparf } else { incr npar }
    }
}
update_timing
set fh [open $out/summary_gls_$corner.txt w]
puts $fh "block chip corner $corner period_ns $period netlists [llength $files] ([join $blocks ,]) clocks ideal (pre-CTS)"
puts $fh "parasitics dc_topo_ss_estimate spef_reads $npar failed $nparf"
puts $fh "leaf_cells [sizeof_collection [get_cells -quiet -hierarchical -filter is_hierarchical==false]] load_s [expr {[clock seconds] - $t0}]"
flush $fh
set strip [envd T65_GLS_STRIP tb_top_gls/u_chip]
# windows: the list file T65_GLS_LIST ("name saif_file" lines; a line "END" closes the list) is re-read every minute,
# so windows can be added while this session runs; each window is processed once its SAIF is complete (<file>.ok,
# written by the testbench after the SAIF).  The session ends when the list is closed and done, or after
# T65_GLS_WAIT_H hours.
set done {}
set tmax [expr {[clock seconds] + int(3600 * [envd T65_GLS_WAIT_H 0])}]
while {1} {
    set closed 0; set pend {}; set pick ""
    set lf [open $::env(T65_GLS_LIST)]
    foreach line [split [read $lf] "\n"] {
        set line [string trim $line]
        if {$line eq "END"} { set closed 1; continue }
        if {[llength $line] != 2 || [lsearch -exact $done [lindex $line 0]] >= 0} continue
        lappend pend [lindex $line 0]
        if {$pick eq "" && [file exists [lindex $line 1].ok]} { set pick $line }
    }
    close $lf
    if {$pick eq ""} {
        if {$closed && [llength $pend] == 0} break
        if {[clock seconds] > $tmax} { puts $fh "windows_not_ready [join $pend { }]"; break }
        after 60000; continue
    }
    lassign $pick name f
    lappend done $name
    set t1 [clock seconds]
    reset_switching_activity
    set rc [catch {redirect $out/rpt_saif_$name.rpt { read_saif -strip_path $strip $f }} e]
    puts $fh "window $name saif [file tail $f] read_rc $rc"
    redirect $out/rpt_switching_$name.rpt { report_switching_activity }
    update_power
    redirect $out/rpt_power_$name.rpt { report_power -nosplit }
    redirect $out/rpt_power_hier_$name.rpt { report_power -nosplit -hierarchy -levels 2 }
    set r [open $out/rpt_power_$name.rpt]
    foreach line [split [read $r] "\n"] {
        if {[regexp {^\s*(Net Switching Power|Cell Internal Power|Cell Leakage Power|Total Power|clock_network|register|combinational|sequential|memory|black_box|io_pad)\s} $line]} { puts $fh "power $name [string trim $line]" }
    }
    close $r
    set r [open $out/rpt_power_hier_$name.rpt]
    foreach line [split [read $r] "\n"] { if {[regexp {^\s*(u_\S+)\s+\(} $line]} { puts $fh "power_hier $name [regsub -all {\s+} [string trim $line] { }]" } }
    close $r
    set r [open $out/rpt_switching_$name.rpt]
    foreach line [split [read $r] "\n"] { if {[regexp {^\s*(Nets|Pins|Registers|Combinational|Sequential|Clock|Primary|Black|Total|Memory|Latch|Flip)} $line]} { puts $fh "switching $name [regsub -all {\s+} [string trim $line] { }]" } }
    close $r
    puts $fh "window_s $name [expr {[clock seconds] - $t1}]"
    flush $fh
}
puts $fh [format "pt_wall_s %d pt_mem_mb %s" [expr {[clock seconds] - $t0}] [mem]]
close $fh
exit
