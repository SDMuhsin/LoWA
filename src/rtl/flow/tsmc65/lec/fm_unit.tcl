# Formality: RTL (reference) vs the DC netlist (implementation) of one block of the hierarchical flow, with the
# block's DC guidance file (.svf).  Environment (lec/run_fm.sh):
#   T65_ROOT  src/rtl of the tree     T65_BLOCK block (common/hier.txt)   T65_OUT the block's DC output dir
#   T65_CORNER library corner of the DC run (ss)                          T65_FM_OUT this run's output dir
# Hierarchical verification: a block whose children are linked as block abstractions (`abs`) is verified with
# those children as black boxes on both sides (hdlin_interface_only: the child RTL modules in the reference, the
# children's netlist designs in the implementation).  Every child is verified in its own run of this script,
# so each cell of the chip is verified exactly once.  Children linked as full netlists (lane's banks) are
# verified inside their parent.
set t0 [clock seconds]
set flow $::env(T65_ROOT)/flow/tsmc65
source $flow/common/site.tcl
source $flow/common/rtl.tcl
source $flow/common/hier.tcl
t65_hier_load
set block  $::env(T65_BLOCK)
set out    $::env(T65_OUT)
set fo     $::env(T65_FM_OUT)
set corner [expr {[info exists ::env(T65_CORNER)] ? $::env(T65_CORNER) : "ss"}]
set mod    [t65_hier $block module]
set params [t65_hier $block params]
set kids   [t65_hier $block children]
set abs    [expr {[t65_hier $block link] eq "abs" && [llength $kids]}]
set fh [open $out/design.txt]; set design [string trim [read $fh]]; close $fh
file mkdir $fo/rpt
set_host_options -max_cores [expr {[info exists ::env(T65_CORES)] ? $::env(T65_CORES) : 8}]
source $flow/lec/fm_setup.tcl
# guidance: the block's own .svf, plus each child verified in this run (T65_FM_FULL_KIDS), whose register merges
# and renames are recorded only in the child's own .svf
set svfs [list $out/$block.svf]
foreach k [split [string map {, " "} [expr {[info exists ::env(T65_FM_FULL_KIDS)] ? $::env(T65_FM_FULL_KIDS) : ""}]]] {
    if {$k ne ""} { lappend svfs [t65_kid_dir [file dirname $out] $k]/$k.svf }
}
set_svf {*}$svfs

read_db -technology_library [t65_db $corner]
# ---- reference: the RTL DC read (same files, defines and site ICG wrapper)
# T65_FM_FULL_KIDS (comma list): children verified inside this run instead of as black boxes (their netlists are
# read in full).  Used for mm: rn_mkgen NCORE 9 has a constant mk_req_ready output that DC propagates out of the
# child, which a black box cannot represent (a false failure of the black-box method, not a design difference).
set fullk [split [string map {, " "} [expr {[info exists ::env(T65_FM_FULL_KIDS)] ? $::env(T65_FM_FULL_KIDS) : ""}]]]
set bbm {}
if {$abs} { foreach k $kids { if {$k ni $fullk} { lappend bbm [t65_hier $k module] } } }
set bbm [lsort -unique $bbm]
set_app_var hdlin_interface_only [join $bbm " "]
set defs [join [lmap d [t65_defines] {string cat "+define+" $d}] ""]
read_verilog -container r -libname WORK -05 -vcs $defs [t65_hw_files]
if {$params eq ""} { set_top r:/WORK/$mod } else { set_top r:/WORK/$mod -parameter [string map {=> =} $params] }
set_app_var hdlin_interface_only ""
# ---- implementation: the DC netlist; abstracted children are not defined in it: their own netlists are read
# for the interface only (pin directions), as black boxes
set ifiles [list $out/$block.v]; set bbi {}
if {$abs} {
    set hroot [file dirname $out]
    foreach k $kids {
        set kd [t65_kid_dir $hroot $k]
        lappend ifiles $kd/$k.v
        if {$k in $fullk} continue
        set fh [open $kd/design.txt]; lappend bbi [string trim [read $fh]]; close $fh
    }
}
set_app_var hdlin_interface_only [join [lsort -unique $bbi] " "]
read_verilog -container i -libname WORK -netlist $ifiles
set_top i:/WORK/$design
set_app_var hdlin_interface_only ""

match
report_unmatched_points                  > $fo/rpt/unmatched.rpt
report_black_boxes                       > $fo/rpt/black_boxes.rpt
# T65_FM_HIER=<level>: Formality's own hierarchical verification (write_hierarchical_verification_script): each
# sub-block at or above that level is verified separately (with the boundary constraints Formality derives and
# then checks at the parent), then the top with those blocks black-boxed.  Per-block results go to
# rpt/hier_results.txt; the final verify below is the top-level one.
if {[info exists ::env(T65_FM_HIER)]} {
    write_hierarchical_verification_script -replace -level $::env(T65_FM_HIER) $fo/hier_script.tcl
    redirect -tee $fo/rpt/hier_run.log { source $fo/hier_script.tcl }
    set ok [verify]
} else {
set ok [verify]
}
report_status                            > $fo/rpt/status.rpt
report_failing_points                    > $fo/rpt/failing.rpt
report_aborted_points                    > $fo/rpt/aborted.rpt
report_unverified_points                 > $fo/rpt/unverified.rpt
report_guidance -summary                 > $fo/rpt/guidance.rpt
if {!$ok} { catch { analyze_points -failing > $fo/rpt/analyze.rpt } }

# ---- summary (numbers and RTL point types only)
set fs [open $fo/summary.txt w]
puts $fs "block $block ref $mod impl $design corner $corner verify_return $ok black_box_children [join $bbm ,] full_children [join $fullk ,]"
set r [open $fo/rpt/status.rpt]
foreach line [split [read $r] "\n"] {
    if {[regexp {(Verification|Passing|Failing|Aborted|Unverified|Not Compared|Unread|Unmatched|Matched|Compare Points|BBPin|DFF|LAT|Port|Cut|Total)} $line]} {
        puts $fs "  [string trim $line]"
    }
}
close $r
set r [open $fo/rpt/unmatched.rpt]; set types {}
foreach line [split [read $r] "\n"] {
    if {[regexp {^\s*[0-9]+ Unmatched} $line]} { puts $fs "  unmatched: [string trim $line]" }
    if {[regexp {^\s+(Ref|Impl)\s+(\S+)\s} $line -> side ty]} { dict incr types "$side $ty" }
}
close $r
dict for {k n} $types { puts $fs "  unmatched_type $k: $n" }
foreach rp {aborted unverified failing} {
    set r [open $fo/rpt/$rp.rpt]; set n 0
    foreach line [split [read $r] "\n"] { if {[regexp {^\s+(Ref|Impl)\s+(DFF|LAT|Port|BBPin|Cut|BBNet|Loop)\S*\s} $line]} { incr n } }
    close $r
    puts $fs "  ${rp}_points_listed $n"
}
set r [open $fo/rpt/guidance.rpt]
foreach line [split [read $r] "\n"] { if {[regexp {^Total\s} $line]} { puts $fs "  guidance (accepted rejected unsupported unprocessed total) $line" } }
close $r
puts $fs "fm_wall_s [expr {[clock seconds] - $t0}]"
close $fs
exit
