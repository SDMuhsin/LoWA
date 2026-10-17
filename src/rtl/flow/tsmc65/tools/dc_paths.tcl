# Worst setup path into each endpoint group of a finished P8b block (reads its .ddc; numbers + RTL names only).
# Environment: T65_ROOT, T65_BLOCK, T65_PERIOD, T65_OUT (build/tsmc65/hier/<block>), T65_PATHS (glob patterns of
# endpoint register pins matched on the full pin name, space separated).  Writes $T65_OUT/paths.txt: per pattern the worst path and its delay
# split over the top-level instances it crosses.  Run by tools/dc_paths.sh.
set flow $::env(T65_ROOT)/flow/tsmc65
source $flow/common/site.tcl
source $flow/common/hier.tcl
source $flow/common/summary.tcl
source $flow/common/dc_lib.tcl
t65_hier_load
set block $::env(T65_BLOCK); set out $::env(T65_OUT); set hroot [file dirname $out]
set_host_options -max_cores 8
t65_dc_setup ss 1 $out/paths_mw
if {[t65_hier $block link] eq "abs"} {
    set refs {}
    foreach k [t65_hier $block children] { set fh [open $hroot/$k/design.txt]; lappend refs [string trim [read $fh]]; close $fh }
    set_top_implementation_options -block_references $refs
}
read_ddc $out/$block.ddc
set fh [open $out/design.txt]; current_design [string trim [read $fh]]; close $fh
link
set ::argv [list $::env(T65_PERIOD) ss $out/paths.sdc [t65_hier $block io]]
source $flow/sdc/gen_sdc.tcl
set_operating_conditions -max [t65_oc ss] -max_library [t65_lib ss] -min [t65_oc ff] -min_library [t65_lib ff]
source $out/paths.sdc
set fo [open $out/paths.txt w]
foreach pat $::env(T65_PATHS) {
    set ends [get_pins -quiet -hierarchical -filter "full_name=~$pat"]
    if {[sizeof_collection $ends] == 0} { puts $fo "group $pat none"; continue }
    set p [get_timing_paths -delay_type max -max_paths 1 -to $ends]
    puts $fo "group $pat [t65_path_line $p]"
    set d [dict create]; set prev 0.0
    foreach_in_collection pt [get_attribute $p points] {
        set a [get_attribute $pt arrival]; set o [get_object_name [get_attribute $pt object]]
        set k [lindex [split $o /] 0]; if {![string match u_* $k]} { set k top }
        dict incr d $k 0; dict set d $k [expr {[dict get $d $k] + $a - $prev}]; set prev $a
    }
    set s ""; dict for {k v} $d { append s [format " %s %.2f" $k $v] }
    puts $fo "  split$s"
}
close $fo
exit
