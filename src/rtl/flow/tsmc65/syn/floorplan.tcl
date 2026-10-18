# DC topographical floorplan of one block (P8c2, PLAN R14b): die / core from the measured area at a target
# utilisation, ports on one side, no hand placement.  Parameters come from common/floorplan.txt (one row per block);
# T65_FP_AREA / T65_FP_UTIL / T65_FP_ASPECT / T65_FP_SIDE / T65_FP_OFFSET override a column (e.g. for a block that
# has no row, or a what-if run).  Sourced by syn/dc_unit.tcl when T65_FP=1, after the constraints and before
# compile_ultra.
#   t65_fp_params <block>        -> dict area util aspect side offset (error if neither the table nor T65_FP_AREA)
#   t65_floorplan <block> <out>  creates the die / core, places the ports' side, writes <out>/rpt/floorplan.rpt and
#                                returns a one-line summary (core width / height, die, ports)
proc t65_fp_params {block} {
    set p [dict create area "" util 0.65 aspect 1.0 side b offset 12]
    set fh [open $::T65_FLOW/common/floorplan.txt]
    while {[gets $fh line] >= 0} {
        set line [string trim $line]
        if {$line eq "" || [string index $line 0] eq "#"} continue
        lassign $line b area util aspect side offset
        if {$b eq $block} { set p [dict create area $area util $util aspect $aspect side $side offset $offset] }
    }
    close $fh
    foreach {k e} {area T65_FP_AREA util T65_FP_UTIL aspect T65_FP_ASPECT side T65_FP_SIDE offset T65_FP_OFFSET} {
        if {[info exists ::env($e)] && $::env($e) ne ""} { dict set p $k $::env($e) }
    }
    if {[dict get $p area] eq ""} { error "no floorplan row for $block (common/floorplan.txt) and no T65_FP_AREA" }
    return $p
}
proc t65_floorplan {block out} {
    set p [t65_fp_params $block]
    dict with p {}
    set core [expr {double($area) / $util}]
    set cw [expr {sqrt($core / $aspect)}]
    set ch [expr {$cw * $aspect}]
    # round the core up to a 1 um grid (DC snaps rows and tracks to the library's site inside it)
    set cw [expr {ceil($cw)}]; set ch [expr {ceil($ch)}]
    set dw [expr {$cw + 2 * $offset}]; set dh [expr {$ch + 2 * $offset}]
    create_die_area -coordinate [list 0 0 $dw $dh]
    create_core_area -coordinate [list [list $offset $offset] [list [expr {$offset + $cw}] [expr {$offset + $ch}]]]
    set ports [get_ports *]
    set_port_side -side $side $ports
    redirect $out/rpt/floorplan.rpt { report_physical_constraints -no_site_row }
    set s [format "floorplan core %.0f x %.0f um (%.3f mm2, util %.2f of area %.0f um2, aspect %.2f), die %.0f x %.0f um, %d ports on side %s" \
        $cw $ch [expr {$cw * $ch / 1e6}] $util $area $aspect $dw $dh [sizeof_collection $ports] $side]
    puts "T65_FP $s"
    return $s
}
