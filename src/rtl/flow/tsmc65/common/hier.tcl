# Reader of common/hier.txt (P8b block table) for DC / PT.
proc t65_hier_load {} {
    set fh [open $::T65_FLOW/common/hier.txt]
    while {[gets $fh line] >= 0} {
        set line [string trim $line]
        if {$line eq "" || [string index $line 0] in {# @}} continue
        lassign $line name mod params kids io link
        set ::T65_HIER($name) [dict create module $mod params [expr {$params eq "-" ? "" : $params}] \
            children [expr {$kids eq "-" ? {} : [split $kids ,]}] io $io \
            link [expr {$link eq "abs" ? "abs" : "full"}] copy [expr {$link eq "copy"}]]
    }
    close $fh
}
proc t65_hier {name key} {
    if {![info exists ::T65_HIER($name)]} { error "unknown block $name (common/hier.txt)" }
    return [dict get $::T65_HIER($name) $key]
}
# Output directory of a child block as its parent links it: build/tsmc65/hier/<child><tag>.  The tag is empty
# (the reference run) unless T65_KID_TAG names the child, e.g. T65_KID_TAG="gm16=_p5.0,vu16=_p5.0" (P8d: the
# chip links the gm16 / vu16 runs made at the chip clock, syn/run_hier.sh T65_TAG).
proc t65_kid_dir {hroot k} {
    set tag ""
    if {[info exists ::env(T65_KID_TAG)]} {
        foreach kv [split [string map {, " "} $::env(T65_KID_TAG)]] {
            if {[regexp {^([^=]+)=(.*)$} $kv -> n t] && $n eq $k} { set tag $t }
        }
    }
    return $hroot/$k$tag
}
