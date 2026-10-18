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
