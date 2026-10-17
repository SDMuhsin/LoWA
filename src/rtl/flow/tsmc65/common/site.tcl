# Site config loader + library helpers for flow/tsmc65 (DC / PT / Formality / tclsh).
# Every kit-specific name and path comes from the git-ignored site/site.tcl (template site.example.tcl).
set ::T65_FLOW [file normalize [file dirname [info script]]/..]
source $::T65_FLOW/site/site.tcl
proc t65_db  {corner} { return $::T65_LIBDIR/$::T65_CORNER_LIB($corner).db }
proc t65_lib {corner} { return $::T65_CORNER_LIB($corner) }
proc t65_oc  {corner} { return $::T65_CORNER_OC($corner) }
