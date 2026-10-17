# RTL file list of the LoWA-TA design for the TSMC-65 flow: every hw/ file plus the block's wrapper.
# The behavioural clock gate hw/mem/tech/mm_icg.v is replaced by the site wrapper (::T65_ICG_RTL)
# that maps mm_icg to the library ICG cell; the memory backend is selected with MM_TECH=2 (t65_defines).
proc t65_hw_files {} {
    set root $::env(T65_ROOT)
    set files [lsort [split [string trim [exec find $root/hw -name *.v]] "\n"]]
    set out {}
    foreach f $files {
        if {[file tail $f] eq "mm_icg.v" && [info exists ::T65_ICG_RTL] && $::T65_ICG_RTL ne ""} {
            lappend out $::T65_ICG_RTL; continue
        }
        lappend out $f
    }
    return $out
}
proc t65_defines {} { return {SYNTHESIS MM_TECH=2} }
