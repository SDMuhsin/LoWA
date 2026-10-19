# P8e (PLAN R16): overrides that turn the P8b/P8d DC / PT flow into the SRAM-macro variant, without editing it.
# ::T65_SRAM_LATCH = 1 (latch-RF baseline run): only the summary procs below are defined.
# Sourced after common/{site,rtl,hier,summary,dc_lib}.tcl by sram/dc_sram.tcl (and sram/pt_sram.tcl):
#   - memory backend MM_TECH=3 (tsmc65_sram, hw/mem/tech/mm_bank_t65s.v);
#   - the stand-in hw/mem/tech/mm_sram_macro.v replaced by the site wrapper site/mm_sram_macro_site.v;
#   - the macros' compiled .db files (sram/mc_gen.sh) added to the link library of each corner, macros are black
#     boxes for synthesis (timing, area and power from their .db, no netlist).
# Environment (sram/run_sram.sh, from site/sram_site.sh): T65_MC_OUT (macro views), T65_MC_CORNERS (ss, tt, ff
# cold, ff hot characterisation corner names), T65_MC_SHAPES (words:bits:compiler:mux:instance ...).
set ::T65_SRAM_RTL $::T65_FLOW/site/mm_sram_macro_site.v
array set ::T65_MC_CORNER [list ss [lindex [split $::env(T65_MC_CORNERS) ,] 0] \
                                tt [lindex [split $::env(T65_MC_CORNERS) ,] 1] \
                                ff [lindex [split $::env(T65_MC_CORNERS) ,] 2]]
set ::T65_MC_CELLS {}
foreach s $::env(T65_MC_SHAPES) { lappend ::T65_MC_CELLS [lindex [split $s :] 4] }
proc t65_mc_dbs {corner} {
    set l {}
    foreach c $::T65_MC_CELLS { lappend l $::env(T65_MC_OUT)/$c/${c}_$::T65_MC_CORNER($corner)_syn.db }
    return $l
}
if {!$::T65_SRAM_LATCH} {
    proc t65_defines {} { return {SYNTHESIS MM_TECH=3} }
    if {[info commands t65_hw_files] ne ""} {          ;# DC only (PT reads netlists)
        rename t65_hw_files t65_hw_files_latch
        proc t65_hw_files {} {
            set out {}
            foreach f [t65_hw_files_latch] {
                if {[file tail $f] eq "mm_sram_macro.v"} { lappend out $::T65_SRAM_RTL } else { lappend out $f }
            }
            return $out
        }
    }
    if {[info commands t65_dc_setup] ne ""} {
        rename t65_dc_setup t65_dc_setup_latch
        proc t65_dc_setup {corner topo out} {
            t65_dc_setup_latch $corner $topo $out
            set_app_var link_library [concat [get_app_var link_library] [t65_mc_dbs $corner]]
            if {$corner ne "ff"} {
                foreach m [t65_mc_dbs $corner] f [t65_mc_dbs ff] { set_min_library $m -min_version $f }
            }
            puts "T65_SRAM link [llength [t65_mc_dbs $corner]] macro libraries ($corner)"
        }
    }
}
# macro instances of the current design (leaf cells whose reference is one of the generated macros)
proc t65_mc_cells {} {
    set f [join [lmap c $::T65_MC_CELLS {string cat "ref_name==$c"}] " || "]
    return [get_cells -quiet -hierarchical -filter "is_hierarchical==false && ($f)"]
}
# summary lines of the SRAM variant (numbers only): macro count / area, logic area, macro access time vs period
proc t65_sram_summary {path period} {
    set fh [open $path a]
    set m [t65_mc_cells]
    set ma 0.0
    foreach_in_collection c $m { set ma [expr {$ma + [get_attribute $c area]}] }
    redirect -variable ra {report_area -nosplit}
    set tot 0.0; regexp -line {^Total cell area:\s+([0-9.]+)} $ra -> tot
    puts $fh [format "sram_macros %d macro_area_um2 %.1f logic_area_um2 %.1f total_cell_area_um2 %.1f" \
        [sizeof_collection $m] $ma [expr {$tot - $ma}] $tot]
    foreach r [lsort -unique [get_attribute -quiet $m ref_name]] {
        set n [sizeof_collection [filter_collection $m "ref_name==$r"]]
        puts $fh "sram_ref [lsearch -exact $::T65_MC_CELLS $r] count $n"
    }
    # worst macro clock -> Q arc (read access, all margin settings, as DC times it) and worst path through a macro output
    set wq NA; set ws NA
    if {[sizeof_collection $m]} {
        set p [get_timing_paths -from [get_pins -quiet -of_objects $m -filter "direction==out"] -max_paths 1]
        if {[sizeof_collection $p]} { set ws [get_attribute $p slack] }
        set p2 [get_timing_paths -to [get_pins -quiet -of_objects $m -filter "direction==in"] -max_paths 1]
        if {[sizeof_collection $p2]} { set wq [get_attribute $p2 slack] }
    }
    puts $fh "sram_out_path_slack $ws sram_in_path_slack $wq period_ns $period"
    close $fh
}
