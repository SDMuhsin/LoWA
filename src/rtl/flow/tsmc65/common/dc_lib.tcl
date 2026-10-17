# DC library / physical setup shared by syn/dc_block.tcl (P8a scout) and syn/dc_unit.tcl (P8b hierarchical units).
#   t65_dc_setup <corner> <topo> <out>   target / link / min libraries; in topographical mode a fresh Milkyway
#                                        library under <out> with the corner's max RC and the ff (best) min RC
proc t65_dc_setup {corner topo out} {
    set_app_var search_path [concat . $::T65_LIBDIR [get_app_var search_path]]
    set_app_var target_library [list [t65_db $corner]]
    set_app_var synthetic_library dw_foundation.sldb
    set_app_var link_library [list * [t65_db $corner] dw_foundation.sldb]
    if {$corner ne "ff"} { set_min_library [t65_db $corner] -min_version [t65_db ff] }
    if {[llength $::T65_DONT_USE]} {
        set_dont_use [get_lib_cells [lmap c $::T65_DONT_USE {string cat [t65_lib $corner]/$c}]]
    }
    if {$topo} {
        set mw $out/t65_mw
        file delete -force $mw
        create_mw_lib -technology $::T65_TF -mw_reference_library $::T65_MW_REF $mw
        open_mw_lib $mw
        t65_rc_setup $corner ff
    }
}
