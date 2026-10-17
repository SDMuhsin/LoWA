# Site config template for flow/tsmc65 (copy to site.tcl, git-ignored, and fill in every <PLACEHOLDER>).
# Sourced by common/site.tcl (DC / PT / Formality) and sdc/gen_sdc.tcl (plain tclsh).
# Corners: ss = setup signoff (slow, hot), tt = typical / power, ff = hold (fast, cold).
set ::T65_KIT $::env(T65_KIT)

# ---- standard-cell timing libraries (.db)
set ::T65_LIBDIR <STDCELL_DB_DIR>
array set ::T65_CORNER_LIB {ss <STDCELL_LIB_SS> tt <STDCELL_LIB_TT> ff <STDCELL_LIB_FF>}
array set ::T65_CORNER_OC  {ss <OPCOND_SS> tt <OPCOND_TT> ff <OPCOND_FF>}
array set ::T65_CORNER_PVT {ss {<V> <T>} tt {<V> <T>} ff {<V> <T>}}

# ---- physical data for DC topographical mode
set ::T65_TF      <TECHFILE>
set ::T65_MW_REF  <MILKYWAY_REFERENCE_LIB>
set ::T65_TLU     <RC_TECH_DIR>
set ::T65_TLU_MAP <RC_TECH_LAYER_MAP>
array set ::T65_CORNER_TLU {ss <RC_FILE_WORST> tt <RC_FILE_TYPICAL> ff <RC_FILE_BEST>}
# RC model setup command of your kit's format (max corner file, min corner file); DC only
proc t65_rc_setup {max min} {
    <RC_SETUP_CMD> <MAX_OPT> $::T65_TLU/$::T65_CORNER_TLU($max) <MIN_OPT> $::T65_TLU/$::T65_CORNER_TLU($min) <MAP_OPT> $::T65_TLU_MAP
}

# ---- reference cells
set ::T65_NAND2       <NAND2_X1_CELL>
set ::T65_ICG_PATTERN <ICG_CELL_GLOB>
set ::T65_DRV_IN      {<INPUT_DRIVER_CELL> <OUTPUT_PIN>}
set ::T65_DRV_CLK     {<CLOCK_DRIVER_CELL> <OUTPUT_PIN>}
set ::T65_LOAD_PIN    <LOAD_CELL>/<INPUT_PIN>
set ::T65_DONT_USE    {}                ;# lib-cell globs DC must not use (optional)

# ---- site RTL: mm_icg mapped to the library ICG (replaces hw/mem/tech/mm_icg.v); "" = behavioural model
set ::T65_ICG_RTL [file dirname [info script]]/mm_icg_site.v
