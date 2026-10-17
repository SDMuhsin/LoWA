# Block table of the TSMC-65 flow: name -> top module (an input-register wrapper, so every path is register to
# register as in the GF180 flow), elaboration parameters, wrapper file (relative to src/rtl) and the instance
# of the design under test inside the wrapper.  P8a scout blocks; P8b adds the remaining units.
array set ::T65_BLOCK {
    fma   {top wrap_fp_fma       params {STAGES=>2}                     wrap flow/synth/wrap_fp_fma.v    dut u_dut}
    pe    {top wrap_gm_pe        params {FMA_STAGES=>2}                 wrap flow/synth/wrap_gm_pe.v     dut u_dut}
    lane  {top wrap_vu_lane      params {}                              wrap flow/synth/wrap_vu_lane.v   dut u}
    sfu   {top wrap_fp_sfu       params {}                              wrap flow/synth/wrap_fp_sfu.v    dut u_dut}
    gm4   {top wrap_gm_top       params {ROWS=>4,COLS=>4,NSFU=>2}       wrap flow/synth/wrap_gm_top.v    dut u_dut}
    gm16  {top wrap_gm_top       params {}                              wrap flow/synth/wrap_gm_top.v    dut u_dut}
    bank768 {top wrap65_mm_sram_bank params {WORDS=>768,WIDTH=>32,AW=>10} wrap flow/tsmc65/wrap/wrap65_mm_sram_bank.v dut u_dut}
    bank256 {top wrap65_mm_sram_bank params {WORDS=>256,WIDTH=>32,AW=>8}  wrap flow/tsmc65/wrap/wrap65_mm_sram_bank.v dut u_dut}
}
proc t65_block {name key} {
    if {![info exists ::T65_BLOCK($name)]} { error "unknown block $name (common/blocks.tcl)" }
    return [dict get $::T65_BLOCK($name) $key]
}
