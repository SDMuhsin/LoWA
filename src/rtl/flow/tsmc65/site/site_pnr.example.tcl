# P&R site config template for flow/tsmc65/pnr (copy to site_pnr.tcl, not tracked, and fill in every <PLACEHOLDER>).
# Sourced by pnr/pnr_common.tcl after site.tcl (which supplies the techfile, RC files and corner libraries).
set ::T65_ICC2_BIN   <ICC2_BIN_DIR>                  ;# directory of icc2_shell / icc2_lm_shell
set ::T65_LEF        <STD_CELL_LEF_FOR_THE_METAL_STACK>
set ::T65_SITE       <CORE_SITE_NAME>
set ::T65_PG         {<POWER_NET> <GROUND_NET>}
set ::T65_TIE        {<TIE_HIGH_CELL> <TIE_LOW_CELL>}
set ::T65_PNR_DONT_USE {<LIB_CELL_GLOB> ...}        ;# cells the optimiser must not add
set ::T65_CTS_CELLS  {<CLOCK_BUFFER_GLOB> <CLOCK_INVERTER_GLOB>}
set ::T65_LAYER_MIN  <LOWEST_SIGNAL_LAYER>
set ::T65_LAYER_MAX  <HIGHEST_SIGNAL_LAYER>
set ::T65_PG_LAYERS  {<RAIL_LAYER> <MESH_VERTICAL_LAYER> <MESH_HORIZONTAL_LAYER>}
set ::T65_PIN_LAYERS {<PIN_LAYER_1> <PIN_LAYER_2>}
