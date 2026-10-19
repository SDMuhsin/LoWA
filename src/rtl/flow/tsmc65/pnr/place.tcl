# Stage 2: timing-driven placement and pre-CTS optimisation (ideal clocks).  All DC clock gates are kept.
source $::env(T65_ROOT)/flow/tsmc65/pnr/pnr_common.tcl
t65p_open_stage floorplan
set_app_options -name place.coarse.continue_on_missing_scandef -value true
set_app_options -name opt.tie_cell.max_fanout -value 8
set_app_options -name place_opt.flow.merge_clock_gates -value false
place_opt
t65p_log "place_opt done"
redirect [t65p_rpt place_congestion.rpt] { report_congestion }
t65p_save_stage place
t65p_stage_reports place
puts "STAGE_OK place"
exit
