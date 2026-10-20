# Stage 4: global routing of the signal nets (clock nets are routed by clock_opt) for parasitics; no track / detail
# routing, no optimisation.  Writes the netlist (with the clock tree and the optimisation buffers), the parasitics of
# each corner and the clock-tree summary for PrimeTime / PrimePower (sta/run_pt_unit.sh with T65_POST_*).
source $::env(T65_ROOT)/flow/tsmc65/pnr/pnr_common.tcl
t65p_open_stage cts
route_global
t65p_log "route_global done"
redirect [t65p_rpt groute_congestion.rpt] { report_congestion }
t65p_save_stage groute
t65p_stage_reports groute
set o $::env(T65_PNR)/out
file mkdir $o
write_verilog -exclude {physical_only_cells leaf_module_declarations pg_objects} $o/[t65p_design].v
write_parasitics -format spef -output $o/[t65p_design]
t65p_log "outputs written"
puts "STAGE_OK groute"
exit
