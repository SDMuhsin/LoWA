# Outputs of the post-CTS block before global routing (stage snapshot <design>_cts): netlist with the clock tree and
# the optimisation / hold buffers, and ICC2's parasitics of the placement-based (virtual-route) RC estimate that
# place_opt / clock_opt optimised against, per corner.  Written to $T65_PNR/out_cts/ for sta/run_pt_unit.sh
# (T65_POST_DIR).  Read-only on the database (the working block is not saved).
source $::env(T65_ROOT)/flow/tsmc65/pnr/pnr_common.tcl
t65p_open_stage cts
set o $::env(T65_PNR)/out_cts
file mkdir $o
write_verilog -exclude {physical_only_cells leaf_module_declarations pg_objects} $o/[t65p_design].v
write_parasitics -format spef -output $o/[t65p_design]
t65p_log "outputs written"
puts "STAGE_OK cts_out"
exit
