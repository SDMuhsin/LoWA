# Stage 3: clock tree synthesis, clock-net routing and post-CTS optimisation (setup at ss, hold at ff); clocks are
# propagated from here on.  Clock nets: double spacing on the mid metals.
source $::env(T65_ROOT)/flow/tsmc65/pnr/pnr_common.tcl
t65p_open_stage place
create_routing_rule t65_ck_2s -multiplier_spacing 2
set_clock_routing_rules -rules t65_ck_2s -min_routing_layer M3 -max_routing_layer M6
set_clock_tree_options -target_skew [envd T65_CTS_SKEW 0.10]
set_app_options -name clock_opt.hold.effort -value [envd T65_HOLD_EFFORT high]
# T65_CTS_ROUTE=0: skip clock_opt's detail routing of the clock nets (route_clock); they are then globally routed with
# the signal nets in groute.tcl, so clock and signal parasitics come from the same global route.
if {[envd T65_CTS_ROUTE 1]} {
    clock_opt
} else {
    clock_opt -from build_clock -to build_clock
    t65p_log "build_clock done"
    clock_opt -from final_opto -to final_opto
}
t65p_log "clock_opt done"
redirect [t65p_rpt cts_clock_qor.rpt]     { report_clock_qor }
redirect [t65p_rpt cts_clock_latency.rpt] { report_clock_qor -type latency }
redirect [t65p_rpt cts_clock_summary.rpt] { report_clock_qor -type summary }
t65p_save_stage cts
t65p_stage_reports cts
puts "STAGE_OK cts"
exit
