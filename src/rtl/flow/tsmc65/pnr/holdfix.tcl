# Optional stage after cts: hold fixing at ss as well as ff.  The cts stage optimises hold at ff only (the fast
# corner); with the real clock tree the skew at ss is larger (slower clock buffers, ICG branches), so some short
# paths also fail hold at ss.  This stage enables hold on func_ss and reruns clock_opt's final optimisation.
source $::env(T65_ROOT)/flow/tsmc65/pnr/pnr_common.tcl
t65p_open_stage cts
set_scenario_status func_ss -hold true
set_app_options -name clock_opt.hold.effort -value [envd T65_HOLD_EFFORT high]
clock_opt -from final_opto -to final_opto
t65p_log "hold fix done"
t65p_save_stage holdfix
t65p_stage_reports holdfix
puts "STAGE_OK holdfix"
exit
