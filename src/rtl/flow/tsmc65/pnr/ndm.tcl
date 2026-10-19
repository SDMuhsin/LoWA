# ICC2 reference library (icc2_lm_shell): cell LEF (frame views) + the ss / tt / ff timing libraries on the techfile.
# Environment: T65_ROOT, T65_NDM (output dir; writes $T65_NDM/t65_std.ndm)
set flow $::env(T65_ROOT)/flow/tsmc65
source $flow/common/site.tcl
source $flow/site/site_pnr.tcl
set_app_options -name lib.workspace.keep_all_physical_cells -value true
create_workspace t65_std -technology $::T65_TF -flow normal
read_lef $::T65_LEF
foreach c {ss tt ff} { read_db [t65_db $c] }
check_workspace
commit_workspace -force -output $::env(T65_NDM)/t65_std.ndm
puts "NDM_OK"
exit
