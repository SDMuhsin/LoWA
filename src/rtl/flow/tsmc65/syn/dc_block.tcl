# DC synthesis of one LoWA-TA block on the TSMC-65 standard-cell library (compile_ultra, DesignWare, clock gating).
# Environment (set by syn/run_block.sh):
#   T65_ROOT   src/rtl of the tree        T65_BLOCK  block name (common/blocks.tcl)
#   T65_PERIOD clock period, ns           T65_CORNER ss | tt | ff (default ss)
#   T65_TOPO   1 topographical (default) | 0 wire-load
#   T65_CORES  max cores (default 8)      T65_OUT    output directory (stays on the server)
set t0 [clock seconds]
set flow $::env(T65_ROOT)/flow/tsmc65
source $flow/common/site.tcl
source $flow/common/rtl.tcl
source $flow/common/blocks.tcl
source $flow/common/summary.tcl

set block  $::env(T65_BLOCK)
set period $::env(T65_PERIOD)
set corner [expr {[info exists ::env(T65_CORNER)] ? $::env(T65_CORNER) : "ss"}]
set topo   [expr {[info exists ::env(T65_TOPO)] ? $::env(T65_TOPO) : 1}]
set cores  [expr {[info exists ::env(T65_CORES)] ? $::env(T65_CORES) : 8}]
set out    $::env(T65_OUT)
set top    [t65_block $block top]
set params [t65_block $block params]
set dut    [t65_block $block dut]
file mkdir $out/rpt
set_host_options -max_cores $cores

# ---------------------------------------------------------------- libraries
source $flow/common/dc_lib.tcl
t65_dc_setup $corner $topo $out
set_svf $out/$block.svf

# ---------------------------------------------------------------- read RTL
define_design_lib WORK -path $out/work
analyze -format verilog -define [t65_defines] [concat [t65_hw_files] [list $::env(T65_ROOT)/[t65_block $block wrap]]]
if {$params eq ""} { elaborate $top } else { elaborate $top -parameters $params }
set design [get_object_name [current_design]]
link

# ---------------------------------------------------------------- constraints
set ::argv [list $period $corner $out/$block.sdc]
source $flow/sdc/gen_sdc.tcl
set_operating_conditions -max [t65_oc $corner] -max_library [t65_lib $corner] -min [t65_oc ff] -min_library [t65_lib ff]
if {!$topo} { set_wire_load_mode enclosed }
source $out/$block.sdc
set_fix_hold [get_clocks clk]
check_design -nosplit > $out/rpt/check_design.rpt
check_timing          > $out/rpt/check_timing_pre.rpt

# clock gating of enabled flop banks (the latch-RF word gates are explicit mm_icg instances)
set_clock_gating_style -sequential_cell latch -positive_edge_logic integrated \
    -control_point before -minimum_bitwidth 4 -max_fanout 32

# ---------------------------------------------------------------- compile
# no retiming: the netlist keeps the RTL register boundaries (Formality, bit-exactness of the pipeline)
compile_ultra -no_autoungroup -gate_clock
compile_ultra -incremental -no_autoungroup

# ---------------------------------------------------------------- outputs (stay on the server)
change_names -rules verilog -hierarchy
write -format ddc -hierarchy -output $out/$block.ddc
write -format verilog -hierarchy -output $out/$block.v
write_sdc $out/$block.out.sdc
set_svf -off
report_qor                                > $out/rpt/qor.rpt
report_area -hierarchy -nosplit           > $out/rpt/area_hier.rpt
report_timing -max_paths 10 -nworst 1 -nosplit > $out/rpt/timing_max.rpt
report_timing -delay_type min -max_paths 5 -nosplit > $out/rpt/timing_min.rpt
report_constraint -all_violators -nosplit > $out/rpt/violators.rpt
check_timing                              > $out/rpt/check_timing.rpt
report_clock_gating -nosplit              > $out/rpt/clock_gating.rpt
t65_summary $out/summary.txt $block $design $params $period $corner $topo $dut $t0
exit
