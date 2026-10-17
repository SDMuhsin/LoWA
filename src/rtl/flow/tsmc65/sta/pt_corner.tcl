# PrimeTime timing of a P8b block netlist (syn/dc_unit.tcl output) at another library corner (tt / ff), with the
# SDC regenerated for that corner and the DC topographical parasitics (ss RC estimate) when they were written.
# Pre-CTS, ideal clocks: setup and hold at the corner are informative.  Environment (syn/run_pt.sh):
#   T65_ROOT, T65_BLOCK, T65_PERIOD, T65_CORNER (tt | ff), T65_OUT (build/tsmc65/hier/<block>)
set t0 [clock seconds]
set flow $::env(T65_ROOT)/flow/tsmc65
source $flow/common/site.tcl
source $flow/common/hier.tcl
source $flow/common/summary.tcl
t65_hier_load
set block  $::env(T65_BLOCK)
set period $::env(T65_PERIOD)
set corner $::env(T65_CORNER)
set out    $::env(T65_OUT)
set fh [open $out/design.txt]; set design [string trim [read $fh]]; close $fh
set_app_var search_path [concat . $::T65_LIBDIR]
set_app_var link_path [list * [t65_db $corner]]
read_verilog $out/$block.v
current_design $design
link_design
set ::argv [list $period $corner $out/pt_$corner.sdc [t65_hier $block io]]
source $flow/sdc/gen_sdc.tcl
source $out/pt_$corner.sdc
set par 0
if {[file exists $out/$block.spef]} { if {![catch {read_parasitics -format spef $out/$block.spef}]} { set par 1 } }
update_timing
set fh [open $out/pt_$corner.summary.txt w]
puts $fh "block $block top $design corner $corner period_ns $period parasitics [expr {$par ? {dc_topo_ss_estimate} : {none}}]"
t65_put_timing $fh
puts $fh [format "pt_wall_s %d" [expr {[clock seconds] - $t0}]]
close $fh
exit
