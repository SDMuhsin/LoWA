# P8e (PLAN R16): sta/pt_unit.tcl (PrimeTime / PrimePower of one block's complete netlist) run unchanged on a run of
# sram/run_sram.sh: its common/ sources and the hier root are replaced (T65_HROOT = build/tsmc65/hier_<variant>_p<P>),
# and in the SRAM variant the macros' .db of the corner are added to the link path (sram/sram_lib.tcl).
set flow $::env(T65_ROOT)/flow/tsmc65
source $flow/common/site.tcl
source $flow/common/hier.tcl
source $flow/common/summary.tcl
set ::T65_SRAM_LATCH [expr {$::env(T65_VARIANT) eq "latch"}]
source $flow/sram/sram_lib.tcl
set fh [open $flow/sta/pt_unit.tcl]; set code [read $fh]; close $fh
set ns [regsub -all -line {^source \$flow/common/\S+\.tcl[ \t]*$} $code {} code]
set nh [regsub -line {^set hroot .*$} $code {set hroot $::env(T65_HROOT)} code]
set nl [regsub -line {^set_app_var link_path \[concat \* \[t65_db \$corner\] \$etms\]$} $code \
    {set_app_var link_path [concat * [t65_db $corner] $etms [expr {$::T65_SRAM_LATCH ? {} : [t65_mc_dbs $corner]}]]} code]
if {$ns != 3 || $nh != 1 || $nl != 1} { puts "T65_ERROR pt_sram: pt_unit.tcl changed shape ($ns $nh $nl)"; exit 1 }
uplevel #0 $code
