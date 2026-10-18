# P8e (PLAN R16): syn/dc_unit.tcl run unchanged except for the SRAM-macro overrides (sram/sram_lib.tcl) and an extra
# summary block.  The common/ files are sourced here first; dc_unit.tcl's own `source $flow/common/...` lines and its
# final `exit` are removed from its text before it is evaluated, so its flow is the P8b/P8d one line for line.
# T65_VARIANT=latch runs the latch-RF baseline through the same script (no overrides, same extra summary / power).
# Environment: as syn/dc_unit.tcl, plus T65_MC_OUT / T65_MC_CORNERS / T65_MC_SHAPES (sram/run_sram.sh).
set flow $::env(T65_ROOT)/flow/tsmc65
source $flow/common/site.tcl
source $flow/common/rtl.tcl
source $flow/common/hier.tcl
source $flow/common/summary.tcl
source $flow/common/dc_lib.tcl
set ::T65_SRAM_LATCH [expr {[info exists ::env(T65_VARIANT)] && $::env(T65_VARIANT) eq "latch"}]
source $flow/sram/sram_lib.tcl
set fh [open $flow/syn/dc_unit.tcl]; set code [read $fh]; close $fh
set ns [regsub -all -line {^source \$flow/common/\S+\.tcl[ \t]*$} $code {} code]
set ne [regsub -line {^exit[ \t]*$} $code {} code]
if {$ns != 5 || $ne != 1} { puts "T65_ERROR dc_sram: dc_unit.tcl changed shape (sources $ns, exit $ne)"; exit 1 }
uplevel #0 $code
# ---------------------------------------------------------------- SRAM variant summary + power (numbers only)
current_design [string trim [exec cat $out/design.txt]]
if {[catch {t65_sram_summary $out/summary.txt $period} e]} { puts "T65_ERROR sram summary: $e" }
# DC power at the run's corner with the vectorless defaults P8d uses in PrimePower (toggle 0.1 / cycle, static 0.5)
set_app_var power_default_toggle_rate 0.1
set_app_var power_default_static_probability 0.5
redirect $out/rpt/power.rpt { report_power -nosplit }
redirect $out/rpt/power_hier.rpt { report_power -nosplit -hierarchy }
if {[catch {
    redirect -variable pr { report_power -nosplit -cell -only [t65_mc_cells] }
    set fh [open $out/summary.txt a]
    redirect -variable pt { report_power -nosplit }
    if {[regexp -line {^Total\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)} $pt -> a au b bu c cu d du]} {
        puts $fh "power_total internal $a $au switching $b $bu leakage $c $cu total $d $du"
    }
    redirect $out/rpt/power_macros.rpt { puts $pr }
    foreach l [split $pr "\n"] { if {[regexp {^Total\s} $l]} { puts $fh "power_macros [regexp -all -inline {\S+} $l]" } }
    close $fh
} e]} { puts "T65_NOTE sram power: $e" }
puts "T65_SRAM_DONE"
exit
