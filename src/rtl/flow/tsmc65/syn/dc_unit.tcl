# DC topographical synthesis of one block of the P8b hierarchical unit flow (block table common/hier.txt).
# The block's module is elaborated with its full16 parameters and compiled without a wrapper: its own ports carry
# the generator's IO delays (sdc/gen_sdc.tcl, io_pct of the table).  Every child block was compiled before (its
# own run of this script); its generic elaboration is replaced by the mapped .ddc and kept dont_touch, so each
# unique sub-block is compiled once and its timing is still analysed in the parent through its netlist.
# After the compile, the block's own sub-designs are renamed <block>__<design> so that the mapped designs of
# different blocks never collide when a later parent reads several .ddc files.
# Environment (syn/run_unit.sh):
#   T65_ROOT  src/rtl of the tree     T65_BLOCK block name      T65_PERIOD clock period, ns
#   T65_CORNER ss | tt | ff (ss)      T65_TOPO 1 | 0 (1)        T65_CORES max cores (8)
#   T65_OUT   output dir (build/tsmc65/hier/<block>, stays on the server)
#   T65_INCR  1: compile_ultra -incremental after the first pass (1)
#   T65_SPEF  1: write the topographical parasitics for the PT corner runs (1)
set t0 [clock seconds]
set flow $::env(T65_ROOT)/flow/tsmc65
source $flow/common/site.tcl
source $flow/common/rtl.tcl
source $flow/common/hier.tcl
source $flow/common/summary.tcl
source $flow/common/dc_lib.tcl
t65_hier_load
proc envd {k d} { expr {[info exists ::env($k)] ? $::env($k) : $d} }

set block  $::env(T65_BLOCK)
set period $::env(T65_PERIOD)
set corner [envd T65_CORNER ss]
set topo   [envd T65_TOPO 1]
set cores  [envd T65_CORES 8]
set incr   [envd T65_INCR 1]
set spef   [envd T65_SPEF 1]
set out    $::env(T65_OUT)
set hroot  [file dirname $out]
set mod    [t65_hier $block module]
set params [t65_hier $block params]
set kids   [t65_hier $block children]
set io     [t65_hier $block io]
file mkdir $out/rpt
set_host_options -max_cores $cores
t65_dc_setup $corner $topo $out
set_svf $out/$block.svf

# ---------------------------------------------------------------- read RTL
define_design_lib WORK -path $out/work
analyze -format verilog -define [t65_defines] [t65_hw_files]
if {$params eq ""} { elaborate $mod } else { elaborate $mod -parameters $params }
set design [get_object_name [current_design]]
puts "T65_DESIGN $design"

# ---------------------------------------------------------------- mapped children
# all descendants (children of children too): their generic designs go first, then each .ddc is read once
proc t65_desc {b} { set r {}; foreach k [t65_hier $b children] { lappend r $k; lappend r {*}[t65_desc $k] }; return [lsort -unique $r] }
# The child run elaborated its module alone; if the parent's instantiation names the same parameter set
# differently (other order), the design is matched by its set of name tokens and the mapped one renamed.
proc t65_tokens {n} { lsort [split $n _] }
set kid_design [dict create]
set kid_file   [dict create]
foreach k [t65_desc $block] {
    set f $hroot/$k/design.txt
    if {![file exists $f]} { puts "T65_ERROR child $k not built ($f)"; exit 1 }
    set fh [open $f]; set kn [string trim [read $fh]]; close $fh
    dict set kid_file $k $kn
    set pn $kn
    if {[sizeof_collection [get_designs -quiet $kn]] == 0} {
        set pn ""
        foreach c [get_object_name [get_designs -quiet [t65_hier $k module]*]] {
            if {[t65_tokens $c] eq [t65_tokens $kn]} { set pn $c }
        }
        if {$pn eq ""} {
            puts "T65_ERROR child $k: design $kn is not in the elaborated hierarchy of $design; candidates:\
                  [get_object_name [get_designs -quiet [t65_hier $k module]*]]"
            exit 1
        }
        puts "T65_MATCH child $k $kn as $pn"
    }
    dict set kid_design $k $pn
    remove_design $pn
}
set owned {}
foreach k $kids {
    set kn [dict get $kid_file $k]
    set pn [dict get $kid_design $k]
    if {[sizeof_collection [get_designs -quiet $pn]] == 0} {
        set before [get_object_name [get_designs]]
        read_ddc $hroot/$k/$k.ddc
        foreach d [get_object_name [get_designs]] { if {$d ni $before} { lappend owned $d } }
    }
    # a grandchild loaded by an earlier child's .ddc under its own run's name
    dict for {g gn} $kid_file {
        set gp [dict get $kid_design $g]
        if {$gn ne $gp && [sizeof_collection [get_designs -quiet $gn]] && ![sizeof_collection [get_designs -quiet $gp]]} {
            rename_design $gn $gp; lappend owned $gp
        }
    }
}
current_design $design
if {![link]} { puts "T65_ERROR link failed"; exit 1 }
set kid_cells {}
foreach k [t65_desc $block] {
    set kn [dict get $kid_design $k]
    set c [get_cells -quiet -hierarchical -filter "ref_name==$kn"]
    puts "T65_CHILD $k $kn [sizeof_collection $c]"
    if {$k in $kids} {
        if {[sizeof_collection $c] == 0} { puts "T65_ERROR child $k has no instance"; exit 1 }
        set_dont_touch $c true
        set kid_cells [add_to_collection $kid_cells $c]
    }
}

# ---------------------------------------------------------------- constraints
set ::argv [list $period $corner $out/$block.sdc $io]
source $flow/sdc/gen_sdc.tcl
set_operating_conditions -max [t65_oc $corner] -max_library [t65_lib $corner] -min [t65_oc ff] -min_library [t65_lib ff]
if {!$topo} { set_wire_load_mode enclosed }
source $out/$block.sdc
# hold is not fixed before CTS (ideal clocks); the ff hold numbers in the summary are informative only
check_design -nosplit > $out/rpt/check_design.rpt
check_timing          > $out/rpt/check_timing_pre.rpt
set_clock_gating_style -sequential_cell latch -positive_edge_logic integrated \
    -control_point before -minimum_bitwidth 4 -max_fanout 32

# ---------------------------------------------------------------- compile (no retiming: RTL register boundaries kept)
compile_ultra -no_autoungroup -gate_clock
if {$incr} { compile_ultra -incremental -no_autoungroup }

# ---------------------------------------------------------------- own sub-designs get the block prefix
set refs [lsort -unique [get_attribute -quiet [get_cells -quiet -hierarchical -filter is_hierarchical==true] ref_name]]
set nren 0
foreach d $refs {
    if {$d in $owned || [string match "${block}__*" $d]} continue
    if {[dict exists $kid_design $d]} continue
    if {[lsearch -exact [dict values $kid_design] $d] >= 0} continue
    rename_design $d ${block}__$d
    incr nren
}
puts "T65_RENAMED $nren"
current_design $design
change_names -rules verilog -hierarchy
set design [get_object_name [current_design]]

# ---------------------------------------------------------------- outputs (stay on the server)
write -format ddc -hierarchy -output $out/$block.ddc
write -format verilog -hierarchy -output $out/$block.v
write_sdc $out/$block.out.sdc
if {$topo && $spef} { catch { write_parasitics -output $out/$block.spef } }
set_svf -off
set fh [open $out/design.txt w]; puts $fh $design; close $fh
report_qor                                > $out/rpt/qor.rpt
report_area -hierarchy -nosplit           > $out/rpt/area_hier.rpt
report_timing -max_paths 10 -nworst 1 -nosplit > $out/rpt/timing_max.rpt
report_timing -delay_type min -max_paths 10 -nosplit > $out/rpt/timing_min.rpt
report_constraint -all_violators -nosplit > $out/rpt/violators.rpt
check_timing                              > $out/rpt/check_timing.rpt
report_clock_gating -nosplit              > $out/rpt/clock_gating.rpt
t65_unit_summary $out/summary.txt $block $design $params $period $corner $topo $io $kid_design $out/rpt/area_hier.rpt $t0
exit
