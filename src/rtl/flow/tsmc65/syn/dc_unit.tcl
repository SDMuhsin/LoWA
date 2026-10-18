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
set abs    [expr {[t65_hier $block link] eq "abs" && [llength $kids]}]
file mkdir $out/rpt
set_host_options -max_cores $cores
t65_dc_setup $corner $topo $out
set_svf $out/$block.svf

# ---------------------------------------------------------------- read RTL
define_design_lib WORK -path $out/work
analyze -format verilog -define [t65_defines] [t65_hw_files]
# module `@<block>/<instance>`: the sub-design that instance has inside <block>'s elaboration (parameters as the
# parent passes them, so the design name is the parent's by construction)
if {[string index $mod 0] eq "@"} {
    lassign [split [string range $mod 1 end] /] cb inst
    set cm [t65_hier $cb module]; set cp [t65_hier $cb params]
    if {$cp eq ""} { elaborate $cm } else { elaborate $cm -parameters $cp }
    set c [get_cells -quiet $inst]
    if {[sizeof_collection $c] != 1} { puts "T65_ERROR no instance $inst in $cm"; exit 1 }
    current_design [get_attribute $c ref_name]
} elseif {$params eq ""} { elaborate $mod } else { elaborate $mod -parameters $params }
set design [get_object_name [current_design]]
puts "T65_DESIGN $design"

# ---------------------------------------------------------------- mapped children
# Each direct child's generic elaboration is replaced by its mapped .ddc.  Every design inside a child's .ddc
# carries that child's prefix (see the renaming below), so .ddc files of different children never collide.
# The child run elaborated its module alone; if the parent's instantiation names the same parameter set
# differently (other order), the design is matched by its set of name tokens and the mapped one renamed.
# link `abs` (6th column): children are read as their block abstractions (interface logic only; the
# abstraction is written into every block's .ddc), for parents whose full netlist would not fit in memory.
proc t65_tokens {n} { lsort [split $n _] }
set abs_refs {}; set abs_designs {}
set kid_design [dict create]
foreach k $kids {
    set f $hroot/$k/design.txt
    if {![file exists $f]} { puts "T65_ERROR child $k not built ($f)"; exit 1 }
    set fh [open $f]; set kn [string trim [read $fh]]; close $fh
    set pn $kn
    # per-parent copy (link `copy`): the child's design is <child>__<design>; read it, re-link the instances of
    # the elaborated <design> to it and drop the elaborated one
    set orig [string range $kn [string length "${k}__"] end]
    if {[t65_hier $k copy] && [string match "${k}__*" $kn] && [sizeof_collection [get_designs -quiet $orig]]} {
        if {$abs} { lappend abs_refs $kn; set_top_implementation_options -block_references $abs_refs }
        set before [get_object_name [get_designs]]
        read_ddc $hroot/$k/$k.ddc
        if {$abs} { foreach d [get_object_name [get_designs]] { if {$d ni $before} { lappend abs_designs $d } } }
        current_design $design            ;# read_ddc made the child current
        set oc [get_cells -quiet -hierarchical -filter "ref_name==$orig"]
        puts "T65_COPY child $k: [sizeof_collection $oc] instances of $orig -> $kn"
        if {[sizeof_collection $oc] == 0} { puts "T65_ERROR child $k: no instance of $orig"; exit 1 }
        change_link -all_instances $oc $kn
        remove_design $orig
        dict set kid_design $k $kn
        continue
    }
    if {[sizeof_collection [get_designs -quiet $kn]] == 0} {
        set pn ""
        foreach c [get_object_name [get_designs -quiet *]] {
            if {[t65_tokens $c] eq [t65_tokens $kn]} { set pn $c }
        }
        if {$pn eq ""} { puts "T65_ERROR child $k: design $kn is not in the elaborated hierarchy of $design"; exit 1 }
        puts "T65_MATCH child $k $kn as $pn"
    }
    dict set kid_design $k $pn
    remove_design $pn
    if {$abs} { lappend abs_refs $kn; set_top_implementation_options -block_references $abs_refs }
    set before [get_object_name [get_designs]]
    read_ddc $hroot/$k/$k.ddc
    if {$abs} { foreach d [get_object_name [get_designs]] { if {$d ni $before} { lappend abs_designs $d } } }
    if {$pn ne $kn} { rename_design $kn $pn }
}
current_design $design
if {![link]} { puts "T65_ERROR link failed"; exit 1 }
dict for {k kn} $kid_design {
    set c [get_cells -quiet -hierarchical -filter "ref_name==$kn"]
    puts "T65_CHILD $k $kn [sizeof_collection $c]"
    if {[sizeof_collection $c] == 0} { puts "T65_ERROR child $k has no instance"; exit 1 }
    if {[catch {set_dont_touch $c true} e]} { puts "T65_NOTE dont_touch $k: $e" }
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

# ---------------------------------------------------------------- every sub-design gets the block prefix
# (children linked as full netlists included: their own sub-designs already carry the child's prefix, so names
#  nest, e.g. lane__mm_sram_bank_WORDS256_WIDTH32_AW8 and lane__bank256__mm_icg; block abstractions keep
#  their names, so a parent never links two abstractions that contain the same nested abstraction)
set refs [lsort -unique [get_attribute -quiet [get_cells -quiet -hierarchical -filter is_hierarchical==true] ref_name]]
set nren 0
foreach d $refs {
    if {[string match "${block}__*" $d] || $d eq $design} continue
    if {$d in $abs_designs} continue       ;# block abstractions cannot be renamed (UIMG-48)
    rename_design $d ${block}__$d
    incr nren
}
puts "T65_RENAMED $nren"
# per-parent copy: the top design itself carries the block prefix (before the abstraction is written)
if {[t65_hier $block copy]} {
    rename_design $design ${block}__$design
    current_design ${block}__$design
    set design ${block}__$design
    puts "T65_COPY_NAME $design"
}
if {!$abs} { dict for {k kn} $kid_design { dict set kid_design $k ${block}__$kn } }
current_design $design
change_names -rules verilog -hierarchy
set design [get_object_name [current_design]]

# ---------------------------------------------------------------- outputs (stay on the server)
write -format verilog -hierarchy -output $out/$block.v
write_sdc $out/$block.out.sdc
if {$topo && $spef} { catch { write_parasitics -output $out/$block.spef } }
set_svf -off
report_qor                                > $out/rpt/qor.rpt
report_area -hierarchy -nosplit           > $out/rpt/area_hier.rpt
report_timing -max_paths 10 -nworst 1 -nosplit > $out/rpt/timing_max.rpt
report_timing -delay_type min -max_paths 10 -nosplit > $out/rpt/timing_min.rpt
report_constraint -all_violators -nosplit > $out/rpt/violators.rpt
check_timing                              > $out/rpt/check_timing.rpt
report_clock_gating -nosplit              > $out/rpt/clock_gating.rpt
# P8c: the worst path of every violating endpoint group, timed in this session (a fresh session that reloads the
# .ddc cannot match all abstraction pins, see tools/dc_paths); the text goes into the summary
if {[catch {set fo [open $out/groups.txt w]; t65_group_paths $fo $out/rpt/violators.rpt; close $fo} e]} {
    puts "T65_NOTE group paths: $e"
}
if {[catch {t65_unit_summary $out/summary.txt $block $design $params $period $corner $topo $io $kid_design \
        $out/rpt/area_hier.rpt $t0 $abs $hroot} e]} { puts "T65_ERROR summary: $e" }
# the block abstraction (for parents linking with `abs`) goes into the .ddc with the full netlist
if {$topo && [catch {create_block_abstraction} e]} { puts "T65_NOTE create_block_abstraction: $e" }
write -format ddc -hierarchy -output $out/$block.ddc
set fh [open $out/design.txt w]; puts $fh $design; close $fh
exit
