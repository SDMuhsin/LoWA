# Stage 1: import the block's DC netlists (the block and every netlist below it), MCMM, square core at T65_UTIL,
# signal pins on the boundary, power grid (core ring, mesh on the two top metals, cell rails).  Core-only block, no IO ring.
source $::env(T65_ROOT)/flow/tsmc65/pnr/pnr_common.tcl
t65p_host
file delete -force [t65p_lib]
create_lib [t65p_lib] -technology $::T65_TF -ref_libs [list $::env(T65_NDM)/t65_std.ndm]
set files [t65p_netlists]
set d [t65p_design]
t65p_log "read_verilog [llength $files] netlists, top $d"
read_verilog -top $d $files
current_block $d
link_block
t65p_log "linked"
t65p_layers
t65p_scenarios
t65p_cell_usage
initialize_floorplan -core_utilization [envd T65_UTIL 0.65] -side_ratio {1 1} \
    -core_offset {12 12 12 12} -site_def $::T65_SITE -flip_first_row true
t65p_log "floorplan initialised"
set_block_pin_constraints -self -allowed_layers $::T65_PIN_LAYERS -sides {1 2 3 4}
place_pins -self
# power grid
lassign $::T65_PG vdd vss
create_net -power $vdd
create_net -ground $vss
create_port -direction in -port_type power  $vdd
create_port -direction in -port_type ground $vss
connect_pg_net -net $vdd [get_ports $vdd]
connect_pg_net -net $vss [get_ports $vss]
connect_pg_net -automatic
lassign $::T65_PG_LAYERS rail_l v_l h_l
create_pg_ring_pattern t65_ring -horizontal_layer $h_l -horizontal_width 3 -horizontal_spacing 1.5 \
    -vertical_layer $v_l -vertical_width 3 -vertical_spacing 1.5
set_pg_strategy t65_ring -core -pattern [list {name: t65_ring} [list nets: [list $vdd $vss]] {offset: {2 2}}]
create_pg_mesh_pattern t65_mesh -layers [list \
    [list [list vertical_layer: $v_l]   {width: 2} {spacing: interleaving} {pitch: 40} {offset: 10}] \
    [list [list horizontal_layer: $h_l] {width: 2} {spacing: interleaving} {pitch: 40} {offset: 10}]]
set_pg_strategy t65_mesh -core -pattern [list {name: t65_mesh} [list nets: [list $vdd $vss]]] -extension {{stop: outermost_ring}}
create_pg_std_cell_conn_pattern t65_rails -layers [list $rail_l]
set_pg_strategy t65_rails -core -pattern [list {name: t65_rails} [list nets: [list $vdd $vss]]]
compile_pg -strategies {t65_ring t65_mesh t65_rails}
t65p_log "pg done"
redirect [t65p_rpt floorplan_design.rpt] { report_design -floorplan }
t65p_save_stage floorplan
t65p_stage_reports floorplan
puts "STAGE_OK floorplan"
exit
