# Unit-level STA constraints (llmdocs/asic_flow_tutorial.md section 5 style), used on the Yosys
# netlist of flow/synth/wrap_<unit>.v (inputs registered in the wrapper, outputs registered in
# the unit), so every timing path that matters is register to register.
create_clock -name clk -period $::env(PERIOD) [get_ports clk]
set_clock_uncertainty -setup 0.50 [get_clocks clk]
set_clock_uncertainty -hold  0.10 [get_clocks clk]
set_input_delay  8 -clock clk [delete_from_list [all_inputs] [get_ports {clk}]]
set_output_delay 8 -clock clk [all_outputs]
set_load 0.07291 [all_outputs]
set_driving_cell -lib_cell gf180mcu_as_sc_mcu7t3v3__inv_2 -pin Y [delete_from_list [all_inputs] [get_ports clk]]
