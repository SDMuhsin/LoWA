# Site tool environment template (tcsh). Copy to site.csh (git-ignored) and fill in.
# Sourced by common/env.csh: Synopsys DC (+ PrimeTime / PrimePower / Formality for P8c) and T65_KIT.
source <TOOL_SETUP_SCRIPT_DC> >& /dev/null
source <TOOL_SETUP_SCRIPT_PT> >& /dev/null
source <TOOL_SETUP_SCRIPT_FM> >& /dev/null
setenv T65_KIT <STDCELL_KIT_ROOT>
# RTL simulator for the P8d power-activity runs (sim/run_saif.sh sources it on its own)
setenv T65_VCS_CSH <TOOL_SETUP_SCRIPT_VCS>
# Gate-level power simulation (P9b, sim/run_gls.sh): the library cells' Verilog models (combinational cells are used
# from it; the sequential ones are replaced by site/gls_cells.v, template gls_cells.example.v)
setenv T65_CELL_V <STDCELL_VERILOG_MODELS>
