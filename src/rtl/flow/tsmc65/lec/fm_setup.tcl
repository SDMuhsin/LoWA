# Formality setup of the LoWA-TA latch register file and clock gates (site independent: no library or cell
# names; sourced by lec/fm_unit.tcl before any design is read).
# - synopsys_auto_setup: the Formality defaults that match DC synthesis semantics (undriven nets, DW, SVF use).
# - Clock gates: mm_bank_t65's write gates are the site mm_icg wrapper, i.e. the same library ICG cell in the RTL
#   and in the netlist, matched as an instance.  compile_ultra -gate_clock adds latch-based ICGs on enabled
#   registers that are not in the RTL; verification_clock_gate_hold_mode any lets Formality treat a gated clock
#   plus its enable as the RTL's enabled register (hold is checked for any enable polarity).
# - Latch RF: the RTL infers one transparent-high latch per word bit (mm_bank_t65, clocked by the ICG output)
#   and DC maps each to a library latch of the same name; Formality matches latches as state points like flops.
#   No latch is converted, merged or excluded here.
# - Nothing is black-boxed here except what lec/fm_unit.tcl declares for compiled children, and nothing is
#   excluded from verification (no set_dont_verify, no set_constant, no user matches).
set_app_var synopsys_auto_setup true
set_app_var verification_clock_gate_hold_mode any
set_app_var verification_failing_point_limit 0
