# Signoff SDC for mb_chip (SIGNOFF_SDC_FILE): max_transition 1.5 ns (= MAX_TRANSITION_CONSTRAINT,
# the std-cell liberty default_max_transition) and NO blanket set_max_capacitance -> cap is checked
# against the liberty per-pin max_capacitance. The PDK default 0.2 pF is unattainable: each SRAM
# CLK pin alone is 0.314 pF (ss). Same policy as librelane/fp16_signoff.sdc.
set ::mb_max_tran $::env(MAX_TRANSITION_CONSTRAINT)
set ::mb_max_cap ""
source [file join [file dirname [file normalize [info script]]] base.sdc]
