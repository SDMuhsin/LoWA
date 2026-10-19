# P9b status: measured-activity chip power

## 2026-10-19 19:55 UTC
- State: gate-level simulation flow built. VCS zero-delay simulation of the 17 chip netlists of the flat-chip
  PrimeTime run at 5.5 ns (latch RF, P8f mm), driven by `tb/power/tb_top_gls.sv` (tb_top without RTL-internal
  probes). The sequential library cells are replaced by delayed behavioural models (`site/gls_cells.v`) to avoid the
  zero-delay clock-gate race. The VCS compile takes 2 min 40 s and 2.4 GB.
- Mid-step windows: `tb/power/mk_cut_stim.py` starts the program at unit op k, with the DRAM image the interpreter
  produces for ops < k. An RTL check of the cut at op 44 (LOWA_SG) is running in Verilator.
- Next: measure the gate-level simulation speed, plan the windows, then run PrimePower on them.
- ETA: the plan is set once the speed is known.

## 2026-10-19 20:15 UTC
- Speed measured. With no SAIF, the embedding vector op runs at about 500 cycles/s. With SAIF recording and three
  simulations plus PrimePower on the host, it is about 250 cycles/s for ADAMW, about 40 cycles/s for LN_FWD and
  under 40 cycles/s for the GEMMs. A whole reduced step (10.36 M cycles) would take far too long, and the five
  full segments started at 20:00 were stopped.
- Plan now: 11 sampled windows of 30-100 k cycles (tb/power/win/p9b_plan.txt), one simulation per op type, each
  started at its op with a cut stimulus. That is 3 GEMMs (QKV with merge, FFN1 with GELU, FFN backward), 6 vector
  ops (EMB, SMAX_FWD, LN_FWD, LN_BWD, SMAX_BWD, GN_ROWS), LOWA_SG and ADAMW. Two run at a time. One PrimePower
  session loads the flat chip once and processes each SAIF as it lands. Per-phase power comes from these windows,
  and the step energy is composed with phase times.
- ETA: simulations about 2-3 h, PrimePower 1 h to load plus about 10-20 min per window, so numbers around
  02:00 UTC.
