# P9c status: placement + clock tree (ICC2)

Goal: replace the "pre-layout, no clock tree" caveat of the 5.5 ns chip timing and the tt power with post-placement /
post-CTS numbers (ICC2 floorplan, place_opt, clock_opt, global route; PrimeTime / PrimePower on the result).
Branch p9c, server tau-02, clone ~/lowa_tsmc65/p9c (netlists read from the P8 checkout, read-only).

## 2026-10-19 19:45 UTC
- Flow written: `flow/tsmc65/pnr/` (ndm, floorplan, place, cts, groute, run_pnr.sh), adapted from the template flow;
  `sta/pt_unit.tcl` gains a post-layout mode (T65_POST_DIR: ICC2 netlist + SPEF, propagated clocks).
- Step 1 (feasibility): lw (346,958 cells, 98 k latch-RF bits, 37 k flops) running through the flow on 12 threads
  at 5.5 ns. Reference library: 34 s. Floorplan: 63 s, 2.2 GB (core 1.41 x 1.41 mm at 0.65). place_opt running.
- tau-02: idle apart from one interactive desktop session (load < 1, 25 GB free).
- Decision (a / b / c): pending the lw place / CTS times.

## 2026-10-19 20:20 UTC: feasibility result and decision
- lw place_opt (12 threads, host shared with an unrelated DC job of the account): **25 min wall, 4.1 GB peak**; ICC2
  setup WNS +0.0002 ns at ss (with its +0.10 ns extra margin), 1,257 hold endpoints at ff before CTS.
- Scaling by cell count (memory is close to linear in ICC2: 1.4 GB floorplan / 4.1 GB place for 0.35 M cells):
  flat chip 5.19 M cells -> ~55-60 GB, gm16 3.27 M -> ~35-40 GB for place_opt alone; clock_opt needs more.
  The host has 30 GB (about 17 GB available now). Runtime would be ~7-10 h for place_opt of the chip, before CTS.
- **Decision: (a) is not achievable; (b) as specified (gm16 flat) is not achievable either (memory).**
  Proceed with a measured sub-set: **lw, mm (P8f) and vu16** placed + CTS + global route (1.90 M cells, 37 % of the
  chip's cells, 6.8 of 18.04 mm2), each with post-CTS PrimeTime ss / tt and PrimePower tt vectorless against the same
  unit's pre-layout run at 5.5 ns. vu16 (16 FMA lanes + latch-RF banks) is the structural analogue of gm16 (FMA PE
  array + latch-RF banks); its measured post-/pre-layout factors are applied to gm16, labelled as an extrapolation.
- Running: lw clock_opt (12 threads), mm floorplan->groute (4 threads). vu16 next (~14 GB est.).
