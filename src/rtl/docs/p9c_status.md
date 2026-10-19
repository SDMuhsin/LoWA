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
