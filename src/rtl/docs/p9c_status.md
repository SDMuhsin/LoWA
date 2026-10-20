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

## 2026-10-19 21:20 UTC: quota incident
- The shared home quota filled at ~21:05 UTC; vu16 floorplan failed in save_lib, and the lw / mm clock_opt sessions
  (62 / 22 min in, both in clock-net detail routing) would have failed at their save. Both were stopped by PID.
- Per the PI, run dirs, databases, parasitics, PT outputs and logs moved to /tmp/lwta_p9c on tau-02 (mode 700,
  symlinked from the clone's build/tsmc65/{pnr,pt,summary}); about 2.3 GB, /tmp has 80 GB free. To be deleted when the
  numbers are final. The lw / mm placements (finished before the quota hit) are intact.
- Relaunched: vu16 floorplan->groute (8 threads), lw cts->groute (6), then mm cts->groute (6). clock_opt now runs
  build_clock + final_opto without clock detail routing (T65_CTS_ROUTE=0); clock and signal nets get the same global route.
- Measured so far: mm place_opt 31.5 min / 4.4 GB (4 threads); pre-layout references at 5.5 ns (same units, same
  method as P8): lw ss setup WNS 0 (flop +1.179), lw tt vectorless 33.9 mW (clock network 54 %), mm ss +0.172.

## 2026-10-20 00:20 UTC
- lw done through global route (0.65 target, 6 threads): clock_opt 1 h 57 min / 4.9 GB, route_global 57 min / 4.5 GB.
  - CTS (ICC2 report_clock_qor): 135,110 sinks, 11 levels, 8,234 clock repeaters (32,494 um2) + clock-gate cells;
    max latency 1.20 ns / global skew 0.59 ns at ss (0.52 / 0.24 ns at ff); post-CTS ICC2 setup WNS -0.006 ns
    (2 endpoints, with +0.10 ns ICC2 margin), hold -0.003 ns (54 endpoints at ff).
  - **Global route is heavily congested at 0.65**: 19.4 % of GRCs overflow (place-stage estimate was 0.29 %); with
    those detours ICC2 setup at ss is -5.40 ns (latch-RF read muxes). Not a usable timing result as is.
- Running: lw post-CTS PrimeTime with (i) the global-route SPEF and (ii) ICC2's placement-based RC estimate of the
  post-CTS database; lw again at 0.50 utilisation (does the congestion go away? die-area datum); mm clock_opt;
  vu16 clock_opt (place took 1 h 56 min / 8 threads).
