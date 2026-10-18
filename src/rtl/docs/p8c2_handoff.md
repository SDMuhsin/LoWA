# P8c2 handoff

Done (pushed: 072d71f RTL/flow/tests, bbf0b18 flow knobs):
- R14a RTL repairs, all regressions bit-exact (lint, test-unit, test-e2e reduced); cycle tables in docs/p8c2_report.md.
- R14c overlap check (tb/mem/mm_ovl_chk.svh): 0 overlaps in test-mem and e2e.
- R14b floorplan step (flow/tsmc65/syn/floorplan.tcl, common/floorplan.txt). Works on lane and lw (both close);
  gm16 / vu16 floorplanned runs stop in DC's placer (internal error) with pe / lane abstractions; vu16 with full
  lanes places but gives -5.25 ns (dont_touch lanes spread over the die).
- DC at 4.0 ns, no floorplan, commit 072d71f: gm16 -0.0095 / -2.12 (497 endpoints), vu16 -0.698 / -640.

Left:
- exit-rule candidate runs (checkout, bbf0b18): vu16 at 4.75 done: -0.272 / -69.9 (540 endpoints), 52 min, 6.6 GB.
  The first vu16 5.0 run hit the server disk quota (its -1.11 is not valid); space was freed (diagnostic run dirs and
  the wip P8a scout netlists deleted) and the chain relaunched: vu16 at 5.0 gave -1.11 again with a failed DDC write (quota still hit; the quota is
  account-wide, project dir 4.9 GB), so it is invalid; gm16 at 4.1 was stopped. Free account space first.
  If vu16 misses 5.0, run 5.25 (`T65_PERIOD=5.25 T65_TAG=_p5.25 FORCE=1 flow/tsmc65/syn/run_hier.sh vu16`).
  log build/tsmc65/hier_cand.log gets CAND_DONE. Read:
  `flow/tsmc65/sync.sh run 'grep -E "^(setup_wns|unit_area|time_)" build/tsmc65/summary/hier_vu16_p*.txt build/tsmc65/summary/hier_gm16_p4.1.txt'`
- fill __RESULT__ / __DC__ in docs/p8c2_report.md, `make sync65-results`, commit, `git push -q origin main`,
  `make sync65-pull`.
