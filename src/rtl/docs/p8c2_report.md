# P8c2 report: round-2 timing repairs, floorplanned DC of gm16 / vu16, mm read/write ordering

Phase P8c2 of the TSMC-65 port (PLAN R14), the last timing round before P8d. Only what R14 allows was implemented:
(a) four RTL repairs, (b) a DC topographical floorplan for gm16 and vu16, (c) the justification of the mm
read/write ordering. The clock (4.0 ns, ss 0.9 V 125 C, tsmc65 latch RF), the SDC generator, the IO budgets and the
flow options are unchanged; there are no false paths or multicycle paths. DC ran on tau-03 (8 cores, one job at a
time) in the server checkout at commit 072d71f, the pushed commit with all hw/ and flow/ changes of this phase
(later commits only add this report and result summaries). Netlists and full reports stay on the server.

**Result.**
- **gm16: -2.14 (P8b) / -1.31 (P8c) -> -0.0095 ns** at 4.0 ns (TNS -2.12, 497 endpoints) with the R14a repairs.
- **vu16: -1.59 (P8c) -> -0.698 ns** at 4.0 ns (TNS -640, 1,837 endpoints).
- R14b: the floorplan step is in flow/tsmc65 and works (lane, lw: placement area and port side honoured, both
  close), but **DC topographical's placer stops with an internal error on floorplanned gm16 and vu16** whenever
  they link the pe / lane abstractions; with the lanes as full dont_touch netlists vu16 places but gets -5.25 ns.
  So no floorplanned gm16 / vu16 result exists.
- **Exit rule not finished.** vu16 at 4.75 ns: -0.272 ns (no floorplan). The vu16 5.0 ns run gave -1.11 ns, but the
  server account hit its disk quota during it (DDC write failed, and the run's Milkyway library writes are suspect),
  so that number is not valid; gm16's 4.1 ns run was stopped for the same reason. The chip clock is therefore not
  set by this phase: it is at least 5.0 ns (vu16 misses 4.75; rounded up to the 0.25 ns grid). See section 6.
- All regressions bit-exact vs L1 (lint, test-unit, test-e2e reduced). Cycles: e2e reduced step -0.004 %;
  full-step estimate +0.24 % (writer register on real_smax).
- mm write-first ordering kept: no compiler / sequencer traffic issues an overlapping read / write pair in one
  cycle (argument in section 3, check in every integration and e2e test: 0 overlaps).

## 1. RTL repairs (R14a; hw/, tech-neutral)

| unit | repair | files | latency effect |
|---|---|---|---|
| gm | **registered merge request**: the lane hit is decoded from the mg_* port and registered per (buffer, lane); we / address / data are registered once; the read-data select is delayed one more cycle | gm_wfeed | an access reaches the W bank 1 cycle later; merge read data 2 cycles after the request (was 1) |
| lw | merge RMW pipeline follows: read data captured at s5 (was s4), add at s6, write offered at s10 (was s7, 7 cycles after the read: odd, so a read and a write of hits issued every other cycle never meet on the port); in-flight hazard window s0..s7 (was s0..s5); merge_busy released after s11, when the last write has reached the bank | lw_merge | +3 cycles per merged tile (pipeline drain), plus rare extra hazard waits |
| gm | **drain-chain registers**: one register on every PE-to-PE hop of the column drain chains (`DRS` = 1, parameter of gm_top / gm_array / gm_drain; 15 x 16 x 32 flops). gm_drain injects row r `XR(r)` = 15 - r cycles earlier, so every word still reaches the chain bottom in the same cycle | gm_array, gm_drain, gm_top | **none** (all words arrive at the adder in the same cycles; real GEMM cases identical to P8c) |
| vu | **cross-lane operand register** after the 16:1 `xl_row` select in vu_top | vu_top, vu_seq | none between rows (xl_row changes one cycle before the issue); at the start of a cross-lane op the lanes' last totals need one more settle cycle: vu_seq start settle 2 -> 3, **+1 cycle per cross-lane op** |
| vu | **registered writer beat**: the selected lane's beat (data, word count) is registered after the V:1 lane select; S_LANE takes a written group from the register when it was sampled from that lane while the lane was ready and no pop of it was in flight | vu_writer | +1 cycle only when the writer waits for a lane to become ready. Rows beyond the tile are still dropped on the lane's ready bit (a first version that also waited for the register there cost elem +16 % / gnorm +11 %; fixed before the regressions below) |

Why every drain hop and not every 4 rows (the P8c wording): the failing drain group was all 2,046 `dr_s`
endpoints, one per hop, so a register every 4 rows would have cut 3 of 15 hops. Because the earlier injection
absorbs the extra registers (row r injects at offset `(LAMBDA-1)*(15-r) - (15-r) >= 0`), a register on every hop
costs no cycles. Correctness of the earlier injection: rows above the last capture one cycle earlier per row and
now inject `LAMBDA-2` = 2 cycles later per row than the last row, whose schedule is unchanged, so the last row
stays the binding case of gm_ctrl's capture-before-injection rule (DOFF), and gm_ctrl's last-injection bound (LI)
only becomes more conservative. A word passing row r occupies the chain slot of its own arrival cycle, so an
injection never overwrites another word.

Latencies were changed only in the RTL (lw_merge, vu_seq, gm_drain). One testbench model changed with the port
contract it models: tb_lw_top's W-buffer model stands in for gm_wfeed's merge port and now registers the request
like gm_wfeed (read data two cycles after the request). No check was weakened.

## 2. Regressions (all bit-exact vs L1, nothing weakened)

| target | result |
|---|---|
| make lint | PASS: 55 files, 141 Verilator runs, Yosys 55 tops, Icarus 55 tops |
| make test-unit (fp, gemm, vec, lowa, mem, rng, ctrl) | PASS: TEST-FP, TEST-GEMM, TEST-VEC, TEST-LOWA, TEST-MEM [generic], TEST-RNG, TEST-CTRL (6/6) |
| make test-e2e (1 layer, B 2, L 32, 2 steps) | PASS: ALLWRITES 3,389,288 words and 24/24 result tensors bit-exact per step, losses unchanged (0.748631597, 0.58404243) |

Cycle deltas against P8c (same tests; P8c logs build/p8c_test_unit.log, build/p8c_test_e2e.log):

| case | P8c | P8c2 | delta |
|---|---|---|---|
| gm_top real (qproj, scores, pv, attnout, ffn1_256, dv, dgelu, ffn1) and every synthetic gm case | 9,697,593 ... 38,791,261 | identical | 0 |
| gm_top + mm_top real_ffn1_256 | 2,576,252 | 2,576,252 | 0 |
| lw_top merge_real_row / _trn (144 tiles) | 15,135 | 15,420 | +285 (+1.9 %; +2 per tile drain + hazard waits) |
| lw_gemm merge_gemm (9,216 tiles, GEMM-bound) | 9,697,594 | 9,697,594 | 0 (merge hold total 908,778 -> 935,210, never on the critical path) |
| lw_top + mm_top adamw_real_head | 5,329,371 | 5,329,371 | 0 |
| vu_top real_ln_fwd / ln_bwd / emb / smax_bwd | 2,666,543 / 4,706,341 / 6,090,781 / 5,824,549 | each +1 | +1 |
| vu_top real_smax | 9,314,354 | 9,338,928 | +24,574 (+0.26 %) |
| vu_top real_gnorm | 363,711 | 363,720 | +9 |
| vu_top real_loss | 53,624 | 53,673 | +49 |
| vu_top + mm_top real_ln_fwd | 2,965,567 | 2,965,568 | +1 |
| vu_top + rn_mkgen real_smax | 9,240,626 | 9,265,200 | +24,574 (+0.27 %) |
| vu_top + rn_mkgen real_emb / real_smax_bwd | 6,078,493 / 5,701,661 | each +1 | +1 |
| vu_top synthetic (elem, ln_fwd, ln_bwd, smax, loss, gnorm) | | | +1 ... +481 |

- **e2e reduced step:** 10,359,831 -> 10,359,455 cycles (-376, -0.004 %); gm_busy unchanged, vu_busy -377,
  merge_busy +6, lw_busy 0.
- **Full-step estimate (12 layers, B 32, L 128; not run).** GEMM ops: 0 (identical cycles). Merge: hidden behind
  the GEMM (merge_gemm unchanged). Vector ops: the full-size unit cases are +1 except real_smax (+24,574 per
  forward softmax, one per layer, from the writer register when the writer waits on the lanes) and real_gnorm
  (+9 per step): about +0.30 M cycles on the 12-layer step (P8c estimate base ~123 M cycles), **+0.24 %**. The
  reduced step does not show it (its softmax rows are 32 wide).

## 3. mm read/write ordering (R14c): write-first kept, justified and checked

Since P8c a read is ordered when it enters its port's request register, so a write beat accepted in a cycle in
which an overlapping read request is *offered* (not yet registered) goes first; before P8c that beat waited.
The two orderings differ exactly on such pairs. They never occur in the traffic the compiler and sequencer
generate:

1. **Different units.** `place_fences` (sw/compile.py) inserts a FENCE before any op whose read or write ranges
   overlap a write range (or whose write ranges overlap a read range) of an unfenced earlier op of another unit.
   A FENCE waits for that unit's op to retire, and ct_seq retires an op only after the engine's done (all its reads
   returned) and its write channel is idle (all writes B-acknowledged). Overlapping ops of two units therefore never
   run at the same time.
2. **Same unit, different ops.** Each unit runs one op at a time and the next op starts after that retirement.
   The merge engine runs inside the GEMM op; the GEMM op's read ranges include the hit / directory / coefficient
   tables and the taps, which do not overlap its outputs.
3. **Same op.** Only an op whose own read and write ranges overlap can produce such a pair. In the full program
   (12 layers, B 32, L 128; 413 unrolled ops) and in the reduced e2e program, no op's read ranges overlap its write
   ranges, except ADAMW, which updates p, m, v in place (its `ranges()` entry lists them as writes, which covers
   the reads for the fence analysis). lw_adamw reads each word of p, m and v exactly once, in ascending order, and
   writes a word only after the read containing it has returned. A read request offered at the time of a write beat
   therefore covers only words beyond those already returned, which are disjoint from the words the beat writes.

Check that runs in the tests: `tb/mem/mm_ovl_chk.svh` watches mm_top's engine ports in every integration test
(gm / vu / lw on mm_top, test-mem) and in the e2e testbench, and counts every write beat accepted in a cycle in which
an overlapping read is offered (word granular on the beat's mask; mm_top's conservative 8-word range is counted
separately). The testbench fails on a word overlap. Results: test-mem integration, 14 runs: 721,866 same-cycle
offered-read / accepted-write pairs, 0 range overlaps, 0 word overlaps. e2e reduced (2 steps): 324,402 pairs, 0 range overlaps, 0 word overlaps. The mm unit tests (tb_mm_unit)
are not covered by this check: they drive mm_top directly and test its ordering rules, not compiler traffic.

Also fixed: the mm_top header still described the read-first rule; tb/top/run_e2e.py could run a stale simulator
binary after a failed Verilator build (it now deletes the binary before building and passes the tb/mem include
path).

## 4. Floorplan step (R14b; flow/tsmc65)

- `syn/floorplan.tcl` (`t65_floorplan`): core area = measured area / utilisation, core aspect, core offset, die =
  core + offset; `create_die_area`, `create_core_area`, `set_port_side` for every port (clock included); no cell is
  placed by hand. The physical constraints are written to `rpt/floorplan.rpt`, the summary gets a `T65_FP` line.
- `common/floorplan.txt`: one row per block (area, utilisation, aspect, side, offset, and where the area comes
  from); `T65_FP_AREA` / `_UTIL` / `_ASPECT` / `_SIDE` / `_OFFSET` override a column.
- Parameters used: utilisation 0.65 (the template's MQ65_UTIL), square core (the template's `-side_ratio {1 1}`),
  12 um core offset (the template's), all ports on the bottom edge. Areas: the P8c unit totals (gm16 11,170,038 um2,
  vu16 4,364,134 um2, abstracted children included, since their cells are placed in this core).
- `syn/dc_unit.tcl` applies it before compile_ultra when `T65_FP=1` (topographical mode only).
- `syn/run_unit.sh` / `run_hier.sh` / Makefile: `T65_TAG` gives a variant its own output directory and summary
  (e.g. `hier/gm16_fp`), and T65_FP / T65_TAG apply to the named blocks only, never to their children.
  Usage: `make syn65u-gm16 FORCE=1 T65_FP=1 T65_TAG=_fp`.
- Checked first on the lane (220,487 um2 -> 583 x 583 um core): placement area and port sides are honoured
  (report_physical_constraints), and the lane still closes 4.0 ns.

## 5. DC results (ss 0.9 V 125 C)

| block | floorplan | period | WNS / TNS (ns), failing endpoints | unit area um2 (memory) | DC wall / peak RSS | commit |
|---|---|---|---|---|---|---|
| gm16 | no | 4.0 | **-0.0095 / -2.12**, 497 (P8c: -1.314 / -11,327) | 11,304,180 (5,845,328) | 96 min / 10.7 GB | 072d71f |
| gm16 | yes (pe abstraction) | 4.0 | no result: DC placer internal error (OPT-1405) at the first placement | | - | 072d71f |
| vu16 | no | 4.0 | **-0.698 / -640**, 1,837 (P8c: -1.594 / -1,462) | 4,312,191 (2,316,339) | 49 min / 6.6 GB | 072d71f |
| vu16 | yes (lane abstraction) | 4.0 | no result: same placer error (also without the core-area command, and with abstractions as macros) | | - | 072d71f / wip |
| vu16 | yes (lanes as full dont_touch netlists) | 4.0 | -5.25 / -1,499, 2,936 (lane-internal paths up to 8.8 ns: the fixed lane netlists are spread over the 2.6 mm die) | 4,193,581 (cells) | 59 min / 8.4 GB | wip = bbf0b18 flow |
| lw (check of the step) | yes (bank-copy abstractions) | 4.0 | +0.001 / 0 (unfloorplanned: 0 / 0) | 1,307,070 | 25 min | wip |
| lane (check of the step) | yes | 4.0 | 0 / 0 | 220,365 | 6 min / 2.7 GB | wip |
| lane | no | 4.0 | 0 / 0 (flop +0.002) | 220,487 | 5 min / 2.7 GB | 072d71f |
| bank512_gm / bank256_gm / bank256_vu | no | 4.0 | 0 / 0 | 121,710 / 60,956 / 60,956 | 2-5 min | 072d71f |
| vu16 (exit rule) | no | 4.75 | -0.272 / -69.9, 540 | 4,330,564 | 52 min / 6.6 GB | bbf0b18 |
| vu16 (exit rule) | no | 5.0 | -1.11 (invalid: disk quota hit during the run, DDC write failed) | | 44 min / 6.6 GB | bbf0b18 |
| gm16 (exit rule) | no | 4.1 | not run (stopped: disk quota) | | | |

Integrity (same scans as P8c): latches only in mm_bank_t65 (gm16 24,576 words, vu16 8,192), 0 unresolved
references, 0 check_design issues, 0 errors apart from PROC-3; gm16 removals unchanged (200 / 588 / 667).
The unit area grew by 134,142 um2 on gm16 (drain-chain and merge request registers) and fell by 51,943 um2 on vu16.
The floorplan used the P8c areas (11,170,038 / 4,364,134 um2), 1.2 % below the new gm16 total.

**Remaining paths, gm16 at 4.0 ns (no floorplan).** 19 groups, all within -0.01 ns: PE-to-PE operand hops
(`b_s`, `a_s`, `ctl_s`), the new drain registers, X-feeder bank enable -> `q_r`, ACC bank -> X row capture, merge
read data -> `mg_rdata` port, epilogue inputs. The P8c groups (-1.31 ns: merge port -> W bank, drain chain, bank
decode) are gone.

**Remaining paths, vu16 at 4.0 ns (no floorplan).** 60 groups; the worst are the lanes' own outputs / inputs in the
parent, i.e. wire inside the lane abstraction region:
- lane bank `q_r` (920 endpoints, -0.61): `b_iss_addr` broadcast register -> lane -> bank read (top 1.52, bank 2.52);
- writer `lane_ob` (255, -0.70): lane `ob_rsel` -> `ob_data` pin 3.57 ns inside the lane region, writer 0.89;
- seq `nseq` / `sl_pc` / `sl_cool` (-0.68): lane `sfo_cnt` -> minimum over 16 lanes (top 3.29);
- `xl_lane_val` (31, -0.67): lane `xl_val_r` -> pin 2.83 inside the lane region, 16:1 select 1.60;
- column-vector bank `q_r` (-0.61) and cross-lane FMA stage 1 from `x_st` (-0.59).

**Bank-in-parent pessimism (R14b).** The floorplanned gm16 / vu16 runs give no result, so the question can only be
answered where the floorplan runs: lw links its bank copies as abstractions and closes with and without the
floorplan; in gm16 the bank-internal groups (X-feeder word enable -> `q_r`, ACC bank -> capture) are down to
-0.009 ns without a floorplan. The vu16 lane-bank group (-0.61) remains, with 2.5 ns in the bank abstraction
against ~2.1 ns in the bank's own run.

**Not implemented (beyond R14), proposals for the PI:**
1. Floorplanned DC of gm16 / vu16 needs a placement-capable child representation: either a DC version / option
   whose placer accepts these abstractions in a user floorplan (the error is an internal placer sanity check; a
   case for Synopsys support), or physical sub-blocks (lane / PE) with fixed outlines, which is hand floorplanning
   and outside R14b.
2. vu16 RTL (all P8c proposals not signed off in R14): a registered minimum over the 16 lanes' `sfo_cnt` (+1 POP
   window), per-group copies of the broadcast register, and the lane's `ob_data` / `xl_val` driven from next-state
   registers at the lane boundary (no latency; 4,096 + 512 flops).


## 6. Exit rule (R14): state and what is left

- Neither block meets 4.0 ns after (a)+(b) (b gave no runnable floorplan for these two blocks).
- vu16 misses at 4.75 ns (-0.272). Its 5.0 ns run is invalid (quota). Left: free space on the server account (the
  quota is account-wide; this project's directory is 4.9 GB), re-run vu16 at 5.0 (and 5.25 if it misses), gm16 at
  4.1 (or 4.25). Commands: docs/p8c2_handoff.md.
- Every other unit (fp units, pe, banks and bank copies, lane, lw, mm, seq, axil, rng) passed 4.0 ns in P8b / P8c
  (lane, pe, bank copies re-run here at 4.0 ns: pass), so they meet any chip clock >= 4.0 ns without change.
- Server clean-up done in this phase: the failed floorplan run directories and the wip tree's P8a scout netlists
  were deleted (their summaries are kept); no files outside the project directory remain.
