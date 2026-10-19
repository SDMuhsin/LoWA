# P8 report: LoWA-TA on TSMC 65 nm, through synthesis (P8a-P8d)

Phase P8 (PLAN R11-R16) ports the full16 configuration of LoWA-TA (16 x 16 FP32 FMA PEs, 16-lane vector
unit, LoWA engine, memory system, sequencer) to TSMC 65 nm GP standard cells (RVT) and takes it through
logic synthesis, formal equivalence and pre-layout timing / power. This report closes P8 (phase P8d, R15). It
is pre-layout only: there is no floorplan, placement, clock tree or routing (section 8).

All numbers come from tool runs on the CMC server tau-03 (Synopsys DC W-2024.09-SP2 topographical, Formality
and PrimeTime / PrimePower Q-2019.12-SP2, VCS S-2021.09) in the server checkout of the pushed commits named in
section 9. Netlists, full reports, SAIF files and library data stay on the server; only scripts and summary
numbers are in the repository.

## 1. Summary

| item | result |
|---|---|
| chip clock | **7.75 ns (129.0 MHz) at ss 0.9 V 125 C, chip level** (PLAN R17). Unit-level closure: 5.25 ns (R14 / R15 exit rule: vu16 misses 4.75 and 5.0 ns, closes 5.25 ns); the flat chip misses 5.25 ns by 2.27 ns on cross-unit paths through mm_top's combinational ready outputs, and closes 7.75 ns (flop slack +0.229 ns) |
| standard-cell area (ss netlists, cells only) | **18.05 mm2**, of which **8.89 mm2** latch register file (37,376 words, 146 KiB) |
| leaf cells / flops / latches | 5,211,998 / 415,461 / 1,196,032 (+ 37,376 word ICGs, + DC clock gates) |
| setup, PrimeTime on the complete chip netlist | __PTSS__ |
| setup at tt / ff | __PTTTFF__ |
| hold (pre-CTS, ideal clocks, informative) | __PTHOLD__ |
| Formality, RTL vs netlist | __FMSUM__ |
| average power, tt 1.0 V 25 C, 129.0 MHz | __POWER__ |
| full-size training step (12 layers, B 32, L 128) | **3,290.6 M cycles = 25.5 s** at 129.0 MHz (DRAM: 256-bit AXI = 4.13 GB/s peak, 3.76 GB/s average demand); 17.3 s at the unit-level 5.25 ns |

__SUMMARY_NOTES__

## 2. Chip clock (exit rule)

R12 fixed the target at 4.0 ns. After two RTL timing rounds (P8c, P8c2) gm16 and vu16 still missed it, so the
R14 exit rule applies: the chip clock is the slowest unit's closed period on the 0.25 ns grid, with no further RTL
changes. P8d re-ran the candidates on a clean disk quota (P8c2's 5.0 ns run had hit the quota):

| block | period | WNS / TNS (ns), failing endpoints | unit area um2 | DC wall / peak RSS | status |
|---|---|---|---|---|---|
| vu16 (P8c2) | 4.75 | -0.272 / -69.9, 540 | 4,330,564 | 52 min / 6.6 GB | misses |
| vu16 | 5.0 | **-1.114 / -608.5, 982** | 4,290,642 | 47 min / 6.6 GB | misses (reproduces P8c2's -1.11) |
| **vu16** | **5.25** | **0 / 0** (worst flop endpoint +0.0001) | 4,297,890 | 46 min / 6.6 GB | **closes** |
| **gm16** | **5.25** | **0 / 0** (flop 0.000) | 11,166,587 | 90 min / 13.4 GB | closes |
| every other unit (lw, mm, seq, axil, lane, pe, banks, rng, fp units) | 4.0 | 0 / 0 | | | closes 4.0 (P8b / P8c / P8d) |

- vu16 at 5.0 ns is worse than at 4.75 ns. Its worst paths at 5.0 ns run from the lanes' `ob_rsel` registers to the
  writer's `lane_ob` registers, with 4.42 ns inside the lane abstraction's region (3.57 ns at 4.0 ns). The lane
  abstractions are fixed, dont_touch netlists; how far DC topographical places the writer from them, and so the
  wire estimate on those pins, changes from run to run with the constraint. The result at 5.0 ns repeated
  exactly in two runs (P8c2, P8d), so it is not noise of one run.
- So the **unit-level** closed period is **5.25 ns**. gm16, the second limiter (-0.0095 ns at 4.0 ns), closes 5.25 ns
  in one run. The flat-chip PrimeTime run then shows cross-unit paths that the unit runs cannot see (section
  4.2); per R17 the **chip clock is the chip-level closed period, 7.75 ns**. The units keep their netlists
  (5.25 / 4.0 ns), which are tighter than 7.75 ns needs.
- The units that closed 4.0 ns were not re-synthesised at 5.25 ns; their 4.0 ns netlists are used (tighter
  than needed). The chip-level PrimeTime run (section 4) times them all at 5.25 ns.

## 3. Area

DC unit totals at ss (cell area; a unit total = the unit's own cells + instances x each abstracted child's own
run). PrimeTime's per-instance count on the flat netlists (section 4) reproduces every unit's cell area, flop and
latch count exactly.

| unit | netlist (period) | cells | area mm2 | of which latch RF mm2 | flops | latches (RF bits) |
|---|---|---|---|---|---|---|
| gemm, gm_top 16 x 16 (256 PEs) | gm16 (5.25) | 3,272,195 | **11.167** | 5.845 (24,576 words) | 181,373 | 786,432 |
| vec, vu_top V = 16 | vu16 (5.25) | 1,221,897 | **4.298** | 2.316 (9,728 words) | 112,086 | 311,296 |
| lowa, lw_top | lw (4.0) | 346,958 | **1.300** | 0.730 (3,072 words) | 36,806 | 98,304 |
| mem, mm_top (8 R / 2 keep / 4 W channels, 2 mask generators) | mm (4.0) | 357,861 | **1.219** | 0 | 78,991 | 0 |
| ctrl, ct_seq + ct_dma | seq (4.0) | 13,065 | 0.063 | 0 | 6,203 | 0 |
| ctrl, top_axil | axil (4.0) | 22 | 0.0001 | 0 | 2 | 0 |
| top_chip glue | chip (5.25) | 0 | 0 | 0 | 0 | 0 |
| **chip** | | **5,211,998** | **18.046** | **8.891** | **415,461** | **1,196,032** |

- The latch RF (R11a: no usable SRAM compiler) is 49 % of the cell area. Per word: 32 latches, a write-data
  register share and a word ICG; ~238 um2 per word (bank512: 121,710 um2 / 512).
- gm16's PE array is 3.95 mm2 (256 x 15,412 um2); the rest of gm16 outside the RF is feeders, drain, epilogue
  (16 SFUs), loader and control.
- top_chip has no logic of its own: its netlist (211 lines) only connects the six unit instances.
- No floorplan, so no die area. At the template's 0.65 utilisation the cell area alone needs a ~27.8 mm2 core
  (~5.3 mm square), before power grid, IO ring and any routing-driven growth.

## 4. Timing

### 4.1 DC (ss, per block)

See section 2 for gm16 / vu16; lw (rebuilt in P8d, see 6.3) +0.0002 ns and mm 0.000 ns at 4.0 ns.

The chip-level DC run (top_chip linking the six units as block abstractions, and the abstractions below them)
links cleanly but its timing is not usable: DC topographical reports OPT-1442 (the abstracted units' cells have
no locations in the chip's physical context, which needs a chip floorplan with physical blocks) and then times
unplaced cells (-1.99 ns on a path inside gm16's epilogue that closes in gm16's own run). Since top_chip has no
logic, the compile has nothing to optimise; the chip netlist is pure interconnect and is verified by Formality
(section 5). Chip-level timing is taken from PrimeTime on the complete netlist instead.

### 4.2 PrimeTime, complete netlists (pre-layout)

Method: each unit's netlist is read together with the netlists of every block it links as an abstraction (so the
PEs, lanes, banks and mask generators are all present), each block's DC topographical parasitics (ss RC estimate)
are annotated on every instance of that block (321 SPEF reads in gm16; 0 failures), clocks are ideal, and the SDC
is the DC one (5.25 ns, 0.25 ns setup / 0.05 ns hold uncertainty, IO 20 %) regenerated per corner. tt and ff
reuse the ss RC estimate. The chip run reads all 17 netlists (5,211,998 leaf cells).

Per unit (complete netlist, 5.25 ns). WNS / TNS in ns, violating endpoints in brackets; "flop" = worst endpoint
that is a flop or output (latch-RF endpoints borrow and sit at 0).

| unit | leaf cells | ss setup | ss flop WNS | tt setup (flop WNS) | ff setup (flop WNS) | ff hold WNS / TNS [endpoints] |
|---|---|---|---|---|---|---|
| gm16 | 3,272,195 | **-0.0059 / -0.006 [1]** | -0.0059 | 0 (+0.776) | 0 (+0.936) | -0.0053 / -5.76 [5,337] |
| vu16 | 1,221,897 | **-0.0228 / -0.025 [2]** | -0.0228 | 0 (+1.295) | 0 (+1.768) | -0.0015 / -0.54 [514] |
| lw | 346,958 | 0 / 0 | +0.979 | 0 (+2.165) | 0 (+2.797) | -0.0015 / -0.12 [115] |
| mm | 357,861 | +0.737 | +0.737 | +1.563 | +2.010 | -0.0014 / -0.02 [15] |
| seq | 13,065 | +1.205 | +1.205 | +2.720 | +3.393 | -0.0016 / -0.03 [29] |
| axil | 22 | +2.455 | +2.455 | +2.629 | +2.718 | +0.044 |

- ss misses in PrimeTime, where DC closed: gm16 one endpoint by 5.9 ps (`u_array/g_r_15__g_c_0__u_pe/dr_s_reg_10_ ->
  u_drain/g_add_0__u_add` FMA stage, the drain-chain exit of the last PE row into the drain adder); vu16 two
  endpoints by 22.8 ps (`u_seq/x_st_reg -> u_seq/u_xfma` stage 1, the cross-lane FMA). In PrimeTime the PE / lane
  internals are timed with their own SPEF instead of DC's abstraction model, which accounts for these few ps.
  They are within DC-vs-PT correlation and far below what placement will change; they are reported as misses.
- Hold at ss and tt: no violators in any unit.

Chip (flat, all 17 netlists): **the chip does not meet 5.25 ns at ss.**

| chip, flat (5,211,998 cells, 356 SPEF reads) | setup WNS / TNS (ns) [endpoints] | flop WNS | hold WNS [endpoints] |
|---|---|---|---|
| ss, 5.25 ns | **-2.272 / -569.9 [476]** | -2.272 | +0.050 [0] |
| tt / ff, 5.25 ns | not run (stopped when R17 moved the chip clock to 7.75 ns) | | |
| **ss, 7.75 ns (chip clock)** | **0 / 0 [0]** (latch-RF endpoints borrow) | **+0.229** | +0.050 [0] |
| tt, 7.75 ns | 0 / 0 [0] | +3.076 | +0.016 [0] |
| ff, 7.75 ns | 0 / 0 [0] | +3.436 | -0.0053 / -6.47 [6,012] (**pre-CTS, informative; fixed by hold buffering in P&R**, 4.3) |

Worst setup endpoint per unit (ss, 5.25 ns): u_gm -2.272 [11], u_mm -1.748 [41], u_lw -1.666 [419], u_vu -0.119
[5], u_seq +1.205, u_axil +3.507. Every unit closes 5.25 ns in its own complete-netlist run (table above), so
these are **paths that cross units**. The worst one starts and ends in gm_top's epilogue (`u_ep/as_rp_ohr_reg ->
u_ep/asm_cred_reg`, arrival 7.23 ns): gm's write address/data mux drives `wr_addr` out of gm_top (3.85 ns in
gm's own run), mm_top's write port computes `wr_ready` from it combinationally (`mm_wrport`: `wr_ready = !blk &&
...`, with the read/write overlap block), and gm takes `wr_ready` back into `wr_pop -> asm_cred` in the same
cycle. The unit runs could not see this: each side of the boundary was given 20 % of the period as IO budget on
the assumption of registered unit outputs (docs/rtl_conventions.md), and mm_top's ready outputs are not
registered (mm's own run shows the input-to-output paths `rd_req_len -> rd_req_ready`, `... -> wr_ready`). lw
(merge / AdamW channels) and vu (loader / writer channels) talk to mm the same way.

**Cross-unit paths (for P8f).** Flat chip at 7.75 ns, ss: every path into a flop or output with slack below 2.5 ns
(arrival above ~5.0 ns, i.e. the paths that fail or nearly fail at 5.25 ns), grouped by the sequence of units
it crosses (20,000-path search, T65_XPATHS):

| units crossed | endpoints | worst slack at 7.75 ns | arrival ns | worst path | via mm |
|---|---|---|---|---|---|
| gm > mm > gm | 8 | +0.229 | 7.227 | `u_gm/u_ep/as_rp_ohr_reg -> u_gm/u_ep/asm_cred_reg` (gm 3.98, mm 2.16, gm 1.09) | yes |
| gm > mm | 5 | +0.752 | 6.733 | `u_gm/u_ep/as_rp_ohr_reg -> u_mm/g_wr_0__u_wr/a_i_reg` | yes |
| lw > mm > lw | 395 | +0.834 | 6.625 | `u_lw/cur_reg -> u_lw/u_ad/wr_data_reg` | yes |
| vu > mm | 12 | +1.883 | 5.603 | `u_vu/u_ld/bs_mine_reg -> u_mm/g_rd_2__u_rd/u_mf/count_reg` | yes |
| lw > mm | 3 | +2.222 | 5.296 | `u_lw/cur_reg -> u_mm/g_wr_2__u_wr/a_i_reg` | yes |
| **seq > vu** | 3 | +2.381 | 5.116 | `u_seq/vu_p_reg -> u_vu/u_ld/rd1_req_addr_reg` | **no** |

The only group that does not pass through mm is **seq > vu** (sequencer descriptor register into vu_loader's
read-address register, 5.12 ns arrival): it has no combinational feedthrough, but seq's output stage plus vu's
input logic exceed the 20 % / 80 % split the unit runs assumed. At 5.25 ns it would miss by about 0.12 ns. No other
pair of units (gm-vu, gm-lw, seq-gm, seq-lw, seq-mm, axil) has a path with arrival above 5.0 ns.

So with the present RTL the chip-level period at ss, pre-layout, is about **5.25 + 2.27 = 7.52 ns**, i.e.
7.75 ns on the 0.25 ns grid (129 MHz), unless the ready handshakes between the engines and mm_top are
registered (an RTL change, not allowed in P8d; it needs PI sign-off). __CHIPDECISION__

### 4.3 Hold

**All hold numbers here are pre-CTS and informative; they are not sign-off figures.** What they mean: with ideal
clocks (zero skew, zero insertion delay) PrimeTime checks that data launched at a clock edge does not reach the
capturing flop or latch before that same edge plus the 0.05 ns hold uncertainty, at the fast corner (ff 1.1 V
-40 C, where cells are fastest). The flat chip at 7.75 ns has **6,012 endpoints** at ff with negative hold slack
(sum over units 6,010 in the unit runs), **all between -0.0053 ns and 0**, TNS -6.47 ns; they are short
flop-to-flop hops (PE-to-PE operand and drain registers, bank write registers) whose data delay is a few ps below
the 0.05 ns margin. No hold fixing was done (R13: no pre-CTS hold buffers). In place and route, the real clock tree
replaces the ideal clock (its skew can make hold better or worse per path), and the router's hold-fixing step
inserts delay buffers on the paths that still violate; violations of a few ps are routinely fixed that way, at a
small area cost. There are no hold violations at ss or tt.

## 5. Formal equivalence (Formality)

Method (flow/tsmc65/lec): RTL (the files, defines and site ICG wrapper DC read) as reference, the DC netlist as
implementation, with the block's DC guidance file (.svf). Site-independent setup in `lec/fm_setup.tcl`:
`synopsys_auto_setup`, `verification_clock_gate_hold_mode any` for the clock gates `compile_ultra -gate_clock`
inserted; the latch RF needs nothing special (the RTL latches and the library latches match as state points, the
RF's word ICGs are the same library cell on both sides). No `set_dont_verify`, `set_constant`, user matches or
black boxes beyond the following hierarchical rule: a block that links its children as abstractions is verified
with those children as black boxes on both sides, and each child is verified in its own run, so every cell of the
chip is covered exactly once.

| block (netlist) | children black-boxed | result | passing points | failing | unverified / aborted | unmatched (impl LATCG) | wall |
|---|---|---|---|---|---|---|---|
| pe (gm_pe) | - | **SUCCEEDED** | 530 | 0 | 0 / 0 | 4 | 25 s |
| bank512_gm, bank256_gm (latch RF) | - | **SUCCEEDED** | 16,992 / 8,544 | 0 | 0 / 0 | 2 each | 26 / 18 s |
| gm16 (gm_top, 5.25 ns) | 256 pe, 64 banks | **SUCCEEDED** | 98,086 | 0 | 0 / 0 | 1,655 | 7 min |
| lane (vu_lane, with its two banks) | - | **SUCCEEDED** | 22,746 | 0 | 0 / 0 | 177 | 82 s |
| bank256_vu | - | **SUCCEEDED** | 8,544 | 0 | 0 / 0 | 2 | 18 s |
| vu16 (vu_top, 5.25 ns) | 16 lanes, 6 banks | **SUCCEEDED** | 32,070 | 0 | 0 / 0 | 840 | 109 s |
| bank256_lw / bank512_lw / bank768_lw | - | **SUCCEEDED** | 8,544 / 16,992 / 25,440 | 0 | 0 / 0 | 2 each | 18-98 s |
| lw (lw_top) | 6 banks | **SUCCEEDED** | 37,175 | 0 | 0 / 0 | 1,073 | 262 s |
| rng9 / rng3 (rn_mkgen) | - | **SUCCEEDED** | 1,474 / 747 | 0 | 0 / 0 | 1 / 4 | 93 / 42 s |
| mm (mm_top), rng black-boxed | 2 rn_mkgen | FAILED (method) | 77,008 | 1 port | 2,571 unverified (3 h cap) | 2,715 | 3 h |
| **mm (mm_top), rng verified inside, rng .svf added** | - | **INCONCLUSIVE** | 78,038 | **0** | **3,451 unverified** (3 h cap), 0 aborted | 2,720 (+3 ref DFF1X) | 3 h 9 min |
| mm, Formality hierarchical verification | per sub-block | __MMHIER__ |
| seq (ct_seq + ct_dma) | - | **SUCCEEDED** | 8,942 | 0 | 0 / 0 | 206 | 76 s |
| axil (top_axil) | - | **SUCCEEDED** | 89 | 0 | 0 / 0 | 0 | 8 s |
| chip (top_chip) | 6 units | **SUCCEEDED** | 6,918 (6,484 black-box pins, 434 ports) | 0 | 0 / 0 | 0 | 19 s |

- "Unmatched" points are all implementation `LATCG`: the latches inside the clock gates DC inserted, which have
  no RTL counterpart; they are absorbed by the clock-gate hold mode. No reference point is unmatched.
- Rejected guidance: change_names / rename_design entries of the block-prefix renaming that the netlist reader
  resolves by name anyway; verification succeeds, so they are harmless (Formality's own note).
- **mm is not proven.** First run (rn_mkgen black-boxed): 1 failing point, `mk_req_ready[1]`; DC propagated a
  constant output of rn_mkgen NCORE 9 (`ph == PLAST` is always true for 9 cores) out of the child, which a black
  box cannot represent, so this is a limitation of the black-box method, not a design difference. Second run
  (rn_mkgen verified inside mm without its own .svf): 87 failing rn_mkgen registers, all register merges recorded
  only in the child's .svf. Third run (child .svf added): **0 failing, 78,038 passing, 3,451 unverified at the 3 h
  cap** (all in the 8 read ports `mm_rdport`: request queue `q_a` 1,770, response FIFO `u_mf/mem` 472, `q_n`
  413, `q_br` 354, `rq_lo` 180, `q_v` 64). Status: **INCONCLUSIVE, 3,451 unverified points**; not a pass.
  The 3 unmatched reference points are constant-1 registers DC removed (OPT-1206).

## 6. Integrity

### 6.1 Registers: DC vs RTL

RTL reference: Yosys after `proc` and its own unloaded / constant-register removal (tools/rtl_regs.sh; latch
counts include one latch per word ICG, which DC maps to library ICG cells). DC counts include the children.

| unit | DC flops | RTL flops (Yosys opt) | kept | DC latches | RTL latches - word ICGs | DC removals (OPT-1206 / 1207 / 1215 messages) |
|---|---|---|---|---|---|---|
| gm16 | 181,373 | 182,774 | 99.2 % | 786,432 | 811,008 - 24,576 = 786,432 | 200 / 588 / 667 |
| vu16 | 112,086 | 113,982 | 98.3 % | 311,296 | 321,024 - 9,728 = 311,296 | 347 / 44 / 22 |
| lw | 36,806 | 38,150 | 96.5 % | 98,304 | 101,376 - 3,072 = 98,304 | 492 / 457 / 38 |
| mm | 78,991 | 79,376 | 99.5 % | 0 | 0 | 31 / 256 / 0 |
| seq / axil | 6,203 / 2 | 6,214 / 2 (P8b) | 99.8 % | 0 | 0 | 132 / 2 / 7, 0 |

Latch counts are exact (words x 32). The flop differences are registers DC reports as constant, unloaded or
merged (each removal has its OPT message; Formality's guidance accepts them).

### 6.2 Logs

Every block run of the chip (gm16 and vu16 at 5.25 ns, all others at 4.0 ns):
- latches inferred only in `mm_bank_t65` (`integ_latch_other 0`);
- 0 unresolved references (LINK-5), 0 black boxes or unresolved lines in check_design;
- 0 errors apart from the harmless PROC-3, except the chip run's 9: OPT-1442 (x8) and ABS-251, both about
  placement of abstractions in a non-floorplanned topographical top (section 4.1);
- no exceptions: no false paths, multicycle paths, case analysis or disabled arcs; dont_touch only on the
  instances of compiled children.

### 6.3 Stale runs found and rebuilt

The server checkout still held P8b runs of lw and mm (before the P8c / P8c2 repairs of lw_merge, mm_top and
mm_rdport; the P8c runs had been made in a work tree that no longer exists). A Formality trial on the old mm run
exposed it (the P8c seed register missing). Both were rebuilt in P8d at 4.0 ns from the current RTL, with lw's
three bank copies; every other block's run postdates its last RTL change (checked against `git log -- hw`).

## 7. Power (PrimePower, average, tt)

Two methods, both stated per number. PrimePower (pt_shell, `power_analysis_mode averaged`) on the complete unit
netlists of section 4.2 at tt (1.0 V, 25 C), with the ss RC estimate, ideal clocks, 5.25 ns.

**(a) Vectorless, explicit toggle rates.** `power_default_toggle_rate 0.1` (10 % of the clock rate on every
primary input and black-box output), `power_default_static_probability 0.5`; PrimePower propagates activity
through the logic and the sequential cells. Clock nets toggle at the clock rate; ICG enables get probability 0.5,
so half of the latch-RF word gates open every cycle. That is far above the real latch-RF activity (one word per
bank per cycle at most), so (a) is an upper-side estimate for the RF clock power.

| unit | switching W | internal W | leakage W | total W | of which clock network |
|---|---|---|---|---|---|
| gm16 | 0.0465 | 0.2674 | 0.0510 | **0.3648** | 0.2350 (64 %) |
| vu16 | 0.0214 | 0.0632 | 0.0189 | **0.1035** | |
| lw | 0.0057 | 0.0238 | 0.0057 | **0.0352** | |
| mm | 0.0169 | 0.0305 | 0.0060 | **0.0535** | |
| seq | 0.0001 | 0.0011 | 0.0003 | 0.0016 | |
| axil | 0.0000 | 0.0000 | 0.0000 | 0.0000 | |
| **chip (sum of units; top_chip has no cells)** | **0.0907** | **0.3860** | **0.0820** | **0.559** | |

**(a2) The same, on the flat chip at the chip clock (7.75 ns, 129.0 MHz).** Here only the chip's own ports get the
default rates; every unit input is driven by the propagated activity of the unit that produces it.

| flat chip, tt, 7.75 ns, vectorless | switching W | internal W | leakage W | total W |
|---|---|---|---|---|
| u_gm | 0.0245 | 0.179 | 0.0506 | **0.254** (70 %) |
| u_vu | 0.0059 | 0.0417 | 0.0189 | **0.0665** |
| u_lw | 0.0029 | 0.0149 | 0.0057 | **0.0236** |
| u_mm | 0.0010 | 0.0086 | 0.0061 | **0.0157** |
| u_seq / u_axil | 0.0001 | 0.0007 | 0.0003 | 0.0011 |
| **chip** | **0.0345** | **0.2447** | **0.0816** | **0.361** (clock network 0.218) |

**(b) RTL SAIF from VCS (real workload, bounded).** __SAIFSECTION__

Energy per full-size step (17.3 s): __ENERGY__

## 8. Full-size training step at the chip clock

`python3 docs/calc/budget.py --fmhz 181.818` (full16, 12 layers, B 32, L 128, CLS-pruned):

| quantity | value |
|---|---|
| step cycles (GEMM / vector / LoWA overlapped) | **3,290.6 M** (GEMM 2,777.9 M, vector 506.7 M, LoWA 261.8 M, optimiser + misc 6.0 M; serial 3,552.4 M) |
| **step time at the chip clock, 5.5 ns (181.8 MHz)** | **18.1 s** overlapped, 19.5 s serial |
| step time without the mm fix (7.75 ns) / unit-level 5.25 ns / R12's 4.0 ns | 25.5 s / 17.3 s / 13.2 s |
| DRAM port | 256-bit AXI4 at the core clock = 32 B / cycle = **5.82 GB/s** peak at 181.8 MHz; the model charges GEMM at 0.993 and vector ops at 0.548 of it (P6a / P6b measured) |
| DRAM traffic per step | 95.95 GB; average demand **5.30 GB/s**, GEMM-phase peak 5.82 GB/s |

- The cycle count does not depend on the clock because the DRAM model scales with it (32 B per core cycle); a
  DRAM that delivers less than 5.8 GB/s at 182 MHz would make the GEMM phases DRAM-bound.
- **Cycle deltas of the timing repairs** (measured on the unit and e2e tests; docs/p8c_report.md,
  docs/p8c2_report.md, docs/p8f_report.md): P8c: GEMM ops +1 to +2 cycles per op, vector ops a few cycles per
  phase, e2e reduced step +3,066 (+0.030 %). P8c2: GEMM 0, vector ops +1 per op, real_smax +24,574 per forward
  softmax (writer register), real_gnorm +9; e2e reduced step -376. P8f (mm skid buffers): e2e reduced step
  -0.002 %, integration cases -2,047 ... +6 cycles. For the full step that is about +0.04 M (P8c) + 0.30 M (P8c2)
  - ~0.07 M (P8f, its e2e -0.002 % scaled to the full step) = **about +0.3 M cycles, +0.01 %** of budget.py's 3,290.6 M. (The P8c / P8c2 reports quoted
  +0.03 % / +0.24 % against a ~123 M-cycle base; against the full step it is 0.01 %.) budget.py does not model
  these per-op constants; they change the step time by less than 2 ms.
- docs/budget.md states the clock (section 1).

## 9. Limitations (honest list)

- **Pre-layout only.** No floorplan, placement, CTS or routing. Wires are DC topographical estimates per block
  (each block placed alone, as a compact region); clocks are ideal; hold is informative only. A floorplan-aware run
  of gm16 / vu16 was attempted in P8c2 and stopped in DC's placer (internal error) with the abstractions.
- **Latch register file instead of SRAM** (R11a): 49 % of the area and the main reason the die is large; P8e
  (R16) explores an SRAM-macro variant.
- **Hierarchical timing.** Units were synthesised and timed with their children as block abstractions; the
  chip-level DC timing is not usable (OPT-1442, section 4.1). Chip-level timing comes from PrimeTime on the flat
  netlist, with block-level SPEF estimates stitched per instance; inter-unit wires at chip level are not estimated
  beyond the 20 % IO budget each unit was constrained with.
- **Power** is pre-layout: no clock tree (ideal clock nets), estimated wires, __POWERLIM__
- **test-e2e-full was not re-run after the P8c / P8c2 repairs.** The repairs were checked bit-exact with lint,
  test-unit and the reduced test-e2e (P8c / P8c2 reports); the 12-layer e2e (5 h) last ran before them.
- **tsmc65-backend simulations skipped** (P8a, R12): real_ffn1_256 and one real_ln_fwd sensitivity row on the
  latch-RF backend (~10 h each in Verilator). The latch RF is covered by the --quick set, Formality of every
  bank, and the VCS power runs of section 7 (which simulate it, but do not check results).
- The mm Formality run __MMLIM__
- Units that closed 4.0 ns were not re-synthesised at 5.25 ns (area / power slightly above a 5.25 ns synthesis).

## 10. Reproduction

From `src/rtl`, with the git-ignored site files in `flow/tsmc65/site/` (templates `*.example.*`):

```
make sync65-pull                      # server checkout = pushed HEAD
make syn65-exit                       # vu16 at 5.0 / 5.25 ns, gm16 at the closed period  (log build/tsmc65/hier_exit.log)
make p8d65 P8D_PERIOD=5.25            # stages: stale (lw, mm at 4.0), chip DC, Formality of every block,
                                      #   PrimeTime per unit ss / tt / ff + PrimePower vectorless at tt  (log build/tsmc65/p8d.log)
make saif65-gm saif65-vu P8D_PERIOD=5.25 SAIF_CYC=200000   # VCS RTL SAIF of real_dgelu / real_ln_fwd
                                      #   (vectors: copy build/vec/{gemm,}/real/<case>.desc, .mem.hex to build/tsmc65/vec/{gm,vu}/)
make p8d65 P8D_PERIOD=5.25 P8D_STAGES=ptsaif               # PrimePower of gm16 / vu16 with those SAIFs
make sync65-results                   # summaries -> flow/tsmc65/results/ (git-ignored)
```

Chip-level runs (after `p8d65`), on the server in `src/rtl`:

```
export T65_KID_TAG=gm16=_p5.25,vu16=_p5.25 T65_PERIOD=5.25
FORCE=1 flow/tsmc65/syn/run_hier.sh chip                                # chip DC
flow/tsmc65/lec/run_fm.sh chip                                          # chip Formality
T65_FM_FULL_KIDS=rng9,rng3 T65_FM_TAG=_full T65_FM_TIMEOUT=05:00:00 flow/tsmc65/lec/run_fm.sh mm
T65_PT_SUFFIX=_flat T65_PT_AREA=1 flow/tsmc65/sta/run_pt_unit.sh chip ss  # flat chip PrimeTime (also tt vectorless, ff)
```

Files: `flow/tsmc65/run_p8d.sh`, `syn/run_exit.sh`, `lec/{fm_setup,fm_unit}.tcl`, `lec/run_fm.sh`,
`sta/{pt_unit.tcl,run_pt_unit.sh}`, `sim/run_saif.sh`, `tb/power/saif_ctl.sv`; summaries
`build/tsmc65/summary/{hier,fm,ptu}_*.txt` on the server.

Commits: __COMMITS__
