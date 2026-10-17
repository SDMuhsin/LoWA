# P8b report: hierarchical DC synthesis of the full16 units at 4.0 ns (TSMC 65 nm)

Phase P8b of the TSMC-65 port (PLAN R11, R12). Every full16 unit was synthesised with DC W-2024.09 in
topographical mode:
- corner ss 0.9 V 125 C (RVT), clock 4.0 ns, tsmc65 memory backend (latch register files)
- tau-03, 8 cores, one job at a time
- code taken from the server git checkout (`T65_TREE=checkout`; P8b commits e495cdd .. 662ce3c)

Netlists, .ddc files and full reports stay on the server. Only the summary text files in
`flow/tsmc65/results/hier_*.txt`, `paths_*.txt` and `pt_*.txt` come back.

**Result.**
- Met at 4.0 ns: every leaf block and every unit except gemm, vec and mem. That includes the fp units, the three
  bank shapes, gm_pe, vu_lane, rn_mkgen x2, ct_seq, top_axil and lw_top.
- Missed: **gm_top 16x16 (-2.14 ns)**, **vu_top V=16 (-1.08 ns)** and **mm_top (-0.08 ns)**. All three misses
  are on paths that cross the unit: control and data fanned out over 4-11 mm2, plus logic depth in front of the
  latch-RF read mux. Section 5 gives each path and a proposed fix. None was implemented: every fix is an RTL
  pipeline change and needs PI sign-off (R11c, R12).

## 1. Flow (flow/tsmc65, new in P8b)

| file | role |
|---|---|
| `common/hier.txt` | Block table. Columns: name, module (or `@block/instance`), full16 parameters, children, IO %, link mode. The `@unit` lines map units to blocks. |
| `common/hier.tcl` | Reader of the table (DC / PT). |
| `common/dc_lib.tcl` | Library and Milkyway setup, factored out of `syn/dc_block.tcl` and shared with it. |
| `syn/dc_unit.tcl` | One block. The steps are listed below this table. |
| `syn/run_unit.sh` | One run. Output goes to `build/tsmc65/hier/<block>/`. The summary, GNU time and integrity scan go to `build/tsmc65/summary/hier_<block>.txt`. |
| `syn/run_hier.sh` | Bottom-up batch over units / blocks. It adds the dependencies and reuses finished blocks. `FORCE=1` rebuilds the named blocks, `FORCE=all` rebuilds everything. End marker `HIER_DONE`. |
| `common/summary.tcl` | Adds `t65_unit_summary`. Memory area = cells inside `mm_sram_bank` instances. In `abs` mode it also writes the unit totals (own cells + instances x child summary). Errors in it are caught and written to the file. |
| `tools/dclog_check.sh` | Integrity scan of dc.log: Presto latch inferences inside / outside `mm_bank_t65`, LINK-5 / unresolved / black-box lines, OPT-1206/1207/1215 register removals, errors, top warning codes. |
| `tools/rtl_regs.sh` | Local Yosys count of the RTL registers per block: flops + register-array bits, and latches. It is the reference for the register check. |
| `tools/summarize_hier.sh` | One line per block. |
| `tools/dc_paths.tcl/.sh` | Reads a finished block's .ddc and gives the worst path into each endpoint group, with its delay split over the top-level instances. |
| `sta/pt_corner.tcl`, `syn/run_pt.sh` | PrimeTime tt / ff timing of a block netlist. |

What `syn/dc_unit.tcl` does for one block:
- elaborates the module with its full16 parameters, without a wrapper;
- replaces each child's generic elaboration with the child's mapped .ddc, linked as a block abstraction (`abs`)
  or as a full dont_touch netlist;
- generates the SDC, then `compile_ultra -gate_clock` plus an incremental pass, with no retiming;
- prefixes every sub-design that is not an abstraction with `<block>__`;
- writes the outputs, the summary and the block abstraction.

Makefile targets:
- `make syn65u-<unit|block> [FORCE=1|all]`
- `make syn65u-all [UNITS=...]`
- `make syn65u-status`, `make syn65u-results`
- `make sync65-results`

**Hierarchy.**
- Leaves are compiled once, as flat blocks:
  - the three bank shapes 256/512/768 x 32;
  - gm_pe, rn_mkgen NCORE 9 / 3;
  - the fp units, ct_seq and top_axil.
- vu_lane links its two banks as full dont_touch netlists.
- Every other parent links its children as **block abstractions**:
  - gm_top: 256 PEs, 32 x bank512, 32 x bank256;
  - vu_top: 16 lanes, 6 x bank256;
  - lw_top: 2 banks of each shape;
  - mm_top: the two mask generators.
- In those parents, DC compiles all remaining logic together, so paths across sub-units are optimised in one run.
  For gm_top that is the array glue, feeders, drain, epilogue, loader and ctrl.
- The parameters are the RTL defaults that top_chip uses: gm_top ROWS=COLS=NSFU=16, vu_top V=16, lw_top defaults,
  mm_top NR 8 / NM 2 / NW 4 / AXL 3 / MK_DRAM 0, ct_seq GP 36 / VP 25 / LP 10. These equal
  docs/calc/budget.py cfg full16 (R=C=V=16). The elaborated design names match the parent's instantiation, and
  `T65_CHILD` in each summary confirms the instance counts.

**Why abstractions, not full netlists with dont_touch (measured).**
- The first batch linked every child as a full mapped netlist with dont_touch. DC topographical then places the
  children's cells over the whole parent area, and the children cannot be re-buffered.
- Paths that close in the child's own run then failed in the parent:
  - vu16: -2.49 ns, inside lanes (`stf_rp -> u_b0 q_r`);
  - mm: -1.97 ns, inside rn_mkgen (`ph_reg`, two cells of 1.8 / 2.7 ns driving long estimated wires).
- With abstractions, a child's internal paths are timed in its own run, as a compact region (which is how a
  hierarchical layout places it). The parent times all interface paths: from the parent's logic into each child's
  first register, and from each child's last register out.
- This also keeps gm_top 16x16 at 9 GB.
- vu_lane keeps full banks, so that vu_top never holds the bank256 abstraction twice. It closes 4.0 ns that way.

## 2. Constraints (honesty)

- **Clock.** 4.0 ns, setup uncertainty 0.25 ns, hold 0.05 ns, transition 0.15 ns. Max transition 0.4 ns, max
  fanout 16.
- **IO delays (`sdc/gen_sdc.tcl`, documented there).**
  - Units have no wrapper. Every input and output port gets 20 % of P (0.8 ns).
    - On an input, this stands for the producer's clock-to-q plus the inter-unit route. The RTL rule is
      registered outputs.
    - On an output, it leaves P - 20 % for the unit's own output stage.
  - Bank leaves use 40 %: their address comes through the parent's control logic. This is tighter, not looser.
  - The P8a scout used input-flop wrappers. P8b does not.
- **No exceptions.** No false paths, multicycle paths, case analysis or disabled arcs.
- **dont_touch** is set only on the instances of compiled children: vu_lane's two banks, and the abstractions,
  which DC treats as fixed anyway. Each child has its own run and summary, and the unit totals add them.
  - Nothing else is dont_touch.
  - Constant inputs of a child are not propagated into it, which is pessimistic.
- **Hold.**
  - No hold fixing before CTS. P8a used `set_fix_hold`; P8b does not, so that area carries no pre-CTS hold
    buffers.
  - The ff hold columns are min analysis with the ff library and best RC, on ideal clocks. They are informative only.
- **Topographical estimates, pre-CTS.** No floorplan, so DC places each unit as a free-form region.

## 3. Results at 4.0 ns, ss

Area = total cell area in um2. For `*` rows it is the unit total: own cells + instances x child summary. Memory
= cells inside latch-RF banks (latches, write register, read mux, word ICGs). WNS / TNS are setup over all
endpoints. Latch endpoints that borrow sit at 0. "flop WNS" is the worst slack into flops and outputs, which
equals WNS whenever a block misses. Hold = endpoints with negative hold slack at ff. Wall and RSS are for one
dc_shell (GNU time).

| unit | block (module) | area um2 | of which memory | WNS / TNS (ns) | hold viol. @ff | DC wall | peak RSS |
|---|---|---|---|---|---|---|---|
| fp | fma (fp_fma S2) | 13,582 | 0 | +0.000 / 0 | 0 | 2 min | 2.3 GB |
| fp | add (fp_add S2) | 3,335 | 0 | +0.001 / 0 | 0 | 2 min | 2.3 GB |
| fp | mul2 / mul1 (fp_mul S2 / S1) | 5,968 / 6,376 | 0 | +0.000 / 0 | 0 | 2 min | 2.3 GB |
| fp | cmp / i2f (S1) | 1,178 / 1,537 | 0 | +0.028 / +0.001 | 0 | 2 min | 2.3 GB |
| fp | sfu (fp_sfu LF 2) | 24,413 | 0 | +0.000 / 0 | 34 | 2 min | 2.3 GB |
| mem leaf | bank256 / bank512 / bank768 | 60,956 / 121,710 / 182,238 | all | 0 (flop +0.008 / +0.000 / +0.000) | 0 | 2 / 4 / 5 min | 2.3-2.5 GB |
| gemm | pe (gm_pe) | 15,412 | 0 | +0.000 / 0 | 23 | 1 min | 2.3 GB |
| **gemm** | **gm16 (gm_top 16x16, NSFU 16)** | **11,161,593 \*** | **5,845,328** | **-2.136 / -34,230** (14,509 flop endpoints) | 705 | 106 min | 9.0 GB |
| vec | lane (vu_lane) | 219,407 | 121,913 | 0 (flop +0.001) | 35 | 7 min | 2.6 GB |
| **vec** | **vu16 (vu_top V=16)** | **4,345,047 \*** | **2,316,339** | **-1.082 / -4,102** (13,302 flop endpoints) | 4 | 71 min | 6.3 GB |
| lowa | lw (lw_top) | 1,307,480 \* | 729,809 | 0 / 0 | 88 | 19 min | 3.5 GB |
| rng | rng9 / rng3 (rn_mkgen 9 / 3 cores) | 141,285 / 50,332 | 0 | +0.000 / +0.006 | 2 / 14 | 5 / 3 min | 2.4 GB |
| **mem** | **mm (mm_top 8/2/4)** | **1,211,204 \*** | 0 | **-0.078 / -1.37** | 0 | 26 min | 3.6 GB |
| ctrl | seq (ct_seq + ct_dma) | 62,869 | 0 | +0.000 / 0 | 31 | 3 min | 2.4 GB |
| ctrl | axil (top_axil) | 69 | 0 | +1.712 / 0 | 0 | 2 min | 2.3 GB |

- **Sum of the chip's units.**
  - gm16 + vu16 + lw + mm + seq + axil = **18.09 mm2** of cells at ss, before top-level glue.
  - Of that, **8.89 mm2** is latch RF: 37,376 words.
- **Critical paths (start -> end, at the worst flop endpoint).**
  - fp units, pe, sfu, lane, lw: the first FMA stage (operand register or input port -> `u_fma/u_d2`), as in P8a.
    Arrival 3.74-3.77 ns, closed.
  - banks: `addr -> q_r` (read mux, 2.1-2.2 ns after the 1.6 ns input delay).
  - rng: Threefry rounds (`u_tf/g_u_*` cut registers).
  - seq: `u_dma/off -> u_dma/row_s`.
  - gm16 / vu16 / mm: see section 5.
- **tt / ff corners** (PrimeTime Q-2019.12, `results/pt_<block>_<corner>.txt`).
  - Coverage: every block with a full netlist (the fp units, banks, pe, lane, rng9/3, seq, axil).
  - Setup: the ss-compiled netlists, with the DC topographical ss parasitics, have setup slack of +1.14 to +1.47 ns
    at tt and +1.77 to +2.27 ns at ff. Latch-RF blocks borrow at 0.
  - Hold at ff: the same few violators as DC (lane 34, sfu 31, seq 29, pe 19, rng3 13, rng9 2), all at -0.0016 ns
    or better. The clock is ideal and the 0.05 ns hold uncertainty dominates.
  - Hold at tt: no violators.
  - Parents with abstractions have no complete flat netlist and were not timed in PT.

## 4. Integrity checks

**Registers: DC versus the RTL** (Yosys after `proc` with its own unloaded / constant removal; register arrays
counted as SIZE x WIDTH; latch counts without the one-per-word ICG latch). DC numbers are after compile; for
`*` units they include the children.

| block | DC flops | RTL flops (Yosys) | DC latches | RTL latches | DC removals (OPT-1206 const / 1207 unloaded / 1215 merged messages) |
|---|---|---|---|---|---|
| fma / add / mul2 / mul1 | 126 / 76 / 95 / 33 | 127 / 78 / 96 / 33 | 0 | 0 | 0/0/1, 0/0/2, 0/0/1, 0 |
| cmp / i2f / sfu | 69 / 33 / 968 | 69 / 33 / 969 | 0 | 0 | 0, 0, 0/0/1 |
| bank256 / 512 / 768 | 64 each | 64 each | 8,192 / 16,384 / 24,576 | same | 0 |
| pe | 428 | 429 | 0 | 0 | 4/1/0 |
| lane | 5,409 | 5,500 | 16,384 | 16,384 | 2/20/58 |
| gm16 \* | 171,468 | 172,869 | 786,432 | 786,432 | 200/588/667 |
| vu16 \* | 110,650 | 112,499 | 311,296 | 311,296 | 341/44/13 |
| lw \* | 36,713 | 38,057 | 98,304 | 98,304 | 492/457/38 |
| rng9 / rng3 | 1,455 / 728 | 1,529 / 761 (raw 2,165 / 971) | 0 | 0 | 17/1234/40, 0/396/33 |
| mm \* | 78,623 | 79,008 | 0 | 0 | 31/256/0 |
| seq / axil | 6,203 / 2 | 6,214 / 2 | 0 | 0 | 2/7/0 (128 bits of `lterm` are constant by design), 0 |

- Latch counts are exact in every unit: words x 32.
- gm16 holds 256 PEs x 428 flops: 256 PE instances, each one FMA (`T65_CHILD pe ... 256`).
- DC keeps 98.4-99.9 % of the RTL flops. The rest are constant, unloaded or merged registers, each reported by DC
  (OPT-1206/1207/1215).
- **Dead logic in the RTL (for the PI).** rn_mkgen has about 700 flops whose outputs never reach a port: Threefry
  output bits beyond the 16 mask bits. Yosys removes 636 of them too.

**Logs** (all P8b runs):
- Latches are inferred only in `mm_bank_t65` (`integ_latch_other 0` in every block).
- 0 unresolved references (LINK-5).
- 0 black boxes or unresolved lines in check_design.
- 0 errors apart from the harmless PROC-3.

**Capacity versus docs/memory_map.md.**
- The RTL's latch RF is 37,376 words (146 KiB):
  - gm_top 24,576: x feeder 8,192, w feeder 8,192, drain 8,192;
  - vu_top 9,728: lane banks 8,192 + 6 x 256;
  - lw_top 3,072.
- memory_map.md and R11a quote 89 KiB / 22,784 words. The differences:
  - the map's ACC is 4,096 words, against 8,192 for the drain;
  - the vector-unit banks are not in the map;
  - lw_merge has 1,024 words more.
- The map, not the RTL, looks out of date. This is for the PI to confirm. Nothing was shrunk.

## 5. Misses at 4.0 ns: paths and proposed fixes (none implemented; RTL changes need PI sign-off)

**gm16, WNS -2.136 ns** (arrival 5.86 ns of 3.75 ns allowed).
- Worst path: `u_loader/tf_cp` (the tile-FIFO read pointer) -> `tf[tf_cp]` -> x / w feeder write enable / address
  OR-reduction -> bank address mux (`wen ? waddr : raddr`) -> 512:1 latch read mux -> `q_r`.
- Delay split: loader 1.52 ns (26 cells), feeder 1.71 ns (20 cells), bank read mux 2.38 ns (18 cells; 2.15 ns in
  the bank's own run).
- The read path depends on the *write* decision, because the bank port is shared through `wen ? waddr : raddr`.
- The same 2.1 ns class of violations sits on these endpoint groups (by total TNS):
  - PE drain shift `u_pe/dr_s`;
  - feeder bank `q_r` / `wd_r` (read and write);
  - x feeder row registers `a_s`;
  - drain `acc_old_s` / `g_add`;
  - epilogue `acc2`.
  (Grouped from the run's violator report. `tools/dc_paths` could not re-time gm16 after reloading the .ddc: in a
  fresh session most abstraction pins do not match, so its gm16 output is not used.)
- Proposed fix (RTL, PI):
  1. Register the loader's write request (valid / address / data) at each feeder input. The bank port then sees a
     register plus one mux level. Write latency +1 is harmless for the loader's tile copy, and the
     write / read conflict check moves with it.
  2. If still short: register the bank read address one stage earlier in gm_ctrl, with read latency +1 and the
     array skew / issue offsets moved by one cycle.
  3. For the broadcast groups: add one pipeline register per row / column group on the drain-shift and feeder
     -> array buses. The systolic skew already tolerates fixed extra latency per row and column.
- Each change needs an L2 re-run. Without them, gm16 needs about 6.2 ns, so even the R12 fallback (5.0 ns)
  would not close gm16.

**vu16, WNS -1.082 ns.**
- Worst groups, by TNS:
  1. Lane slot registers `u_lane/r3` (written from the shared constant-bank output `cv1_d`, broadcast to 16 lanes)
     and lane bank `q_r` (issue address from u_seq).
  2. Loader hold buffers `u_ld/g_hold_*/f_d`: the 256-bit DRAM data `rd0_dat` is written into 16 lanes' buffers
     (-1.0 ns). `results/paths_vu16.txt` gives the worst path: `u_ld/b0_r` (burst control register) ->
     write-select decode and fan-out to 16 x 256 bits -> `f_d`, 57 points, arrival 4.71 ns.
  3. Writer `u_wr/g_data`.
  4. u_seq: `xl_sel -> 16:1 cross-lane operand mux -> u_xfma` stage 1 (-1.08 ns).
  5. u_seq: `sl_busy` / `sl_cool` from lane `sfu_out_cnt`. That lane output is combinational
     (`sfo_wp - sfo_rp`), against the registered-output rule.
- Proposed fix (RTL, PI):
  1. Register `rd0_dat`, the hold-buffer write selects and the constant-bank read data once per group of 4 lanes
     (one broadcast stage).
  2. Register lane `sfu_out_cnt` and `acc_cnt`, and adjust the u_seq credit logic by one cycle.
  3. Register the selected cross-lane operand before u_xfma (the cross-lane reduction gets +1 latency).
- Without them, vu16 would close at about 5.0 ns (arrival 4.83 ns plus uncertainty).

**mm, WNS -0.078 ns.**
- `mk_seed[*]` (input port, 0.8 ns) -> rn_ks key schedule -> `u_mk/i2b_q`, `i3b_q`.
- `rd_req_addr -> rd_req_ready`: a combinational port-to-port path (read-port arbitration), so it uses both
  0.8 ns IO budgets.
- Proposed fix: register `mk_seed` at the mm_top input. The seed is a per-step CSR value, so one cycle of latency
  is invisible if it is written before the step starts. Whether `rd_req_ready` needs a skid register depends on
  the real top-level driver; P8c will time it with its actual source instead of the 20 % budget.

## 6. P8c: top-level script (prepared, not run) and estimate

- The `chip` line of `common/hier.txt` (top_chip, children gm16 vu16 lw mm seq axil, all `abs`) is the P8c run:
  `make syn65u-chip`, or `flow/tsmc65/syn/run_hier.sh chip` (it reuses every finished unit).
- **Estimate.** In `abs` mode, memory scales with the parent's own cells plus the retained interface logic:
  - vu16: 120k own cells and 22 abstractions -> 6.3 GB, 71 min;
  - gm16: 356k own cells and 320 abstractions -> 9.0 GB, 106 min.
  top_chip has a few thousand glue cells plus six unit abstractions, which carry their interface logic only.
  Estimate **4-8 GB and 30-60 min** on 8 cores, within the 24 GB limit.
- **One blocker to fix first (not a design change).**
  - Abstraction designs keep their names, because DC cannot rename an abstraction (UIMG-48).
  - So gm16, vu16 and lw each contain abstractions named `mm_sram_bank_WORDS256_WIDTH32_AW8` (and gm16 / lw also
    `..._WORDS512_...`), and top_chip would read the same design name from three .ddc files.
  - Fix in the flow: per-parent leaf copies whose own run renames the top design to `<block>__<design>` before
    the abstraction is written (e.g. bank256_gm, bank256_vu, bank256_lw), linked with `change_link` in the
    parent. The banks are a few minutes each; the parents (gm16 106 min, vu16 71 min, lw, mm) must be rerun.
- After that, P8c adds Formality and pre-layout PT / PP. The RTL fixes of section 5, if signed off, come first,
  and with them the L2 regressions.
