# P8e report: SRAM-macro variant of the TSMC 65 nm synthesis numbers (PLAN R16)

P8e asks one question: what would the on-chip memory cost if the 146 KiB of `mm_sram_bank` storage were compiled
SRAM / register-file macros instead of the latch register file (the buildable baseline of P8a-P8d)?  The kit's ARM
memory compilers turn out to work for front-end views once three host problems are worked around (section 2).  So
this report uses compiled macros (liberty timing / power, vendor Verilog models, footprints), not the CACTI
fallback.  The kit has no GDS or LVS for these macros, so this variant can be synthesised but **not taped out**.
The latch RF stays the design of record.

Everything kit-derived (liberty, .db, Verilog models, footprints, netlists, full reports) stays on the server under
`~/lowa_tsmc65/sram/`.  The compiler names, the kit path and the macro instance names appear only in git-ignored
site files.  This report names the compilers generically: **RF** is the single-port register-file compiler,
**HD-SRAM** the single-port high-density SRAM compiler, and **RF-2P** the two-port register-file compiler.

## 1. Method

| step | what | where |
|---|---|---|
| views | compiler run headless with a system Java 8 (datatable, liberty for 4 corners, Verilog model, footprint); liberty libraries renamed per file and compiled to .db with Library Compiler | `flow/tsmc65/sram/mc_gen.sh`, `lib2db.tcl`; site `site/sram_site.sh` |
| backend | `mm_sram_bank` TECH = 3 `tsmc65_sram` = `mm_bank_t65s`: one macro per bank plus a q-hold adapter | `hw/mem/tech/mm_bank_t65s.v`, stand-in `hw/mem/tech/mm_sram_macro.v`; site wrapper `site/mm_sram_macro_site.v` |
| contract | `tb/mem/tb_mm_sram_bank.sv` (hold check on) with the vendor models in VCS, every shape | `flow/tsmc65/sram/sim_banks.sh` |
| DC | `syn/dc_unit.tcl` evaluated unchanged through a wrapper that only swaps the backend define, the RTL file of the macro and the link libraries; the macros are black boxes linked from their .db; the latch baseline runs through the same wrapper (`T65_VARIANT=latch`) | `sram/dc_sram.tcl`, `sram_lib.tcl`, `run_sram.sh` |
| PT / PP | `sta/pt_unit.tcl` evaluated unchanged with the macro .db in the link path; PrimePower vectorless at tt (toggle 0.1 / cycle, static probability 0.5), as P8d does | `sram/pt_sram.tcl`, `run_pt_sram.sh` |
| macro power | leakage and per-access energy from the liberty, and from the compiler datatable | `sram/mc_power.py` |

No P8d file was edited.  The DC and PT wrappers check that the scripts they evaluate still have the expected shape
(five / three `source` lines, one final `exit`) and stop otherwise.  Clock: **5.25 ns at ss 0.9 V 125 C**, the
chip clock P8d settled under the R15 exit rule (its batch runs `run_p8d.sh 5.25`; `docs/p8_report.md` was not yet on
main).  The banks were first run at the provisional 5.0 ns and then rerun at 5.25 ns.  The 5.0 ns numbers are in
section 5.1.

### 1.1 Port contract and adapter

The compiled macros have active-low chip and write enables, a 3-bit margin-adjust input (tied to 000, the fastest
legal setting at every corner; the datatable marks illegal settings as 999 and 000 is legal) and a retention pin
(tied inactive).  They are **write-through**: q shows d after a write.  The RF and HD-SRAM compilers both reject
`write_thru off` ("should be one of: on").  The `mm_sram_bank` contract says writes never change q, so
`mm_bank_t65s` adds a 1-bit `rd_last` flag and a WIDTH-bit hold register: on a write that follows a read, the
current q is captured and q switches to the register until the next read.  Read latency (1), address / data
timing and the one-access-per-cycle rule are unchanged.  Cost: 33 flops and a 32-bit 2:1 mux, 350 um2 per bank.

## 2. Compiler attempts

| # | attempt | result |
|---|---|---|
| 1 | stock launcher (ksh script, bundled 32-bit JRE) | `Error: could not find libjava.so` / `could not find Java 2 Runtime Environment` |
| 2 | system Java 8 (`/usr/bin/java` 1.8.0) running the generator's main class headless, base dir = kit tree on NFS | the Java front end starts and parses options, then each view generator fails: `ERROR: could not stat(<kit>/lib/<compiler>_dtgen)`.  Cause: the 32-bit helpers use 32-bit `stat()`, and the kit files on the NFS have inode numbers above 2^32 (e.g. 6.46e9), so `stat` fails with EOVERFLOW |
| 3 | compiler dir copied to local `/tmp` (inode numbers ~1.9e7), system Java | datatable, Verilog, liberty: **succeed**.  Footprint fails: it calls `<java.home>/../jre_tiling/bin/java` |
| 4 | system JRE copied to local scratch, with `jre_tiling/bin/java` linked to it | footprint (LEF-like `.vclef` + antenna file): **succeeds**.  All needed views now generate in about 30 s per macro |
| 5 | `-corners` with 3 corners | the compiler requires exactly 4: ss 0.9 V 125 C, tt 1.0 V 25 C, ff 1.1 V -40 C, ff 1.1 V 125 C |
| 6 | Library Compiler (2019.12) on the liberty | 12 of 12 .db files, 0 errors.  Every liberty names its library `USERLIB`, so `mc_gen.sh` renames each one to `<inst>_<corner>_syn` first; otherwise several macros and corners cannot be linked together |

Total time to working views: about 1 h.  The `/tmp` staging (compiler copies + JRE, ~0.6 GB) is removed with
`mc_gen.sh clean`.  The server-side views take 35 MB.

## 3. Macros

Sweep (`mc_gen.sh sweep`).  ss values; macro area includes the compiler's default power ring; core = bit-cell array
plus periphery without the ring.  Every candidate cycles far below the 4.0-5.25 ns window, so the choice is the
smallest area.

| shape | compiler / mux | core um2 | macro um2 | access ns (ss, margin 000) | min cycle ns (ss) | note |
|---|---|---|---|---|---|---|
| 256 x 32 | RF mux 2 | 13,829 | **20,508** | 1.001 | 0.925 | **chosen** |
| 256 x 32 | RF mux 4 | 15,399 | 23,200 | 0.994 | 0.866 | |
| 256 x 32 | HD-SRAM mux 8 | 21,258 | 46,840 | 1.009 | 1.442 | |
| 256 x 32 | RF-2P mux 2 | 22,736 | 31,523 | - | - | two-port, not needed |
| 512 x 32 | RF mux 4 | 22,082 | **30,743** | 1.076 | 0.958 | **chosen** |
| 512 x 32 | HD-SRAM mux 8 | 26,892 | 53,488 | 1.015 | 1.450 | |
| 512 x 32 | HD-SRAM mux 16 | 39,108 | 80,172 | 1.126 | 1.609 | |
| 768 x 32 | HD-SRAM mux 8 | 32,782 | **60,438** | 1.022 | 1.458 | **chosen** (RF stops at 512 words) |
| 768 x 32 | HD-SRAM mux 16 | 44,761 | 86,379 | 1.130 | 1.614 | |

RF word ranges: 8-128 (mux 1), 16-256 (mux 2), 32-512 (mux 4).  HD-SRAM muxes: 8 / 16 / 32, 512+ words at mux 16.
A 768-word bank could also be RF 512 + RF 256 (51,251 um2, 15 % smaller) with an address-split adapter.  Only two
banks have this shape, so it was not done.

Power per macro (`mc_power.py`, 200 MHz):

| shape | leakage tt / ss, mW | read / write energy tt, pJ (liberty) | read / write energy tt, pJ (datatable) |
|---|---|---|---|
| 256 x 32 RF | 0.050 / 0.091 | 6.05 / 8.62 | 7.29 / 10.38 |
| 512 x 32 RF | 0.074 / 0.120 | 8.41 / 11.00 | 9.73 / 12.88 |
| 768 x 32 HD-SRAM | 0.181 / 0.404 | **0 / 0** (see below) | 8.39 / 12.30 |

The HD-SRAM liberty carries **zero clock-pin internal energy**; its only non-zero internal power is on the write
enable.  DC and PrimePower therefore under-report that macro's dynamic power.  For the two 768-word banks, the
datatable value (read / write AC currents of core + periphery, mA/MHz x V) is used as a correction.  For the RF
macros the datatable value is 1.2x the liberty value, which validates the unit reading.

## 4. Contract tests

| test | where | result |
|---|---|---|
| `tb_mm_sram_bank`, vendor full model, timing checks on, path delays zeroed (`+delay_mode_zero`) | server VCS, 256 / 512 / 768 x 32, 100,000 ops each | **PASS** 3 / 3, 0 timing-check violations |
| `tb_mm_sram_bank`, vendor unit-delay model | server VCS, same shapes | **PASS** 3 / 3 |
| same, vendor full model with its default 1.000 ns clock-to-Q path delay | server VCS | FAIL (about 22.7 % mismatches): the testbench samples q 1 ns after the edge, exactly the model's output delay, so it reads the previous value.  This is a sampling artefact of the shared testbench, which assumes zero-delay models; the zero-delay and unit-delay runs above are the valid checks |
| adapter mutation (q = macro q, no hold) with the stand-in | local Icarus | FAIL 9,461 / 20,000, as expected: the hold check catches write-through |
| `make lint` | local | PASS (57 files; `mm_sram_bank -GTECH=3` added to the backend sweep) |
| `make test-mem-banks` | local, 4 backends x 3 shapes x Icarus / Verilator | PASS 24 / 24 |
| `tb/mem/run_tests.py --quick --tech tsmc65_sram` | local, stand-in macro | PASS (TEST-MEM [tsmc65_sram]) |

## 5. DC results at 5.25 ns, ss (banks)

Both variants come from the same wrapper at the same clock and corner (DC topographical, io 40 %).  DC builds the
macros' physical cells from the logical library (OPT-1413: no physical library cell for the macro; size from the
.db area).  Power: PrimePower vectorless at tt on the DC netlists (sram/run_pt_sram.sh), the P8d method.

| bank | variant | area um2 (macro + logic) | setup WNS ns | hold WNS ns | PP tt total mW | of which leakage mW |
|---|---|---|---|---|---|---|
| 256 x 32 | latch RF | 60,758 | 0.000 (met) | 0.113 | 0.851 | 0.243 |
| 256 x 32 | macro | **20,858** (20,508 + 350) | +1.677 | 0.138 | 0.815 | 0.052 |
| 512 x 32 | latch RF | 121,159 | 0.000 (met) | 0.137 | 1.581 | 0.484 |
| 512 x 32 | macro | **31,093** (30,743 + 350) | +1.603 | 0.142 | 1.067 | 0.076 |
| 768 x 32 | latch RF | 181,670 | 0.000 (met) | 0.129 | 2.336 | 0.729 |
| 768 x 32 | macro | **60,788** (60,438 + 350) | +1.636 | 0.141 | 0.454 (about 1.19 with the datatable correction) | 0.183 |

A latch-RF slack of 0.000 means "met": DC stops optimising at zero.  Macro banks: the worst path is
macro clock -> q output (1.22 ns arrival at ss, of which ~1.0 ns is the macro access time, the rest the adapter mux
and output buffering), and the worst input path (address / data / enables into the macro) has +2.4 ns.  The
macros' own minimum cycle (0.93-1.46 ns at ss) is far below 4.0 ns, so at bank level the macro variant also meets
the original 4.0 ns target.  The per-parent copies (`bank*_gm`, `_vu`, `_lw`) give identical numbers.

### 5.1 At 5.0 ns (first run)

The macro banks had +1.53 / +1.45 / +1.48 ns, with identical areas.  The latch RF had 60,740 / 121,081 / 181,593 um2,
all met.  PP tt total: macro 0.853 / 1.116 / 0.467 mW, latch 0.877 / 1.630 / 2.410 mW.

## 6. On-chip memory roll-up (108 banks)

The RTL instantiates 108 banks: 72 x 256, 34 x 512 and 2 x 768 words (gm16 64 = 32 WBUF x 256 + 16 XBUF x 512 +
16 ACC x 512; vu16 38 x 256; lw 2 x 768 + 2 x 256 + 2 x 512), 37,376 words in total.  memory_map.md section 3
prints "106 banks" in its total row, but its own rows sum to 108.  Bank-level numbers at 5.25 ns (ss area, tt
vectorless power) times the bank counts:

| | latch RF | SRAM macros | macro / latch |
|---|---|---|---|
| memory area mm2 | 8.857 | **2.681** (macros 2.643 + adapters 0.038) | 0.30 |
| memory power, PP tt vectorless, mW | 119.7 | 95.9 (97.3 with the HD-SRAM correction) | 0.80 |
| of which leakage, mW | 35.4 | 6.7 | 0.19 |
| bank setup slack at ss, ns | 0.000 (met) | >= +1.60 | |

### 6.1 Units at 5.25 ns, ss (DC topographical; banks linked as block abstractions, as in P8d)

Unit area = own cells + children (`unit_area_um2` of the summary); memory = the units' mm_sram_bank instances.
Latch lw was rerun through the same wrapper; latch vu16 / gm16 are P8d's 5.25 ns unit runs (`hier_<unit>_p5.25`,
their bank / lane / PE children are the 4.0 ns runs).  The macro variant's children are all compiled at 5.25 ns.

| unit | variant | unit area mm2 | memory mm2 | logic mm2 | setup WNS ns | hold WNS ns (pre-layout) | source |
|---|---|---|---|---|---|---|---|
| lw | latch RF | 1.281 | 0.727 | 0.554 | +0.0007 | -0.0013 | DC, this phase |
| lw | macro | **0.771** | 0.225 | 0.545 | +0.0016 | -0.090 | DC, this phase |
| vu16 | latch RF | 4.298 | 2.316 | 1.982 | 0.000 | -0.0011 | P8d |
| vu16 | macro | **2.691** | 0.793 | 1.898 | **-0.510** (TNS -178, 533 endpoints) | -0.0084 | DC, this phase |
| gm16 | latch RF | 11.167 | 5.845 | 5.321 | 0.000 | -0.0055 | P8d |
| gm16 | macro | (running) estimate **6.984** | 1.662 | 5.321 | - | - | estimate: latch logic + 32 x 256 + 32 x 512 macro banks |

vu16 macro variant: it misses 5.25 ns by 0.51 ns, while P8d's latch run meets it.  The violating groups are
mostly the unit-spanning paths that P8c / P8c2 already saw in vu16 at earlier clocks:
- lane output buffer -> writer `u_wr/lane_ob_reg` (237 endpoints, -0.51 ns; 4.16 ns in the lane, 1.33 in the writer);
- lane `sfo_cnt` -> sequencer `u_seq` state (about 140 endpoints, -0.51 ns, 4.84 ns of top-level wire).
P8d's own vu16 runs moved between -1.11 ns (5.0) and met (5.25) on these paths, so DC-topographical run-to-run
placement spread is of this size.  One group is caused by the macros: column-vector bank -> broadcast register
`b_cv*_reg` (24 endpoints, -0.46 ns).  4.45 ns of that path is inside the bank abstraction: the macro output and
adapter mux drive the 16-lane broadcast, and as a dont_touch child the bank cannot be re-buffered by the parent.
The latch RF's q register has the same structure but a flop clock-to-q instead of a ~1.0 ns macro access.  DC power
(ss, default vectorless): 120.6 mW, leakage 16.9 mW.

The gm16 estimate (labelled) assumes the units' non-memory logic is unchanged.  lw supports this: its logic is
0.554 mm2 with latch banks and 0.545 mm2 with macros.  DC power of lw at ss (DC default vectorless, both variants
through the same wrapper): latch 26.5 mW (leakage 8.0), macro 22.8 mW (leakage 5.0).  The lw macro variant has more
pre-layout hold violations (456 endpoints, worst -0.09 ns, total -10.8 ns).  The summary does not name the paths; the likely source (not verified) is the
macros' input hold requirements at the ff corner.  Pre-layout hold is left to layout in both variants, as in P8d.


## 7. Limitations

- **Not buildable:** the compilers list GDS and LVS views as not available.  This variant is a what-if for area,
  timing and power only; the latch RF remains the design of record.
- Macro physical size comes from the liberty area (footprint `.vclef` agrees: e.g. 377.7 x 141.6 um = 53,488 um2 for
  an HD-SRAM 512 x 32).  DC topographical places the macros as cells sized from the .db.  There is no floorplan and
  no macro placement / pin-access or power-ring overhead beyond the compiler's own ring.  Unit-level wire estimates
  therefore do not see macro blockages.
- Margin-adjust pins are tied to 000 (fastest legal).  DC times the worst arc over all margin settings (it does not
  use `when` conditions for delay), which is conservative.
- Power uses vectorless activity: the macros' enables get static probability 0.5, i.e. an access every other cycle,
  while latch-RF nets toggle at 0.1 per cycle.  Absolute power is a method artefact for both variants; the
  ratios are what to read.  The HD-SRAM liberty lacks clock energy (section 3).
- The datatable energy units (mA/MHz) are inferred, cross-checked against the RF liberty energies (ratio 1.2).
