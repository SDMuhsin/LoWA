# P9c report: placement and clock tree (ICC2) for LoWA-TA on TSMC 65 nm

P9c set out to replace the "pre-layout, no clock tree" caveat on P8's chip timing (5.5 ns at ss) and power
(0.475 W vectorless at tt) with post-placement / post-CTS numbers. The full chip could not be placed on the available
server (section 2). Three of the six units were measured instead: lw, mm and vu16, which hold 1.90 M of the chip's 5.19 M
leaf cells and 6.8 of its 18.04 mm2. Each was taken through floorplan, timing-driven placement, clock tree synthesis
with post-CTS optimisation, and global routing in ICC2. The results were then timed in PrimeTime and analysed in
PrimePower, with the same method as each unit's pre-layout P8 run. The chip-level figures below that use them are
extrapolations and are labelled as such.

All tool runs were on tau-02 (ICC2 / PrimeTime / PrimePower Q-2019.12-SP2), from the pushed p9c branch. The run
directories, netlists, parasitics and full reports stay on the server; only scripts and summary numbers are here.

## 1. Summary

| item | result | label |
|---|---|---|
| scope | lw, mm (P8f), vu16 placed + CTS + globally routed: **1.90 M cells (37 % of the chip), 6.8 mm2 of 18.04**; gm16 (3.27 M), seq, axil not placed | measured |
| why not the chip / gm16 | ICC2 place_opt needs ~12-13 KB per cell (vu16: 15.6 GB for 1.22 M cells): chip ~65 GB, gm16 ~40 GB > the 30 GB shared host | measured scaling |
| clock tree (ICC2, ss / ff) | lw 135 k sinks, 11 levels, 8,234 repeaters, latency 1.20 / 0.52 ns, global skew 0.59 / 0.24 ns; mm 84 k sinks, 4,672 repeaters, 1.24 / 0.55, 0.67 / 0.30; vu16 423 k sinks, 14 levels, 25,479 repeaters (61,357 um2), 1.22 / 0.56, 0.38 / 0.17 | post-CTS |
| setup, ss, 5.5 ns, register to register (sign-off SDC, 0.25 ns uncertainty) | vu16 **-0.392 ns** (3,377 endpoints), mm **-0.286 ns** (529), with global-route parasitics; lw -0.101 ns (placement RC estimate; its global route is congested, section 4.4) | post-CTS, global route, no post-route optimisation |
| period that closes at ss (PT sweep, same SDC scaled) | vu16 **5.9 ns**, mm 5.8 ns, lw 5.7 ns (estimate); with ICC2's placement estimate instead of the global route: vu16 5.7 ns | post-CTS |
| hold | register to register: clean at ss and ff in all three units (worst +0.005 ns, vu16 ff) | post-CTS |
| tt power, vectorless, vs the same unit pre-layout | vu16 116.4 / 99.7 mW (**+17 %**), mm 57.5 / 44.1 (**+30 %**), lw 42.8 / 33.9 (**+26 %**); clock network +47 % / +47 % / +43 %; the added clock power per sink is 58-68 nW | post-CTS, global route |
| chip, extrapolated | power **about 0.56-0.59 W** vectorless at tt (P8: 0.475 W pre-layout); post-CTS ss period **about 5.9 ns (169 MHz)** before post-route optimisation; gm16 and the cross-unit paths not measured | extrapolation |
| core area at 0.65 target utilisation | lw 2.00 mm2 (1.41 mm square), mm 1.87 mm2, vu16 6.61 mm2 (2.57 mm square); chip at the same density 18.04 / 0.65 = 27.8 mm2 of core, more if the latch-RF banks need lower density (section 4.4) | measured per unit / arithmetic |

## 2. Feasibility and decision (step 1)

- lw (346,958 cells) ran first as the probe. On 12 threads, on a host shared with an unrelated DC job, floorplan took
  63 s / 2.2 GB and place_opt took 25 min / 4.1 GB. vu16 (1.22 M cells) then needed 15.6 GB for place_opt and 13.0 GB
  for clock_opt.
- ICC2 memory grows about linearly with cell count, at about 12-13 KB per cell in place_opt. The flat chip (5.19 M
  cells) would need about 65 GB and gm16 (3.27 M) about 40 GB, before clock_opt. The host has 30 GB, shared with others.
- **Decision (docs/p9c_status.md, 20:20 UTC):**
  - (a), the full chip placed with CTS, is not achievable.
  - (b), gm16 flat, is not achievable either, for memory reasons.
  - Instead, three units were measured: lw (latch-RF banks of 256-768 words plus AdamW / merge logic), mm (flops only,
    P8f netlist) and vu16 (16 FMA lanes plus 256-word latch-RF banks).
  - vu16 is the structural analogue of gm16, which is FMA PEs plus 256 / 512-word latch-RF banks. Its post-/pre-layout
    factors are applied to gm16, labelled as an extrapolation.
- Hierarchical place and route of the chip, with gm16 as a block placed once and reused, would be the way to measure the
  chip on this host. It needs block abstracts and a chip floorplan, and was not attempted in the time available.

## 3. Method

Flow: `flow/tsmc65/pnr/` (new; adapted from the template flow that P8 used for synthesis and signoff), driven by
`run_pnr.sh <block> [first] [last]`. Each stage is one icc2_shell session that saves a snapshot.

| stage | what |
|---|---|
| ndm | reference library: cell LEF + ss / tt / ff timing libraries on the techfile |
| floorplan | the unit's DC netlists (the unit plus every block below it: banks, lanes, mask generators), linked flat; square core at 0.65 target utilisation, 12 um core offset; pins on the four sides (two mid metals); power ring + 2 um / 40 um mesh on the two top metals + cell rails |
| place | place_opt (timing-driven; clock gates kept, not merged) |
| cts | clock_opt build_clock + final_opto: CTS with the clock buffer / inverter cells only, 0.10 ns skew target, double-spacing clock routing rule on the mid metals, post-CTS setup optimisation at ss and hold fixing at ff (high effort) |
| holdfix | (vu16, mm) final_opto again with hold also enabled at ss (section 4.3; it changed nothing material) |
| groute | route_global of all nets (clock nets included, with their rule); no track or detail routing, no post-route optimisation; netlist + ICC2 SPEF per corner written |
| cts_out | (alternative) netlist + ICC2 SPEF of the post-CTS database before global routing, i.e. ICC2's placement-based RC estimate that place_opt / clock_opt optimised against |

- Constraints: the synthesis SDC (`sdc/gen_sdc.tcl`) at 5.5 ns for ss and ff. It has 0.25 ns setup / 0.05 ns hold
  uncertainty, 20 % IO delays and a 0.5 ns max transition. ICC2 adds +0.10 ns setup / +0.02 ns hold margin and a 0.25 ns
  clock transition target. MCMM: func_ss (setup, cworst RC), func_ff (hold, cbest RC).
- PrimeTime / PrimePower: `sta/run_pt_unit.sh` with `T65_POST_DIR` (new post-layout mode of `pt_unit.tcl`). It reads the
  ICC2 netlist, which includes the clock tree, optimisation and hold buffers, plus the ICC2 SPEF. It uses the sign-off
  SDC unchanged (0.25 ns setup uncertainty) and **propagated clocks**.
  - ss / tt use the cworst SPEF, as in P8, where tt reused the ss RC; ff uses the cbest SPEF.
  - Power is vectorless as in P8: toggle rate 0.1 on primary inputs, static probability 0.5, averaged.
- Reference: the same unit's pre-layout run at 5.5 ns, with the same script, constraints and activity settings
  (`T65_PT_SUFFIX=_p5.5_pre`). That run uses DC topographical SPEF per block and ideal clocks.
- Labels: "post-CTS, global-route parasitics" = groute SPEF; "post-CTS, placement RC estimate" = cts_out SPEF.
  Neither is detail-routed or sign-off extracted.

## 4. Results per unit

### 4.1 Implementation (ICC2)

| unit | cells pre -> post-CTS | cell area mm2 pre -> post | core (0.65 target) | utilisation post | place_opt | clock_opt | route_global | GRC overflow after global route |
|---|---|---|---|---|---|---|---|---|
| lw | 346,958 -> 358,062 | 1.300 -> 1.327 | 1,414 x 1,413 um = 2.00 mm2 | 0.664 | 25 min / 4.1 GB (12 thr) | 1 h 57 min / 4.9 GB (6 thr) | 57 min / 4.5 GB | **19.4 %** (place-stage estimate 0.29 %) |
| mm (P8f) | 334,912 -> 344,180 | 1.216 -> 1.264 | 1,368 x 1,366 um = 1.87 mm2 | 0.676 | 31.5 min / 4.4 GB (4 thr) | 41 min / 4.7 GB (6 thr) | 3 min / 3.9 GB | 0.38 % |
| vu16 | 1,221,897 -> 1,218,948 | 4.298 -> 4.256 | 2,571 x 2,570 um = 6.61 mm2 | 0.644 | 1 h 56 min / 15.6 GB (8 thr) | 2 h 48 min / 13.0 GB (8 thr) | 7 min / 12.0 GB | 0.36 % |

- Cells grow by the clock tree, high-fanout buffering and hold buffers (lw +3.2 %, mm +2.8 %). vu16 shrinks slightly
  because ICC2 downsizes cells that DC sized against its own wire model.
- Pre-layout area is DC's cell area; post is ICC2's leaf-cell area without fillers.

### 4.2 Clock tree (ICC2 report_clock_qor; sinks = flops + latches, ICGs are internal tree cells)

| unit | sinks | levels | repeaters (area um2) | clock std-cell area um2 (incl. ICGs) | max latency ss / ff (ns) | global skew ss / ff (ns) | clock transition violations at ss |
|---|---|---|---|---|---|---|---|
| lw | 135,110 | 11 | 8,234 (32,494) | 59,431 | 1.20 / 0.52 | 0.59 / 0.24 | 717 |
| mm | 84,065 | 11 | 4,672 (13,811) | 32,635 | 1.24 / 0.55 | 0.67 / 0.30 | 114 |
| vu16 | 423,382 | 14 | 25,479 (61,357) | 148,267 | 1.22 / 0.56 | 0.38 / 0.17 | 253 |

The 0.10 ns skew target is met at neither corner. The skew comes mainly from the clock-gate branches: the ICGs sit at
different depths, and the word-gated latch subtrees hang under ICGs that DC inserted. The skew is in the propagated
clocks of every timing number below.

### 4.3 Timing (PrimeTime, propagated clocks, 5.5 ns, sign-off SDC with 0.25 ns setup uncertainty)

Register to register (r2r) and IO are reported separately. A unit's IO delays are referenced to the ideal clock at the
port, while its internal clock arrives after the tree's ~1.2 ns insertion delay. ICC2 adjusts IO latencies during CTS;
PrimeTime does not. Input-to-register hold at ss therefore shows violations (mm -0.21 ns on 385 skid-buffer inputs, lw
-0.028 ns) that do not exist on chip, where the driver is another unit's flop on the same clock tree. Those violations
are excluded below.

| unit | parasitics | r2r setup ss WNS [endpoints] | worst path | closes at (ss sweep) | r2r hold ss / ff | pre-layout ss at 5.5 ns (flop WNS) |
|---|---|---|---|---|---|---|
| vu16 | global route | **-0.392 [3,377]** | lane 14 SFU FMA (ia_reg -> u_fma divider stage) | **5.9 ns** | +0.069 / +0.005 | +0.227 |
| vu16 | placement estimate | -0.135 [754] | lane 7 FMA (opb_reg -> u_fma) | 5.7 ns | | |
| mm | global route | **-0.286 [529]** | mask generator core (rn_mkgen u_tf) | **5.8 ns** | +0.096 / +0.009 | +0.172 |
| mm | placement estimate | -0.103 [286] | mask generator core | | | |
| lw | placement estimate | -0.101 [198] | AdamW denominator (u_ad/u_den) | 5.7 ns | +0.099 / +0.009 | +1.179 |
| lw | global route (congested) | -4.02 [1,731] | latch-RF read mux (u_sg banks) | not usable | | |

- IO paths, setup: positive in every unit at ss (in-to-reg +0.88 to +1.22 ns, reg-to-out +0.61 to +0.69 ns).
- With the 0.25 ns uncertainty, which P8 set for "jitter + pre-CTS skew", the skew is counted twice once clocks are
  propagated. With a post-CTS uncertainty of 0.10 ns, every setup WNS above improves by exactly 0.15 ns: vu16 -0.242,
  mm -0.136 (global route).
- No post-route optimisation (route_opt) was run, since detail routing was out of scope. In a complete flow it acts on
  exactly these global-route paths. The global-route numbers are therefore an upper bound on the loss, and the
  placement-estimate numbers are what the optimiser saw.
- tt: r2r setup has positive slack in all three units (flop WNS +1.81 / +1.88 / +1.87 ns); ff hold is clean.

### 4.4 lw: congestion of the large latch-RF banks

lw's global route overflows 19.4 % of its routing cells at 0.65 target utilisation, against 0.29 % predicted at
placement. The detours make the latch-RF read muxes in u_sg (banks of 512 and 768 words) up to 4-5 ns slow.
Two reruns tested whether more area fixes it:
- 0.50 target utilisation (core 2.60 mm2): overflow 14.5 %, ICC2 setup -4.70 ns after global route.
- 0.50 target plus a 0.55 local placement-density cap with high congestion effort: overflow 6.5 % (place-stage
  estimate 0.21 %), still -4.51 ns.
So spreading the cells reduces the overflow but does not remove the hot spots (max overflow 57-64 tracks per routing
cell).
vu16 (256-word banks) and mm (no RF) route with 0.36-0.38 % overflow. gm16 uses 512-word banks (32 of its 64 banks),
so it may share lw's problem. This is unresolved; with SRAM macros (P8e) it disappears.

### 4.5 Power (PrimePower, tt 1.0 V 25 C, 181.8 MHz, vectorless: toggle 0.1, static probability 0.5)

| unit | pre-layout total (clock network) mW | post-CTS, global route total (clock network) mW | post-CTS, placement estimate | total change | clock change | added clock power per sink |
|---|---|---|---|---|---|---|
| lw | 33.9 (18.4) | 42.8 (26.3) | 42.8 (26.7) | +26 % | +43 % | 58 nW |
| mm | 44.1 (12.2) | 57.5 (17.9) | 57.6 (18.2) | +30 % | +47 % | 68 nW |
| vu16 | 99.7 (54.3) | 116.4 (79.7) | 118.0 (81.5) | +17 % | +47 % | 60 nW |

- The added power is mainly the clock tree's net switching: vu16's clock-network switching rises from 0.97 to 24.6 mW,
  since the ideal pre-layout clock nets carried no wire load, while the clock-pin internal power barely changes
  (52.9 -> 54.2 mW).
- Data-path power moves both ways. vu16's combinational power falls from 32.2 to 23.0 mW (ICC2 downsizing); mm's rises
  from 22.9 to 29.7 mW (buffering and real wires).
- Power is insensitive to the parasitics source: global route vs placement estimate agree within 1.5 %, and lw's
  congestion barely moves it.

## 5. Chip-level extrapolation

**Power (tt, vectorless, 181.8 MHz).** Two ways, applied to P8's flat-chip pre-layout figure, 0.475 W
(u_gm 0.337, u_vu 0.086, u_lw 0.031, u_mm 0.020, seq / axil 0.0015 W):

1. **Per-unit total factor.** The measured post/pre ratio is used for lw (1.263), mm (1.304) and vu16 (1.168), and
   vu16's ratio for gm16 and seq: 0.394 + 0.100 + 0.039 + 0.026 + 0.002 = **0.56 W**.
2. **Clock-tree adder per sink.** The added clock power per sink is 58-68 nW in all three units, measured over
   0.64 M sinks. Across the chip's 1,616,567 sinks (420,535 flops + 1,196,032 latches) that is +0.094 to +0.110 W, so
   **0.57-0.59 W**, assuming no net change in data-path power (the measured changes were -9 to +8 mW per unit).

So the chip is **about 0.56-0.59 W** vectorless at tt with a clock tree, +18-23 % over 0.475 W. gm16 is 71 % of the
chip power and was not placed, so this is an extrapolation from vu16, its closest measured analogue.

For P9b's measured-activity chip power (0.80 W step average, ideal clocks): the clock-tree adder is set by the tree's
wiring and by how often the ICGs open. Vectorless runs open them half the time, more than real workloads do for the
word-gated latch RF. So the vectorless adder, **+0.09 to +0.11 W**, is an upper bound for the measured-activity case.
With it, the measured-activity step average is about 0.9 W or less.

**Timing (ss).**
- Measured post-CTS, global-route register-to-register WNS at 5.5 ns: vu16 -0.39, mm -0.29 ns. lw is -0.10 ns on the
  placement estimate and congested in the global route.
- Degradation from pre-layout at the same period: vu16 0.62 ns, mm 0.46 ns.
- gm16 has +0.24 ns pre-layout at 5.5 ns (its 5.25 ns run had -0.006 ns). With vu16's degradation it would sit near
  -0.4 ns.
- The chip's post-CTS period before post-route optimisation is therefore **about 5.9 ns (169 MHz)**, set by vu16
  (measured) and gm16 (extrapolated). With a post-CTS uncertainty of 0.10 ns instead of 0.25 ns it is about 5.75 ns.
- Cross-unit paths were not placed. P8's worst, seq > vu, had +0.131 ns pre-layout.
- Post-route optimisation, which was not run, would recover part of the loss. **The 5.5 ns chip clock is therefore
  not confirmed post-CTS; it remains a pre-layout figure.**

**Area.**
- At the template's 0.65 target the units' cores are 1.87-6.61 mm2, at 0.64-0.68 final utilisation.
- The chip at the same density needs about 27.8 mm2 of core (~5.3 mm square) before the IO ring. This is the
  arithmetic P8 gave, now backed by three placed units.
- lw's congestion (4.4) means the large latch-RF banks need lower local density than 0.65, so 27.8 mm2 is a lower
  bound for the latch-RF chip.

## 6. What remains estimated, and why

- **Not placed:** gm16 (63 % of the cells, 71 % of the power), seq, axil, and the chip-level interconnect between
  units. Every chip-level number in section 5 is an extrapolation.
- **No detail routing, no post-route optimisation, no sign-off extraction.** Parasitics are ICC2's global-route
  extraction (cworst at ss / tt, cbest at ff) or its placement estimate. Clock nets are globally routed, not detail
  routed (clock_opt's route_clock was skipped to fit the time; T65_CTS_ROUTE=0).
- **lw's global route is congested**; its timing is quoted on the placement estimate only.
- **Unit context:** each unit is placed alone in a square core with pins on all four sides and IO delays at 20 % of
  the period. On chip its inputs come from other units' flops on the same clock tree. The input-to-register hold
  violations at ss are artefacts of this (section 4.3).
- **Skew target missed:** 0.38-0.67 ns global skew at ss, against a 0.10 ns target. A CTS setup with ICG-aware
  balancing or skew groups would lower it; this was not tuned.
- **Power** is vectorless (same settings as P8), tt, with ss RC at tt as in P8. P9b's gate-level SAIF was not applied
  to the placed netlists: it was recorded on the pre-layout chip netlist on tau-03, and the placed units are on tau-02.
- **Disk:** after the home quota filled (21:05 UTC), run directories were moved to tau-02 local /tmp/lwta_p9c (PI
  approval). It peaked at 31 GB. It was **deleted on 2026-10-20 at 05:15 UTC** once the numbers were final. The summary
text files (59) and launcher logs were kept in the server clone (`~/lowa_tsmc65/p9c/src/rtl/build/tsmc65/summary`,
`logs_p9c`). The ICC2 databases, netlists and SPEF are gone; they can be rebuilt with section 8 (about 8 h on 8-12
threads for the three units).

## 7. Proposed paper text

Recommended (replaces the "pre-layout, no clock tree" caveat for power; timing stays pre-layout):

> "To check the effect of physical design, three of the six units (lw, mm and vu16; 37 % of the cells) were placed and
> clock-tree synthesised in ICC2 and analysed with global-route parasitics. The clock tree adds 17-30 % to unit power
> (clock network +43-47 %, about 60 nW per sink). Extrapolated to the chip, this gives about 0.56-0.59 W vectorless at
> tt instead of 0.475 W. Without post-route optimisation, the placed units lose 0.3-0.6 ns of setup slack at ss and
> close 5.7-5.9 ns, so the 5.5 ns clock is a pre-layout figure, and about 5.9 ns (169 MHz) is expected after clock-tree
> synthesis. The flat chip (5.2 M cells) exceeds the memory of the available server for place and route."

Short form (for a table footnote):

> "Pre-layout timing; post-CTS placement of three units (37 % of cells) indicates about 5.9 ns before post-route
> optimisation and +18-23 % power from the clock tree."

## 8. Reproduction

On the server, in a clone of the p9c branch, `src/rtl`, with the site files (`flow/tsmc65/site/site.tcl`,
`site_pnr.tcl` from `site_pnr.example.tcl`) and `build/tsmc65/hier` holding (or linking) P8's DC runs:

```
export T65_PERIOD=5.5 T65_CTS_ROUTE=0
T65_CORES=12 flow/tsmc65/pnr/run_pnr.sh lw                        # ndm, floorplan, place, cts, groute
T65_KID_TAG=mm=_p8f,rng9=_p8f,rng3=_p8f T65_PNR_TAG=_p8f flow/tsmc65/pnr/run_pnr.sh mm floorplan
T65_KID_TAG=vu16=_p5.25 T65_PNR_TAG=_p5.25 T65_CORES=8 flow/tsmc65/pnr/run_pnr.sh vu16 floorplan cts
T65_KID_TAG=vu16=_p5.25 T65_PNR_TAG=_p5.25 flow/tsmc65/pnr/run_pnr.sh vu16 holdfix holdfix
T65_KID_TAG=vu16=_p5.25 T65_PNR_TAG=_p5.25 T65_GR_FROM=holdfix flow/tsmc65/pnr/run_pnr.sh vu16 groute groute
T65_OUT_FROM=cts flow/tsmc65/pnr/run_pnr.sh lw cts_out cts_out     # placement-estimate outputs
# post-CTS PrimeTime / PrimePower (likewise mm, vu16 with their tags / dirs):
T65_POST_DIR=$PWD/build/tsmc65/pnr/vu16_p5.25/out_gr_holdfix T65_KID_TAG=vu16=_p5.25 \
    flow/tsmc65/sta/run_pt_unit.sh vu16 ss                     # + tt vectorless, ff; T65_PT_SWEEP="5.6 ... 6.25"
# pre-layout reference of the same unit:
T65_KID_TAG=vu16=_p5.25 T65_PT_SUFFIX=_p5.5_pre flow/tsmc65/sta/run_pt_unit.sh vu16 tt vectorless
```

lw at 0.50 utilisation: `T65_UTIL=0.50 T65_PNR_TAG=_u50`; with a density cap: `T65_PLACE_DENSITY=0.55 T65_PNR_TAG=_u50d`.
