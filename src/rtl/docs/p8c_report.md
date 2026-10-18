# P8c report: timing repairs of gm16 / vu16 / mm at 4.0 ns, bank copies, on-chip capacity

Phase P8c of the TSMC-65 port (PLAN R13). Only the RTL repairs signed off in R13 were implemented. The clock
(4.0 ns, ss 0.9 V 125 C), the SDC generator, the IO budgets and the flow options are unchanged; there are no false
paths or multicycle paths. DC runs: tau-03, 8 cores, one job at a time, in the server work tree (`T65_TREE=wip`,
the same hw/ and flow/ as this commit). Netlists and full reports stay on the server.

**Result.**
- **mm closes 4.0 ns** (was -0.08 ns).
- **gm16 improves from -2.14 to -1.31 ns** and **vu16 changes from -1.08 to -1.59 ns** (TNS -4,102 -> -1,462, failing flop endpoints 13,302 -> 1,553), but neither closes with the
  signed-off repairs alone. Section 4 lists the remaining paths and the proposed next repairs. They are outside R13
  and were not implemented.
- All regressions are bit-exact vs L1 and unchanged (lint, test-unit, test-e2e reduced). Cycle cost: +0.03 % on
  the reduced e2e step.
- The bank-copy flow fix (section 5) works: gm16 and lw link their own bank designs. On-chip capacity is reconciled
  (section 6): the RTL is right, the docs and budget.py were out of date.

## 1. RTL repairs (hw/, tech-neutral)

| unit | repair (R13) | files | latency effect |
|---|---|---|---|
| gm | feeder write request registered per lane, after the lane decode (`wen_s / waddr_s / wdata_s`, X and W feeders) | gm_xfeed, gm_wfeed | write lands +1 cycle; `tile_done` delayed +1 in gm_top, so gm_ctrl and the merge engine still start after the last write |
| gm | X-feeder bank read-address stage (`ren_s / raddr_s`), computed from the row readers' next state | gm_xfeed | none: the read issues in the same cycle as before (the W feeder already reads from the pass-counter registers) |
| vu | broadcast register stage: every seq -> lane control signal is registered once in vu_top: issue, keep bit, micro-op, acc / max / output control, cross-lane select and result, s2 clear | vu_top | lanes run 1 cycle behind vu_seq. Column-vector banks are still read with the seq issue and their data is registered, so it arrives in step with the lanes |
| vu | rd0 beat broadcast stage in the loader (data, count, address, one-hot target lane) | vu_loader | holder write +1; the full check counts the staged beat; ld_done waits for the stage |
| vu | registered lane outputs: `sfu_out_cnt`, `acc_cnt` (occupancy counters), `st_busy`, `ob_room` (next-state registers) | vu_lane | none (same cycle as the old combinational value) |
| vu | registered lane outputs: `acc_idle`, `xl_val` (registered copies) | vu_lane | +1 cycle |
| mm | `mk_seed` registered at the mm_top input | mm_top | +1 cycle after a CSR write; the seed is a per-step value |
| mm | `rd_req_ready` decoupled: request register per read port; ready = register empty or moving on | mm_rdport, mm_top | +1 cycle from request to AR; still one request per cycle. RAW is checked on the registered request; WAR is checked against the registered request (was: the offered request) |

**Latency assumptions that were updated** (vu_seq, in the RTL, nothing in a testbench):
- POP in-flight window 3 -> 4.
- STORE limiter: the lane store FIFO is 4 deep and a decision now has 4 cycles unseen. A STORE is not decided in a
  cycle whose 4 predecessors all decided one.
- Lane `ob_room` threshold: `ob_wpos < 5` -> `< 4`. The broadcast register adds one more push in flight.
- ACC room stays at `acc_cnt < 4`: with 4 pushes in flight the FIFO fills to exactly its depth of 8.
- Drain wait 6 -> 7 cycles; flush wait 2 -> 4 cycles (broadcast register + registered acc_idle); cross-lane settle
  1 -> 3 cycles before the first row.

**Not changed, checked:**
- sw/ (the compiler) assumes no engine latency.
- ct_seq retires on engine done + 2 cycles + write-channel idle. The write path is unchanged.
- Memory ordering: a read's order point is now its acceptance into the port register. A write offered in the same
  cycle as an overlapping read used to wait for the read; now it can go first. Consecutive commands keep their order,
  because a read already in a port register blocks the write. The mm unit tests issue one command per handshake and
  pass unchanged.

## 2. Regressions (all bit-exact vs L1, nothing weakened)

| target | result |
|---|---|
| make lint | PASS: 55 files, 141 Verilator runs, Yosys, Icarus |
| make test-unit (fp, gemm, vec, lowa, mem, rng, ctrl) | PASS: TEST-FP, TEST-GEMM, TEST-VEC, TEST-LOWA, TEST-MEM [generic], TEST-RNG, TEST-CTRL (6/6), every vector bit-exact |
| make test-e2e (1 layer, B 2, L 32, 2 steps) | PASS: ALLWRITES 3,389,288 words and 24/24 result tensors bit-exact per step, losses unchanged |

Cycle deltas against the P8a logs (same tests, pre-repair RTL):

| real case | P8a cycles | P8c cycles | delta |
|---|---|---|---|
| gm_top real_qproj / scores / pv / attnout / ffn1_256 / dv / dgelu / ffn1 | 9,697,592 ... 38,791,260 | each +1 | +1 (tile_done +1 at the end of the op) |
| lw_top + mm_top (real) | 5,329,369 | 5,329,371 | +2 |
| gm_top + mm_top (real) | 2,576,250 | 2,576,252 | +2 |
| gm_top + rn_mkgen (real) | 9,698,537 | 9,698,538 | +1 |
| vu_top + mm_top (real) | 2,963,526 | 2,965,567 | +2,041 (+0.069 %) |
| vu_top + rn_mkgen (real) | 9,216,047 | 9,240,626 | +24,579 (+0.267 %) |

Other real cases have identical cycle counts. The vector unit pays for the extra fixed waits (drain, flush,
cross-lane settle) and the STORE limiter, once per phase or step.

- **e2e reduced step:** 10,356,765 -> 10,359,831 cycles (+3,066, +0.030 %).
  - gm_busy +51, vu_busy +2,202 (+0.26 %), lw_busy +10, merge_busy +12.
- **Full-step estimate (12 layers, not run).** The per-op costs repeat per layer: about +3 k cycles per layer of
  vector ops and +50 per layer of GEMM ops. That gives +37 k cycles on the 12-layer step, +0.03 %.

## 3. DC at 4.0 ns, ss (after the repairs)

| block | area um2 (unit total) | of which memory | WNS / TNS (ns) | p8b WNS | DC wall | peak RSS |
|---|---|---|---|---|---|---|
| mm (mm_top 8/2/4) | 1,218,577 | 0 | **0 / 0** | -0.078 | 29 min | 3.6 GB |
| lane (vu_lane) | 220,487 | 121,913 | 0 (flop +0.002) | 0 | 7 min | 2.6 GB |
| gm16 (gm_top 16x16) | 11,170,038 | 5,845,328 | **-1.314 / -11,327** (11,039 flop endpoints) | -2.136 | 93 min | 13.2 GB |
| vu16 (vu_top V=16) | 4,364,134 | 2,316,339 | **-1.594 / -1,462** (1,553 flop endpoints) | -1.082 | 57 min | 6.6 GB |
| lw (lw_top) | 1,307,480 | 729,809 | 0 / 0 (identical to p8b, now with its bank copies) | 0 | 21 min | 3.5 GB |
| bank copies (x_gm, x_vu, x_lw) | 60,956 / 121,710 / 182,238 | all | 0 (identical to p8b) | 0 | 2-5 min | 2.4 GB |

**Integrity checks.** Same scans as p8b:
- latches only in `mm_bank_t65`: gm16 24,576 words;
- 0 unresolved references, 0 check_design issues, 0 errors apart from PROC-3;
- gm16 removals unchanged (200 / 588 / 667);
- the new registers add 2,147 flops to gm16.

## 4. Remaining misses (not repaired: outside R13)

Each block's run now writes `groups_<block>.txt`: the worst path of every violating endpoint group, timed in the
session (`t65_group_paths`). It replaces dc_paths, which cannot re-time abstraction pins after a reload.

**gm16, -1.31 ns.** The worst groups, with the delay split over the hierarchy:

| group (n endpoints) | WNS | path | split |
|---|---|---|---|
| W-feeder bank `q_r` / word ICG enables (1,024 / 5,353) | -1.31 / -1.08 | input `mg_lane` (merge RMW port, 0.8 ns IO budget) -> lane hit decode -> bank address mux -> latch read mux / word decode | wfeed 2.05-2.86, bank 1.06-2.23 |
| PE drain chain `dr_s` (2,046) | -1.31 | `dr_s` of PE(r, c) -> PE(r+1, c) | PE 1.37, array wire 1.59, PE 2.09 |
| ACC bank `q_r` (512); X row capture `a_s` (512) | -1.31 / -1.29 | bank `q_r` -> 16:1 lane capture mux | bank 2.97, xfeed 1.58 |
| X-feeder word ICG enables (2,093) | -1.19 | `wen_s` (new register) -> bank word decode -> ICG E | xfeed 1.10, bank 3.73 |
| feeder `wdata_s` / `waddr_s` (1,536 / 400) | -1.15 / -1.24 | loader tile FIFO -> beat routing -> new per-lane registers | loader + feeder |
| ACC word ICG enables (3,518) | -1.03 | drain `par5` -> bank word decode | drain 1.37, bank 3.29 |

The repairs removed the tile-FIFO -> write-decision -> read-mux path (-2.14 ns). What is left is mostly wire:
- the bank abstractions' own decode and read mux take 2.2-3.7 ns in the parent, against about 2.1 ns in the
  bank's own run;
- the PE-to-PE drain hop crosses 1.6 ns of estimated wire.

Without a floorplan, DC topographical spreads 64 banks and 256 PEs over 11 mm2.

Proposals (for the PI; none implemented):
1. Register the merge RMW request at the gm_wfeed input. The merge read data arrives +1 cycle later, and the
   lw_merge RMW pipeline needs one more stage.
2. Register the loader's beat (data, routing) before the lane decode. This is a second write stage: tile_done +2.
3. A pipeline register on the drain chain every 4 rows. Drain injection offsets +1 per group, in gm_drain's schedule.
4. Run the block abstractions in a floorplanned context (bank and PE placement). The bank-internal decode paths
   (ICG E, 3.3-3.7 ns) are a placement effect: they close in the bank's own run.

Without 3 and 4, gm16 needs about 5.3 ns.

**vu16, -1.59 ns.**
- The repairs removed the old groups: the loader hold buffers (-1.01, 7,683 endpoints), the lane r3 / cvec, the
  status -> seq credit cones, and most lane-bank paths.
- What remains are the lane abstractions' own output and input paths in the parent. A lane register drives its
  output pin through 1.2-3.9 ns of estimated wire inside the lane region, then one cross-lane stage:

| group (n) | WNS | path | split |
|---|---|---|---|
| writer `g_data` (256) | -1.59 | lane `ob_rsel` -> `ob_data` (2:1) -> writer 16:1 lane select | lane 3.85, writer 1.46 |
| cross-lane FMA stage 1 (97) | -1.58 | lane `xl_val_r` -> 16:1 `xl_row` mux -> u_xfma | lane 2.11, top 1.33, xfma 1.73 |
| seq `nseq` / `sl_pc` (94) | -1.57 | lane `sfo_cnt` (register) -> min over 16 lanes -> issue / turn logic | lane 1.24, top 3.65 |
| lane bank `q_r` (802) | -1.35 | `b_iss_v` (new broadcast register) -> lane bank read | top 0.60, lane 1.84, bank 2.61 |

Proposals (none implemented):
1. Register the cross-lane operand after the 16:1 mux. This costs no latency: `xl_row` already changes one cycle
   before the issue (p8b fix 3).
2. Register the writer's selected lane beat. ob_data from next state needs a writer pop-protocol change.
3. Reduce the 16 lane counts to the minimum in a registered tree in vu_top (+1 to the POP window).
4. Duplicate the broadcast register per group of 4 lanes, kept from merging.
5. Time the lanes in a floorplanned context: the 1.2-3.9 ns inside the lane region is placement spread.

Without them, vu16 needs about 5.6 ns.

**mm** closes; the `wr_addr -> wr_ready` path (-0.07 in p8b) closes too.

## 5. Flow: per-parent bank copies (P8c blocker of p8b section 6)

- `common/hier.txt` adds bank256_gm, bank512_gm, bank256_vu, bank256_lw, bank512_lw and bank768_lw, with link
  column `copy`.
- A copy's run renames its top design to `<block>__<design>` before it writes the abstraction.
- A parent with a copy child reads the copy's .ddc, `change_link -all_instances`es its elaborated bank instances to
  the copy design, and removes the elaborated one.
- gm16 links 32 x bank512_gm + 32 x bank256_gm; lw links its three copies; vu16 links bank256_vu.
- The lane keeps its full-netlist banks (`lane__...` names).
- The chip-level link therefore reads no design name from two .ddc files.
- The top level was not run (P8d).

Other flow changes:
- `t65_group_paths`: worst path per violating group, written in the session to `summary/groups_<block>.txt`.
- dc_paths: removes a stale Milkyway library first (MWUI-290) and fixes a split bug.
- `site.example.sh` honours `T65_TREE` from the environment.

## 6. On-chip capacity: RTL 146 KiB vs 89 KiB in the docs

The RTL is correct; docs/memory_map.md section 3 and budget.py still described the P0 plan. Per unit, in 32-bit
words:

| | P0 plan (docs until P8b) | RTL (hw/) | why |
|---|---|---|---|
| X + W buffers | 16,384 | 16,384 | same |
| ACC | 4,096 | 8,192 | gm_drain keeps two halves: block n's epilogue reads one while block n+1 accumulates (P3) |
| vector unit | 0 (flop transpose stagers) | 9,728 | P4 built 2 x 256-word tile buffers per lane + 6 column-vector banks |
| LoWA | 1,792 | 3,072 | P5 added the atom-centre table (256) and the merge directory / coefficient buffers (2 x 512) |
| DMA FIFOs | 512 | 0 | flops in hw/mem, not RAM |
| **total** | **22,784 = 89 KiB** | **37,376 = 146 KiB** | |

Corrected:
- docs/memory_map.md section 3 (per-bank table);
- docs/calc/budget.py (`onchip_words`; macro equivalents from the RTL words, so full16 = 584 GF180 macros =
  39.6 mm2, core estimate 122 mm2);
- docs/budget.md (table and note).

No buffer size was changed. The R11a text ("89 KiB") is a ruling and was left as written. The figure that
applies is 146 KiB.
