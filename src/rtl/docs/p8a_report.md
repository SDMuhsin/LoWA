# P8a report: memory backends, TSMC-65 flow scaffold, block-level clock scout

Phase P8a of the TSMC-65 port (PLAN R11). Contents: (1) `mm_sram_bank` technology backends,
(2) regression evidence, (3) the `flow/tsmc65` tree, (4) a block-level DC clock scout at
ss 0.9 V 125 C, and (5) a clock recommendation plus a sizing estimate for the hierarchical full16 run.

## 1. Memory backends (R11a)

`hw/mem/mm_sram_bank.v` is now a thin wrapper. Its port contract is unchanged (written in its header):
one access per cycle, no byte enables, `q` valid the cycle after a read and held until the next read,
and a write is visible to a read in the next cycle. The backend is chosen by parameter `TECH`, whose
default is the macro `MM_TECH`.

| `MM_TECH` | backend | file (hw/mem/tech/) | default when |
|---|---|---|---|
| 0 | generic: behavioural array | `mm_bank_generic.v` | simulation (no `SYNTHESIS`) |
| 1 | gf180: sram256x8 macro tiles (the previous body) | `mm_bank_gf180.v`, `gf180mcu_ocd_ip_sram__sram256x8m8wm1.v` | `SYNTHESIS` (the Yosys/OpenSTA GF180 flow is unchanged) |
| 2 | tsmc65: latch register file, one clock-gated word per address | `mm_bank_t65.v`, `mm_icg.v` | selected explicitly (`flow/tsmc65` passes `MM_TECH=2`) |

- **tsmc65 (`mm_bank_t65`).** The design follows the template's `mq65_latch_rf`, but it is `WORDS` deep
  (not `2**AW`), so a 768-word bank has 768 words.
  - A write at edge N captures `d` in `wd_r`.
  - The word's `mm_icg` enable (`en & we & addr == w`) is latched while clk is low. The word latches are
    transparent for the high phase after N and close holding `wd_r`.
  - A read loads `q_r` from a `WORDS:1` mux. Writes never touch `q_r`.
  - The word latches are inferred by DC. No library cell names appear in `hw/`.
- **`mm_icg.v`** is the behavioural gate (enable latched while clk is low) that every simulator uses.
  - DC never reads it. `flow/tsmc65/common/rtl.tcl` substitutes the git-ignored
    `flow/tsmc65/site/mm_icg_site.v`, which maps `mm_icg` to the library ICG (template
    `mm_icg_site.example.v`).
- **gf180 deviation.** gf180 does not fully meet the "q held across writes" clause: its macros reload Q on
  a write to the last-read macro row. This was true before P8a too.
  - `tb_mm_sram_bank` shows the deviation when its hold check is on. No unit depends on it: every
    regression passes on the generic backend, which does hold `q`.
- **Lint (fixed decision 6).** All three simulators accept every file.
  - `flow/lint.sh` lints every backend through the wrapper (768x32 and 300x24 shapes), plus `lw_sg` and
    `gm_top` 4x4 on tsmc65.
  - Yosys still forbids latches everywhere except in `mm_bank_t65` / `mm_icg`, where it now *requires* them.
  - The exception is recorded in `docs/rtl_conventions.md`.

## 2. Regression evidence

All runs use the generic backend unless the row says otherwise. Every simulation has a cycle limit
(testbench watchdog / `+maxcyc`) or a wall-clock timeout. No testbench or tolerance was changed.

| target | result |
|---|---|
| `make lint` | PASS: 55 files / 55 modules, Verilator -Wall 141 runs, Yosys 55 tops, Icarus 55 tops |
| `make test-unit` | PASS: test-fp (43,559,600 vectors, 0 mismatches, 117 runs), test-gemm, test-vec, test-lowa, test-mem (47/47 rows), test-rng, test-ctrl (6/6) |
| `make test-mem` | PASS 47/47 (33 mm_unit AXI cases + 14 integration runs incl. the 3 real cases) |
| `make test-e2e` (reduced: 1 layer, B 2, L 32, 2 steps) | PASS: ALLWRITES 3,389,288 words per step bit-exact, 24/24 result tensors bit-exact vs the interpreter, losses 0.748631597 / 0.58404243; step 1 = 10,356,787 cycles (identical to P7a) |
| `make test-mem-banks` (new) | PASS 18/18: contract test on generic / gf180 / tsmc65 x 3 shapes (768x32, 256x32, 300x24) x Icarus + Verilator, 100k random ops each |
| `make test-mem-t65` (`--quick`) | PASS 41/41: 33 mm_unit cases + gm_top / gm_top+rn_mkgen `ep_bias_drop_resid`, vu_top `elem`, `emb` (nominal + stress DRAM) on the tsmc65 latch RF |
| tsmc65, real case `adamw_real_head` (lw_top, nominal DRAM) | PASS: 1,776,390 vectors, 5,329,369 cycles (57 min wall) |
| tsmc65, real case `real_ln_fwd` (vu_top, nominal DRAM) | PASS: 3,153,920 vectors, 2,963,526 cycles (2.6 h wall) |
| `make test-mem-gf180` | PASS 47/47 (the same set on the gf180 macro models) |

**Not run:**
- `test-e2e-full` (5 h), as instructed.
- The tsmc65 backend on the 16x16 `real_ffn1_256` integration case (all three of its DRAM settings) and on the `wfair=0` sensitivity row of `real_ln_fwd`. Verilator
  runs the latch RF 60-75x slower than the array (measured: `emb` 370 s vs 6 s, `real_ln_fwd` 9,321 s vs 145 s).
  `real_ffn1_256` (522 s on the array) would need about 10 h per DRAM setting, so it was skipped.
- Building `gm_top` with latch RFs also needs `-O0` C++ (`run_tests.py` adds it for tsmc65 only). At
  `-Os` the 500 MB of generated C++ did not finish in over an hour.
- `make test-e2e` itself ran ~4x slower in wall time than the P7a run (6,532 s); the cycle counts were
  identical. A gf180 build run at the same time showed the same rate, so the cause is the host, not the
  backend.

## 3. Flow tree `flow/tsmc65`

```
flow/tsmc65/
  sync.sh                 pull (git ff of the server checkout) | wip (rsync hw/ + flow/ to a pre-push tree) |
                          site | results | run '<cmd>' | tree
  site/                   README + *.example.* templates (tracked); site.tcl/.csh/.sh, mm_icg_site.v (git-ignored)
  common/site.tcl         loads site.tcl; t65_db / t65_lib / t65_oc
  common/rtl.tcl          hw/ file list with the site ICG wrapper swapped in; defines {SYNTHESIS MM_TECH=2}
  common/blocks.tcl       block table: top (input-register wrapper), parameters, wrapper file, DUT instance
  common/summary.tcl      summary writer: numbers + RTL instance names only
  common/env.csh          tool environment (sources site.csh)
  sdc/gen_sdc.tcl         SDC generator (period, corner, IO %): clock, uncertainty, IO delays, drivers, load, DRV
  syn/dc_block.tcl        DC topographical compile of one block (compile_ultra -gate_clock + incremental, no retiming)
  syn/run_block.sh        one run -> build/tsmc65/syn/<block>_<corner>_p<P>/, summary -> build/tsmc65/summary/
  syn/run_scout.sh        sequential batch block:period ..., end marker SCOUT_DONE
  tools/summarize.sh      one line per run: area, WNS/TNS, flop WNS, hold WNS, wall time, peak RSS, worst path
  wrap/wrap65_mm_sram_bank.v   input-register wrapper for a bank
```

**Constraints** (`gen_sdc.tcl`):
- Clock uncertainty 0.25 ns setup and 0.05 ns hold; clock transition 0.15 ns.
- IO delay 20 % of the period. Blocks sit inside input-flop wrappers, as in the GF180 flow.
- Max transition min(0.5, 0.1 P); max fanout 16.
- Flop-to-latch paths into the RF use the tool default: capture at the closing edge of the same high
  phase, with borrowing. A borrowing latch endpoint therefore reports slack 0, so the summary also
  reports the worst path into flops (`flopWNS`).

**Makefile targets** (run on tau-03 through `~/bin/tau3`):
- `make syn65-<block> [PERIODS=...]` and `make syn65-scout`
- `make syn65-status` and `make syn65-results`
- `make sync65-pull | sync65-wip | sync65-site`
- `test-mem-t65`, `test-mem-gf180` and `test-mem-banks` (local)

**Getting code onto the server.**
- The normal path is git: commit and push locally, then `make sync65-pull` (`git pull --ff-only` in
  `~/lowa_tsmc65/LoWA`).
- This scout ran before any push. It used `make sync65-wip`, which rsyncs `src/rtl/hw` and `src/rtl/flow`
  to a separate work tree `~/lowa_tsmc65/wip/src/rtl`; `T65_TREE=wip` is set in the git-ignored site.sh.
- After the push, set `T65_TREE=checkout`.

## 4. Clock scout (DC W-2024.09 topographical, ss / 0.9 V / 125 C, RVT, tau-03, 8 cores, one job at a time)

Every block sits inside an input-register wrapper. Area = total cell area of the block incl. the wrapper.
WNS = worst setup slack over all endpoints; latch endpoints that borrow sit at 0.0000.
flopWNS = worst slack into flops / outputs. Hold is pre-CTS with an ideal clock.

| block | config | P = 6.67 ns | P = 5.0 ns | P = 4.0 ns | extra (faster) |
|---|---|---|---|---|---|
| fma | fp_fma STAGES 2 | 12,555 um2, WNS +0.000 | 12,787, +0.002 | 13,806, +0.000 | 3.33: 14,402 met; 2.86: 15,085 met; 2.5: 17,348 met |
| pe | gm_pe (FMA 2, 4 acc) | 14,707, +0.002 | 16,071, +0.000 | 16,083, +0.002 | 3.33: 16,878 met; 2.86: 18,098 met; 2.5: 20,046 met (+0.0000) |
| sfu | fp_sfu LF 2 | 23,464, +0.002 | 23,766, +0.001 | 24,664, +0.000 | 3.33: 25,538 met; 2.86: 25,904 met |
| lane | vu_lane (2 x 256x32 latch RF, FMA, acc FMA, SFU) | 216,245, 0 / flop +0.028 | 219,259, 0 / +0.001 | 221,749, 0 / +0.006 | 3.33: 225,771, 0 / +0.000 met |
| gm4 | gm_top 4x4, NSFU 2 (full-size feeders: 14,336 latch words) | 3,965,346, 0 / +0.093 | 3,991,126, 0 / +0.017 | 4,038,143, 0 / +0.003 | 3.33: 4,069,933, 0 / +0.000 met (at the edge) |
| bank768 | mm_sram_bank 768x32, tsmc65 | 181,989, 0 / +3.301 | 181,920, 0 / +1.904 | 182,510, 0 / +1.140 | 3.33: 0 / +0.679; 2.86: 0 / +0.372 |
| bank256 | mm_sram_bank 256x32, tsmc65 | | 61,047, 0 / +2.108 | | |

No run has a negative setup slack, so TNS is 0.000 everywhere. DC stops optimising once a constraint is
met, so "+0.00x" means the period was closed, not that the block sits at its limit. The faster rows bound
the limit.

**Critical paths** (start -> end, instance names from the RTL; arrival in ns at the tightest period run):

| block | period | worst path into flops | arrival | what it is |
|---|---|---|---|---|
| fma | 2.5 | `a_r` (wrapper input) -> `u_d2` stage register | 2.27 | first FMA stage: operand -> multiplier / alignment, cut point d2 |
| pe | 2.5 | `a_s` (PE operand register) -> `u_fma/u_d2` | 2.25 | the same FMA first stage |
| sfu | 2.86 | `ia` / `ib` -> `u_fma/u_d2` | 2.60 | the same FMA first stage (SFU's internal FMA) |
| lane | 3.33 | `opb` -> `u_fma/u_d2`; `u_sfu/ib` -> `u_sfu/u_fma/u_d2`; `af_rp` -> `u_afma/u_d2` | 3.07-3.10 | FMA first stage plus the lane's operand mux / FIFO read in front of it |
| gm4 | 3.33 | `u_loader/tf_cp[0]` -> `u_wfeed/g_buf_1__g_lane_7__u_bank/.../q_r` | 3.09 | loader copy pointer -> W-feeder bank address -> 256:1 latch read mux -> `q_r` |
| gm4 | 4.0 | `u_ctrl/ps_buf` -> `u_xfeed/.../u_bank/.../q_r` | 3.73 | feeder buffer select -> bank address -> 512:1 latch read mux -> `q_r` |
| gm4 | 5.0 | `u_ctrl/oc_b1[0]` -> `u_ctrl/oc_b1[31]` | 4.71 | 32-bit address counter / adder in gm_ctrl (relaxed run; not a limiter) |
| gm4 | 6.67 | `u_drain/u_d2` -> `u_drain/g_add_0__u_add/u_d2` | 6.32 | drain fp_add (relaxed run; not a limiter) |
| bank768 | 2.86 | `addr_r` -> `q_r` | 2.22 | 768:1 read mux (period-independent; ~1.9 ns of logic) |
| bank768 | all | `wd_r` -> word latch (borrowing) | 0.50-0.76 | write data into the open word latch, must fit in P/2 |

**DC runtime and memory per block** (wall clock of one dc_shell with 8 cores; peak RSS of the process
tree from GNU time; the same at every period within ±10 %):

| block | leaf cells (4.0 ns) | DC wall | peak RSS |
|---|---|---|---|
| fma / pe / sfu | 5.0k / 5.4k / 8.4k | 2.4-3.0 min | 2.36-2.38 GB |
| bank256 / bank768 | 16k / 48k (8,192 / 24,576 latches) | 2.2 / 5.7 min | 2.40 / 2.53 GB |
| lane | 63k (16,384 latches) | 6.0-6.8 min | 2.60 GB |
| gm4 | 1.08M (458,752 latches, 15,294 ICGs) | 75-90 min (101 min at 3.33 ns) | 7.9-8.2 GB |

About 2.3 GB of every footprint is the tool plus the library / Milkyway load. Above that, memory and
runtime scale with cell count, and in these blocks the cell count is dominated by the latch RFs.

**Latch RF cost.** About 238 um2 per 32-bit word, from both bank sizes:
- 256x32: 0.061 mm2. 768x32: 0.182 mm2.
- Of the 768x32 bank: latches 0.124 mm2, read mux + decode 0.053 mm2, ICGs 0.005 mm2.
- The full16 on-chip capacity of 89 KiB (22,784 words, docs/memory_map.md) therefore projects to about
  5.4 mm2 of latch RF at ss.

## 5. Recommendation

**Clock target: 4.0 ns (250 MHz) at ss 0.9 V 125 C.**

- Every scouted block closes 4.0 ns with non-negative slack: fp units, PE, lane, gm_top 4x4 and the
  largest bank. The fp units and PE also close 3.33, 2.86 and even 2.5 ns, at +4 % / +9 % / +26 % area
  over 4.0 ns.
- The lane closes 3.33 ns, and gm4 closes 3.33 ns only at the edge (flop WNS +0.0000).
- The block-level numbers are pre-CTS and come from topographical estimates with ideal clocks. Three
  things are not modelled and will cost margin on full16:
  1. CTS skew across thousands of gated latch clocks (gm4 alone has 15k ICGs).
  2. Top-level wires between the array, the feeders and the memory system.
  3. The PT/PP signoff in P8c.
- 4.0 ns keeps roughly 15-20 % of headroom over the fastest period every block met (3.33 ns), which is
  the usual pre-layout margin.
- 5.0 ns (the provisional target) closes with large margin but gives away 25 % throughput for nothing
  visible at block level. 6.67 ns is not needed.

**What limits the clock, in order.**
1. The first FMA stage (operand register -> multiplier / alignment -> cut point d2), in every fp user:
   fma, pe, sfu and the lane's two FMAs. It reaches 2.25-2.6 ns of arrival at 2.5-2.86 ns. On the lane,
   the operand mux / accumulator FIFO read sits in front of it (3.07-3.10 ns at 3.33).
2. The GEMM feeders' read path: control state (gm_ctrl buffer select, gm_loader copy pointer) -> bank
   address -> latch read mux -> `q_r` (3.73 ns arrival at 4.0 ns; at 3.33 ns it closes at +0.000). This is the tsmc65-specific path. A latch RF read mux is slower
   than a macro read, and it sits behind control logic.
3. The gm_ctrl 32-bit address counters appear in the 5.0 ns run (4.71 ns arrival), unoptimised there.
4. The latch RF itself is not limiting. Its write needs only 0.5-0.8 ns of the half cycle (2.0 ns at
   4.0 ns), and its bare read path is 2.2 ns.

An FMA STAGES change or retiming is therefore not needed for 250 MHz. If the PI wants more than
~300 MHz, the first change would be a third FMA stage or an extra register in front of the feeder read
mux; either needs PI sign-off and an L2 re-run (R11c).

**Sizing of the hierarchical full16 run (R11d), from the measured scaling.**
- A flat full16 gm_top would hold ~20.5k latch words (feeders + drain, 80 KiB) and 256 PEs, about
  2.6-3M leaf cells. Extrapolating gm4 (1.08M cells, 7.9 GB, 80 min), that is roughly 15-20 GB and
  4-6 h, which is too close to the 30 GB shared limit.
- **Plan:** compile each unique leaf once with `compile_ultra`, then `set_dont_touch` it and link the
  `.ddc` into its parent.
  - Leaves: gm_pe (~3 min), each bank shape (256x32, 512x32, 768x32: 2-6 min each), fp_sfu, fp_add,
    vu_lane (~7 min).
  - Parents: gm_top and vu_top (glue + feeder control around 256 / 16 instances), lw_top and mm_top.
  - Top: top_chip (glue only).
- With leaves pre-compiled, each parent run should stay near the 2.4-4 GB / sub-hour class seen here.
  The largest RSS will be the top link of ~3M instances, estimated at under 12 GB with 8 cores.
- Each run uses at most 8 cores and runs alone (one big job at a time).
