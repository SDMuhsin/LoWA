# P6a report: memory system (hw/mem), AXI4 master + DMA channels + DRAM model

## 1. Architecture
`mm_top` = NR engine read channels (`mm_rdport`) + NM keep-bit channels (`mm_mkport`, test stand-in) + NW write
channels (`mm_wrport`) on one AXI4 master port (the chip boundary, ruling R3). Full-chip set: gm rd0/rd1, vu rd0/rd1,
lw rd0/rd1/rd2 (NR = 7), gm/vu mk (NM = 2), gm/vu/lw wr (NW = 3). The engines are unchanged: they already generate the
2-D strided ROW, TRN, gather and keep-bit index request streams (P3-P5); a channel turns each (byte address, 1..64 words)
request into bursts and back into in-order 8-word beats with any 4-byte alignment.

**AXI4 parameters.** Data 32 x 2^AXL bit, default **256 bit** (spec / budget: the GEMM loader takes 1 beat = 32 B per
cycle = 800 MB/s at 25 MHz; 64 and 128 bit are supported and tested for a narrower FPGA link, at proportionally lower
bandwidth). INCR only, full-size beats, partial words by WSTRB, <= 16 beats, never across 4 KiB. ARID = read channel
(engines 0..NR-1, keep bits NR..), AWID = write channel; per-ID in order, different IDs may complete in any order and R
beats may interleave. RREADY = BREADY = 1: a burst is granted only when its channel's return FIFO has room for all
its beats (RFL0 = 64 beats for the GEMM tile stream, 16 otherwise), so no channel can block another. Outstanding:
MAXR = 8 read bursts, MAXW = 4 write bursts (parameters).

**Arbitration.** Round robin per burst over the channels that hold a granted-able burst, one AR and one AW slot
(registered). Justification: no starvation (a waiting channel is served within NR+NM-1 grants), and the heavy GEMM
stream still gets the bus whenever the others are idle; fixed priority would starve LoWA-SG / vector streams beside the
GEMM. W follows AW order (W-order FIFO).

**Same-address ordering (word granular, at acceptance time).** A read request is not accepted while it overlaps a
pending write (accepted, not B-acknowledged) of any channel; a write beat is not accepted while it overlaps a pending
read (accepted, last beat not arrived), a read request offered in the same cycle, or a pending write of another write
channel (WAW across AWIDs). Hence every read returns exactly the writes accepted before it, in any interleaving, with no
deadlock (writes never wait for a younger read). A blocked access flushes the write channels' open bursts.
`wr_idle` per write channel = all its writes B-acknowledged: the flag the P6b sequencer must wait for before signalling a
consumer on another unit.

**Write path.** Beat -> AXI lines (1-2 at 256 bit), same-line merge (later wins), coalescing into INCR bursts up to 16
beats (closed on discontinuity, 4 KiB, 16 beats, 3 idle cycles, or flush).

**Keep bits (ruling R2, PI condition).** The mk interface is the engines' plain keep-bit stream; `MK_DRAM = 1` serves it
from a bit-packed DRAM region (`MASK_STANDIN`, 0x8000_0000, outside the memory map) as a **test stand-in**, `MK_DRAM = 0`
passes it to an external source (the P6b Threefry generator) with no engine change. **Open deviation**: the stand-in adds
DRAM traffic the design does not have (one AXI line per 256 keep bits, i.e. 1/32 of the masked data bytes); it is
counted separately and must be excluded from bandwidth figures (section 4).

## 2. DRAM model (tb/mem/tb_dram.sv, test model only)
Sparse region image (memory_map placement, `mm_image.py`), latency, shared R+W bandwidth cap (token bucket),
random backpressure, ID reordering, R interleaving, protocol checks (burst type/size/alignment/length/4 KiB/
WLAST/W-before-AW), per-ARID R-beat counters (`DRAMID`, separates the keep-bit stand-in) and R-idle causes
(`DRAMIDLE`: free slot with no read outstanding / reads still in latency).
**Nominal** (`NOM`): lat 24 cycles (~1 us), wlat 8, bw 3200 (= 32 B/cycle = 800 MB/s at 25 MHz), `+wfair=1`: R and W
share the tokens.  **Stress**: lat 100, wlat 40, 400 MB/s, 30 % AR/AW/W backpressure, ID reordering, R
interleaving (+ 30 % client stalls in the unit tb).  `+wfair=0` (sensitivity, section 4): a W beat is accepted only
with a spare token while R streams, i.e. strict read priority.

## 3. Tests (`make test-mem`, clean build, 18 min 55 s wall, **TEST-MEM PASS**, build/test-mem/summary.txt)
Golden: unit cases from an independent sequential python reference on the memory_map image (gen_mm.py);
integration cases from L1 (gen_integ.py through each engine's existing generator), compared by the engine tbs.
Every integration run has a cycle watchdog (`+maxcyc`, ~4x nominal).

| tb | cases | DRAM knobs | result |
|---|---|---|---|
| mm_unit 256 / 128 / 64-bit AXI | burst, pattern (ROW / TRN / unaligned / gather / mask), write, order, outst | nominal + stress | 30/30 PASS |
| mm_unit 256-bit, MAXR 8 / 4 | outst (every slot used, never more) | lat 200, no bw cap | 2/2 PASS (max_out 8 / 4) |
| mm_unit 256-bit, MAXR 4 | order | stress | PASS |

| engine + mm_top (bit-exact vs L1) | cycles nominal | stress | other | P3-P5 ideal memory |
|---|---|---|---|---|
| gm ep_bias_drop_resid | 6,343 | 11,756 | | 6,123 |
| vu elem | 41,315 | 82,387 | | |
| vu emb | 216,357 | 667,849 | | |
| lw adamw_real_head | 5,329,369 | | | 5,329,353 |
| gm real_ffn1_256 (256x3072x768) | **2,576,250** | | no bw cap 2,434,146; wfair=0 3,841,682 | 2,434,140 |
| vu real_ln_fwd | **5,255,246** | | wfair=0 5,247,054; MAXR 32 / RQL 5 / RFL 5 3,676,233 | 2,656,302 |

**real_ffn1_256 did not hang**: the earlier >20 min run was the read-priority DRAM setting (3.84 M cycles at
~180 us/s of simulated time = 14 min wall); it completes bit-exact in every setting.

## 4. Bandwidth efficiency vs budget
Efficiency = useful beats (R + W, keep-bit stand-in excluded) / cycles of the run.

| case | DRAM beats | DRAM-bound (1 beat/cycle) | compute-bound | measured | efficiency |
|---|---|---|---|---|---|
| gm real_ffn1_256, nominal | 2,557,440 | 2,557,440 | 2,434,146 | 2,576,250 | **0.993** |
| gm real_ffn1_256, read-priority DRAM | 2,557,440 | 2,557,440 | 2,434,146 | 3,841,682 | 0.666 |
| vu real_ln_fwd, nominal | 1,623,040 | 1,623,040 | 2,656,302 | 5,255,246 | **0.309** |
| lw adamw_real_head | 518,120 | | 5,329,353 | 5,329,369 | 0.097 (compute-bound) |

- **GEMM.** budget.py's bytes per MAC (0.125 B at Br = Bc = 64) match the measured traffic exactly (2,360,832
  R beats for 603,979,776 MACs); the phase needs 0.97 R beats/cycle + 0.08 W, so at 800 MB/s it is bus-bound by
  construction; the memory system reaches 0.993 of the link.  With read-priority the output writes starve, the
  drain and the array stall, and 1,421,054 R slots go unused with *no* read outstanding (DRAMIDLE): the loss is
  the DRAM's R/W policy, not the master; raising MAXR to 16 or RFL to 32 changes nothing (3,841,682 cycles in all
  three variants).  At 1600 MB/s the case is compute-bound again (2,434,146).
- **Vector.** The vector unit issues 1-beat bursts (1,185,792 ARs for 1,228,800 R beats; 393,728 AWs for 394,240
  W beats).  Root cause: request **order** in the engine, not missing coalescing in hw/mem.  vu_loader walks
  "for each 8-word column block, one request per row" (row index inner, `r0_r`, then `r0_cb`) and vu_writer writes
  "one 8-word group of every row, then the next group": consecutive requests of a channel are one row stride
  apart, never contiguous, so mm_rdport / mm_wrport cannot merge them (mm_wrport already merges contiguous beats
  into <= 16-beat bursts).  With 1-beat bursts the bound is outstanding requests / latency (RQ = 8 per channel,
  MAXR = 8: ~8 beats per ~26 cycles = 0.31, as measured).  Memory-side alternative measured: MAXR 32, RQL 5,
  RFL 5 -> 3,676,233 cycles (1.43x), not adopted (4x pending-range comparators in the timing-critical RAW / WAR
  path, +area).  **Change needed in vu (not made, P4 RTL):** (1) vu_loader rd0: row-major order, one request per
  row of the whole segment (len = seg_len <= 64 words = one <= 8-beat burst), which needs the lane holding register
  to absorb up to 8 consecutive beats for one lane (8-beat per-lane FIFO, or a beat-wide (8-word) write port into
  the lane tile buffer); (2) vu_writer OUT: emit a row's groups back to back (row-major), which needs a per-lane
  segment output buffer (<= 64 words) instead of the 8-word lockstep output register; the existing wrport
  coalescing then forms <= 16-beat bursts with no hw/mem change.  Expected: real_ln_fwd back to its compute bound
  (~2.66 M cycles, the DRAM beats are 0.61 of it).
- **budget.py** now charges every GEMM row max(engine, bytes / (32 B x 0.993)) and every vector op max(kernel,
  element-passes x 5.50 B / (32 B x 0.309)) (5.50 B per element-pass measured on real_ln_fwd; P0 assumed 4.0);
  `--dram-rprio` uses 0.666, `--no-dram` the old compute-only time.  full16, L = 128: step **129.4 s -> 135.0 s**
  (GEMM 2,728.5 -> 2,777.9 M cycles, vector 499.3 -> 591.3 M); 189.0 s with a read-priority DRAM.  docs/budget.md
  updated (table, P6a paragraph, GLUE s/step).

**Keep-bit stand-in traffic (ruling R7, excluded above).**  ARID NR (= 2) beats: ep_bias_drop_resid 22 of 3,694 R+W beats
(0.6 %), elem 861 of 7,291 (11.8 %), emb 3,561 of 52,542 (6.8 %); the real cases carry no mask.  This traffic
disappears with the P6b on-chip generator (MK_DRAM = 0).

## 5. Area and slack (`make synth-mem`, clean, 8 min 51 s, **SYNTH-MEM PASS**; mm_top NR 7, NM 2, NW 3, 256 bit, incl. wrapper input flops)

| | cells | area tt | ss raw | **ss after repair** | tt raw | hold ss |
|---|---|---|---|---|---|---|
| before (P6a draft) | 307,829 | 9.79 mm2 | -22.05 | **-2.79** | -5.20 | +0.46 |
| + cm_rstbuf reset tree only | 307,504 | 9.79 mm2 | -18.16 | -6.34 | -0.78 | +0.47 |
| **final** (+ rdport packer enables) | 305,561 | **9.74 mm2** | -12.25 | **+2.61** | -0.10 | +0.46 |

`repair_design` without placement inserts no buffers (docs/p3_report.md section 6), so RTL fanout is what STA sees.
Fixes: (1) the -2.79 ns path was `rst_n` (input port) fanning out to every control flop of mm_top: one kept-
hierarchy `cm_rstbuf` per channel port + one for the arbiters / W-order FIFO (13 copies; all blocks leave reset in
the same cycle, one cycle after rst_n; cycle counts unchanged).  (2) That exposed mm_rdport's packer: `flush`
(228 loads) -> `take` / dat select (257 loads) -> `dat` D, -6.34 ns.  A flush now also emits `comb` (words
0..hc-1 are hold[0..hc-1], the rest are past the request length, don't-care by the channel contract), so `dat`
has one data source and one load enable, and the enables are separate single-level 256-load nets.  Final worst
path: wrapper input flop of `rd_req_addr` -> lo + len - 1 -> range compares against the pending writes ->
`blk` -> `req_ready` -> request-queue write enables (two ~90-load nets), +2.61 ns.  Hold is pre-layout (P8).
**FIFO -> SRAM (mm_sram_bank) evaluated, none qualifies.**  Every large FIFO here needs one 256-bit (W: 264-bit)
push *and* one pop per cycle; the sram256x8 macro is single-port, 8 bit wide, 0.0678 mm2.  The rd0 return
FIFO (64 x 256 = 16,384 flops ~ 0.97 mm2 + its one-hot read mux, ~1.3 mm2) would need 2 x 32 macros (even/odd
banks, 4.34 mm2) for full rate, or 32 macros (2.17 mm2) at half rate; the 16-deep FIFOs are 4x smaller still.
Break-even would need >= ~100-deep 1-port or much narrower FIFOs.  Flops stay; area is dominated by the return
FIFOs (16,384 + 6 x 4,096 flops), the W FIFOs (3 x 16 x 264) and their read muxes.

## 6. Deviations
1. Keep bits from DRAM (test stand-in only, section 1; traffic in section 4), removable with MK_DRAM = 0.
2. Address generation for ROW / TRN / gather / mask stays in the engines' loaders (P3-P5 put it there); the
   memory system serves arbitrary (address, length) requests. DMA_COPY (ISA 0x30) not implemented (P6b).
3. EMB integration uses gen_vec's compacted vocabulary rows at EMB_WORD (the full 154 MB table does not fit the
   model).
4. Engine tbs gained `+define+MM_DRAM` blocks; their default path is unchanged.  mm_attach_gm.svh takes
   `MM_MAXR / MM_RQL / MM_RFL` defines for the outstanding experiments (defaults = mm_top defaults).
5. The nominal DRAM model shares bandwidth between R and W (`+wfair=1`); the earlier strict-read-priority
   setting is kept as a reported sensitivity (GEMM 0.666).  Which policy the real controller has is off-die (R3).
6. The vector unit's 1-beat request pattern costs ~2x on LN-class ops at 800 MB/s (section 4); the fix is a P4
   RTL change, not made here.
7. `make lint` PASS (44 modules, 3 min 32 s); `make model-check` PASS (GATE L1 vs L0, 1 min 23 s).
