# LoWA-TA budget (P0)

All numbers: `python3 src/rtl/docs/calc/budget.py [--L n] [--fmhz f]`. "Measured" = from a tool run on
this machine; everything else is an **assumption flagged for P2/P8 to measure**.

## 1. Anchors

| quantity | value | status |
|---|---|---|
| fp32 RNE adder (GanAccelAsic `mb_fp32_add_rne`), Yosys/abc, tt | 24,143 um2 combinational | measured (src/syn/build) |
| same adder registered, ss_125C_3v00 | 26.2 ns logic, 26.93 ns reg-to-reg, +12.13 ns slack at 40 ns; routed +14.08 ns | measured (llmdocs/asic_flow_tutorial.md worked example) |
| fp16 multiplier (11 x 11 significand), tt | 14,793 um2 | measured |
| fp16xfp16 + fp32 accumulate lane, with regs | 52,520 um2 | measured |
| dfxtp_2 flop | 59.3 um2 | measured (cell area) |
| sram256x8 (ocd) | 301.3 x 224.9 um = 0.0678 mm2 for 256 B; min period 17.09 ns, CLK->Q 12.6-14.8 ns (ss) | LEF / Liberty |
| dla_pe 15,913 um2 | not used beyond "an int PE is ~10x smaller than an fp32 FMA" | weak anchor |
| **fp32 FMA, `hw/fp/fp_fma` STAGES = 4** | **119,410 um2** incl. its 363 pipeline flops; closes 40 ns at ss with +16.5 ns (STAGES = 2: 104,189 um2, +9.6 ns) | **measured (P2, `make synth-fp`, Yosys tt, pre-layout)** |
| **fp32 FMA PE (lambda = 4)** | **132,929 um2** = measured FMA + (4 x 32 acc + 100 operand) flops. P0 assumed 171,679 um2 (-23 %) | measured FMA + flop count |
| SFU per vector lane, `hw/fp/fp_sfu` (LF = 2, 4 threads, own FMA) | 222,855 um2 (P0 assumed 60,000 on top of the lane FMA) | measured (P2) |
| std-cell utilisation 0.60, macro halo x1.25 | | assumption |

Clock: the adder alone is 26.2 ns at ss, so a single-cycle FMA would be ~60 ns. With lambda = 4 stages
(mul tree / tree + align / add / normalise + round) each stage is ~15-20 ns, inside **40 ns (25 MHz) at
ss_125C_3v00**, the point the prior chip closed on this flow. SRAM (17.1 ns min period) is not the limit.
33-40 MHz looks reachable but is not claimed before P2 measures the FMA stages.

## 2. Chosen PE count: 16 x 16 = 256 (parameterised R, C, V, lambda)

| config | PEs | step @ L=128 (overlapped / serial) | DRAM peak | macros (KiB) | logic mm2 | core mm2 |
|---|---|---|---|---|---|---|
| **full16** | 256 | **3.29 G cycles = 131.6 s** / 142.1 s (P6b DRAM-charged; P6a 135.0 s; compute-only 129.4 / 139.8 s; read-priority DRAM 185.6 s) | 800 MB/s | 584 (146) | 43.5 | **122** |
| full16-lean (XBUF, ACC 1/4) | 256 | 297.3 s (DRAM-charged, P6b; P6a 300.7 s; compute-only 297.0 s) | 2,000 MB/s | 392 (98) | 43.5 | 106 |
| mid8 | 64 | 475.1 s (not DRAM-bound) | 400 MB/s | 296 (74) | 13.3 | 47 |
| shuttle2x1 | 2 | 14,662 s | 100 MB/s | 9 (2) | 0.9 | 2.2 |
| shuttle1x1 | 1 | 28,144 s | 100 MB/s | 9 (2) | 0.7 | 2.0 |

(P2: logic and core use the measured FMA and SFU areas; P0's assumption gave 51.5 / 116, 51.5 / 104, 14.9 / 40,
0.8 / 2.4, 0.7 / 2.1. `budget.py --p0` reproduces those.)

(P8c: the macros column counts the on-chip RAM the RTL actually instantiates, `budget.py onchip_words`:
full16 37,376 words = 146 KiB = gemm 24,576 (X / W buffers 2 x 4,096 each + ACC 2 halves x 4,096) + vector
unit 9,728 (2 x 256-word tile buffers per lane + 6 column-vector banks) + LoWA 3,072 (dY / X rows 2 x 768, atom
accumulator 256, atom centres 256, merge directory + coefficients 2 x 512); the DMA FIFOs are flops.  P0 to P8b
quoted 356 macros = 89 KiB, the P0 plan: single ACC, no vector-unit RAM, LoWA 1,792 words, DMA FIFOs as
macros.  Core estimates follow; the cycle and DRAM numbers do not depend on it.  docs/p8c_report.md section 5.)

Why 16 x 16: (a) operand SRAM grows with R + C while compute grows with R * C, so the smallest array at
which PE area and macro area are of the same order is ~16 x 16 (8 x 8: macros 12.2 mm2 vs PEs 11.0 mm2,
per-step traffic 1.9x higher); (b) at lambda = 4 the X side needs only R/4 = 4 words/cycle; (c) 32 x 32
would be a ~400 mm2 core (4x PE area) which is not a credible GF180 die. The full16 core (~116 mm2,
~10.8 mm square before pads) is already a full-reticle-class GF180 die: it is a **projection**, not a
shuttle design.

Step breakdown, full16, L_PAD = 128 (cycles): GEMM 2,728.5 M (array utilisation 0.964 with the P3-measured
engine model, docs/p3_report.md section 5: full chunks stream without the P0 skew bubble, 100 + 24/chunk
cycles per block, loader-bound when M < 64, GELU epilogues at 0.43 / 0.32 elements per cycle; the loss is
the CLS-pruned layer-11 attention with M = 1 and the B = 32 head GEMMs), vector 54.9 M, LoWA sampled grad
241.3 M (overlapped with the array; serial if not), optimizer + embeddings 1.0 M.
MACs/step: 673.7 G with CLS pruning (722.0 G without). LoWA sampled products 1.21 G MAC-equivalents;
merge hits 19.7 M RMW/step.

Real GLUE lengths (full16, step time at the mean padded batch length, rounded up to 16; approximate):

| task | L_PAD | s/step | 30-epoch run |
|---|---|---|---|
| CoLA | 32 | 30.1 | 8,040 steps = 2.8 days |
| STS-B | 80 | 79.5 | 5,400 steps = 5.0 days |
| MRPC | 96 | 96.1 | 3,450 steps = 3.8 days |
| RTE | 128 | 131.6 | 2,340 steps = 3.6 days |
| SST-2 | 48 | 45.5 | 63,150 steps = 33 days |
| MNLI | 96 | 96.1 | 368,160 steps = 410 days |

For scale: the same step on one A40 GPU takes well under a second. The ASIC case rests on energy
and on-device training, not speed; P8 must report energy/step (open question 7).

**P6a: DRAM-charged step time (`budget.py`, default; `--no-dram` = compute only, `--dram-rprio` = read-priority
DRAM).**  Each GEMM costs max(engine cycles, its bytes / (32 B x EFF_GEMM)) and each vector op max(kernel cycles,
element-passes x 5.50 B / (32 B x 0.309)); the efficiencies are measured with hw/mem + the DRAM model at
800 MB/s, 24-cycle latency (docs/p6a_report.md section 4; keep-bit stand-in traffic excluded, ruling R7):
GEMM real_ffn1_256 2,557,440 beats in 2,576,250 cycles = **0.993** with R and W sharing the bus (nominal), 0.666
(3,841,682 cycles) if the DRAM serves writes only on spare bandwidth while reads stream; vector real_ln_fwd
1,623,040 beats in 5,255,246 cycles = **0.309** (the vector unit's 1-beat requests). full16, L = 128: GEMM
2,728.5 -> 2,777.9 M cycles, vector 499.3 -> 591.3 M, step **129.4 s -> 135.0 s** (overlapped; serial 139.8 ->
145.5 s); with a read-priority DRAM 189.0 s.  The GLUE table uses the DRAM-charged numbers; the breakdown
paragraph above keeps the compute-only figures of P3/P4.

**P6b: vector burst order (ruling R8).**  vu_loader now requests each lane's row segment as contiguous 4-beat
(32-word) bursts into a 4-beat per-lane holder FIFO (was: one 1-beat request per row per 8-word block).  Measured
real_ln_fwd on hw/mem + the nominal DRAM model: 5,255,246 -> **2,963,526 cycles**, vector efficiency 0.309 ->
**0.548** (1,623,040 beats; compute bound 2,656,302 = 1.12x).  budget.py EFF_VEC = 0.548: full16, L = 128, vector
591.3 -> 506.7 M cycles, step **135.0 s -> 131.6 s** (overlapped; serial 145.5 -> 142.1 s; read-priority DRAM
189.0 -> 185.6 s).  Writes stay row-interleaved 1-beat (row-contiguous write visits measured slower, docs/p6b_report.md).

## 3. DRAM bandwidth and the SRAM trade
Bytes per MAC from operand reloads = 4 (1/Br + 1/Bc). full16 holds Br = Bc = 64 -> 0.125 B/MAC ->
32 B/cycle in GEMM phases = **800 MB/s at 25 MHz**, 94.6 GB/step at L = 128 (87.9 GB of it operand
reloads). Halving the traffic needs Br = Bc = 128: ACC 4x (256 macros, +17 mm2). The lean variant saves
144 macros (9.8 mm2) for 2.5x the bandwidth. The default keeps 800 MB/s because a 256-bit AXI at
core clock is internal; what the off-chip link can carry is open question 3.

## 4. Area (full16, assumptions above)
PE array 34.0 mm2 (measured FMA, P2); 16 vector lanes incl. measured SFU + transpose stagers ~7.6 mm2;
LoWA unit ~0.3 mm2; 8 Threefry instances ~0.8 mm2; sequencer + DMA + CSRs ~0.8 mm2 (assumed) -> logic
43.5 mm2 -> 72.5 mm2 placed at 0.60; 356 macros = 24.1 mm2 -> 30.2 mm2 with halo. **Core ~103 mm2**
(P0: 116 mm2), plus pad ring and DRAM interface. The FMA is now a pre-layout Yosys number (tt cell area);
the remaining uncertainty is placement utilisation and the not-yet-built blocks.

## 5. What fits on the shuttle block (1675 x 1110 um = 1.86 mm2, 11 pads)
- Macros: the prior chip placed 10 sram256x8 in this block, i.e. 0.68 mm2 of macros (~0.85 mm2 with
  halo), leaving ~1.0 mm2 for cells, ~0.6 mm2 of cell area at 0.60 utilisation.
- With the measured FMA (0.10-0.12 mm2 at STAGES 2-4) that is about **5 fp32 FMAs plus control**, but the
  SFU (0.22 mm2 incl. its own FMA) is larger than P0 assumed. The smallest complete core (1 PE, 1 vector
  lane + SFU, 1 Threefry, sequencer, 12 macros) is ~2.3 mm2 (P0: 2.1): **it still does not fit**. P2
  measured the FMA PE at 0.13 mm2 (lambda 4), above the ~0.11 mm2 P0 named as the fit condition; it would
  fit only with the lane FMA shared with the SFU (P4's choice) and fewer macros.
- 11 pads cannot carry an AXI4 DRAM interface in any case (PLAN fixed decision 7).
So the honest plan is: P8 runs PnR of the **full16 core netlist as a macro-backed block** (timing,
area, power at a stated PE count), and separately of a shuttle-sized **test vehicle** (FMA + SFU +
LoWA merge/sampled engine + a serial scan/DRAM-model port) if the PI wants silicon. The full chip is a
projection.
