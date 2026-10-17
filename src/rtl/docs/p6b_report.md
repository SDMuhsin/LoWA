# P6b report: on-chip dropout mask generator (R7) + vector-unit burst order (R8)

## 1. Dropout mask generator (hw/rng, ruling R7)

**Contract** = model/rng.py `keep_mask` (docs/dataflow.md section 7): element e of the mask of (seed, step, site, layer)
uses Threefry-2x32-13 block c0 = e >> 1, c1 = step << 8 | site << 4 | layer, key (seed[31:0], seed[63:32]), word e & 1;
keep = (word >> 8) >= 1677722.

| module | function |
|---|---|
| `rn_ks` | key schedule + counter formation (combinational): x1i = c1 + k1, the 3 injection pairs (ks[j mod 3], ks[(j+1) mod 3] + j), ks2 = 0x1BD11BDA ^ k0 ^ k1 |
| `rn_tf` | one Threefry block per cycle: 14 ARX units (x0 = c0 + k0, 13 rounds with the 3 key injections) + the two keep compares, fully unrolled, `STAGES` evenly spread registers (default 3, latency 3); raw words are a test port |
| `rn_mkgen` | one mk channel: config registers (seed, step, site, layer), NCORE `rn_tf` pipelines, a request needs NB = MKW/2 + 1 blocks (p0 = idx >> 1 ... p0 + 8; the 9th only for odd idx), issued in NSEQ = ceil(NB / NCORE) cycles (ready in the last), tags beside the cores, assembly + odd shift, registered response. Latency STAGES + 2 = 5 cycles. |

**Order.** The mk interface is unchanged (request = element index, response = the MKW keep bits of idx .. idx + MKW - 1,
in order, no ready). The engines already generate the index streams (GEMM epilogue: C element index of each 16-column
group; vector: base + row * stride + 16 g, rows inner), and the backward pass regenerates a mask by issuing the same
indices with the same (step, site, layer), so a pure function of the index serves every consumer in its own order.
"GEMM input-side dropout": the GEMM engine has none (P3 left drop-bwd on the A operand to an ELEM pass, dataflow.md
section 2), so its only consumer is the epilogue.

**Throughput.** Vector unit: vu_mask issues up to 1 request (16 bits) per cycle = **16 bits/cycle** peak (V = 16 lanes x
1 element). GEMM 16x16: the epilogue drains DW = ROWS*COLS/64 = 4 elements per cycle = **4 bits/cycle** (one 16-bit
request per 4 cycles; 8x8: 1 bit/cycle). rn_mkgen sustains MKW / NSEQ bits per cycle: NCORE 9 = 16 bits/cycle (vector
channel), NCORE 3 = 5.33 bits/cycle (GEMM channel), measured 1.000 / 3.000 cycles per request gap-free. The 13 rounds +
4 adds fit 3 stages at 40 ns with +10.5 ns ss slack (2 stages: +2.1 ns).

**Integration.** mm_top `MK_DRAM = 0` is now the default: one rn_mkgen per keep-bit channel (`MKNC0` = 3 for channel 0
= GEMM, `MKNC` = 9 for the others = vector, `MKST` = 3), configured by new ports `mk_seed[63:0]`, `mk_step[23:0]`,
`mk_site / mk_layer[4*NM]` (the P6c sequencer drives them from the step descriptor and the instruction). The external
`xmk_*` pass-through ports are removed. `MK_DRAM = 1` keeps mm_mkport (DRAM stand-in) for debug; tb_mm_unit and the
existing stand-in integration cases select it explicitly.

## 2. Tests (`make test-rng`, TEST-RNG PASS, build/test-rng/summary.txt)
Goldens only from model/rng.py (`threefry2x32_13`, `KAT`, `keep_mask`) and, end to end, from L1 via the P3/P4 generators.

| test | vectors | result |
|---|---|---|
| rn_tf + rn_ks, STAGES 3: Random123 KAT (3) + 1,048,576 random (c0, c1, key; key per 256) + corners, 10 % gaps | 1,048,579 | PASS |
| rn_tf STAGES 1 / 2 / 7 / 14 (65,539 each, 30 % gaps) | 262,156 | PASS |
| rn_mkgen, **real step-B attention-probability mask, layer 0** (32x12x128x128, 6.29 M bits), vu_mask order, NCORE 9 | 393,216 requests | PASS, 1.000 cycle/request |
| rn_mkgen, **real step-B FFN-output (hidden) mask, layer 5** (4096x768), GEMM epilogue order, NCORE 3 | 196,608 | PASS, 3.000 cycles/request |
| rn_mkgen random config, random odd/even idx: MKW 16 x NCORE 9/3/1/5 x STAGES 3/2/1, MKW 4 x NCORE 3/2/1, MKW 2 x NCORE 1/2; gap 0 and 30 % | 20 x 60,004 | PASS (throughput = ceil(NB/NCORE) checked) |
| vu_top + rn_mkgen (MK_GEN): **real_smax** (attn L0) | 6,389,760 words, 393,216 responses | PASS, 0 keep-bit mismatches vs L1 |
| vu_top + rn_mkgen: **real_smax_bwd** (attn L11, backward regeneration) | 6,291,456 / 786,432 | PASS, 0 |
| vu_top + rn_mkgen: **real_emb** (emb, hidden) | 3,145,728 / 196,608 | PASS, 0 |
| gm_top 16x16 + rn_mkgen (NCORE 3): **real_attnout** (attn_out L0, epilogue dropout) | 3,145,728 / 196,608 | PASS, 0; 9,698,537 cycles = the ideal-mask run (GEMM rate met) |
| gm_top + rn_mkgen: ep_bias_drop_resid | 2,627 / 185 | PASS |
| test-mem: gm_top + mm_top (MK_DRAM = 0) ep_bias_drop_resid, nominal + stress DRAM | 2,627 | PASS, 0 keep-bit beats on AXI (stand-in: 22), 6,342 / 11,750 cycles |

A mutated golden is reported as FAIL (checked by hand on tb_rn_tf).

## 3. Vector-unit burst order (ruling R8)
**vu_loader rd0:** one request per row per block of 8 x 2^HBL words (default HBL = 2: 32 words = a 4-beat burst), rows
inner (a segment <= 32 words is one burst per row); the per-lane 1-beat holding register became a 2^HBL-beat FIFO
(data + word count + tile address + operand per entry), so a burst lands at 1 beat/cycle while 16 lanes drain one word
per cycle each. Beats still carry 8 consecutive words of one row; lane = row, the tile-buffer contents and the
reduction order are unchanged (numerics identical).
**vu_writer:** a `WK` option writes up to WK groups of a row back to back (row-contiguous, coalesced by mm_wrport), but
it measured slower (row visits stall the lockstep lanes), so the default stays WK = 1 (P4 order); see the table.

real_ln_fwd (4096 x 768) on vu_top + mm_top + tb_dram, nominal DRAM (800 MB/s, lat 24, `+wfair=1`):

| variant | cycles | ARs / AWs | efficiency (1,623,040 beats / cycles) |
|---|---|---|---|
| P6a (1-beat requests) | 5,255,246 | 1,185,792 / 393,728 | 0.309 |
| **HBL 2 (4-beat row bursts), WK 1 — adopted** | **2,963,526** | 301,056 / 393,728 | **0.548** |
| HBL 3 (8-beat) | 2,984,014 | 153,600 / 393,728 | 0.544 |
| HBL 2 + WK 2 (NOB 2) | 3,348,576 | 301,056 / 117,248 | 0.485 |
| HBL 2 + WK 4 (NOB 4) | 3,356,732 | 301,056 / 209,408 | 0.484 |
| read-priority DRAM (`+wfair=0`), adopted | 3,250,246 | | 0.499 |

The compute bound (ideal memory) is 2,656,302 (P4); 2,963,526 is 1.12x of it (P6a: 1.98x). Ideal-memory cycles of the
real vector cases move by +0.4..0.9 % (holder FIFO drain order), all bit-exact.
**Budget** (docs/calc/budget.py EFF_VEC = 0.548, docs/budget.md): full16, L = 128, vector 591.3 -> 506.7 M cycles, step
**135.0 s -> 131.6 s** nominal (serial 145.5 -> 142.1 s; read-priority DRAM 189.0 -> 185.6 s; full16-lean 300.7 -> 297.3 s).

## 4. Area and slack (40 ns, ss_125C_3v00 after repair_design)

| block | area tt | ss slack | note |
|---|---|---|---|
| rn_mkgen NCORE 9, STAGES 3 (vector rate) | 0.684 mm2 (25,392 cells, 1,608 flops) | **+10.45 ns** | `make synth-rng` PASS |
| rn_mkgen NCORE 3, STAGES 3 (GEMM rate) | 0.263 mm2 (9,524 cells) | **+14.11 ns** | |
| rn_mkgen NCORE 9, STAGES 2 | 0.649 mm2 | +2.14 ns | sensitivity |
| vu_top V = 16 (HBL 2) | 18.16 mm2 std cells (P4: 16.69) | **+2.56 ns** (P4 +5.30) | `make synth-vec` PASS; +1.47 mm2 = the 4-beat holder FIFOs (16 x 4 x 256 bit) |
| mm_top full chip (NR 7, NM 2 generators, NW 3) | 10.22 mm2 (324,879 cells; P6a 9.74) | **-3.79 ns (FAIL, open)** | `make synth-mem` FAIL; same RTL with MK_DRAM = 1: +11.64 ns, 9.73 mm2 |

## 5. Gates
`make test-rng` PASS, `make test-vec` PASS (all unit, stress, V = 4, Icarus and real cases bit-exact), `make test-mem`
PASS (47 rows incl. the generator case; test-mem-quick PASS again after the WAR edit below), `make synth-vec` / `synth-rng` PASS,
**`make synth-mem` FAIL (-3.79 ns, section 6.0)**, `make lint` PASS
(47 modules, 125 Verilator runs incl. new rn_* / vu_loader HBL / vu_writer WK sweeps), `make model-check` PASS
(GATE L1 vs L0). test-gemm / test-lowa not re-run: hw/gemm and hw/lowa are unchanged; tb_gm_top changed only inside
`ifdef MK_GEN`.

## 6. Deviations / open points
0. **(Closed by P6c: flush registered, +0.73 ns, docs/p6c_report.md section 1.)** mm_top missed 40 ns at ss with the generators (-3.57 ns, then -3.79 ns after one fix attempt). Critical
   path: wrapper flop `q_ra` (an engine read-request address) -> WAR overlap of the offered read request vs the write
   beat (`oi` in mm_top's g_wr loop) -> `w_blk` -> mm_wrport `wr_ready` -> write-queue `q_line` enables (wr channel 2).
   The delay is in two high-fanout nets (nor3_2 7.7 ns + 11.8 ns), not in the arithmetic and not in rn_mkgen; with the
   identical RTL at MK_DRAM = 1 the block closes +11.64 ns, so ABC's mapping of this net changed when the keep-bit AR
   requesters left the arbiter. The one attempt (single-subtract overlap test for `oi`, logically equivalent, kept,
   test-mem-quick PASS) did not help. Suggested fix (P6c/P8): register `w_blk` (or the same-cycle read-offer term) and
   take the write beat one cycle later, or split the `wr_ready` -> queue-enable net with kept-hierarchy buffers as P6a
   did for reset (cm_rstbuf style). See docs/p6b_handoff.md.
1. Writes stay row-interleaved 1-beat bursts (393,728 AWs): the row-contiguous writer (WK > 1) is implemented and
   bit-exact but slower (section 3); the remaining 12 % over the compute bound is mostly the write side.
2. The 4-beat lane FIFOs cost +1.47 mm2 (+8.8 % vu_top) and 2.7 ns of slack; HBL = 1 would halve it (not measured).
3. Measured with the vector rd0 on mm_top channel 0 (64-beat return FIFO, as P6a). In the full-chip map vu rd0 is not
   channel 0 (16-beat FIFO, 4 bursts of 4 beats in flight); P6c/P7 should give vu rd0 the deeper FIFO or re-measure.
4. mm_top port change: `xmk_*` removed, `mk_seed / mk_step / mk_site / mk_layer` added; MK_DRAM default 1 -> 0.
5. Configuration must be stable from one cycle before a channel's first request until it is idle (P6c sequencer rule);
   the generator holds no per-request state beyond its 5-cycle pipeline.
6. GEMM channel uses NCORE 3 (16 bits per 3 cycles >= 4 bits/cycle); real_attnout cycles are unchanged vs ideal masks.
