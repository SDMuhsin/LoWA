# P4 report: vector / SFU unit (hw/vec), L2 bit-exact vs L1

Results (final RTL): `make test-vec` PASS (V = 16 Verilator unit + real roberta-base cases, stress with memory stalls,
V = 4 Verilator and Icarus smoke; every case bit-exact vs L1, section 3), `make synth-vec` PASS (vu_top V = 16:
**+5.30 ns** setup slack at ss_125C_3v00 after repair_design, section 5), `make lint` PASS (32 modules),
`make gen-check` PASS (vu_ucode ROM up to date, micro-programs reproduce the L1 entry points op by op),
`make model-check` PASS after the L1 additions (`mse` entry point; canonical NaN in the primitives, section 6).

## 1. What exists
| file | role |
|---|---|
| `flow/gen/vu_ucode.py` -> `hw/vec/vu_ucode.v` | kernel micro-programs (one per pass: LN sums, softmax exp/scale, CE, ...) and the per-op phase tables; `check` runs every vector op through a Python emulator of the lane on the L1 primitives and compares with the L1 per-op entry points |
| `hw/vec/vu_lane.v` | one lane: 2 x 256-word tile buffers (mm_sram_bank), control / operand register stages, one fp_fma under micro-ops, accumulator unit (fp_fma + chunk logic), max / min (fp_cmp), fp_sfu behind in / out FIFOs, STORE FIFO, 2 x 8-word output registers, lane scalars s0..s3 |
| `hw/vec/vu_seq.v` | sequencer: planner (tiles x phases x row segments, two-entry step queue, double buffering), element issue, slot turns, in-order FIFO-op gating, cross-lane sequential sum, scalar phases |
| `hw/vec/vu_loader.v` | rd0 tile loads (row beats -> lane holding registers -> tile buffers, embedding gather + position ids), rd1 loads (per-row scalars, ids, column vectors / key-mask rows) |
| `hw/vec/vu_mask.v` | dropout keep bits over the mask channel, per-lane FIFOs |
| `hw/vec/vu_writer.v` | output path: lane registers -> 256-bit beats with word masks (any word alignment), per-row stats and scalar writes |
| `hw/vec/vu_top.v` | the unit: 16 lanes, 6 column-vector banks, reset tree (cm_rstbuf per block and per lane), status reductions |
| `tb/vec/gen_vec.py`, `tb_vu_top.sv`, `run_tests.py` | cases (unit + real), memory model, driver (`make test-vec`) |
| `flow/synth/wrap_vu_{lane,top}.v`, `synth_vec.sh`, `run_vec.py` | synthesis / STA (`make synth-vec`) |
| `model/hwmodel`: `mse` entry point, `lw_mse` | MSE loss + gradient (spec.md 4, written out in docs/arith.md style in lwta.c) |

## 2. Architecture
**Descriptors.** vu_top executes one vector descriptor (`d_*`, held from `start` to `done`): op id
(LN_FWD, LN_BWD, SMAX_FWD, SMAX_REC, SMAX_BWD, EMB, CE, MSE, GN_ROWS, GN_FIN, E_ADD / E_MUL / E_SCALE /
E_SCALER / E_DROP / E_TANH / E_TANHB / E_COPY), rows x cols, operand bases / row strides (A, B, out),
column vectors (gamma, beta, token-type row; cv0 doubles as the key-mask bit row of batch b), per-row
scalars (stats in / out, labels, weights), scalar output base, keep-bit index base / row stride, ids base
(gather), constants C0..C3 (INV_D, eps, 0.125, inv_b, 2/B) and DS.  The ISA's GNORM is a sequence of
28 GN_ROWS + GN_FIN + 28 E_SCALER descriptors (section 6).

**Lane = row.** Every row reduction (LN 768, softmax L, CE C, GN rows) runs sequentially inside one lane,
so the order is the contract's and does not depend on V.  A tile is V = 16 rows; a row longer than the
256-word tile buffer is processed in segments: the lane accumulator keeps the chunk partial / count across
segments (runs of KC = 64 from +0, partials added in order), so d = 768 and the directed 4096-wide rows give
the contract's chunked sums; softmax rows (<= 256) stay resident across their phases and the exp pass
stores e_j in place.  Cross-row sums (CE over b, GN over rows) are a sequential `fma(v, 1, acc)` over the
lanes in row order (chunked for GN, carried across tiles and reset per tensor); the 28-term norm total is
`fma(n_i, n_i, s2)` in a lane scalar across descriptors.

**Lane datapath.** One fp_fma (STAGES 2) executes every rounded op as `fma` (`mul = fma(x, y, -0)`,
`add = fma(x, 1, y)`, `sub = fma(x, 1, -y)`, bit-identical to L1 as in P2).  NS = 8 barrel slots take turns
(slot = cycle mod 8); each slot holds one element (r0 = A[j], r1 = B[j], r2..r4 = cvec[j], temp r5, keep /
key-mask / label flags) and runs the kernel's micro-program; a result is written back 4 cycles after its
turn, before the slot's next turn.  Ops that enter an ordered resource (SFU push / pop, MAX, ACC0/1, OUT,
STORE) are gated by per-type element-sequence counters, so FIFO order = element order whatever the slot
timing.  The accumulator unit is a second fp_fma with feedback (`p = fma(x, y, p)`, every 64 pushes
`tot = add(tot, p)`, flush at the row end; sequential mode for softmax / CE / the LoWA-style sums); the
max unit is fp_cmp with one-cycle forwarding; the SFU is fp_sfu (LF 2, 4 threads) with 8-deep in / out
FIFOs; dropout / key-mask zeroing is a flag on the op (`keep ? mul(x, DS) : +0`), the CE one-hot is the
operand `is_y ? -1 : -0` of `sub(p, onehot)`.  Control and operand register stages (one-hot slot and
source selects through cm_ohmux) keep the sequencer fanout and the operand muxes off the FMA path.

**Memory side** = the GEMM engine's (P6 attaches the same DMA): rd0 tile beats (one 8-word beat per row per
8-word column block; consecutive beats go to different lanes, each lane drains its holding register one
word per cycle into its single-port bank), rd1 (scalars, ids, column vectors: 64-word requests, one word per
cycle into the cvec banks), wr (8-word groups per row, split at the 32-byte boundary with word masks), mk
(16 keep bits per request, element index = base + row * stride + j).  Steps double-buffer the tile buffers
and the column-vector banks (parity per step); dual-operand steps (LN_BWD, SMAX_BWD, EMB, E_ADD / E_MUL /
E_TANHB) use both buffers and are not overlapped.  Embedding gather: the loader reads the 16 ids of the
tile, computes the position ids (`pos = nz ? cnt + 2 : 1`, running count per sequence of `d_emb_l` tokens,
restarted at tile boundaries that start a sequence) and issues `base + id * stride` row reads for the word
and position tables; `e = add(add(w, tt0), pos)` is recomputed in every LN pass (bit-identical).

## 3. Verification (L2 = bit-exact vs L1)
Goldens only from `model/hwmodel` entry points: `ln_fwd ln_bwd smax_fwd smax_bwd ce mse gnorm_clip tanh
tanh_bwd drop add mul scale recip`.  The real cases replay `model/hwmodel/step.py` on the step-B batch with the
entry points hooked (inputs copied before the call, outputs recorded), so the data are exactly the L1
step's tensors (layer-0 softmax with the real key mask and the chip's dropout masks; layer-11 softmax
backward and LN2 backward; layer-0 LN1; the embedding path on the real ids with a compacted word table; the
real logits / labels; the 28 real gradient tensors).  The testbench compares every written word with `!==`,
counts elements and mismatches, prints one PASS / FAIL line; the memory model (in-order read channels with
programmable latency, random stalls on every channel) is the only non-L1 component.  Fault injection (one
corrupted expected word) fails the case.

| config | set | vectors | mismatches | cycles (model x) |
|---|---|---|---|---|
| V=16 | elem: add / mul / scale / drop / tanh / tanh-bwd / copy on 20 x 50 with corner rows, 1 x 4096, 7 x 1, 3 x 777 drop | 13,434 | 0 | 33,603 |
| V=16 | ln_fwd: 37x200, 16x64, 20x65 (chunk tails), 7x1, 3x4096, 17x768, 1x128; rows of identical values, subnormal (DAZ), max magnitude, +-0, tiny | 35,373 | 0 | 83,017 |
| V=16 | ln_bwd: the same shapes with L1 stats | 35,075 | 0 | 133,670 |
| V=16 | smax: fwd + recompute + bwd on (B,H,L) = (2,3,48) with a fully masked batch, (1,1,128), (1,2,1), (3,1,16), (1,1,4), (1,1,256); identical-value and max-magnitude rows; dropout | 291,042 | 0 | 353,353 |
| V=16 | emb: gather with pads (id 1), two sequences, d = 200 and 768, dropout | 58,752 | 0 | 110,809 |
| V=16 | loss: CE (32,3) (32,2) (5,16) (1,2) (17,3) with zero-weight rows / B_ACT < B, identical and max-magnitude logits; MSE B = 32, 5, 1 | 339 | 0 | 5,423 |
| V=16 | gnorm: 9 tensors incl. 1x256 x3, 70x200, 1x77, 3x77, 1x3, 2x4096, 65x64 (row and column chunk tails), a zero row; GN_FIN, scale by c | 27,442 | 0 | 48,879 |
| V=16 stress | elem, smax, loss, gnorm x {stall 25 % lat 3, stall 40 % lat 60, lat 1} | 12 runs | 0 | |
| V=16 | **real_smax**: layer-0 softmax fwd, 32x12x128x128 with the real key mask + dropout, P' + m, r | 6,389,760 | 0 | 9,216,055 (1.01) |
| V=16 | **real_smax_bwd**: layer-11 dS from the L1 P, dP', masks | 6,291,456 | 0 | 5,775,388 (1.08) |
| V=16 | **real_ln_fwd**: layer-0 LN1, 4096 x 768, y + mu + rstd | 3,153,920 | 0 | 2,656,302 (1.00) |
| V=16 | **real_ln_bwd**: layer-11 LN2 backward, 4096 x 768 | 3,145,728 | 0 | 4,669,468 (1.09) |
| V=16 | **real_emb**: the real batch (ids, positions, token type, LN, dropout) 4096 x 768 | 3,145,728 | 0 | 6,035,492 (1.10) |
| V=16 | **real_loss**: CE on the real logits (loss + dz) + head tanh and tanh-bwd on the real 32 x 768 | 49,217 | 0 | 53,401 |
| V=16 | **real_gnorm**: the 28 real tensors -> 28 norms, G, c, all 598,274 scaled gradients | 598,304 | 0 | 364,365 (0.92) |
| V=4 | smax_small (1,1,48) fwd + bwd, ln_small 6 x 70 fwd + bwd, each plain and stall 30 % lat 7 | 4 runs | 0 | |
| V=4 Icarus 13 (4-state) | smax_small (fwd + bwd), ln_small (fwd + bwd) at +lat=5 (MEMW 14, MASKW 13; compile 14 s) | 5,556 | 0 | 21,818 / 5,108 |

The final `make test-vec` table (33 rows, every one PASS, 0 mismatches; cycles vs the budget model in brackets, the
model having been fitted to an earlier run of the same cases) is `build/test-vec/summary.txt`.  Earlier development runs on the same
cases found and fixed: loader address pipeline off by one row (B operand), writer draining lanes in the
wrong order (deadlock), output-register credit by word count (partial registers), POP / keep-bit gating
lagging the FIFO pointers by two cycles, and -- Icarus only, 4-state -- the loader's rd1 job selection
read descriptor fields inside Verilog functions called from `assign` / `always @*`, which ivl does not put
in the sensitivity list (the job pointer stayed X and the unit hung; P3 saw the same class): the enables and
lengths are now wires (section 6).

## 4. Cycles vs model
Measured on the real tensors, V = 16, cycles per element per lane (= cycles / (tiles x row length)):

| op | measured | what bounds it |
|---|---|---|
| LN_FWD (768) | 13.51 | 3 passes x 3 segments, each segment re-loaded (2 cycles / column / operand on the 256-bit channel), output 2 cycles / column, per-tile scalar phases |
| LN_BWD | 23.75 | dual operand (z, dy) not overlapped: 4 cycles / column of loads per pass x 2 passes |
| EMB | 30.70 | dual gathered operands x 3 passes + 3 column vectors |
| SMAX_FWD (128) | 23.44 | exp pass SFU-bound: 13 cycles / element, plus max (1) and scale / output (2) passes, stats, key-mask row |
| SMAX_BWD | 14.69 | dual (P, dP') load 4 + rowdot 2 + dS / output 2 |
| ELEM (dual, 1024 x 768) | 7.13 | loads 4 + output 2 (single-operand ops ~3) |
| E_TANH | 25 | SFU-bound |
| GN rows (28 real tensors) | 9.74 | 24 tensors are one 256-word row: per-descriptor start-up, XL and sqrt dominate |

`docs/calc/budget.py` (`VEC_RATE`, `vec_cycles`, `step`) now uses these rates (P0 charged one element-pass
per lane per cycle).  Step budget, full16, L = 128: vector **499.3 M cycles** (P0 model 54.9 M), GEMM
2,728.5 M, total overlapped 3,228.9 M = **129.2 s** (P3 report: 111.4 s).  The levers, in order: the dual-
operand steps are not double-buffered (LN_BWD / EMB / SMAX_BWD: ~45 % of the vector time), one SFU per lane
(exp 1/13 per cycle: 2 x 12 x 6.3 M exps = 122 M cycles, the figure P2 anticipated), one 256-bit write
channel (2 cycles per column), and LN over 768 re-loading each segment per pass.

## 5. Synthesis and STA (`make synth-vec`, `build/synth/vec_table.md`)
Yosys 0.62 `synth -flatten -booth`, abc -D 40000, GF180 7-track tt library, OpenSTA at 40 ns; ss_125C_3v00
setup slack raw and after OpenROAD `repair_design` (resize only, as in P3).

| unit | parameters | cells | area tt (um2, std cells) | ss raw | ss repaired | tt raw |
|---|---|---|---|---|---|---|
| vu_lane | NS 8 | 32,422 | 962,132 | -2.31 | **+3.56** | +8.93 |
| vu_top | V 16 | 566,206 | 16,688,778 (incl. wrapper flops) | -17.86 | **+5.30** | -2.03 |

SRAM macros are blackboxes: 2 x 4 per lane (tile buffers) + 6 x 4 column-vector banks = **152 macros**
(10.3 mm2 at 0.0678 mm2 each; memory_map.md assumed no vector SRAM).  One lane = 0.96 mm2 (fp_fma x 2
0.21, fp_sfu 0.22, fp_cmp 0.01, 8 slots x 6 regs + FIFOs + output registers ~3.7 k flops, muxes);
16 lanes 15.4 mm2 + top 1.3 mm2, against P0's 7.6 mm2 for 16 lanes with stagers.  The first lane
synthesis failed at -5.99 ns after repair: the worst path was the sequencer's slot select (one flop, 17.9
ns clk-to-Q into 16 lanes x 192 mux selects) through the operand muxes into the FMA multiplier.  The fix is
structural, as in P3: a lane-local control register with replicated one-hot slot / source selects
(cm_ohmux) and an operand register before the FMA; the unit's reset is a cm_rstbuf tree (4 blocks + 16
lanes, 21 kept instances in the netlist).  Hold after repair is -2.1 ns pre-layout (no hold fixing in this
flow, P8), as in P3.

## 6. Deviations and open points (for the PI)
1. **Vector SRAM.** 152 sram256x8 macros (2 x 256-word tile buffers per lane, 6 column-vector banks) instead
   of the flop stagers of memory_map.md; with them a row segment is loaded once per pass instead of once
   per element-pass.  Rows up to 256 words are resident across the phases of an op; LN / GN rows of 768 (and
   the 4096-wide directed rows) are segmented, with the chunked sums carried across segments.  Softmax rows
   must fit one segment (L_PAD <= 128 does).
2. **Throughput.** Measured per-op rates (section 4) make the vector unit 499 M cycles per step (15 % of the
   step) instead of P0's 54.9 M; step time 129.2 s instead of 111.4 s.  Double-buffering the dual-operand
   steps (a third buffer, +64 macros) would remove about a third of that.
3. **GNORM / LOSS as descriptor sequences.** GNORM = 28 GN_ROWS (per tensor, XL state reset per tensor, G
   accumulator reset by the first) + GN_FIN (G, c written; c kept in lane scalar s3) + 28 E_SCALER (scale by
   s3); the P6 compiler emits them (the ISA's single GNORM with a tensor table is not implemented in the
   unit).  CE / MSE take inv_b (and 2/B_ACT) as descriptor constants (computed by the sequencer / host for
   the step), labels as int32 words, loss weights as fp32 per row; the loss is written as one word.
4. **MSE in L1.** No entry point existed; `lw_mse` / `H.mse` were added: `d = sub(z, y)`,
   `l_b = mul(mul(d, d), w_b)`, `tot = add(tot, l_b)` (b ascending), `loss = mul(tot, inv_b)`,
   `dz = mul(mul(d, inv2b), w_b)`, `inv2b = mul(2, inv_b)` (exact) -- spec.md's formulas left to right.
5. **Canonical NaN in L1.** `lwta.c`'s inline `fadd / fsub / fmul / ffma` returned x86's 0xFFC00000 for a
   generated NaN inside composite ops (a fully masked softmax row: `P = mul(0, recip(+0))`), while the
   exported element-wise entry points (and the RTL, arith.md section 1) return 0x7FC00000.  The
   canonicalisation moved into the primitives; non-NaN results are unchanged and `make model-check` still
   PASSes.  No real-data case produces a NaN.
6. **Position ids** are computed in the loader from the ids (`1 + cumsum(id != 1) * (id != 1)` per sequence of
   `d_emb_l` tokens, L_PAD a multiple of 16); the sequencer does not need to precompute them.
7. **Masks.** Key-mask rows are bit-packed words (bit j of word j >> 5 of batch row b, `km_rs` bytes per
   row), as the STEP_IN layout; dropout keep bits come from the mask channel 16 per request, re-fetched by
   every pass that needs them (deterministic PRNG).
8. **E_SCALER / E_SCALE, dropout backward.** The A-operand dropout backward of dataflow.md section 2 is the
   E_DROP descriptor (P3 left it to P4); E_SCALE takes the scalar from C0, E_SCALER from lane scalar s3.
9. **Icarus.** Signals read inside a function are not in the sensitivity of the `assign` / `always @*` that
   calls it under ivl: the loader's job enables / lengths are wires now (the functions take only arguments).
   Testbench memories MEMW 14 / MASKW 13 for the V = 4 smoke (2^17 words took 11 min to compile, P3's finding).
10. Timing closes after `repair_design` only (raw -17.9 ns at the top: unbuffered high-fanout descriptor and
    issue nets into 16 lanes, which resizing alone repairs), as the GEMM engine did; the lane itself is +3.56 ns.
