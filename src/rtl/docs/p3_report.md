# P3 report: GEMM engine (hw/gemm), L2 bit-exact vs L1

Results (sections 5-7, final RTL): `make test-gemm` PASS (Verilator 16x16 / 8x8 / 4x4 + Icarus 4x4, all bit-exact vs L1),
`make synth-gemm` PASS (gm_top 16x16 +4.23 ns at ss after repair, section 6), `make lint` PASS (25 modules),
`make model-check` PASS (no L1 numerics changed; no new L1 entry point was needed: the
goldens use `gemm_raw`, `gemm`, `gemm_nn`, `add_bias`, `scale`, `gelu`, `gelu_bwd`, `drop`, `add`,
`lowa_merge`, `smax_fwd`, `ln_fwd` and `fma`).

## 1. What exists
| file | role |
|---|---|
| `hw/mem/gf180mcu_ocd_ip_sram__sram256x8m8wm1.v` | the GF180 256 x 8 macro: behavioural model, `SYNTHESIS`-guarded `(* blackbox *)` (vendor name = file name, the one exception to the `mm_` prefix) |
| `hw/mem/mm_sram_bank.v` | WORDS x WIDTH single-port bank from the macro (4 macros per 32-bit word, 256-word rows, row mux on Q) |
| `hw/gemm/gm_pe.v` | PE: fp_fma (STAGES 2) with the LAMBDA-deep feedback, +0 restart, LAMBDA result registers, drain-chain register, overrun check |
| `hw/gemm/gm_array.v` | ROWS x COLS array: a right, b / ctl down, drain chains down the columns |
| `hw/gemm/gm_xfeed.v` | X tiles: 2 x 8 lanes x 512 words (64 macros each), addressing skew, lane layout `(k + row) mod 8` |
| `hw/gemm/gm_wfeed.v` | W tiles: 2 x 16 lanes x 256 words, shared read + rotator + skew registers, lane layout `(c + k + 4q) mod 16`, merge RMW port |
| `hw/gemm/gm_loader.v` | tile fetch: ROW / TRN burst address generation, beat -> 8 lanes routing, buffer-free gating |
| `hw/gemm/gm_drain.v` | chunk-sum engine: injection schedule, DW = 4 fp_add lanes, ACC = 16 lanes x 512 words (2 halves x 2 parities x 4) |
| `hw/gemm/gm_pow2.v` | exact power-of-two scaling (= L1 `mul(x, 2^e)`) |
| `hw/common/cm_ohmux.v` | one-hot AND-OR read mux with replicated select slices (fanout control, section 6) |
| `hw/common/cm_rstbuf.v` | one registered reset copy, `(* keep_hierarchy *)` so synthesis keeps every instance (reset tree, section 6) |
| `hw/gemm/gm_gelu.v` | GELU / GELU' on a pool of NSFU = 16 fp_sfu with a 64-entry reorder buffer and the fp_mul / fp_add / fp_fma glue |
| `hw/gemm/gm_epilogue.v` | ACC read-out, bias / scale / activation / dropout / residual lanes, group assembler, beat writer, bias + aux + mask prefetch |
| `hw/gemm/gm_ctrl.v` | descriptor sequencer: block loop, loader feed, pass scheduling, drain scheduling, epilogue hand-off |
| `hw/gemm/gm_top.v` | the engine: all of the above behind the memory-side valid/ready channels |
| `tb/gemm/gen_pe.py`, `tb_gm_pe.sv` | PE vectors (L1 fma chains) and testbench |
| `tb/gemm/gen_gemm.py`, `tb_gm_top.sv`, `run_tests.py` | engine cases (unit + real data, all goldens from L1), memory model, driver (`make test-gemm`) |
| `flow/synth/wrap_gm_{pe,array,top}.v`, `synth_gemm.sh`, `run_gemm.py` | synthesis / STA (`make synth-gemm`) |

## 2. Architecture
**Array and PE.** Output-stationary ROWS x COLS (16 x 16) of gm_pe. A PE holds one fp_fma
(`FMA_STAGES = 2`, the P2 production setting) and keeps its LAMBDA = 4 chunk partials in flight
inside the FMA pipeline: the op for W row `c*4+q` at k is issued 4 cycles after the one at k-1, so
its c operand is the FMA output delayed 4 - 2 = 2 cycles. `ctl.first` (k = 0 of a chunk) selects +0
instead of the feedback: this is the contract's `p = +0; p = fma(a_k, b_k, p)`. At `ctl.last` the
result of each q is captured in `res[q]` and held until the drain injects it. The PE adds only a 2:1
mux on the FMA's c input to fp_fma's critical path: 130,068 um2 with +15.3 ns of setup slack at
ss_125C_3v00 (FMA_STAGES = 3: 134,487 / +16.6 ns; 4: 141,154 / +19.6 ns), so 2 stages is the choice
(section 6). Operands enter from the left (a, held 4 cycles per k by the feeder) and from the top
(b and the control word, one hop register per PE each); the chunk results leave through a 32-bit
drain register chain down each column.

**Pass and skew.** A pass streams one K chunk (KT <= 64 words) for one 16 x 64 micro-tile:
`4*KT` cycles. PE(r, c) consumes (k, q) r + c cycles after PE(0, 0), which consumes it 3 cycles
after `ps_start`. The X side gets its skew from addressing (row r's reads start r cycles later; a
token shifts down the rows), the W side from COLS*(COLS-1)/2 skew registers after the lane
rotator: the W read of all 16 lanes is unskewed (one address per cycle), and section 2.1 shows why
an addressed skew is impossible on the W side with one write per lane per cycle. Consecutive full
passes stream with no bubble (the W counter restarts in its last cycle; the X rows of two passes
overlap on disjoint lanes when KT mod 8 = 0, otherwise the controller waits for the X feeder).

**2.1 Buffer layouts (the P6 DMA contract).** Each operand lane is one 32-bit mm_sram_bank
(single port, one access per cycle), and every DMA beat (256 bits = 8 consecutive words of a DRAM
burst) must hit 8 distinct lanes so the write path takes one beat per cycle:
- X block (64 rows x 64 k): 8 lanes x 512 words. Word (r, k), r = block row = mt*16 + rr, lives in
  lane `(k + r) mod 8`, address `{mt, rr, k >> 3}`. ROW beats (one row, k0..k0+7) and TRN beats
  (rows r0..r0+7, one k) each cover 8 distinct lanes, and the 4 rows that need a word in the same
  cycle of a pass (rows phi + 4i read k = K - i) hit lanes K + phi + 3i, distinct because
  gcd(3, 8) = 1.
- W block (64 W rows n = 4c + q, 64 k): 16 lanes x 256 words. Word (n, k) lives in lane
  `(c + k + 4q) mod 16` (with 8 lanes, COLS <= 8: `(c + k + 2q) mod 8`), address `{k, q}`. ROW beats
  (8 consecutive k of one n) and TRN beats (n0..n0+7 at one k = two c's x four q's: {0, 4, 8, 12} and
  {1, 5, 9, 13}) cover 8 distinct lanes; the 16 columns read the same (k, q) in a cycle, so the
  rotation (k + 4q) mod 16 puts lane data on its column. An addressed skew (column c reading
  (k, q) c cycles later) would need a lane map injective on {(c, kq = T - c)} for every T *and*
  on both beat shapes; writing lane = c + sigma(kq) the first condition forces sigma(d) - d to be a
  permutation of Z16 over every window of 16 consecutive d, hence sigma(d + 16) = sigma(d), and
  then a ROW beat's k and k + 4 collide. So the W skew is in registers (0.23 mm2 at 16 x 16).
- ACC (2 halves x 64 x 64 partials): 16 lanes x 512 words, lane = (half, q & 1, j), address
  `{mt, r, g, q >> 1}` for output column 4*(4g + j) + q of micro-tile mt, row r.
- DMA bursts: ROW mode = one burst per matrix row (KT words), TRN mode = one burst per k (64 words
  of DRAM row k = the 64 lanes); words beyond the burst length are masked, nothing else is padded.
  The loader requests W before X so that the LoWA merge engine (P5, `mg_*` port on the next W
  buffer, `merge_busy` gates the block start) gets ~512 cycles before the block starts.

**2.2 Chunk sum (the contract).** After a pass the drain engine moves the 1024 partials out in
column groups of DW = 4 (one group = 64 cycles, 4 groups = 256 cycles = one full pass, so drains
of back-to-back full passes are back to back) in the order row 15 .. 0, q = 0 .. 3; PE(r, c) injects
res[q] 3*(15 - r) + q cycles after the group start, the word travels 15 - r hops and no two words
of a column collide (4r - 4r' = q - q' is impossible). Each word meets its ACC partial in an fp_add
(`ACC = add(ACC, p)`, old = +0 for the first chunk: `add(+0, p)` is really performed, because
p = -0 must give +0) and is written back 5 cycles after the read; reads and writes alternate the
two parity lanes every cycle, so one single-port lane never sees both. K tails: the last chunk's
pass is `4*KT` cycles with `last` at k = KT - 1, exactly the contract's short chunk. Output blocks
alternate ACC halves: block n's epilogue reads half n & 1 while block n + 1 accumulates in the
other (section 7, deviation 1).

**2.3 Drain / pass scheduling (gm_ctrl).** With S = ps_start + 3 the last k of a pass is captured
in PE(r, c), q at S + 4(KT-1) + q + r + c + 1 + L (L = FMA_STAGES); the injection of that word is at
Ds + g*64 + 3(15 - r) + q, Ds = the drain's injection origin. The controller starts the drain at
A = ps_start + DD + 1, DD = 4*KT + DOFF, DOFF = ROWS + DW + L + 1 - LAMBDA = 19 (the derivation gives
>= 18; one cycle of margin), so Ds = A + 2 satisfies capture + 1 <= injection for every (r, c, q).
A pass may start only if (a) the drain it schedules starts when the engine is free,
A' >= A_prev + 256, and (b) its first capture comes after the previously scheduled drain's last
injection, ps_start' + 4*KT' + L + 2 - 4 >= A_prev + 2 + 3*60 + 45 (margin 1). For full chunks both
hold with margin (A' - A_prev = 256 exactly, 514 >= 503), so there is no bubble; for short chunks
(KT < 64) the controller waits. Every PE flags a capture into an undrained register (`err[0]`),
the drain engine a start while busy (`err[3]`): neither fires in any test.

**2.4 Epilogue.** Per output block, after its last chunk's drains, the epilogue reads the ACC
half (DW = 4 words per cycle, column groups of 16, rows < M only, a micro-tile only once its
final sums are written and never while a drain touches that half), and applies, in the dataflow.md
order: `add(acc, bias[col])` (fp_add), `mul(y, 2^e)` (gm_pow2, exact), GELU or GELU' (gm_gelu),
`keep ? mul(y, DS) : +0` (fp_mul, keep bits from the mask channel, element index = C word offset),
`add(y, aux)` (fp_add, aux = residual from the rd1 channel). GELU-fwd stores u to `out2` and g to
C. gm_gelu serialises the elements (1 per cycle; 2 cycles each in backward mode), computes
`t = mul(u, A)`, `w = mul(mul(u, -0.5), u)` with fp_mul / gm_pow2, sends erf(t) (and exp(w)) to any of
the 16 fp_sfu that is ready, re-orders the results by tag in a 64-entry ROB, and finishes with
`add(1, er)`, `mul(., 0.5)`, `mul(h, s)` (fwd) or `pdf = mul(ex, B)`, `fma(u, pdf, cdf)`,
`mul(dg, f)` (bwd), op for op as docs/arith.md section 4 (the SFU's special cases are the SFU's).
The 16 SFUs would allow 0.67 (fwd) / 0.43 (bwd) elements per cycle; measured end to end (ROB,
credits, serializer) the epilogue sustains 0.43 (fwd) / 0.32 (bwd) against the 0.33 the FFN GEMMs
produce: FFN1 + GELU is array-bound (measured 1.000 x the array model), dgelu is co-limited
(1.004 x, section 5). Groups of 16 columns are assembled and written as two 256-bit beats with a word mask
(columns >= N). The epilogue of block n runs during block n + 1; block n + 2 (same ACC half) is
held until it is done.

**2.5 Interfaces (P6 / P5 / P4).** `rd0` tile requests (addr, len <= 64 words) + 256-bit in-order
beats; `rd1` bias / aux requests + beats; `wr` 256-bit beats + 8-bit word mask; `mk` keep-bit
requests (element index) + 16-bit responses; `mg_*` single-word RMW on the next W buffer +
`w_tile_done` + `merge_busy` (P5); the dropout-backward on the A operand is left to an ELEM pass
(P4, dataflow.md allows either; section 7). The descriptor (`d_*`) carries M, N, K0..K2
(GEMM_ACC2 = up to three K segments with their own A / B bases and the same strides; segment lengths
multiples of 64 except the last keep the contract's chunk sequence), batch counts and strides,
ROW / TRN modes, C / bias / aux / out2 bases, epilogue flags, scale exponent and DS.

## 3. Verification (L2 = bit-exact vs L1)
Goldens: `tb/gemm/gen_pe.py` applies the L1 `fma` entry point in the contract order (p = +0,
k ascending, one chain per output); `tb/gemm/gen_gemm.py` calls `H.gemm_raw` (batched, arbitrary
strides) and the per-op epilogue entry points, and for the real cases replays
`model/hwmodel/step.py`'s layer-0 forward on the step-B batch (`model/runs`, P1): embeddings, L1
`lowa_merge` with the step-B coefficients for Wq / Wv, Q / K / V, scores, `smax_fwd` with the
chip's masks, O = P'V, the attention-output projection, LN1 and FFN1. Testbenches compare every
written word with `!==`, count elements and mismatches, and print one PASS / FAIL line; the memory
model (word-addressed, in-order read channels with a programmable latency, random stalls on every
channel) is the only non-L1 component. Error flags of the engine are checked at the end of every run.

Tables: section 5.

## 4. Cycle model (docs/calc/budget.py)
`gemm_cycles` was rewritten from the measurements of section 5 (the P0 model charged R + C per
pass): per output block and K chunk the engine needs `max(micro-tiles x 4 x 64, (rows + cols) x KT / 8)`
cycles (array vs loader beats; a short last chunk costs a full drain slot) + 24, per block + 100,
a block cannot finish faster than its epilogue (`elements / ep_rate`, ep_rate = 4 words per cycle,
GELU 0.43, GELU' 0.32), plus the first tile load and the last block's epilogue once per
descriptor (batches stream).  The P0 loader assumption was never a bound; the measured loader is
(1 beat/cycle, exactly the array rate for 64-row blocks: M < 64 is loader-bound, which is what
the `full16-lean` config now shows).  Step budget (budget.md): GEMM 2,728.5 M cycles, array
utilisation 0.964, step 111.4 s overlapped (P0 model: 2,961.7 M / 0.889 / 120.7 s).

## 5. Results: L2 bit-exact vs L1 (Verilator unless stated; final RTL except where noted)
| config | set | elements | mismatches | cycles (model, measured/model) |
|---|---|---|---|---|
| gm_pe FMA_STAGES 2 | directed 168 + random 20,000 chunks, gaps 0 / 5; Icarus stages 2 / 3 / 4 | all | 0 | - |
| gm_top 4x4 NSFU 2 | 17 unit cases x {no stall, stall 25 % lat 3, stall 40 % lat 60} = 51 runs | 51 PASS | 0 | - |
| gm_top 4x4 NSFU 2, Icarus 13 (4-state) | chunk_k65, partial_small, mode_abt, ep_bias_drop_resid, ep_gelu at +lat=5 (MEMW 14, MASKW 12; compile 9 s, runs 1-21 s) | 1,574 | 0 | cycles = Verilator +lat=1 + 4 (645 / 509 / 9,801 / 5,105 / 6,375) |
| gm_top 8x8 NSFU 4 | 14 cases x {plain, stall 30 % lat 7} = 28 runs (final RTL, `build/test-gemm/unit8_final/`) | 31,664 | 0 | - |
| gm_top 16x16 | 27 unit cases (K = 1 .. 4160, partial M/N, batches, A^T / B^T, 3 segments, every epilogue) | 27 PASS | 0 | model 0.54-1.04 x (tiny GEMMs: fill/tail dominated) |
| real_qproj  Q = X Wq^T, LoWA-merged Wq, 4096 x 768 x 768 (final RTL) | e2e | 3,145,728 | 0 | 9,697,592 (9,737,216, 0.996), util 0.973 |
| real_scores S = Q K^T, 128 x 128 x 64, batch 384 | e2e | 6,291,456 | 0 | 1,814,584 (1,765,376, 1.028), util 0.867 |
| real_pv O = P' V (V TRN), 128 x 64 x 128, batch 384 | e2e | 3,145,728 | 0 | 1,618,232 (1,688,576, 0.958), util 0.972 |
| real_dv dV = P'^T dO (TRN x TRN), 128 x 64 x 128, batch 384 | e2e | 3,145,728 | 0 | 1,618,232 (1,688,576, 0.958), util 0.972 |
| real_attnout + bias + dropout + residual, 4096 x 768 x 768 (final RTL) | e2e | 3,145,728 | 0 | 9,698,537 (9,737,216, 0.996), util 0.973 |
| real_ffn1_256 + bias + GELU (u, g stores), 256 x 3072 x 768 | e2e | 1,572,864 | 0 | 2,434,140 (2,435,840, 0.999), util 0.969 |
| real_dgelu dg x GELU'(u), 256 x 3072 x 768 | e2e | 786,432 | 0 | 2,442,860 (2,444,341, 0.999), util 0.966 |
| real_ffn1 + bias + GELU, full 4096 x 3072 x 768 (final RTL) | e2e | 25,165,824 | 0 | 38,791,260 (38,951,221, 0.996), util 0.973 |
Cycles = testbench clock cycles from `start` to `done`; util = MACs / (cycles x 256).  Real-data
goldens: step-B batch through the L1 layer-0 forward (section 3).  All error flags 0 in every run.
`make test-gemm` (final RTL, from a clean `build/test-gemm`; all simulators rebuilt): 2,554 s wall,
`TEST-GEMM PASS` in `build/test-gemm/summary.txt` (46 rows, 0 mismatches; real_ffn1's `util` column
reads 0.108 because the testbench's 32-bit MAC counter wraps at 2^32 for that case, `active` 0.973 is
the valid figure).  `make lint` PASS (25 modules, 102 s), `make model-check` PASS (`GATE L1 vs L0: PASS`,
66 s), `make synth-gemm` PASS (769 s).

## 6. Synthesis, STA and the fanout rewrite
Yosys 0.62 `synth -flatten -booth`, abc -D 40000, GF180 7-track, OpenSTA; ss_125C_3v00 setup
slack at 40 ns raw and after OpenROAD `repair_design` (resize only: with no placement it inserts
no buffers, so RTL fanout is what counts).
| unit | parameters | area tt (um2, std cells) | cells | ss raw | ss repaired | tt raw |
|---|---|---|---|---|---|---|
| gm_pe | FMA_STAGES 2 (3 / 4) | 130,068 (134,487 / 141,154) | - | +15.27 (+16.55 / +19.58) | - | - |
| gm_array | 4x4 / 8x8 / 16x16 | 2.09 / 8.32 / 33.20 M | - | +14.78 / +14.78 / -6.37 | +14.89 (16x16) | - |
| gm_top | 16x16, NSFU 16, first P3 RTL | 42.85 M | 1,599,708 | -162.06 | -63.82 | -107.43 |
| gm_top | 16x16, NSFU 16, fanout rewrite (rev c: `build/synth/gm_top_16x16c`) | 42.78 M | 1,595,738 | -7.29 | **+0.86** | +4.49 |
| gm_top | 16x16, NSFU 16, rev d: rev c + 7 `(* keep *)` reset flops, merged by Yosys into one (`build/synth/p3_revs/gm_top_16x16d`) | 42.79 M | 1,595,875 | -9.53 | -0.50 | +4.29 |
| gm_top | **16x16, NSFU 16, final RTL**: cm_rstbuf reset tree + ctrl reset fix (`build/synth/gm_top_16x16`, `make synth-gemm`) | **42.80 M** | **1,597,394** | **-6.37** | **+4.23** | **+4.49** |
SRAM macros are blackboxes (0.0678 mm2 each, 128 ACC + 128 X + 32 W = 288 in gm_top, counted in
budget.md).  Hold after repair is -1.76 ns pre-layout (no hold fixing in this flow; P8).
`make synth-gemm` (all seven rows of `build/synth/gemm_table.md`, in parallel): 769 s wall, `SYNTH-GEMM PASS`.

**The reset path (rev c +0.86 / rev d -0.50 / final +4.23).**  The worst path of rev c and rev d
was not the reset net's own load but a write enable gated by it: reset flop Q (8.8 / 9.3 ns with
1,044 / 1,028 loads) -> nand2 (-> or2 in rev d) -> `nor3_2` = `~bf_wp[1] & ~bf_wp[0] & (rst & we)`
driving the 221 enable inputs of `u_ctrl.bf` entry 0 (27.7 ns) -> mux2 -> D.  Rev d lost 1.4 ns
to the slower launch flop and the extra gate; the structure was the same.  `repair_design`
without placement inserts no buffers (`Buffers` column 0 in every run; 22,094 fanout violations
remain, 1,276-1,337 instances resized), so the RTL structure of every high-fanout net is what
STA sees.  Fix: `hw/common/cm_rstbuf.v`, a one-flop module with `(* keep_hierarchy *)`;
gm_top instantiates 8 copies (one per sub-block + one for the GELU pool through the new
`gm_epilogue.rst_n_g` port).  Yosys `opt_merge` merges identical `$dff` cells of one module but
never across a kept hierarchy boundary; a 7-instance test and the engine netlist both keep every
copy (`cm_rstbuf` instances in `nl.v`: 8; loads per copy: array 1 (its own per-row flops),
xfeed 216, wfeed 85, loader 12, drain 17, epilogue 132, ctrl 31, gelu 558).  The launch flop of
the old path now drives 31 loads and the path is gone; the remaining worst path is the
256-load write enable of `u_ep.af` entry 0 (`nor3_2`, 32.0 ns of its 35.0 ns arrival), +4.23 ns.
Every copy adds one cycle of reset (all blocks release in the same cycle, as rev d), cycle
counts are identical to rev d in all 27 unit cases.  The flow's setattr keeps `t:cm_rstbuf` as
well (belt and braces; the module attribute alone suffices).
The -63.8 ns came from register files read and written through binary pointers: Yosys lowers
`file[ptr*W +: W]` into partial-decode nets, and single nets (ROB read pointer, assembler read
pointer, W rotator select, ROB write decode) ended up driving 1,000-6,400 gate inputs (7-74 ns
each at ss).  Fix, all in RTL: `cm_ohmux` (one-hot AND-OR mux, REP replicated `(* keep *)`
select copies so that one select bit drives <= ~64 AND gates) for the GELU ROB head, the
assembler beat select, the aux / mask / GELU FIFOs, the bias and aux lane selects, the W
rotator (buffer + lane folded into a registered one-hot per column) and the X capture; explicit
per-entry write enables from one-hot write pointers for the ROB, aux / GELU FIFOs and bias
registers; one registered reset copy per sub-block; one extra register on the GELU return path
(the only latency change: +2 .. +128 cycles on the GELU unit cases, +64 on real_ffn1_256).  In
the final netlist the largest nets are the gelu reset copy (558), `u_wfeed.rbuf_c[0]` (417),
the xfeed capture-select root (502) and per-entry write enables of <= 256 loads; all are what
PnR's placed `repair_design` buffers (the tutorial's 96-load `rst_n` case), none is a datapath
net.

## 7. Deviations and open points (for the PI)
1. ACC double-buffered by output block: 128 macros (+64 vs memory_map.md) so block n's epilogue
   overlaps block n+1; without it every block would expose its epilogue (~1,100 cycles).
2. FMA_STAGES = 2 (P2 production setting) with the LAMBDA = 4 feedback; W skew in registers
   (addressed skew impossible on the W side, section 2.1).
3. Dropout-backward on the A operand is not in the engine (ELEM pass, allowed by dataflow.md).
4. A short K chunk costs a full 256-cycle drain slot (only the last chunk of a GEMM).
5. Measured GELU / GELU' epilogue rates 0.43 / 0.32 el/cycle (16 SFUs): GELU' is co-limiting at
   K = 768 (1.004 x the array bound); a GEMM with gelu-bwd and K < ~600 would be epilogue-bound.
6. Loader = 1 beat/cycle = the array rate for 64-row blocks; M < 64 blocks (layer-11 CLS rows,
   head GEMMs) are loader-bound.  budget.py models it; the `full16-lean` config is not viable.
7. Pre-layout hold -1.76 ns after repair (no hold fixing in this flow; P8 inserts hold buffers
   after CTS as in the tutorial's mb_chip).  Setup closes at +4.23 ns after repair (section 6).
   The flow's `repair_design` is resize-only (no placement: 0 buffers inserted, 1,336 instances
   resized, 22,094 fanout violations left), so every high-fanout net is as the RTL structures it;
   the remaining worst path is a 256-load write enable (`u_ep.af` entry 0, nor3 32.0 ns), which
   only PnR's placed buffering shortens.
8. Reset bug found by the Icarus (4-state) smoke, fixed: in `gm_ctrl` the reset of `buf_inuse`,
   `rel_pend` and `have_a` sat inside the `else` branch of `if (!rst_n)` (dead code), so those
   flops were never reset.  Verilator's 2-state start hid it; under Icarus `buf_inuse` stayed X,
   `x_hold` / `w_hold` were X and the loader never accepted a beat (`rd0_dat_ready` X for the whole
   run = the "hang" that was attributed to the compile).  The three resets are now in the reset
   branch; cycle counts are unchanged in every Verilator case.
9. Icarus specifics, both in the testbench / loader only: (a) `iverilog` time grows super-linearly
   with the memory-model array sizes (2^24 keep bits + 2^16 words: > 2.5 h; 2 x 2^16: 311 s;
   2^14 words + 2^12 keep bits: 9 s), so the keep-bit array is now a testbench parameter `MASKW`
   (24 for Verilator, 12 for the Icarus cases, which use <= 2,627 keep bits and < 2^13 words);
   every hw module alone (4x4 gm_top included) compiles in <= 1 s.  (b) A continuous
   bit-select `w_busy[c_buf]` whose index was X at time 0 was never re-evaluated by ivl even after
   every input became known; `gm_loader.buf_free` is written as explicit 2:1 muxes (same logic).
