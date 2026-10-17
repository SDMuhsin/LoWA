# P6c report: mm_top timing, sequencer / ISA decoder (hw/ctrl), compiler + program interpreter (sw/)

## 1. mm_top timing (left open by P6b), closed
The real critical path was the **flush** request, not the write-beat accept. It ran from the wrapper flop `q_ra`
(read request address) through `hi = lo + len - 1` and the pending-write compares (RAW `r_blk`), then `rd_wait`,
then mm_wrport `flush`, then `fl_go`, and ended at the write-queue `q_line` enables. That is about 17 ns of
arithmetic and then three high-fanout nets, 43 ns in total. The fix registers `rd_wait` (`rd_wait_q`, one flop)
before it reaches the write ports. `flush` only closes open write bursts early, so that an access blocked by
them makes progress; the blocked access keeps it asserted. The RAW / WAR / WAW acceptance rules are unchanged.
**`make synth-mem` PASS: +0.73 ns at ss after repair** (raw -1.80, tt +8.45, hold +0.24, 10.23 mm2,
325,503 cells). **`make test-mem` PASS**: all 47 rows bit-exact. The cycle counts are identical to P6b
(ep_bias_drop_resid 6,342; real_ffn1_256 2,576,250; real_ln_fwd 2,963,526).

## 2. ISA as built (docs/isa.md section 7, sw/isa.py)
The P0 text needed three changes:
- **Variable descriptor length.** Descriptors are 1-3 slots of 64 B, because a GEMM plus its merge needs 36
  payload words.
- **Layer-relative address terms.** Each descriptor carries REL nibbles. At decode, every word gets
  `+ TERM[sel]`, where the terms are: layer terms ±STR[k] per LOOP iteration, the loop index, STEP_T, and the
  slot delta. One loop body therefore addresses layer l's weights, stash, LoWA tables and RNG layer field.
- **Exact engine payloads.** The payload words are the engine descriptor fields of gm_top, lw_top merge,
  vu_top and lw_top main.

Other definitions:
- **Dependence rule.** Issue is in program order. WAIT / CLR are checked at issue. A unit runs one op at a time.
  An op retires on engine done (plus merge done), then 2 cycles, then the unit's memory write channel idle
  (P6a `wr_idle`); SIG is set at retire. FENCE waits for units to be idle; END waits for all of them.
- **Host CSRs:** run-enable, soft reset, RESET_T, IRQ_EN, DOORBELL (slot), STEP_DONE, STEP_T, PROG_BASE,
  SLOT_DELTA, SEED, ERR, PERF.

## 3. Sequencer (hw/ctrl/ct_seq.v + ct_dma.v)
- **Datapath.** A 48-word decode buffer is filled by one 48-word fetch per descriptor. The REL adds run serially,
  one word per cycle (one adder and a 16:1 term mux). Each unit has its own descriptor register
  (GEMM 36 / VEC 25 / LOWA 10 / DMA 9 words), held until retire. There are 15 event flags, a non-nested LOOP
  with 12 stride registers, and the CSR file. The dropout generator configuration comes out as `mk_seed`
  (SEED) and `mk_step` (STEP_T); site and layer are payload fields. The fetch path is about 80 cycles per
  descriptor, against op lengths of 10^3-10^6 cycles.
- **DMA_COPY (ISA 0x30, open since P6a).** It copies n2 x n1 rows of n0 words with byte strides, in chunks of
  up to 64 words. It shares the fetch read channel and has its own write channel.
- **`make synth-ctrl` PASS:** 0.861 mm2 tt (7,039 flops incl. the wrapper's input flops), **+1.61 ns ss after
  repair** (raw -22.69: reset and enable fanout before repair; one cm_rstbuf is added for each of ct_seq and
  ct_dma), hold +0.42.
- **`make test-ctrl` PASS (6/6).** tb_ct_seq uses stub engines (random durations, random write-drain delays, a
  merge stub) and a memory with random latency, gaps and backpressure. A python checker compares the RTL log
  against the reference walk of the program (sw/isa.py decode plus the LOOP rule), never against RTL output.
  It checks:
  - every start's payload words, in program order;
  - issue order;
  - one op per unit;
  - retire only after done, write idle and merge done;
  - FENCE ordering;
  - WAIT/SIG ordering;
  - the CSRs after END: STEP_T + 1, STEP_DONE, IRQ and its acknowledge, mk_step.

| case | content | unit ops | checks | cycles |
|---|---|---|---|---|
| real_reduced | the compiled reduced program (1 layer, B 2, L 32) | 56 | 6,760 | 50,227 |
| real_full | **the compiled full-step program** (LOOP 12 up / 11 down) | 408 | 2,874,830 | 2,968,280 |
| flags, flags_s0 | WAIT/SIG/CLR chains GEMM->VEC->LOWA, MERGE, FENCE, LOOP x 3 down with layer / slot / STEP_T terms, DMA->VEC flag | 19 each | 162 each | ~1,140 |
| dma | DMA_COPY 1/2/3-level copies, 70-word rows (chunk crossing), unaligned source, stride-0 fill; memory checked word by word | 4 | 3,098 | 513 |
| badop | illegal opcode -> ERR.decode = 1, halt, no IRQ | 1 | 8 | 112 |

Mutation check: a mutant that retires without waiting for write idle and has a wrong loop term is reported as
FAIL ("retired before its writes drained").

## 4. Compiler (sw/compile.py) and its golden test (`make compile`, `make compiler-check`)
**Inputs.** Model config (layers, B, L, task CE/MSE, C) → program + DRAM image, using:
- the real roberta-base checkpoint (model/runs/model_fp32.npz, dumped by l0_ref.py's loader);
- LoWA tables from sw/lowa_tables.py;
- the R6 HPs;
- pre-step coefficients, head and Adam state, and the beta^s running products in OPT_SCAL;
- STEP_IN slot data.

**Program for one full step: 141 descriptors, 16,640 bytes** (413 unit / fence ops unrolled):

| op | count |
|---|---|
| GEMM | 31 |
| GEMM_ACC2 | 1 |
| VEC | 25 |
| LOSS | 1 |
| GNORM | 29 |
| LOWA_SG | 4 |
| LOWA_SCAL | 1 |
| ADAMW | 2 |
| DMA_COPY | 3 |
| FENCE | 37 |
| LOOP | 2 |
| SETCSR | 4 |
| END | 1 |

The schedule:
- EMB, then DMA_COPY X0 → STASH_X0, then LOWA_SCAL;
- LOOP 12 up {11 forward ops};
- head forward, CE, head backward (dout cleared by a stride-0 DMA_COPY from CONST zeros);
- LOOP 11 down {19 backward ops} and a separate layer-0 body;
- 28 GN_ROWS + GN_FIN, a DMA_COPY of the clip coefficient into both AdamW scalar blocks, ADAMW coef, ADAMW head
  (clip applied in lw_adamw), END.

The reduced program (1 layer, B 2, L 32) has 87 descriptors and 10,112 bytes.

**Dropout sites (P6b deviation 3).** Every site has a consumer:
- forward: EMB (vu), attention probabilities (SMAX_FWD), attn_out and ffn_out (GEMM epilogue), cls1 / cls2
  (vu E_DROP);
- backward: ffn_out and attn_out on dz2 / dz1 (vu E_DROP before the dg / dO GEMMs, because the GEMM has no
  input-side dropout);
- attention: SMAX_BWD for dP, plus E_DROP of the recomputed P to rebuild P' for dV;
- cls1 / cls2: vu E_DROP.

All masks are regenerated from (seed, STEP_T, site, layer, element index).

**Ordering.** FENCEs are placed automatically from the read/write byte ranges of every op. It is a conservative
union over loop iterations. In the backward pass, dV, dQ and the recomputed X_l go into layer l's dead V / Q / K
stash slots, so LOWA_SG(q_l, v_l) overlaps dX(l), LN_BWD and the GEMMs of layer l-1.

**Interpreter (sw/interp.py).** It decodes exactly as the sequencer does, using the same TERM / LOOP rules. It
executes each engine descriptor with the L1 entry points, following the engine contracts encoded by the P3-P5
generators:
- gemm_raw over the descriptor strides / segments, with the merge rebuilt from the image's hit + dir tables
  through H.lowa_merge, then the epilogue chain;
- ln / smax / ce / gnorm / drop / tanh;
- lowa_sg from the centre table;
- step_scalars and adamw.

**`make compiler-check`: COMPILER-CHECK PASS.** The full step-B program (B 32, L 128, 12 layers) through the
interpreter (618 descriptors executed, 18 s) is **bit-exact against model/hwmodel/step.py** run fresh on the
same inputs, for all 24 result tensors:
- loss 0.677555978 and the logits;
- all 24 x 256 coefficient gradients and the 4 head gradients;
- G and the clip coefficient;
- post-step coefficients and head;
- post-step Adam m and v.

## 5. Deviations / not done
1. **NOT DONE: the reduced program on sequencer + gm_top + vu_top + lw_top + mm_top + DRAM model (deliverable
   3, second half).** Paused per the PI plan; see docs/p6c_handoff.md. The program, image, golden
   (interpreter) and unit-level decode checks of that exact program are ready.
2. ISA changes as in section 2 (slot length, TAG 9 bits, REL terms, ADAMW on LOWA, no REDUCE_ROWS / dim-sel /
   region-base CSRs / HWIN / single-op debug). Queue depth 1 per unit (P0: 2).
3. memory_map.md section 5:
   - STEP_IN slot 1 moved to 0x7C9C_A000, because int32 ids do not fit two slots;
   - new LOWA_DIR region;
   - SCR_ATT unused (attention is processed for the whole batch per descriptor).
4. No CLS pruning (D6): layer 11 runs all rows. Results are identical; this costs speed only.
5. Bias gradients use the ones-GEMM. That is exact only for B ≤ 64 (the compiler asserts it).
6. The program is compiled per (B, L, C). The last partial batch needs its own program, or the P0 dim scaling.
7. The interpreter runs ops in program order; a fence that a concurrent schedule needs but the program lacks
   would only show up on RTL. place_fences is conservative. The RTL system run (item 1) is the test that would
   catch it.
8. Gates: section 6.

## 6. Gates
`make synth-mem` PASS (+0.73), `make test-mem` PASS, `make synth-ctrl` PASS (+1.61), `make test-ctrl` PASS (6/6),
`make compile`, `make compiler-check` PASS, `make lint` PASS (49 modules, 127 Verilator runs), `make model-check` PASS (GATE L1 vs L0).
