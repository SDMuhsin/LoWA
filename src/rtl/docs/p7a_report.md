# P7a report: chip top, end-to-end training steps on the full RTL

## 1. Chip top (hw/top)
- `top_chip.v`: ct_seq + ct_dma, gm_top (16x16), vu_top (V 16), lw_top, and mm_top (NR 8, NM 2, NW 4, 256-bit AXI,
  MK_DRAM = 0, so both dropout keep-bit channels come from the on-chip Threefry generators).
- `top_axil.v`: AXI4-Lite slave (32-bit, 12-bit address) in front of the P6c CSR bus.
  - One transaction at a time. A write needs AW and W together; a read gets R one cycle after AR.
  - Responses are always OKAY. WSTRB is ignored, so only full-word CSR writes are supported.
- **Chip boundary:** clk, rst_n, irq, the AXI4-Lite slave and the AXI4 master. Nothing else crosses it.
- **Channel map:**

  | channel | assignment |
  |---|---|
  | read 0-7 | gm rd0 (tiles), gm rd1, vu rd0, vu rd1, lw rd0, lw rd1, lw rd2 (merge), ct (fetch + DMA) |
  | write 0-3 | gm, vu, lw, ct |
  | keep 0 | gm epilogue; site / layer from GEMM payload word 4 [11:8] / word 5 [3:0] |
  | keep 1 | vu; site / layer from VEC payload word 0 [15:12] / word 1 [3:0] |

  Payload words are wired to the engine ports in sw/isa.py field order. ct `wr_idle` = {lw, vu, gm, ct}.
- **`make lint` PASS:** 51 files / 51 modules. Verilator -Wall 129 runs, Yosys 51 tops, Icarus 51 tops.

## 2. Test bench and method (tb/top)
- **tb_top.sv** connects top_chip to the tb_dram AXI4 slave with nominal knobs (lat 24, wlat 8, 32 B/cycle,
  `wfair=1` per ruling R9). A host model runs a script that does:
  - AXI-Lite CSR writes and reads;
  - host DRAM-window writes (ruling R3: the host load path into the same DRAM);
  - wait for IRQ (every simulation has a +maxcyc limit);
  - dumps of DRAM regions.
  It counts host bytes and per-unit busy cycles, and writes an op log.
- **tb_dram gained `load_bin`**: a `$fread` image load. `$readmemh` of the 12-layer image took longer than 7 minutes.
- **run_e2e.py** does the following:
  - compiles with sw/compile.py;
  - builds the DRAM model from the image regions plus every op's write range. Read-only ranges outside them stay
    unmapped, so stray reads answer SLVERR and are counted as errors.
  - runs the interpreter step by step (golden);
  - with `--l1`, runs two chained L1 steps at the same (B, L): step 2 takes step 1's post-state, step_s + 1 and
    the next B sequences;
  - checks interpreter vs L1, then RTL vs both;
  - **ALLWRITES**: compares every word of every byte range any op writes (stash, scratch, grads, state, loss)
    after every step;
  - compares the 24 result tensors.
- **Host traffic per step:**
  - Step 1's slot-0 inputs are part of the one-time image.
  - Before step 2 the host writes only ids, key-mask words and labels into slot 1, rings DOORBELL (slot 1),
    waits for the IRQ and writes STEP_DONE (acknowledge).
  - The per-row loss weights of slot 1 are constant and written once.
  - Step 2 uses the post-step coefficients, head, Adam m / v and OPT_SCAL beta^t that step 1 left in DRAM. STEP_T
    is advanced by END.
- **Debug aids** (not tests):
  - `--stop-after k` replaces the descriptor after unit op k with END, on both RTL and interpreter.
  - `tb/top/e2e_debug.py` attributes mismatching words to the last op whose write range covers them.

Build: Verilator 5.044 with `--threads 8`, which runs about 5.5x faster than one thread (≈16 k cycles/s against
≈2.9 k). MEMW 27 (512 MiB model).

## 3. Reduced program (1 layer, B 2, L 32), `make test-e2e`: PASS
- 2 steps × 87 descriptors, 10,356,787 cycles / step. **ALLWRITES** (3,389,288 words) and **all 24 result
  tensors are bit-exact vs the interpreter after both steps.** Losses are 0.748631597 / 0.58404243. Simulation
  time 1,528 s.
- The step is dominated by the head AdamW: 592 k parameters at 9 cycles each = 5.33 M cycles. This is
  shape-independent.

### Bug found and fixed at the root: gm_drain's epilogue block window
**Symptom.** In the first full-RTL run, the reduced step had 697 wrong words in row 0 of the attention output
projection (attn_out), with gm_top err[3] (ACC port conflict) set. It was localised as follows:
1. e2e_debug pointed to op 9 (Wo GEMM).
2. `--stop-after 9` and the GEMM error monitor pointed to op 8, P·V: M 32, N 64, **K 32**, 24 batched blocks.
3. tb_gm_top reproduced it on its own. Every GEMM with K < 64 (one short chunk) and more than one output block
   fails with 16 wrong words + err. K = 64 passes.

**Cause.** gm_drain keeps one activity window (`act_cnt` / `act_half`, TOT + 6 cycles: reads plus the 5-cycle RMW
write tail). The window blocks epilogue reads of that ACC half. With short-K passes, drains run back to back, so the
next block's drain is accepted on the *other* half exactly when the previous drain ends. The single window then
switched halves, and the previous half's write tail (5 cycles) was left unprotected. The epilogue, already reading
the earlier micro-tiles of that half, collided with it.

**Fix.** One window counter per half (`act_cnt0` / `act_cnt1`). The P3 drain schedule is unchanged; no rule was
loosened.

**Regression cases** in `tb/gemm/run_tests.py` UNIT_CASES (expected values from L1 `gemm_raw` via gen_gemm.py):
`custom_32_64_32_2_12_0_1` (the reduced-L P·V) and `custom_64_128_16_1_1_0_0`. Both PASS, as do K 16 / 48 and the
head shapes (1x2x2, 2x768x2, 768x768x2 with A transposed).

**P3 gates re-run on the fixed RTL:**
- `make synth-gemm` PASS: gm_top +5.22 ns ss after repair (raw -6.37), tt +4.49, hold +0.24, 42.53 mm2.
- `make test-gemm`: **PASS** (TEST-GEMM PASS, including the two new cases).

## 4. Full-depth step (12 layers, roberta-base, real weights / LoWA tables / RTE tokens), `make test-e2e-full`
Only batch × seq are reduced: **B 2, L 32**. d, layers, k and the atom shape are unchanged.
- **Estimate.** budget.py estimated 55.6 M cycles / step at this shape without CLS pruning (54.1 M with pruning),
  ≈ 1 h / step at the measured simulation speed.
- **Cycle limit.** 200 M for the 2-step run.
- **L1 at this shape.** step.py takes (B, L) from its inputs, so no L1 change was needed. The inputs are the first
  B rows / L tokens of the step-B batch; step 2 uses rows 2-3. The interpreter matches L1 here as well (CHECK
  interp vs L1).

RESULT: section 8.

## 5. Two steps and host bytes
| quantity | value |
|---|---|
| per-step host → device (measured, reduced) | 280 B = 272 B DRAM window (ids 4·64 + key mask 4·2·1 + labels 4·2) + 8 B AXI-Lite (DOORBELL, STEP_DONE ack) |
| device → host per step | 0 B (the loss can be read from LOSS_OUT, optional) |
| formula | 4·B·L + 4·B·ceil(L/32) + 4·B + 8 |
| full shape B 32, L 128 | 16,384 + 512 + 128 + 8 = **17,032 B** |
| spec (memory_map.md section 4) | 8,848 B |

Why 17,032 B differs from the 8,848 B spec:
- **ids are int32.** The vu EMB reads int32 (P6c deviation, memory_map.md section 5). That is +8,192 B; u16 ids
  would need a vu EMB change.
- **No 16 B step descriptor.** B / L are compiled into the program.
- **AXI-Lite traffic is counted:** +8 B.

## 6. Cycle accounting
`tb/top/acct.py` reads the op log and matches each op to the program walk per unit. The budget column is
budget.py's per-op model at the same shape (GEMM: max(array/loader, bytes / (32 B · EFF_GEMM)); vector: VEC_RATE).

Reduced run, per step (10,356,765 cycles running):

| class | measured (start→done) | budget | ratio |
|---|---|---|---|
| ADAMW (head 592 k + coef 512) | 5,334,168 | 5,333,778 | 1.00 |
| GEMM (W-stream + small) | 3,771,480 | 3,686,567 | 1.02 |
| LOWA_SG (2 modules) | 359,795 | 355,584 | 1.01 |
| VEC total | 851,148 | 911,182 | 0.93 |
| DMA_COPY (3) | 104,708 | 12,288 | 8.5 |

The engines overlap (lw beside the GEMMs), so the busy counters do not sum to the step:

| counter | cycles |
|---|---|
| gm busy | 3.77 M |
| array active | 3.23 M |
| vu busy | 0.85 M |
| lw busy | 5.69 M |
| merge | 0.31 M |
| sequencer with all units idle | 2,117 |
| fetch / decode | 110 k |
| DRAM R beats | 3.93 M |
| DRAM W beats | 0.51 M |
| DRAM idle | 5.99 M (58 %, mostly during AdamW) |

Gaps over 10 %:
- **DMA_COPY.** Not modelled in budget.py. ct_dma copies 64-word chunks, one read round trip each, so it is
  latency-bound at ≈ 1 word/cycle. It overlaps the GEMMs, so it is not on the critical path.
- **VEC E_TANHB / GN_ROWS / E_DROP are faster than budgeted.** budget.py charges every element op at the
  1024 x 768 ELEM rate. With 2 rows (head ops) the lanes are mostly idle, but the per-column cost scales by
  ceil(rows/16) in both, so the budget over-charges these short ops.
- **LN / SMAX / EMB are 1.2-1.5x the budget.** At B·L = 64 rows, a 16-row tile has a fixed per-tile
  scalar-phase cost and the DRAM row latency (24 cycles) is exposed. VEC_RATE was measured at 4,096 rows.
- **CE.** 515 vs 120 cycles; this is fixed latency.

Full-depth accounting: section 8.

## 7. Gates
| gate | result |
|---|---|
| `make lint` | PASS |
| `make model-check` | PASS (GATE L1 vs L0) |
| `make synth-gemm` | PASS (+5.22 ns ss) |
| `make test-gemm` | PASS |
| `make test-e2e` | PASS |

`make test-e2e-full`: **PASS** (2 steps, section 8).

## 8. Results of the long runs
### 8.1 Full-depth step 1 (12 layers, B 2, L 32): **PASS, bit-exact vs L1**
`make test-e2e-full` = `run_e2e.py --layers 12 --batch 2 --seq 32 --steps 2 --l1 --threads 8 --maxcyc 200000000`.
- 141 descriptors (618 executed per step).
- The DRAM model holds 94 regions, 359 MiB. The interpreter matches L1 at this shape (24 / 24 tensors, both steps).
- **Step 1 on RTL: 61,619,595 cycles.**
  - **ALLWRITES PASS**: 8,346,904 words (every byte range the program writes: 12 layers of stash, scratch,
    grads, post-step state).
  - **24 / 24 result tensors bit-exact vs the interpreter and vs the L1 step**: loss 0.785167277 (RTL = L1),
    logits, all 24 x 256 coefficient gradients, head gradients, G / clip, post-step coefficients + head, Adam m / v.
  - No AXI or engine errors.
- **Simulation time.** Step 1 took ≈ 3.3 h wall at 8 threads. The host was shared (load average ≈ 60 on 48 cores
  from unrelated GanAccelAsic jobs plus `make test-gemm`), so the speed was 5-16 k cycles/s against 16 k
  unloaded.

### 8.2 Step 2 of the 12-layer run: **PASS, bit-exact vs two chained L1 steps**
`E2E l12_b2_s32_n2 PASS`.
- **Host traffic before step 2:** the host wrote 272 B through the DRAM window (slot-1 ids, mask, labels), then
  rang doorbell slot 1 and acknowledged the IRQ (8 B AXI-Lite). In total 280 B in, 0 B out.
- **Step 2: 61,620,280 cycles** (IRQ at cycle 123,239,875), using step 1's on-chip-updated PARAM / ADAM_M / ADAM_V /
  OPT_SCAL and STEP_T.
- **ALLWRITES PASS** (8,346,904 words). **24 / 24 tensors bit-exact vs the interpreter and vs L1 step 2**:
  loss 0.230305105 = L1.
- Both steps took 17,921 s wall (5.0 h) on the loaded host.

### 8.3 Cycle accounting, full depth (step 1)
budget.py at B 2, L 32, without CLS pruning (as compiled):

| unit | RTL busy | budget.py | gap |
|---|---|---|---|
| step | **61.62 M** | 55.56 M overlapped | +10.9 % |
| GEMM (gm busy) | 47.04 M; array active 42.84 M | gemm 45.59 M | +3.2 % |
| vector (vu busy) | 6.80 M | vec 4.57 M | +49 % |
| LoWA unit (lw busy: SG + AdamW + SCAL) | 9.72 M | lowa 4.09 M + opt 5.39 M = 9.49 M | +2.4 % |
| merge (lw merge queue) | 8.95 M | — (hidden behind the GEMM by design) | — |
| DMA_COPY | 0.10 M | — | — |
| sequencer, no unit busy | 15.0 k | — | — |
| fetch / decode | 0.14 M | — | — |
| DRAM R beats | 46.0 M | — | — |
| DRAM W beats | 2.85 M | — | — |
| DRAM idle | 13.5 M (22 %) | — | — |

Per op class (op log partial at the time of writing; `tb/top/acct.py build/e2e/l12_b2_s32_n2` now gives the full table):
- GEMM W-stream 1.03 x budget;
- attention / head GEMMs 0.97 x;
- LOWA_SG 1.02 x;
- LN_FWD / LN_BWD 1.33 x;
- SMAX 1.2-1.5 x;
- EMB 1.38 x.

Totals from the complete op log, measured vs the per-op budget model:

| unit | measured | budget | ratio |
|---|---|---|---|
| GEMM | 47.04 M | 45.60 M | 1.03 |
| VEC | 6.80 M | 5.57 M | 1.22 |
| LOWA_SG | 4.33 M | 4.27 M | 1.02 |
| ADAMW | 5.38 M | 5.38 M | 1.00 |
| DMA_COPY | 0.10 M | — (not modelled) | — |

Explanation of the gaps over 10 %:
1. **Vector unit, +2.2 M.** VEC_RATE was measured on 4,096-row tensors. At N = 64 rows the per-tile scalar
   phases and the DRAM row latency are not amortised (section 6). This is a small-shape effect. It also shows
   that the vector ops are serial with the GEMMs in this program: the vector unit waits on FENCEs, as budget.py
   assumes.
2. **Step +6.1 M over the overlapped model.** It breaks down as:
   - GEMM +1.4 M;
   - vector +2.2 M;
   - the LoWA sampled gradient not fully hidden: lw busy is 9.72 M, of which 5.33 M is the serial AdamW at the
     end. The layer-0 LOWA_SG runs after the last backward GEMM and before GN_ROWS, so ≈ 0.4 M is exposed;
   - the rest is GEMM hand-offs and fence drains.
   budget.py's own serial model (58.2 M pruned) brackets the measured step from below.
3. **budget.py charges CLS pruning (54.1 M).** The compiler does not prune (P6c deviation 4).

## 9. Open items
- `make test-e2e` takes ≈ 25 min plus a 7 min build. The head AdamW (5.3 M cycles) dominates the reduced step.
- ids are int32 (17,032 B / step at full shape vs the 8,848 B spec).
- No CLS pruning.
- WSTRB is ignored on AXI-Lite.
