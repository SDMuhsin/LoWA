# LoWA-TA memory map (P0)

All byte counts: `python3 src/rtl/docs/calc/memmap.py` (layout) and `calc/budget.py` (traffic), for
roberta-base, B = 32, L_PAD = 128 (the worst case; every per-step number scales down with L_PAD).

## 1. DRAM layout (4 KiB aligned, one image, 1.947 GiB -> a 2 GiB device is enough, 4 GiB addressable)

| region | base | bytes | written by | content |
|---|---|---|---|---|
| PROG | 0x0000_0000 | 262,144 | host once | descriptors (isa.md) |
| CONST | 0x0004_0000 | 4,096 | host once | gamma, h[0..24] (diamond order), lr0 per group, wd, betas, eps, fp32 1/w and 1/(T-w), dropout threshold 1677722 and scale 0x3F8E38E4, LN eps |
| EMB_WORD | 0x0004_1000 | 154,414,080 | host once | [50265 x 768] |
| EMB_POS | 0x0938_4000 | 1,579,008 | host once | [514 x 768] |
| EMB_TT_LN | 0x0950_6000 | 9,216 | host once | token_type row 0, LN_emb gamma/beta |
| LAYERS | 0x0950_9000 | 340,217,856 | host once | 12 x (Wq,bq,Wk,bk,Wv,bv,Wo,bo,LN1,W1,b1,W2,b2,LN2); W row-major [out x in] as in torch; `LAYER_STRIDE` = 28,351,488 |
| LOWA_CTR | 0x1D97_E000 | 24,576 | host once | 24 x 256 centres (ci << 16 \| cj) in `sample_centers` order |
| LOWA_HITS | 0x1D98_4000 | 614,400 | host once | 24 x 6,400 merge hits, 32 bit each (dataflow.md section 3) |
| PARAM | 0x1DA1_A000 | 2,396,172 | host once, device RW | 6,144 coefs (zero-init) + head (C <= 3) |
| GRAD | 0x1DC6_4000 | 2,396,172 | device | same layout |
| ADAM_M, ADAM_V | 0x1DEA_E000, 0x1E0F_8000 | 2 x 2,396,172 | device | same layout, zero-init |
| OPT_SCAL | 0x1E34_2000 | 4,096 | device | t, beta1^t, beta2^t, lr_t, clip coef, loss |
| STEP_IN | 0x1E34_3000 | 24,576 | host per step | 2 slots x 8,848 B |
| STASH_X0 | 0x1E34_9000 | 12,582,912 | device | layer-0 input after embedding dropout |
| STASH | 0x1EF4_9000 | 1,364,459,520 | device | 12 x 113.7 MB (below) |
| SCR_ACT | 0x7048_9000 | 201,326,592 | device | 4 x [N x 3072] ping-pong activations / gradients |
| SCR_ATT | 0x7C48_9000 | 4,718,592 | device | S/P', dP', dS for one sequence (12 heads) x 2 |
| SCR_CLS | 0x7C90_9000 | 786,432 | device | head activations / grads |
| LOSS_OUT | 0x7C9C_9000 | 64 | device | loss, status |

Backbone = 124,055,040 params = 496.2 MB fp32 (PLAN's "~500 MB"). Trainable = 6,144 + 592,899 (C = 3).

**Stash per layer** (N = 4096): z1, z2, Q, K, V (5 x 12.58 MB) + u (50.33 MB) + LN stats 4N +
softmax stats 2*B*12*L (0.46 MB) = 113.7 MB. Not stashed (recomputed or regenerated): P/P' (from Q,
K, m, r), LN outputs (from z, mu, rstd), dropout masks (PRNG), layer inputs X_l, l >= 1 (LN2 of z2).
With CLS pruning (D6) layer 11 stores K, V for all tokens and the rest for 32 rows only: total stash
1.290 GB.

## 2. Per-step DRAM traffic (full16 config, L_PAD = 128)

| stream | GB/step | note |
|---|---|---|
| GEMM operand reloads | 87.85 | 0.125 B/MAC x 674 GMAC + outputs; set by on-chip block size Br = Bc = 64 |
| stash write + read | 2.58 | |
| vector passes | 3.51 | LN, softmax, LN-bwd, recompute |
| LoWA sampled grad | 0.58 | dY row + X row per token per module |
| merge hit lists | 0.079 | re-read with every W-block reload of Wq, Wv |
| embedding gather, optimizer | 0.035 | |
| **total** | **94.6** | 784 MB/s average over a 120.7 s step; 800 MB/s peak in GEMM phases |

The DRAM traffic is dominated by operand reloads because a GF180 SRAM macro holds 256 B in 0.068 mm2:
on-chip reuse is area-bound, not bandwidth-bound (budget.md section 3 shows the trade).

## 3. On-chip RAM banking (full16: R = C = 16, lambda = 4; as built in hw/, reconciled in P8c)

Every on-chip RAM is an `mm_sram_bank` (32-bit words, one port, read latency 1) with a technology backend
(rtl_conventions.md): generic (simulation), gf180 (`gf180mcu_ocd_ip_sram__sram256x8m8wm1`, 256 x 8 bit,
301.3 x 224.9 um, 4 macros per 32-bit x 256-word lane) or tsmc65 (latch register file, docs/p8b_report.md).

| unit / bank | banks x words | organisation | words | KiB | ports per cycle |
|---|---|---|---|---|---|
| WBUF x 2 (gm_wfeed) | 32 x 256 | 2 buffers x 16 lanes; lane (c + k + 4q) mod 16, address {k, q}; Kc = 64 | 8,192 | 32 | all lanes of one buffer read {k, q}; DMA writes 8 lanes / merge RMW on the other |
| XBUF x 2 (gm_xfeed) | 16 x 512 | 2 buffers x 8 lanes; lane (k + r) mod 8; Br = 64 rows x Kc = 64 | 8,192 | 32 | R / lambda = 4 row reads; DMA writes 8 lanes on the other buffer |
| ACC x 2 halves (gm_drain) | 16 x 512 | 2 halves x 2 parities x 4 lanes; block n's epilogue reads one half while block n+1 accumulates in the other | 8,192 | 32 | 4-word RMW stream + 4-word epilogue read |
| vector tile buffers (vu_lane) | 32 x 256 | 16 lanes x 2 row-segment buffers (A / B or double buffer) | 8,192 | 32 | 1 issue read + 1 load / store write per lane |
| column vectors (vu_top) | 6 x 256 | cvec0..2 x 2 parities (double buffered) | 1,536 | 6 | 1 read (issue) + 1 load write |
| LoWA rows (lw_sg) | 2 x 768 | dY row, X row | 1,536 | 6 | 1 word / cycle each |
| LoWA atom accumulator + atom centres (lw_sg) | 2 x 256 | coefficient-gradient accumulators; atom centres (i, j) | 512 | 2 | 1 RMW per atom / 1 read |
| LoWA merge directory + coefficients (lw_merge) | 2 x 512 | block directory (2 module slots x 256), coefficients (2 x 256) | 1,024 | 4 | 1 read each |
| **total** | **106 banks** | | **37,376** | **146** | |

The DMA return / write FIFOs (hw/mem) and the vector writer's output registers are flops.  In GF180 macros the
146 KiB are 584 macros (39.6 mm2); on TSMC 65 nm the latch RF is 8.89 mm2 of cells at ss (p8b_report.md section 3).
Until P8c this section listed the P0 plan, 356 macros = 89 KiB = 22,784 words: a single 4,096-word ACC (the
RTL double-buffers it, P3), no vector-unit RAM (P4 built per-lane tile buffers and column-vector banks
instead of the flop transpose stagers), LoWA 1,792 words (P5 added the atom-centre table and the merge directory / coefficient buffers), and
512 words of DMA FIFOs counted as macros.  `docs/calc/budget.py onchip_words` now counts the RTL geometry.

Hit lists are streamed; the 25 taps h_e are held in flops; the atom centres are in the lw_sg bank above.

## 4. Host <-> device bytes

| when | bytes | content |
|---|---|---|
| once, before training | 496.2 MB weights + 0.64 MB LoWA tables + 2.4 MB head init + 0.26 MB program | through the host DRAM window or the DRAM's own host port (open question) |
| per step, in | **8,848 B** at B = 32, L = 128 | ids u16 (8,192), attention mask bits (512), labels (128: int32 class or fp32 STS-B score), step descriptor (16: L_PAD, B_ACT, slot, flags); the per-row loss weight is bit-packed in the descriptor |
| per step, out | 0 required, 4 B optional | loss |
| end of training | 2.4 MB | 6,144 coefs + head |

## 5. P6c additions (sw/compile.py places every operand; binding)
| region | base | bytes | content |
|---|---|---|---|
| STEP_IN slot 0 | 0x1E34_3000 (STEP_IN) | 24,576 | ids int32 [N] (vu EMB reads int32, so P0's 2 x 8,848 B slots do not fit), key-mask bits [B x ceil(L/32)], labels int32 [B], loss weights fp32 [B] |
| STEP_IN slot 1 | 0x7C9C_A000 | 24,576 | same; the program addresses slot 0 + TERM[15] (CSR SLOT_DELTA = 0x5E68_7000) |
| LOWA_DIR | 0x7C9D_0000 | 24,576 | 24 x 256 block-directory words (sw/lowa_tables.py `dir`), used by the merge engine |
CONST words: 0 gamma, 1..25 h taps, 32..41 LOWA_SCAL constants (1/w, 1/(T-w), lr0 x 2, wd, b1, b2, 1-b1, 1-b2, eps),
64..127 ones (bias-gradient GEMM), 128..895 zeros (dout clear by DMA_COPY). OPT_SCAL words: 0..23 LOWA_SCAL block
(group scalars at 8 and 16, clip copied to 15 and 23 by DMA_COPY), 32 / 33 GN_FIN [G, c], 64 + 8 i per-tensor norms.
SCR_ACT holds whole-batch attention buffers (3 x B·12·L² words) and the forward / backward activation buffers
(compile.py `Cfg.scr`); SCR_ATT is unused (the program processes all sequences in one descriptor). SCR_CLS: head
activations / gradients (8 x [B x 768]). In backward, dV / dQ / X_l (recomputed) are written into the dead V / Q / K
stash slots of layer l, so the LoWA sampled gradient of layer l reads per-layer addresses and can overlap layer l-1.
Embedding rows: the image carries only the rows the batch uses (the full table is identical in layout).
