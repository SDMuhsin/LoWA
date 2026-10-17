# LoWA-TA architecture spec (P0)

Scope: one complete LoWA training step of `roberta-base` (PLAN fixed decision 1), i.e. exactly what
`src/train_glue.py --optimizer adamw-sgwa` + `src/sgwa_adapter.py` do per optimizer step, on chip.
Reference run flags (`scratchpad/s3/jobs_*.tsv`, `scripts/a1_run.sh`): `--dtype float32
--per_device_train_batch_size 32 --max_length 128 --adapter_target_modules query,value --sgwa_k 256
--sgwa_K 3 --sgwa_seed 777 --sgwa_scaling 0.552427 --learning_rate 0.166667 --classifier_lr 5e-3
--weight_decay 0.01 --num_train_epochs 30 --num_warmup_steps <task>`.
Numbers in this doc come from `calc/budget.py` and `calc/memmap.py`.

## 0. What the reference actually does (read from the code, transformers 4.45.2, torch 2.5.1)

| item | reference behaviour | source |
|---|---|---|
| model | `AutoModelForSequenceClassification(roberta-base)`, SDPA attention, `model.train()` | train_glue.py:1199, 2464 |
| LoWA | `SGWALinear` on every `query`/`value` (24 modules), W = W0 + gamma * sum_j c_j psi_j, merged by one `index_add` | sgwa_adapter.py:221-248 |
| gamma | **0.552427** in S.3/A.1 runs (`--sgwa_scaling`); the derived default 150/sqrt(2mn) = 0.138107 is the "sgwaA" config. Program constant, not hardwired. | s3_plan.py:21 |
| trainable | 24 x 256 coefs + `classifier.dense` (768x768 + 768) + `classifier.out_proj` (C x 768 + C); everything else `requires_grad=False` | sgwa_adapter.py:286-299 |
| optimizer | `torch.optim.AdamW` (the `-sgwa` suffix only selects the adapter), betas (0.9, 0.999), eps 1e-8, default foreach kernel; two groups: coefs lr 0.166667, head lr 5e-3, **weight_decay 0.01 on both, biases included** | train_glue.py:766-776, 2201-2294 |
| schedule | HF `get_scheduler("linear")`: f(s) = s/max(1,w) for s < w, else max(0, (T-s)/max(1,T-w)); s = completed steps, so the **first update runs at lr = 0** | train_glue.py:2356 |
| clipping | `clip_grad_norm_(model.parameters(), 1.0)` every step (default `--grad_clipping 1.0`), global L2 over the 28 trainable tensors | train_glue.py:480, 2491 |
| batches | `DataCollatorWithPadding`: **dynamic padding** to the batch max length (<= 128), shuffled, last batch short (no drop_last) | train_glue.py:2148-2179 |
| dropout | hidden 0.1 (embeddings, attention-output, FFN-output, classifier x2), attention-probs 0.1 (inside SDPA) | roberta-base config |
| loss | CE (mean) for classification; **MSE (mean) for STS-B** (`num_labels=1` -> `problem_type="regression"`) | modeling_roberta.py:1357-1372 |
| sampled grad | reference runs used `--sgwa_sampled_grad 0` (dense autograd). The chip uses the sampled gradient (fixed decision 4); `verify_sgwa` G10 shows equality to fp32 rounding. | sgwa_adapter.py:47-56 |

Measured padded lengths (roberta tokenizer, 5 random epochs, mean of per-batch max length):
CoLA 22.8, SST-2 40.8, STS-B 71.2, MRPC 80.2, MNLI 89.3, RTE 127.9 (RTE: 99 % of batches hit 128).

## 1. Decisions taken in P0

| # | decision | why |
|---|---|---|
| D1 | **Dropout ON, generated on chip** by a counter-based PRNG (Threefry-2x32-13, ARX only, no multipliers); masks are a pure function of (seed, step, site, layer, element) and are **regenerated in backward**, never stored. CSR `DROP_EN=0` gives a dropout-free debug mode. | Training without dropout is a different procedure from `train_glue` (all S.3 numbers used p=0.1). A counter PRNG is bit-reproducible by the L1 hwmodel and needs no mask stash (154 M mask bits/step at L=128). |
| D1a | Consequence: the chip can never reproduce a `train_glue` run bit for bit (PyTorch uses Philox inside CUDA kernels and inside SDPA). L0 for the L1-vs-L0 check is the same HF model with eager attention and **mask injection** (dropout modules replaced by modules that apply the chip's masks). Agreement with the published accuracy can only be distributional (5 seeds, same protocol). | |
| D2 | **STS-B / MSE in scope.** CE for C in {2, 3}, MSE for C = 1; selected by CSR `LOSS_MODE`. | S.3 and A.1 both run STS-B; MSE costs one SFU-free op. |
| D3 | **Fixed tile length, dynamic length per step.** One program; dimensions in descriptors scale with `L_PAD` (multiple of 16, <= 128) from the step descriptor. A short last batch is padded by the host with dummy rows (`<s>` + pads, mask = [1,0,...], loss weight 0); the loss divides by `B_ACT`. | Pad tokens and dummy rows contribute exact zeros to every token reduction (proof: section 5), so padding is numerically neutral under a fixed reduction order; dynamic L recovers 1.4-4.1x per step on real GLUE batches. |
| D4 | **Output-stationary GEMM, sequential FMA reduction** over k ascending, accumulator init +0.0, bias added once at the end. | Gives a reduction order that is independent of array size (the P1 contract). |
| D5 | **Atom-major sampled gradient**: acc_j accumulates sum_t sum_e h_e (dY[t,i_je] x X[t,l_je]) directly per atom (256 accumulators/module), instead of first forming S over the union U (<= 6400 cells) as `_SGWASampledLinear` does. Same k*s products, different summation order (flagged). | 1 KiB of accumulators instead of 25.6 KiB per module; no union table. |
| D6 | Last-layer **CLS pruning**: in layer 11 only the 32 CLS rows of Q, attention output, O-proj, FFN, and their backward are computed (K, V for all tokens). Exact: no other row of layer-11 output reaches the head. | -6.7 % MACs (722 -> 674 GMAC at L=128). |
| D7 | Activation stash in DRAM, recompute only what is cheap (softmax P from stashed Q, K, row max, 1/sum; LN outputs from stashed pre-LN sums and stats). | 1.29 GB stash at L=128; DRAM capacity is not the limit, SRAM is. |
| D8 | Clock target **25 MHz (40 ns) at ss_125C_3v00**, same as the prior GanAccelAsic chip that closed there; FMA pipelined to lambda = 4 stages. | Proven sign-off point on this flow; SRAM macro min period is 17.1 ns (ss lib). |

## 2. Block diagram

```
                host (PCIe/FPGA bridge)                       device DRAM (>= 2 GiB, AXI4 slave)
                   |  AXI4-Lite (CSRs + DRAM window)              ^  AXI4 256-bit @ core clk
                   v                                              |
 +----------------------------------------------------------------+--------------------------+
 |  CSR / host IF --> SEQUENCER (descriptor fetch, scoreboard, loops over L_PAD)            |
 |                        |  issue                                                           |
 |        +---------------+-------------------+----------------------+-------------------+  |
 |        v                                   v                      v                   v  |
 |  DMA engine (4 rd + 2 wr streams,   GEMM ENGINE              VECTOR/SFU UNIT       LoWA UNIT
 |   row->lane / transpose modes)      R x C output-stationary  V lanes, lane = row   merge-on-load
 |        |   ^                         FMA PEs, lambda accs     FMA + exp ln erf tanh  sampled grad
 |        |   |                         XBUF x2 | WBUF x2 | ACC  recip rsqrt sqrt      step scalars (lr,
 |        |   |                         epilogue: bias, scale,   max/sum, PRNG masks    beta^t), clip coef
 |        |   |                         GELU, dropout, residual  transpose stagers      row bufs, 256 accs
 |        +---+----- SRAM banks (sram256x8 macros) ---------------------------------------------+
 |  Threefry-2x32-13 mask generators (V/2 instances) shared by GEMM epilogue and vector unit  |
 +--------------------------------------------------------------------------------------------+
   WBUF fill path:  DRAM W0 block --> DMA --> [LoWA merge: W += (gamma c_j) h_e for hits in block] --> WBUF
```

## 3. Units

**GEMM engine.** R x C PEs (default 16 x 16), each one pipelined fp32 FMA (latency lambda = 4, RNE,
FTZ) and lambda accumulators. A pass computes an output micro-tile of R rows x (C*lambda) columns over
one K chunk of Kc = 64: PE (r,c) holds X[r,k] for lambda cycles while W rows c*lambda..c*lambda+3 stream
past, so the X side needs only R/lambda words/cycle. Operand modes per side: `ROW` (lane = matrix row,
stream along the row) and `TRN` (lane = matrix column; the DMA scatters one DRAM row across lanes).
Epilogue on drain (fused, V-wide): `+bias`, `*scale` (0.125 for scores, exact), `GELU` (stores pre-act
u to stash and g to scratch), `dropout(site)`, `+residual`, `GELU-bwd(u)`, `dropout-bwd(site)`, mask-row.
Each K chunk starts at +0.0 in the PE and is added into ACC at drain (`acc = add(acc, p)`, Kc = 64,
P1 contract, dataflow.md section 2).

**Vector / SFU unit.** V lanes (default 16); **lane = one row** for every row reduction (LN 768, softmax
128, CE C), lane = one element for element-wise ops, so reduction order never depends on V. Each lane:
fp32 FMA, add, mul, max, compare, select; SFU (shared pipelined block per lane): exp, ln, erf, tanh,
recip, rsqrt, sqrt (algorithms + ulp chosen and measured in P2/P4). V x V transpose stagers in flops turn
row-major DRAM bursts into one-word-per-lane streams.

**LoWA unit.** (a) *Merge engine* on the WBUF fill path: for every W block of a LoWA module it walks
the block's slice of the precompiled hit list and does `W[cell] = W[cell] + (gamma*c_j)*h_e` in hit-list
order (dataflow.md section 3). (b) *Sampled-gradient engine*: per token, per atom, reads the 7-word
windows of dY and X rows and accumulates 25 products into acc_j (D5). (c) *Step-scalar engine*: lr_t for
both groups, beta1^t, beta2^t, bias corrections, clip coefficient. The element-wise AdamW itself runs on
the vector lanes (avoids a second SFU), under P5's control/tests.

**DMA / AXI4.** AXI4 master, 256-bit data at core clock (needs 32 B/cycle in GEMM phases, see
budget.md), INCR bursts up to 16 beats, 4 read + 2 write streams, address generators with 3 nested
(count, stride) loops, the two lane modes, and a gather mode for embedding rows. AXI4-Lite slave for
CSRs and a host DRAM window.

**SRAM banks.** All from `gf180mcu_ocd_ip_sram__sram256x8m8wm1` (256 x 8 bit; 4 macros = 256 fp32 words
of one 32-bit lane). `sram64x8` (fd, 431.9 x 232.9 um) is *larger* than the 256x8 ocd macro and is not
used. Banking: memory_map.md section 3.

**Sequencer.** Fetches 64-byte descriptors from DRAM (`PROG`), issues them to the four units with a
scoreboard (per-unit busy + 16 event flags), runs the whole step (isa.md).

## 4. Per-op responsibility, forward and backward

N = B * L_PAD tokens, d = 768, h = 12, dh = 64, F = 3072. "stash" = written to DRAM for backward.
Masks M_* come from the PRNG; s = DROP_SCALE = fp32(1/(1-0.1)) = 0x3F8E38E4.

| op (HF) | forward (unit) | backward (unit) | stash |
|---|---|---|---|
| positions | pos = 1 + cumsum(id != 1) * (id != 1) (sequencer, per row) | none | - |
| embeddings | DMA gather word[id], pos[pos]; e = (w + tt0) + p; LN_emb; dropout M_emb (VEC) | none: frozen, nothing upstream is trainable | X0 = dropout(LN(e)) |
| Q, K, V proj | GEMM ROW x ROW, W = merged (Q, V) / plain (K); +bias | dX = [dQ dK dV] [Wq;Wk;Wv] as one GEMM, K = 2304 (skipped for layer 0); W in TRN mode, Q/V merged again on load | Q, K, V |
| LoWA grad | - | sampled engine on (dQ, X) and (dV, X): grad c_j = gamma * acc_j | - |
| scores | S = (Q K^T) * 0.125 (GEMM, epilogue scale), per (b, head) | recomputed: same GEMM | - |
| mask + softmax | VEC: m = max over unmasked j; e = exp(S - m), masked -> 0; r = recip(sum e); P = e * r; P' = M_att ? P*s : 0 | dP = M_att ? (dO V^T)*s : 0; rd = sum_j P dP; dS = (P*(dP - rd)) * 0.125 | row m, r |
| context | O = P' V (GEMM ROW x TRN) | dV = P'^T dO (TRN x TRN), dP' = dO V^T (ROW x ROW) | - |
| attn out | y = O Wo^T + bo; z1 = x + M_ao(y)*s (epilogue); h1 = LN1(z1) (VEC) | dO = drop(dz1) Wo (GEMM, drop-bwd on the A operand, P1); LN1-bwd (VEC) | z1, mu1, rstd1 |
| FFN1 + GELU | u = h1 W1^T + b1; g = GELU(u) (epilogue) | du = (dg) * GELU'(u) (epilogue of FFN2-dX GEMM) | u |
| FFN2 | y = g W2^T + b2; z2 = h1 + M_fo(y)*s; out = LN2(z2) | dg = drop(dz2) W2 (GEMM, drop-bwd on A); dh1 = du W1 + dz2 (GEMM, epilogue residual) | z2, mu2, rstd2 |
| head | x = M_c1(x_cls)*s; a = tanh(x Wd^T + bd); z = M_c2(a)*s Wo^T + bo | dWo = dz^T a', dbo = sum dz; da = dz Wo, drop-bwd, *(1 - a^2); dWd = da^T x, dbd = sum da; dx_cls = da Wd, drop-bwd -> layer-11 output grad (CLS rows only) | x_cls, a, masks regenerated |
| loss | CE: lse = m + ln(sum exp(z - m)), loss = mean_w(lse - z_y); MSE: mean_w (z - y)^2 | CE: dz = (softmax(z) - onehot) * w / B_ACT; MSE: dz = (z - y) * (2/B_ACT) * w | - |
| clip | per tensor n_i = sqrt(sum g^2) (28 tensors, torch order), G = sqrt(sum n_i^2), c = min(1, recip(G + 1e-6)) since max_norm = 1; g *= c | | |
| AdamW | VEC lanes, per element (spec of order in dataflow.md section 6) | | m, v in DRAM |

Exact-erf GELU as torch: g = (u*0.5) * (1 + erf(u * 0x3F3504F3)); GELU'(u) = 0.5*(1 + erf(u/sqrt2)) +
u * exp(-0.5 u^2) * 0.3989423 (torch `GeluCUDAKernelImpl` backward form). LN: eps = 1e-5, biased var.

Backward through frozen layers computes **only dX**. No dW is ever formed for a frozen weight; the only
weight gradients are dWd, dWo of the head (B = 32 reduction) and the 6144 coefficient gradients.

## 5. Why padding is exact
Masked keys get P = 0 exactly (exp is forced to 0, not computed from finfo.min), so they add +0 to
every P'V sum; in backward dS[:, j] = P[:, j] * (...) = 0 and dK[j] = dV[j] = 0 for a masked key j;
the head reads only CLS. By induction over layers every pad row (and every dummy batch row, whose loss
weight is 0) carries dY = 0, so its LoWA products are +-0 and its terms in every sum are +-0.
fma(a, +-0, acc) = acc for finite acc. The only visible effect is the sign of an exact zero.

## 6. Arithmetic (fixed decision 5) and what deviates from PyTorch
fp32 RNE everywhere, FTZ on inputs and outputs (documented; PyTorch on GPU keeps subnormals).
Deliberate differences from the reference, all reproduced exactly by L1:
GEMM = sequential FMA (cuBLAS: blocked, unspecified); division = recip * mul (softmax, AdamW m/denom,
CE/MSE 1/B); sqrt via SFU; index_add collisions summed in hit-list order (CUDA: atomics, unordered);
atom-major sampled sum (D5); LN two-pass stats (torch: Welford); step scalars (lr_t, beta^t) computed in
fp32 on chip (torch: Python double then cast); dropout RNG stream (D1).

## 7. Open questions for PI
1. **Sampled-gradient order (D5).** Fixed decision 4 is written as per-cell S then sum over taps. The
   chip sums per atom directly (256 accumulators instead of a 6,400-cell union buffer). Same products,
   different fp32 order. Accept, or require the S-first order (+25.6 KiB SRAM per module, +100 macros)?
2. **Dropout (D1).** On-chip Threefry masks, regenerated in backward; matches `train_glue` only in
   distribution, never bit for bit. Alternative: train with dropout off (a different procedure from
   every S.3 number). Which is the accepted reference?
3. **Off-chip DRAM link.** full16 needs ~800 MB/s sustained (32 B/cycle at 25 MHz). GF180 has no open
   DDR PHY; the realistic attachment is an FPGA holding the DRAM controller over a wide parallel bus.
   Pin count and IO speed of that link are undefined, and so is the host's initial 500 MB load path
   (slow AXI4-Lite window vs a host port on the DRAM side).
4. **Full-size target.** full16 projects to a ~116 mm2 core (the FMA area is a +-40 % assumption).
   Is that the projection the paper should state, or mid8 (~40 mm2, 3.7x slower)?
5. **Shuttle.** The smallest complete core (~2.1 mm2) does not fit the 1.86 mm2 block under the FMA
   assumption. Choose: PnR the full16 core as a block only (no silicon), or design a separate test
   vehicle (FMA + SFU + LoWA engines + serial port).
6. **FTZ.** PyTorch on GPU keeps subnormals; the chip flushes. Gradients of order 1e-38 are possible in
   the backward of deep layers. P1 must measure whether any reference value is subnormal on a real step.
7. **Energy.** Speed is ~100x below one GPU (120.7 s/step at L = 128). Should P8 report energy/step
   (power x time incl. DRAM traffic of 94.6 GB/step) as the primary figure of merit?
8. **Step scalars in fp32 on chip** (lr_t, beta^t, bias corrections) vs torch's Python-double values.
   Alternative: a host-precomputed per-step scalar table (8.8 MB for MNLI), which some would read as
   the schedule not running on chip.
9. **AdamW datapath** runs on the vector lanes (to avoid a second SFU); PLAN P5 lists AdamW under the
   LoWA unit. P5 would own its control and tests. OK?
10. **gamma for the reference run.** S.3 "sgwa" used 0.552427 with lr 0.166667; the derived default is
    0.138107 ("sgwaA", lr 0.666667). Which config is L0's reference step?
11. **L0 must run `--sgwa_sampled_grad 1`** (the reference runs used dense autograd). G10 says the two
    agree to fp32 rounding; confirm that the published numbers can be cited for a sampled-grad chip.
12. **CLS pruning (D6)** of layer 11 changes the work, not the math. OK to count it in the budget?
13. **SFU list grows**: sqrt (clip norm, AdamW), ln (loss value only) are needed in addition to PLAN's
    exp, tanh, erf, rsqrt, recip. Correctly rounded sqrt or a documented ulp bound?
14. **Masks depend on L_PAD** (element index uses the padded shape). Harmless for training, but L0 must
    use the same padded shape. OK?
