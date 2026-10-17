# LoWA-TA instruction set (P0)

The sequencer executes **tile-level macro-ops** described by fixed 64-byte descriptors (16 x 32-bit
words, little-endian) stored in DRAM region `PROG`. There are no scalar instructions; every loop over
tiles, heads and K chunks is done in hardware inside a macro-op. One training step = one program run.

## 1. Descriptor common header (word 0)

| bits | field | meaning |
|---|---|---|
| 7:0 | `OP` | opcode (table below) |
| 11:8 | `UNIT` | 0 seq, 1 DMA, 2 GEMM, 3 VEC, 4 LOWA |
| 15:12 | `WAIT` | wait until event flag `WAIT` is set (0xF = none) |
| 19:16 | `SIG` | set event flag `SIG` when this op retires (0xF = none) |
| 20 | `CLR` | clear flag `WAIT` on issue (consume) |
| 31:21 | `TAG` | debug tag, copied to `PERF_LAST_TAG` |

Issue rule: in program order; an op issues when its unit's queue (depth 2) has room and its `WAIT`
flag is set. Units run concurrently; ordering between units is **only** through flags (the compiler
places them). This is what lets the LoWA sampled-grad op of layer l run beside the GEMMs of layer l-1.

## 2. Dimension and address encoding
A **dim** field is 16 bits: `[15:14] sel, [13:0] imm`; value = imm * {1, L_PAD, B*L_PAD, B}[sel].
An **addr** is a 32-bit byte offset into DRAM plus an optional 8-bit `layer` multiplier
(`addr + layer * LAYER_STRIDE[region]`), so one layer body is emitted once and looped by `LOOP`.
A **stride** is 32 bits, bytes, with the same `sel` scaling in its top 2 bits.

## 3. Opcodes

| OP | name | unit | operands (words 1..15) | does |
|---|---|---|---|---|
| 0x01 | `GEMM` | GEMM | A addr/stride/mode, B addr/stride/mode, out addr/stride, M, N, K (dims), batch count + A/B/out batch strides, `MERGE` (module id or none), epilogue word (below), bias addr, aux addr (residual / u / mask-row) | C[M x N] = A[M x K] B^T in R x C*lambda tiles, all K chunks, all batches (heads); merge-on-load if MERGE; epilogue on drain |
| 0x02 | `GEMM_ACC2` | GEMM | as GEMM with a second (A, B) pair concatenated along K | [dQ dK dV][Wq;Wk;Wv] (K = 3 x 768 in order Q, K, V) |
| 0x10 | `EMB` | VEC+DMA | ids addr, pos rule, word/pos/tt/LN addrs, out addr | gather + add + LN_emb + dropout(site EMB) -> X0 |
| 0x11 | `LN_FWD` | VEC | in, gamma, beta, out, stats addrs, rows | two-pass LN, writes mu, rstd |
| 0x12 | `LN_BWD` | VEC | z, stats, gamma, dy, dx, residual-grad addrs | recomputes xhat, writes dz (+ residual grad) |
| 0x13 | `SMAX_FWD` | VEC | S, mask, P', stats addrs, rows, dropout site | max, exp, sum, recip, scale, mask, dropout |
| 0x14 | `SMAX_BWD` | VEC | P (recomputed), dP', dS addrs, dropout site | dP = drop-bwd(dP'), rowdot, dS (x 0.125) |
| 0x15 | `ELEM` | VEC | in0, in1, out, n, sub-op (add, mul, scale, drop, drop-bwd, tanh, tanh-bwd, gelu, gelu-bwd, copy) | element-wise |
| 0x16 | `LOSS` | VEC | logits, labels, weights, dlogits, loss out, mode (CE/MSE), C | loss + dlogits |
| 0x17 | `REDUCE_ROWS` | VEC | in, out, rows, cols | column sum over rows (bias grads) |
| 0x20 | `LOWA_SG` | LOWA | module id, dY addr/stride, X addr/stride, tokens (dim), grad addr | sampled coefficient gradient for one module |
| 0x21 | `LOWA_SCAL` | LOWA | step-scalar block addr | lr_t (2 groups), 1-lr_t*wd, beta1^t, beta2^t, step sizes; writes OPT_SCAL |
| 0x22 | `GNORM` | VEC+LOWA | grad region, tensor table addr (28 entries: offset, length) | per-tensor norms, total norm, clip coef c, g *= c |
| 0x23 | `ADAMW` | VEC | param/grad/m/v addrs, length, group (0 coef, 1 head) | one AdamW update of a contiguous range |
| 0x30 | `DMA_COPY` | DMA | src, dst, 3-level (count, stride) | general copy (stash copies, head buffers) |
| 0x31 | `FENCE` | seq | flag mask | wait for all listed units idle |
| 0x32 | `LOOP` | seq | body length, count, layer-index register, direction (+1 / -1) | hardware loop; sets `layer` for addr fields |
| 0x33 | `SETCSR` | seq | csr id, value | e.g. PRNG site/layer, perf marker |
| 0x3F | `END` | seq | - | retire step: t += 1, write LOSS_OUT, STEP_DONE += 1, IRQ |

Epilogue word (GEMM): `[3:0]` op chain id from {none, bias, bias+scale, bias+GELU(store u, g),
bias+drop+residual, drop-bwd, gelu-bwd(u), +residual}, `[11:4]` dropout site id, `[12]` row-mask enable,
`[31:16]` scale exponent (power-of-two scales only, exact).

## 4. A training step as one program (forward 12 layers, head, backward 12 layers, optimizer)
```
EMB                                   ; X0
LOWA_SCAL                             ; scalars for this step (needs t only)
LOOP 12 up {                          ; layer l
  GEMM q (MERGE=q_l) ; GEMM k ; GEMM v (MERGE=v_l)      -> stash Q,K,V
  GEMM S=QK^T*0.125 (batch = B*12 heads) ; SMAX_FWD ; GEMM O=P'V
  GEMM o-proj (bias+drop+residual -> z1) ; LN_FWD (z1 -> h1)
  GEMM ffn1 (bias+GELU, stash u) ; GEMM ffn2 (bias+drop+residual -> z2) ; LN_FWD (z2 -> next X)
}                                      ; layer 11 body is a separate CLS-pruned copy
ELEM drop(x_cls) ; GEMM dense ; ELEM tanh ; ELEM drop ; GEMM out_proj ; LOSS
GEMM dWo ; REDUCE_ROWS dbo ; GEMM da ; ELEM drop-bwd, tanh-bwd ; GEMM dWd ; REDUCE_ROWS dbd ; GEMM dx_cls
LOOP 12 down {
  LN_BWD (z2) ; GEMM dg = drop(dz2) W2 (drop-bwd on A; gelu-bwd epilogue -> du) ; GEMM dh1 = du W1 (+residual)
  LN_BWD (z1) ; GEMM dO = drop(dz1) Wo (drop-bwd on A)
  GEMM S recompute ; SMAX recompute (stats) ; GEMM dP' = dO V^T ; SMAX_BWD
  GEMM dV = P'^T dO ; GEMM dQ = dS K ; [l>0] GEMM dK = dS^T Q
  [l>0] LN_FWD recompute X_l = LN2_{l-1}(z2_{l-1}) into scratch (l = 0: STASH_X0)
  LOWA_SG q_l (SIG f) ; LOWA_SG v_l (SIG f)           ; runs beside the next GEMMs
  [l>0] GEMM_ACC2 dX = [dQ dK dV][Wq;Wk;Wv] (MERGE q/v) (+residual dz1)
}
FENCE all ; GNORM ; ADAMW coef ; ADAMW head ; END
```
Exact op count per step is emitted by the P6 compiler; at ~40 descriptors per layer body the program
is a few KiB (the 256 KiB `PROG` region leaves room for unrolled debug variants).

## 5. Host kickoff
1. Once: host writes `PROG`, `CONST`, weights, LoWA tables, head init, zeroes `PARAM.coef`, `ADAM_*`,
   sets `PROG_BASE`, region bases, `SEED`, `T_TOTAL`, `WARMUP`, then `CTRL.RESET_T`.
2. Per step: host writes the step descriptor + ids + mask + labels + weights into `STEP_IN[slot]`
   (8,848 B at B=32, L=128), then writes `DOORBELL = slot | (1 << 31)`. Two slots: host fills slot
   s^1 while slot s runs.
3. Device runs the program, at `END` sets `STEP_DONE`, raises IRQ (if enabled), writes loss.
4. End of training: host reads `PARAM` (6,144 coefs + head).

## 6. CSRs (AXI4-Lite, 32-bit)

| addr | name | access | content |
|---|---|---|---|
| 0x000 | `ID` | R | 'LWTA' |
| 0x004 | `CFG` | R | R, C, V, lambda, macro counts (synthesis parameters) |
| 0x008 | `CTRL` | RW | bit0 run-enable, bit1 soft reset, bit2 RESET_T, bit3 DROP_EN, bit4 IRQ_EN, bit5 single-op debug |
| 0x00C | `STATUS` | R | busy per unit, current PC, error code |
| 0x010 | `DOORBELL` | W | [0] slot, [31] go |
| 0x014 | `STEP_DONE` | R | completed steps (wraps) |
| 0x018 | `STEP_T` | RW | optimizer step counter t (Adam), s = t (scheduler) |
| 0x020-0x05C | region bases | RW | PROG, CONST, EMB, LAYERS, LOWA, PARAM, GRAD, ADAM_M, ADAM_V, STEP_IN, STASH, SCR, LOSS_OUT, plus `LAYER_STRIDE` |
| 0x060 | `SEED_LO/HI` | RW | Threefry key |
| 0x068 | `T_TOTAL`, `WARMUP` | RW | schedule integers (also in CONST as fp32 reciprocals) |
| 0x070 | `ERR` | R/W1C | FP invalid seen, AXI error, descriptor decode error |
| 0x080-0x0FC | `PERF_*` | R | cycles per unit busy, DRAM bytes rd/wr, stall cycles, last tag |
| 0x100 | `HWIN_ADDR` / `HWIN_DATA` | RW | host DRAM window (indirect, auto-increment) |

## 7. As built (P6c) — binding; supersedes sections 1-3, 5 and 6 where they differ
Single source of the encoding: `sw/isa.py` (compiler, interpreter and sequencer tests import it); RTL: `hw/ctrl/ct_seq.v`.

**Descriptor.** 1-3 slots of 64 B (the P0 fixed 64 B cannot hold a GEMM: 36 payload words). Word 0 header:
`[7:0] OP, [11:8] UNIT (0 seq, 1 DMA, 2 GEMM, 3 VEC, 4 LOWA), [15:12] WAIT, [19:16] SIG, [20] CLR, [22:21] NSL-1,
[31:23] TAG` (TAG shrank from 11 to 9 bits for NSL). Words 1..2·NSL are **REL** words: nibble i (word 1 + i/8, bits
4(i mod 8)) selects the term added to descriptor word i at decode: `word += TERM[sel]`. Payload p0.. follows.
`TERM[0] = 0`, `TERM[1..12]` = layer terms (0 at LOOP start, ± `STR[k]` per iteration; `STR[k]` set by SETCSR k),
`TERM[13]` = loop index, `TERM[14]` = STEP_T, `TERM[15]` = SLOT_DELTA if the doorbell slot is 1 else 0. This is how
one loop body addresses layer l (weights, stash, LoWA tables, RNG layer field), the step counter reaches
LOWA_SCAL, and the STEP_IN slot is selected. The P0 `dim` sel scaling is not implemented: a program is compiled for
one (B, L); see the P6c report.

| OP | name | UNIT | NSL | payload (sw/isa.py) |
|---|---|---|---|---|
| 0x01 / 0x02 | GEMM / GEMM_ACC2 | 2 | 3 | gm_top descriptor (m, n, k0-2, nb1/2, flags incl. MERGE, sexp, mask site / layer, A/B bases x3 + strides, C, bias, aux, o2, ds) + lw_top merge descriptor (nb, mb, kc0-2, ms0-2, hits / dir / coef a and b, h, gamma) |
| 0x10 / 0x16 / 0x22 | VEC / LOSS / GNORM | 3 | 2 | vu_top descriptor (vop = flow/gen/vu_ucode.py VOPS index, st_en, x/g reset, mask site / layer, rows, cols, rows_per_b, emb_l, 16 addresses / strides, c0-c3, ds) |
| 0x20 / 0x21 / 0x23 | LOWA_SG / LOWA_SCAL / ADAMW | 4 | 1 | lw_top main descriptor (op, flag, a0-a7, len) |
| 0x30 | DMA_COPY | 1 | 1 | src, dst, n0 (words), n1, src / dst stride 1, n2, src / dst stride 2 (bytes) |
| 0x31 | FENCE | 0 | 1 | unit mask (bit0 DMA, bit1 GEMM incl. its merge, bit2 VEC, bit3 LOWA): wait until those units are idle |
| 0x32 | LOOP | 0 | 1 | body bytes (the descriptors after LOOP), count, start index, dir (0 up, 1 down); not nested |
| 0x33 | SETCSR | 0 | 1 | id, value (1..12: STR[id]) |
| 0x3F | END | 0 | 1 | waits until all units idle; STEP_T += 1, STEP_DONE += 1, IRQ |
| 0x00 | NOP | 0 | 1 | |

ELEM / LN_FWD / ... are VOPs of the one VEC opcode; ADAMW is a LOWA op (lw_adamw); REDUCE_ROWS does not exist (the
head bias gradients are a 1 x B ones-GEMM, exact for B <= 64; the compiler asserts it). An illegal (OP, UNIT, NSL)
sets ERR.decode and halts.

**Issue / retire (dependency rule).** Descriptors issue in program order. A unit op issues when its WAIT flag is
set (or WAIT = 0xF) and its unit is free; CLR clears the flag at issue. Each unit runs **one op at a time** (queue
depth 1 plus the decoder's staging buffer; P0 said 2). An op retires when the engine's done has been seen (a GEMM
with MERGE also needs lw_top's merge done), two cycles have passed, and the unit's memory write channel is idle
(all writes B-acknowledged, P6a); then the unit is free and the SIG flag (if any) is set. Hence a consumer that
waits for a flag, a FENCE or a following op of the same unit always reads the producer's data from DRAM.
Flag rule for the compiler: a flag is signalled again only by an op placed after the consumer that cleared it.
The P6c compiler orders cross-unit dependences with FENCE only (computed from every op's read / write byte ranges),
so the LoWA sampled gradient runs beside the following GEMM / vector ops until a reader of its output or a writer
of its inputs.

**Host interface (CSR bus, 32-bit, one access / cycle; P7 puts the AXI4-Lite slave in front).**
`0x000 ID 'LWTA'`, `0x008 CTRL` ([0] run-enable, [1] soft reset (self-clearing), [2] RESET_T (STEP_T := 0),
[3] DROP_EN (stored, masks are always on), [4] IRQ_EN, [5] reserved (single-op debug not built)),
`0x00C STATUS` ([3:0] units busy DMA/GEMM/VEC/LOWA, [5] running, [6] error, [31:16] PC[21:6]),
`0x010 DOORBELL` (W: [31] go, [0] slot; ignored unless run-enable and idle), `0x014 STEP_DONE` (R; any write
acknowledges the IRQ), `0x018 STEP_T`, `0x020 PROG_BASE`, `0x024 SLOT_DELTA` (byte distance STEP_IN slot 1 - slot 0),
`0x060 / 0x064 SEED_LO / HI` (Threefry key = mk_seed; mk_step = STEP_T[23:0]), `0x070 ERR` (W1C, [0] decode),
`0x080 PERF_CYCLES` (since the doorbell), `0x084 LAST_TAG`, `0x088 PERF_DESC` (descriptors issued), `0x08C FLAGS`,
`0x090 PC`. Region-base CSRs, T_TOTAL / WARMUP and HWIN are not built: the compiler embeds absolute addresses and
the schedule integers in the program, and the host loads DRAM through the DRAM's own host port (ruling R3).
IRQ is a level, raised at END while CTRL.IRQ_EN.

**Memory.** The sequencer has one read channel (descriptor fetch: one 48-word request per descriptor; shared with
DMA_COPY reads) and one write channel (DMA_COPY) on mm_top (NR + 1, NW + 1 in P7's map).
