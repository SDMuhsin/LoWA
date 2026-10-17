# LoWA-TA dataflow and the accumulation-order contract (P0)

Sections 2-7 are the **P1 hwmodel contract**: L1 must reproduce every rounding listed here, and L2 must
match L1 bit for bit. Everything is fp32 RNE with FTZ (inputs and results that are subnormal become
+-0 with the sign kept). `fma(a,b,c)` = a*b+c with one rounding. `mul`, `add` round once each.
Nothing in the contract depends on R, C, V or lambda: those change only the speed.

## 1. Step schedule (order of macro-ops, see isa.md section 4)
```
EMB -> LOWA_SCAL -> [L0 .. L11 forward] -> head fwd + LOSS -> head bwd
    -> [L11 .. L0 backward, LOWA_SG(q_l), LOWA_SG(v_l) overlapped with the next GEMMs]
    -> FENCE -> GNORM -> ADAMW(coef) -> ADAMW(head) -> END
```
Within a layer, GEMMs run back to back on the array; vector passes (LN, softmax) run between them; the
WBUF of the next GEMM is prefetched and merged during the current one. The only cross-unit overlap the
budget counts is the LoWA sampled-gradient engine running beside the array (it has its own DRAM
streams).

## 2. GEMM: tiling, operand modes, reduction order

C[M x N] = A[M x K] * B^T, B stored [N x K] (torch's [out x in] for weights).
Loop nest (output-stationary, full16 values in brackets):
```
for nb in 0 .. N/Bc-1          (Bc = C*lambda = 64)       W block row
  for mb in 0 .. M/Br-1        (Br = 64)                  token block
    for kc in 0 .. K/Kc-1      (Kc = 64), in ascending kc
      DMA:  XBUF[next] <- A[mb, kc]   WBUF[next] <- B[nb, kc]  (+ LoWA merge, section 3)
      for mt in 0 .. Br/R-1    micro-tile of R = 16 rows, partials from ACC bank (mt mod 4)
        PE(r,c) for k in kc*Kc .. kc*Kc+63:   for q in 0..3:  acc[q] = fma(A[r,k], B[c*4+q,k], acc[q])
      (after the last kc: drain through the epilogue)
```
**Reduction order (contract, changed in P1, see arith.md section 2):** for every output element and each
K chunk of Kc = 64 (the last may be short), the PE does `p = +0.0; p = fma(a_k, b_k, p)` for k ascending
within the chunk. The ACC drain adder then does `acc = add(acc, p)`, with acc starting at +0.0 and the
chunks taken in ascending order. After that comes `y = add(acc, bias)` if there is a bias, and then the
epilogue chain in the order written in isa.md (scale, activation, dropout, residual add). Kc is fixed
at 64 by the contract, independent of R, C, V and lambda. (P0 had one FMA chain over all K. On the real
step it was about 3x less accurate than cuBLAS fp32 and failed the P1 gate, so it was replaced, see
p1_report.md.) `GEMM_ACC2` continues the same chunk sequence through the second (A, B) pair: k runs over
the Q-part, then the K-part, then the V-part (K = 2304 = 36 chunks).
**Dropout backward sits on the A operand, not in the epilogue.** For `dO = dz1' Wo` and `dg = dy2 W2`, the
mask belongs to the GEMM input (`dz1' = drop(dz1)`, `dy2 = drop(dz2)`, each `keep ? mul(x, DS) : 0`).
It is applied on the A-side load path or by an ELEM pass before the GEMM. The epilogue applies only the
gelu-bwd of dg.

Operand modes (the DMA does the layout, the array never sees a transpose):

| GEMM | A side (lanes = rows of C) | B side (lanes = cols of C) | M x N x K |
|---|---|---|---|
| X W^T (all projections, fwd) | X ROW | W ROW | N x out x in |
| dY W (all dX, bwd) | dY ROW | W TRN (row o of W scattered across k-lanes) | N x in x out |
| S = Q K^T | Q ROW (head slice) | K ROW | L x L x 64, batch B*12 |
| O = P' V | P' ROW | V TRN | L x 64 x L |
| dP' = dO V^T | dO ROW | V ROW | L x L x 64 |
| dV = P'^T dO | P' TRN | dO TRN | L x 64 x L |
| dQ = dS K | dS ROW | K TRN | L x 64 x L |
| dK = dS^T Q | dS TRN | Q TRN | L x 64 x L |
| head dW = dZ^T X | dZ TRN | X TRN | out x in x B (reduction over the batch, b ascending) |

ROW: DRAM row r, words k..k+63 -> lane r, consecutive addresses. TRN: DRAM row k (a 64-byte burst of
16 words) -> one word into each of 16 lanes at the same address. Both are plain INCR bursts.

## 3. LoWA merge-on-load (forward Q/V projections and the backward dX GEMMs that use Wq, Wv)

Per module, the compiler expands `SGWALinear.__init__` exactly: centres `randperm(m*n, seed 777)[:256]`,
rows/cols = reflect(ci + di), reflect(cj + dj) for the 25 diamond taps in `torch.nonzero(H)` order
(row-major over the 7 x 7 box), giving entry e = 25*j + tap with flat cell `eflat[e]`.
Each entry becomes a 32-bit hit `{blk_row[3:0] (Nb), blk_col[3:0] (kc), lane[3:0], q[1:0], kk[5:0],
tap[4:0], spare}` plus the atom index j implied by a per-block run-length header; hits are sorted by
(W block, entry index e ascending). 6,400 hits/module, ~44 per 64 x 64 block.

When the DMA has written W0 block (nb, kc) into WBUF[next], the merge engine, before the buffer is
released to the array, executes for each hit of that block in list order:
```
v   = mul(gamma, c_j)          (once per hit; identical every time, so recomputation is harmless)
val = mul(v, h_tap)
W[cell] = add(W[cell], val)    (read-modify-write of one WBUF word)
```
This is the reference's `vals = (scaling*coef)[:,None]*hval; W0.index_add(eflat, vals)` with a defined
order for colliding cells (overlapping atoms, reflected duplicates): ascending e. For TRN-mode loads
(backward) the same hits are used with lane/address swapped. Merged values are never written back to
DRAM. Cost: ~44 RMW per block vs 1,024 array cycles per block: hidden behind the double buffer.

## 4. LoWA sampled coefficient gradient (no dense m x n buffer)

Reference: `_SGWASampledLinear.backward`: S[u] = sum_t G[t,i_u] X[t,l_u] over the union U, then
grad c_j = gamma * sum_e h_e S[cell(j,e)]. Chip (D5, flagged for PI): accumulate per atom directly.

Inputs: dY = dQ_l or dV_l rows [N x 768] (written by the attention-backward GEMMs), X = X_l rows
[N x 768] (STASH_X0 for l = 0, otherwise LN2 recompute). For a module and t = 0..N-1 ascending:
```
load row dY[t, :] and row X[t, :] into the two 768-word row buffers
for j = 0..255:
   read window  yw[0..6] = dY[t, refl(ci_j - 3 .. ci_j + 3)],  xw[0..6] = X[t, refl(cj_j - 3 .. cj_j + 3)]
   for tap = 0..24 (nonzero(H) order, (di, dj) offsets):
        p      = mul(yw[di+3], xw[dj+3])
        acc_j  = fma(h_tap, p, acc_j)          acc_j starts at +0.0 for each module and step
grad c_j = mul(gamma, acc_j)
```
Why no m x n buffer: each (t, j) needs only 7 + 7 words of the two current rows; the state that
persists across tokens is 256 accumulators. Products touch exactly the k*s = 6,400 cells per token
(the union duplicates are re-multiplied, which is what makes the union table unnecessary).
Window reads: 1 word/cycle per row buffer, so 7 cycles per atom with 4 multipliers + 4 FMAs
(25 products in 7 cycles); the 25-long FMA chain of one atom is interleaved across atoms to hide the
FMA latency, which does not change the per-atom order. Last layer, query module: only the 32 CLS rows
are fed (all other dQ rows are exactly zero, D6).

## 5. Vector unit orders (lane = row)
- All LN sums below are **chunked sums** (arith.md section 2, runs of 64, partials added in order, P1).
- LN fwd: `s = sum x_i (add, i ascending); mu = mul(s, INV_D)` with INV_D = fp32(1/768);
  `q = sum fma(x_i - mu, x_i - mu, .)` i ascending; `var = mul(q, INV_D)`; `rstd = rsqrt(add(var, 1e-5))`;
  `xhat_i = mul(sub(x_i, mu), rstd)`; `y_i = fma(xhat_i, g_i, b_i)`. Stash mu, rstd.
- LN bwd: `gy_i = mul(dy_i, g_i)`; `a = sum gy_i`, `b = sum fma(gy_i, xhat_i, .)` (one pass, i ascending,
  xhat recomputed exactly as above); `dx_i = mul(rstd, sub(sub(gy_i, mul(a, INV_D)), mul(xhat_i, mul(b, INV_D))))`.
- softmax fwd: m = max over unmasked j; `e_j = exp(sub(s_j, m))` (masked: 0); `sum` j ascending;
  `r = recip(sum)`; `P_j = mul(e_j, r)`; `P'_j = keep ? mul(P_j, DROP_SCALE) : 0`. Stash m, r.
  Backward recompute uses the stashed m, r, so P is bit-identical to forward.
- softmax bwd: `dP_j = keep ? mul(dP'_j, DROP_SCALE) : 0`; `rd = sum fma(P_j, dP_j, .)`;
  `dS_j = mul(mul(P_j, sub(dP_j, rd)), 0.125)`.
- GELU, tanh, CE, MSE: spec.md section 4 formulas, evaluated left to right as written.
- Bias gradient (head): `sum over b ascending` (add).

## 6. Optimizer (after the whole backward)
1. Clip: for each of the 28 tensors in `model.parameters()` order (q0, v0, q1, ..., v11 coefs; dense.W,
   dense.b, out_proj.W, out_proj.b), viewed as rows of its last dimension: `rs = sum fma(g, g, .)` along
   each row, then `ss = sum add(rs)` over the rows, both as chunked sums (P1: a single 589,824-term chain
   for dense.W lost 5e-4 of the norm), `n_i = sqrt(ss)`;
   `G = sqrt(sum fma(n_i, n_i, .))`; `c = min(1, recip(add(G, 1e-6)))`; `g = mul(g, c)` (always applied,
   as torch).
2. Step scalars (LOWA_SCAL, fp32): s = t-1 completed steps; `f = s < w ? mul(s, INV_W) : max(0,
   mul(T - s, INV_TW))` (integers converted exactly, INV_W, INV_TW host fp32 constants);
   `lr = mul(lr0, f)`; `b1t = mul(b1t, 0.9)`, `b2t = mul(b2t, 0.999)` (running products from 1.0);
   `bc1 = sub(1, b1t)`, `bc2s = sqrt(sub(1, b2t))`; `step = mul(lr, recip(bc1))`; `decay = sub(1, mul(lr, wd))`.
3. Per element (torch `_multi_tensor_adamw` order): `p = mul(p, decay)`; `m = fma(0.1, sub(g, m), m)`
   (torch lerp, weight 0.1 < 0.5); `v = fma(0.001, mul(g, g), mul(v, 0.999))`;
   `den = add(mul(sqrt(v), recip(bc2s)), 1e-8)`; `p = sub(p, mul(step, mul(m, recip(den))))`.

## 7. Dropout masks
Threefry-2x32-13 with key (SEED_LO, SEED_HI) and counter (c0, c1) = (element_index >> 1, step << 8 |
site << 4 | layer); output word x[element_index & 1]; `keep = (x >> 8) >= 1677722`
(P(drop) = 1677722 / 2^24 = 0.10000002). Site ids: 0 emb, 1 attn-probs, 2 attn-out, 3 ffn-out,
4 cls-1, 5 cls-2; element_index = row-major index in the tensor the mask applies to (attention:
((b*12 + h)*L_PAD + i)*L_PAD + j with L_PAD, so masks depend on L_PAD; documented, harmless).
L0 bridge (P1) injects these masks into an eager-attention HF model.
