# P1 report: L0 bridge, L1 hwmodel, L1 vs L0 gate

`make model-check` (in src/rtl) runs the arithmetic proof, then L0 (GPU cuda:0, only if the dumps are
missing or stale), then L1 on both steps, then the gate. Result: **ARITH PASS, GATE PASS (A and B)**,
49 s when the L0 dumps already exist.

## 1. What exists
| file | role |
|---|---|
| `model/rng.py` | Threefry-2x32-13 + `keep_mask`, the one implementation shared by L0 and L1. Checked against the 3 Random123 KAT vectors |
| `model/l0_ref.py` | L0 bridge. train_glue's `parse_args` on the reference flags, its seeding/model/adapter/loader order, its optimizer resolver, param groups and scheduler. Eager attention with every dropout replaced by the chip's mask (R2). fp32 on cuda:0 and float64 on CPU |
| `model/hwmodel/csrc/lwta.c` | L1 core in C: primitives, SFU algorithms, GEMM, LN, softmax, LoWA merge and sampled grad, CE, GNORM, AdamW |
| `model/hwmodel/__init__.py` | numpy entry points, also the golden-vector API for P2-P5 (arith.md section 5) |
| `model/hwmodel/step.py` | L1 full step (spec section 4 and isa section 4, op by op) |
| `model/hwmodel/test_arith.py`, `fit_poly.py` | arithmetic proof; derivation of the polynomial constants |
| `model/check_l1_l0.py` | the gate |
| `docs/arith.md` | arithmetic contract for P2 (primitives, FTZ, chunked sums, SFU algorithms, constants, measured ulp) |

**Reference step.** roberta-base on RTE, train seed 42 (first seed of the S.2-D/S.3 protocol), with
the R6 hyperparameters: gamma 0.552427, lr 0.166667, classifier lr 5e-3, wd 0.01,
`--sgwa_sampled_grad 1`, warmup 140, 30 epochs (T = 2340 steps), fp32, TF32 off. The batch is 32 real
examples with train_glue tokenisation and `DataCollatorWithPadding`. Both batches pad to L = 128.
The eager-attention build is checked to give the same head init as train_glue's SDPA build.
Two steps were defined before any run:
- **A**: the first optimizer step (s = 0). The linear schedule makes lr = 0 here, so the updated
  coefficients and head equal the old ones exactly, and c = 0 means the merge adds zeros. A alone
  cannot test the merge or the update.
- **B**: s = 140, at peak lr. Start from the state after 140 real train_glue steps (stock dropout,
  batches 1-140 of the same seeded loader), then run batch 141 with the chip's masks. This exercises
  merge-on-load with trained coefficients, bias correction at t = 141, and a non-trivial AdamW update.

Dumps per step and level: inputs, loss, logits, 24 x 256 coef grads, the 4 head grads, total norm,
clip coefficient, post-step coefs/head, Adam m and v, and the lr values.
L0-64 = a deep copy of the fp32 model in float64 (identical parameter values, gamma = fp32(gamma),
DS = fp32(1/0.9)), so it is the exact-arithmetic value of the same function.

## 2. Arithmetic (details and constants in arith.md)
- **FMA and add/mul:** the x86 SSE/AVX-512 units under MXCSR RNE|FTZ|DAZ. They are compared bit for
  bit with an exact integer softfloat reference on 311,565 directed and random cases each, with 0
  mismatches. This also establishes that x86 FTZ is tininess-after-rounding, which is the semantics
  arith.md fixes.
- **SFU, exhaustive over all fp32 inputs in the domain** (max ulp): exp 0.879, recip 0.502,
  rsqrt 0.850, sqrt 0.500 (correctly rounded on every input), tanh 1.082, erf 1.013, ln 0.921.
  - exp: Cody-Waite + degree-6 polynomial + two exact scaling multiplies.
  - recip: linear seed + 3 FMA Newton steps.
  - rsqrt: 0x5F3759DF seed + 3 Newton steps.
  - sqrt: rsqrt + one FMA residual correction.
  - tanh: odd degree-8 polynomial below 1, `1 - 2/(e^2a + 1)` above.
  - erf: degree-6 polynomial below 0.921875, `1 - exp(-(a + a R(a)))` with a degree-9 R above.
- **L1 runtime:** 8.2-8.6 s for a full B = 32, L = 128 step on 24 cores (48 threads). About 5.3 s of
  that is the numpy Threefry mask generation. L1 is bit-identical across thread counts (7 vs 48
  threads checked).

## 3. Deviations from P0 (all inside src/rtl; contract text updated in dataflow/spec/isa)
1. **GNORM order (dataflow section 6).** P0 specified one sequential `fma(g, g)` chain per tensor. For
   `classifier.dense.weight` (589,824 positive terms), small squares are rounded away once the sum is
   large, and the norm comes out **5.1e-4** low. The total norm and clip were 790x worse than fp32,
   which failed the gate. Fix: rows of the last dimension, then the row sums, both as chunked sums.
2. **Chunked accumulation, Kc = 64 (dataflow section 2, LN in section 5).** P0's single FMA chain over
   all K (768-3072) and over LN's 768 terms gives logits 3.4x (A) / 1.9x (B) less accurate than
   cuBLAS fp32. On step B it **fails** the gate: `post_m_dw` max is 22.4x > 10x (table below). Each
   64-long K chunk now starts at +0 in the PE and is added into ACC at drain. This is still
   independent of R, C, V and lambda. Hardware cost: one adder per drain lane, instead of preloading
   ACC into the PE. The same rule applies to the LN sums.
3. **Dropout backward placement.** spec/isa put drop-bwd in the epilogue of `dO = dz1' Wo` (and
   implicitly for `dg`). That is wrong: the mask indexes the GEMM's input, not its output. L1 applies
   it to the A operand, and spec/isa/dataflow are corrected. P3 must provide it on the A-load path
   (or as an ELEM pass).
4. **L0 runs with torch deterministic algorithms (warn_only).** Without them, CUDA `index_add`
   atomics on colliding LoWA cells made the 140-step state of B differ run to run. That is a real
   problem for a gate that `make` reruns. The dumps are now bit-reproducible (checked over 2 runs).
5. Added step B (above). Layer 11 is computed for all rows in L1. CLS pruning (D6) only drops rows
   whose results are exact zeros or never reach the head, so every non-zero value is identical.

Not changed: the LoWA sampled-gradient order D5 (atom-major, one sequential chain per atom over
N x 25 products). Its gate ratios are 0.96x / 0.72x median, so it needs no chunking.

## 4. Gate (criterion fixed before the first run, never moved)
Truth = L0-64. PASS iff, on every quantity, median(L1 err) <= 4 x median(L0-fp32 err) and
max(L1 err) <= 10 x max(L0-fp32 err). There are 24 quantities per step: loss, logits, coef grads,
4 head grads, gnorm, clip, post coef/head (5), and post m and v (10).

Final contract, reproducible dumps: **step A 24/24 PASS, step B 24/24 PASS.** The table shows the
median ratio (L1 err / L0-fp32 err) for the key quantities:

| quantity | A ratio med / max | B ratio med / max | B L1 median err |
|---|---|---|---|
| loss (1 value) | 1.00 / 1.00 | **3.85** / 3.85 | 1.2e-7 |
| logits (64) | 1.25 / 0.91 | 0.83 / 0.60 | 2.3e-7 |
| coef grads (6144) | 0.96 / 0.78 | 0.72 / 0.22 | 5.6e-6 |
| head dW / out_proj W grads | 0.59 / 0.28, 0.64 / 2.82 | 0.62 / 0.47, 0.83 / 0.57 | 1.4e-6 |
| gnorm, clip (1 value each) | 0.63, 0.63 | 1.00, **3.29** | 3.0e-8, 9.9e-8 |
| post coefs (6144) | exact (lr = 0) | 1.00 / 1.00 | 7.3e-5 |
| post m, v coef | 0.96, 0.96 | 0.69, 0.78 | |
| worst max ratio | 2.82 (g_ow) | 6.10 (post_m_dw) | |

P0 order (KC = 0, still with fix 1), same dumps: step A passes; **step B FAILs** on post_m_dw max
(22.35x). Logits median 3.42x (A) and 1.89x (B).

**Honest caveat: the single-value quantities are rounding luck.** On the earlier non-deterministic
dump of B (same batch and contract; the state differed only by the atomics noise), the gate
**FAILED** on gnorm (19.2x) and clip (24.7x). L1's norm error there was 1.8e-7, about 2 ulp. But
L0-fp32's G happened to be exactly the correctly rounded truth (9.3e-9). The 4x bound was therefore
3.7e-8, smaller than the 0.5-ulp error (4.2e-8) a correctly rounded fp32 result can have. The
decomposition shows that the exact norm of L1's gradients is off by -1.5e-7 in both A and B, while
fp32-GPU's is off by -2.5e-7 (A) and -7e-8 (B). Both sit at the same 1-3 ulp noise level. The
reported PASS is the reproducible dump; a different B state could flip a scalar either way. On
vector quantities L1 is consistently at or below fp32-GPU error.

## 5. Answers to P0 open questions
- **Q6, FTZ.** Real subnormals occur in L0-fp32: 17,193 of 657.5 M dropout-module outputs (attention
  probabilities / dropout outputs), 2,902 of 638.7 M gradients at Linear outputs, and 0 in parameter
  gradients (step A; B is similar). The chip flushes them, and the gate above includes that effect.
- **Q13, sqrt.** The chosen algorithm is correctly rounded on all 2.13 G positive inputs (measured),
  so sqrt costs one rsqrt plus 3 ops.
- **Q8, step scalars.** On-chip fp32 lr / beta^t matched torch's double values well enough: post-step
  coefs ratio 1.00 in B.

## 6. Limits
- L1 has CE for any C <= 16 but **no MSE (STS-B) loss yet**. RTE does not need it; P4 must add it to L1 and RTL.
- L1 does not model NaN payloads (canonical NaN only) or timing.
- Masks are generated in numpy (slow but shared with L0 by construction).
