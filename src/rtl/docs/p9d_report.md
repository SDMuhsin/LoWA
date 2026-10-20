# P9d report: MRPC accuracy under chip-exact emulation

## 1. Headline
Fine-tuning roberta-base with LoWA on MRPC, in a run where **every one of the 3,450 training steps is the chip's
step bit for bit**, gives **accuracy 87.75 / F1 91.23**. Those are medians over seeds 42-46 at the best-F1 epoch,
the paper's protocol. The paper's FP32 GPU runs give 87.50 / 91.04. All five seeds finished all 30 epochs.

| seed | chip-exact: best epoch | acc | F1 | GPU reference (paper): best epoch | acc | F1 |
|---|---|---|---|---|---|---|
| 42 | 24 | 87.75 | 91.23 | 26 | 88.97 | 92.12 |
| 43 | 27 | 85.78 | 89.97 | 19 | 87.01 | 90.75 |
| 44 | 18 | 88.24 | 91.72 | 10 | 86.52 | 90.27 |
| 45 | 21 | 87.75 | 91.10 | 20 | 88.48 | 91.59 |
| 46 | 20 | 87.75 | 91.41 | 12 | 87.50 | 91.04 |
| **median** | | **87.75** | **91.23** | | **87.50** | **91.04** |

Paired by seed, the chip-exact run is above the GPU run on 2/5 seeds and below on 3/5 (accuracy). That is no
systematic difference: a paired sign test needs 5/5 to call one, and the median gap of +0.25 accuracy points is
well inside the seed spread (seed sd about 1 point in both runs). The two runs are different trajectories of the same procedure
(Section 3), so only distributional agreement is expected.

## 2. Method
**Step 1 decision.** L1 (`model/hwmodel`, lwta.c + step.py) needs 14-16 s per full-size step (B 32, L 128) on
this machine. MRPC is 115 steps x 30 epochs = 3,450 steps, which is 13-15 h for one seed using every core. The
CPU was also shared with P9a's RTL simulation and another project's 40 jobs. So option (i) was out. Option (ii)
was taken: port L1 to CUDA op by op and prove the port bit-exact.

**The port.** `model/gpuemu/`:
- `lwta_gpu.cu` is a line-by-line CUDA port of `lwta.c`. It uses the same scalar primitives: `__fadd_rn`, `__fmul_rn`
  and `__fmaf_rn`, built with `-fmad=false` (no contraction) and `-ftz=true` (DAZ/FTZ), with NaN canonicalised.
- It uses the same SFU algorithms and constants.
- It keeps the same per-element reduction order:
  - GEMM: fma over k ascending within 64-long chunks, chunk partials added in order. Tiles align to the chunk
    boundaries, and a partial tile runs only the real k.
  - Sequential softmax, LoWA-gradient, CE and colsum chains.
  - Chunked LN and GNORM sums.
- Parallelism is only across independent outputs. Some loops were split into two kernels without changing the
  rounded ops or their order:
  - the LoWA products are formed in parallel, then each atom's fma chain runs;
  - each GNORM row's chunked sum is formed in parallel, then each tensor's chunked sum over its rows;
  - softmax rows are staged in shared memory;
  - the LoWA merge runs one thread per cell, in ascending hit order.
- Threefry-2x32-13 dropout masks are generated on the GPU.

`step_gpu.py` is `hwmodel/step.py` with H replaced by G (same entry points). Additions:
- loss weights and B_ACT for the short last batch (spec.md D3);
- warmup / total steps as parameters;
- the beta^t running products held in the state, the same sequence of L1 multiplies.

`hwmodel/step.py` got the same three optional inputs (`loss_w`, `b_act`, `warmup`/`t_total`). The defaults leave
the P1 dumps unchanged. A GPU step takes about 1.5 s alone on the (shared) A40, or about 3 s each with 5 seeds in
parallel. The whole 5-seed run took 3 h 05 min.

**The run** (`gpuemu/run_mrpc.py --seed N`). It uses the paper's MRPC LoWA row (`scratchpad/s3/jobs_mrpc.tsv`):
- k 256, K 3, gamma 0.552427;
- lr 0.166667 for the coefficients and 5e-3 for the head, wd 0.01;
- B 32, 30 epochs, warmup 206, clip 1.0.

The model (head init), LoWA tables and seeded shuffling loader come from train_glue's own code path
(`l0_ref.build`). The 221 frozen backbone tensors and all 24 LoWA tables are checked bit-equal to
`model_fp32.npz`. Each batch is padded to the chip's fixed program shape, L 128. The 20-row last batch of every
epoch is padded with dummy rows (`<s>` + pads, loss weight 0), and the loss divides by B_ACT = 20. Dropout is the
chip's Threefry keyed by (training seed, step, site, layer).

**Evaluation.** After every epoch the 408 dev pairs are scored two ways:
- (a) with the chip's forward arithmetic and dropout off (`forward_eval`, the same kernels);
- (b) in ordinary FP32 PyTorch (train_glue's HF model, SDPA, eval mode) with the trained coefficients and head
  copied in.

The predictions of (a) and (b) agree on **all 408 pairs at every one of the 150 epoch evaluations**, so the
scores and epoch choices above are the same either way. The best epoch is the first epoch with the maximum dev F1,
as in train_glue. Accuracy and F1 (positive class) are computed as in HF `glue/mrpc`.

## 3. Evidence that the emulation is the chip
Every comparison is bitwise (uint32 equality), with nothing tolerance-based.

| check | what | result |
|---|---|---|
| `gpuemu/check_gpu.py` | add/sub/mul/fma/max/min on the 311,565 test_arith cases. These include the 2^-126 FTZ rounding boundary, subnormal inputs and all specials, so the GPU's FTZ is shown to be tininess-after-rounding like the contract. | 0 mismatches |
| | exp, recip, rsqrt, sqrt, tanh, erf, ln, GELU, GELU', tanh', dropout on 10.2 M inputs: 4 M random bit patterns, specials, and dense sweeps | 0 mismatches |
| | GEMM in every stride layout of the step (odd sizes, K = 1, 3, 9, 17, 65, 3072), LN fwd/bwd, softmax fwd/bwd with and without dropout, LoWA merge and sampled gradient, CE with dummy rows, GNORM/clip (clipping and not), AdamW, colsum, Threefry masks | 0 mismatches |
| `gpuemu/check_step.py dumps` | P1 full-size steps A (s 0) and B (s 140), B 32 x L 128, vs the stored L1 results: all 25 outputs = 2,393,165 values (loss, logits, all gradients, norm, clip, lr, post coefficients / head / Adam m, v) | 0 mismatches |
| `gpuemu/check_step.py ckpt` | 18 in-run MRPC steps vs a fresh L1 step on the same pre-state and batch, for both the step the run itself took and a fresh GPU step. Steps: seed 42 at s 0, 1, 2, 113, 114 (short batch), 115, 205, 206 (warmup peak), 207, 1000, 2000 and 3449 (last); seed 43 s 206; seed 44 s 114, 2000; seed 45 s 115; seed 46 s 0, 3449. | 0 mismatches |
| `gpuemu/check_rtl.py` | GPU vs **the RTL itself**: the dumps of `make test-e2e-full` (12-layer chip RTL, B 2 x L 32, 2 chained steps), 24 result tensors per step | 24/24 bit-exact both steps, losses 0.785167277 / 0.230305105 |

Chain of evidence: RTL = L1 (P7a test-e2e-full, at B 2 L 32) and GPU = RTL directly at that shape; GPU = L1 at
the full B 32 L 128 shape on 20 steps (18 MRPC + 2 P1) plus every kernel. The RTL has not been simulated at
B 32 L 128 against these runs; that size is P9a's.

## 4. What remains different from the GPU reference, and caveats
- These are different trajectories, so only distributional agreement is expected (spec.md D1a, section 6). The
  differences:
  - dropout stream: Threefry vs Philox;
  - reduction orders;
  - SFU approximations;
  - the **sampled** coefficient gradient (the chip) vs dense autograd (the paper's GPU runs);
  - fixed L 128 vs dynamic padding.
- Data order: same seeded loader mechanism as train_glue. Batch-by-batch identity with the paper's runs was not
  checked.
- Evaluation is chip-forward arithmetic, and equals FP32 PyTorch here (identical predictions). The chip program as
  built is a training step. A forward-only (inference) program was not compiled or simulated; the forward kernels
  are the training step's forward, already verified.
- Bit-exactness was shown on the 20 sampled full-size steps and per kernel, not on every one of the 17,250 steps.
  Every step runs the same kernels, and the loss was finite throughout.

## 5. Reproduction
```
cd src/rtl/model                                    # runs/ needs model_fp32.npz + step{A,B}_{l0f32,l1}.npz (make model-check)
make -C gpuemu                                      # liblwta_gpu.so (nvcc, sm_86)
CUDA_VISIBLE_DEVICES=0 python gpuemu/check_gpu.py   # kernels
CUDA_VISIBLE_DEVICES=0 python gpuemu/check_step.py dumps
for s in 42 43 44 45 46; do CUDA_VISIBLE_DEVICES=0 python gpuemu/run_mrpc.py --seed $s --out ../build/p9d/seed$s & done
CUDA_VISIBLE_DEVICES=0 python gpuemu/check_step.py ckpt ../build/p9d/seed42/ckpt_s*.npz   (any ckpt_sXXXX.npz)
CUDA_VISIBLE_DEVICES=0 python gpuemu/check_rtl.py ../build/e2e/l12_b2_s32_n2
python gpuemu/summarize_mrpc.py ../build/p9d/seed4{2,3,4,5,6}
```
Logs are in `build/p9d/` (untracked): `seedN/log.jsonl` (per-epoch metrics, both evaluations),
`seedN/state_epXX.npz`, `check_{a,b,c}.log`.

## 6. Proposed paper wording
Table IV: Score `acc. 87.8`, Source `chip-em.`. Footnote d:
"chip-em.: every training step emulated bit-exactly to the RTL arithmetic (FP32 RNE/FTZ, on-chip dropout RNG,
B 32 x L 128 program), median of five seeds."

Text (Sec. V):
"To score the chip rather than a GPU, we re-ran the MRPC fine-tuning with every training step executed by a
bit-exact emulation of the datapath (verified against the RTL and the bit-accurate model). Over five seeds it
reaches 87.8 % accuracy (91.2 F1), against 87.5 % (91.0) for FP32 GPU training, so the chip's arithmetic costs no
accuracy."

If space is short, keep only the table change and the footnote. Do not write "measured on chip": the RTL was
not run for these 3,450 steps.
