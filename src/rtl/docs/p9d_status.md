# P9d status: MRPC under chip-exact emulation

## 2026-10-19 19:50 UTC: step 1 decision
- Paper MRPC number: `scratchpad/s3` runs `mrpc-sgwa-seed42..46` (train_glue.py, roberta-base, q/v, k 256, K 3,
  gamma 0.552427, lr 0.166667 / head 5e-3, wd 0.01, B 32, max length 128, 30 epochs, warmup 206, dense
  autograd gradient, dynamic padding). Accuracy at the best-F1 epoch: 88.97, 87.01, 86.52, 88.48, 87.50 -> median 87.50 (F1 92.12, 90.75, 90.27, 91.59, 91.04 -> median 91.04; best epochs 26, 19, 10, 20, 12).
- L1 (`model/hwmodel`, lwta.c + step.py) takes 14.0 s and 16.1 s for one full-size step (B 32, L 128, all
  48 cores, the stored step A/B dumps). MRPC = 115 steps/epoch x 30 = 3,450 steps = 13.4-15.4 h for ONE
  seed on the whole machine, and the CPU is shared with P9a's RTL simulation. Parallel seeds do not help (L1
  already uses every core). Option (i) is out.
- Decision: option (ii). Port L1 op by op to CUDA (cuda:0): the same scalar primitives with RNE, explicit DAZ/FTZ,
  the same per-element reduction order (chunked GEMM KC 64, sequential softmax / LoWA / CE sums), the same SFU
  algorithms and Threefry masks. Prove bit-exactness against L1 on whole steps (all outputs) before use.
  Fallback if not bit-exact by ~2026-10-20 02:00 UTC: report (iii).

## 2026-10-19 20:00 UTC: GPU port bit-exact
- `model/gpuemu/` (lwta_gpu.cu, Python entry points, step_gpu.py): L1 ported op by op to CUDA (cuda:0).
- `gpuemu/check_gpu.py`: add/mul/fma on the 311,565 test_arith cases (incl. the 2^-126 FTZ boundary and
  subnormal inputs), every SFU fn and composite on 10.2 M inputs, GEMM in every layout, LN, softmax, LoWA,
  CE, GNORM, AdamW, Threefry masks: 0 mismatches.
- `gpuemu/check_step.py dumps`: full-size steps A (s 0) and B (s 140), B 32 x L 128, all 2,393,165 output
  values of each step (loss, logits, all gradients, norm, clip, lr, post params / m / v): 0 mismatches.
  GPU 1.5-1.7 s/step (the GPU is shared with another job at ~97 % util) vs L1 14-16 s.
- Next: MRPC driver (gpuemu/run_mrpc.py) smoke test, then seeds 42-46 in parallel on cuda:0.

## 2026-10-19 20:25 UTC: MRPC runs going
- Seeds 42-46 started 20:07 UTC, 5 processes on cuda:0, ~3.2 s/step each -> ETA ~23:30 UTC (30 epochs).
- In-run checkpoints vs L1 (check_step.py ckpt): seed 42 s 1, 113, 114 (short batch, 12 dummy rows) bit-exact
  (all 2,393,165 values; both a fresh GPU step and the step the run itself took).
- GPU vs RTL directly (check_rtl.py on make test-e2e-full's dumps, 12 layers, B 2 x L 32, 2 chained steps):
  24/24 result tensors bit-exact, losses 0.785167277 / 0.230305105.
- Epoch 0 dev: chip-forward eval and fp32 PyTorch eval give identical predictions on all 408 pairs.

## 2026-10-19 23:40 UTC: DONE
- All 5 seeds finished 30 epochs (23:13 UTC). Chip-exact median acc 87.75 / F1 91.23 (paper GPU 87.50 / 91.04).
- 18 in-run MRPC steps checked against L1 (incl. s 1000, 2000, 3449): all bit-exact. Report: docs/p9d_report.md.
