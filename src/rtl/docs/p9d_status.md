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
