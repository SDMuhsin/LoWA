# P9a status (measured full-size step cycles)

## 2026-10-19 19:50 UTC
- Branch p9a. Builds: obj_m27_t8 done (MEMW 27, 8 threads); obj_m28_t8 (for the 2-layer full-size image, 521 MiB) building.
- Running: full-size 1-layer step (B 32, L 128), `run_e2e.py --layers 1 --batch 32 --seq 128 --steps 1 --threads 8`,
  bit-exact check vs the interpreter kept (interpreter takes 12 s at this shape).
- Added: `tb/top/seg.py` (per-segment split from the op log), `run_e2e.py --nop-before` (unit ops before a static
  index become NOPs, for isolated segment timing).
- Seen on the P8 12-layer reduced run (B 2, L 32): forward bodies 2,109,264 cycles each (F0 +5, F11 -81),
  backward bodies 2,545,364 / 2,545,416 alternating; step 1 vs 2 differ by 1,229 cycles (vu only).
- budget.py full size: 3,290.6 M CLS-pruned (paper), 3,485.65 M without pruning (as compiled).
- ETA: depends on simulation speed under the shared host load (load average ~60 on 48 cores).

## 2026-10-19 20:35 UTC
- The in-situ full-size 1-layer run ran at ~5 k cycles/s with 8 threads under a host load of 60-170 (other projects);
  ~300 M cycles would need ~17 h, past the freeze. Stopped it.
- Switched to isolated program chunks (run_e2e.py --nop-before / --stop-after) at drain points of the schedule (every
  op starts with all engines idle, except LoWA SG beside GEMM_ACC2 and two small DMA copies; tb/top/drain.py on the
  P8 12-layer log). 10 chunks of the 1-layer program + 7 chunks of the 2-layer program's backward body, all at B 32,
  L 128, single-thread Verilator, in parallel; the same chunks at B 2, L 32 plus in-situ B 2, L 32 runs (1 and 2
  layers) to validate the composition. Composer: tb/top/chunks.py.
- Speed per single-thread sim under the current load: ~1.2-1.5 k cycles/s; the longest chunk is ~42 M cycles, so ETA
  for the full-size numbers ~06:00-08:00 UTC (freeze 12:00).

## 2026-10-20 00:45 UTC
- V1 (in-situ 1-layer reduced, 2 steps, current RTL): E2E PASS; step 1 10,359,277 vs composed 10,359,261 (-16).
- Full-size chunks done: C5, C10, D1a, D4, D5a. Full-size tail ops (GNORM x 7, DMA, AdamW x 2) are cycle-identical
  to the reduced ones (5,534,967 both): the tail does not depend on B / L, as assumed.
- The 8-thread 12-layer reduced in-situ run (V12) slowed to ~0.6 k cycles/s under the host load; added an isolated
  single-thread 12-layer tail chunk (rT12) for the tail-growth term.
- Remaining full-size chunks: ETA 02:00-06:00 UTC.

## 2026-10-19 22:35 UTC
- In-situ 2-layer reduced run on current RTL (B 2, L 32): E2E PASS (ALLWRITES + 24/24 bit-exact), 15,020,725 cycles.
  Its forward body F1 = 2,108,967 (composed from the 1-layer chunks: 2,108,967, equal); backward body B1 = 2,545,138
  (composed from the isolated body chunks: 2,545,141, +3). Tail grows by 7,174 cycles per layer.
  1-layer composition + F + B + tail growth = 15,020,543 vs 15,020,725 measured (-182, -0.001 %).
- Full-size chunks: 2.5-8 h to go.

## 2026-10-19 21:35 UTC
- Reduced (B 2, L 32) chunks all done. Composed 1-layer step 10,359,261 cycles vs 10,359,455 in situ (pre-P8f log):
  -194 (-0.002 %, the size of the P8f change). Shared ops simulated at the end of one chunk and the start of the next:
  equal or +2 cycles (5 boundaries).
- Found: the tail (GNORM over the coefficient gradients, coefficient AdamW) grows with the number of LoWA modules
  (+~7.2 k cycles per layer at any shape); it is not in the 1-layer + 11 x body formula. Taken from the in-situ
  12-layer reduced run (running, current RTL) minus the 1-layer one; shape independence checked by comparing the
  full-size and reduced tail chunks op by op.
- Full-size chunks: 3-9 h to go (ETA 01:00-06:30 UTC).
