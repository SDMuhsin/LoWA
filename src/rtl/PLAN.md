# LoWA-TA: an on-chip training accelerator for LoWA on RoBERTa (PI plan)

Owner: PI (plans, reviews, gates). Implementation proceeds one phase at a time.
A phase that cannot finish in one sitting writes docs/pN_handoff.md and stops; the next sitting
continues from it. Phases are sized to fit; if one cannot, it is split up front.

## Fixed decisions (not changed without PI sign-off)
1. **Workload.** One full LoWA training step of `roberta-base` for GLUE sequence classification, as in
   `src/train_glue.py` + `src/sgwa_adapter.py`: 12 layers, d=768, 12 heads, FFN 3072, seq <= 128, batch 32,
   LoWA on `query` and `value` (24 modules, k=256 centres, K=3 diamond, s=25 taps, seed 777), classifier
   head (dense 768x768 + tanh + out_proj) trainable, everything else frozen. Forward, loss, backward and the
   optimizer update (AdamW, LR schedule) all run on chip.
2. **Host traffic is minimal.** Once: frozen weights, centre tables, program. Per step: token ids, mask,
   labels in; nothing required out (loss optional). End of training: coefficients + head out.
   Everything else (embedding gather, activations stash, gradients, optimizer state) lives in device memory.
3. **Memory.** Frozen backbone (~500 MB fp32) cannot be on chip: device-attached DRAM behind an AXI4 master
   (DRAM model in simulation). On-chip SRAM = banks composed from the GF180 `sram256x8` / `sram64x8` macros.
   LoWA's merge happens **on the weight-load path** (W0 tile + scatter of the 25-tap atoms that hit the tile);
   merged weights are never written back to DRAM.
4. **LoWA backward = sampled gradient.** grad c_j = gamma * sum_t H_t * (dY^T X)[p_jt]: only k*s dot products of
   length b per module, never the dense m*n gradient.
5. **Arithmetic.** IEEE-754 binary32 throughout (the paper's numbers are fp32), round-to-nearest-even,
   subnormals flushed to zero (documented). Transcendentals (exp, tanh, erf/GELU, rsqrt, recip) are hardware
   algorithms with measured ulp error.
6. **RTL.** Synthesizable Verilog-2005 / the SystemVerilog subset accepted by Yosys 0.62, Verilator 5.044
   and Icarus 13 (all three must accept every file). No `real`, no `$` functions, no initial blocks in
   synthesizable RTL. Parameterized array size.
7. **Flow.** GF180MCU-D via LibreLane 3.0.14 as in `/workspace/GanAccelAsic` (read-only; copy what is needed
   into `src/rtl/flow`). Shuttle block limits (1675x1110 um, 11 pads) cannot host a DRAM-attached design, so
   PnR is run on the core (macro-backed) at a stated PE count, and full-size area/power is a stated projection.

## Verification ladder (the only accepted evidence)
- **L0 reference:** PyTorch fp32, the actual `SGWALinear` model, one real step on real GLUE data (cuda:0 only).
- **L1 hwmodel:** numpy/C model of the hardware (same op order, same rounding, same FTZ, same transcendental
  algorithms). Checked against L0 for loss, every coefficient gradient and the updated coefficients
  (relative tolerance stated up front and justified).
- **L2 RTL:** bit-exact against L1, per unit (random + directed + corner cases) and end to end.
- Testbenches never compare RTL against itself or against values printed by the RTL. A test that is skipped
  or shrunk is reported as such.

## Layout
```
src/rtl/
  PLAN.md  docs/        spec, ISA, memory map, per-phase reports
  hw/      common/ fp/ gemm/ vec/ lowa/ mem/ ctrl/ top/      synthesizable RTL only
  tb/      mirrors hw/ (unit tbs) + e2e/
  model/   L0 bridge + L1 hwmodel (python)
  sw/      compiler: model config -> program + memory image
  flow/    lint, synth, PnR configs and scripts
  Makefile one entry point: make lint | test-unit | test-e2e | synth | pnr
```

## Phases (each ends at a PI gate)
- P0 architecture spec + ISA + memory map + dataflow schedule with cycle/byte estimates (docs only)
- P1 L0 bridge + L1 hwmodel, L1 vs L0 on a real step
- P2 fp32 primitives (add, mul, FMA, exp, recip, rsqrt, tanh/erf) + tests vs L1
- P3 GEMM engine (systolic array, tile buffers, transposed modes for backward)
- P4 vector/SFU unit (softmax, LayerNorm, GELU, tanh, CE loss, and their backwards)
- P5 LoWA unit (merge-on-load, sampled gradient, AdamW + schedule for coefs and head)
- P6 memory system (AXI4 DMA, DRAM model, SRAM banks), controller/sequencer, compiler
- P7 top integration + e2e RTL vs L1 (reduced config first, then roberta-base layer and full step via Verilator)
- P8 lint/synth/PnR on GF180, timing/area/power report + full-size projection

## PI rulings on P0 open questions (2026-10-14)
- R1 Coefficient gradient is summed per atom (products accumulated in the dataflow.md order). This is the
  hardware contract; L1 mirrors it. Mathematically identical to fixed decision 4.
- R2 Dropout ON via the on-chip counter-based RNG, regenerated in backward. L0 is run with the chip's masks
  injected so L1-vs-L0 is a like-for-like check; dropout-off mode is kept for debug only.
- R3 The chip boundary is an AXI4 master port. The DRAM controller/PHY and the host load path (a host DMA
  writing the same DRAM) are off-die and modelled in simulation; this is stated, not hidden.
- R4 Full-size projection = 16x16 (also report 8x8). Array size stays a parameter.
- R5 Physical: place-and-route the core as a standalone hard block (no pad ring) at a reduced, stated PE
  count chosen in P8; no shuttle fitting.
- R6 Reference HPs = [S.3] full-search row: gamma 0.552427, lr 0.166667, classifier_lr 5e-3, wd 0.01,
  `--sgwa_sampled_grad 1`.
- R7 (2026-10-15) Dropout RNG: no phase built the on-chip counter-based (Threefry) mask generator; P3-P5
  consume a keep-bit stream (mk channel). P6a may source it from DRAM as a test stand-in only (parameter-
  selected, reported, its traffic excluded from efficiency numbers). The generator is assigned to P6b and
  must be bit-exact vs model/rng.py.
- R8 (2026-10-15) DRAM-aware budget: with measured DMA efficiency the full16 step is 188.9 s (was 129.4 s
  compute-only). Follow-up assigned after P6b: the vector unit's loader/writer issues row-interleaved 1-beat
  requests (lane = row), which the memory system cannot coalesce; restructure vu request order into
  per-row multi-beat bursts (no numerics change) and re-measure.
- R9 (2026-10-15) DRAM model R/W policy: fair read/write sharing (`+wfair=1`) is the nominal setting, as real
  controllers drain writes by watermark; strict read priority stays as a sensitivity row. Budget: full16
  step 135.0 s nominal (189.0 s under read priority).
- Phase split: P6b = on-chip dropout mask generator (R7) + vector-unit burst order (R8);
  P6c = sequencer/ISA decoder + DMA_COPY + compiler (model config -> program + memory image).
- R10 (2026-10-16) Target process: the design will be ported to TSMC 65 nm with CMC tooling. GF180 place-and-
  route is dropped; existing GF180 synthesis/STA numbers remain as an open-source sanity check. After P7, work
  pauses for the user's instruction; P8 is to be redefined as a TSMC-65 porting phase (SRAM-compiler
  abstraction, buffer/clock design-space pass, Genus/Innovus scripts + SDC, handoff package).
- R11 (2026-10-17) P8 = TSMC 65 nm port through synthesis (DC topographical + Formality + pre-layout PT/PP on
  the CMC server; template: GanAccelAsic src_int8_tsmc65). Decisions:
  (a) Memory: no SRAM compiler is usable, so `mm_sram_bank` gets selectable technology backends behind its
      unchanged port contract (same read latency, same write/byte-enable semantics): `gf180` (current macro),
      `generic` (behavioural, the simulation default) and `tsmc65` (latch register file with clock-gated write,
      as in the template's latch RF). Capacity stays at full16 geometry (89 KiB); nothing is shrunk.
      Library cell names live only in git-ignored site files.
  (b) Array: full16 (16x16 PEs, V=16) is the synthesised configuration (R4); 8x8 is a reported row only.
  (c) Clock: provisional target 200 MHz (5.0 ns) at ss 0.9 V 125 C. P8a measures block-level slack at
      5.0 / 4.0 / 6.67 ns and the PI then fixes the target. RTL pipeline changes need PI sign-off and must
      stay bit-exact (L2 regressions re-run).
  (d) Synthesis is hierarchical (each unique block compiled once, then the top), sized for a 16-thread /
      30 GB shared server, one big job at a time.
  Split: P8a memory backends + flow/tsmc65 scaffold + block-level clock scout; P8b all units at the fixed
  clock; P8c top-level full16 synthesis + Formality + pre-layout PT/PP + report.
- R12 (2026-10-17) P8a gate passed (docs/p8a_report.md). Clock target fixed at **4.0 ns (250 MHz) at ss 0.9 V
  125 C**; limiters are FMA stage 1 and the GEMM feeder bank-read path. If full16 top-level synthesis cannot
  close 4.0 ns without RTL changes, fall back to 5.0 ns and report it; no pipeline change without PI sign-off.
  Skipped on the tsmc65 backend (simulation 60-75x slower): real_ffn1_256 and one real_ln_fwd sensitivity row.
- R13 (2026-10-17) P8b gate (docs/p8b_report.md): every unit except gm16 (-2.14 ns), vu16 (-1.08 ns) and mm
  (-0.08 ns) meets 4.0 ns at ss. PI signs off the proposed RTL timing repairs: gm feeder write-request
  register (+ bank read-address stage if needed), vu broadcast register stage + registered lane outputs
  (incl. sfu_out_cnt), mm seed register + decoupled rd_req_ready. Conditions: numerics stay bit-exact
  (lint, test-unit, test-e2e pass unchanged vs L1); cycle-count deltas are measured and reported; any
  latency the compiler or sequencer assumes is updated, not worked around; no change to the clock or to
  constraints. Block abstractions as the parent-link method are accepted. Phases renamed: P8c = timing
  repair + rerun of gm16/vu16/mm (+ reconcile on-chip capacity: RTL 146 KiB vs memory_map.md 89 KiB);
  P8d = top-level full16 synthesis + Formality + pre-layout PT/PP + final report.
