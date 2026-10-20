# P9a report: full-size training-step cycles measured on the RTL

## 1. Result
The paper's full-size step time (12 layers, B 32, L 128) came from docs/calc/budget.py, an analytical model:
3,290.6 M cycles = 18.1 s at 181.8 MHz, "80 % array utilisation". P9a replaces it with a cycle count measured in
cycle-accurate simulation of the chip RTL:

- RTL: current main, hw/top/top_chip including the P8f mm skid buffers. The RTL was not changed in P9a.
- Simulator: Verilator 5.044, tb/top/tb_top.sv.
- DRAM: the tb_dram AXI4 model at its nominal settings (24-cycle read latency, 8-cycle write response, 32 B/cycle shared
  by reads and writes, fair R/W per ruling R9).
- Program and data: the compiled program of sw/compile.py with the real roberta-base weights, the LoWA tables and the
  RTE step-B batch.

| configuration (full16, B 32, L 128 unless noted) | measured on RTL (cycles) | budget.py (cycles) | delta |
|---|---|---|---|
| 1-layer step (complete program: embedding, layer 0 fwd + bwd, head, loss, grad norm, AdamW) | 318,735,804 | 273,418,146 | +16.6 % |
| per added layer (forward body 143,310,459 + backward body 180,468,277) | 323,778,736 | 292,021,119 | +10.9 % |
| **12-layer step, as compiled (no CLS pruning)** | **3,880,380,750** | 3,485,650,455 | **+11.3 %** |
| 12-layer step vs the paper's figure (budget.py with CLS pruning) | 3,880,380,750 | 3,290,582,334 | +17.9 % |
| reduced check: 12-layer step at B 2, L 32, measured in situ in one simulation | 61,634,485 | 55,558,149 | +10.9 % |

- **Step time at the 5.5 ns chip clock (181.8 MHz): 21.34 s** (budget.py: 18.1 s with CLS pruning, 19.17 s without).
  At 7.75 ns, the clock without the P8f fix, it would be 30.07 s.
- **Array utilisation: 0.727.** That is the 722.0 G MACs of the compiled step divided by 256 PEs x the measured
  cycles. The paper says 80 %; budget.py gives 0.800 with pruning and 0.809 without.
- **CLS pruning is not implemented.** The 3,290.6 M figure assumes the last layer computes only the CLS rows. The
  compiler does not do this (P6c deviation 4), so the measured program has no pruning.
  - budget.py puts the pruning saving at 195.1 M cycles. Applying that saving to the measured step gives
    ~3,685 M = ~20.3 s.
  - This is an **estimate, not a measurement**.
- DRAM traffic and bandwidth demand: section 5.

## 2. Method
One full-size step is ~3.9 G cycles. On the shared host (load average 60-170 on 48 cores from other projects the whole
time) Verilator ran this design at:
- ~5 k cycles/s with 8 threads;
- 1.2-1.9 k cycles/s with 1 thread.

A single in-situ full-size 1-layer step (0.32 G cycles) extrapolated to ~17 h, so it was stopped after 10 minutes.
The step was instead measured in pieces that ran in parallel, on the same RTL and the same compiled programs.

1. **The schedule is static and does not depend on data values.**
   - The sequencer issues the compiled descriptor list in order and the engines have fixed latencies.
   - The DRAM model has fixed latency and a token-bucket bandwidth, with no dependence on address or data.
   - Steps 1 and 2 of the reduced runs use the same program with different tokens, masks, dropout draws and
     parameters. They differ by:
     - 167 cycles in 10.36 M (1 layer);
     - 1,229 cycles in 61.64 M (12 layers, P8 log).
   - The whole difference sits in a few vector ops (+40 / +91 cycles each), so the data dependence is below 2e-5.
2. **The chip drains between ops.**
   - tb/top/drain.py reads an op log and lists the ops that start only after every earlier op has retired.
   - Every op of the compiled program does this except three:
     - the LOWA_SCAL at the start;
     - one DMA_COPY beside a head GEMM;
     - the second LoWA sampled gradient of each layer, which runs beside the following GEMM_ACC2.
   - A program piece that starts at such a drain point starts from the in-situ machine state (all engines idle,
     nothing in flight). From then on its timing is the in-situ timing.
3. **Isolated chunks.** A new testbench-side flag, `run_e2e.py --nop-before j`, works with the existing
   `--stop-after k`:
   - The unit ops before static op j become NOPs, one 64 B NOP per descriptor slot, so the program layout and loop
     bodies are unchanged. END goes after op k.
   - The chip then runs only ops j..k, on the full-size DRAM image.
   - Chunks were cut at drain points only, never between the LoWA SG and the GEMM_ACC2 it overlaps.
   - The chunks, all at B 32, L 128 with one Verilator thread each, about 22 simulations at a time:

     | name | what it covers | count |
     |---|---|---|
     | C1..C10 | the 1-layer program (pre, forward body, head, layer-0 backward, tail) | 10 chunks |
     | D1a..D5b | the backward loop body of the 2-layer program, the part a 1-layer program lacks | 7 chunks |
     | S0_2, S6, S10, S31, S39, SB39 | single ops, for the DRAM accounting (section 5) | 6 runs |
4. **Composition** (tb/top/chunks.py). Each chunk [j..k] contributes as follows:
   - If the next chunk starts at the same op k: S(k) - S(j). That op is simulated twice, at the end of one chunk and
     at the start of the next; equal durations confirm the isolation.
   - If the next chunk starts at k + 1: R(k) - S(j) + gap. The gap (FENCE plus descriptor fetch, 1-82 cycles) comes
     from an in-situ run of the same program.
   - The first chunk counts from the sequencer start, and the last chunk counts up to the step IRQ.

   By telescoping, the contributions add up exactly to the step.
5. **Depth.** The program is pre + forward body x n + head + backward body x (n - 1) + layer-0 backward + tail. So:

   step(12) = step(1) + 11 x (forward body + backward body) + tail growth.

   - The tail (GNORM over the coefficient gradients, clip, coefficient AdamW) grows with the number of LoWA modules,
     not with B or L.
   - Its growth, +78,850 cycles for 11 layers, is the in-situ 12-layer tail minus the 1-layer tail at B 2, L 32.
   - The full-size and reduced tail ops are cycle-identical: 5,534,967 cycles both, op by op.
6. **Measured, not modelled.** Every term above is a cycle count from an RTL simulation. The inter-op gaps and the
   tail growth were measured at B 2, L 32, where the ops concerned are the same.

What was composed, exactly:
- step(12) = [C1..C10] + 11 x ([forward body ops 3..13 of C1..C5] + [D1a..D5b]) + [12-layer tail - 1-layer tail].
- The forward body used is layer 0's. It also comes last before the head; at B 2, L 32 that body is 81 cycles shorter
  than the 10 inner bodies (V12: F0..F10 2,109,045-2,109,048, F11 2,108,967). That is the main part of the -1,186
  cycles in section 3.

## 3. Validation of the method
### 3.1 Reduced shape (B 2, L 32), current RTL
The same chunk set was run at B 2, L 32 and composed the same way. It was compared with in-situ runs on the same RTL,
each of which also passes the bit-exact check:

| quantity | in situ | composed from chunks | delta |
|---|---|---|---|
| 1-layer step (V1, the `make test-e2e` configuration; PASS, both steps bit-exact) | 10,359,277 | 10,359,261 | -16 |
| forward body (F1 of the in-situ 2-layer run V2) | 2,108,967 | 2,108,967 | 0 |
| backward body (B1 of V2) | 2,545,138 | 2,545,141 | +3 |
| 2-layer step (V2; PASS) = 1-layer + F + B + tail growth 7,174 | 15,020,725 | 15,020,543 | -182 |
| 12-layer step (V12; PASS vs the interpreter and vs L1) = 1-layer + 11 x (F + B) + tail growth 78,850 | 61,634,485 | 61,633,299 | -1,186 (-0.002 %) |

- **Shared ops at chunk boundaries.** Of 6 boundaries, 4 give equal durations; the other 2 give +2 cycles.
- **V12 per segment (in situ):**
  - forward bodies 2,109,045-2,109,048, and 2,108,967 for the last;
  - backward bodies alternating 2,545,136 / 2,545,184;
  - layer-0 backward 2,177,895;
  - tail 5,613,821; the isolated 12-layer tail chunk gives the same 5,613,821.
- **V12 is also the first 12-layer run of the P8f RTL.**
  - ALLWRITES (8,346,904 words) and 24/24 result tensors are bit-exact vs the interpreter and vs L1; loss 0.785167277.
  - The step takes 61,634,485 cycles. The P8 run before P8f took 61,639,415, so P8f saves 4,930 cycles (0.008 %).

### 3.2 Full size
- **Shared ops.**

  | op | duration at the end of one chunk | duration at the start of the next | difference |
  |---|---|---|---|
  | 6 (QK^T GEMM) | 2,360,720 | 2,360,720 | 0 |
  | 10 (LN_FWD) | 3,602,525 | 3,602,523 | -2 |
  | 31 (E_DROP) | 1,282,923 | 1,282,885 | -38 |
  | 39 (dS K GEMM) | 2,360,720 | 2,360,720 | 0 |
  | 44 (LoWA SG) | 11,445,330 | 11,445,329 | -1 |
  | body 39 | 2,360,720 | 2,360,720 | 0 |

- **Error bound.** 17 chunk starts at -38 to +2 cycles each, plus the boundary effects of section 3.1, come to a few
  thousand cycles in 3.88 G (~1e-6).
- **The same op in different layers** costs the same to within a few cycles. Merge-on-load traffic depends on the
  per-module LoWA tables, but even at B 2, L 32, where the merge is far more exposed (merge queue busy 8.95 M of
  61.6 M cycles), the 11 forward bodies agree to within 3 cycles. At full size the q / v GEMMs carry about +5.7 k
  cycles of merge over the k GEMM (10,043,893 vs 10,038,176). So layer-to-layer variation of the merge is bounded
  well below 0.01 %.
- **Bit-exact check.** Every full-size chunk was checked with `run_e2e.py --check-only`: ALLWRITES plus the 24 result
  tensors vs the interpreter run on the same chunk program. All 23 runs PASS.
  - C1 starts at the real step start, so its check is a real-data full-size bit-exact check of EMB, the X0 copy and
    the Q / K / V / QK^T GEMMs (81 M words).
  - Later chunks start from an image whose skipped producers wrote nothing, so their operands are partly zero. Their
    check shows that the chip and the interpreter agree, not that the full-size numerics are right.

## 4. Where the measured cycles go, and why budget.py is 11 % low
Composed 12-layer step (no pruning):

| segment | cycles | share |
|---|---|---|
| pre (EMB, X0 copy to the stash, LoWA scalars) | 10,754,634 | 0.3 % |
| forward bodies x 12 | 1,719,725,508 | 44.3 % |
| head fwd + loss + head bwd | 2,879,755 | 0.1 % |
| backward bodies x 11 | 1,985,151,047 | 51.2 % |
| layer-0 backward | 156,255,989 | 4.0 % |
| tail (grad norm, clip, AdamW of 6,144 coefficients + 592 k head parameters) | 5,613,817 | 0.1 % |

Per op class: the measured start-to-retire time per op, summed over the composed step, against the per-op terms of
budget.py at the same shape. Overlapped ops are counted in full.

| class | measured | budget.py per-op model | ratio |
|---|---|---|---|
| GEMM | 3,048.70 M | 2,972.96 M | 1.025 |
| vector unit (VEC, LOSS, GNORM) | 672.72 M | 564.47 M | 1.192 |
| LoWA sampled gradient | 275.60 M | 273.09 M | 1.009 |
| AdamW | 5.33 M | 5.33 M | 1.000 |
| DMA_COPY | 4.88 M | 0.79 M (budget.py's step model: 0) | 6.2 |

The +394.7 M cycles over budget.py's step model (3,485.7 M, no pruning) break down as follows:

1. **Vector unit, +166 M** (672.7 M measured vs budget.py's step-level vector term of 506.7 M).
   - The dropout passes E_DROP are 5.14 M cycles per layer (61.7 M per step). budget.py's step model counts them only
     as 0.6 M of "misc".
   - LayerNorm runs 1.26-1.36 x its P4 rate, SMAX_FWD 1.17 x, SMAX_BWD 1.26 x, SMAX_REC 1.08 x and EMB 1.38 x.
   - The P4 rates came from unit testbenches at 4,096 rows. P9a did not analyse the extra cycles further. They are
     not a port-bandwidth limit: LN_FWD on its own uses 45 % of the DRAM port.
2. **LoWA sampled gradient not hidden, +148.8 M.** budget.py assumes the LoWA unit runs entirely beside the array,
   but only the second SG of each layer does: it runs beside the GEMM_ACC2, and 30,356,364 cycles for the pair vs
   30,356,215 for the GEMM alone means it is fully hidden. The first SG (11,445,330 cycles) waits for the LN_FWD
   recompute and runs alone. That makes 11 x 11.45 M plus both SGs of layer 0 (22.89 M).
3. **GEMM, +75.7 M (+2.5 %).** The W-stream GEMMs run at 1.014-1.046 x, and the GELU / GELU' epilogue GEMMs (FFN1,
   FFN2 dX) at 1.03-1.04 x. The attention GEMMs run at 0.99 x.
4. **DMA_COPY, +4.7 M exposed.** Two copies are on the critical path at full size, and budget.py models neither. They
   copy at about one word per cycle, as P7a found.
   - The X0 stash copy at the start: 2.41 M cycles, all exposed.
   - The copy in the head phase: 2.47 M cycles, of which 2.33 M are exposed. At B 2, L 32 it was hidden behind a
     GEMM.

The rest is small: AdamW is exact, and the GNORMs are below the model.

## 5. DRAM traffic and bandwidth demand
TBD (S0_2 pending)

## 6. Proposed paper text
> The full training step (12 layers, batch 32, sequence 128) takes 3.88 G cycles, measured in cycle-accurate
> simulation of the chip RTL (Verilator) with a fixed-latency DRAM model at 32 B/cycle: 21.3 s at the 181.8 MHz
> chip clock, an array utilisation of 73 %. Because one full-step simulation would take days, the step was simulated
> as program chunks that start at points where the chip is idle (the schedule is static and data-independent),
> and composed; the composition reproduces a complete in-situ 12-layer simulation at reduced batch and sequence
> length to 0.002 %.

If the paper keeps the analytical number, it should say so: "18.1 s estimated analytically (with last-layer CLS
pruning, which our compiler does not yet implement); 21.3 s measured on the RTL without it".

## 7. What remains estimated, and caveats
- **CLS pruning** is modelled, not measured: the ~20.3 s figure in section 1.
- **The DRAM is a model**: fixed latency, 32 B/cycle at the core clock, no refresh, banks or page misses.
  - At 181.8 MHz that is 5.82 GB/s.
  - Many ops run at or near that limit (e.g. QK^T: 0.999 beats/cycle), so a real DRAM that delivers less would
    lengthen the step.
  - The cycle count does not depend on the clock only because the model scales with it.
- **The step was composed from chunks, not simulated in one run at full size.** Composition errors are bounded by the
  shared-op checks (0 to -38 cycles per boundary) and the reduced-shape end-to-end check (-0.002 %).
  - The inter-op gaps (1-82 cycles each) and the tail growth (78,850) were measured at B 2, L 32.
  - Using those values at full size assumes the gaps and the tail are independent of B and L. For the tail this was
    checked directly.
- **Data dependence**: below 2e-5 between steps (vector ops only).
- **Full-size numerics** are bit-exact on real data only through the first chunk (EMB .. QK^T). The other chunks agree
  with the interpreter on partly zero operands.
- **Energy per step** rises with the step time: 21.34 s x 0.475 W = 10.1 J vectorless, at most ~12.8 J with the
  0.60 W SAIF-based figure. Both are pre-layout (P8 section 7).

## 8. Reproduction
Branch p9a. Builds: `run_e2e.build(27 | 28, 'build/e2e', threads)`, where 28 is needed for the 2-layer full-size image
(521 MiB).

- **Each chunk.** Use `--memw 28` for 2-layer full-size chunks; --prep-only writes the image, and then
  build/e2e/obj_m27/Vtb_top runs with the run_sim plusargs:

  `python3 tb/top/run_e2e.py --layers {1|2} --batch 32 --seq 128 --steps 1 --nop-before J [--stop-after K] --memw 27 --threads 1 --prep-only`
- **Chunk table.** The (j, k) pairs are in tb/top/chunks.py (L1_CHUNKS, BODY_CHUNKS).
- **Bit-exact check.** Repeat the same arguments with `--check-only`.
- **In-situ references:**

  `run_e2e.py --layers 1 --batch 2 --seq 32 --steps 2` (V1), `--layers 2 ... --steps 1` (V2) and
  `--layers 12 ... --steps 1 --l1 --threads 8` (V12).
- **Composition:**

  `python3 tb/top/p9a_summary.py --insitu1 build/e2e/l1_b2_s32_n2 --insitu2 build/e2e/l2_b2_s32_n1 --tail12 build/e2e/l12_b2_s32_n1`

  and `tb/top/chunks.py --batch 2 --seq 32 ...` for the reduced check.
