# P9a report: full-size training-step cycles measured on the RTL

DRAFT: numbers marked TBD are pending.

## 1. Question and answer
The paper's full-size step time (12 layers, B 32, L 128) came from docs/calc/budget.py, an analytical model:
3,290.6 M cycles = 18.1 s at 181.8 MHz, "80 % array utilisation". P9a replaces it with a cycle count measured in
cycle-accurate simulation of the chip RTL: current main (hw/top/top_chip incl. the P8f mm skid buffers), Verilator
5.044, tb/top/tb_top.sv with the tb_dram AXI4 DRAM model at the nominal knobs (24-cycle read latency, 8-cycle write
response, 32 B/cycle shared by reads and writes, fair R/W per ruling R9), the compiled program of sw/compile.py with
real roberta-base weights, LoWA tables and the RTE step-B batch.

TBD

## 2. Why one simulation of the whole step is not possible, and what was run instead
One full-size step is ~3.5 G cycles. Verilator ran this design at 5 k cycles/s with 8 threads and 1.2-1.9 k cycles/s
with 1 thread on the shared host (load average 60-170 on 48 cores from other projects during the whole phase).
The first attempt, a single in-situ full-size 1-layer step (~0.3 G cycles), extrapolated to ~17 h and was stopped
after 10 minutes.

The step is instead measured piecewise, in parallel, on the same RTL and the same compiled programs:

1. **The schedule is static and data-independent.** The sequencer issues the compiled descriptor list in order; the
   engines have fixed latencies; the DRAM model has fixed latency and a token-bucket bandwidth with no address or
   data dependence. Measured: steps 1 and 2 of the reduced runs (same program, different tokens, masks, dropout
   draws and parameters) differ by 167 cycles in 10.36 M (1 layer) and by 1,229 in 61.64 M (12 layers, P8 log),
   all of it in a few vector ops (+40 / +91 cycles each). Dependence on data is therefore below 2e-5.
2. **The chip drains between ops.** tb/top/drain.py lists, from an op log, which ops start with every earlier op
   retired. In the compiled program every op does except three (the LOWA_SCAL at the start, a DMA_COPY beside a
   head GEMM, and the second LoWA sampled gradient of each layer beside the GEMM_ACC2 that follows it). A program
   piece that starts at a drain point therefore starts from the same machine state as in situ (engines idle,
   nothing in flight), and its timing from that point on is the in-situ timing.
3. **Isolated chunks.** `run_e2e.py --nop-before j --stop-after k` (new flag, testbench side only; the RTL is
   unchanged) replaces the unit ops before static op j by NOPs (one 64 B NOP per descriptor slot, so the program
   layout and loop bodies are unchanged) and puts END after static op k. The chip then runs only ops j..k, on the
   full-size image. Chunks were cut at drain points, never between the LoWA SG and the GEMM_ACC2 it overlaps.
   - 1-layer program (pre, forward body, head, layer-0 backward, tail): 10 chunks C1..C10.
   - 2-layer program, backward loop body only (the part a 1-layer program does not contain): 7 chunks D1a..D5b.
   - Single ops S6, S10, S31, S39, SB39 (the ops shared by two chunks; DRAM beats, see section 5).
   All at B 32, L 128, one Verilator thread each, 22 simulations in parallel.
4. **Composition** (tb/top/chunks.py). A chunk [j..k] contributes S(k) - S(j) when the next chunk starts at the
   same op k (that op is simulated twice, at the end of one chunk and at the start of the next: equal durations
   confirm the isolation), or R(k) - S(j) + gap when the next chunk starts at k + 1, where the gap (FENCE +
   descriptor fetch, 1-82 cycles) is taken from an in-situ run of the same program. The first chunk counts from the
   sequencer start, the last to the step IRQ. Contributions add up to the step exactly by telescoping.
5. **Depth.** The program is pre + forward body x n + head + backward body x (n - 1) + layer-0 backward + tail. So
   step(12) = step(1) + 11 x (forward body + backward body) + tail growth. The tail (GNORM over the coefficient
   gradients, coefficient AdamW) grows with the number of LoWA modules but not with B or L, so its growth is taken
   from in-situ reduced runs (12-layer tail minus 1-layer tail); that it is independent of B and L is checked by
   comparing the full-size and reduced tail ops one by one (section 4).

## 3. Validation of the method at the reduced shape (B 2, L 32), current RTL
The same chunk set was run at B 2, L 32 and composed the same way, against in-situ runs on the same RTL:

| quantity | in situ | composed from chunks | delta |
|---|---|---|---|
| 1-layer step (V1, `make test-e2e` config, PASS bit-exact both steps) | 10,359,277 | 10,359,261 | -16 |
| forward body (in-situ F1 of the 2-layer run V2) | 2,108,967 | 2,108,967 | 0 |
| backward body (in-situ B1 of V2) | 2,545,138 | 2,545,141 | +3 |
| 2-layer step (V2, PASS bit-exact) = 1-layer + F + B + tail growth 7,174 | 15,020,725 | 15,020,543 | -182 |
| 12-layer step (V12, current RTL, PASS bit-exact vs interpreter and L1) = 1-layer + 11 x (F + B) + tail growth 78,850 | 61,634,485 | 61,633,299 | -1,186 (-0.002 %) |

- Shared ops at chunk boundaries (simulated at the end of one chunk and the start of the next): equal at 4 of 6
  boundaries, +2 cycles at the other 2.
- V12 per segment (in situ): forward bodies F0..F10 2,109,045-2,109,048, F11 (the last, before the head) 2,108,967;
  backward bodies alternate 2,545,136 / 2,545,184; layer-0 backward 2,177,895; tail 5,613,821 (the isolated
  12-layer tail chunk, rT12: 5,613,821, identical). The composition uses the 1-layer program's forward body (which is
  also the last one before the head), so it is 81 cycles short for each of the 10 inner forward bodies; that and the
  backward-body alternation are the -1,186 cycles.
- V12 is also the first 12-layer run of the P8f RTL (the mm skid buffers): ALLWRITES (8,346,904 words) and 24/24 result
  tensors bit-exact vs the interpreter and vs L1, loss 0.785167277, 61,634,485 cycles (P8 run before P8f:
  61,639,415; P8f saves 4,930 cycles = 0.008 %).
