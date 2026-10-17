# LoWA-TA RTL conventions (P2; binding for P3-P8)

**Language.** Verilog-2005 plus `always @*` and `generate`. Every file under `hw/` must be accepted by
Yosys 0.62, Verilator 5.044 (`--lint-only -Wall`, clean or waived in the source with a one-line
reason) and Icarus 13 (`-g2005`). In `hw/` there are no `real`s, no `$` system functions (also not
`$signed`, `$clog2`; widths are computed by hand or by plain constant expressions), no `initial` blocks,
no `#` delays and no latches. Testbenches (`tb/`) may use SystemVerilog constructs that both
Verilator and Icarus accept, and system tasks (`$fscanf`, `$display`, `$finish`).
Exception (PLAN R11a): the tsmc65 memory backend `hw/mem/tech/mm_bank_t65.v` (latch register file) and its
clock gate `hw/mem/tech/mm_icg.v` contain latches by design; `flow/lint.sh` requires latches there and
forbids them everywhere else.

**Files and names.**
- One module per file, file name = module name (`hw/fp/fp_fma.v` holds `module fp_fma`).
- The module prefix is the directory: `fp_` (hw/fp), `gm_` (gemm), `vu_` (vec), `lw_` (lowa), `rn_` (rng),
  `mm_` (mem), `ct_` (ctrl), `top_` (top), `cm_` (common).
- Files start with `` `default_nettype none `` and end with `` `default_nettype wire ``.
- Parameters are `UPPER_CASE`, ports and signals `lower_case`. Active-low signals end in `_n`.
  Register outputs of a pipeline stage are named `<name>_s<k>` (value held after stage k).
- Generated RTL (e.g. microcode ROMs) carries a header naming its generator; `make gen-check`
  fails if it is stale.

**Clock and reset.** One clock `clk`, rising edge. One reset `rst_n`: **synchronous, active low**.
Only control state is reset (valid bits, counters, FSM state, busy flags). Datapath registers are
not reset (area); they are never read while their valid bit is 0.

**Pipeline protocol.**
- Arithmetic units are **fixed-latency, valid-only** pipelines: `in_valid` qualifies the inputs in
  a cycle, `out_valid` qualifies the outputs exactly `LATENCY` cycles later. There is no stall and
  no backpressure. A unit that cannot take a new input every cycle exposes `in_ready`
  (combinational, valid/ready in the AXI sense on the input only); its output is still valid-only.
- Every unit has a parameter `STAGES` (= its latency in cycles, >= 1). The unit has a fixed list of
  internal cut points; `STAGES` enables them in a documented priority order and the last register
  is always at the output. `STAGES` larger than the number of cut points appends plain delay
  registers at the output (used to equalise latencies). The module header states the cut points
  and the stage count that closes 40 ns at ss_125C_3v00 (`docs/p2_report.md`).
- Outputs are registered. Inputs are assumed to come from flops of the producer (STA is run with
  input flops in a synthesis wrapper, `flow/synth/wrap_*.v`).

**Directory mirroring.** `tb/<dir>/tb_<module>.sv` tests `hw/<dir>/<module>.v`. Test vectors are
generated into `build/` (git-ignored) by `tb/<dir>/gen_*.py`, which take expected values only from
the L1 entry points (`model/hwmodel`, `import hwmodel as H`). A testbench prints exactly one final
line `PASS <unit> <n> vectors` or `FAIL <unit> <m>/<n> mismatches`; the Makefile greps for it.

**Memory backends.** `mm_sram_bank` is the only on-chip memory primitive. Its technology backend
(`hw/mem/tech/`: generic / gf180 / tsmc65) is chosen by the `MM_TECH` macro behind one port contract;
units never instantiate a backend or a library cell directly. Library cell names live only in
git-ignored site files (`flow/tsmc65/site/`).

**Build outputs** go under `build/` only (simulator objects, vectors, logs, netlists, reports).
