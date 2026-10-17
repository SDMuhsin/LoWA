# P5 report: LoWA unit (hw/lowa), L2 bit-exact vs L1

Results (final RTL): `make test-lowa` PASS (every case bit-exact vs L1: merge-on-load on synthetic and real
layer-0 Wq tiles and inside the GEMM engine, the sampled gradient of a real layer, LOWA_SCAL, AdamW on the real
step-B coefficients and head; Verilator + Icarus smoke, section 3), `make synth-lowa` (section 5), `make lint` PASS
(40 modules), `make model-check` PASS (step B, L1 vs L0; the only L1 addition is the `step_scalars` entry point).

## 1. What exists
| file | role |
|---|---|
| `sw/lowa_tables.py` | per-module tables from the L1 dump (`lowa.<i>.eflat`, `hval`): hit list 6,400 x 32 b sorted by (64 x 64 W block, entry e = 25 j + tap), block directory 256 x 32 b (start[15:0], count[23:16]), centres 256 x 32 b (ci << 16 \| cj) |
| `hw/lowa/lw_fifo.v`, `lw_unpack.v`, `lw_rdseq.v` | flop FIFO (one-hot pointers, cm_ohmux read); 256-bit beat -> word unpacker; sequential word reader over one rd channel (requests <= 64 words, 8 outstanding) |
| `hw/lowa/lw_merge.v` | merge-on-load through gm_wfeed's `mg_*` RMW port (section 2.1) |
| `hw/lowa/lw_sg.v` | sampled coefficient gradient, NL = 4 lanes x 2 chain slots (section 2.2) |
| `hw/lowa/lw_scal.v` | LOWA_SCAL: the step-scalar micro-sequence (fp_i2f, fp_mul, fp_add, fp_cmp, fp_sfu), CONST + OPT_SCAL blocks |
| `hw/lowa/lw_adamw.v` | streaming AdamW (p, m on rd0; g, v on rd1), optional in-engine clip, 3 output beats per 8 elements (section 2.3) |
| `hw/lowa/lw_top.v` | the unit: main queue (d_op 0 SG, 1 SCAL, 2 ADAMW on rd0 / rd1 / wr), merge queue (`m_*`, rd2 + `mg_*`), one cm_rstbuf per engine |
| `model/hwmodel/__init__.py` `step_scalars` | step.py's scalar chain on the existing primitives (lr, bias corrections, decay) as an L1 entry point |
| `tb/lowa/gen_lowa.py`, `tb_lw_top.sv`, `run_tests.py` | cases, memory model (3 rd + 1 wr, latency / stalls), W-buffer model for the merge-only mode, `make test-lowa` |
| `tb/lowa/tb_lw_gemm.sv` | gm_top + lw_top (merge queue) on one memory model: the merge exercised through the engine's own W path (section 3) |
| `flow/synth/wrap_lw_top.v`, `synth_lowa.sh`, `run_lowa.py`, `flow/sta/lowa_blocks.tcl` | synthesis / STA with the four engines kept as hierarchy, per-block slack after repair (`make synth-lowa`) |

## 2. Architecture
**Storage per module (DRAM, bytes):** hits 6,400 x 4 = 25,600, directory 256 x 4 = 1,024, centres 256 x 4 = 1,024,
coefficients 1,024, m / v state 2 x 1,024; the h taps (25 words) are shared.  memory_map.md counted 25,600 + 1,024:
the block directory replaces the per-block run-length headers (+1 KiB per module, 24 KiB total).
**On chip:** lw_merge 2 x 512-word banks (coefficients and directory, two module slots = 2 x 8 sram256x8 macros),
h in 25 registers, a 64-entry hit FIFO; lw_sg 2 x 768-word row banks + a 256-word centre bank + a 256-word
accumulator bank (2 x 12 + 2 x 4 macros), h in 25 registers, 8 x (7 + 7) window words in flops; lw_adamw ~14 k
flops of stream / stage FIFOs (2 x 256 b per stream, 16 / 16 / 32 / 16 / 32 entries between the segments).

**2.1 Merge-on-load (lw_merge).** The engine replays the loader's W tile order (nb, mb, seg, kc) so it knows the
64 x 64 W block of every `w_tile_done`: ROW loads use block (nb, kc), TRN loads (kc, nb) of the same list.  Per
tile it looks the block up in the directory, fetches its hits (one tile ahead, into the FIFO), and from
`w_tile_done` until the last write holds `merge_busy[buf]`, which gates the block start in gm_ctrl.  Per hit:
`v = mul(gamma, c_j)`, `val = mul(v, h_tap)`, mg read, `fp_add`, mg write, in hit-list order (= ascending e, the
contract for colliding cells); hits issue every other cycle, a hit on a cell still in flight waits.  Two module
slots (GEMM_ACC2: Wq in segment 0, Wv in segment 2).  The table load (h + coefficients + directory, ~600 cycles
per slot) happens once per merged GEMM at descriptor start.

**2.2 Sampled gradient (lw_sg).** Per token the dY row and the X row (768 words each) are loaded into two row
banks (single-buffered, 1 word / cycle each); the dispatcher walks the 256 atoms, reading the two reflected
7-word windows (1 word / cycle / bank, 7 cycles per atom) into a free chain slot (NL = 4 lanes x 2 slots,
round robin).  A lane serves its two slots on alternate cycles: `p = mul(yw, xw)`, `acc = fma(h_tap, p, acc)`
for the 25 taps in nonzero(H) order, the 2-stage FMA feeding its own result back without a bubble; acc_j lives
in the accumulator bank across tokens (+0 at the start); at the end `grad_j = mul(gamma, acc_j)`.  The lane's
operands for the slot whose turn is next are selected one cycle ahead and registered (the fp inputs see one mux
level); the window positions are running counters so only the reflection fold is in front of the address
registers.

**2.3 AdamW (lw_adamw).** One element stream: segment A (clip mul, decay mul, g - m, g * g, v * b2, two FMAs, 1
element / cycle), a sqrt pool (NSQ = 2 fp_sfu, round robin, in order), `den = add(mul(sqrt(v), rbc2), eps)`, a
recip pool (NRC = 1), `p = sub(p, mul(step, mul(m, recip(den))))`; 8-element groups are written as three beats
(p, m, v).  The LOWA_SCAL block (24 words) carries the per-group scalars.

## 3. Verification (L2 = bit-exact vs L1; `build/test-lowa/summary.txt`)
Clean `make test-lowa` from an empty build/test-lowa (generation, both Verilator builds, all runs): 28 runs
PASS, `TEST-LOWA PASS`, 5,269 s wall (88 min; the two gm_top GEMM cases take ~37 min each).

| case | what | vectors | mismatches | cycles |
|---|---|---|---|---|
| merge_unit | 128 x 128, 8 atoms: edge-folded (0,0), (127,2), (5,127), overlapping and identical centres, an empty block, 4 tiles; lat 24 / lat 7 + 30 % stalls / Icarus | 16,384 | 0 | 1,141-1,217 (400 mg accesses = 200 hits x RMW) |
| merge_real_row / merge_real_trn | layer-0 Wq, all 256 atoms, step-B coefficients, 144 ROW and 144 TRN tiles | 589,824 each | 0 | 15,135; 12,800 mg accesses (6,400 hits); first tile 707 cycles (table load), later tiles ~2 cycles / hit |
| **merge_gemm** (tb_lw_gemm) | **Q = X Wq^T + bq of layer 0 (4096 x 768 x 768) with gm_top streaming the frozen W0 and lw_merge adding the atoms on load; Q vs L1's GEMM on L1's merged weight** | 3,145,728 | 0 | 9,697,593: 9,216 W tiles merged, 819,200 mg accesses (409,600 hits = 6,400 x 64 row blocks), merge_busy held 98.6 cycles per tile on average (max 312), 0 gated cycles |
| nomerge_gemm (tb_lw_gemm) | the same GEMM on L1's merged weight with the merge engine idle (= P3 real_qproj, the tile-rate reference) | 3,145,728 | 0 | 9,697,592 (= P3 real_qproj) |
| sg_unit | 5 tokens x 256 random atoms; lat 24 / stalls | 256 | 0 | 14,965 |
| sg_edge | 3 tokens at row stride 2 x 768, edge-folded + overlapping atoms; lat 24 / stalls / Icarus | 256 | 0 | 9,405 / 9,307 |
| **sg_real_l0q** | **layer-0 query, all 4,096 tokens of the step-B backward (dQ, X from the hooked L1 step), all 256 gradients** | 256 | 0 | 11,387,945 (2,780 / token) |
| **sg_real_l11q** | layer-11 query, the 32 CLS rows only (every other dQ row is exactly zero, D6); the hooked call equals `stepB_l1.g_coef[22]` | 256 | 0 | 90,025 (2,813 / token) |
| scal_s0 / scal_peak | s = 0 (lr = 0 exactly) and s = 140 (peak lr); lat 24 / stalls / Icarus | 24 each | 0 | 306 / 274 |
| adamw_s0 / adamw_clip / adamw_noclip | n = 100 (8-word tail): m = v = 0 + lr 0; peak lr + in-engine clip; s = 700 no clip; lat 24 / stalls / Icarus | 300 each | 0 | 1,067 / 1,035 |
| **adamw_real_coef** | **all 24 modules' coefficients (6,144) from the real step-B gradients, L1 clip, post p / m / v vs `stepB_l1`** | 18,432 | 0 | 55,463 (9.03 / element) |
| **adamw_real_head** | **dense W + b, out_proj W + b (592,130 elements), real gradients, post p / m / v vs `stepB_l1`** | 1,776,390 | 0 | 5,329,353 (9.00 / element) |

Goldens come only from the L1 entry points (`H.lowa_merge`, `H.lowa_sg`, `H.adamw`, `H.step_scalars`, `H.gemm`);
the real SG / AdamW generators also assert that their inputs and outputs equal the recorded L1 step
(`model/runs/stepB_l1.npz`).  Verilator runs the full set (MEMW 23); Icarus runs sg_edge, scal_s0, adamw_clip and
merge_unit (MEMW 15, +lat=5).

**The sampled-gradient bug of the handoff (atom 1 wrong, everything else exact).** `all_idle` (the condition for
leaving S_FIN and starting the gamma pass) dropped `s_busy` at the last FMA issue, three cycles before the chain's
result reached the lane result register; S_OUT therefore started early and its accumulator read for atom 1 was
pre-empted by the final accumulator write (the bank controls give writes priority), so word 1 was scaled from a
stale read.  Fixed by including the in-flight last FMAs (`res_pend`) in `all_idle`.  No vector or L1 change.

**Merge inside the GEMM engine (tb_lw_gemm).** gm_top's W tiles come from memory (frozen W0); lw_top's merge
descriptor is started one cycle before the GEMM descriptor (events are latched while the merge op is busy), rd2
carries the tables, `mg_*` / `w_tile_done` / `merge_busy` connect to gm_top.  **Result: the merge costs the
GEMM no cycles.**  merge_gemm (W0 + lw_merge) and nomerge_gemm (L1's merged W, merge idle) are both bit-exact
(3,145,728 words, 0 mismatches) and take 9,697,593 vs 9,697,592 cycles; the one cycle is the tb's start stagger
(the merge descriptor is started one cycle before the GEMM's and the count starts there).  The controller was
never held by `merge_busy` (gate_cycles 0: no cycle in which a block was ready except for the merge): a tile's
merge (table hits of the next tile prefetched, ~2 cycles per hit) takes 98.6 cycles on average and at most 312
against the 1,052-cycle W-tile period of the 16 x 16 array, and the ~600-cycle table load at descriptor start
is hidden behind the first X / W tile loads (first w_tile_done ~550 cycles after the start, its merge completes before the
block's compute needs the buffer).  Merge stall from merging: 0 cycles (0.00 %).
Two testbench defects found on the way (no RTL or L1 change): gen_lowa's GEMM descriptor omitted the batch
counts `nb1 = nb2 = 1` (gm_ctrl reads 0 as 65,536 batches, so the GEMM repeated the same 4096 x 768 output
forever), and tb_lw_gemm waited for the one-cycle `done` and `m_done` pulses in the same cycle (they are now
latched).  Simulation time: ~37 min per case (Verilator, one thread, ~4.3 k cycles / s).

## 4. Cycles vs model (docs/calc/budget.py, updated)
| op | measured | model | note |
|---|---|---|---|
| LOWA_SG per token | 2,780 (4,096-token layer-0 query), 2,813 (32 CLS rows) | 256 x 7 + 768 = 2,560 + `LOWA_TOK_OVH` 218 | window-bound dispatch; the overhead is the chain-slot wait (~6 cycles per 8 atoms: the 25-tap chain of a slot, 56 cycles, is one cycle longer than 8 slots x 7), the row-request latency and the chain drain.  241.3 -> 261.8 M cycles per step, hidden beside the array |
| ADAMW per element | 9.0 | `OPT_CYC_EL` 9.0 (was `12 / V` on the vector unit) | sqrt / recip pool bound (NSQ 2, NRC 1); 5.38 M cycles for the 598,274 parameters (0.17 % of the step) |
| merge per tile | ~2 cycles / hit + ~10, table load ~600 cycles per slot once per GEMM | hidden behind the 1,024-cycle block (dataflow.md) | measured in gm_top (merge_gemm): 98.6 cycles per tile (max 312) against the 1,052-cycle tile period, 0 gated cycles: fully hidden, as budget.py assumes (it counts merge bytes only, no cycles) |

## 5. Synthesis and STA (`make synth-lowa`, `build/synth/lowa_table.md`)
| unit | parameters | cells | area tt (um^2, std cells) | slack ss raw (ns) | slack ss repaired (ns) | slack tt raw (ns) | hold ss (ns) | runtime |
|---|---|---|---|---|---|---|---|---|
| lw_top | NL=4, NSQ=2, NRC=1 | 192,577 | 5,886,525 | -5.28 | **+8.36** | +7.61 | +0.24 | 4.4 min |

| block | cells | area tt (um^2, std cells) | slack ss repaired (ns) |
|---|---|---|---|
| lw_sg (sampled gradient) | 52,296 | 1,516,130 | +8.36 |
| lw_adamw (AdamW, NSQ 2 + NRC 1 fp_sfu = 3 x ~210 k inside) | 77,507 | 2,522,430 | +9.13 |
| lw_merge (merge-on-load) | 16,872 | 534,933 | +9.89 |
| lw_scal (LOWA_SCAL, 1 fp_sfu) | 10,046 | 315,870 | +9.26 |

Yosys 0.62 `synth -flatten -booth` with lw_sg / lw_adamw / lw_merge / lw_scal, fp_sfu, mm_sram_bank and cm_rstbuf
kept as hierarchy (`flow/synth/synth_lowa.sh`), abc -D 40000 on gf180mcu_as_sc_mcu7t3v3 tt; OpenSTA at 40 ns with
registered inputs (`wrap_lw_top.v`); 'repaired' = after OpenROAD repair_design (`flow/sta/repair_sta.tcl`, the
pass criterion); per-block slack = the worst path ending in the block after the same repair
(`flow/sta/lowa_blocks.tcl`, `build/synth/lw_top_NL4/blocks_ss.rpt`).  Area excludes the SRAM macros (lw_merge 16,
lw_sg 32 sram256x8).  Timing repairs on the way there: (1) the lane's fp inputs were behind the slot-parity mux, the
tap one-hot decode and the 25:1 h mux (raw -78 ns): the next slot's operands are now selected a cycle ahead and
registered; (2) the window address was `fold(ci + w - 3)` in one cycle: now running counters + fold; (3) the global
slot-parity flop `ph` fanned out to every operand mux select (-3.7 ns after repair): one copy per lane and purpose
(the P3 / P4 cm_rstbuf pattern).  Area: the first synthesis had 52 k flops, 32 k of them lw_adamw's stream and
stage FIFOs, sized far beyond what the 9-cycle / element throughput needs; the FIFOs were halved (2 beats per
stream, 16 / 16 / 32 / 16 / 32 entries) with the issue throttles adjusted (7.7 M -> 5.9 M um^2, cycles unchanged).

## 6. Deviations from the spec / plan
- Block directory (256 words per module) instead of per-block run-length headers in the hit list (+1 KiB DRAM per
  module); the hit entry carries the atom index explicitly.
- Two coefficient / directory slots in lw_merge (one per K segment of GEMM_ACC2), each loaded once per merged GEMM
  (~600 cycles per slot exposed on the first tile only).
- AdamW runs in the LoWA unit with its own sqrt / recip SFUs (spec Q9 left it to the vector unit); budget.py now
  counts 9 cycles per element there.
- lw_sg rows are single-buffered (the budget's +D per token); the layer-11 query feeds only the 32 CLS rows.
- The merge protocol requires the merge descriptor to be started before the GEMM's first `w_tile_done` (the P6
  sequencer issues the LoWA merge op before the GEMM op; tile events are only latched while the merge op is busy).
