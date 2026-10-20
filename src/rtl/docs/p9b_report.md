# P9b report: measured-activity chip power (gate-level SAIF, PrimePower)

P9b replaces the P8 power range (0.475 W vectorless / 0.60 W RTL-SAIF upper bound, docs/p8_report.md section 7)
with power that rests on real switching activity: zero-delay gate-level simulation of the synthesized chip
netlist on the real training program, SAIF per phase, and PrimePower on the complete flat chip at tt. It covers
the latch-RF chip of record (18.04 mm2, 5.5 ns, P8f mm). From the per-phase power it gives the energy of a
full-size training step.

## 1. Results

**What was obtained: the preferred method, at window level.**
- The switching activity is a zero-delay gate-level VCS simulation of the complete synthesized chip (all 17
  netlists of the 5.5 ns flat-chip run). It runs the compiled training program on real data, and the gate-level
  SAIF covers **100.00 % of the 5,368,682 nets and 100.00 % of the leaf cells** (the 1.2 M latch-RF bits
  included).
- The power is PrimePower (averaged) on the flat chip at tt (1.0 V, 25 C), 181.8 MHz.
- The workload is 22 windows of 10-60 k cycles, each inside one op, cut from the real step program (the 1-layer and
  2-layer reduced programs, B 2, L 32). A whole step cannot be simulated at gate level (section 2.2).
- Every window's gate-level run is cycle-exact and bit-exact against the Verilator RTL (section 2.3).
- All numbers are **pre-layout**: ideal clocks (no clock tree), DC wire estimates.

### 1.1 Per-phase power (tt, 181.8 MHz)

| phase | windows (op, W) | phase power |
|---|---|---|
| **GEMM** | Q-projection with LoWA merge 0.868, K 0.867, O 0.872, FFN2 0.871, FFN1 + GELU 0.873; backward dX (2-layer, dense gradients): FFN2-dX 0.867, FFN1-dX 0.874, O-dX 0.868; attention scores 0.726 | **0.864 W** (full-size MAC mix: forward dense 32 %, FFN1 16 %, backward dX 47 %, attention 4.5 %) |
| **vector** | SMAX_FWD 0.433, LN_FWD 0.450, SMAX_REC 0.432, SMAX_BWD 0.418, LN_BWD 0.427 (dense gradients); also EMB 0.419, GN_ROWS 0.435 | **0.433 W** (full-size vector-cycle mix of budget.py) |
| **LoWA** (LOWA_SG, beside the GEMMs) | LOWA_SG alone 0.403 W; the LoWA engine (u_lw) draws 0.042 W in it vs 0.025 W idle | **+0.017 W** on top of the GEMM phase while it runs |
| **optimiser** (ADAMW) | 0.396 | **0.396 W** |
| all engines idle (floor) | ICG clock pins 0.30 W (ideal clock, every cycle) + leakage 0.082 W | about 0.39 W |

- Split of a GEMM window (FFN1, 0.873 W):
  - u_gm 0.747 W, of which the PE array 0.512, weight feed 0.077, drain 0.063, X feed 0.056, epilogue 0.032;
  - u_vu 0.071 (idle), u_mm 0.029, u_lw 0.025.
  - By group: clock network 0.305, combinational 0.454, registers 0.114. Switching 0.284, internal 0.506,
    leakage 0.082.
- The **clock network group is 0.30 W in every window, idle or busy**. It is the internal power of the 48,063 ICGs
  (37,376 of them the latch-RF word gates), whose clock pin sees the ideal root clock every cycle whatever their
  enable. It is a cost of the latch register file, and a clock tree would add buffer power on top (not modelled).
- The 1-layer program's own backward windows read lower (FFN backward 0.553 W, LN_BWD 0.408, SMAX_BWD 0.402,
  LOWA_SG 0.396). There the loss gradient reaches only the 2 CLS rows of dY (every other row is zero), which the
  11 lower layers of the 12-layer step never see. The phase table therefore uses layer 0 of the 2-layer program,
  where the gradients are dense.

### 1.2 Energy per training step (full size: 12 layers, B 32, L 128)

E = P_GEMM t_GEMM + P_vec t_vec + P_opt t_opt + dP_LoWA t_LoWA, with **phase times from budget.py** (docs/p8_report.md
section 8; P9a's measured cycles were not available at the time of writing, see 1.3):

| phase | cycles | time at 181.8 MHz | power | energy |
|---|---|---|---|---|
| GEMM | 2,777.9 M | 15.28 s | 0.864 W | 13.19 J |
| vector | 506.7 M | 2.79 s | 0.433 W | 1.21 J |
| LoWA SG (beside GEMM, no time of its own) | 261.8 M | (1.44 s) | +0.017 W | 0.02 J |
| optimiser + misc | 6.0 M | 0.03 s | 0.396 W | 0.01 J |
| **step** | **3,290.6 M** | **18.1 s** | **0.80 W average** | **14.4 J** |

- GEMM is 91 % of the step energy.
- For comparison, the measured reduced step itself (1 layer, B 2, L 32, 10.36 M cycles = 57 ms, each op at its
  own window's power) averages 0.52 W, or 29.7 mJ. It is dominated by the AdamW of the classifier head (51 % of
  its cycles), which at full size is 0.2 % of the step.

With P9a's measured full-size cycles (docs/p9a_report.md), the same powers give: pending (P9a ETA 06:00-08:00 UTC);
the step energy scales with the GEMM time at 0.864 W per second of GEMM and the vector time at 0.433 W.

### 1.3 Against the P8 range

The measured average, 0.80 W over a full-size step and 0.86-0.87 W in a GEMM, is **above** P8's 0.60 W "RTL-SAIF
upper bound". It is not a contradiction of a measurement.
- P8's figure annotated only about 10 % of the sequential outputs (8 % of the nets) from RTL and let PrimePower
  propagate the rest, the latch-RF arrays included. P8's GEMM workload was also a VCS RTL run.
- In this phase the VCS RTL simulation of gm_top (MM_TECH=2) was found **not to match** the Verilator RTL, which
  is the simulator of every bit-exact test, on 12 of the 15 stimuli whose first 80 k cycles contain GEMM work.
  It writes fewer DRAM beats and gives different write signatures; the 7 vector / LoWA / AdamW-only stimuli all
  match. For example, the attention-score
  GEMM (op 6) does not finish within 80 k cycles in VCS RTL, while it finishes in 16,247 cycles in Verilator and
  at gate level. The cause was not traced.
- The vectorless 0.475 W assumed 10 % input toggle rates. The GEMM datapath with real FP32 operands toggles
  much more (switching power 0.28 W against 0.048 W vectorless).
- Both P8 figures are superseded by this one.

### 1.4 Proposed paper sentence

"Power was obtained from a zero-delay gate-level simulation of the complete synthesized netlist running the
training program on real data, with the resulting switching activity (100 % of nets annotated) analysed in
PrimePower at the typical corner (1.0 V, 25 C) and 181.8 MHz before place and route. The chip draws 0.86 W while
the GEMM array is busy and 0.43 W in vector phases, for 14.4 J per full-size training step (0.80 W average over
18.1 s); clock-tree power is not included."

## 2. Method

### 2.1 Design and corner

- Netlists: the 17 block netlists of the flat-chip PrimeTime run at 5.5 ns (docs/p8_report.md section 4.2,
  T65_KID_TAG `gm16=_p5.25,vu16=_p5.25,mm=_p8f,rng9=_p8f,rng3=_p8f`), read-only from the P8 server checkout:
  5,189,049 leaf cells, 420,535 flops, 1,196,032 RF latches.
- PrimePower (pt_shell Q-2019.12-SP2, `power_analysis_mode averaged`), tt 1.0 V 25 C, 5.5 ns clock (181.8 MHz).
  The flat chip, all 356 per-instance DC-topographical SPEF reads (ss RC estimate, as in P8), the DC SDC
  regenerated for tt, ideal clocks. The design is loaded once (48 min, 7.5 GB) and each window's SAIF is applied
  in turn (`reset_switching_activity`, `read_saif`, `update_power`), 2-6 min per window
  (`flow/tsmc65/sta/pt_gls_power.tcl`, `run_pt_gls.sh`).

### 2.2 Gate-level simulation

- VCS S-2021.09, zero delay: no SDF, `+nospecify +notimingcheck`. The combinational cells come from the
  library's Verilog models. The sequential cells (flops, RF latches, ICGs; 27 cell types) are replaced by
  behavioural models in the git-ignored site file `flow/tsmc65/site/gls_cells.v` (template
  `gls_cells.example.v`). In those models every flop or latch output moves 0.05 ns after its clock edge or
  enable, while the clock gates stay zero-delay.
  - Why: the library models clock their state through zero-delay UDPs, so with ideal clocks a flop behind a
    clock gate (each latch-RF word and the DC clock gates) would sample its data a delta cycle after the
    ungated flops had already updated. This is the classic zero-delay clock-gating race.
  - All state starts at 0 (the RTL simulations use Verilator `--x-initial fast`).
  - Compile: 3 min, 2.4 GB. A run uses 4.8 GB, or 6.5 GB with SAIF recording.
- Testbench: `tb/power/tb_top_gls.sv`. It is tb/top/tb_top.sv (same top_chip hookup, DRAM model with the
  nominal knobs, AXI4-Lite host script) with only nets that exist in both the RTL and the netlist:
  - top_chip has no logic, so its unit-to-unit wires keep their RTL names;
  - the clock is the chip clock (5.5 ns), so SAIF durations need no rescaling;
  - SAIF windows (`$toggle_start / $toggle_stop / $toggle_report`, gate-level monitoring on, region u_chip) are
    given in cycles from the start of the first unit op;
  - it prints a DRAM write-stream signature (all accepted W beats: data, strobes) at every heartbeat.
- Workload: the compiled training program of the e2e test (tb/top/run_e2e.py, 1 layer, B 2, L 32, one step,
  10.36 M cycles; real roberta-base weights and SST-2 tokens). The full-size program cannot be simulated at gate
  level: the GEMM runs at 4-19 cycles/s, the vector ops at 20-60 cycles/s.
- Windows deep in the step (`tb/power/mk_cut_stim.py --cut k`): the program starts at unit op k.
  - The interpreter (sw/interp.py) executes every descriptor before op k on the initial image, so the DRAM image
    holds exactly the data those ops produce.
  - The descriptors before op k (except SETCSR) become NOPs, slot for slot.
  - A pending event flag of a skipped producer is dropped from its first consumer's WAIT.
  - Check: the RTL run of the cut at op 44 reproduces every op duration of the uncut program: 179,919 /
    179,919 / 1,275 / 1,271 / 191,831 / ... / 5,329,393 cycles (the first op differs by 1 cycle).
- One window per op type, 10-60 k cycles inside the op, after its first operand loads (tb/power/win/p9b_plan.txt).

### 2.3 Gate level = RTL (functional check of the power simulation)

Reference: the same testbench on the RTL in Verilator, the simulator of the bit-exact e2e tests
(`tb/power/run_ref_verilator.sh`; the SAIF code is compiled out). Every stimulus was run to 80 k cycles.
`tb/power/cmp_vref.py` checks two things against the gate-level run of each window:
- every heartbeat (every 10 k cycles: cycle, unit busy flags, DRAM write beats, and a 64-bit signature over every
  accepted W beat's data, strobes and address);
- the op log (cycle of every unit start / done / retire).

Result: **CMP_VREF ALL OK** for all 22 windows. Write
beats with data inside the gate-level runs:

| windows with DRAM writes before their last heartbeat | write beats compared (signature equal) |
|---|---|
| v_emb, v_smax, v_ln, v_lnb, v_smaxb (1-layer) | 1,024 / 1,836 / 3,060 / 1,536 / 2,048 |
| v2_lnb, v2_smaxr, v2_smaxb (2-layer) | 1,044 / 2,240 / 2,688 |
| o_adamw (head AdamW, 6 op-log events) | 2,552 |
| g_att (attention-score GEMMs, epilogue writes) | 1,536 |

- The long GEMMs write their first C block only after 10-20 k cycles, after the windows' last heartbeat. For them
  the check is the cycle-exact busy flags and op log; the GEMM datapath's bit-exactness is shown by g_att.
- An earlier 175 k-cycle run from reset (no SAIF) matched cycle for cycle: EMB done / retire, DMA, LOWA_SCAL and
  the first GEMM start.
- The flops' 0.05 ns output delay therefore changes no cycle, and the netlist (Formality-verified, P8) behaves as
  the RTL on these workloads.

### 2.4 What is and is not measured

Measured:
- Switching activity of every net of the chip netlist: 5,368,682 nets, 100.00 % annotated from the gate-level
  SAIF, and 100.00 % of the leaf cells fully annotated. This includes the 1.2 M latch-RF bits, their word ICGs and
  the DC clock gates. PrimePower propagates nothing.
- It is taken on real data (roberta-base weights, real tokens) running the compiled training program, in one
  window per op type.

Not measured, or estimated:
- **Pre-layout.** There is no clock tree: clock nets are ideal, so clock-buffer power is missing, while the clock
  pins of the flops, latches and ICGs are counted. Wires are DC topographical estimates (ss RC, used at tt). P9c's
  placement / CTS is not used here.
- **Windows, not whole ops.** Each window is 10-60 k cycles of one op, at the reduced shape (B 2, L 32), not the
  full-size shape (B 32, L 128).
  - GEMM tiles are the same 64 x 64 blocks at either size. The array-row occupancy (M = 64 rows per block) and the
    data statistics are those of the reduced batch.
  - Vector ops at full size have longer rows for softmax (L 128 instead of 32).
- **Phase power to step energy is a composition.**
  - Phase power is the weighted mean of the window powers. GEMM is weighted by the full-size MAC mix of
    budget.py's GEMM list; vector by budget.py's full-size vector-cycle mix (SMAX_FWD, LN_FWD, SMAX_REC +
    SMAX_BWD, LN_BWD).
  - Phase times are budget.py's (or P9a's measured cycles, where stated).
  - The LoWA engine runs beside the GEMMs. Its extra power is u_lw's power in the LOWA_SG window minus u_lw's
    power in the GEMM windows, not a measured overlapped window.
  - The attention-score GEMMs and DMA copies are counted inside GEMM time at the GEMM-phase power.
- tt corner only, averaged power (no time-based peak power).
- Latch-RF variant only. The SRAM-macro variant (P8e) has no gate-level run.

## 3. Reproduction

Locally (src/rtl), the stimuli (design data: program and DRAM images):

```
python3 tb/top/run_e2e.py --layers 1 --batch 2 --seq 32 --steps 1 --prep-only --out build/p9b/e2e   # l1 = cut 0
for k in 3 4 6 7 9 10 11 12 30 32 40 44 48 53; do python3 tb/power/mk_cut_stim.py --cut $k --out build/p9b/stim/c$k; done
for k in 60 62 63 66 68 70 74; do python3 tb/power/mk_cut_stim.py --layers 2 --cut $k --out build/p9b/stim/L2c$k; done
# copy dram.reg / dram.bin / host.txt of each to the server build/tsmc65/stim/<l1|c<k>|L2c<k>>/
tb/power/run_ref_verilator.sh 80000 build/p9b/stim/*        # Verilator RTL references (EDA environment of make test-e2e)
```

Server (src/rtl of the P9b tree; netlists read-only from the P8 checkout, `T65_HIER=<P8 checkout>/build/tsmc65/hier`;
the bulky simulation directories were symlinked to tau-03's local /tmp, section 4):

```
GLS_DIR=gls2 flow/tsmc65/sim/run_gls.sh build                       # 1-layer windows (MEMW 24)
GLS_DIR=gls3 GLS_MEMW=25 flow/tsmc65/sim/run_gls.sh build           # 2-layer windows (71 MiB image)
GLS_DIR=gls2 GLS_PAR=3 GLS_NOBUILD=1 GLS_WINLIST=<list> flow/tsmc65/sim/run_p9b.sh            # tb/power/win/p9b_plan.txt
GLS_DIR=gls2 GLS_PAR=3 GLS_NOBUILD=1 GLS_WINLIST=<list> GLS_PLAN=tb/power/win/p9b_plan2.txt flow/tsmc65/sim/run_p9b.sh g_k g_att g_o
GLS_DIR=gls3 GLS_PAR=3 GLS_NOBUILD=1 GLS_WINLIST=<list> GLS_PLAN=tb/power/win/p9b_plan3b.txt flow/tsmc65/sim/run_p9b.sh
T65_GLS_WAIT_H=15 flow/tsmc65/sta/run_pt_gls.sh gls1 tt <list>      # one PrimePower session, each SAIF as it lands
```

Then, locally: `python3 tb/power/cmp_vref.py <fetched sim.out / oplog.txt of each window> build/p9b/stim
tb/power/win/p9b_plan*.txt` and `python3 tb/power/p9b_energy.py <summary_gls_tt.txt> [--gemm --vec --lowa --opt
<M cycles> --src <text>]`.

Run times on tau-03 (shared, 3 simulations + PrimePower at once):
- GLS compile 3 min, 2.4 GB. A window run uses 6.5 GB with SAIF.
- Speed: GEMM 3.5-19 cycles/s, vector 20-60, LoWA / AdamW 100-250.
- The GEMM windows take 20-80 min, the others 10-25 min.
- The PrimePower load takes 48 min and 7.5 GB, then 90-110 s per window.
- A gate-level SAIF of the chip is 1.5 GB.

## 4. Files, server data, commits

- Tracked:
  - `tb/power/{tb_top_gls.sv, mk_cut_stim.py, run_ref_verilator.sh, cmp_vref.py, cmp_oplog.py, p9b_energy.py}`;
  - `tb/power/win/p9b_plan{,2,3,3b}.txt`;
  - `flow/tsmc65/sim/{run_gls.sh, run_p9b.sh, run_p9b_ref.sh, p9b_cmp.sh, p9b_status.sh}`;
  - `flow/tsmc65/sta/{pt_gls_power.tcl, run_pt_gls.sh}`;
  - `flow/tsmc65/site/gls_cells.example.v`; site.example.csh gains `T65_CELL_V`.
- Git-ignored site file: `flow/tsmc65/site/gls_cells.v` (library cell names).
- No RTL change.
- `run_p9b_ref.sh` / `p9b_cmp.sh` are the VCS-RTL comparison. It is kept for the record of the VCS-RTL GEMM
  mismatch (1.3); the reference of record is Verilator (`run_ref_verilator.sh`, `cmp_vref.py`).
- Server: scripts and the PrimePower summary are under ~/lowa_tsmc65/p9b on the NFS home.
  - The NFS quota ran full during the phase, so the PI approved putting the GLS builds, run directories, stimuli
    and SAIFs on tau-03's local `/tmp/lwta_p9b` (chmod 700).
  - The SAIFs were deleted after PrimePower had read them, and the directory is deleted at the end of the phase.
  - Netlists, SAIFs and full reports never left the server; only the summary numbers here did.
