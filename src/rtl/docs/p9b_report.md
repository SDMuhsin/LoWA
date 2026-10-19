# P9b report: measured-activity chip power (gate-level SAIF, PrimePower)

P9b replaces the P8 power range (0.475 W vectorless / 0.60 W RTL-SAIF upper bound, docs/p8_report.md section 7)
with power that rests on real switching activity: zero-delay gate-level simulation of the synthesized chip
netlist on the real training program, SAIF per phase, and PrimePower on the complete flat chip at tt. It covers
the latch-RF chip of record (18.04 mm2, 5.5 ns, P8f mm). From the per-phase power it gives the energy of a
full-size training step.

__RESULTS__

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

Each stimulus was also run on the RTL (VCS, the same testbench, `GLS_RTL=1`, flow/tsmc65/sim/run_p9b_ref.sh), and
`flow/tsmc65/sim/p9b_cmp.sh` compares the two runs.

__CHECK__

### 2.4 What is and is not measured

__LIMITS__

## 5. Reproduction

Locally (src/rtl):

```
python3 tb/top/run_e2e.py --layers 1 --batch 2 --seq 32 --steps 1 --prep-only --out build/p9b/e2e   # l1 stimulus
for k in 3 7 10 11 30 32 40 44 48 53; do python3 tb/power/mk_cut_stim.py --cut $k --out build/p9b/stim/c$k; done
# copy dram.reg / dram.bin / host.txt of each to the server build/tsmc65/stim/<l1|c<k>>/
```

Server (src/rtl of the P9b tree, netlists read from the P8 checkout with T65_HIER=<P8 checkout>/build/tsmc65/hier):

```
flow/tsmc65/sim/run_gls.sh build                                   # GLS compile (GLS_DIR selects the build dir)
GLS_PAR=3 flow/tsmc65/sim/run_p9b.sh                               # one SAIF window per line of tb/power/win/p9b_plan.txt
T65_GLS_WAIT_H=15 flow/tsmc65/sta/run_pt_gls.sh gls1 tt <window list>   # PrimePower, processes each SAIF as it lands
flow/tsmc65/sim/run_p9b_ref.sh 80000; flow/tsmc65/sim/p9b_cmp.sh   # RTL references and the comparison
python3 tb/power/p9b_energy.py <summary ptgls_gls1_tt.txt>         # phase power and step energy (locally)
```
