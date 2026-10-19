# P9a report: full-size training-step cycles measured on the RTL

DRAFT: numbers marked TBD are pending.

## 1. Question and answer
The paper's full-size step time (12 layers, B 32, L 128) came from docs/calc/budget.py, an analytical model:
3,290.6 M cycles = 18.1 s at 181.8 MHz, "80 % array utilisation". P9a replaces it with a cycle count measured in
cycle-accurate simulation of the chip RTL (current main: hw/top/top_chip incl. the P8f mm skid buffers, Verilator
5.044, tb/top/tb_top.sv + the tb_dram AXI4 DRAM model at the nominal knobs: 24-cycle read latency, 8-cycle write
response, 32 B/cycle shared by reads and writes, fair R/W, ruling R9).

TBD

## 2. Method
TBD
