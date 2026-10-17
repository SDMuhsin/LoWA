# P6a handoff (closed; only PI decisions / later-phase items remain)

P6a is complete: docs/p6a_report.md (TEST-MEM PASS clean 18 m 55 s, SYNTH-MEM PASS +2.61 ns ss after repair,
9.74 mm2, lint PASS, model-check PASS, budget.py / budget.md DRAM-charged).

Open, not P6a work:
1. **Vector unit request order (P4 RTL, PI decision).** vu_loader / vu_writer issue row-interleaved 1-beat
   requests; real_ln_fwd runs 5.26 M cycles at 800 MB/s vs 2.66 M compute-bound.  Exact change: report section 4.
2. **DRAM R/W policy (R3, off-die).** Nominal model shares bandwidth (+wfair=1, GEMM eff 0.993); a read-priority
   controller gives 0.666 (budget.py --dram-rprio, step 189.0 s vs 135.0 s).
3. **P6b:** Threefry keep-bit generator (MK_DRAM = 0 path ready), DMA_COPY (ISA 0x30), sequencer must wait on
   `wr_idle` before handing data to another unit.
