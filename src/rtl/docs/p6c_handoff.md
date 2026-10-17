# P6c handoff (paused per the PI plan)

Done (docs/p6c_report.md):
- mm_top timing closed: +0.73 ns ss; synth-mem and test-mem PASS.
- ISA as built: docs/isa.md section 7 and sw/isa.py.
- hw/ctrl: ct_seq + ct_dma. synth-ctrl +1.61 ns, 0.86 mm2; test-ctrl PASS 6/6, including the full and reduced
  compiled programs against the reference decode.
- sw/compile.py: full step = 141 descriptors, 16,640 B.
- sw/interp.py + sw/check_compiler.py: COMPILER-CHECK PASS, bit-exact vs the L1 step on all 24 result tensors.

## Open: deliverable 3b, the reduced program on RTL (sequencer + 3 engines + mm_top + tb_dram)
Ready:
- `python3 sw/compile.py --layers 1 --batch 2 --seq 32` writes build/compile/prog_l1_b2_s32.bin and
  image_l1_b2_s32.npz (37 MB, sparse regions: "r%08x" byte address -> uint32 words).
- Golden: `interp.Machine` on the same image (check_compiler.run_interp / results with a Cfg(1, 2, 32)). Compare
  LOSS_OUT, GRAD (512 coef + head), PARAM / ADAM_M / ADAM_V, and the logits at Cfg.cls["z"].

Steps:
1. Build a system tb (tb/ctrl/tb_ct_sys.sv). Instantiate:
   - ct_seq;
   - gm_top (16x16);
   - vu_top;
   - lw_top;
   - mm_top with NR = 8 (append the ct read channel last) and NW = 4 (ct write last), MK_DRAM = 0;
   - tb_dram (tb/mem/mm_dram_inst.svh, nominal knobs).
2. Wire the fields:
   - Payload words map to engine ports in sw/isa.py field order:
     - GEMM_FIELDS words 0..35 -> gm d_* (m, n, k*, nb*, trn / ep flags, sexp) and lw_top m_* (words 26..35 +
       m_trn = word 3 bit 25);
     - VEC_FIELDS -> vu d_*;
     - LOWA_FIELDS -> lw d_op / d_flag / d_a0..a7 / d_len.
   - mk_site / mk_layer: channel 0 = GEMM payload word 4 [11:8] / word 5 [3:0]; channel 1 = VEC word 0 [15:12] /
     word 1 [3:0].
   - Write channels: wr_idle order {LOWA, VEC, GEMM, DMA} = ct `wr_idle[3:0]`.
3. Image loading: tb_dram reads the "regions + $readmemh" files of tb/mem/mm_image.py. Convert the npz with
   mm_image.Image.put per region. The PROG region needs ≥ 48 words of slack after END, because fetch reads 48 words.
4. Kick-off CSRs, as in tb_ct_seq: PROG_BASE 0, SLOT_DELTA 0x5E687000, STEP_T 140, SEED = stepB `inp["seed"]` (42;
   check `inp["seed"]`), CTRL 0x11, DOORBELL 0x80000000. Wait for irq with a cycle limit.
   - Expected: the GEMM rate is bound by W streaming, about 2-4 M cycles.
5. Check items to verify on first run:
   - gm_top `m_mb` semantics: compile.py uses ceil(M/64); confirm Br in gm_ctrl.
   - vu GN_ROWS / GN_FIN o2 word count (W0 / W03 phases).
   - lw_scal writing words 15 / 23: compile.py copies clip after LOWA_SCAL, so either is fine.
   - The ones-GEMM with M = 1 and the N = C = 2 GEMMs: engine small-shape support.
   - The E_DROP with c_rs = L·768 (strided CLS rows).
6. Report the cycles in p6c_report.md section 5 item 1, and add the case to tb/ctrl/run_tests.py.

Other follow-ups: CLS pruning (D6) in compile.py; per-(B, L) programs (or dim sel); queue depth 2.
