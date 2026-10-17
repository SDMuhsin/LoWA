# P6b handoff (paused by PI instruction)

Done (docs/p6b_report.md): hw/rng (rn_ks, rn_tf, rn_mkgen) bit-exact vs model/rng.py, integrated in mm_top as the default
keep-bit source (MK_DRAM = 0); `make test-rng` PASS (KAT, 1M random, real step-B attn L0 / ffn_out L5 masks, generator-
sourced real_smax / real_smax_bwd / real_emb / real_attnout / ep_bias_drop_resid); vu_loader per-row 4-beat bursts
(HBL = 2), real_ln_fwd 5,255,246 -> 2,963,526 cycles (eff 0.548), budget 135.0 -> 131.6 s; `make test-vec`, `test-mem`,
`synth-vec` (+2.56 ns), `synth-rng` (+10.45 / +14.11 ns) PASS.

## Open: `make synth-mem` FAIL, mm_top -3.79 ns at ss after repair (40 ns)
- Path: wrap flop `q_ra[100]` (rd_req_addr of read channel 3) -> mm_top g_wr[2] WAR term `oi` (offered read request vs
  write beat) -> `w_blk[2]` -> mm_wrport `wr_ready` -> `u.g_wr[2].u_wr.q_line[*]` enables. Reports:
  build/synth/mm_top_full/repair_ss.rpt (first path), netlist build/synth/mm_top_full/nl.v.
- Diagnosis: the time is in two high-fanout nets (nor3_2: 7.7 ns, 11.8 ns) feeding the write-queue enables; arithmetic is
  short. Same RTL with MK_DRAM = 1 closes +11.64 ns (build/synth/mm_top_dram), so it is a mapping/fanout effect of the
  changed channel set, not rn_mkgen logic (rn_mkgen alone has +10 ns).
- Tried once: `oi` rewritten as one 31-bit subtract + small compares (equivalent; kept; test-mem-quick PASS): -3.79 ns.
- Next step (one of): (a) register the WAR block decision (`w_blk`) and accept the beat a cycle later (re-run test-mem:
  order cases); (b) buffer the `wr_ready`/`blk` -> queue-enable net with kept-hierarchy duplicate drivers (as P6a's
  cm_rstbuf / rdport enables); (c) re-check slack after placement in P8.
- Then: `make synth-mem` and update docs/p6b_report.md section 4 / 6.0.
