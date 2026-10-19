# P9b status: measured-activity chip power

## 2026-10-19 19:55 UTC
- State: gate-level simulation flow built. VCS zero-delay simulation of the 17 chip netlists of the flat-chip
  PrimeTime run at 5.5 ns (latch RF, P8f mm), driven by `tb/power/tb_top_gls.sv` (tb_top without RTL-internal
  probes). The sequential library cells are replaced by delayed behavioural models (`site/gls_cells.v`) to avoid the
  zero-delay clock-gate race. The VCS compile takes 2 min 40 s and 2.4 GB.
- Mid-step windows: `tb/power/mk_cut_stim.py` starts the program at unit op k, with the DRAM image the interpreter
  produces for ops < k. An RTL check of the cut at op 44 (LOWA_SG) is running in Verilator.
- Next: measure the gate-level simulation speed, plan the windows, then run PrimePower on them.
- ETA: the plan is set once the speed is known.

## 2026-10-19 20:15 UTC
- Speed measured. With no SAIF, the embedding vector op runs at about 500 cycles/s. With SAIF recording and three
  simulations plus PrimePower on the host, it is about 250 cycles/s for ADAMW, about 40 cycles/s for LN_FWD and
  under 40 cycles/s for the GEMMs. A whole reduced step (10.36 M cycles) would take far too long, and the five
  full segments started at 20:00 were stopped.
- Plan now: 11 sampled windows of 30-100 k cycles (tb/power/win/p9b_plan.txt), one simulation per op type, each
  started at its op with a cut stimulus. That is 3 GEMMs (QKV with merge, FFN1 with GELU, FFN backward), 6 vector
  ops (EMB, SMAX_FWD, LN_FWD, LN_BWD, SMAX_BWD, GN_ROWS), LOWA_SG and ADAMW. Two run at a time. One PrimePower
  session loads the flat chip once and processes each SAIF as it lands. Per-phase power comes from these windows,
  and the step energy is composed with phase times.
- ETA: simulations about 2-3 h, PrimePower 1 h to load plus about 10-20 min per window, so numbers around
  02:00 UTC.

## 2026-10-19 21:15 UTC
- Gate-level vs RTL check (both VCS, the same testbench, l1 stimulus, first 175 k cycles): the 9 op-log events
  match cycle for cycle. They cover EMB (130 k cycles), DMA, LOWA_SCAL and the first GEMM start (CMP_OPLOG SAME).
- GEMM gate-level speed with SAIF is about 4 cycles/s, so the GEMM windows were cut to 15 k cycles (g_qkv,
  g_ffn1b). g_ffn1 keeps 40 k cycles.
- The NFS home quota ran full at 21:05; the server account is shared with the other phases. The PI approved writing
  the SAIFs and stimuli to tau-03 local `/tmp/lwta_p9b` (chmod 700, symlinked from the run directories). It will be
  deleted once the numbers are final.

## 2026-10-20 00:00 UTC
- Bug found and fixed: `$toggle_report` rejects a reg-vector file name, so the first window runs (20:13-22:10)
  wrote no SAIF and were repeated with a string name. Every SAIF is now checked before PrimePower takes it.
- Batch 1 is done: 11 windows, PrimePower on the flat chip at tt, 100.00 % of nets and of leaf cells annotated
  from the gate-level SAIF.

  | window | W |
  |---|---|
  | GEMM QKV (merge on) | 0.868 |
  | GEMM FFN1 (GELU) | 0.873 |
  | GEMM FFN bwd | 0.553 |
  | EMB | 0.419 |
  | SMAX_FWD | 0.433 |
  | LN_FWD | 0.450 |
  | LN_BWD | 0.408 |
  | SMAX_BWD | 0.402 |
  | GN_ROWS | 0.435 |
  | LOWA_SG | 0.396 |
  | ADAMW | 0.396 |

  The floor with every engine idle is about 0.39 W: the clock pins of the 48 k ICGs at full rate (0.30 W, ideal
  clock) plus leakage (0.082 W).
- Gate level = RTL: on v_emb (1,024 write beats) and v_smax (544) the DRAM write-stream signatures match the RTL
  run, and the op logs match cycle for cycle.
- Finding: the backward of the 1-layer program runs on CLS-only gradients (all dY rows but 2 are zero), so its
  backward windows under-read. Batch 3 re-measures the backward on layer 0 of the 2-layer program, where the
  gradients are dense (MEMW 25 build). Batch 2 adds more forward GEMM types (K, attention scores, O, FFN2).
- ETA: batches 2 and 3 about 02:30, PrimePower about 03:00.
