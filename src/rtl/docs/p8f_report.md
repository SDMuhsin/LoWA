# P8f: registered mm_top handshakes (PLAN R17)

Status: complete. **With the fix, the flat chip closes at 5.50 ns at ss (181.8 MHz). P8d's chip-level figure was 7.75 ns.**
No cross-unit path through mm_top is left among the near-critical paths. The one cross-unit path that remains, seq > vu, does not touch mm and
is out of scope (PI). It misses 5.25 ns by 0.119 ns. Branch `p8f`; the gm, lw, vu and seq RTL is untouched.

## 1. Problem

P8d's flat-chip PrimeTime (docs/p8_report.md section 4.2) misses 5.25 ns at ss by -2.272 ns. The violators cross unit
boundaries through mm_top's ready outputs. The worst path is gm epilogue -> `wr_addr` -> mm_wrport `wr_ready`
(combinational through the WAR / WAW overlap compare) -> gm `wr_pop` -> credit, with arrival 7.23 ns. The other
worst endpoints are lw (-1.67) and vu (-0.12), through the same kind of handshakes. In mm's own run the
input-to-output paths `wr_addr -> wr_ready` (2.95 ns at 4.0) and the register-to-output path
`rq_len_reg -> RAW compare -> rd_req_ready` (2.95 ns) take up the whole period. The unit runs could not see this,
because both sides of each boundary were budgeted on the assumption of registered unit outputs.

## 2. Change (mm side only; gm / lw / vu RTL untouched)

| Port (toward) | Before | P8f | Latency / throughput |
|---|---|---|---|
| `rd_req_ready` (gm / vu / lw read channels) | `!rq_v \|\| acc`: the request register, then the RAW compare, then `acc` | flop: a two-entry skid buffer (`hw/mem/mm_skid.v`) replaces mm_rdport's request register; `rd_req_ready` = skid entry empty | unchanged: a request reaches the head register in the same cycle as it reached `rq` before; one request per cycle |
| `wr_ready` (write channels) | `!blk && (stage A free)`, with `blk` = WAR / WAW compare of the *offered* `wr_addr` | flop: a two-entry skid buffer in front of mm_wrport's stage A; the WAR / WAW compare runs on the buffer head (registers) | +1 cycle from handshake to stage A (writes are posted); one beat per cycle |
| `rd_dat_ready` consumption (round 2) | the packer advanced on `!dat_valid \|\| rd_dat_ready` (engine ready -> take -> FIFO pops, credits) | an output skid entry behind `rd_dat`: the packer advances when that entry is empty (registered); `rd_dat_ready` only moves the entry into `rd_dat` | unchanged: a beat goes to `rd_dat` whenever it is free; one beat per cycle |
| `mk_req_ready` (rn_mkgen) | `ph == PLAST` (decode of a register) | flop loaded with `ph_next == PLAST` | identical value in every cycle, no change |
| `wr_idle` (sequencer), `rd_idle` (testbench only) | AND of the channel's state bits | flop: `idle && !handshake` of the previous cycle (rise one cycle later, fall in the same cycle as before, so it is never early) | +1 cycle on each write-channel idle rise (op retirement) |

`rd_dat_valid` / `rd_dat` (flops, now with the skid entry behind them), `mk_rsp_valid` / `mk_rsp_keep` and `err` were already flops. The AXI master outputs go off chip
and are out of scope (chip-level reg-to-out slack +1.96 ns in P8d).

**Ordering (R/W guarantees kept).** Accesses are still ordered by their handshake on the engine port. A request
sitting in a skid buffer has already been handshaken, so it has to keep its place against the other channels. Each buffer entry
carries 2-bit age counters: for a read entry, one per write channel, giving the number of that channel's buffered beats that
are older; for a write entry, one per read port and one per other write channel. The counter is set at the
handshake from the other buffers' occupancy, net of what leaves in that cycle, and is decremented when the other
buffer's head leaves. Because the buffers are FIFOs, the older entries are always the first `c` entries
(head, then the skid entry). The checks are:

- a read head leaves its buffer (into the pending queue) only when it overlaps no pending write (stage A, pending line,
  open burst, burst queue until B, as before) and none of the older buffered beats;
- a write head leaves its buffer (into stage A) only when it overlaps no pending read (queue entry until its last beat),
  none of the older buffered reads, no pending write of another channel, and none of the other channels' older buffered beats.

A younger overlapping access cannot pass an older one, because it waits on the older one's buffer entry and then on its pending
entry. An older access never waits for a younger one: a younger overlapping access cannot have reached a pending set, and the
buffered-entry checks only look at older entries. So there is no deadlock. Same-cycle handshakes keep the P8c rules:
write before read, and beats of two write channels unordered, as before. mm_ovl_chk.svh is unchanged and watches the same port-level pairs.
`flush` (close open bursts) is asserted by a blocked read head or by a write head waiting on another channel.

Not changed: mm_mkport (the MK_DRAM = 1 debug stand-in, not in the chip) keeps its combinational ready. The unit
testbenches drive mm_top only through its ports.

Cost (round 1): per read port 2 x 37 payload bits + 2 x 2·NW counter bits, minus the old 37-bit request register; per write
channel 2 x 294 payload bits + 2 x 2·(NR+NW) counter bits. At full16 (NR 8, NW 4) that is about 3.0 k flops more
(mm had 79 k). Round 2 adds 256 + 1 flops per read port. Area figures come from DC (section 4).

## 3. Regressions (bit-exact vs L1, nothing weakened)

Final RTL (round 2, commit 6c3da6f; round 1 passed the same set, cycles noted where they differ). All pass, run fastest first in the p8f worktree. Baselines are the same targets run on origin/main (fc109fe) in a
detached worktree on the same machine. No testbench, vector or check was changed.

| Target | Result | Cycles vs main |
|---|---|---|
| make lint | LINT PASS: 56 files / 56 modules; Verilator -Wall 142 runs, Yosys 56, Icarus 56 | - |
| test-mem-banks | PASS (3 backends x Icarus / Verilator) | - (no mm_top) |
| test-ctrl | PASS 6/6 | identical (ct_seq on its own, no mm_top) |
| test-rng | PASS (round 1; round 2 changes only mm_rdport, which test-rng does not instantiate) | identical (5 cycle figures; rn_mkgen ready is the same value in every cycle) |
| test-mem | PASS 47/47 (mm_unit: 5 cases x 3 widths x nominal / stress, incl. `order`; integration gm / gmg / vu / lw on mm_top) | table below |
| test-e2e (reduced: 1 layer, B 2, L 32, 2 steps) | PASS: every written word (3,389,288 per step) and the results bit-exact vs the interpreter; loss 0.58404243 | step 1 10,359,277 / step 2 10,359,444 cycles vs main 10,359,455 / 10,359,660: **-178 / -216 (-0.002 %)** (round 1: 10,358,917 / 10,359,084) |

Same-cycle read / write overlap check (tb/mem/mm_ovl_chk.svh): test-mem integration 14 runs, 721,776 pairs, 0 range overlaps,
0 word overlaps; e2e 323,852 pairs, 0 / 0.

test-mem integration cycles (DRAM knobs as in tb/mem/run_tests.py):

| case | DRAM | main | P8f | delta |
|---|---|---|---|---|
| gm ep_bias_drop_resid | nominal | 6,351 | 6,353 | +2 (+0.03 %) |
| gm + rn_mkgen ep_bias_drop_resid | nominal | 6,351 | 6,352 | +1 |
| vu elem | nominal | 37,510 | 37,486 | -24 (-0.06 %) |
| vu emb | nominal | 115,935 | 115,841 | -94 (-0.08 %) |
| lw adamw_real_head | nominal | 5,329,371 | 5,329,371 | 0 |
| gm real_ffn1_256 | nominal | 2,576,252 | 2,576,252 | 0 |
| vu real_ln_fwd | nominal | 2,965,568 | 2,965,562 | -6 |
| gm ep_bias_drop_resid | stress | 11,818 | 11,824 | +6 (+0.05 %) |
| gm + rn_mkgen ep_bias_drop_resid | stress | 11,801 | 11,800 | -1 |
| vu elem | stress | 64,886 | 64,814 | -72 (-0.11 %) |
| vu emb | stress | 289,712 | 289,679 | -33 (-0.01 %) |
| gm real_ffn1_256 | +bw=100000 | 2,434,148 | 2,434,148 | 0 |
| gm real_ffn1_256 | +wlat variant | 3,841,685 | 3,841,684 | -1 |
| vu real_ln_fwd | +wlat variant | 3,250,248 | 3,248,201 | -2,047 (-0.06 %) |

(Round 1 differed only in vu emb: -22 nominal, -217 stress.) Small decreases come from the write skid: a beat no longer stalls the engine while it waits for its WAR / WAW
check or for stage A, because the two buffer entries absorb it. Small increases come from the extra cycle to stage A and
from the registered idle flag.

## 4. Synthesis and timing (TSMC 65, ss 0.9 V 125 C, pre-layout)

Server flow: `flow/tsmc65/run_p8f.sh`. The unchanged blocks (top_chip, gm16_p5.25, vu16_p5.25, lw, seq, axil and their
children) are P8d's DC runs, linked read-only. rng9, rng3 and mm are re-synthesised from this branch at 5.25 ns
(DC topographical, ss, IO budget 20 %). The flat-chip PrimeTime setup is the same as P8d's: all 17 netlists, every unit's DC SPEF,
the top's own SPEF skipped, ideal clocks.

### 4.1 Round 1 (commit 86a26a6: request skid buffers, registered idle flags, registered mk_req_ready)

| block @ 5.25 ns, ss | setup WNS (ns) | area (um2) | flops | DC wall / peak RSS |
|---|---|---|---|---|
| rng9 (rn_mkgen, 9 cores) | +0.0001 | 111,199 | 1,480 | 5 min / 2.5 GB |
| rng3 (rn_mkgen, 3 cores) | +0.0037 | 39,244 | 729 | 4 min / 2.4 GB |
| mm (own + children) | +0.0008 | 1,191,833 (P8d at 4.0 ns: 1,218,577) | 82,009 (P8d: 78,991) | 42 min / 3.7 GB |

DC integrity scan for mm: 0 unresolved references, 0 latches, check_design clean. The worst mm paths are now internal:
buffer head -> RAW compare -> pending queue, and write head -> compare -> another buffer's age counter.

Flat chip, ss, 5.25 ns: **setup WNS -0.735 ns, TNS -34.66 ns, 516 endpoints** (P8d: -2.272 / -569.9 / 476);
hold +0.050, 0 violators.

| unit (endpoints in) | P8d WNS [viol] | round 1 WNS [viol] |
|---|---|---|
| u_gm | -2.272 [11] | -0.006 [1] |
| u_mm | -1.748 [41] | -0.735 [510] |
| u_lw | -1.666 [419] | 0.000 [0] |
| u_vu | -0.119 [5] | -0.119 [5] |
| u_seq / u_axil | +1.205 / +3.507 | +1.205 / +3.507 |

The gm `wr_addr -> wr_ready -> credit` loop and all the lw paths are gone. The worst remaining path is
`u_vu/u_ld/bs_mine_reg -> (vu 3.54 ns) -> rd_dat_ready -> (mm 2.17 ns) -> u_mm/g_rd_2__u_rd/u_mf/count_reg`. It runs
from the vu loader's dat_ready, which is combinational and late in vu, through the mm_rdport packer (`o_free -> take -> FIFO pops`).
That is a combinational path entering an mm_top handshake input, still in R17's scope (mm side).

Period sweep in the same PT session (clock and IO delays scaled): 5.50 -0.485 [27], 5.75 -0.235 [7], **6.00 closes**
(WNS 0.000, the latch write path at its borrow limit). Round 1 alone therefore takes the chip from 7.75 ns to **6.0 ns**.

### 4.2 Round 2 (mm_rdport output skid entry)

The packer now advances when a registered output skid entry is empty. `rd_dat_ready` only moves that entry into
`dat`, the head register, with no added latency and one beat per cycle.

mm DC at 5.25 ns, round 2: **setup WNS -0.038 ns** (TNS -9.6, 687 endpoints), hold clean, 1,216,340 um2 (own + rng),
84,065 flops (+5,074 vs P8d's mm), integrity clean. All violators are mm-internal ordering paths (buffer head ->
RAW / WAR compare -> pending-queue load or another buffer's age counter, ~4.2 ns in mm_top glue) plus
`mk_layer -> rn_mkgen x1i_q` (a configuration input). The same logic met +0.0008 in round 1, so this is DC QoR noise
from the extra 2 k flops, not a structural path.

Flat chip, ss, round 2 (commit 6c3da6f), 5.25 ns, informative row: **setup WNS -0.119 ns, TNS -22.48 ns, 1,082
endpoints**. Hold +0.050, 0 violators.

| unit (endpoints in) | worst path at 5.25 ns | WNS [viol] | crosses mm? |
|---|---|---|---|
| u_vu | `u_seq/vu_p_reg -> u_vu/u_ld/rd1_req_addr_reg` (seq 1.41 + vu 3.71) | -0.119 [5] | no: seq > vu, out of P8f scope (PI) |
| u_mm | `u_mm/g_wr_3__u_ws/h_d_reg -> u_mm/g_rd_7__u_rs/h_c_reg` (mm 5.04, internal) | -0.058 [1,076] | no: mm-internal |
| u_gm | `u_gm/.../u_pe/dr_s_reg -> u_gm/u_drain/.../g_reg_r_reg` (gm 5.00, internal) | -0.006 [1] | no: gm-internal |
| u_lw | latch write path at its borrow limit | 0.000 [0] | - |
| u_seq / u_axil | - | +1.205 / +3.507 | - |

At 5.50 ns, re-timed in the same session, the chip **closes: WNS 0.000, 0 violators**.

P8d's cross-unit groups through mm (docs/p8_report.md section 4.2, flat chip at 7.75 ns) and how each is handled:

| group (P8d) | P8d worst path | handled by |
|---|---|---|
| gm > mm > gm | `u_ep/as_rp_ohr_reg -> u_ep/asm_cred_reg` (7.23 ns) | `wr_ready` is a flop (write skid) |
| gm > mm | `u_ep/as_rp_ohr_reg -> u_mm/g_wr_0__u_wr/a_i_reg` | `wr_addr` goes only into the skid register; the compare runs on the head |
| lw > mm > lw | `u_lw/cur_reg -> u_lw/u_ad/wr_data_reg` | `wr_ready` / `rd_req_ready` are flops |
| vu > mm | `u_vu/u_ld/bs_mine_reg -> u_mm/g_rd_2__u_rd/u_mf/count_reg` | round 2: `rd_dat_ready` only moves the output skid entry |
| lw > mm | `u_lw/cur_reg -> u_mm/g_wr_2__u_wr/a_i_reg` | as gm > mm |

### 4.3 Chip period: flat PT at 5.50 ns (commit 49feaed, same netlists as round 2)

The flat chip closes at 5.50 ns, ss: setup WNS 0.000 (the lw latch write path at its borrow limit; flop-to-flop WNS
**+0.131 ns**), TNS 0, 0 violators. Hold +0.050, 0 violators. Per unit: mm +0.192, gm 0.000, vu 0.000 (latch paths),
lw 0.000, seq +1.455, axil +3.707. PT peak RSS 7.6 GB, 50 min.

Cross-unit search (T65_XPATHS 20,000 worst endpoints with slack below 0.75 ns, grouped by the sequence of units crossed):
one group only, **u_seq > u_vu, 78 endpoints, worst +0.131 ns** (`u_seq/vu_p_reg_417_ -> u_vu/u_ld/rd1_req_addr_reg_31_`,
arrival 5.116 ns). No path that crosses mm_top has slack below 0.75 ns at 5.5 ns (P8d at 7.75 ns: five groups through mm,
worst +0.229).

| period (ss) | flat-chip setup WNS / TNS (ns) [endpoints] | limiter |
|---|---|---|
| 5.25 (round 2) | -0.119 / -22.48 [1,082] | seq > vu -0.119 (out of scope); mm internal -0.058; gm internal -0.006 |
| **5.50** | **0.000 / 0 [0]**, flop WNS +0.131 | closes |
| 5.25 (round 1) | -0.735 / -34.66 [516] | vu > mm (rd_dat_ready), fixed in round 2 |
| 6.00 (round 1) | 0.000 / 0 [0] | closed in round 1 |
| 7.75 (P8d, no fix) | closes; 5.25: -2.272 / -569.9 [476] | gm > mm > gm (wr_ready) |

Closed chip period on the 0.25 ns grid: **5.50 ns**. Even with the seq > vu path fixed, 5.25 would also need about
0.06 ns more in mm's internal ordering compare (mm DC -0.038 at 5.25 in round 2, +0.0008 in round 1).

## 5. Reproduce

```
make lint; make test-mem-banks; make compile; make test-ctrl; make test-rng; make test-mem; make test-e2e
# server (P8d's checkout linked read only for the unchanged blocks; T65_CORES 6)
cd ~/lowa_tsmc65/p8f/src/rtl || exit 1
nohup flow/tsmc65/run_p8f.sh ~/lowa_tsmc65/LoWA/src/rtl link mm > build/tsmc65/p8f_mm.log 2>&1 < /dev/null &
nohup flow/tsmc65/run_p8f.sh ~/lowa_tsmc65/LoWA/src/rtl pt > build/tsmc65/p8f_pt.log 2>&1 < /dev/null &
```

Files: `hw/mem/mm_skid.v` (new), `hw/mem/mm_top.v`, `hw/mem/mm_rdport.v`, `hw/mem/mm_wrport.v` (comments only),
`hw/rng/rn_mkgen.v`; `flow/tsmc65/run_p8f.sh` (new), `flow/tsmc65/sta/pt_unit.tcl` (`T65_PT_SWEEP`: re-time the same
PT session at further periods until setup closes). Summaries on the server: `build/tsmc65/summary/{hier_mm,hier_rng9,
hier_rng3,ptu_chip_ss_flat}.txt`.
