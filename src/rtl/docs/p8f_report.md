# P8f: registered mm_top handshakes (PLAN R17)

Status: in progress (results land below as they finish).

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
| `mk_req_ready` (rn_mkgen) | `ph == PLAST` (decode of a register) | flop loaded with `ph_next == PLAST` | identical value in every cycle, no change |
| `wr_idle` (sequencer), `rd_idle` (testbench only) | AND of the channel's state bits | flop: `idle && !handshake` of the previous cycle (rise one cycle later, fall in the same cycle as before, so it is never early) | +1 cycle on each write-channel idle rise (op retirement) |

`rd_dat_valid` / `rd_dat`, `mk_rsp_valid` / `mk_rsp_keep` and `err` were already flops. The AXI master outputs go off chip
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

Cost: per read port 2 x 37 payload bits + 2 x 2·NW counter bits, minus the old 37-bit request register; per write
channel 2 x 294 payload bits + 2 x 2·(NR+NW) counter bits. At full16 (NR 8, NW 4) that is about 3.0 k flops more
(mm had 79 k). Area figures come from DC (section 4).

## 3. Regressions (bit-exact vs L1, nothing weakened)

__REGR__

## 4. Synthesis and timing (TSMC 65, ss 0.9 V 125 C, pre-layout)

__TIMING__

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
