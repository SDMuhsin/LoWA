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

__CHANGE__

## 3. Regressions (bit-exact vs L1, nothing weakened)

__REGR__

## 4. Synthesis and timing (TSMC 65, ss 0.9 V 125 C, pre-layout)

__TIMING__

## 5. Reproduce

__REPRO__
