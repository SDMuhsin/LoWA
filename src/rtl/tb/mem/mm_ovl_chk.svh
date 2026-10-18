// Same-cycle read/write overlap check on mm_top's engine ports (P8c2, PLAN R14c).  Included inside a testbench
// module after the mm_top instance; MMO_INST names that instance (default u_mm), MMO_NR / MMO_NW its channel counts.
// Since P8c a read request is ordered when it enters its port register, so a write beat accepted in a cycle in
// which an overlapping read request is offered (rd_req_valid, not yet in the register) goes first; before P8c
// (read-first) that beat waited.  The two orderings differ exactly on such pairs.  This check counts them
// mmo_ovl counts pairs in which a word the beat writes (wr_mask) lies in the read [addr, addr + len - 1] (the
// testbench fails on mmo_ovl != 0); mmo_rng counts pairs that overlap on mm_top's conservative range rule (the
// beat's 8 words, mask ignored), and mmo_pairs every same-cycle offered read / accepted write pair (reference).
`ifndef MMO_INST
`define MMO_INST u_mm
`endif
    integer mmo_ovl = 0, mmo_rng = 0, mmo_pairs = 0;
    always @(posedge clk) begin : mmo_chk
        integer gr, gw, k;
        reg [29:0] rlo, rhi, wlo, whi;
        reg        hit;
        if (rst_n) for (gr = 0; gr < MMO_NR; gr = gr + 1) for (gw = 0; gw < MMO_NW; gw = gw + 1)
            if (`MMO_INST.rd_req_valid[gr] && `MMO_INST.wr_valid[gw] && `MMO_INST.wr_ready[gw]) begin
                rlo = `MMO_INST.rd_req_addr[32*gr+2 +: 30];
                rhi = rlo + {23'd0, `MMO_INST.rd_req_len[7*gr +: 7]} - 30'd1;
                wlo = `MMO_INST.wr_addr[32*gw+2 +: 30];
                whi = wlo + 30'd7;
                mmo_pairs = mmo_pairs + 1;
                if (rlo <= whi && wlo <= rhi) mmo_rng = mmo_rng + 1;
                hit = 1'b0;
                for (k = 0; k < 8; k = k + 1)
                    if (`MMO_INST.wr_mask[8*gw + k] && rlo <= wlo + k && wlo + k <= rhi) hit = 1'b1;
                if (hit) begin
                    if (mmo_ovl < 5) $display("MMOVL rd ch %0d [%h, %h] wr ch %0d [%h, %h] mask %b", gr, rlo, rhi, gw, wlo, whi,
                                              `MMO_INST.wr_mask[8*gw +: 8]);
                    mmo_ovl = mmo_ovl + 1;
                end
            end
    end
