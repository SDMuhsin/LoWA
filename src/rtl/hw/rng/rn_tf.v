`default_nettype none
// rn_tf: one Threefry-2x32-13 block (Random123 threefry2x32, 13 rounds), fully unrolled and pipelined,
// one block per cycle, followed by the dropout keep compare of both output words (model/rng.py:
// keep = (word >> 8) >= DROP_THRESH = 1677722).
// Datapath (14 units, all ARX, no multiplier):
//   unit 0       x0 = a + b (a + b = c0 + k0; rn_mkgen passes (idx >> 1) + k0 and the pair offset),
//                x1 = x1i (= c1 + k1, precomputed by the caller)
//   unit 1 + r   round r = 0..12: x0 += x1; x1 = rotl(x1, ROT[r mod 8]) ^ x0;
//                after rounds 3 / 7 / 11 the key injection j = 1 / 2 / 3: x0 += ija, x1 += ijb
//                (ija = ks[j mod 3], ijb = ks[(j+1) mod 3] + j, precomputed by the caller, rn_ks)
//   the keep compare follows unit 13 in the same stage.
// Cut points: STAGES (1..14) registers, the one after unit u is present when
// floor((u+1) * STAGES / 14) > floor(u * STAGES / 14), i.e. evenly spread; the last is always at the
// output.  Latency = STAGES.  The injection words must be stable while a block is in flight (the key
// is a per-stream constant in rn_mkgen).  ss_125C_3v00 slack at 40 ns (rn_mkgen, NCORE 9): STAGES 3 +12.0 ns (default), 2 +2.5 ns.
module rn_tf #(
    parameter STAGES = 3
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] x1i,
    input  wire [31:0] i1a, i1b, i2a, i2b, i3a, i3b,
    output wire        out_valid,
    output wire [31:0] out_x0,
    output wire [31:0] out_x1,
    output wire [1:0]  keep                  // {keep(x1), keep(x0)}
);
    localparam [23:0] THRESH = 24'd1677722;
    // s0 / s1 / sv: inputs of unit u at [u]; [14] = after unit 13
    wire [32*15-1:0] s0, s1;
    wire [14:0]      sv;
    wire [2*15-1:0]  sk;                     // keep bits (meaningful at [14] only)
    assign s0[31:0] = a;                     // unit 0 adds b
    assign s1[31:0] = x1i;
    assign sv[0]    = in_valid;
    assign sk[1:0]  = 2'b00;
    genvar u;
    generate
        for (u = 0; u < 14; u = u + 1) begin : g_u
            wire [31:0] p0 = s0[32*u +: 32];
            wire [31:0] p1 = s1[32*u +: 32];
            wire [31:0] n0, n1;
            if (u == 0) begin : g_init
                assign n0 = p0 + b;
                assign n1 = p1;
            end else begin : g_round
                localparam integer R   = u - 1;
                localparam integer RM  = R % 8;
                localparam integer ROT = (RM == 0) ? 13 : (RM == 1) ? 15 : (RM == 2) ? 26 : (RM == 3) ? 6 :
                                         (RM == 4) ? 17 : (RM == 5) ? 29 : (RM == 6) ? 16 : 24;
                wire [31:0] r0 = p0 + p1;
                wire [31:0] r1 = {p1[31-ROT:0], p1[31:32-ROT]} ^ r0;
                if (R == 3) begin : g_inj1
                    assign n0 = r0 + i1a;
                    assign n1 = r1 + i1b;
                end else if (R == 7) begin : g_inj2
                    assign n0 = r0 + i2a;
                    assign n1 = r1 + i2b;
                end else if (R == 11) begin : g_inj3
                    assign n0 = r0 + i3a;
                    assign n1 = r1 + i3b;
                end else begin : g_plain
                    assign n0 = r0;
                    assign n1 = r1;
                end
            end
            wire [1:0] nk;
            if (u == 13) begin : g_keep
                assign nk = {n1[31:8] >= THRESH, n0[31:8] >= THRESH};
            end else begin : g_nokeep
                assign nk = 2'b00;
            end
            if (((u + 1) * STAGES) / 14 > (u * STAGES) / 14) begin : g_cut
                reg [31:0] q0, q1;
                reg [1:0]  qk;
                reg        qv;
                always @(posedge clk) begin
                    if (!rst_n) qv <= 1'b0;
                    else        qv <= sv[u];
                    q0 <= n0; q1 <= n1; qk <= nk;
                end
                assign s0[32*(u+1) +: 32] = q0;
                assign s1[32*(u+1) +: 32] = q1;
                assign sv[u+1] = qv;
                assign sk[2*(u+1) +: 2] = qk;
            end else begin : g_wire
                assign s0[32*(u+1) +: 32] = n0;
                assign s1[32*(u+1) +: 32] = n1;
                assign sv[u+1] = sv[u];
                assign sk[2*(u+1) +: 2] = nk;
            end
        end
    endgenerate
    assign out_valid = sv[14];
    assign out_x0    = s0[32*14 +: 32];
    assign out_x1    = s1[32*14 +: 32];
    assign keep      = sk[2*14 +: 2];
    /* verilator lint_off UNUSEDSIGNAL */   // the keep slots of the inner units are constant 0
    wire unused_sk = ^sk[2*14-1:0];
    /* verilator lint_on UNUSEDSIGNAL */
endmodule
`default_nettype wire
