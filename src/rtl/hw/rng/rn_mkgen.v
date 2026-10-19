`default_nettype none
// rn_mkgen: on-chip dropout keep-bit generator for one engine's mk channel (PLAN rulings R2 / R7,
// spec D1, docs/dataflow.md section 7).  Bit-exact contract: model/rng.py keep_mask.
// mk channel (the engines' interface, unchanged from P3-P5 / P6a): a request carries an element index
// idx (row-major index into the tensor the mask applies to); the response, in request order and with no
// ready, carries the MKW keep bits of elements idx .. idx + MKW - 1 (bit i = element idx + i).  The
// engines generate the index streams themselves (GEMM epilogue: C element index; vector unit:
// base + row * stride + 16 g; the backward pass regenerates a mask by requesting the same indices with
// the same configuration), so the generator serves any order.
// Element e uses Threefry block c0 = e >> 1, c1 = step << 8 | site << 4 | layer, key = seed, and word
// e & 1 of it; keep = (word >> 8) >= 1677722.  A request needs blocks p0 .. p0 + MKW/2 (NB = MKW/2 + 1,
// p0 = idx >> 1; the last block is used only for odd idx).
// Throughput: NCORE rn_tf pipelines evaluate NCORE blocks per cycle; a request takes NSEQ = ceil(NB /
// NCORE) cycles (mk_req_ready is high in its last one), so the channel sustains MKW / NSEQ keep bits per
// cycle: NCORE = NB = 9 gives MKW = 16 bits every cycle (the vector unit's peak, 1 request per cycle),
// NCORE = 3 gives 16 bits every 3 cycles (the 16x16 GEMM epilogue drains DW = 4 elements per cycle).
// The request must be held (valid, idx stable) until ready, as every engine's mk port does.
// Latency: request (last phase) -> response = STAGES + 2 cycles.  Configuration (seed, step, site,
// layer) is registered; it must be stable from the cycle before the first request until `idle`.
module rn_mkgen #(
    parameter MKW    = 16,                   // keep bits per response (even, >= 2)
    parameter NCORE  = 9,                    // Threefry pipelines (1 .. MKW/2 + 1)
    parameter STAGES = 3                     // rn_tf pipeline registers
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire [63:0]    seed,
    input  wire [23:0]    step,
    input  wire [3:0]     site,
    input  wire [3:0]     layer,
    input  wire           mk_req_valid,
    output reg            mk_req_ready,   // registered (P8f, PLAN R17): = (ph == PLAST)
    input  wire [31:0]    mk_req_idx,
    output reg            mk_rsp_valid,
    output reg  [MKW-1:0] mk_rsp_keep,
    output wire           idle
);
    localparam NB   = MKW / 2 + 1;           // blocks per request
    localparam NSEQ = (NB + NCORE - 1) / NCORE;
    localparam SEG  = 2 * NCORE;             // keep bits per cycle of the cores
    localparam CW   = SEG * NSEQ;            // >= MKW + 2
    localparam integer PLI = NSEQ - 1;
    localparam integer NCI = NCORE;
    localparam [3:0] PLAST = PLI[3:0];
    localparam [7:0] NC8   = NCI[7:0];

    // ---------------------------------------------------------------- configuration registers
    wire [31:0] k0_w, x1i_w, i1a_w, i1b_w, i2a_w, i2b_w, i3a_w, i3b_w;
    rn_ks u_ks (.seed(seed), .step(step), .site(site), .layer(layer), .k0(k0_w), .x1i(x1i_w),
        .i1a(i1a_w), .i1b(i1b_w), .i2a(i2a_w), .i2b(i2b_w), .i3a(i3a_w), .i3b(i3b_w));
    reg  [31:0] k0_q, x1i_q, i1a_q, i1b_q, i2a_q, i2b_q, i3a_q, i3b_q;
    always @(posedge clk) begin
        k0_q <= k0_w; x1i_q <= x1i_w; i1a_q <= i1a_w; i1b_q <= i1b_w;
        i2a_q <= i2a_w; i2b_q <= i2b_w; i3a_q <= i3a_w; i3b_q <= i3b_w;
    end

    // ---------------------------------------------------------------- issue: phase ph = 0 .. NSEQ-1
    reg  [3:0]  ph;
    reg  [7:0]  po;                          // ph * NCORE
    // P8f (PLAN R17): mk_req_ready is a flop holding (ph == PLAST), loaded with the next phase's value, so mm_top's
    // mk port has no combinational output (same value in every cycle as before, no cycle change)
    wire [3:0]  ph_n = !mk_req_valid ? ph : (mk_req_ready ? 4'd0 : ph + 4'd1);
    wire        fire = mk_req_valid && mk_req_ready;
    reg         v_s0, last_s0, odd_s0;
    reg  [3:0]  ph_s0;
    reg  [7:0]  po_s0;
    reg  [31:0] t_s0;                        // (idx >> 1) + k0
    always @(posedge clk) begin
        if (!rst_n) begin
            ph <= 4'd0; po <= 8'd0; v_s0 <= 1'b0; mk_req_ready <= (PLAST == 4'd0);
        end else begin
            v_s0 <= mk_req_valid;
            mk_req_ready <= (ph_n == PLAST);
            if (mk_req_valid) begin
                ph <= ph_n;
                if (mk_req_ready) po <= 8'd0;
                else po <= po + NC8;
            end
        end
        last_s0 <= mk_req_ready; odd_s0 <= mk_req_idx[0]; ph_s0 <= ph; po_s0 <= po;
        t_s0 <= {1'b0, mk_req_idx[31:1]} + k0_q;
    end

    // ---------------------------------------------------------------- Threefry cores
    wire [NCORE-1:0] cv;
    wire [SEG-1:0]   kb;                     // bit 2k = keep(x0 of core k), 2k+1 = keep(x1 of core k)
    genvar k;
    generate
        for (k = 0; k < NCORE; k = k + 1) begin : g_core
            localparam [7:0] K8 = k;
            wire [7:0] off = po_s0 + K8;
            /* verilator lint_off UNUSEDSIGNAL */   // the raw words are a test port of rn_tf
            wire [31:0] ox0, ox1;
            /* verilator lint_on UNUSEDSIGNAL */
            rn_tf #(.STAGES(STAGES)) u_tf (.clk(clk), .rst_n(rst_n), .in_valid(v_s0),
                .a(t_s0), .b({24'd0, off}), .x1i(x1i_q),
                .i1a(i1a_q), .i1b(i1b_q), .i2a(i2a_q), .i2b(i2b_q), .i3a(i3a_q), .i3b(i3b_q),
                .out_valid(cv[k]), .out_x0(ox0), .out_x1(ox1), .keep(kb[2*k +: 2]));
        end
    endgenerate
    // request tags travel beside the cores (same latency)
    wire        t_last, t_odd;
    wire [3:0]  t_ph;
    cm_delay #(.W(6), .N(STAGES), .RST(0)) u_tag (.clk(clk), .rst_n(rst_n),
        .d({last_s0, odd_s0, ph_s0}), .q({t_last, t_odd, t_ph}));
    wire        ov = cv[0];

    // ---------------------------------------------------------------- assembly
    wire [CW-1:0] cat;
    generate
        if (NSEQ == 1) begin : g_one
            assign cat = kb;
            /* verilator lint_off UNUSEDSIGNAL */
            wire unused_ph = ^t_ph;
            /* verilator lint_on UNUSEDSIGNAL */
        end else begin : g_seq
            reg [SEG*(NSEQ-1)-1:0] acc;          // keep bits of the earlier phases
            genvar p;
            for (p = 0; p < NSEQ - 1; p = p + 1) begin : g_acc
                localparam [3:0] P4 = p;
                always @(posedge clk)
                    if (ov && !t_last && t_ph == P4) acc[SEG*p +: SEG] <= kb;
            end
            assign cat = {kb, acc};
        end
    endgenerate
    /* verilator lint_off UNUSEDSIGNAL */   // bits beyond MKW + 1 (unused blocks of the last phase)
    wire [CW-1:0] sh = t_odd ? {1'b0, cat[CW-1:1]} : cat;
    wire unused_cv = ^cv;
    /* verilator lint_on UNUSEDSIGNAL */
    reg  [5:0] infl;                         // requests in flight
    always @(posedge clk) begin
        if (!rst_n) begin
            mk_rsp_valid <= 1'b0; infl <= 6'd0;
        end else begin
            mk_rsp_valid <= ov && t_last;
            infl <= infl + {5'd0, fire} - {5'd0, mk_rsp_valid};
        end
        if (ov && t_last) mk_rsp_keep <= sh[MKW-1:0];
    end
    assign idle = (infl == 6'd0) && !mk_req_valid;
endmodule
`default_nettype wire
