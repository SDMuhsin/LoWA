`default_nettype none
// gm_loader: fetches the X and W tiles of a block from the memory side and writes them into the
// feeders' buffers in the lane layouts of gm_xfeed / gm_wfeed (docs/p3_report.md section 2).
// Tile requests come from gm_ctrl (valid/ready, in order; a FIFO of depth 4 lets the loader run
// one block ahead of the array).  Per tile the issuer sends row bursts to rd0 (in order):
//   ROW mode: one burst per matrix row i (addr = base + i*rs, kt words):   X: i = row, W: i = W row n
//   TRN mode: one burst per k        (addr = base + k*rs, n words):        the DMA "transpose"
// and the consumer, which mirrors the same sequence, routes the 8 words of every returned beat to
// their lanes (always 8 distinct lanes by the layouts); words beyond the burst length are masked.
// The beats of a tile are written only once the target buffer is free (x_busy / w_busy = the
// feeder is reading it or the controller has not released the block), which is what lets the
// issuer request the next block's tiles early (the memory side queues the data).  tile_done pulses (with the kind / buffer) when the last beat was written;
// W tiles are requested before X tiles so that the LoWA merge engine (P5) gets the W buffer
// while the X tile streams in.
module gm_loader #(
    parameter QW     = 2,                   // log2 LAMBDA
    parameter KW     = 6,                   // log2 KC
    parameter XL     = 8,
    parameter XLW    = 3,
    parameter XAW    = 9,                   // X lane address width
    parameter RW     = 4,                   // log2 ROWS
    parameter MTW    = 2,                   // log2 (BR/ROWS)
    parameter WLW    = 4,                   // log2 WL
    parameter WAW    = 8,                   // W lane address width (KW + QW)
    parameter NW     = 7                    // width of n_rows / n_cols counts (<= 64 -> 7)
) (
    input  wire              clk,
    input  wire              rst_n,
    // tile requests from the controller
    input  wire              tq_valid,
    output wire              tq_ready,
    input  wire              tq_kind,       // 0 = X, 1 = W
    input  wire              tq_buf,
    input  wire              tq_trn,
    input  wire [31:0]       tq_addr,
    input  wire [31:0]       tq_rs,
    input  wire [NW-1:0]     tq_n,          // rows (X) / W rows (W) in the tile, 1 .. BR / BC
    input  wire [KW:0]       tq_kt,         // 1 .. KC
    // memory read channel 0
    output reg               rd_req_valid,
    input  wire              rd_req_ready,
    output reg  [31:0]       rd_req_addr,
    output reg  [6:0]        rd_req_len,    // words, 1 .. 64
    input  wire              rd_dat_valid,
    output wire              rd_dat_ready,
    input  wire [255:0]      rd_dat,
    // buffer status / writes
    input  wire [1:0]        x_busy,
    input  wire [1:0]        w_busy,
    output wire              xw_valid,
    output wire              xw_buf,
    output wire [XL*XLW-1:0] xw_lane,
    output wire [XL*XAW-1:0] xw_addr,
    output wire [XL*32-1:0]  xw_data,
    output wire [XL-1:0]     xw_we,
    output wire              ww_valid,
    output wire              ww_buf,
    output wire [8*WLW-1:0]  ww_lane,
    output wire [8*WAW-1:0]  ww_addr,
    output wire [8*32-1:0]   ww_data,
    output wire [7:0]        ww_we,
    output reg               tile_done,
    output reg               tile_done_kind,
    output reg               tile_done_buf
);
    // ------------------------------------------------------------------ tile FIFO (depth 4)
    localparam TW = 1 + 1 + 1 + 32 + 32 + NW + KW + 1;
    reg [TW-1:0] tf [0:3];
    reg [2:0]    tf_wp, tf_rp;              // issuer read pointer
    reg [2:0]    tf_cp;                     // consumer read pointer
    wire tf_full = (tf_wp - tf_cp) == 3'd4;
    assign tq_ready = ~tf_full;
    wire tq_fire = tq_valid & tq_ready;
    always @(posedge clk) begin
        if (!rst_n) tf_wp <= 3'd0;
        else if (tq_fire) begin
            tf[tf_wp[1:0]] <= {tq_kind, tq_buf, tq_trn, tq_addr, tq_rs, tq_n, tq_kt};
            tf_wp <= tf_wp + 1'b1;
        end
    end

    // ------------------------------------------------------------------ issuer
    wire          i_avail = (tf_rp != tf_wp);
    wire [TW-1:0] i_t = tf[tf_rp[1:0]];
    wire          i_trn = i_t[TW-3];
    wire [31:0]   i_base = i_t[TW-4 -: 32];
    wire [31:0]   i_rs   = i_t[TW-36 -: 32];
    wire [NW-1:0] i_n    = i_t[KW+1 +: NW];
    wire [KW:0]   i_kt   = i_t[KW:0];
    reg           i_act;
    reg  [31:0]   i_addr;
    reg  [NW-1:0] i_cnt;                    // bursts issued so far
    wire [NW-1:0] i_nreq = i_trn ? {{(NW-KW-1){1'b0}}, i_kt} : i_n;
    wire [6:0]    i_len  = i_trn ? {{(7-NW){1'b0}}, i_n} : {{(7-KW-1){1'b0}}, i_kt};
    wire          i_fire = rd_req_valid & rd_req_ready;
    always @(posedge clk) begin
        if (!rst_n) begin
            i_act <= 1'b0; rd_req_valid <= 1'b0; tf_rp <= 3'd0; i_cnt <= {NW{1'b0}};
        end else begin
            if (!i_act) begin
                if (i_avail) begin
                    i_act <= 1'b1; i_addr <= i_base; i_cnt <= {NW{1'b0}};
                    rd_req_valid <= 1'b1; rd_req_addr <= i_base; rd_req_len <= i_len;
                end
            end else if (i_fire) begin
                /* verilator lint_off WIDTHEXPAND */
                if (i_cnt == i_nreq - 1'b1) begin
                /* verilator lint_on WIDTHEXPAND */
                    i_act <= 1'b0; rd_req_valid <= 1'b0; tf_rp <= tf_rp + 1'b1;
                end else begin
                    i_cnt <= i_cnt + 1'b1;
                    i_addr <= i_addr + i_rs;
                    rd_req_addr <= i_addr + i_rs;
                end
            end
        end
    end

    // ------------------------------------------------------------------ consumer
    wire          c_avail = (tf_cp != tf_wp);
    wire [TW-1:0] c_t   = tf[tf_cp[1:0]];
    wire          c_kind = c_t[TW-1];
    wire          c_buf  = c_t[TW-2];
    wire          c_trn  = c_t[TW-3];
    wire [NW-1:0] c_n    = c_t[KW+1 +: NW];
    wire [KW:0]   c_kt   = c_t[KW:0];
    wire [NW-1:0] c_nreq = c_trn ? {{(NW-KW-1){1'b0}}, c_kt} : c_n;
    wire [6:0]    c_len  = c_trn ? {{(7-NW){1'b0}}, c_n} : {{(7-KW-1){1'b0}}, c_kt};
    /* verilator lint_off UNUSEDSIGNAL */   // (len + 7) / 8 <= 8 fits in 4 bits
    wire [6:0]    c_len7 = (c_len + 7'd7) >> 3;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [3:0]    c_nb   = c_len7[3:0];                  // beats per burst (<= 8)
    reg  [NW-1:0] c_i;                      // burst index
    reg  [3:0]    c_b;                      // beat index within the burst
    // explicit 2:1 muxes rather than w_busy[c_buf] / x_busy[c_buf]: Icarus 13 never re-evaluates a
    // continuous bit-select whose index was X at time 0 (c_buf comes from the un-reset tile FIFO),
    // so rd_dat_ready stayed X for the whole run under 4-state simulation (docs/p3_report.md section 7)
    wire          w_free   = c_buf ? ~w_busy[1] : ~w_busy[0];
    wire          x_free   = c_buf ? ~x_busy[1] : ~x_busy[0];
    wire          buf_free = c_kind ? w_free : x_free;
    wire          c_fire = rd_dat_valid & c_avail & buf_free;
    assign rd_dat_ready = c_avail & buf_free;
    /* verilator lint_off WIDTHEXPAND */
    wire          c_last_b = (c_b == c_nb - 1'b1);
    wire          c_last_i = (c_i == c_nreq - 1'b1);
    /* verilator lint_on WIDTHEXPAND */
    always @(posedge clk) begin
        if (!rst_n) begin
            tf_cp <= 3'd0; c_i <= {NW{1'b0}}; c_b <= 4'd0; tile_done <= 1'b0;
        end else begin
            tile_done <= 1'b0;
            if (c_fire) begin
                if (c_last_b) begin
                    c_b <= 4'd0;
                    if (c_last_i) begin
                        c_i <= {NW{1'b0}};
                        tf_cp <= tf_cp + 1'b1;
                        tile_done <= 1'b1; tile_done_kind <= c_kind; tile_done_buf <= c_buf;
                    end else begin
                        c_i <= c_i + 1'b1;
                    end
                end else begin
                    c_b <= c_b + 1'b1;
                end
            end
        end
    end

    // word j of the beat: its (row/k) coordinates and lane / address in each layout
    genvar j;
    generate
        for (j = 0; j < 8; j = j + 1) begin : g_w
            // position along the burst
            wire [6:0]  pos = {c_b, 3'd0} + j;
            wire        inb = (pos < c_len);
            // X layouts: ROW: r = c_i, k = pos;  TRN: k = c_i, r = pos
            /* verilator lint_off UNUSEDSIGNAL */   // bit 6: positions are < 64 inside a tile
            wire [6:0]  xr = c_trn ? pos : {{(7-NW){1'b0}}, c_i};
            wire [6:0]  xk = c_trn ? {{(7-NW){1'b0}}, c_i} : pos;
            /* verilator lint_on UNUSEDSIGNAL */
            /* verilator lint_off WIDTHTRUNC */
            wire [XLW-1:0] xlane = xk[XLW-1:0] + xr[XLW-1:0];
            wire [MTW-1:0] xmt = xr[RW +: MTW];
            wire [RW-1:0]  xrr = xr[RW-1:0];
            wire [KW-XLW-1:0] xkh = xk[KW-1:XLW];
            /* verilator lint_on WIDTHTRUNC */
            assign xw_lane[j*XLW +: XLW] = xlane;
            assign xw_addr[j*XAW +: XAW] = {xmt, xrr, xkh};
            // W layouts: ROW: n = c_i, k = pos;  TRN: k = c_i, n = pos;  c = n / LAMBDA, q = n mod LAMBDA
            /* verilator lint_off UNUSEDSIGNAL */   // bit 6 / high bits of wc: positions < 64
            wire [6:0]  wn = c_trn ? pos : {{(7-NW){1'b0}}, c_i};
            wire [6:0]  wk = c_trn ? {{(7-NW){1'b0}}, c_i} : pos;
            wire [6:0]  wc = wn >> QW;
            /* verilator lint_on UNUSEDSIGNAL */
            wire [QW-1:0] wq = wn[QW-1:0];
            /* verilator lint_off WIDTHTRUNC */
            wire [WLW-1:0] wlane = wc[WLW-1:0] + wk[WLW-1:0] + ({{(WLW-QW){1'b0}}, wq} << (WLW - QW));
            wire [KW-1:0]  wkk = wk[KW-1:0];
            /* verilator lint_on WIDTHTRUNC */
            assign ww_lane[j*WLW +: WLW] = wlane;
            assign ww_addr[j*WAW +: WAW] = {wkk, wq};
            assign xw_data[j*32 +: 32] = rd_dat[j*32 +: 32];
            assign ww_data[j*32 +: 32] = rd_dat[j*32 +: 32];
            assign xw_we[j] = inb;
            assign ww_we[j] = inb;
        end
    endgenerate
    assign xw_valid = c_fire & ~c_kind;
    assign ww_valid = c_fire &  c_kind;
    assign xw_buf = c_buf;
    assign ww_buf = c_buf;
endmodule
`default_nettype wire
