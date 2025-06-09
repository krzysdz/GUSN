// Emulated asymmetric SDP - read is a multiple of write, but not necessarily a power of 2
module sdp_mem #(
    parameter int WRITE_WIDTH = 8,
    parameter int READ_WIDTH_MUL = 27,
    parameter int NUM_CHUNKS = 35
) (
    input logic clk,
    input logic rst,
    input logic [WRITE_WIDTH-1:0] wdata,
    input logic we, // write and increment write pointer
    input logic w_next_chunk, // move write pointer to next chunk (when current isn't filled, but starting new layer)
    input logic rp_inc,
    input logic [$clog2(NUM_CHUNKS)-1:0] rp_load_val,
    input logic rp_load,
    output logic [$clog2(NUM_CHUNKS)-1:0] rp,
    output logic [READ_WIDTH_MUL*WRITE_WIDTH-1:0] rdata
);
    logic [$clog2(NUM_CHUNKS)-1:0] rp_c;
    logic [$clog2(NUM_CHUNKS)-1:0] wp_chunk;
    logic [READ_WIDTH_MUL-1:0] wp_byte;

`ifndef QUARTUS
    logic [READ_WIDTH_MUL*WRITE_WIDTH-1:0] data[NUM_CHUNKS];

    genvar i;
    for (i = 0; i < READ_WIDTH_MUL; ++i) begin : g_byte_write
        always_ff @(posedge clk) begin
            if (we)
                if (wp_byte[i])
                    data[wp_chunk][i*WRITE_WIDTH +: WRITE_WIDTH] <= wdata;
        end
    end
`else
    (* ramstyle = "no_rw_check" *) logic [READ_WIDTH_MUL-1:0][WRITE_WIDTH-1:0] data[NUM_CHUNKS];
    always_ff @(posedge clk) begin
        if (we) begin
            if (READ_WIDTH_MUL > 0) if (wp_byte[0]) data[wp_chunk][0] <= wdata;
            if (READ_WIDTH_MUL > 1) if (wp_byte[1]) data[wp_chunk][1] <= wdata;
            if (READ_WIDTH_MUL > 2) if (wp_byte[2]) data[wp_chunk][2] <= wdata;
            if (READ_WIDTH_MUL > 3) if (wp_byte[3]) data[wp_chunk][3] <= wdata;
            if (READ_WIDTH_MUL > 4) if (wp_byte[4]) data[wp_chunk][4] <= wdata;
            if (READ_WIDTH_MUL > 5) if (wp_byte[5]) data[wp_chunk][5] <= wdata;
            if (READ_WIDTH_MUL > 6) if (wp_byte[6]) data[wp_chunk][6] <= wdata;
            if (READ_WIDTH_MUL > 7) if (wp_byte[7]) data[wp_chunk][7] <= wdata;
            if (READ_WIDTH_MUL > 8) if (wp_byte[8]) data[wp_chunk][8] <= wdata;
            if (READ_WIDTH_MUL > 9) if (wp_byte[9]) data[wp_chunk][9] <= wdata;
            if (READ_WIDTH_MUL > 10) if (wp_byte[10]) data[wp_chunk][10] <= wdata;
            if (READ_WIDTH_MUL > 11) if (wp_byte[11]) data[wp_chunk][11] <= wdata;
            if (READ_WIDTH_MUL > 12) if (wp_byte[12]) data[wp_chunk][12] <= wdata;
            if (READ_WIDTH_MUL > 13) if (wp_byte[13]) data[wp_chunk][13] <= wdata;
            if (READ_WIDTH_MUL > 14) if (wp_byte[14]) data[wp_chunk][14] <= wdata;
            if (READ_WIDTH_MUL > 15) if (wp_byte[15]) data[wp_chunk][15] <= wdata;
            if (READ_WIDTH_MUL > 16) if (wp_byte[16]) data[wp_chunk][16] <= wdata;
            if (READ_WIDTH_MUL > 17) if (wp_byte[17]) data[wp_chunk][17] <= wdata;
            if (READ_WIDTH_MUL > 18) if (wp_byte[18]) data[wp_chunk][18] <= wdata;
            if (READ_WIDTH_MUL > 19) if (wp_byte[19]) data[wp_chunk][19] <= wdata;
            if (READ_WIDTH_MUL > 20) if (wp_byte[20]) data[wp_chunk][20] <= wdata;
            if (READ_WIDTH_MUL > 21) if (wp_byte[21]) data[wp_chunk][21] <= wdata;
            if (READ_WIDTH_MUL > 22) if (wp_byte[22]) data[wp_chunk][22] <= wdata;
            if (READ_WIDTH_MUL > 23) if (wp_byte[23]) data[wp_chunk][23] <= wdata;
            if (READ_WIDTH_MUL > 24) if (wp_byte[24]) data[wp_chunk][24] <= wdata;
            if (READ_WIDTH_MUL > 25) if (wp_byte[25]) data[wp_chunk][25] <= wdata;
            if (READ_WIDTH_MUL > 26) if (wp_byte[26]) data[wp_chunk][26] <= wdata;
            if (READ_WIDTH_MUL > 27) if (wp_byte[27]) data[wp_chunk][27] <= wdata;
            if (READ_WIDTH_MUL > 28) if (wp_byte[28]) data[wp_chunk][28] <= wdata;
            if (READ_WIDTH_MUL > 29) if (wp_byte[29]) data[wp_chunk][29] <= wdata;
            if (READ_WIDTH_MUL > 30) if (wp_byte[30]) data[wp_chunk][30] <= wdata;
            if (READ_WIDTH_MUL > 31) if (wp_byte[31]) data[wp_chunk][31] <= wdata;
            if (READ_WIDTH_MUL > 32) if (wp_byte[32]) data[wp_chunk][32] <= wdata;
            if (READ_WIDTH_MUL > 33) if (wp_byte[33]) data[wp_chunk][33] <= wdata;
        end
    end
`endif
    always_comb begin
        rp_c = rp;
        if (rp_load) begin
            rp_c = rp_load_val;
        end else if (rp_inc) begin
            rp_c = (rp == NUM_CHUNKS-1) ? 0 : rp + 1;
        end
    end
    always_ff @(posedge clk) begin
        if (rst) begin
            rp <= 0;
            wp_chunk <= 0;
            wp_byte <= 1;
        end else begin
            if (w_next_chunk || (we && wp_byte[READ_WIDTH_MUL-1])) begin
                wp_chunk <= (wp_chunk == NUM_CHUNKS-1) ? 0 : wp_chunk + 1;
                wp_byte <= 1;
            end else if (we) begin
                wp_byte <= wp_byte << 1;
            end

            rp <= rp_c;
            rdata <= data[rp_c];
        end
    end

`ifdef SIM_ONLY
    logic [$clog2(READ_WIDTH_MUL+1)-1:0] byte_idx;
    assign byte_idx = $clog2(wp_byte) + 1;
    always_ff @(posedge clk) begin
        if (we) begin
            $display("[%0t] Writing chunk %0d (%0d/%0d): %0d", $time, wp_chunk, byte_idx, READ_WIDTH_MUL, $signed(wdata));
        end
    end
`endif
endmodule
