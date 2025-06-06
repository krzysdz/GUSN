module top_qi8 (
    input logic clk,
    input logic rx,
    input logic rst_btn,
    output logic tx,
    output logic [3:0] last_result
);
    localparam int Rate = 921600;
`ifdef CLK_F
    localparam int ClkFreq = `CLK_F;
`else
    localparam int ClkFreq = 100_000_000;
`endif
    localparam int Delta = 64'd2**32 * 16 * Rate / ClkFreq;

    initial last_result = 4'hF;

    logic reset_btn_r;
    logic [1:0] n_start;
    logic global_reset;
    assign global_reset = reset_btn_r || n_start == 2'b01;

    logic rx_r;
    logic rx_byte_ready;
    logic rx_byte_ready_old;
    logic [7:0] rx_data;
    logic [9:0] bytes_received;
    logic byte_just_received_c;
    logic byte_just_received_r;

    logic [3:0] result;
    logic done;
    logic done_r;
    logic just_done_c;

    logic uart_rx_clk;
    logic uart_rx_clk_old;
    logic uart_tx_clk;
    logic [3:0] uart_scale_cnt;
    uart_baud bg(.CLK(clk), .nRST(!global_reset), .ACC_DELTA(Delta), .RXC(uart_rx_clk));

    assign byte_just_received_c = rx_byte_ready && !rx_byte_ready_old;
    assign just_done_c = done && !done_r;

    always_ff @(posedge clk) begin
        if (global_reset) begin
            uart_scale_cnt <= 0;
            bytes_received <= 0;
            last_result <= 4'hF;
        end else begin
            if (uart_rx_clk)
                uart_scale_cnt <= uart_scale_cnt + 1;
            if (bytes_received >= 10'd784)
                bytes_received <= 0;
            else if (byte_just_received_c)
                bytes_received <= bytes_received + 1;
        end
        reset_btn_r <= rst_btn;
        rx_r <= rx;
        rx_byte_ready_old <= rx_byte_ready;
        uart_rx_clk_old <= uart_rx_clk;
        uart_tx_clk <= uart_scale_cnt == 4'hF && uart_rx_clk && !uart_rx_clk_old;
        byte_just_received_r <= byte_just_received_c;
        done_r <= done;
        n_start <= {n_start[0], 1'b1};
        last_result <= result;
    end

    uart_rx i_rx(.clk(clk), .nRST(!global_reset), .uart_clk(uart_rx_clk), .rxd(rx_r), .ready(rx_byte_ready), .data(rx_data), .error());
    uart_tx i_tx (
        .rst(global_reset),
        .clk(clk),
        .uart_clk(uart_tx_clk),
        .tx_rq(just_done_c),
        .data(8'h30 + result),
        .tx_busy(),
        .txd(tx)
    );

    net_proc i_net(
        .clk(clk),
        .start(bytes_received == 10'd784),
        .done(done),
        .max_idx_10(result),
        .ext_mem_rst(bytes_received == 0 && byte_just_received_c),
        .ext_mem_we(byte_just_received_r),
        .ext_mem_wdata(rx_data)
    );
endmodule
