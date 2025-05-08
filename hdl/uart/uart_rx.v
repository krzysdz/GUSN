`ifndef UART_RX_V
`define UART_RX_V

module uart_rx (
    input clk,
    input nRST,
    input uart_clk,
    input rxd,
    output reg ready,
    output reg [7:0] data,
    output reg error
);
    localparam SIdle = 2'd0;
    localparam SReceive = 2'd1;
    localparam SParity = 2'd2;
    localparam SEnd = 2'd3;

    reg [1:0] state;
    reg [3:0] bits;
    reg [3:0] cnt;

    always @(posedge clk, negedge nRST) begin
        if (!nRST) begin
            data <= 0;
            error <= 0;
            state <= SIdle;
            ready <= 0;
            cnt <= 0;
            bits <= 0;
        end else if (uart_clk) begin
            if (state == SIdle) begin
                if (!rxd) begin
                    state <= SReceive;
                    cnt <= 9;
                    bits <= 0;
                    error <= 0;
                    ready <= 0;
                end
            end else if (state == SReceive) begin
                cnt <= cnt + 1;
                if (cnt == 0) begin
                    data <= {rxd, data[7:1]};
                    bits <= bits + 1;
                    if (bits == 4'd8)
                        state <= SParity;
                end
            end else if (state == SParity) begin
                cnt <= cnt + 1;
                if (cnt == 0) begin
                    error <= ^data != rxd; // ^{data, rxd}
                    state <= SEnd;
                end
            end else if (state == SEnd) begin
                cnt <= cnt + 1;
                if (cnt == 0) begin
                    error <= error || !rxd;
                    ready <= !(error || !rxd);
                    state <= SIdle;
                end
            end
        end
    end
endmodule
`endif
