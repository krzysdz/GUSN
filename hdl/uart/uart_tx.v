`ifndef UART_TX_V
`define UART_TX_V

module uart_tx (
    input rst,
    input clk,
    input uart_clk,
    input tx_rq,
    input [7:0] data,
    output tx_busy,
    output txd
);
    reg [3:0] bit_counter;
    reg [9:0] shreg;
    reg rq_saved;

    assign tx_busy = bit_counter != 11;

    always @(posedge clk, posedge rst) begin
        if (rst) begin
            bit_counter <= 4'd11;
            shreg <= 10'b1111111111;
            rq_saved <= 0;
        end else begin
            if (uart_clk) begin
                if (tx_busy) begin
                    bit_counter <= bit_counter + 1;
                    shreg <= {1'b1, shreg[9:1]};
                end else if (rq_saved) begin
                    bit_counter <= 0;
                    shreg[0] <= 0;
                    rq_saved <= 0;
                end
            end
            if (!tx_busy && tx_rq && !rq_saved) begin
                rq_saved <= 1;
                shreg <= {^data, data, 1'b1};
            end
        end
    end

    assign txd = shreg[0];
endmodule

`endif
