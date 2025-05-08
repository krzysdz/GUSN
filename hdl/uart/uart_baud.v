`ifndef UART_BAUD_V
`define UART_BAUD_V

module uart_baud #(
    parameter ACC_W = 32
) (
    input nRST,
    input [ACC_W-1:0] ACC_DELTA,
    input CLK,
    output RXC
);
    reg [ACC_W-1:0] acc;
    reg co;

    assign RXC = co;

    always @(posedge CLK, negedge nRST) begin
        if (!nRST) begin
            acc <= 0;
            co <= 0;
        end else
            {co, acc} <= acc + ACC_DELTA;
    end
endmodule
`endif
