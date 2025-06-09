module lab_top (
    input logic clk,
    input logic rx,
    input logic rst_btn,
    output logic tx,
    output logic [6:0] segments
);
    logic [3:0] last_result;
    logic fast_clk;

    top_qi8 i_top(.clk(fast_clk), .rx(rx), .rst_btn(!rst_btn), .tx(tx), .last_result(last_result));
    hex_disp i_hex(.data(last_result), .seg(segments));

    altera_pll #(
        .reference_clock_frequency("50.0 MHz"),
        .operation_mode("direct"),
        .number_of_clocks(1),
        .output_clock_frequency0("100.0 MHz")
    ) i_pll(
        .rst(1'b0),
        .refclk(clk),
        .outclk({fast_clk})
    );
endmodule
