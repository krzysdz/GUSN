module own_top (
    input logic clk,
    input logic rx,
    input logic rst_btn,
    output logic tx,
    output logic [3:0] d0_anodes,
    output logic [7:0] d0_segments
);
    logic [3:0] last_result;
    logic [6:0] segments;

    top_qi8 i_top(.clk(clk), .rx(rx), .rst_btn(rst_btn), .tx(tx), .last_result(last_result));
    hex_disp i_hex(.data(last_result), .seg(segments));

    assign d0_anodes = 4'b1110;
    assign d0_segments = {1'b1, segments};
endmodule
