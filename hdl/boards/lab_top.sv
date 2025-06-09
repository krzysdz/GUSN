module lab_top (
    input logic clk,
    input logic rx,
    input logic rst_btn,
    output logic tx,
    output logic [6:0] segments
);
    logic [3:0] last_result;

    top_qi8 i_top(.clk(clk), .rx(rx), .rst_btn(rst_btn), .tx(tx), .last_result(last_result));
    hex_disp i_hex(.data(last_result), .seg(segments));
endmodule
