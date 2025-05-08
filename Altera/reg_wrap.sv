module reg_wrap #(
    parameter int WIDTH = 32
) (
    input clk,
    input logic [WIDTH-1:0] a,
    input logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] c
);
    logic [WIDTH-1:0] a_r;
	 logic [WIDTH-1:0] b_r;
	 logic [WIDTH-1:0] c_c;

    always @(posedge clk) begin
	     a_r <= a;
		  b_r <= b;
		  c <= c_c;
	 end
	 
	 fp_mul #(.EXP_W(8), .FRAC_W(23)) i_mul(.a(a_r), .b(b_r), .product(c_c));
endmodule
