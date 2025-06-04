// This MAC has (optionally) registered inputs and supports no pipelining (because that's not available on Cyclone V)
// Performed operation is prev_result*acc+(A+C)*B
// The use_dsp attribute is necessary for Vivado to infer DSP usage. A pipelined version works without it.
(* use_dsp = "yes" *) module poor_mac #(
    parameter bit REGISTER_INPUTS = 0,
    parameter int A_WIDTH = 8,
    parameter int B_WIDTH = 8,
    parameter int C_WIDTH = 8,
    parameter int O_WIDTH = 22, // max(A_WIDTH, C_WIDTH) + 1 + B_WIDTH + clog2(ceil(28**2 / 27))
    parameter logic PREADDER_SUB = 0
) (
    input logic clk,
    input logic en,
    input logic signed [A_WIDTH-1:0] a,
    input logic signed [B_WIDTH-1:0] b,
    input logic signed [C_WIDTH-1:0] c,
    input logic acc,
    output logic signed [O_WIDTH-1:0] result
);
    // Internal signals (registered or not)
    logic                      s_en;
    logic signed [A_WIDTH-1:0] s_a;
    logic signed [B_WIDTH-1:0] s_b;
    logic signed [C_WIDTH-1:0] s_c;
    logic                      s_acc;
    // Choose between accumulate and just multiplication result. Quartus requires it to be a separate wire.
    logic signed [O_WIDTH-1:0] acc_src_c;
    assign acc_src_c = s_acc ? result : 0;

    generate
        if (REGISTER_INPUTS) begin : g_reg_in
            logic                      en_r;
            logic signed [A_WIDTH-1:0] a_r;
            logic signed [B_WIDTH-1:0] b_r;
            logic signed [C_WIDTH-1:0] c_r;
            logic                      acc_r;

            assign s_en = en_r;
            assign s_a = a_r;
            assign s_b = b_r;
            assign s_c = c_r;
            assign s_acc = acc_r;

            always_ff @(posedge clk) begin
                en_r <= en;
                a_r <= a;
                b_r <= b;
                c_r <= c;
                acc_r <= acc;
            end
        end else begin : g_no_reg_in
            assign s_en = en;
            assign s_a = a;
            assign s_b = b;
            assign s_c = c;
            assign s_acc = acc;
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (s_en) begin
            if (PREADDER_SUB) begin : g_preadder_subtracts
                result <= acc_src_c + ((s_a - s_c) * s_b);
            end else begin : g_preadder_adds
                result <= acc_src_c + ((s_a + s_c) * s_b);
            end
`ifdef SIM_ONLY
            // b == 0 check is for simulation only, to stop propagating X. Anything times 0 is 0, but in Verilog simulation x * 0 is x.
            if (s_b == 0) result <= acc_src_c;
`endif
        end
    end
endmodule
