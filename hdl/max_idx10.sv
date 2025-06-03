// 4-cycle machine finding index of maximum of 4 signed numbers
// uses 3 time-multiplexed comparators
module max_idx10 #(
    parameter int WIDTH = 8
) (
    input logic clk,
    input logic reset,
    input logic signed [WIDTH-1:0] inputs[10],
    input logic start,
    output logic [3:0] idx,
    output logic done
);
    typedef enum logic[1:0] {
        S1_FIRST4,
        S2_INTERMEDIATE,
        S3_LAST_IN_AND_IM_R,
        S4_RESULT
    } max_idx_state_t;

    max_idx_state_t state;
    logic signed [WIDTH-1:0] tmp_results[7];
    logic [3:0] tmp_idx[4];
    // 1st cmp
    logic signed [WIDTH-1:0] in1_a_n;
    logic signed [3:0] in1_a_d;
    logic signed [WIDTH-1:0] in1_b_n;
    logic signed [3:0] in1_b_d;
    logic signed [WIDTH-1:0] out1_n;
    logic signed [3:0] out1_d;
    // 2nd cmp
    logic signed [WIDTH-1:0] in2_a_n;
    logic signed [3:0] in2_a_d;
    logic signed [WIDTH-1:0] in2_b_n;
    logic signed [3:0] in2_b_d;
    logic signed [WIDTH-1:0] out2_n;
    logic signed [3:0] out2_d;
    // 3rd cmp
    logic signed [WIDTH-1:0] in3_a_n;
    logic signed [3:0] in3_a_d;
    logic signed [WIDTH-1:0] in3_b_n;
    logic signed [3:0] in3_b_d;
    logic signed [WIDTH-1:0] out3_n;
    logic signed [3:0] out3_d;

    always_comb begin
        in2_a_n = 0;
        in2_a_d = 0;
        in2_b_n = 0;
        in2_b_d = 0;
        in3_a_n = 0;
        in3_a_d = 0;
        in3_b_n = 0;
        in3_b_d = 0;
        unique0 case (state)
            S1_FIRST4 : begin
                in1_a_n = inputs[0];
                in1_a_d = 4'd0;
                in1_b_n = inputs[1];
                in1_b_d = 4'd1;
                in2_a_n = inputs[2];
                in2_a_d = 4'd2;
                in2_b_n = inputs[3];
                in2_b_d = 4'd3;
                in3_a_n = inputs[4];
                in3_a_d = 4'd4;
                in3_b_n = inputs[5];
                in3_b_d = 4'd5;
            end
            S2_INTERMEDIATE : begin
                in1_a_n = tmp_results[0];
                in1_a_d = tmp_idx[0];
                in1_b_n = tmp_results[1];
                in1_b_d = tmp_idx[1];
                in2_a_n = tmp_results[3];
                in2_a_d = 4'd6;
                in2_b_n = tmp_results[4];
                in2_b_d = 4'd7;
                in3_a_n = tmp_results[5];
                in3_a_d = 4'd8;
                in3_b_n = tmp_results[6];
                in3_b_d = 4'd9;
            end
            S3_LAST_IN_AND_IM_R : begin
                in1_a_n = tmp_results[0];
                in1_a_d = tmp_idx[0];
                in1_b_n = tmp_results[1];
                in1_b_d = tmp_idx[1];
                in2_a_n = tmp_results[2];
                in2_a_d = tmp_idx[2];
                in2_b_n = tmp_results[3];
                in2_b_d = tmp_idx[3];
            end
            S4_RESULT : begin
                in1_a_n = tmp_results[0];
                in1_a_d = tmp_idx[0];
                in1_b_n = tmp_results[1];
                in1_b_d = tmp_idx[1];
            end
        endcase
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= S1_FIRST4;
            done <= 0;
        end else begin
            unique case (state)
                S1_FIRST4: begin
                    if (start) begin
                        state <= S2_INTERMEDIATE;
                        done <= 0;
                        tmp_results[0] <= out1_n;
                        tmp_results[1] <= out2_n;
                        tmp_results[2] <= out3_n;
                        tmp_results[3] <= inputs[6];
                        tmp_results[4] <= inputs[7];
                        tmp_results[5] <= inputs[8];
                        tmp_results[6] <= inputs[9];
                        tmp_idx[0] <= out1_d;
                        tmp_idx[1] <= out2_d;
                        tmp_idx[2] <= out3_d;
                    end
                end
                S2_INTERMEDIATE : begin
                    state <= S3_LAST_IN_AND_IM_R;
                    tmp_results[0] <= out1_n;
                    tmp_results[1] <= out2_n;
                    tmp_results[3] <= out3_n;
                    tmp_idx[0] <= out1_d;
                    tmp_idx[1] <= out2_d;
                    tmp_idx[3] <= out3_d;
                end
                S3_LAST_IN_AND_IM_R : begin
                    state <= S4_RESULT;
                    tmp_results[0] <= out1_n;
                    tmp_results[1] <= out2_n;
                    tmp_idx[0] <= out1_d;
                    tmp_idx[1] <= out2_d;
                end
                S4_RESULT : begin
                    state <= S1_FIRST4;
                    done <= 1;
                end
                default: state <= S1_FIRST4;
            endcase
        end
    end
    always_ff @(posedge clk) begin
        if (state == S4_RESULT)
            idx <= out1_d;
    end

    cmp_ad #(.NUM_W(WIDTH), .AD_W(4)) cmp_i1(
        .a_num(in1_a_n), .a_data(in1_a_d),
        .b_num(in1_b_n), .b_data(in1_b_d),
        .o_num(out1_n), .o_data(out1_d)
    );
    cmp_ad #(.NUM_W(WIDTH), .AD_W(4)) cmp_i2(
        .a_num(in2_a_n), .a_data(in2_a_d),
        .b_num(in2_b_n), .b_data(in2_b_d),
        .o_num(out2_n), .o_data(out2_d)
    );
    cmp_ad #(.NUM_W(WIDTH), .AD_W(4)) cmp_i3(
        .a_num(in3_a_n), .a_data(in3_a_d),
        .b_num(in3_b_n), .b_data(in3_b_d),
        .o_num(out3_n), .o_data(out3_d)
    );
endmodule