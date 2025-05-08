`include "datatypes.svh"

module net_proc #(
    parameter int MUL_WIDTH = 27,
    parameter int NUM_CHUNKS = 35
) (
    input logic clk,
    input logic start,
    output logic done,
    output logic [3:0] max_idx_10, // max index of first 10 memory items

    input logic ext_mem_rst,
    input logic ext_mem_we,
    input logic [7:0] ext_mem_wdata
);
    localparam int InstAddrW = 13;
    localparam int SingleMacW = 24;
    localparam int AccAllW = 25;

    logic running;
    full_inst_t instruction;
    logic [InstAddrW-1:0] inst_ptr;
    (* ram_style = "block" *) logic [$bits(full_inst_t)-1:0] prog_mem[2**InstAddrW];
    initial $readmemb("prog.dat", prog_mem);

    always_ff @(posedge clk) begin
        if (start) begin
            inst_ptr <= 0;
            instruction <= '{1'b0, 1'b0, {27*8{1'b0}}, 1'b0, 1'b0, NOP, 46'd0};
        end else begin
            if (running && instruction.proc_inst != FIN) begin
                inst_ptr <= inst_ptr + 1;
                instruction <= full_inst_t'(prog_mem[inst_ptr]);
            end
        end
    end
    always_ff @(posedge clk) begin
        if (start || ext_mem_rst || instruction.proc_inst == FIN)
            running <= start;
    end

    logic signed [SingleMacW-1:0] mac_outs[27];
    logic signed [7:0] in_offset_neg;
    logic signed [7:0] out_offset;
    logic [$clog2(NUM_CHUNKS)-1:0] saved_rptr;
    logic [$clog2(NUM_CHUNKS)-1:0] mem_rptr;
    param_arr_t mem_data;
    logic signed [AccAllW-1:0] mac_sum;
    logic signed [AccAllW-1:0] final_reg;
    logic signed [AccAllW-1:0] final_op_result_c;
    logic [3:0] max_idx_10_c;

    always_comb begin
        unique case (instruction.proc_inst)
            SUM_ALL:
                final_op_result_c = mac_sum;
            SCALE_AND_BIAS:
                final_op_result_c = (final_reg + instruction.proc_data.scale_data.bias) * $signed({1'b0, instruction.proc_data.scale_data.mult}) + out_offset;
            SHR:
                final_op_result_c = final_reg >>> instruction.proc_data.shift_data.shift;
            CLAMP:
                // Max is constant 127 (i8::MAX)
                final_op_result_c = (final_reg > 8'sd127) ?
                                        8'sd127 :
                                        ((final_reg < instruction.proc_data.clamp_data.min) ?
                                            instruction.proc_data.clamp_data.min :
                                            final_reg);
                // final_op_result_c = (final_reg > instruction.proc_data.clamp_data.max) ?
                //                         instruction.proc_data.clamp_data.max :
                //                         ((final_reg < instruction.proc_data.clamp_data.min) ?
                //                             instruction.proc_data.clamp_data.min :
                //                             final_reg);
            default: final_op_result_c = final_reg;
        endcase
    end
    always_ff @(posedge clk)
        if (running)
            final_reg <= final_op_result_c;

    // This is going to have atrocious latency
    always_comb begin
        max_idx_10_c = 0;
        for (int i = 1; i < 10; ++i)
            if (mem_data[i] > mem_data[max_idx_10_c]) max_idx_10_c = i;
    end

    always_ff @(posedge clk)
        if (instruction.proc_inst == SET_OUT_OFFSET)
            out_offset <= instruction.proc_data.offset_data.offset;

    always_ff @(posedge clk) begin
        if (start) in_offset_neg <= 0;
        else if (instruction.proc_inst == WRITE && instruction.proc_data.write_data.next_chunk)
            in_offset_neg <= out_offset;
    end

    always_ff @(posedge clk) begin
        if (start) saved_rptr <= 0;
        else if (instruction.save_rptr) saved_rptr <= mem_rptr;
    end

    always_ff @(posedge clk)
        if (running && instruction.proc_inst == FIN)
            max_idx_10 <= max_idx_10_c;

    always_ff @(posedge clk)
        if (start) done <= 0;
        else if (running && instruction.proc_inst == FIN)
            done <= 1;

    sdp_mem #(
        .WRITE_WIDTH(8),
        .READ_WIDTH_MUL(27),
        .NUM_CHUNKS(35)
    ) i_data_mem (
        .clk(clk),
        .rst(ext_mem_rst),
        .wdata(ext_mem_we ? ext_mem_wdata : $unsigned(final_reg[7:0])),
        .we(ext_mem_we | (instruction.proc_inst == WRITE)),
        .w_next_chunk(~ext_mem_we & instruction.proc_data.write_data.next_chunk),
        .rp_inc(instruction.mul_en),
        .rp_load_val(saved_rptr),
        .rp_load(instruction.load_rptr),
        .rp(mem_rptr),
        .rdata(mem_data)
    );

    generate
        genvar i;
        for (i = 0; i < 27; ++i) begin : g_mac
            poor_mac #(.REGISTER_INPUTS(0), .O_WIDTH(SingleMacW), .PREADDER_SUB(1)) i_mac(
                .clk(clk),
                .en(instruction.mul_en),
                .a(mem_data[i]),
                .b(instruction.weights[i]),
                .c(in_offset_neg),
                .acc(instruction.mul_acc),
                .result(mac_outs[i])
            );
        end
    endgenerate

    adder_tree3 #(.WIDTH_IN(SingleMacW), .WIDTH_OUT(AccAllW), .N(27)) i_sum_tree(.inputs(mac_outs), .sum(mac_sum));
endmodule
