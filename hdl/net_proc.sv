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
    localparam int AccAllW = 24;
    localparam int MulW = 15; // this is width of unsigned
    localparam int PreShrW = AccAllW + MulW + 2; // 1, because mul will be signed; 1 for additions ovf

    config_t conf_data_raw[1];
    config_t conf;
    initial $readmemb("config.dat", conf_data_raw);
    assign conf = config_t'(conf_data_raw[0]);

    logic running;
    logic wait_wr;
    full_inst_t instruction;
    logic [InstAddrW-1:0] inst_ptr;
    full_inst_t prog_mem[2**InstAddrW];
    initial $readmemb("prog.dat", prog_mem);

    always_ff @(posedge clk) begin
        if (start) begin
            inst_ptr <= 0;
            instruction <= '{1'b0, 1'b0, {27*8{1'b0}}, 1'b0, 1'b0, NOP, 15'd0};
        end else begin
            if (running && !wait_wr && instruction.proc_inst != FIN) begin
                inst_ptr <= inst_ptr + 1;
                instruction <= prog_mem[inst_ptr];
            end
        end
    end
    always_ff @(posedge clk) begin
        if (start || ext_mem_rst || instruction.proc_inst == FIN)
            running <= start;
    end

    logic signed [7:0] in_offset_neg;
    // Parameters set by instructions
    logic signed [7:0] out_offset;
    logic [MulW-1:0] multiplier;
    logic [5:0] shift;
    logic signed [7:0] act_min;
    // Memory read pointer saving
    logic [$clog2(NUM_CHUNKS)-1:0] saved_rptr;
    logic [$clog2(NUM_CHUNKS)-1:0] mem_rptr;
    // Internal wires
    param_arr_t mem_data;
    logic signed [SingleMacW-1:0] mac_outs[27];
    // Final neuron processing pipeline
    logic signed [AccAllW-1:0] mac_sum; // this is already after 2 steps
    logic signed [45:0] offset_rd;
    logic signed [PreShrW-1:0] mul_no_shr;
    logic signed [46:0] mul_w_off_rd;
    logic signed [46:0] mul_shifted;
    logic signed [7:0] activated;
    logic signed [7:0] bias_ppl [3];
    logic [6:0] is_write_ppl;
    logic [6:0] last_in_layer_ppl;

    assign wait_wr = |last_in_layer_ppl;

    // Setting (most) parameters
    always_ff @(posedge clk) begin
        if (start) begin
            out_offset <= 0;
            multiplier <= 0;
            shift <= 0;
            act_min <= 0;
        end else if (running && !wait_wr) begin
            unique case (instruction.proc_inst)
                SET_OUT_OFFSET: out_offset <= instruction.proc_data.oof_d.out_offset;
                SET_MUL: multiplier <= instruction.proc_data.mul_d.mul;
                SET_SHR: begin
                    assert (instruction.proc_data.shr_d.shift <= 38)
                        else $error("shift outside [0, 38] range");
                    shift <= instruction.proc_data.shr_d.shift;
                end
                SET_MIN: act_min <= instruction.proc_data.act_d.act_min;
                default: ;
            endcase
        end
    end
    // Memory read pointer control
    always_ff @(posedge clk) begin
        if (start)
            saved_rptr <= 0;
        else if (instruction.save_rptr)
            saved_rptr <= mem_rptr;
    end
    // Shifted offset and rounding
    // this is calculated at most 1 cycle after setting shift or offset,
    // but won't be needed until 3? more and is set only once per layer (eol blocks until write)
    always_ff @(posedge clk) begin
        offset_rd <= {{38{out_offset[7]}}, out_offset, 1'b1} << shift;
    end
    // Neuron finalizing pipeline
    always_ff @(posedge clk) begin
        if (start) begin
            is_write_ppl <= 0;
            last_in_layer_ppl <= 0;
            in_offset_neg <= conf.input_quant_offset;
        end else begin
            is_write_ppl <= {is_write_ppl[5:0], instruction.proc_inst == STORE && !wait_wr};
            last_in_layer_ppl <= {last_in_layer_ppl[5:0],
                                (instruction.proc_inst == STORE && !wait_wr && instruction.proc_data.st_d.last_in_layer)};
            if (last_in_layer_ppl[6]) begin
                in_offset_neg <= out_offset;
            end
        end
        for (int n = 2; n > 0; n--) bias_ppl[n] <= bias_ppl[n-1];
        bias_ppl[0] <= instruction.proc_data.st_d.bias;
        mul_no_shr <= (mac_sum + bias_ppl[2]) * $signed({1'b0, multiplier});
        mul_w_off_rd <= mul_no_shr + offset_rd;
        mul_shifted <= $signed(mul_w_off_rd[46:1]) >>> shift;
`ifdef SIM_ONLY
        if (is_write_ppl[5]) $display("Pre-act %0d", mul_shifted);
`endif
        activated <= mul_shifted > 8'sd127 ?
                        8'sd127 : (
                            mul_shifted < act_min ?
                                act_min :
                                mul_shifted
                        );
    end

    sdp_mem #(
        .WRITE_WIDTH(8),
        .READ_WIDTH_MUL(27),
        .NUM_CHUNKS(35)
    ) i_data_mem (
        .clk(clk),
        .rst(ext_mem_rst),
        .wdata(ext_mem_we ? (ext_mem_wdata + conf.input_quant_offset) : $unsigned(activated)),
        .we(ext_mem_we | is_write_ppl[6]),
        .w_next_chunk((~ext_mem_we & last_in_layer_ppl[6]) | start),
        .rp_inc(instruction.mul_en && !wait_wr),
        .rp_load_val(saved_rptr),
        .rp_load(instruction.load_rptr),
        .rp(mem_rptr),
        .rdata(mem_data)
    );

    generate
        genvar i;
        for (i = 0; i < 27; ++i) begin : g_mac
            poor_mac #(
                .REGISTER_INPUTS(0),
                .O_WIDTH(SingleMacW),
                .PREADDER_SUB(1)
            ) i_mac (
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

    adder_tree3 #(
        .WIDTH_IN(SingleMacW),
        .WIDTH_OUT(AccAllW),
        .N(27),
        .PIPELINE(1)
    ) i_sum_tree (
        .clk(clk),
        .inputs(mac_outs),
        .sum(mac_sum)
    );

    typedef logic signed [7:0] t_mem_10_num [9:0];
    typedef logic signed [7:0] t_10_num [10];
    t_mem_10_num first_10_unpacked;
    t_10_num first_10_ok_order;
    assign first_10_unpacked = t_mem_10_num'(mem_data[9:0]);
    assign first_10_ok_order = {<<8{first_10_unpacked}};
    max_idx10 #(
        .WIDTH(8)
    ) i_max_idx (
        .clk(clk),
        .reset(start),
        .inputs(first_10_ok_order),
        .start(!wait_wr && instruction.proc_inst == FIN && !done),
        .idx(max_idx_10),
        .done(done)
    );
endmodule
