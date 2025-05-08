module expwrap(
    input logic clk,
    input logic rst,
    input logic sth,
    input logic we, // write and increment write pointer
    input logic w_next_chunk, // move write pointer to next chunk (when current isn't filled, but starting new layer)
    input logic rp_inc,
    input logic [4:0] rp_load_val,
    input logic rp_load,
    output logic o
);
    logic [7:0] wdata;
    logic [27*8-1:0] rdata_r;
    logic [27*8-1:0] rdata;
    
    sdp_mem tm(.*);
    
    always_ff @(posedge clk) begin
        if (rst) wdata <= 0;
        else wdata <= {wdata[6:0], sth};
    end
            
    always_ff @(posedge clk) begin
        if (rst) rdata_r <= 0;
        else begin
            if (rp_inc || rp_load)
                rdata_r <= rdata;
            else
                rdata_r <= {rdata_r[27*8-2:0], rdata_r[27*8-1]}; 
        end
    end
    
    assign o = rdata_r[0];
endmodule
