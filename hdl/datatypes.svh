typedef enum logic [2:0] {
    NOP = 3'd0,
    SET_OUT_OFFSET = 3'd1,
    SET_MUL = 3'd2,
    SET_SHR = 3'd3,
    SET_MIN = 3'd4,
    STORE = 3'd5,
    FIN = 3'd7
} nonvec_inst_t;

typedef struct packed {
    logic [6:0] _pad;
    logic signed [7:0] out_offset;
} oof_d_t;
typedef struct packed {
    logic [14:0] mul;
} mul_d_t;
typedef struct packed {
    logic [8:0] _pad;
    logic [5:0] shift;
} shr_d_t;
typedef struct packed {
    logic [6:0] _pad;
    logic signed [7:0] act_min;
} act_d_t;
typedef struct packed {
    logic [5:0] _pad;
    logic last_in_layer;
    logic signed [7:0] bias;
} st_d_t;

`ifndef QUARTUS
typedef union packed {
    oof_d_t oof_d;
    mul_d_t mul_d;
    shr_d_t shr_d;
    act_d_t act_d;
    st_d_t st_d;
} u_proc_data_t;
`endif

typedef logic signed [26:0][7:0] param_arr_t;

typedef struct packed {
    logic mul_en;
    logic mul_acc;
    param_arr_t weights;
    logic save_rptr;
    logic load_rptr;
    nonvec_inst_t proc_inst;
`ifndef QUARTUS
    u_proc_data_t proc_data;
`else
    logic [14:0] proc_data;
`endif
} full_inst_t;
