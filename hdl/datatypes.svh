typedef enum logic [2:0] {
    NOP = 0,
    SET_OUT_OFFSET = 1,
    SET_MUL = 2,
    SET_SHR = 3,
    SET_MIN = 4,
    STORE = 5,
    FIN = 7
} nonvec_inst_t;

typedef union packed {
    struct packed {
        logic [6:0] _pad;
        logic signed [7:0] out_offset;
    } oof_d;
    struct packed {
        logic [14:0] mul;
    } mul_d;
    struct packed {
        logic [8:0] _pad;
        logic [5:0] shift;
    } shr_d;
    struct packed {
        logic [6:0] _pad;
        logic signed [7:0] act_min;
    } act_d;
    struct packed {
        logic [5:0] _pad;
        logic last_in_layer;
        logic signed [7:0] bias;
    } st_d;
} u_proc_data_t;

typedef logic signed [26:0][7:0] param_arr_t;

typedef struct packed {
    logic mul_en;
    logic mul_acc;
    param_arr_t weights;
    logic save_rptr;
    logic load_rptr;
    nonvec_inst_t proc_inst;
    u_proc_data_t proc_data;
} full_inst_t;
