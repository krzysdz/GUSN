typedef enum logic [2:0] {
    NOP = 0,
    SUM_ALL = 1,
    SCALE_AND_BIAS = 2,
    SHR = 3,
    CLAMP = 4,
    WRITE = 5,
    SET_OUT_OFFSET = 6,
    FIN = 7
} nonvec_inst_t;

typedef union packed {
    logic [45:0] _empty;
    struct packed {
        logic signed [7:0] bias;
        logic [37:0] mult;
    } scale_data;
    struct packed {
        logic [39:0] _pad;
        logic [5:0] shift;
    } shift_data;
    struct packed {
        logic [29:0] _pad;
        logic signed [7:0] min;
        logic signed [7:0] max;
    } clamp_data;
    struct packed {
        logic [44:0] _pad;
        logic next_chunk;
    } write_data;
    struct packed {
        logic [37:0] _pad;
        logic signed [7:0] offset;
    } offset_data;
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
