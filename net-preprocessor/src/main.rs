#[allow(
    non_snake_case,
    unsafe_op_in_unsafe_fn,
    unused_imports,
    unused_variables
)]
#[path = "../target/flatbuffers/schema_generated.rs"]
mod schema_flatbuffers;

use num_traits::{Float, PrimInt, ToPrimitive, Unsigned};
use schema_flatbuffers::tflite::Tensor;
use schema_flatbuffers::tflite::{self, ActivationFunctionType};
use std::collections::VecDeque;
use std::fs::{self};
use std::io::Read;

#[derive(Debug)]
struct FullyConnectedLayer {
    weights: Vec<i8>,
    /// In TensorFlow these are i32, but really in the data they fit in i8
    biases: Vec<i8>,
    /// input_offset is negated zero point - range [-127, 128], so let's store non-negated and subtract in Verilog (pre-adders can subtract),
    /// actually, since it is just previous output offset *-1, it can be omitted from instructions
    input_offset_neg: i8,
    /// Output offset if it has changed from previous one
    output_offset: Option<i8>,
    /// This is really u15 (logic [14:0])
    multiplier: u16,
    /// Shift right, range [0, 38], then 1 more has to be performed and rounding (LSB) added
    shift: u8,
    /// Quantized INT8 ReLU (NOT ReLU6 or ReLUN1To1) activation:
    /// - clamp between min and max
    /// - max is i8::MAX
    /// - min is max((0 / output.scale) + output.zero_point, i8::MIN) => max(output.zero_point, -128)
    /// Details in (read these 3 functions in bottom to top order)
    /// https://github.com/tensorflow/tflite-micro/blob/c3cded749ed601cd9c2868c61afcd871ee5b0844/tensorflow/lite/kernels/kernel_util.cc#L344-L411
    act_min: Option<i8>,
}

impl FullyConnectedLayer {
    /// Number of layer outputs
    fn outputs_len(&self) -> usize {
        self.biases.len()
    }

    /// Number of layer inputs (outputs from previous layer)
    fn inputs_len(&self) -> usize {
        assert_eq!(
            self.weights.len() % self.outputs_len(),
            0,
            "Number of weights must be inputs * outputs"
        );
        self.weights.len() / self.outputs_len()
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum NonvecInst {
    /// do nothing
    Nop,
    /// set offset that will be added to outputs in current layer and subtracted from inputs of the next one \
    /// can be skipped if previous layer has the same one
    SetOutOffset(i8),
    /// Set u15 multiplier
    SetMul(u16),
    /// Set right shift after multiplication (0 to 38 inclusive)
    SetShr(u8),
    /// Set minimum value for activation, -128 if no fused activation function
    SetMin(i8),
    /// finalize neuron processing using given bias, then write result to memory and if (true) do all layer finishing tasks:
    /// - move output offset to input
    /// - block next instruction execution until result is written
    Store { bias: i8, last_in_layer: bool },
    /// end processing - finalize (max), send results and halt
    Fin,
}

impl NonvecInst {
    fn bitstring(&self) -> String {
        match self {
            NonvecInst::Nop => "000".to_owned() + &"0".repeat(15),
            NonvecInst::SetOutOffset(offset) => format!("001{:0>7}{:0>8b}", 0, offset),
            NonvecInst::SetMul(mul) => format!("010{:0>15b}", mul),
            NonvecInst::SetShr(shr) => format!("011{:0>9}{:0>6b}", 0, shr),
            NonvecInst::SetMin(min) => format!("100{:0>7}{:0>8b}", 0, min),
            NonvecInst::Store {
                bias,
                last_in_layer,
            } => format!("101{:0>6}{}{:0>8b}", 0, *last_in_layer as u8, bias),
            NonvecInst::Fin => "111".to_owned() + &"0".repeat(15),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CustInst {
    /// perform multiplication
    mul_en: bool,
    /// add multiplication result to previous, **disable for first mul in neuron!**
    mul_acc: bool,
    /// array of 27 i8 multiplication coefficients
    weights: [i8; 27],
    /// save read pointer, use for first mul **in layer**
    save_rptr: bool,
    /// load read pointer, use with last mul of neuron, except for last in layer
    load_rptr: bool,
    /// (variable) instruction to non-vector part
    proc_inst: NonvecInst,
}

impl CustInst {
    fn bitstring(&self) -> String {
        format!(
            "{mul_en}{mul_acc}{weights}{s_rp}{l_rp}{proc_inst}",
            mul_en = self.mul_en as u8,
            mul_acc = self.mul_acc as u8,
            weights = self
                .weights
                .iter()
                .rev()
                .map(|w| format!("{:0>8b}", w))
                .collect::<Vec<String>>()
                .concat(),
            s_rp = self.save_rptr as u8,
            l_rp = self.load_rptr as u8,
            proc_inst = self.proc_inst.bitstring()
        )
    }
}

impl From<&NonvecInst> for CustInst {
    fn from(value: &NonvecInst) -> Self {
        CustInst {
            mul_en: false,
            mul_acc: false,
            weights: [0; 27],
            save_rptr: false,
            load_rptr: false,
            proc_inst: *value,
        }
    }
}

/// State of (initial) quantization in current graph
#[derive(Debug, PartialEq)]
enum QuantState {
    /// Quantization layer not found yet
    NotSet,
    /// Data is quantized with given zero point (scale assumed to be 1)
    Set(i8),
    /// Quantization has been changed by a non-quantization layer
    /// This is to be subtracted at output and added on input
    FCOutput(i8),
}

fn clog2<T: PrimInt + Unsigned>(x: T) -> u32 {
    T::max_value().count_ones() - x.leading_zeros()
}

fn generate_inst_stream(layers: &[FullyConnectedLayer]) -> Vec<CustInst> {
    let mut instructions = Vec::new();

    // Don't emit unnecessary instructions if values did not change
    // Data about output offset changes is stored in layer data, but these are not.
    // I don't want to rework it. This code is already a mess.
    let mut current_mul = None;
    let mut current_shift = None;
    let mut current_act_min = None;

    for layer in layers {
        check_layer_acc_max(layer);

        // Instructions that must be executed before ending first neuron in layer
        let mut layer_pre_first_write = Vec::new();

        if let Some(out_off) = layer.output_offset {
            layer_pre_first_write.push(NonvecInst::SetOutOffset(out_off));
        }
        if current_mul != Some(layer.multiplier) {
            layer_pre_first_write.push(NonvecInst::SetMul(layer.multiplier));
            current_mul = Some(layer.multiplier);
        }
        if current_shift != Some(layer.shift) {
            layer_pre_first_write.push(NonvecInst::SetShr(layer.shift));
            current_shift = Some(layer.shift);
        }
        if current_act_min.is_none_or(|am| am != layer.act_min.unwrap_or(-128)) {
            let am = layer.act_min.unwrap_or(-128);
            layer_pre_first_write.push(NonvecInst::SetMin(am));
            current_act_min = Some(am);
        }
        let neuron_count = layer.outputs_len();
        let input_count = layer.inputs_len();
        let weights_iter = layer.weights.chunks_exact(input_count).enumerate();

        for (neuron_idx, neuron_weights) in weights_iter {
            let first_neuron_in_layer = neuron_idx == 0;
            let last_neuron_in_layer = neuron_idx == neuron_count - 1;
            let bias = layer.biases[neuron_idx];

            let mut multiplications = neuron_weights
                .chunks(27)
                .map(|chunk| {
                    let mut a = [0i8; 27];
                    a[0..chunk.len()].copy_from_slice(chunk);

                    CustInst {
                        mul_en: true,
                        mul_acc: true,
                        weights: a,
                        save_rptr: false,
                        load_rptr: false,
                        proc_inst: NonvecInst::Nop,
                    }
                })
                .collect::<VecDeque<CustInst>>();
            if let Some(first) = multiplications.front_mut() {
                if first_neuron_in_layer {
                    // first multiplication in layer must save pointer
                    first.save_rptr = true;
                } else {
                    // others issue write from previous neuron
                    first.proc_inst = NonvecInst::Store {
                        bias: layer.biases[neuron_idx - 1],
                        last_in_layer: false,
                    }
                }
                // first multiplication in neuron does not accumulate
                first.mul_acc = false;
            }
            // last multiplication in neuron, except for last one in layer, should restore pointer
            if !last_neuron_in_layer {
                if let Some(last) = multiplications.back_mut() {
                    last.load_rptr = true;
                }
            }
            if first_neuron_in_layer {
                let setup_instr_count = layer_pre_first_write.len();
                let available_mul_count = multiplications.len();
                let dummy_muls_to_add = setup_instr_count.saturating_sub(available_mul_count);
                for setup in layer_pre_first_write[0..dummy_muls_to_add].iter().rev() {
                    multiplications.push_front(CustInst {
                        mul_en: false,
                        mul_acc: false,
                        weights: [0; 27],
                        save_rptr: false,
                        load_rptr: false,
                        proc_inst: *setup,
                    });
                }
                for (mul, si) in multiplications
                    .iter_mut()
                    .zip(layer_pre_first_write.iter())
                    .skip(dummy_muls_to_add)
                {
                    mul.proc_inst = si.clone();
                }
            }
            instructions.extend(multiplications.iter());
            if last_neuron_in_layer {
                instructions.push(CustInst {
                    mul_en: false,
                    mul_acc: false,
                    weights: [0; 27],
                    save_rptr: false,
                    load_rptr: false,
                    proc_inst: NonvecInst::Store {
                        bias: bias,
                        last_in_layer: true,
                    },
                });
            }
        }
    }
    instructions.push(CustInst {
        mul_en: false,
        mul_acc: false,
        weights: [0; 27],
        save_rptr: false,
        load_rptr: false,
        proc_inst: NonvecInst::Fin,
    });

    instructions
}

fn process_fully_connected(
    buffers: &flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<tflite::Buffer>>,
    input: &Tensor,
    output: &Tensor,
    bias: &Tensor,
    weights: &Tensor,
    act_func: ActivationFunctionType,
    set_output_offset: bool,
) -> FullyConnectedLayer {
    let bias_buf = buffers.get(bias.buffer().try_into().unwrap());
    let weights_buf = buffers.get(weights.buffer().try_into().unwrap());
    let input_quant = input.quantization().unwrap();
    let output_quant = output.quantization().unwrap();
    let weights_quant = weights.quantization().unwrap();
    let bias_quant = bias.quantization().unwrap();
    // println!("Bias {:?} - buff {:?}", bias, bias_buf);
    assert_eq!(
        weights_quant.scale().unwrap().len(),
        1,
        "Per-channel quantization is not supported"
    );
    assert_eq!(input_quant.scale().unwrap().len(), 1);
    assert_eq!(bias_quant.scale().unwrap().len(), 1);
    assert_eq!(output_quant.scale().unwrap().len(), 1);
    let input_prod_scale = input_quant.scale().unwrap().get(0).to_f64().unwrap()
        * weights_quant.scale().unwrap().get(0).to_f64().unwrap();
    let bias_scale = bias_quant.scale().unwrap().get(0).to_f64().unwrap();
    let output_scale = output_quant.scale().unwrap().get(0).to_f64().unwrap();
    // TensorFlow checks it, so the check is unnecessary, but better make sure
    assert!((input_prod_scale - bias_scale).abs() / output_scale <= 0.02);
    assert!(input_prod_scale >= 0.0);
    let multiplier = input_prod_scale / output_scale;

    let (mantissa, exponent, sign) = multiplier.integer_decode();
    assert_eq!(
        sign, 1,
        "Quantized multiplier must be non-negative (TFLITE_DCHECK(quantized_multiplier >= 0);)"
    );
    // println!(
    //     "Mult {}, decoded {:?}, f32 {:?}",
    //     multiplier,
    //     multiplier.integer_decode(),
    //     multiplier.to_f32().unwrap().integer_decode()
    // );
    // TensorFlow's QuantizeMultiplier uses frexp to effectively end up with
    // let mant_norm = mantissa.to_f64().unwrap().log2().ceil().to_i8().unwrap();
    let mant_norm = clog2(mantissa);
    let quantized_shift = exponent + mant_norm.to_i16().unwrap();
    let mult_sh = mant_norm - 31;
    let quantized_multiplier = mantissa >> mult_sh;
    // Change range of quantized shift from [-31, 7] to [-38, 0] and make it positive;
    let adjusted_sh = mult_sh - 7 + 23;
    let adj_quantized_multiplier = mantissa >> adjusted_sh;
    let adj_quantized_shift = -quantized_shift + 7 - 23 + 31 - 1;
    println!(
        "TensorFlow's quantized multiplier: {} with shift {}, adjusted to [-38, 0] and u15 {} ({} bits) >> ({} + 1)",
        quantized_multiplier,
        quantized_shift,
        adj_quantized_multiplier,
        clog2(adj_quantized_multiplier),
        adj_quantized_shift
    );
    let tf_mul_reconstructed =
        quantized_multiplier as f64 * 2.0.powf((quantized_shift - 31) as f64);
    let net_mul_reconstructed =
        adj_quantized_multiplier as f64 * 2.0.powf(-(adj_quantized_shift + 1) as f64);
    let mul_diff = tf_mul_reconstructed - net_mul_reconstructed;
    println!(
        "Q: {:e}, A: {:e}, diff {:e}",
        tf_mul_reconstructed, net_mul_reconstructed, mul_diff
    );
    assert!(adj_quantized_shift >= 0);
    assert!(adj_quantized_shift <= 38);
    assert!(adj_quantized_multiplier < 1u64 << 15);

    let input_offset_neg = input_quant.zero_point().unwrap().get(0).to_i8().unwrap();
    let weights_offset_neg = weights_quant.zero_point().unwrap().get(0).to_i8().unwrap();
    let output_offset = output_quant.zero_point().unwrap().get(0).to_i8().unwrap();
    // println!(
    //     "Offsets - in={}, weight={}, out={}",
    //     input_offset, weights_offset, output_offset
    // );
    assert_eq!(
        weights_offset_neg, 0,
        "INT8 quantization requires 0 as zero point"
    );

    let bias_vec: Vec<i8> = bias_buf
        .data()
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes(x.try_into().unwrap()) as i8)
        .collect();
    assert_eq!(
        bias_vec.len(),
        bias.shape().unwrap().get(0).try_into().unwrap()
    );
    // assert!(
    //     bias_vec
    //         .iter()
    //         .all(|x| (i8::MIN as i32) <= *x && *x <= (i8::MAX as i32))
    // );
    // println!("Bias vec {:?}", bias_vec);
    // println!("Output {:?}", output);
    assert!(act_func == ActivationFunctionType::NONE || act_func == ActivationFunctionType::RELU);

    FullyConnectedLayer {
        weights: weights_buf
            .data()
            .unwrap()
            .iter()
            .map(|x| i8::from_le_bytes([x]))
            .collect(),
        biases: bias_vec,
        output_offset: if set_output_offset {
            Some(output_offset)
        } else {
            None
        },
        input_offset_neg,
        multiplier: adj_quantized_multiplier as u16,
        shift: adj_quantized_shift.try_into().unwrap(),
        // act_min is max(output.zero_point, i8::MIN), if output.zero_point < i8::MIN, the conversion to i8 will fail and the default will be used
        act_min: if act_func == ActivationFunctionType::NONE {
            None
        } else {
            Some(output_offset.try_into().unwrap_or(i8::MIN))
        },
    }
}

/// Check if max value will fit in accumulators (width specified by BITS constant in first line)
fn check_layer_acc_max(layer: &FullyConnectedLayer) {
    const BITS: u32 = 24;
    const MIN_REPR: i64 = -(1 << (BITS - 1));
    const MAX_REPR: i64 = (1 << (BITS - 1)) - 1;

    let min_input = i8::MIN as i64 - layer.input_offset_neg as i64;
    let max_input = i8::MAX as i64 - layer.input_offset_neg as i64;

    let (acc_min, acc_max) = layer
        .weights
        .chunks_exact(layer.inputs_len())
        .zip(layer.biases.iter())
        .map(|(chunk, bias)| {
            let bias = *bias as i64;
            let (min, max) = chunk.iter().fold((0, 0), |(acc_min, acc_max), w| {
                let max = (*w as i64) * max_input;
                let min = (*w as i64) * min_input;
                if w.is_negative() {
                    (acc_min + max, acc_max + min)
                } else {
                    (acc_min + min, acc_max + max)
                }
            });
            (min + bias, max + bias)
        })
        .fold((0, 0), |(min, max), (l_min, l_max)| {
            (min.min(l_min), max.max(l_max))
        });

    println!(
        "Accumulators in layer will be in range [{}, {}] and require {} bits as signed integers",
        acc_min,
        acc_max,
        clog2(acc_max.unsigned_abs()).max(clog2(acc_min.unsigned_abs())) + 1
    );
    assert!(MIN_REPR <= acc_min && acc_min <= MAX_REPR);
    assert!(MIN_REPR <= acc_max && acc_max <= MAX_REPR);
}

/// Write network config data "config.dat", with network-specific parameters.
fn write_config(input_quant_offset: i8) {
    let input_quant_param = format!("{:0>8b}", input_quant_offset);
    let config_data = [input_quant_param];
    let bin_data = config_data.join("\n");
    fs::write("../hdl/config.dat", bin_data).unwrap();
}

fn main() {
    println!(
        "Remember that it only works if layers are purely sequential and output from each of them is immediately consumed by the next one."
    );
    let mut f = std::fs::File::open("../model_notnorm_nosoft.tflite").unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).expect("file reading failed");

    let model = tflite::root_as_model(&buf).unwrap();
    let opcodes = model.operator_codes().unwrap();
    // println!("{:?}", opcodes);
    let subgraphs = model.subgraphs().unwrap();
    let buffers = model.buffers().unwrap();
    assert_eq!(
        subgraphs.len(),
        1,
        "Only single subgraph/function is supported"
    );
    for graph in subgraphs {
        let name = graph.name().unwrap();
        println!("Processing subgraph {}", name);
        let operators = graph.operators().unwrap();
        let tensors = graph.tensors().unwrap();
        let mut quant_zero_point = QuantState::NotSet;
        let mut layers: Vec<FullyConnectedLayer> = Vec::new();
        let mut prev_output = None;
        for op in operators {
            let op_idx: usize = op.opcode_index().try_into().unwrap();
            let op_code = opcodes.get(op_idx).builtin_code();
            let options_type = op.builtin_options_type();
            print!("Operation {:?}", op_code);
            let inputs_idx_v = op.inputs().unwrap();
            let outputs_idx_v = op.outputs().unwrap();
            assert!(inputs_idx_v.len() >= 1);
            assert_eq!(outputs_idx_v.len(), 1);
            let input_idx = inputs_idx_v.get(0);
            let output_idx = outputs_idx_v.get(0);
            let input = tensors.get(input_idx as usize);
            let output = tensors.get(output_idx as usize);
            if op_code == tflite::BuiltinOperator::QUANTIZE {
                assert_eq!(input.type_(), tflite::TensorType::FLOAT32);
                assert!(
                    input
                        .quantization()
                        .is_none_or(|q| q.details_type() == tflite::QuantizationDetails::NONE)
                );
                let output_quant = output.quantization().unwrap();
                assert_eq!(
                    output_quant.scale().unwrap().len(),
                    1,
                    "There must be a single quantization parameter set"
                );
                // I don't want to bother with quantization that scales anything
                assert_eq!(output_quant.scale().unwrap().get(0), 1.0);
                let zero_point = output_quant.zero_point().unwrap().get(0) as i8;
                println!(", scale 1, zero point {}", zero_point);
                assert_eq!(quant_zero_point, QuantState::NotSet);
                quant_zero_point = QuantState::Set(zero_point);
                continue;
            }
            if op_code != tflite::BuiltinOperator::FULLY_CONNECTED {
                if let QuantState::Set(zero) = quant_zero_point {
                    assert_eq!(
                        output.quantization().unwrap().zero_point().unwrap().get(0),
                        zero as i64
                    );
                    assert_eq!(output.quantization().unwrap().scale().unwrap().get(0), 1.0);
                }
                println!(" - skipping");
                continue;
            }
            assert_eq!(options_type, tflite::BuiltinOptions::FullyConnectedOptions);
            let options = op.builtin_options_as_fully_connected_options().unwrap();
            assert_eq!(options.asymmetric_quantize_inputs(), false);
            assert_eq!(
                options.weights_format(),
                tflite::FullyConnectedOptionsWeightsFormat::DEFAULT
            );
            assert_eq!(options.quantized_bias_type(), tflite::TensorType::FLOAT32);
            print!(
                ", activation function {:?}",
                options.fused_activation_function()
            );
            println!("");
            assert!(
                prev_output.is_none_or(|out| out == input_idx),
                "Fully connected layers must be connected back-to-back"
            );
            prev_output = Some(output_idx);
            let output_zero_point =
                output.quantization().unwrap().zero_point().unwrap().get(0) as i8;
            let input_zero_point = input.quantization().unwrap().zero_point().unwrap().get(0) as i8;
            let set_output_offset = match quant_zero_point {
                QuantState::Set(zp) => {
                    assert_eq!(
                        zp, input_zero_point,
                        "First fully connected layer must have same input zero point as preceding quantization output (ignoring quantization)"
                    );
                    true
                }
                QuantState::FCOutput(zp) => {
                    assert_eq!(
                        zp, input_zero_point,
                        "Fully connected layer must have same input zero point as the output of previous one (input_offset_neg <= output_offset)"
                    );
                    output_zero_point != zp
                }
                QuantState::NotSet => true,
            };
            quant_zero_point = QuantState::FCOutput(output_zero_point);
            assert!(inputs_idx_v.len() == 3);
            let weights = tensors.get(inputs_idx_v.get(1).try_into().unwrap());
            let bias = tensors.get(inputs_idx_v.get(2).try_into().unwrap());
            layers.push(process_fully_connected(
                &buffers,
                &input,
                &output,
                &bias,
                &weights,
                options.fused_activation_function(),
                set_output_offset,
            ));
        }

        let inst_stream = generate_inst_stream(&layers);
        let bin_data = inst_stream
            .iter()
            .map(|inst| inst.bitstring())
            .collect::<Vec<String>>()
            .join("\n");
        fs::write("../hdl/prog.dat", bin_data).unwrap();
        fs::write("../hdl/prog.txt", format!("{:#?}", inst_stream)).unwrap();

        let in_quant_offset = layers[0].input_offset_neg;
        write_config(in_quant_offset);
    }
}
