# Instruction processing

How things should be prepared and executed

## Processing start

Nothing to do (besides reset) if quantization offset is same as first layer input offset (asserted in generator).

**Above sentence is false!** Not doing quantization (adding/subtracting zero point aka offset) means problems with signedness and wrong value interpretation (first layer would have to use unsigned inputs, while the rest uses signed).

## Per layer

### Before calculating final result of first neuron

- set output offset (except l0) if changed from previous
- set multiplier
- set shift (this could also set rounding add)
- set activation min threshold

Sizes of above parameters:

- output offset: 8-bit signed
- multiplier: 15-bit unsigned (this leaves 1 bit for rounding)
- shift amount: 6-bit unsigned (range [0, 38], following shift by 1 after rounding add)
- act min: 8-bit signed

Shift amount and activation min threshold can be combined (14 bits <= 15 multiplier bits) to save a single instruction (4 => 3).

## Per neuron

After multiplications are done:

1. accumulate
2. pre-add bias, multiplicate, post-add rounding
3. shift right
4. activation (if necessary)
5. add output offset
6. write result

The process above is sequential with parameters from layer (mult, round (calculated), sh, act_min, output_offset) and neuron-specific params:

- bias: 8-bit signed
- is last in layer (to move write pointer and copy output offset to input offset): boolean/1-bit unsigned

Single instruction-triggered FSM of known cycle length and pad software with NOPs?

### ~~FSM states~~ pipeline stages

1. ~~WAIT~~
2. ACC_1 - first layer of accumulation
3. ACC_2 - second layer of accumulation
4. ACC_3 - final accumulated result
5. MUL_RD - biased, multiplied, rounding added (DSP)
6. SHR - shifted right
7. ACT_OFF - activation and offset, ready to write

**Remember that activation is always performed (even when it shouldn't be, it's just saturate then) and thresholds must take `output_offset` into account.** The thresholds are always `act_min` - `output_offset` and `act_max` - `output_offset` (`act_max` is 127). These could be precomputed whenever `act_min` or `output_offset` change.\
An easier solution is to first add output offset, then do the threshold/saturation. Care must be taken not to cause overflow, but there is enough margin for i8 addition not to overflow.\
**Solution:** Add offset together with round.

**Block next operations if there is an operation marked as last in layer!**

## Processing end

Index of maximum entry of first 10 elements. Can use 5 comparators and time multiplexing - 4 steps.

## Other notes

Last write in layer must be a separate instruction. Merging it with next layer may cause problems if that layer needs this value (could be allowed if that is not the case).
