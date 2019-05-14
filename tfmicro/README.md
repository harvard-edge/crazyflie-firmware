# Tensorflow Micro on the Crazyflie 2.0

We will try to compile TF Micro on the Crazyflie 2.0, which definitely should
be able to handle the library size. The specs on the default Crazyflie:

* 1 MB flash memory
* Cortex M4 ARM chip

However, all of the default firmware is written in C. Therefore, in this segment
we will compile TF Micro for the microcontoller using C++ (`arm-none-eabi-g++`),
expose some libraries to C, and compile everything in the end in C so the
default firmware works (`arm-non-eabi-gcc`).

## Benchmarking TF Micro Models

We will be trying to use fully connected layers, and for testing we will create 
a few models of varying size in `./train` and make sure that they work in
the regular C++ Tensorflow Lite environment. This is used before using a DQN
trained in Microsoft Airsim for benchmarking purposes (checking model size,
model inference times, etc.)

## Useful Commands

### Freezing Keras Graph into Tensorflow Frozen Graph

```bash
python tensorflow/python/tools/freeze_graph.py \
--input_meta_graph=/tmp/keras_model.ckpt.meta \
--input_checkpoint=/tmp/keras_model.ckpt \
--output_graph=/tmp/keras_frozen.pb \
--output_node_names="<output_node_name_printed_in_step_1>" \
--input_binary=true
```

### Convert TF Frozen Graph to TFLite/Micro

```bash
bazel run tensorflow/lite/toco:toco -- \
--input_file=/tmp/tiny_conv.pb --output_file=/tmp/tiny_conv.tflite \
--input_shapes=1,49,43,1 --input_arrays=Reshape_1 --output_arrays='labels_softmax' \
--inference_type=QUANTIZED_UINT8 --mean_values=0 --std_values=2 \
--change_concat_input_ranges=false
```

## Deploying

Deploying a trained model trained in simulator on Microsoft Airsim can be done
in the same steps, by first quantizing, converting to byte array in C, and loading
it into the Crazyflie Flash Memory. More info is 
[here.](WORKFLOW.md)

## Gotchas

### std::round

Due to the `arm-none-eabi` compiler not supporting everything, some things
just aren't there for C++, such as `std::round`. Therefore, `std::round` has
been replaced by `::round` for Tensorflow.

