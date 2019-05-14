# MicroNets Workflow

## Console Output

Almost all of the output from the code will be sent through the onboard radio
to your computer with the CrazyFlie Dongle. You will need to download the
CrazyFlie client and install it, which can be found
[here](https://github.com/bitcraze/crazyflie-clients-python). Make sure to
follow all the installation steps so you can run `cfclient` and open the
window.

Then, to see console output:

1. Turn on the CrazyFlie normally (not in pairing mode).
2. Press `Scan` to find the radio device and then press `Connect`.
3. Go to the `Console` tab and you should see the debug output from the Crazyflie.

To print to this console, we can use `DEBUG_PRINT` or `consolePrintf` in the
code. This is an easy way to see debug output from our TF-Micro model. 

## Training a Model

### Training the Model

First, train your model in tensorflow. You can use 
[Microsoft AirSim](https://github.com/microsoft/AirSim) with a
custom environment or [OpenAI gym](https://gym.openai.com/)
for creating an environment and training a DQN for the crazyflie.
Currently, the only layers that I have tested are __fully connected__ and
__convolutional__ layers. I don't think any other layers are supported
on the crazyflie.

### Quantization

After training the tensorflow model the next steps are freezing the tensorflow
graph and quantizing to 8-bit.
The process depends on how you defined your tensorflow graph, but examples
of freezing can be found in `tfmicro/train/freeze_simple.py` and
`tfmicro/train/freeze_conv.py`.
Then, you need to clone the tensorflow repository, compile it, and quantize
with the following command:

```bash
cd ${TF_HOME}

bazel run tensorflow/lite/toco:toco -- \
    --input_file=${CHECKPT_DIR}/${MODEL_NAME}.pb \
    --output_file=$(pwd)/${MODEL_NAME}.tflite \
    --input_shapes=1,${NUM_INPUTS} \
    --input_arrays=input \
    --output_arrays='labels_softmax' \
    --inference_type=QUANTIZED_UINT8 \
    --change_concat_input_ranges=false
```

replacing the variable as necessary to match the parameters of your model.

## Deploying the Model

Once the model has been quantized, turn it into a C array and load it in using
the TF-Micro API.

```bash
xxd -i [tf_micro_model.tflite] >> tfmicro_models.cc
```

Then, you can load in the model using TF-Micro - again, see the example code
in `src/deck/drivers/src` to see how everything is loaded in.

### Writing Deployment Code

To deploy the model, the easiest way is to create a `.c` file in the
`src/deck/drivers/src` folder.
Essentially, we write a code that pretends to be a Crazyflie Extension deck,
and when compiling we force the Crazyflie to load that code in first.
See `tfmicrodemo.c` inside this folder for a template.
Furthermore, you must edit the Makefile - see all of the comments in the
Makefile that are labeled as `TF Micro Compilation`, you'll have to edit those
areas with names for your own C file.

### Wrapping TF Micro

All of the original Crazyflie code was compiled in C and TF-Micro requires some
C++ extensions, so there are wrappers needed to call the TF-Micro library.
The wrappers can be extended - they are at `tfmicro/machinelearning.h` and
`tfmicro/machinelearning.cc`.
