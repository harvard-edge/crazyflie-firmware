#!/bin/bash
# Trains a few different FC model sizes for benchmarking the inference time
# and power consumption when quantized the deployed onto the crazyflie.

TF_HOME=${HOME}/tensorflow
CURR_DIR=$(pwd)
NUM_INPUTS=4
CHECKPT=18000
CHECKPT_DIR=/tmp/tf/experiment
OUTPUT_NODE=labels_softmax
QUANTIZE=1
CURR_DIR=$(pwd)

for FCSIZE in 20 40 80 120 160 240 280 290 300 310 320 360 400 440 
do
    MODEL_NAME=fc_${FCSIZE}
    CHECKPT_FILE=${CHECKPT_DIR}/${MODEL_NAME}.ckpt-${CHECKPT}
    echo "Looking for ${CHECKPT_FILE}"
    if [ ! -f "${CHECKPT_FILE}.index" ];
    then 
        python3 train_simple.py \
            --fc-size ${FCSIZE} \
            --model_architecture ${MODEL_NAME} \
            --train_dir ${CHECKPT_DIR} \
            --batch_size 20000
    else
        echo "Already found a trained file matching name."
    fi

    python3 freeze_simple.py \
        --model_architecture=${MODEL_NAME} \
        --fc_size ${FCSIZE} \
        --quantize=${QUANTIZE} \
        --output_file=${CHECKPT_DIR}/${MODEL_NAME}.pb \
        --start_checkpoint=${CHECKPT_DIR}/${MODEL_NAME}.ckpt-${CHECKPT} \
        --num_inputs ${NUM_INPUTS}

    cd ${TF_HOME}

    bazel run tensorflow/lite/toco:toco -- \
        --input_file=${CHECKPT_DIR}/${MODEL_NAME}.pb \
        --output_file=${CURR_DIR}/${MODEL_NAME}.tflite \
        --input_shapes=1,${NUM_INPUTS} \
        --input_arrays=input \
        --output_arrays='labels_softmax' \
        --inference_type=QUANTIZED_UINT8 \
        --change_concat_input_ranges=false
    cd -
done
