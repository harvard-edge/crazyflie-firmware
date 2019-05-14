#!/bin/bash
# Trains a few different conv model sizes for benchmarking the inference time
# and power consumption when quantized the deployed onto the crazyflie.

TF_HOME=${HOME}/tensorflow
CURR_DIR=$(pwd)
NUM_INPUTS=4
CHECKPT=12000
CHECKPT_DIR=/tmp/tfmicro/conv
OUTPUT_NODE=labels_softmax
QUANTIZE=1
CURR_DIR=$(pwd)

for HISTORY_SIZE in 10
do
    MODEL_NAME=conv_${HISTORY_SIZE}
    CHECKPT_FILE=${CHECKPT_DIR}/${MODEL_NAME}.ckpt-${CHECKPT}
    echo "Looking for ${CHECKPT_FILE}"
    if [ ! -f "${CHECKPT_FILE}.index" ];
    then 
        python3 train_conv.py \
            --history_size ${HISTORY_SIZE} \
            --input_size ${NUM_INPUTS} \
            --model_architecture ${MODEL_NAME} \
            --train_dir ${CHECKPT_DIR} \
            --batch_size 8000
    else
        echo "Already found a trained file matching name."
    fi


    python3 freeze_conv.py \
        --history_size ${HISTORY_SIZE} \
        --quantize=${QUANTIZE} \
        --output_file=${CHECKPT_DIR}/${MODEL_NAME}.pb \
        --start_checkpoint=${CHECKPT_DIR}/${MODEL_NAME}.ckpt-${CHECKPT} \
        --num_inputs ${NUM_INPUTS}

    cd ${TF_HOME}

    bazel run tensorflow/lite/toco:toco -- \
        --input_file=${CHECKPT_DIR}/${MODEL_NAME}.pb \
        --output_file=${CURR_DIR}/${MODEL_NAME}.tflite \
        --input_shapes=1,$(($NUM_INPUTS * $HISTORY_SIZE)) \
        --input_arrays=input \
        --output_arrays='labels_softmax' \
        --inference_type=QUANTIZED_UINT8 \
        --change_concat_input_ranges=false

    cd -
done
