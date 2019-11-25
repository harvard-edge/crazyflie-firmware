//
// Created by bart on 11/20/19.
//


#include "mlp_inference.h"
#include "weights.h"
#include "debug.h"
#include "FreeRTOS.h"

#define LAYER_1_SIZE 20
#define LAYER_2_SIZE 20
#define LAYER_3_SIZE 3

float layer_1[LAYER_1_SIZE] = {0};
float layer_2[LAYER_2_SIZE] = {0};
float layer_3[LAYER_3_SIZE] = {0};


int float_inference(float* input, int size){


    //layer 1
    float_matmul(input, size,weights_1,120,layer_1,LAYER_1_SIZE);
    float_bias_add(layer_1, LAYER_1_SIZE, bias_1);
    float_relu(layer_1,LAYER_1_SIZE);

    float_matmul(layer_1, LAYER_1_SIZE,weights_2,400,layer_2,LAYER_2_SIZE);
    vTaskDelay(M2T(10));
    float_bias_add(layer_2, LAYER_2_SIZE, bias_2);
    float_relu(layer_2,LAYER_2_SIZE);

    float_matmul(layer_2, LAYER_2_SIZE,weights_3,60,layer_3,LAYER_3_SIZE);
    float_bias_add(layer_3, LAYER_3_SIZE, bias_3);

    int command = argmax_float(layer_3,3);

    zero_tensor(layer_1,LAYER_1_SIZE);
    zero_tensor(layer_2,LAYER_2_SIZE);
    zero_tensor(layer_3,LAYER_3_SIZE);

    return command;

}

void float_matmul(float* input, int input_size, float* matrix, int matrix_size, float* output, int output_size){
    int i = 0;
    while (i< matrix_size){
        for (int k=0; k< input_size; k++ ){
            for (int j = 0; j < output_size; j++) {
                output[j] += + matrix[i]*input[k];
                i++;
            }
        }
    }
    return;

}

void float_bias_add(float* input, int input_size, float* matrix){
    for(int i = 0; i<input_size; i++){
        input[i] += matrix[i];
    }
    return;
}

void float_relu(float* input, int input_size){
    for (int i = 0 ; i< input_size; i++){
        if(input[i]<0){
            input[i] = 0;
        }
    }
    return;
}

void zero_tensor(float* tensor, int tensor_size){
    for(int i = 0; i< tensor_size; i++){
        tensor[i] = 0.0;
    }

}


int argmax_float(float* array, int size){
    float max = array[0];
    int max_ind = 0;
    for (int i = 1; i< size; i++){
        if (array[i]>max){
            max = array[i];
            max_ind = i;
        }
    }
    return max_ind;
}

