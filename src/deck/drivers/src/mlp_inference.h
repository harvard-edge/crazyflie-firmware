//
// Created by bart on 11/20/19.
//

int float_inference(float* input, int size);
void float_matmul(float* input, int input_size, float* matrix, int matrix_size, float* output, int output_size);
void float_bias_add(float* input, int input_size, float* matrix);
void float_relu(float* input, int input_size);
int argmax_float(float* array, int size);
void zero_tensor(float* tensor, int tensor_size);



