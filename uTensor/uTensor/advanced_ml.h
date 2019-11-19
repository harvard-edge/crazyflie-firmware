// Choices of model in tfmicro_models.h
// Examples are: fc_tflite, micro_conv_tflite, etc
#ifndef UTENSOR_MODEL 
#define UTENSOR_MODEL test
#endif

// type of model. uint8_t if quantized, usually float if not.
#define model_type uint8_t

#ifdef __cplusplus
extern "C" {
#endif

int inference(float*, unsigned int, float *output);
void inference_test(float*, unsigned int, float *output);

int utensor_test_load(int x);
int utensor_test(int x);



struct CWrappedRamTensor;
typedef struct CWrappedRamTensor CWrappedRamTensor;
CWrappedRamTensor* CWrappedRamTensor_create(float *arr);
CWrappedRamTensor* CWrappedRamTensor_set(CWrappedRamTensor*, float *arr);
void destroy_tensor(CWrappedRamTensor* tensor);

// This one uses our wrapper
void inference_new(CWrappedRamTensor* tensor, float *output);

#ifdef __cplusplus
}
#endif
