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

void utensor_test_load(int x);
int utensor_test(int x);

#ifdef __cplusplus
}
#endif
