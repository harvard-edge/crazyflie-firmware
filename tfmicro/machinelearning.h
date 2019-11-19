// Choices of model in tfmicro_models.h
// Examples are: fc_tflite, micro_conv_tflite, etc
#ifndef TFMICRO_MODEL
#define TFMICRO_MODEL source_seeking
#endif

// type of model. uint8_t if quantized, usually float if not.
#define model_type uint8_t

#ifdef __cplusplus
extern "C" {
#endif

int testDoubleFunction(int x);
int machine_learning_test(int n);

struct CTfLiteModel; // An opaque type that we'll use as a handle
typedef struct CTfLiteModel CTfLiteModel;
const CTfLiteModel* CTfLiteModel_create();
void CTfLiteModel_destroy(CTfLiteModel*);
int CTfLiteModel_version(CTfLiteModel* v);
int CTfLiteModel_input_size(CTfLiteModel* v);


struct CTfInterpreter; // An opaque type that we'll use as a handle
typedef struct CTfInterpreter CTfInterpreter;
int CTfInterpreter_create_return_version(const CTfLiteModel*, int);
int CTfLiteModel_dimensions(const CTfLiteModel* c_model, uint8_t* arena, size_t size, int dim);

// Actual inference functions
void CTfInterpreter_simple_fc(const CTfLiteModel* c_model, uint8_t* tensor, int alloc_size, uint8_t* input, int* result);
int inference_uint8(const CTfLiteModel*, uint8_t* tensor, int alloc_size, uint8_t* input, size_t input_size, int* result);
int inference_float32(const CTfLiteModel*, uint8_t* tensor, int alloc_size, float* input, size_t input_size, float* result);

#ifdef __cplusplus
}
#endif
