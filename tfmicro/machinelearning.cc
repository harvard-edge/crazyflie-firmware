/* An attempt at creating C extensions to TF-Micro in order to get it to run
on a Crazyflie 2.0. The reason for this is because all of the firmware for
the crazyflie is written in C, and although we can compile TF-Micro for the
Cortex M4, we still have to link it to the main loop in the firmware.

Approach will be to expose the common functions we need in C, compile TF Micro
in C++, and then link it to the main loop later.
==============================================================================*/
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "machinelearning.h"

// Our machine learning models we're putting in :)
#include "tfmicro_models.h"

extern "C" {

	void setup() {
		tflite::ErrorReporter* error_reporter = nullptr;
		tflite::MicroInterpreter* interpreter = nullptr;
		TfLiteTensor* input = nullptr;
		TfLiteTensor* output = nullptr;
		int inference_count = 0;

		// Create an area of memory to use for input, output, and intermediate arrays.
		// Finding the minimum value for your model may require some trial and error.
		constexpr int kTensorArenaSize = 2 * 1024;
		uint8_t tensor_arena[kTensorArenaSize];
	}

	const CTfLiteModel* CTfLiteModel_create() {
		const tflite::Model* model = nullptr;
		model = tflite::GetModel(source_seeking);
		return reinterpret_cast<const CTfLiteModel*>(model);
	}

	// check to see if our version of tflite matches the model
	int CTfLiteModel_check(const CTfLiteModel* wrapped_model) {
		auto model = reinterpret_cast<const tflite::Model*>(wrapped_model);
		if (model->version() != TFLITE_SCHEMA_VERSION) {
			return 1;
		}
    	return 0;
	}

	int CTfLiteModel_version(const CTfLiteModel* wrapped_model) {
		auto model = reinterpret_cast<const tflite::Model*>(wrapped_model);
		return model->version();
	}

	void CTfLiteModel_destroy(const CTfLiteModel* wrapped_model) {
		auto model = reinterpret_cast<const tflite::Model*>(wrapped_model);
		delete model;
	}

	
}


/*
extern "C" {
	
	// TfLiteModel wrapper functions


	void CTfLiteModel_destroy(CTfLiteModel* v) {
		delete reinterpret_cast<tflite::Model*>(v);
	}
	

	int CTfLiteModel_version(CTfLiteModel* v) {
		return static_cast<int>(reinterpret_cast<tflite::Model*>(v)->version());
	}	
	

	
	// TF Micro Interpreter wrapper functions
	 
	int CTfInterpreter_create_return_version(const CTfLiteModel* c_model, int alloc_size) {
		tflite::MicroErrorReporter micro_error_reporter;
		tflite::ErrorReporter* error_reporter = &micro_error_reporter;
		::tflite::ops::micro::AllOpsResolver resolver;

		const tflite::Model* model = reinterpret_cast<const tflite::Model*>(c_model);
		uint8_t tensor_arena[alloc_size];
		tflite::SimpleTensorAllocator tensor_allocator(tensor_arena, alloc_size);
		tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator, error_reporter);
		TfLiteTensor* model_input = interpreter.input(0);
		return model_input->dims->size;
	}
	
	int CTfLiteModel_dimensions(const CTfLiteModel* c_model, uint8_t* arena, size_t size, int dim) {
		tflite::MicroErrorReporter micro_error_reporter;
		tflite::ErrorReporter* error_reporter = &micro_error_reporter;
		::tflite::ops::micro::AllOpsResolver resolver;

		const tflite::Model* model = reinterpret_cast<const tflite::Model*>(c_model);
		tflite::SimpleTensorAllocator tensor_allocator(arena, size);
		tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator, error_reporter);
		TfLiteTensor* model_input = interpreter.input(0);
		return model_input->dims->data[dim];
	}

	// Actual methods used for inferencing
	void CTfInterpreter_simple_fc(const CTfLiteModel* c_model, 
			uint8_t* tensor, int alloc_size, uint8_t* input, int* result) {
		tflite::MicroErrorReporter micro_error_reporter;
		tflite::ErrorReporter* error_reporter = &micro_error_reporter;
		::tflite::ops::micro::AllOpsResolver resolver;

		const tflite::Model* model = reinterpret_cast<const tflite::Model*>(c_model);
		tflite::SimpleTensorAllocator tensor_allocator(tensor, alloc_size);
		tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator, error_reporter);
		TfLiteTensor* model_input = interpreter.input(0);

		memcpy(model_input->data.uint8, input, 20 * sizeof(uint8_t));

		TfLiteStatus invoke_status = interpreter.Invoke();
		if (invoke_status != kTfLiteOk) {
			return;
		}

		TfLiteTensor* output = interpreter.output(0);

		int NUM_CLASSES = 3;
		for (int i = 0; i < NUM_CLASSES; i++) {

			result[i] = static_cast<int>(output->data.uint8[i]);
		}
	}

	// Run a single inference on some model, using uint8 quantization.
	// 	Input: TFLiteModel, already wrapped in C.
	// 		   tensor: buffer used for performing operations.
	// 		   alloc_size: memory to allocate for tensor operations
	// 		   result: address to array to write results into
	// 	Output:
	// 		Error code:
	// 			0: success
	// 			-1: Evaluation step did not return OK
	
	int inference_uint8(const CTfLiteModel* c_model, 
			uint8_t* tensor, int alloc_size,
			uint8_t* input, size_t input_size, int* result) {
		tflite::MicroErrorReporter micro_error_reporter;
		tflite::ErrorReporter* error_reporter = &micro_error_reporter;
		::tflite::ops::micro::AllOpsResolver resolver;

		const tflite::Model* model = reinterpret_cast<const tflite::Model*>(c_model);
		tflite::SimpleTensorAllocator tensor_allocator(tensor, alloc_size);
		tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator, error_reporter);
		TfLiteTensor* model_input = interpreter.input(0);

		memcpy(model_input->data.uint8, input, input_size * sizeof(uint8_t));

		TfLiteStatus invoke_status = interpreter.Invoke();
		if (invoke_status != kTfLiteOk) {
			return -1;
		}

		TfLiteTensor* output = interpreter.output(0);

		int NUM_CLASSES = 3;
		for (int i = 0; i < NUM_CLASSES; i++) {
			result[i] = static_cast<int>(output->data.uint8[i]);
		}
		return 0;
	}

	int inference_float32(const CTfLiteModel* c_model, 
			uint8_t* tensor, int alloc_size,
			float* input, size_t input_size, float* result) {
		tflite::MicroErrorReporter micro_error_reporter;
		tflite::ErrorReporter* error_reporter = &micro_error_reporter;
		::tflite::ops::micro::AllOpsResolver resolver;


		const tflite::Model* model = reinterpret_cast<const tflite::Model*>(c_model);
		tflite::SimpleTensorAllocator tensor_allocator(tensor, alloc_size);
		tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator, error_reporter);
		TfLiteTensor* model_input = interpreter.input(0);

		if (model_input->type != kTfLiteFloat32) {
			return -2;
		}

		memcpy(model_input->data.f, input, input_size * sizeof(float));

		TfLiteStatus invoke_status = interpreter.Invoke();
		if (invoke_status != kTfLiteOk) {
			return -1;
		}

		TfLiteTensor* output = interpreter.output(0);

		int NUM_CLASSES = 3;
		for (int i = 0; i < NUM_CLASSES; i++) {
			result[i] = static_cast<float>(output->data.f[i]);
		}
		return 0;
	}




}


extern "C" int machine_learning_test(int n) {
	return 0;
}


extern "C" int testDoubleFunction(int x) {
	return x * 2;
};
*/