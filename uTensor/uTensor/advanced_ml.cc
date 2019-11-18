/* An attempt at creating C extensions to TF-Micro in order to get it to run
on a Crazyflie 2.0. The reason for this is because all of the firmware for
the crazyflie is written in C, and although we can compile TF-Micro for the
Cortex M4, we still have to link it to the main loop in the firmware.

Approach will be to expose the common functions we need in C, compile TF Micro
in C++, and then link it to the main loop later.
==============================================================================*/

#include "eprintf.h"
#include "console.h"
#include "advanced_ml.h"
#include "frozen_model.hpp"
#include "uTensor/core/tensor.hpp"


#define UTENSOR_OUTPUT_NODE "deepq/model/action_value/fully_connected_2/BiasAdd:0"
#define BATCH_SIZE 1
#define INPUT_SIZE 6

extern "C" {

    /* A wrapper class for WrappedRamTensor, so we can use it in C. */
    CWrappedRamTensor* CWrappedRamTensor_create(float *arr) {
        Tensor* input_x = new WrappedRamTensor<float>({BATCH_SIZE, INPUT_SIZE}, arr);
        return reinterpret_cast<CWrappedRamTensor*>(input_x);
    }


    CWrappedRamTensor* CWrappedRamTensor_set(CWrappedRamTensor* tensor, float *arr) {
        WrappedRamTensor<float>* new_tensor = reinterpret_cast<WrappedRamTensor<float>*>(tensor);
        new_tensor->setPointer(arr) ;
        tensor = reinterpret_cast<CWrappedRamTensor*>(new_tensor);
        return tensor;
    }

    void destroy_tensor(CWrappedRamTensor* tensor){
        free(tensor);
    }

	/* Does one inference, returns the results.*/
	int inference(float *arr, unsigned int size, float *output) {
		float input_data[size];
		// Create a context for the model, we always use a batch size of 1.
		Context ctx;
		Tensor* input_x = new WrappedRamTensor<float>({BATCH_SIZE, size}, arr);
		get_frozen_model_ctx(ctx, input_x);
		ctx.eval();
		input_x = NULL;

		S_TENSOR pred_tensor = ctx.get(UTENSOR_OUTPUT_NODE);

		for (int i = 0; i < pred_tensor->getSize(); i++) {
		    output[i] = *(pred_tensor->read<float>(i, 0));
		}

		return 1;
	}



	void inference_new(CWrappedRamTensor* input_x, float *output) {
        // Create a context for the model, we always use a batch size of 1.
        Context ctx;
        get_frozen_model_ctx(ctx, reinterpret_cast<WrappedRamTensor<float>*>(input_x));
        ctx.eval();
        S_TENSOR pred_tensor = ctx.get(UTENSOR_OUTPUT_NODE);

        for (int i = 0; i < pred_tensor->getSize(); i++) {
            output[i] = *(pred_tensor->read<float>(i, 0));
        }
    }



}






extern "C" int utensor_test(int x) {
	return x * 2;
};
