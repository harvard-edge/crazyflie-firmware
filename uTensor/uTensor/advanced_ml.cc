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


#define UTENSOR_INPUT_DIM 784

extern "C" {

	void utensor_test_load(int x) {
		float input_data[UTENSOR_INPUT_DIM];
		Context ctx;
		// get_frozen_model_ctx();
		Tensor* input_x = new WrappedRamTensor<float>({1, UTENSOR_INPUT_DIM}, (float*) input_data);
		return;
	}
}


extern "C" int utensor_test(int x) {
	return x * 2;
};
