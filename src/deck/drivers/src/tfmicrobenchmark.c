/* TFMicro Test Script: Tests to see the delay incurred by TF-Micro and
how fast the chip can process these neural networks */
#include "deck.h"
#include "system.h"
#include "commander.h"
#include "range.h"  // get the 6axis distance measurements

#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"
#include "machinelearning.h"
#include "sysload.h"
#include "sequencelib.h"


static void tfMicroBenchTask()
{
	systemWaitStart();
	DEBUG_PRINT("Original heap space: %d\n", xPortGetFreeHeapSize());
	consolePrintf("Try allocating actual ML model and tensor intermediaries (fc)\n");
	const CTfLiteModel* model = CTfLiteModel_create();
	uint8_t tensor_alloc[12000];
	consolePrintf("Finish allocating actual ML model (fc)\n");
	vTaskDelay(M2T(1000));
	// systemDump();
	DEBUG_PRINT("Current heap space: %d\n", xPortGetFreeHeapSize());

	DEBUG_PRINT("Try allocating actual ML model (fc)\n");
	DEBUG_PRINT("Version number: %d\n", CTfLiteModel_version(model));
	vTaskDelay(M2T(1000));
	
	uint64_t start, end;
	distances d;
	int r[9];
	DEBUG_PRINT("Dimension 0 of input: %d\n", CTfLiteModel_dimensions(model, tensor_alloc, 12000, 0));
	DEBUG_PRINT("Dimension 1 of input: %d\n", CTfLiteModel_dimensions(model, tensor_alloc, 12000, 1));
	vTaskDelay(M2T(1000));
	model_type input[6] = {40, 120, 120, 120, 120, 120};
	int command;
	int NUM_RUNS_PER_TIMING = 100;
	for (int i = 0 ; i < 10000; i++) {
		start = usecTimestamp();
		for (int j = 0; j < NUM_RUNS_PER_TIMING; j++) {
			getDistances(&d);
			input[0] = (model_type) ( d.front / 10);
			input[1] = (model_type) ( d.right / 10);
			input[2] = (model_type) ( d.back / 10);
			input[3] = (model_type) ( d.left / 10);
			input[4] = 100; 
			input[5] = 100;
			// CTfInterpreter_simple_fc(model, tensor_alloc, 12000, input, r);
			CTfInterpreter_simple_conv(model, tensor_alloc, 12000, input, 6, r, 9);
		}
		end = usecTimestamp();
		DEBUG_PRINT("time taken: %lld us\n", (end - start));
		vTaskDelay(M2T(1000));
	}
	end = usecTimestamp();
	DEBUG_PRINT("Elapsed time: %lld\n", (long long) (end - start));
	DEBUG_PRINT("Are you ready for this?\n");

	/**
	 * Function that checks how long it takes to run ML models given the current
	 * voltage. We can run the battery dry and see if it changes the amount of time
	 * each inferencing step takes for different models.
	 */
	for (;;) {vTaskDelay(M2T(1000));}
}

static void init() {
	xTaskCreate(tfMicroBenchTask, "tfMicroBenchTask",
		5000 /* Stack size in terms of WORDS (usually 1 or 2 bytes) */,
		NULL, /*priority*/3, NULL);
}

static bool test() {
	return true;
}

const DeckDriver tf_micro_benchmark = {
	.vid = 0,
	.pid = 0,
	.name = "tfMicroBench",
	.usedGpio = 0,
	.init = init,
	.test = test,
};

DECK_DRIVER(tf_micro_benchmark);
