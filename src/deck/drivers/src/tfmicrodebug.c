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


#define TENSOR_ALLOC_SIZE 10000

#define model_type uint8_t
// #define model_type float


static void tfMicroDebugTask()
{
	systemWaitStart();

	DEBUG_PRINT("Original heap space: %d\n", xPortGetFreeHeapSize());
	consolePrintf("Try allocating actual ML model and tensor intermediaries (fc)\n");
	const CTfLiteModel* model = CTfLiteModel_create();
	uint8_t tensor_alloc[TENSOR_ALLOC_SIZE];
	consolePrintf("Finish allocating actual ML model (fc)\n");
	vTaskDelay(M2T(1000));
	// systemDump();
	DEBUG_PRINT("Current heap space: %d\n", xPortGetFreeHeapSize());

	DEBUG_PRINT("Try allocating actual ML model (fc)\n");
	DEBUG_PRINT("Version number: %d\n", CTfLiteModel_version(model));
	vTaskDelay(M2T(1000));
	
	uint64_t start, end;
	distances d;
	start = usecTimestamp();
	int r[9];
	vTaskDelay(M2T(1000));
	model_type input[6] = {40, 120, 120, 120, 120, 120};
	int command;
	for (int i = 0 ; i < 10000; i++) {
		getDistances(&d);
		input[0] = (model_type) ( d.front / 10);
		input[1] = (model_type) ( d.right / 10);
		input[2] = (model_type) ( d.back / 10);
		input[3] = (model_type) ( d.left / 10);
		input[4] = 100; 
		input[5] = 100;
		DEBUG_PRINT("Inputs: %d %d %d %d %d %d\n", input[0], input[1], input[2],
			input[3], input[4], input[5]);
		CTfInterpreter_simple_fc(model, tensor_alloc, TENSOR_ALLOC_SIZE, input, r);
		DEBUG_PRINT("Result: %d %d %d %d %d %d %d %d %d\n", r[0], r[1], r[2], r[3],
			r[4], r[5], r[6], r[7], r[8]);
		command = argmax(r, 9);
		DEBUG_PRINT("Command: %d\n", command);
		vTaskDelay(M2T(1000));
	}
	end = usecTimestamp();
	DEBUG_PRINT("Elapsed time: %lld\n", (long long) (end - start));
	DEBUG_PRINT("Are you ready for this?\n");
	for (;;) { vTaskDelay(M2T(1000)); }
}

/* 2500 stack size too large and it crashes */
/* 1900 stack size seems alright */
static void init() {
	xTaskCreate(tfMicroDebugTask, "tfMicroDebugTask",
		5000 /* Stack size in terms of WORDS (usually 4 bytes) */,
		NULL, /*priority*/3, NULL);
}

static bool test() {
	return true;
}

const DeckDriver tf_micro_debug = {
	.vid = 0,
	.pid = 0,
	.name = "tfMicroDebug",

	.usedGpio = 0,  // FIXME: set the used pins

	.init = init,
	.test = test,
};

DECK_DRIVER(tf_micro_debug);
