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


#define TENSOR_ALLOC_SIZE 6000

static void check_multiranger_online() {
	DEBUG_PRINT("Checking if multiranger ToF sensors online...\n");

	distances d;
	for (int j = 0; j < 10; j++) {
		getDistances(&d);
		vTaskDelay(M2T(100));
	}
	if (d.left == 0 && d.right == 0) {
		// most likely the ranger deck isn't attached correctly
		DEBUG_PRINT("Most likely ranger deck not attached correctly\n");
		for (;;) {
			vTaskDelay(M2T(1000));
		} 
	}
}

static void tfMicroDemoTask()
{
	static setpoint_t setpoint;
	systemWaitStart();

	const CTfLiteModel* model = CTfLiteModel_create();
	uint8_t tensor_alloc[TENSOR_ALLOC_SIZE];
	int r[3];
	uint8_t input[5] = {40, 120, 120, 120, 120};

	DEBUG_PRINT("Starting the advanced machine learning...\n");
  float HOVER_HEIGHT = 1.1;
  
  // Start in the air before doing ML 
  flyVerticalInterpolated(0.0f, HOVER_HEIGHT, 6000.0f);
  vTaskDelay(M2T(500));
	distances d;
  int command = 0;
  float ESCAPE_SPEED = 0.1;
  for (int j = 0; j < 1000; j++) {
    getDistances(&d);

        /* Defining the input to the network*/
        // obs avoidance will
//		input[0] = (uint8_t) ( d.front / 10);
//		input[1] = (uint8_t) ( d.right / 10);
//		input[2] = (uint8_t) ( d.back / 10);
//		input[3] = (uint8_t) ( d.left / 10);
//		input[4] = (uint8_t) ( d.up / 10);
//		input[5] = (uint8_t) ( d.down / 10);


		input[0] = (uint8_t) ( d.right / 10);
		input[1] = (uint8_t) ( d.front / 10);
		input[2] = (uint8_t) ( d.left / 10);
		input[3] = (uint8_t) ( d.down / 10);
		input[4] = (uint8_t) (5);

		DEBUG_PRINT("LASERS: %i %i %i %i",input[0],input[1],input[2],input[3]);

//        input[0] = (uint8_t)(1);
//        input[1] = (uint8_t)(1);
//        input[2] = (uint8_t)(1);
//        input[3] = (uint8_t)(1);
//        input[4] = (uint8_t)(1);

      CTfInterpreter_simple_fc(model, tensor_alloc, TENSOR_ALLOC_SIZE, input, r);
		DEBUG_PRINT("Q-Vals: %i %i %i \n",r[0],r[1],r[2]);
		command = argmax(r, 3);
		DEBUG_PRINT("Command: %i\n", command);
		switch (command) {
				case 0:
				    setHoverSetpoint(&setpoint, ESCAPE_SPEED, 0, HOVER_HEIGHT, 0);
					break;
				case 1:
				    setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT, 60);
                    vTaskDelay(M2T(100));
                    break;
				case 2:
				    setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT, -60);
				    vTaskDelay(M2T(100));
                    break;

                default:
                    setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT, 0);
                    break;
		} 
		commanderSetSetpoint(&setpoint, 3);
    vTaskDelay(M2T(40));
  }
  
	// Slowly lower to a safe height before quitting, or else CRASH!
  flyVerticalInterpolated(HOVER_HEIGHT, 0.1f, 1000.0f);
	for (;;) { vTaskDelay(M2T(1000)); }
}

static void init() {
	xTaskCreate(tfMicroDemoTask, "tfMicroDemoTask",
		4500 /* Stack size in terms of WORDS (usually 4 bytes) */,
		NULL, /*priority*/3, NULL);
}

static bool test() {
	return true;
}

const DeckDriver tf_micro_demo = {
	.vid = 0,
	.pid = 0,
	.name = "tfMicroDemo",

	.usedGpio = 0,  // FIXME: set the used pins

	.init = init,
	.test = test,
};

DECK_DRIVER(tf_micro_demo);
