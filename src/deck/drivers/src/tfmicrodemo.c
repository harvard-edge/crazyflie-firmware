/* TFMicro Test Script: Tests to see the delay incurred by TF-Micro and
how fast the chip can process these neural networks */
#include "deck.h"
#include "system.h"
#include "commander.h"
#include "range.h"  // get the 6axis distance measurements
#include "log.h"
#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"
#include "sysload.h"
#include "sequencelib.h"
#include "sensor.h"
#include "mlp_inference.h"
#include "deck_analog.h"
// uTensor related machine learning


#define TENSOR_ALLOC_SIZE 6000
#define SUBTRACT_VAL 60
#define STATE_LEN 5
#define NUM_STATES 4
#define YAW_INCR 8
//#define SENS_MIN 35000
#define SENS_MIN 0
#define SENS_MAX 65000
#define TRUE 1
#define FALSE 0
#define GOAL_THRES 245
#define GOAL_THRES_COUNT 3
#define DIST_MIN 90
#define RAND_ACTION_RATE 30

#define DEBUG_VALUES false

float R_s;

void yaw_incr(int *yaw){
    int yaw_out = *yaw + YAW_INCR;
    if(yaw_out>180){
        yaw_out -= 360;
    }
    *yaw = yaw_out;
    return;
}


void yaw_decr(int *yaw){
    int yaw_out = *yaw - YAW_INCR;
    if(yaw_out<-180){
        yaw_out += 360;
    }
    *yaw = yaw_out;
    return;
}



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


float get_distance(uint16_t sensor_read){
    if(sensor_read<SENS_MIN){
        sensor_read = SENS_MIN;
    }
    if(sensor_read>SENS_MAX){
        sensor_read=SENS_MAX;
    }
    float frac = 1.0*(float)(sensor_read-SENS_MIN)/(SENS_MAX-SENS_MIN) ;
    return frac;
}


static void update_state(uint8_t *meas_array, distances d,uint8_t dist){
    //Step 1: move entire array by 1 state
    for(int i = (STATE_LEN*NUM_STATES-1);i>=STATE_LEN;i--)
    {
        *(meas_array+i) = *(meas_array+i-STATE_LEN);
    }
    //Step2: update the first state
    *(meas_array) = (uint8_t) ( d.right * 0.06375);
    *(meas_array+1) = (uint8_t) ( d.front * 0.06375);
    *(meas_array+2) = (uint8_t) ( d.left * 0.06375);
    *(meas_array+3) = (uint8_t) ( d.back * 0.06375);
    *(meas_array+4) = (uint8_t) (dist);

}


static void tfMicroDemoTask()
{
    static setpoint_t setpoint;
	systemWaitStart();

    uint32_t pin = 10;
    int r[3];
    float HOVER_HEIGHT = 0.8;
    float raw_read;

    flyVerticalInterpolated(0.0f, HOVER_HEIGHT, 6000.0f);
    vTaskDelay(M2T(500));
    distances d;
    getDistances(&d);

    for (int j = 0; j < 100000; j++) {
        getDistances(&d);

        if (d.front/10 <20)
        {
            break;
        }
        raw_read = analogReadVoltage(pin);
        DEBUG_PRINT("Raw read %f \n",raw_read);
        R_s = (3.0/raw_read-1)*70;
//        DEBUG_PRINT("Gas read: %f \n",R_s);
        setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT, 0);
        commanderSetSetpoint(&setpoint, 3);
        vTaskDelay(M2T(150));

    }

  // flyVerticalInterpolated(HOVER_HEIGHT, 0.1f, 1000.0f);
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

LOG_GROUP_START(gas)
LOG_ADD(LOG_FLOAT,R,&R_s)
LOG_GROUP_STOP(gas)
