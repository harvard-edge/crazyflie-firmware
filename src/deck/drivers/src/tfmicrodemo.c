/* TFMicro Test Script: Tests to see the delay incurred by TF-Micro and
how fast the chip can process these neural networks */
#include <time.h>
#include <stdlib.h>

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
#include "sensor.h"

// uTensor related machine learning


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

int argmax_float(float* array, int size){
    float max = array[0];
    int max_ind = 0;
    for (int i = 1; i< size; i++){
        if (array[i]>max){
            max = array[i];
            max_ind = i;
        }
    }
    return max_ind;
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
    int command = 0;
	static setpoint_t setpoint;
	systemWaitStart();

	float ESCAPE_SPEED = 1.0;
    float HOVER_HEIGHT = 1.0;
    float rotate_threshold = 2.0;
    // Start in the air before doing ML
    //flyVerticalInterpolated(0.0f, HOVER_HEIGHT, 6000.0f);
    vTaskDelay(M2T(500));
    distances d;
    getDistances(&d);
    float front_sensor = d.front*0.001;
    int yaw = 0;
    srand(time(NULL));
    int r = rand();

    // main loop
    for (int j = 0; j < 10000; j++) {
        getDistances(&d);

        // safety statement -- kill drone when hand is over < 20 cm
        if(d.up/10 < 20)
        {
            break;
        }

        front_sensor = d.front*0.001;    // used for obs avoidance

        if (front_sensor < rotate_threshold) {
            yaw  = rand()%33;
            command = rand()%2+1;
        }
        else{
            command = 0;
        }

        vTaskDelay(M2T(200));

        switch (command) {
          case 0:
              setHoverSetpoint(&setpoint, ESCAPE_SPEED, 0, HOVER_HEIGHT, 0);
              commanderSetSetpoint(&setpoint, 3);
              vTaskDelay(M2T(100));
              break;
          case 1:
              for (int i = 0; i<yaw;i++) {
                  setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT,54);
                  commanderSetSetpoint(&setpoint, 3);
                  vTaskDelay(M2T(100));
              }
              break;
            case 2:
                for (int i = 0; i<yaw;i++) {
                    setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT,-54);
                    commanderSetSetpoint(&setpoint, 3);
                    vTaskDelay(M2T(100));
                }
                break;
      }
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
