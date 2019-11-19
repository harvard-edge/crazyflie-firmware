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
#include "sensor.h"

// uTensor related machine learning
#include "advanced_ml.h"


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
	static setpoint_t setpoint;
	systemWaitStart();



//    DEBUG_PRINT("Now trying to call uTensor\n");
//    uint64_t start_time, end_time;
//    int num_runs = 20;
//
//    start_time = usecTimestamp();
//    float test_measurements[6];
//    for (int i = 0 ; i < num_runs; i++) {
//        for (int j = 0; j < 6; j++) {
//            test_measurements[j] = i * 0.3;
//        }
//        int res = inference(test_measurements, 5);
//    }
//    end_time = usecTimestamp();
//    DEBUG_PRINT("Time taken for %d inferences: %lld us\n", num_runs, (end_time - start_time));
//    DEBUG_PRINT("Time taken per inference: %lld us\n", (end_time - start_time) / num_runs);
//
//    DEBUG_PRINT("FINISHED\n");




	float r[3];
	float input[6] = {0.599,0.325999,0.08,0.32,0.74,0.74};
    uint16_t sensor_read = 0;
    uint8_t sensor_mode = 0;
	DEBUG_PRINT("Starting the advanced machine learning...\n");
    float HOVER_HEIGHT = 0.8;
    // Start in the air before doing ML
    //flyVerticalInterpolated(0.0f, HOVER_HEIGHT, 6000.0f);
    vTaskDelay(M2T(500));
    distances d;
    getDistances(&d);
    TSL2591_init();
    uint8_t rand_arr[10] = {2, 2, 1, 1, 2, 1, 1, 1, 2, 1};

    float c = 0;
    float c_f = 1.0;

    float old_dist = 0;
    float dist = 0;
    int yaw = 0;
    int command = 0;
    float ESCAPE_SPEED = 0.5;
    uint8_t goal_count = 0;
    uint8_t found_goal = FALSE;
    uint8_t rand_count = 0;

    // CWrappedRamTensor* wrapped_input = CWrappedRamTensor_create(input);

    for (int j = 0; j < 10000; j++) {
        getDistances(&d);
        if (d.up / 10 < 20) {
            break;
        }

        vTaskDelay(M2T(300));
        sensor_read = read_TSL2591(sensor_mode);
        dist = get_distance(sensor_read);

        c = dist - old_dist;
        c_f = 0.9 * c_f + 0.1 * c;

        old_dist = dist;
        //vTaskDelay(M2T(300));
        //DEBUG_PRINT("FRONT : %f\n",(float)(d.front)*0.001);
        input[0] = d.right * 0.001;
        input[1] = d.front * 0.001;
        input[2] = d.left * 0.001;
        input[3] = d.back * 0.001;
        input[4] = 0.5 * (c - c_f);
        input[5] = 2 * c_f - 1;


//        {
//            vTaskDelay(M2T(200));
//            // inference(input, 6, &r[0]);
//            CWrappedRamTensor *wrapped_input = CWrappedRamTensor_create(&input[0]);
//
//            DEBUG_PRINT("Free heap: %d bytes\n", xPortGetFreeHeapSize());
//
//            // CWrappedRamTensor_set(wrapped_input, &input[0]);
//            inference_new(wrapped_input, &r[0]);
//        }

        {
            vTaskDelay(M2T(50));
            inference_test(input, 6, &r[0]);
            DEBUG_PRINT("Iteration %d\n", j);
        }

        if (DEBUG_VALUES) {
            for (int i = 0; i < 6; i++) {
                DEBUG_PRINT("%f \n", input[i]);
            }
        }


        // DEBUG_PRINT("%i \n",res);
        // command = argmax(res, 3);
        command = argmax_float(r, 3);




		// DEBUG_PRINT("Command: %i\n", command);
        switch (command) {
          case 0:
//              setHoverSetpoint(&setpoint, ESCAPE_SPEED, 0, HOVER_HEIGHT, (float)(yaw));
              setHoverSetpoint(&setpoint, ESCAPE_SPEED, 0, HOVER_HEIGHT, 0);
              commanderSetSetpoint(&setpoint, 3);
              vTaskDelay(M2T(150));
              break;
          case 1:
//              yaw_incr(&yaw);
//              setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT,(float)(yaw));
                setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT, 54);
                commanderSetSetpoint(&setpoint, 3);
              vTaskDelay(M2T(150));

//              vTaskDelay(M2T(100));
              break;
          case 2:
//                yaw_decr(&yaw);
//                setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT, (float)(yaw));
                setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT, -54);
                commanderSetSetpoint(&setpoint, 3);
                vTaskDelay(M2T(150));

//              vTaskDelay(M2T(100));
              break;
          default:
//                setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT, (float)(yaw));
                setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT, 0);
                commanderSetSetpoint(&setpoint, 3);
                vTaskDelay(M2T(150));
              break;
      }
//
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
