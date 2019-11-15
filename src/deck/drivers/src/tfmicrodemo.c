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
#define SENS_MIN 11000
#define SENS_MAX 55000
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


uint8_t get_distance(uint16_t sensor_read){
    if(sensor_read<SENS_MIN){
        sensor_read = SENS_MIN;
    }
    if(sensor_read>SENS_MAX){
        sensor_read=SENS_MAX;
    }
    float frac = ((float)sensor_read-SENS_MIN)/(SENS_MAX-SENS_MIN) ;
    return (uint8_t)(frac*255);
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


    DEBUG_PRINT("HELLO WORLD\n");
    DEBUG_PRINT("THIS IS WFU\n");

    int test_int = 2;
    int y = utensor_test(test_int);
    DEBUG_PRINT("Original: %d\n", test_int);
    DEBUG_PRINT("Doubled with Library: %d\n", y);

    DEBUG_PRINT("Now trying to call uTensor\n");
    uint64_t start_time, end_time;
    int num_runs = 20;

    start_time = usecTimestamp();
    float test_measurements[6];
    for (int i = 0 ; i < num_runs; i++) {
        for (int j = 0; j < 6; j++) {
            test_measurements[j] = i * 0.3;
        }
        int res = inference(test_measurements, 5);
    }
    end_time = usecTimestamp();
    DEBUG_PRINT("Time taken for %d inferences: %lld us\n", num_runs, (end_time - start_time));
    DEBUG_PRINT("Time taken per inference: %lld us\n", (end_time - start_time) / num_runs);

    DEBUG_PRINT("FINISHED\n");



    /*
	const CTfLiteModel* model = CTfLiteModel_create();
	uint8_t tensor_alloc[TENSOR_ALLOC_SIZE];
	int r[3];
	uint8_t full_meas[20] = {40, 120, 120, 120, 120, 40, 120, 120, 120, 120, 40, 120, 120, 120, 120, 40, 120, 120, 120, 120};
	uint8_t input[5] = {40, 120, 120, 120, 120};
    uint16_t sensor_read = 0;
    uint8_t sensor_mode = 0;
	DEBUG_PRINT("Starting the advanced machine learning...\n");
    float HOVER_HEIGHT = 0.8;
    // Start in the air before doing ML
    flyVerticalInterpolated(0.0f, HOVER_HEIGHT, 6000.0f);
    vTaskDelay(M2T(500));
    distances d;
    getDistances(&d);
    TSL2591_init();
    uint8_t rand_arr[10] = {2, 2, 1, 1, 2, 1, 1, 1, 2, 1};


    uint8_t dist =0;
    int yaw = 0;
    int command = 0;
    float ESCAPE_SPEED = 0.3;
    uint8_t goal_count = 0;
    uint8_t found_goal = FALSE;
    uint8_t rand_count = 0;
    for (int j = 0; j < 10000; j++) {
        getDistances(&d);
        //DEBUG_PRINT("yaw: %d \n",yaw);

        // Defining the input to the networ
        // obs avoidance will
//		input[0] = (uint8_t) ( d.front / 10);
//		input[1] = (uint8_t) ( d.right / 10);
//		input[2] = (uint8_t) ( d.back / 10);
//		input[3] = (uint8_t) ( d.left / 10);
//		input[4] = (uint8_t) ( d.up / 10);
//		input[5] = (uint8_t) ( d.down / 10);
        if(d.up/10 < 20)
        {
            break;
        }
//        if(dist>GOAL_THRES){
//            goal_count++;
//            if(goal_count>=GOAL_THRES_COUNT){
//                break;
//            }
//        }
//        DEBUG_PRINT("%i \n",sensor_read);
//        dist = 128;
    //    vTaskDelay(M2T(500));
        sensor_read = read_TSL2591(sensor_mode);
        dist = get_distance(sensor_read);

        //vTaskDelay(M2T(300));
        //DEBUG_PRINT("FRONT : %f\n",(float)(d.front)*0.001);
		input[0] = (uint8_t) ( d.right* 0.06375);
		input[1] = (uint8_t) ( d.front * 0.06375);
		input[2] = (uint8_t) ( d.left * 0.06375);
		input[3] = (uint8_t) ( d.back * 0.06375);
		input[4] = (uint8_t) dist;
		update_state(&full_meas, d,dist);
		//DEBUG_PRINT("full meas: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i \n",full_meas[0],full_meas[1],full_meas[2],full_meas[3],full_meas[4],full_meas[5],full_meas[6],full_meas[7],
		 //       full_meas[8],full_meas[9], full_meas[10],full_meas[11],full_meas[12],full_meas[13],full_meas[14],full_meas[15],full_meas[16],full_meas[17],full_meas[18],full_meas[19]);
        DEBUG_PRINT("sensor %i \n",sensor_read);
        DEBUG_PRINT("dist %i \n",dist);
		// subtract from laser readings, this creates a save zone around objects
//		for(int i=0;i<4;i++)
//        {
//		    if(input[i]>SUBTRACT_VAL){
//		        input[i] = input[i] - SUBTRACT_VAL;
//		    }
//		    else{
//		        input[i] = 0;
//		    }
//        }
//		DEBUG_PRINT("LASERS: %i %i %i %i \n",input[0],input[1],input[2],input[3]);

//        input[0] = (uint8_t)(1);
//        input[1] = (uint8_t)(1);
//        input[2] = (uint8_t)(1);
//        input[3] = (uint8_t)(1);
//        input[4] = (uint8_t)(1);

        CTfInterpreter_simple_fc(model, tensor_alloc, TENSOR_ALLOC_SIZE, full_meas, r);

		//DEBUG_PRINT("Q-Vals: %i %i %i \n",r[0],r[1],r[2]);
		command = argmax(r, 3);

//        if(j%RAND_ACTION_RATE==0)
//        {
//            if(command != 0){
//                if(command ==1)
//                {
//                    command = 2;
//                }
//                else{
//                    command = 1;
//                }
//            }
//            else{
//                command = rand_arr[rand_count%10];
//                rand_count++;
//            }
//
//        }
		//DEBUG_PRINT("Command: %i\n", command);
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
                vTaskDelay(M2T(40));
              break;
      }
//
}

	// Slowly lower to a safe height before quitting, or else CRASH!

    */
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
