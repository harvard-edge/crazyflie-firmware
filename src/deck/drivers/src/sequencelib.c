/** 
 * sequencelib.c
 * 
 * A few nice helper functions to help us fly using the six forms of measurements
 * given by the Ranger deck as well as the Flow camera (points downward). 
 */
#include "sequencelib.h"
#include "range.h"
#include "commander.h"

int argmax(int* arr, int size) {
	int curr = 0;
	for (int i = 0; i < size; i++) {
		curr = max(curr, arr[i]);		
	}

	for (int i = 0; i < size; i++) {
		if (arr[i] == curr) {
			return i;
		}
	}
	return 0;
}

/* argmax for floats*/
int argmax_f(float* arr, int size) {
    int curr = 0;
    for (int i = 0; i < size; i++) {
        curr = max(curr, arr[i]);
    }

    for (int i = 0; i < size; i++) {
        if (arr[i] == curr) {
            return i;
        }
    }
    return 0;
}

void getDistances(distances* d) {
    d->front = rangeGet(rangeFront);
    d->back = rangeGet(rangeBack);
    d->left = rangeGet(rangeLeft);
    d->right = rangeGet(rangeRight);
    d->up = rangeGet(rangeUp);
    d->down = rangeGet(rangeDown);
}

void printDistances(distances d) {
    consolePrintf("Front %.2f; Back %.2f; Left %.2f; Right %.2f; Top %.2f; Bottom %.2f;\n",
                  (double) d.front, (double) d.back , (double) d.left,
                  (double) d.right, (double) d.up, (double) d.down);
}

void setHoverSetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate) {
    setpoint->mode.z = modeAbs;
    setpoint->position.z = z;

//    setpoint->mode.yaw = modeAbs;
    setpoint->mode.yaw = modeVelocity;
//    setpoint->attitude.yaw = yawrate;
    setpoint->attitudeRate.yaw = yawrate;

    setpoint->mode.x = modeVelocity;
    setpoint->mode.y = modeVelocity;
    setpoint->velocity.x = vx;
    setpoint->velocity.y = vy;

    setpoint->velocity_body = true;
}

void flyVerticalInterpolated(float startz, float endz, float interpolate_time) {
    setpoint_t setpoint;
    int CMD_TIME = 100;   // new command ever CMD_TIME seconds
    int NSTEPS = (int) interpolate_time / CMD_TIME; 
    for (int i = 0; i < NSTEPS; i++) {
        float curr_height = startz + (endz - startz) * ((float) i / (float) NSTEPS);
        setHoverSetpoint(&setpoint, 0, 0, curr_height, 0); 
        commanderSetSetpoint(&setpoint, 3);
        vTaskDelay(100);
    }
}