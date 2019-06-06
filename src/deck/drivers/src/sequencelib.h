#ifndef SEQUENCE_LIB_GUARD_H 
#define SEQUENCE_LIB_GUARD_H

/** 
 * sequencelib.h
 * 
 * A few nice helper functions to help us fly using the six forms of measurements
 * given by the Ranger deck as well as the Flow camera (points downward). 
 */
#include "range.h"
#include "commander.h"
#include "debug.h"
#include "system.h"

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
    
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


typedef struct ranger_distances {
    float front;
    float back;
    float left;
    float right;
    float up;
    float down;
} distances;

extern void getDistances(distances* d);
extern void printDistances(distances d);

extern void setHoverSetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate);
extern void flyVerticalInterpolated(float startz, float endz, float interpolate_time);

extern int argmax(int*, int);
extern int argmax_f(float*, int);

#endif