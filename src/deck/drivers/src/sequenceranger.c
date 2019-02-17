/* TFMicro Test Script: Tests to see if we can control the motors using the
attached FlowDeck and get measurements. */
#include "deck.h"
#include "system.h"
#include "commander.h"
#include "range.h"  // get the 6axis distance measurements

#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"

#define DEBUG_MODULE "SEQ"

// Gets the ranges from the multiranger deck extension
// in meters.
static void getRanges(float *front, float *back, float *left, float *right) {
  *front = rangeGet(rangeFront);
  *back = rangeGet(rangeBack);
  *left = rangeGet(rangeLeft);
  *right = rangeGet(rangeRight);
}

static void setHoverSetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate)
{
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;

  setpoint->mode.yaw = modeVelocity;
  setpoint->attitudeRate.yaw = yawrate;

  setpoint->mode.x = modeVelocity;
  setpoint->mode.y = modeVelocity;
  setpoint->velocity.x = vx;
  setpoint->velocity.y = vy;

  setpoint->velocity_body = true;
}

static void sequenceRangerTask()
{
  static setpoint_t setpoint;

  systemWaitStart();

  vTaskDelay(M2T(1000));
  DEBUG_PRINT("Starting sequence ...\n");

  float front, back, left, right;

  for (int j = 0; j < 100; j++) {
    getRanges(&front, &back, &left, &right);
    setHoverSetpoint(&setpoint, 0, 0, 0.9, 200);
    commanderSetSetpoint(&setpoint, 3);
    vTaskDelay(M2T(100));
  }

  // end of routine.
  for (;;) {
    vTaskDelay(1000);
  }
}

static void sequenceRangerInit() {
  xTaskCreate(sequenceRangerTask, "sequenceRanger", 2*configMINIMAL_STACK_SIZE, NULL,
              /*priority*/3, NULL);
}

static bool sequenceRangerTest() {
  return true;
}

const DeckDriver sequence_ranger_deck = {
  .vid = 0,
  .pid = 0,
  .name = "tfclassic",

  .usedGpio = 0,  // FIXME: set the used pins

  .init = sequenceRangerInit,
  .test = sequenceRangerTest,
};

DECK_DRIVER(sequence_ranger_deck);
