/* TFMicro Test Script: Tests to see if we can control the motors using the
attached FlowDeck and get measurements. */
#include "deck.h"
#include "system.h"
#include "commander.h"

#include "FreeRTOS.h"
#include "task.h"


#include "debug.h"

#define DEBUG_MODULE "SEQ"

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


/**
 * Takes in commands of the form of an array. Each array will be read
 * in the format 
 *  x_dst, y_dst, z, yawRate, time (ms) to give that command, ...
 * x_dst controls forward and back, positive goes forward, negative goes back
 * y_dst controls left and right, positive goes left, negative goes right
 * n represents how many tuples of 5 numbers there are.
 */
static void commandsToSequence(setpoint_t *setpoint, float* cmds, int n) {
  for (int i = 0; i < n; i++) {
    float dx       = cmds[5 * i + 0];
    float dy       = cmds[5 * i + 1];
    float z        = cmds[5 * i + 2];
    float yawrate  = cmds[5 * i + 3];
    float cmd_time = cmds[5 * i + 4];

    for (int j = 0; j < cmd_time / 100; j++) {
      setHoverSetpoint(setpoint, dx, dy, z, yawrate);
      commanderSetSetpoint(setpoint, 3);
      vTaskDelay(M2T(100));
    }
  }
}

static void sequenceTask()
{
  static setpoint_t setpoint;

  systemWaitStart();

  vTaskDelay(M2T(1000));
  DEBUG_PRINT("Starting sequence ...\n");

  /*
  // this floats and goes in one circle, then lands
  float commands[] = {
    0, 0, 0.2, 0, 2000,
    0, 0, 0.5, 0, 4000,
    0.8, 0, 0.5, 72.0, 8000,
    0, 0, 0.5, 0, 2000,
    0, 0, 0.3, 0, 1000,
    0, 0, 0.1, 0, 1000,
  };
  int num_commands = 6;
  */

  float commands[] = {
    0, 0, 0.6, 0, 4000,
    0, 0, 1.2, 0, 4000,
    0.5, 0, 1.2, 0, 2000,
    -0.5, 0, 1.2, 0, 2000,
    0, 0.5, 1.2, 0, 2000,
    0, -0.5, 1.2, 0, 2000,
    0, 0, 1.2, 0, 3000,
    0, 0, 0.6, 0, 3000,
    0, 0, 0.3, 0, 3000,
  };
  int num_commands = 9;

  commandsToSequence(&setpoint, commands, num_commands);

  // end of routine.
  for (;;) {
    vTaskDelay(1000);
  }
}

static void sequenceInit() {
  xTaskCreate(sequenceTask, "sequence", 2*configMINIMAL_STACK_SIZE, NULL,
              /*priority*/3, NULL);
}

static bool sequenceTest() {
  return true;
}

const DeckDriver sequence_deck = {
  .vid = 0,
  .pid = 0,
  .name = "bcSequence",

  .usedGpio = 0,  // FIXME: set the used pins

  .init = sequenceInit,
  .test = sequenceTest,
};

DECK_DRIVER(sequence_deck);
