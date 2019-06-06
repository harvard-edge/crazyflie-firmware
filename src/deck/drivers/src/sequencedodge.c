/* TFMicro Test Script: Tests to see if we can control the motors using the
attached FlowDeck and get measurements. */
#include "deck.h"
#include "system.h"
#include "commander.h"
#include "range.h"  // get the 6axis distance measurements

#include "FreeRTOS.h"
#include "task.h"
#include "sequencelib.h"

#include "debug.h"

#define DEBUG_MODULE "SEQ"


static void sequenceDodgeTask()
{
  static setpoint_t setpoint;

  systemWaitStart();

  vTaskDelay(M2T(500));
  DEBUG_PRINT("Starting sequence ...\n");

  float HOVER_HEIGHT = 0.6;

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
  
  flyVerticalInterpolated(0.0f, HOVER_HEIGHT, 3000.0f);

  int ticksDetected = 0;
  int maxTicksDetected = 16;
  float ESCAPE_SPEED = 0.1;
  for (int j = 0; j < 5000; j++) {
    getDistances(&d);
    // printDistances(d);
    ticksDetected = min(ticksDetected, maxTicksDetected);
    // If there is someone coming from above or the sides!!!!!
    if (d.up < 500 || d.left < 500 || d.right < 500 || d.front < 500 || d.back < 500) {
      ticksDetected++;
      // find the direction with the largest distance of free space
      if (d.left >= d.right && d.left >= d.front && d.left >= d.back) {
        setHoverSetpoint(&setpoint, 0, ticksDetected * ESCAPE_SPEED, HOVER_HEIGHT, 0); 
        commanderSetSetpoint(&setpoint, 3);
      }
      else if (d.right >= d.left && d.right >= d.front && d.right >= d.back) {
        setHoverSetpoint(&setpoint, 0, -ticksDetected * ESCAPE_SPEED, HOVER_HEIGHT, 0); 
        commanderSetSetpoint(&setpoint, 3);
      }
      else if (d.front >= d.left && d.front >= d.right && d.front >= d.back) {
        setHoverSetpoint(&setpoint, ticksDetected * ESCAPE_SPEED, 0, HOVER_HEIGHT, 0); 
        commanderSetSetpoint(&setpoint, 3);
      }
      else {
        setHoverSetpoint(&setpoint, -ticksDetected * ESCAPE_SPEED, 0, HOVER_HEIGHT, 0); 
        commanderSetSetpoint(&setpoint, 3);
      }
    } else {
      ticksDetected = 0;
      setHoverSetpoint(&setpoint, 0, 0, HOVER_HEIGHT, 0); 
      commanderSetSetpoint(&setpoint, 3); 
    }
    vTaskDelay(M2T(10));
  }
  
  flyVerticalInterpolated(HOVER_HEIGHT, 0.2f, 3000.0f);
  
  // end of routine.
  for (;;) {
    vTaskDelay(M2T(1000));
  }
}

static void sequenceDodgeInit() {
  xTaskCreate(sequenceDodgeTask, "sequenceDodge", 2*configMINIMAL_STACK_SIZE,
              NULL, /*priority*/3, NULL);
}

static bool sequenceDodgeTest() {
  return true;
}

const DeckDriver sequence_dodge_deck = {
  .vid = 0,
  .pid = 0,
  .name = "sequenceDodge",

  .usedGpio = 0,  // FIXME: set the used pins

  .init = sequenceDodgeInit,
  .test = sequenceDodgeTest,
};

DECK_DRIVER(sequence_dodge_deck);
