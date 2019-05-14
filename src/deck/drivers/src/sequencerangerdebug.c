/* TFMicro Test Script: Tests to see if we can control the motors using the
attached FlowDeck and get measurements. */
#include "deck.h"
#include "system.h"
#include "commander.h"
#include "range.h"  // get the 6axis distance measurements

#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"

#include "sequencelib.h"

// number of milliseconds that we wait for before getting 6DOF measurements
#define SEQUENCE_RANGER_DEBUG_POLLING 100
#define DEBUG_MODULE "SEQ"


static void sequenceRangerDebugTask()
{
  systemWaitStart();
  vTaskDelay(M2T(1000));
  DEBUG_PRINT("Starting sequence ...\n");

  distances d;
  for (;;) {
    getDistances(&d);
    printDistances(d);
    vTaskDelay(SEQUENCE_RANGER_DEBUG_POLLING);
  }
}

static void sequenceRangerDebugInit() {
  xTaskCreate(sequenceRangerDebugTask, "sequenceRangerDebug", 
              2*configMINIMAL_STACK_SIZE, NULL,
              /*priority*/3, NULL);
}

static bool sequenceRangerDebugTest() {
  return true;
}

const DeckDriver sequence_ranger_debug_deck = {
  .vid = 0,
  .pid = 0,
  .name = "sequenceRangerDebug",

  .usedGpio = 0,  // FIXME: set the used pins

  .init = sequenceRangerDebugInit,
  .test = sequenceRangerDebugTest,
};

DECK_DRIVER(sequence_ranger_debug_deck);
