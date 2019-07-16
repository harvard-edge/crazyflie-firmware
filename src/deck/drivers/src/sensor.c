//
// Created by vj-reddi on 6/18/19.
// TSL2591 Adafruit Lightsensor driver for Bitcraze Crazyflie
//

//#define DEBUG_MODULE "sensordeck"

#include "debug.h"
#include "deck.h"
#include "i2cdev.h"
#include "task.h"
#include "system.h"
#include "sysload.h"
#include "FreeRTOS.h"
#include "commander.h"
#include "sensor.h"

// --- function to get status in byte format --- //
// --- Bit 0 is ALS Valid. Bit 4 is ALS Interrupt. Bit 5 is No-persist Interrupt. --- //
void get_status(void){
    uint8_t x;
    i2cdevReadByte(I2C1_DEV,TSL2591_ADDR,TSL2591_COMMAND_BIT | TSL2591_REGISTER_DEVICE_STATUS,&x); //chan0
    //DEBUG_PRINT("STATUS : %u \n",x);
}

// --- set interrupt settings ---- //
void init_interrupt(void){

    uint16_t lowerThreshold = 0;
    uint16_t upperThreshold = 65535;
    tsl2591Persist_t persist = TSL2591_PERSIST_ANY;
    enable();
    i2cdevWriteByte(I2C1_DEV, TSL2591_ADDR, TSL2591_CLEAR_INT,0);
    i2cdevWriteByte(I2C1_DEV, TSL2591_ADDR, TSL2591_COMMAND_BIT | TSL2591_REGISTER_PERSIST_FILTER,TSL2591_PERSIST_60);
    i2cdevWriteByte(I2C1_DEV, TSL2591_ADDR, TSL2591_COMMAND_BIT | TSL2591_REGISTER_THRESHOLD_AILTL,lowerThreshold);
    i2cdevWriteByte(I2C1_DEV, TSL2591_ADDR, TSL2591_COMMAND_BIT | TSL2591_REGISTER_THRESHOLD_AILTH,lowerThreshold >> 8);
    i2cdevWriteByte(I2C1_DEV, TSL2591_ADDR, TSL2591_COMMAND_BIT | TSL2591_REGISTER_THRESHOLD_AIHTL,upperThreshold);
    i2cdevWriteByte(I2C1_DEV, TSL2591_ADDR, TSL2591_COMMAND_BIT | TSL2591_REGISTER_THRESHOLD_AIHTH,upperThreshold >> 8);
    disable();

}

// --- write enable to sensor (switch on) --- //
void enable(void){
    if(!i2cdevWriteByte(I2C1_DEV, TSL2591_ADDR, TSL2591_COMMAND_BIT | TSL2591_REGISTER_ENABLE,
                    TSL2591_ENABLE_POWERON | TSL2591_ENABLE_AEN | TSL2591_ENABLE_AIEN | TSL2591_ENABLE_NPIEN)){
        DEBUG_PRINT("Enable write FAILED \n");
    }
}

// --- write disable to sensor (switch off) --- //
void disable(void){
    if(!i2cdevWriteByte(I2C1_DEV, TSL2591_ADDR, TSL2591_COMMAND_BIT | TSL2591_REGISTER_ENABLE, TSL2591_ENABLE_POWEROFF))
    {
        DEBUG_PRINT("Disable write FAILED \n");
    }
}

// -- call this function to initialize the sensor --- //
void TSL2591_init(void){
    DEBUG_PRINT("Get ready...\n");
    systemWaitStart();

    int found_sens = 0;
    //Using deckport
    if(i2cdevInit(I2C1_DEV))
    {
        DEBUG_PRINT("I2C INIT \n");
    }
    else{
        DEBUG_PRINT("I2C ERROR \n");
    }

    // getting ID
    uint8_t id;
    while(found_sens==0){
        i2cdevReadByte(I2Cx,TSL2591_ADDR,TSL2591_COMMAND_BIT|TSL2591_REGISTER_DEVICE_ID,&id);
        // check ID
        if( id == 80)
        {
            DEBUG_PRINT("TSL2591 FOUND!\n");
            found_sens =1;
        }
        else{
            DEBUG_PRINT("Looking for sensor... \n");
        }
        vTaskDelay(M2T(200));

    }


    //get_status(); //show interrupt status

    // writing gain and integration to device
    enable();
    if(i2cdevWriteByte(I2Cx, TSL2591_ADDR, TSL2591_COMMAND_BIT | TSL2591_REGISTER_CONTROL,
                       TSL2591_INTEGRATIONTIME_300MS | TSL2591_GAIN_MED)){
        disable();
        DEBUG_PRINT("Gain and integration time set. \n");
    }
    //disable();
    init_interrupt(); //turning off interupts
    enable();

}
// returns light intensity
// mode: 0 = full spectrum
// mode: 1 = IR Only
uint16_t read_TSL2591(uint8_t mode)
{

    get_status();   //get interrupt status

    // vars to read data to
    uint8_t test_flag;
    uint16_t x;
    uint16_t y;
    //read single sensor value
    //enable();
    if(i2cdevRead16(I2Cx,TSL2591_ADDR,I2CDEV_NO_MEM_ADDR,1,&test_flag)) //test if still online
    {
        //enable();
        // allow ADC to do its work
        for (uint8_t d=0; d<=TSL2591_INTEGRATIONTIME_300MS; d++)
        {
            vTaskDelay(M2T(120));
        }


        i2cdevRead(I2Cx,TSL2591_ADDR,TSL2591_REGISTER_CHAN0_LOW,2,&y); //chan0 - Full + IR
        i2cdevRead(I2Cx,TSL2591_ADDR,TSL2591_REGISTER_CHAN1_LOW,2,&x); //chan1 - IR only
        //disable();
        get_status();
        //DEBUG_PRINT("Full %u, Infra: %u\n",y,x);
    }
    else {
        DEBUG_PRINT("Device lost.. \n");
    }
    //mode 0: return full
    if(mode ==0){
        return y;
    }
    //mode 1: return IR
    else if (mode == 1){
        return x;
    }

}

