# Tensorflow Micro on the Crazyflie 2.0

We will try to compile TF Micro on the Crazyflie 2.0, which definitely should
be able to handle the library size. The specs on the default Crazyflie:

    * 1 MB flash memory
    * Cortex M4 ARM chip

However, all of the default firmware is written in C. Therefore, in this segment
we will compile TF Micro for the microcontoller using C++ (`arm-none-eabi-g++`),
expose some libraries to C, and compile everything in the end in C so the
default firmware works (`arm-non-eabi-gcc`).


## Creating the TF Micro Model

We will be trying to use fully connected layers, and for testing we will train
a few models of varying size in `./train` and make sure that they work in
the regular C++ Tensorflow Lite environment.

## Testing out on Laptop

Since we are compiling tf-micro for ARM to deploy on the crazyflie, in order
to test the linking on computer it may be worth it to run the binary using
an emulator.

```bash
sudo apt get install qemu   # Ubuntu using APT
```

Then, you can run the binaries with

```bash
qemu-arm tfmicrotest
```
