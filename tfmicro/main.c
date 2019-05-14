#include <stdio.h>
#include "machinelearning.h"

int main(int argc, char** argv) {
    printf("Hello world, this is advanced machine learning\n");
    int i = 1;
    printf("Test function doubles a number: %d\n", i);
    printf("Double: %d\n", testDoubleFunction(i));
    printf("Starting machine learning test from C++ library...\n");

    machine_learning_test(0);

    return 0;
}
