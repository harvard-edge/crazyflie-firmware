# Check which architecture we are compiling for. If we are compiling on ARM,
# we assume that we are runing a test, and we will compile with standard g++
# instead of cross compile


ifeq ($(shell uname -m),armv7l)
# Use these settings when on a raspberry pi
AS      = as
CC      = gcc
CXX    = g++
LD      = gcc
OBJCOPY = objcopy
OBJDUMP = objdump
SIZE    = size 
MAIN_TEST = main.c
else
AS      = $(GCC_BIN)arm-none-eabi-as
CC      = $(GCC_BIN)arm-none-eabi-gcc
CXX    = $(GCC_BIN)arm-none-eabi-g++
LD      = $(GCC_BIN)arm-none-eabi-gcc
OBJCOPY = $(GCC_BIN)arm-none-eabi-objcopy
OBJDUMP = $(GCC_BIN)arm-none-eabi-objdump
SIZE    = $(GCC_BIN)arm-none-eabi-size 
endif

PROCESSOR = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16


SRCS := \
tensorflow/lite/experimental/micro/micro_error_reporter.cc \
tensorflow/lite/experimental/micro/micro_mutable_op_resolver.cc \
tensorflow/lite/experimental/micro/simple_tensor_allocator.cc \
tensorflow/lite/experimental/micro/debug_log.cc \
tensorflow/lite/experimental/micro/debug_log_numbers.cc \
tensorflow/lite/experimental/micro/micro_interpreter.cc \
tensorflow/lite/experimental/micro/kernels/depthwise_conv.cc \
tensorflow/lite/experimental/micro/kernels/softmax.cc \
tensorflow/lite/experimental/micro/kernels/all_ops_resolver.cc \
tensorflow/lite/experimental/micro/kernels/fully_connected.cc \
tensorflow/lite/c/c_api_internal.c \
tensorflow/lite/core/api/error_reporter.cc \
tensorflow/lite/core/api/flatbuffer_conversions.cc \
tensorflow/lite/core/api/op_resolver.cc \
tensorflow/lite/kernels/kernel_util.cc \
tensorflow/lite/kernels/internal/quantization_util.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/preprocessor.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/no_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/yes_features_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.cc \
tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.cc \
machinelearning.cpp \
tfmicro_models.cc \
main.c \

OBJS := \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SRCS))))

RASPIOBJS = OBJS
RASPIOBJS += main.c

INCLUDES := \
-I. \
-I./third_party/gemmlowp \
-I./third_party/flatbuffers/include

CCFLAGS += -fpermissive -Os
CXXFLAGS += -Os -fpermissive -DNDEBUG -std=c++11 -g -DTF_LITE_STATIC_MEMORY
CXXFLAGS += -fno-math-errno -DARM_MATH_CM4 -D__FPU_PRESENT=1 -D__TARGET_FPU_VFP
CCFLAGS += -fno-math-errno -DARM_MATH_CM4 -D__FPU_PRESENT=1 -D__TARGET_FPU_VFP
#Flags required by the ST library
CXXFLAGS += -DSTM32F4XX -DSTM32F40_41xxx -DHSE_VALUE=8000000 -DUSE_STDPERIPH_DRIVER
CCFLAGS += -DSTM32F4XX -DSTM32F40_41xxx -DHSE_VALUE=8000000 -DUSE_STDPERIPH_DRIVER
PROCESSOR = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16

COMPILE_CC = $(CC)
COMPILE_CXX = $(CXX)
COMPILE_CCFLAGS = $(CCFLAGS) $(PROCESSOR) -Os
COMPILE_CXXFLAGS = $(CXXFLAGS) $(PROCESSOR) -Os

LDFLAGS += -lm 
LDFLAGS += --specs=nosys.specs --specs=nano.specs 
LDFLAGS += $(PROCESSOR) -Wl,-Map=$(PROG).map,--cref,--gc-sections,--undefined=uxTopUsedPriority 

# Use the DEBUG flag to compile on x86 for testing purposes to see if
# linking works on a more accessible device.
%.o: %.cc
	$(COMPILE_CXX) $(COMPILE_CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(COMPILE_CXX) $(COMPILE_CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(COMPILE_CC) $(COMPILE_CCFLAGS) $(INCLUDES) -c $< -o $@

tfmicrotest : $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@


raspitest: $(RASPIOBJS)
	$(CXX) $(LDFLAGS) $(RASPIOBJS) \
	-o $@

all: tfmicrotest 


clean :
	$(RM) $(OBJS) 

