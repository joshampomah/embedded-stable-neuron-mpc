# ARM Cortex-M4F cross-compilation toolchain for STM32L4
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake -DEMBEDDED_TARGET=ON ..

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Toolchain binaries
set(CMAKE_C_COMPILER   arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_AR           arm-none-eabi-ar)
set(CMAKE_OBJCOPY      arm-none-eabi-objcopy)
set(CMAKE_OBJDUMP      arm-none-eabi-objdump)
set(CMAKE_SIZE         arm-none-eabi-size)

# Cortex-M4 with hardware FPU (single-precision)
set(CPU_FLAGS "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
# -ffp-contract=fast: Allow fused multiply-add (VFMA) on ARM VFPv4.
# VFMA is 1 cycle vs VMUL+VADD = 2 cycles, and more accurate (single rounding).
# Previously off for PIQP compatibility; custom IPM has iterative refinement.
set(OPT_FLAGS "-O3 -funroll-loops -ffp-contract=fast -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti")

set(CMAKE_C_FLAGS_INIT   "${CPU_FLAGS} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${CPU_FLAGS} ${OPT_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${CPU_FLAGS}")

set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-Wl,--gc-sections -specs=nosys.specs -specs=nano.specs"
)

# Don't try to run test executables during CMake configure
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Search paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
