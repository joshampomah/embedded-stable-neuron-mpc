cmake_minimum_required(VERSION 3.16)
project(stable_neuron_qp_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option: embedded target uses float32, desktop uses double
option(EMBEDDED_TARGET "Build for ARM Cortex-M4F (float32)" OFF)
# Option: use custom Mehrotra IPM instead of PIQP
option(USE_CUSTOM_IPM "Use custom Mehrotra IPM instead of PIQP" ON)

if(EMBEDDED_TARGET)
    add_compile_definitions(EMBEDDED_TARGET)
endif()

# ==============================================================================
# Dependencies via FetchContent
# ==============================================================================
include(FetchContent)

# Eigen 3.4
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# PIQP (header-only dense solver) — only needed when not using custom IPM
if(NOT USE_CUSTOM_IPM)
    FetchContent_Declare(
        piqp
        GIT_REPOSITORY https://github.com/PREDICT-EPFL/piqp.git
        GIT_TAG        v0.4.2
        GIT_SHALLOW    TRUE
    )
    set(PIQP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PIQP_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(piqp)
endif()

# CMSIS headers (only needed for embedded firmware)
if(EMBEDDED_TARGET)
    # CMSIS_5 core headers (core_cm4.h, cmsis_gcc.h)
    FetchContent_Declare(
        cmsis_5
        GIT_REPOSITORY https://github.com/ARM-software/CMSIS_5.git
        GIT_TAG        5.9.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(cmsis_5)
    if(NOT cmsis_5_POPULATED)
        FetchContent_Populate(cmsis_5)
    endif()

    # STM32L4 device headers (stm32l476xx.h)
    FetchContent_Declare(
        cmsis_device_l4
        GIT_REPOSITORY https://github.com/STMicroelectronics/cmsis_device_l4.git
        GIT_TAG        v1.7.3
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(cmsis_device_l4)
    if(NOT cmsis_device_l4_POPULATED)
        FetchContent_Populate(cmsis_device_l4)
    endif()

    # Create interface library for CMSIS include paths
    add_library(cmsis INTERFACE)
    target_include_directories(cmsis INTERFACE
        ${cmsis_5_SOURCE_DIR}/CMSIS/Core/Include
        ${cmsis_device_l4_SOURCE_DIR}/Include
    )
    target_compile_definitions(cmsis INTERFACE
        STM32L476xx
    )
endif()

# ==============================================================================
# stability-reduced QP Solver Library (DCNN SCP)
# ==============================================================================
add_library(stable_neuron_solver
    src/neuron_classifier.cpp
    src/qp_builder.cpp
    src/solver.cpp
    src/icnn_forward.cpp
    src/jacobian.cpp
    src/scp_controller.cpp
)

# ==============================================================================
# Koopman MPC Solver Library (single QP, no SCP)
# ==============================================================================
add_library(koopman_solver
    src/koopman_encoder.cpp
    src/koopman_controller.cpp
)

if(USE_CUSTOM_IPM)
    target_sources(stable_neuron_solver PRIVATE src/custom_ipm.cpp)
    target_compile_definitions(stable_neuron_solver PRIVATE USE_CUSTOM_IPM)
    target_sources(koopman_solver PRIVATE src/custom_ipm.cpp)
    target_compile_definitions(koopman_solver PRIVATE USE_CUSTOM_IPM)
endif()

set(GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated" CACHE PATH "Directory with generated model headers")

target_include_directories(stable_neuron_solver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GENERATED_DIR}
)

target_link_libraries(stable_neuron_solver PUBLIC Eigen3::Eigen)
if(NOT USE_CUSTOM_IPM)
    target_link_libraries(stable_neuron_solver PUBLIC piqp)
endif()

target_include_directories(koopman_solver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GENERATED_DIR}
)
target_link_libraries(koopman_solver PUBLIC Eigen3::Eigen)
if(NOT USE_CUSTOM_IPM)
    target_link_libraries(koopman_solver PUBLIC piqp)
endif()

if(EMBEDDED_TARGET)
    target_compile_options(stable_neuron_solver PRIVATE
        -fno-exceptions -fno-rtti
    )
    target_compile_options(koopman_solver PRIVATE
        -fno-exceptions -fno-rtti
    )
    # FMA is safe for koopman_encoder and koopman_controller (no LDLT)
    set_source_files_properties(src/koopman_encoder.cpp src/koopman_controller.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=fast"
    )

    # Re-enable FMA for our solver files (VMLA.F32 on Cortex-M4F).
    # LDLT factorization diverges with FMA on float32 (different rounding
    # accumulates in the diagonal computation). So solver.cpp is excluded
    # for PIQP, and custom_ipm.cpp is ALWAYS excluded (same LDLT issue).
    set(FMA_SOURCES
        src/neuron_classifier.cpp
        src/qp_builder.cpp
        src/icnn_forward.cpp
        src/jacobian.cpp
        src/scp_controller.cpp
    )
    if(USE_CUSTOM_IPM)
        list(APPEND FMA_SOURCES src/solver.cpp)
        # Note: custom_ipm.cpp excluded — its LDLT diverges with FMA on float32
    endif()
    set_source_files_properties(${FMA_SOURCES}
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=fast"
    )
endif()

# ==============================================================================
# Desktop Test Binary (built when NOT cross-compiling)
# ==============================================================================
if(NOT CMAKE_CROSSCOMPILING)
    add_executable(test_solver test/hil_main.cpp)
    target_link_libraries(test_solver PRIVATE stable_neuron_solver)

    # HIL solver binary (stdin/stdout pipe for closed-loop testing)
    add_executable(hil_solver test/hil_main.cpp)
    target_link_libraries(hil_solver PRIVATE stable_neuron_solver)

    # Koopman HIL solver binary
    add_executable(hil_koopman test/hil_koopman.cpp)
    target_link_libraries(hil_koopman PRIVATE koopman_solver)
endif()

# ==============================================================================
# Firmware Binary (NUCLEO-L476RG) — only when cross-compiling
# ==============================================================================
if(EMBEDDED_TARGET AND CMAKE_CROSSCOMPILING)
    set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/firmware/stm32l476rg.ld)

    set(FIRMWARE_MAIN firmware/main.cpp CACHE STRING "Firmware main source file")

    add_executable(firmware
        firmware/startup.cpp
        firmware/board.cpp
        ${FIRMWARE_MAIN}
    )

    target_include_directories(firmware PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/firmware
    )

    target_link_libraries(firmware PRIVATE
        stable_neuron_solver
        cmsis
    )

    target_compile_options(firmware PRIVATE
        -fno-exceptions -fno-rtti -Wall -Wextra
    )

    target_link_options(firmware PRIVATE
        -T${LINKER_SCRIPT}
        -Wl,--gc-sections
        -Wl,-Map=firmware.map
    )

    add_custom_command(TARGET firmware POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:firmware> firmware.bin
        COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:firmware> firmware.hex
        COMMAND ${CMAKE_SIZE} --format=berkeley $<TARGET_FILE:firmware>
        COMMENT "Generating firmware.bin, firmware.hex, and size report"
    )

    # --- Koopman firmware ---
    add_executable(firmware_koopman
        firmware/startup.cpp
        firmware/board.cpp
        firmware/main_koopman.cpp
    )

    target_include_directories(firmware_koopman PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/firmware
    )

    target_link_libraries(firmware_koopman PRIVATE
        koopman_solver
        cmsis
    )

    target_compile_options(firmware_koopman PRIVATE
        -fno-exceptions -fno-rtti -Wall -Wextra
    )

    target_link_options(firmware_koopman PRIVATE
        -T${LINKER_SCRIPT}
        -Wl,--gc-sections
        -Wl,-Map=firmware_koopman.map
    )

    add_custom_command(TARGET firmware_koopman POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:firmware_koopman> firmware_koopman.bin
        COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:firmware_koopman> firmware_koopman.hex
        COMMAND ${CMAKE_SIZE} --format=berkeley $<TARGET_FILE:firmware_koopman>
        COMMENT "Generating firmware_koopman.bin/.hex and size report"
    )
endif()
