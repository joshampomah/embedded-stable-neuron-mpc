// Startup code for STM32L476RG — C++, no assembly required.
// Provides vector table, Reset_Handler, and default fault handlers.

#include <cstdint>
#include <cstring>

// Linker-provided symbols
extern uint32_t _estack;
extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

// C++ static constructors
extern void (*__preinit_array_start[])(void);
extern void (*__preinit_array_end[])(void);
extern void (*__init_array_start[])(void);
extern void (*__init_array_end[])(void);

// Forward declarations
extern "C" void Reset_Handler();
extern "C" void Default_Handler();
extern "C" void HardFault_Handler();
extern "C" void NMI_Handler()          __attribute__((weak, alias("Default_Handler")));
extern "C" void MemManage_Handler()    __attribute__((weak, alias("Default_Handler")));
extern "C" void BusFault_Handler()     __attribute__((weak, alias("Default_Handler")));
extern "C" void UsageFault_Handler()   __attribute__((weak, alias("Default_Handler")));
extern "C" void SVC_Handler()          __attribute__((weak, alias("Default_Handler")));
extern "C" void DebugMon_Handler()     __attribute__((weak, alias("Default_Handler")));
extern "C" void PendSV_Handler()       __attribute__((weak, alias("Default_Handler")));
extern "C" void SysTick_Handler()      __attribute__((weak, alias("Default_Handler")));

// Main entry point
extern int main();

// Vector table — placed in .isr_vector section
using ISR_Handler = void (*)();

__attribute__((section(".isr_vector"), used))
const ISR_Handler g_pfnVectors[] = {
    reinterpret_cast<ISR_Handler>(&_estack),  // Initial SP
    Reset_Handler,                             // Reset
    NMI_Handler,                               // NMI
    HardFault_Handler,                         // Hard Fault
    MemManage_Handler,                         // MemManage
    BusFault_Handler,                          // Bus Fault
    UsageFault_Handler,                        // Usage Fault
    nullptr, nullptr, nullptr, nullptr,        // Reserved
    SVC_Handler,                               // SVCall
    DebugMon_Handler,                          // Debug Monitor
    nullptr,                                   // Reserved
    PendSV_Handler,                            // PendSV
    SysTick_Handler,                           // SysTick
    // IRQ 0-81 (STM32L476) — all default
};

extern "C" void Reset_Handler() {
    // 0. Enable FPU (must be first — static constructors may use float)
    // Set CP10 and CP11 to Full Access in CPACR
    *reinterpret_cast<volatile uint32_t*>(0xE000ED88) |= (0xF << 20);
    __asm volatile("dsb");
    __asm volatile("isb");

    // 1. Copy .data from flash to SRAM
    uint32_t* src = &_sidata;
    uint32_t* dst = &_sdata;
    while (dst < &_edata) {
        *dst++ = *src++;
    }

    // 2. Zero .bss
    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }

    // 3. Call C++ static constructors
    for (auto fn = __preinit_array_start; fn < __preinit_array_end; fn++) {
        (*fn)();
    }
    for (auto fn = __init_array_start; fn < __init_array_end; fn++) {
        (*fn)();
    }

    // 4. Call main
    main();

    // 5. If main returns, hang
    while (true) {
        __asm volatile("bkpt #0");
    }
}

extern "C" void Default_Handler() {
    while (true) {
        __asm volatile("bkpt #1");
    }
}

extern "C" void HardFault_Handler() {
    while (true) {
        __asm volatile("bkpt #2");
    }
}
