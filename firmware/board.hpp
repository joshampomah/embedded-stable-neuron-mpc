#pragma once
// Board support for NUCLEO-L476RG (STM32L476RG)
// Bare-metal: clock, UART, DWT cycle counter, LED

#include <cstddef>
#include <cstdint>

namespace board {

// System clock frequency after PLL init
constexpr uint32_t SYSCLK_HZ = 80000000;

// Initialize system clock: MSI 4MHz -> PLL -> 80MHz SYSCLK
void system_init();

// Initialize USART2 at 115200 baud on PA2/PA3 (ST-Link VCP)
void uart_init();

// Blocking single-character transmit
void uart_putc(char c);

// Blocking single-character receive
uint8_t uart_getc();

// Raw binary read (no translation, blocks until all bytes received)
void uart_read_bytes(void* buf, size_t len);

// Raw binary write (no \r\n translation)
void uart_write_bytes(const void* buf, size_t len);

// Print null-terminated string
void uart_puts(const char* s);

// Minimal printf (supports %d, %u, %x, %s, %f, %%)
void uart_printf(const char* fmt, ...);

// Initialize DWT cycle counter for microsecond timing
void dwt_init();

// Read DWT cycle counter as microseconds
uint32_t get_microseconds();

// Initialize LD2 (PA5)
void led_init();

// Toggle LD2
void led_toggle();

}  // namespace board

// Heap diagnostics and control (C linkage, accessible from anywhere)
extern "C" uint32_t heap_used_bytes();
extern "C" uint32_t heap_high_water_bytes();
extern "C" uint32_t heap_capacity_bytes();

// Bump allocator reset: reclaims all heap memory above the saved baseline.
// Call heap_save_baseline() once after any permanent allocations (e.g., stdio).
// Call heap_reset_to_baseline() before re-creating large allocators (PIQP).
extern "C" void heap_save_baseline();
extern "C" void heap_reset_to_baseline();
extern "C" uint32_t heap_alloc_failures();
extern "C" uint32_t heap_alloc_total();
extern "C" uint32_t heap_largest_failed();
extern "C" uint32_t heap_free_reclaimed_count();
extern "C" void heap_reset_overflow();
