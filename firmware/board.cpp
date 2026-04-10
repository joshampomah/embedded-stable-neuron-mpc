// Board support for NUCLEO-L476RG — bare-metal register-level init.
// Clock: MSI 4MHz → PLL → 80MHz | UART: USART2 115200 | DWT: cycle counter

#include "board.hpp"
#include <cstdarg>
#include <cstring>
#include <new>

// CMSIS device header provides all register definitions
#include "stm32l476xx.h"

namespace board {

// ============================================================
// System Clock: MSI 4MHz → PLL → 80 MHz
// ============================================================

void system_init() {
    // 0. Disable FPU Flush-to-Zero and Default-NaN modes.
    // ARM Cortex-M4 FPU enables FTZ by default (FPSCR bit 24), which flushes
    // denormalized floats to zero. This causes PIQP's LDLT factorization to
    // diverge for certain QP formulations (works fine on x86 where FTZ is off).
    // Clearing FZ (bit 24) and DN (bit 25) makes ARM FPU behave like x86.
    uint32_t fpscr = __get_FPSCR();
    fpscr &= ~((1u << 24) | (1u << 25));  // Clear FZ and DN bits
    __set_FPSCR(fpscr);

    // 1. Set flash latency to 4 wait states (required for 80MHz at VOS Range 1)
    FLASH->ACR = FLASH_ACR_LATENCY_4WS | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;
    while ((FLASH->ACR & FLASH_ACR_LATENCY) != FLASH_ACR_LATENCY_4WS) {}

    // 2. Ensure VOS Range 1 (high performance, default after reset)
    PWR->CR1 = (PWR->CR1 & ~PWR_CR1_VOS) | PWR_CR1_VOS_0;

    // 3. Enable MSI at 4 MHz (Range 6, default after reset)
    RCC->CR |= RCC_CR_MSION;
    while (!(RCC->CR & RCC_CR_MSIRDY)) {}

    // 4. Configure PLL: MSI(4MHz) / 1 × 40 / 2 = 80 MHz
    RCC->CR &= ~RCC_CR_PLLON;
    while (RCC->CR & RCC_CR_PLLRDY) {}

    RCC->PLLCFGR = RCC_PLLCFGR_PLLSRC_MSI  // PLL source = MSI
        | (0 << RCC_PLLCFGR_PLLM_Pos)       // PLLM = /1
        | (40 << RCC_PLLCFGR_PLLN_Pos)      // PLLN = ×40
        | RCC_PLLCFGR_PLLREN                 // Enable PLLR output
        | (0 << RCC_PLLCFGR_PLLR_Pos);      // PLLR = /2 (00 = /2)

    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY)) {}

    // 5. Switch SYSCLK to PLL
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) {}

    // 6. AHB, APB1, APB2 prescalers = /1 (default)
}

// ============================================================
// USART2: PA2 (TX, AF7), PA3 (RX, AF7) — ST-Link VCP
// ============================================================

void uart_init() {
    // Enable clocks: GPIOA + USART2
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN;
    RCC->APB1ENR1 |= RCC_APB1ENR1_USART2EN;

    // PA2, PA3: alternate function mode
    GPIOA->MODER = (GPIOA->MODER & ~(GPIO_MODER_MODE2 | GPIO_MODER_MODE3))
        | (0b10 << GPIO_MODER_MODE2_Pos)    // AF mode
        | (0b10 << GPIO_MODER_MODE3_Pos);

    // AF7 for USART2
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~(0xF << (2 * 4)) & ~(0xF << (3 * 4)))
        | (7 << (2 * 4))   // PA2 = AF7
        | (7 << (3 * 4));  // PA3 = AF7

    // High speed output for PA2
    GPIOA->OSPEEDR |= GPIO_OSPEEDR_OSPEED2;

    // Configure USART2: 115200 baud from 80MHz PCLK1
    USART2->CR1 = 0;
    USART2->BRR = SYSCLK_HZ / 115200;  // APB1 = SYSCLK/1 = 80MHz
    USART2->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

void uart_putc(char c) {
    while (!(USART2->ISR & USART_ISR_TXE)) {}
    USART2->TDR = static_cast<uint8_t>(c);
}

uint8_t uart_getc() {
    while (!(USART2->ISR & USART_ISR_RXNE)) {}
    return static_cast<uint8_t>(USART2->RDR);
}

void uart_read_bytes(void* buf, size_t len) {
    auto* p = static_cast<uint8_t*>(buf);
    for (size_t i = 0; i < len; i++) {
        p[i] = uart_getc();
    }
}

void uart_write_bytes(const void* buf, size_t len) {
    auto* p = static_cast<const uint8_t*>(buf);
    for (size_t i = 0; i < len; i++) {
        while (!(USART2->ISR & USART_ISR_TXE)) {}
        USART2->TDR = p[i];
    }
}

void uart_puts(const char* s) {
    while (*s) {
        if (*s == '\n') uart_putc('\r');
        uart_putc(*s++);
    }
}

// Minimal printf — %d, %u, %x, %lu, %s, %f, %%
void uart_printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    char buf[24];
    while (*fmt) {
        if (*fmt != '%') {
            if (*fmt == '\n') uart_putc('\r');
            uart_putc(*fmt++);
            continue;
        }
        fmt++;

        // Width specifier (optional)
        int width = 0;
        bool zero_pad = false;
        if (*fmt == '0') { zero_pad = true; fmt++; }
        while (*fmt >= '0' && *fmt <= '9') {
            width = width * 10 + (*fmt++ - '0');
        }

        // Length modifier
        bool is_long = false;
        if (*fmt == 'l') { is_long = true; fmt++; }

        switch (*fmt++) {
        case 'd': {
            long val = is_long ? va_arg(args, long) : va_arg(args, int);
            if (val < 0) { uart_putc('-'); val = -val; }
            int i = 0;
            if (val == 0) buf[i++] = '0';
            else while (val > 0) { buf[i++] = '0' + (val % 10); val /= 10; }
            for (int p = 0; p < width - i; p++) uart_putc(zero_pad ? '0' : ' ');
            while (i > 0) uart_putc(buf[--i]);
            break;
        }
        case 'u': {
            unsigned long val = is_long ? va_arg(args, unsigned long) : va_arg(args, unsigned);
            int i = 0;
            if (val == 0) buf[i++] = '0';
            else while (val > 0) { buf[i++] = '0' + (val % 10); val /= 10; }
            for (int p = 0; p < width - i; p++) uart_putc(zero_pad ? '0' : ' ');
            while (i > 0) uart_putc(buf[--i]);
            break;
        }
        case 'x': {
            unsigned long val = is_long ? va_arg(args, unsigned long) : va_arg(args, unsigned);
            const char hex[] = "0123456789abcdef";
            int i = 0;
            if (val == 0) buf[i++] = '0';
            else while (val > 0) { buf[i++] = hex[val & 0xF]; val >>= 4; }
            for (int p = 0; p < width - i; p++) uart_putc(zero_pad ? '0' : ' ');
            while (i > 0) uart_putc(buf[--i]);
            break;
        }
        case 's': {
            const char* s = va_arg(args, const char*);
            if (!s) s = "(null)";
            uart_puts(s);
            break;
        }
        case 'f': {
            // Simple float formatting: up to 6 decimal places
            double val = va_arg(args, double);
            if (val < 0) { uart_putc('-'); val = -val; }
            auto ipart = static_cast<unsigned long>(val);
            double fpart = val - ipart;
            // Integer part
            int i = 0;
            if (ipart == 0) buf[i++] = '0';
            else while (ipart > 0) { buf[i++] = '0' + (ipart % 10); ipart /= 10; }
            while (i > 0) uart_putc(buf[--i]);
            uart_putc('.');
            // 6 decimal digits
            for (int d = 0; d < 6; d++) {
                fpart *= 10;
                int digit = static_cast<int>(fpart);
                uart_putc('0' + digit);
                fpart -= digit;
            }
            break;
        }
        case '%':
            uart_putc('%');
            break;
        default:
            uart_putc('?');
            break;
        }
    }
    va_end(args);
}

// ============================================================
// DWT Cycle Counter → microseconds
// ============================================================

void dwt_init() {
    // Enable DWT and ITM via CoreDebug DEMCR
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

    // Unlock DWT (STM32L4 locks DWT by default)
    // LAR is at DWT base + 0xFB0, not defined in all CMSIS versions
    volatile uint32_t* DWT_LAR = reinterpret_cast<volatile uint32_t*>(0xE0001FB0);
    *DWT_LAR = 0xC5ACCE55;

    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t get_microseconds() {
    return DWT->CYCCNT / (SYSCLK_HZ / 1000000);
}

// ============================================================
// LED: LD2 on PA5
// ============================================================

void led_init() {
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN;
    // PA5: general purpose output, push-pull
    GPIOA->MODER = (GPIOA->MODER & ~GPIO_MODER_MODE5) | (0b01 << GPIO_MODER_MODE5_Pos);
}

void led_toggle() {
    GPIOA->ODR ^= GPIO_ODR_OD5;
}

}  // namespace board

// ============================================================
// C linkage: get_microseconds for solver timing
// ============================================================
extern "C" uint32_t get_microseconds() {
    return board::get_microseconds();
}

// ============================================================
// Newlib stubs (nano.specs)
// ============================================================
extern "C" {

int _write(int fd, const char* buf, int len) {
    (void)fd;
    for (int i = 0; i < len; i++) {
        if (buf[i] == '\n') board::uart_putc('\r');
        board::uart_putc(buf[i]);
    }
    return len;
}

// ===========================================================================
// Two-region stack-pop allocator: SRAM2 (32KB) + SRAM1 overflow (4KB).
//
// Primary region: SRAM2 (32KB) — holds all PIQP permanent allocations.
// Overflow region: SRAM1 BSS (4KB) — handles transient peaks during setup
// when permanent allocs + a temporary momentarily exceed 32KB.
//
// Stack-pop reclamation: if free() is called for the most recent allocation,
// the bump pointer is rolled back (in whichever region it was allocated from).
// This handles Eigen's temporary matrices perfectly — Eigen allocates a temp,
// evaluates the expression, then frees it (LIFO order).
//
// Bulk reclamation: heap_reset_to_baseline() reclaims all memory above the
// saved baseline (used when recreating PIQP solver with new dimensions).
// ===========================================================================

// Primary heap: SRAM2 (32KB)
static constexpr uint32_t HEAP_SIZE = 32 * 1024;
static char __attribute__((section(".sram2"))) heap_buf[HEAP_SIZE];
static uint32_t heap_offset = 0;

// Overflow heap: SRAM1 BSS (8KB) — handles transient peaks
static constexpr uint32_t OVERFLOW_SIZE = 8 * 1024;
static char overflow_buf[OVERFLOW_SIZE];  // BSS in SRAM1
static uint32_t overflow_offset = 0;

static uint32_t heap_baseline = 0;
static uint32_t heap_high = 0;
static uint32_t heap_fail_count = 0;
static uint32_t heap_alloc_count = 0;
static uint32_t heap_largest_fail = 0;
static uint32_t heap_free_reclaimed = 0;

// Track most recent allocation for stack-pop reclamation
static void* last_alloc_ptr = nullptr;
static uint32_t last_alloc_size = 0;
static bool last_alloc_overflow = false;  // which region the last alloc came from

static void* bump_alloc(size_t size) {
    // 8-byte alignment (required for Eigen, ARM NEON, double)
    size = (size + 7) & ~static_cast<size_t>(7);

    // Try primary (SRAM2) first
    if (heap_offset + size <= HEAP_SIZE) {
        void* p = &heap_buf[heap_offset];
        __builtin_memset(p, 0, size);
        heap_offset += static_cast<uint32_t>(size);
        heap_alloc_count++;
        if (heap_offset > heap_high) heap_high = heap_offset;
        last_alloc_ptr = p;
        last_alloc_size = static_cast<uint32_t>(size);
        last_alloc_overflow = false;
        return p;
    }

    // Fall through to overflow (SRAM1)
    if (overflow_offset + size <= OVERFLOW_SIZE) {
        void* p = &overflow_buf[overflow_offset];
        __builtin_memset(p, 0, size);
        overflow_offset += static_cast<uint32_t>(size);
        heap_alloc_count++;
        last_alloc_ptr = p;
        last_alloc_size = static_cast<uint32_t>(size);
        last_alloc_overflow = true;
        return p;
    }

    // Both regions full
    heap_fail_count++;
    if (static_cast<uint32_t>(size) > heap_largest_fail)
        heap_largest_fail = static_cast<uint32_t>(size);
    return nullptr;
}

// Stack-pop free: reclaim if freeing the most recent allocation (LIFO pattern)
static void stack_free(void* ptr) {
    if (ptr && ptr == last_alloc_ptr && last_alloc_size > 0) {
        if (last_alloc_overflow) {
            overflow_offset -= last_alloc_size;
        } else {
            heap_offset -= last_alloc_size;
        }
        last_alloc_ptr = nullptr;
        last_alloc_size = 0;
        heap_free_reclaimed++;
    }
    // Non-LIFO frees are silently ignored (permanent allocations)
}

void heap_save_baseline() { heap_baseline = heap_offset; }
void heap_reset_to_baseline() {
    heap_offset = heap_baseline;
    overflow_offset = 0;  // reclaim all overflow
    last_alloc_ptr = nullptr;
    last_alloc_size = 0;
}

uint32_t heap_used_bytes() { return heap_offset + overflow_offset; }
uint32_t heap_high_water_bytes() { return heap_high; }  // primary only
uint32_t heap_capacity_bytes() { return HEAP_SIZE + OVERFLOW_SIZE; }
uint32_t heap_alloc_failures() { return heap_fail_count; }
uint32_t heap_alloc_total() { return heap_alloc_count; }
uint32_t heap_largest_failed() { return heap_largest_fail; }
uint32_t heap_free_reclaimed_count() { return heap_free_reclaimed; }
void heap_reset_overflow() {
    overflow_offset = 0;
    // Clear last_alloc tracking if it pointed to overflow
    if (last_alloc_overflow) {
        last_alloc_ptr = nullptr;
        last_alloc_size = 0;
    }
}

// Override newlib's reentrant allocators (all malloc/free calls route here)
struct _reent;
void* _malloc_r(struct _reent*, size_t size) { return bump_alloc(size); }
void  _free_r(struct _reent*, void* ptr) { stack_free(ptr); }
void* _calloc_r(struct _reent*, size_t n, size_t size) {
    void* p = bump_alloc(n * size);
    if (p) __builtin_memset(p, 0, n * size);
    return p;
}
void* _realloc_r(struct _reent*, void*, size_t size) {
    return bump_alloc(size);  // Allocate new; old data not copied (unused in this firmware)
}

// sbrk stub — should never be called since we override malloc
void* _sbrk(int) { return reinterpret_cast<void*>(-1); }

int _close(int fd) { (void)fd; return -1; }
int _fstat(int fd, void* st) { (void)fd; (void)st; return -1; }
int _isatty(int fd) { (void)fd; return 1; }
int _lseek(int fd, int ptr, int dir) { (void)fd; (void)ptr; (void)dir; return -1; }
int _read(int fd, char* buf, int len) { (void)fd; (void)buf; (void)len; return 0; }
void _exit(int status) { (void)status; while(1) {} }
void _kill(int pid, int sig) { (void)pid; (void)sig; }
int _getpid() { return 1; }

}  // extern "C"

// Override C++ operator new/delete to use stack-pop allocator.
// Eigen's dynamic matrices use operator new internally.
void* operator new(size_t size) { return bump_alloc(size); }
void* operator new(size_t size, const std::nothrow_t&) noexcept { return bump_alloc(size); }
void* operator new[](size_t size) { return bump_alloc(size); }
void  operator delete(void* p) noexcept { stack_free(p); }
void  operator delete(void* p, size_t) noexcept { stack_free(p); }
void  operator delete[](void* p) noexcept { stack_free(p); }
void  operator delete[](void* p, size_t) noexcept { stack_free(p); }
