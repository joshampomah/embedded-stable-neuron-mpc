// Firmware main for NUCLEO-L476RG — Koopman MPC solver
// Self-contained validation and timing benchmark.
//
// 1. Init clock (80MHz), UART (115200), DWT, LED
// 2. Print banner and memory usage
// 3. Run solver 100x with reference test case, report timing stats
// 4. Run once with profiling, report breakdown
// 5. Compare against Python reference, report pass/fail

#include "board.hpp"
#include "koopman_reference_test_case.hpp"
#include "stable_neuron_solver/koopman_controller.hpp"
#include "koopman_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace stable_neuron;
using namespace board;
using namespace koopman_ref;

// Koopman controller — much smaller than SCP (~2 KB vs ~82 KB)
static KoopmanController g_controller;

static void configure_controller() {
    KoopmanParams params;
    params.Q = REF_Q;
    params.R = REF_R;
    params.beta_0 = REF_BETA_0;
    params.u_min = REF_U_MIN;
    params.u_max = REF_U_MAX;
    params.delta_u_max = REF_DELTA_U_MAX;
    g_controller.configure(params);
}

int main() {
    // 1. Hardware init
    system_init();
    uart_init();
    dwt_init();
    led_init();

    // Wait for serial connection
    for (volatile uint32_t i = 0; i < 30000000; i++) {}

    // 2. Banner
    uart_puts("\n\n");
    uart_puts("========================================\n");
    uart_puts("  Koopman MPC - NUCLEO-L476RG\n");
    uart_puts("  STM32L476RG  80MHz  128KB SRAM\n");
    uart_puts("========================================\n\n");

    uart_printf("sizeof(KoopmanController) = %u bytes\n",
                static_cast<unsigned>(sizeof(KoopmanController)));
    uart_printf("sizeof(KScalar) = %u bytes (%s)\n",
                static_cast<unsigned>(sizeof(KScalar)),
                sizeof(KScalar) == 4 ? "float" : "double");
    uart_printf("KN=%d, KD_LIFT=%d, KN_FEATURES=%d\n", KN, KD_LIFT, KN_FEATURES);
    uart_printf("QP: %d vars, %d constraints\n", KN_VARS, KN_INEQ);
    uart_printf("Heap before configure: %lu / %lu bytes\n\n",
                static_cast<unsigned long>(heap_used_bytes()),
                static_cast<unsigned long>(heap_capacity_bytes()));

    // Section sizes
    extern uint32_t _sdata, _edata, _sbss, _ebss, _estack;
    uint32_t data_size = reinterpret_cast<uint32_t>(&_edata) - reinterpret_cast<uint32_t>(&_sdata);
    uint32_t bss_size = reinterpret_cast<uint32_t>(&_ebss) - reinterpret_cast<uint32_t>(&_sbss);
    uart_printf(".data = %lu bytes, .bss = %lu bytes\n",
                static_cast<unsigned long>(data_size),
                static_cast<unsigned long>(bss_size));
    uart_printf("Stack top = 0x%08lx\n\n",
                static_cast<unsigned long>(reinterpret_cast<uint32_t>(&_estack)));

    // 3. Timing benchmark: solve<false>() x N_RUNS
    uart_puts("--- Timing Benchmark ---\n");

    constexpr int N_RUNS = 100;
    uint32_t times_us[N_RUNS];
    KoopmanResult last_result;

    // First run with cold start
    configure_controller();
    {
        uint32_t t0 = get_microseconds();
        last_result = g_controller.solve<false>(REF_Z_K, REF_U_PREV, g_koopman_weights);
        uint32_t t1 = get_microseconds();
        uart_printf("  Cold start: %lu us, feasible=%d\n",
                    static_cast<unsigned long>(t1 - t0),
                    last_result.is_feasible ? 1 : 0);
    }

    // Warm-start runs (realistic: solver reuses previous solution)
    for (int i = 0; i < N_RUNS; i++) {
        uint32_t t0 = get_microseconds();
        last_result = g_controller.solve<false>(REF_Z_K, REF_U_PREV, g_koopman_weights);
        uint32_t t1 = get_microseconds();
        times_us[i] = t1 - t0;
    }

    // Sort for percentiles
    // Simple insertion sort (N_RUNS is small)
    for (int i = 1; i < N_RUNS; i++) {
        uint32_t key = times_us[i];
        int j = i - 1;
        while (j >= 0 && times_us[j] > key) {
            times_us[j + 1] = times_us[j];
            j--;
        }
        times_us[j + 1] = key;
    }

    uint32_t sum_us = 0;
    for (int i = 0; i < N_RUNS; i++) sum_us += times_us[i];
    uint32_t mean_us = sum_us / N_RUNS;

    uart_printf("\n  Runs:  %d (warm-start)\n", N_RUNS);
    uart_printf("  Mean:  %lu us\n", static_cast<unsigned long>(mean_us));
    uart_printf("  P50:   %lu us\n", static_cast<unsigned long>(times_us[N_RUNS / 2]));
    uart_printf("  P95:   %lu us\n", static_cast<unsigned long>(times_us[(int)(N_RUNS * 0.95)]));
    uart_printf("  P99:   %lu us\n", static_cast<unsigned long>(times_us[(int)(N_RUNS * 0.99)]));
    uart_printf("  Min:   %lu us\n", static_cast<unsigned long>(times_us[0]));
    uart_printf("  Max:   %lu us\n", static_cast<unsigned long>(times_us[N_RUNS - 1]));
    uart_printf("  Budget: %lu us / 20000 us = %lu.%lu%%\n",
                static_cast<unsigned long>(mean_us),
                static_cast<unsigned long>(mean_us * 100 / 20000),
                static_cast<unsigned long>((mean_us * 1000 / 20000) % 10));
    uart_printf("  Heap high-water: %lu / %lu bytes\n\n",
                static_cast<unsigned long>(heap_high_water_bytes()),
                static_cast<unsigned long>(heap_capacity_bytes()));

    // Histogram
    uart_puts("  Histogram:\n");
    int bins[] = {0, 0, 0, 0, 0};  // <1ms, 1-2ms, 2-5ms, 5-10ms, 10ms+
    // Re-run to get unsorted timings for histogram
    g_controller.reset();
    configure_controller();
    for (int i = 0; i < N_RUNS; i++) {
        uint32_t t0 = get_microseconds();
        g_controller.solve<false>(REF_Z_K, REF_U_PREV, g_koopman_weights);
        uint32_t t1 = get_microseconds();
        uint32_t dt = t1 - t0;
        if (dt < 1000) bins[0]++;
        else if (dt < 2000) bins[1]++;
        else if (dt < 5000) bins[2]++;
        else if (dt < 10000) bins[3]++;
        else bins[4]++;
    }
    uart_printf("    <1ms:   %d\n", bins[0]);
    uart_printf("    1-2ms:  %d\n", bins[1]);
    uart_printf("    2-5ms:  %d\n", bins[2]);
    uart_printf("    5-10ms: %d\n", bins[3]);
    uart_printf("    >10ms:  %d\n\n", bins[4]);

    // 4. Profiled solve
    uart_puts("--- Profiled Solve ---\n");
    configure_controller();
    // One warm-up
    g_controller.solve<false>(REF_Z_K, REF_U_PREV, g_koopman_weights);
    KoopmanResult prof = g_controller.solve<true>(REF_Z_K, REF_U_PREV, g_koopman_weights);

    uart_printf("  Total:     %lu us\n", static_cast<unsigned long>(prof.total_time_us));
    uart_printf("  Encode:    %lu us\n", static_cast<unsigned long>(prof.encode_time_us));
    uart_printf("  Predict:   %lu us\n", static_cast<unsigned long>(prof.predict_time_us));
    uart_printf("  QP build:  %lu us\n", static_cast<unsigned long>(prof.qp_build_time_us));
    uart_printf("  QP solve:  %lu us\n", static_cast<unsigned long>(prof.qp_solve_time_us));
    uart_printf("  Feasible:  %d\n", prof.is_feasible ? 1 : 0);
    uart_printf("  QP status: %d, iters: %d\n", prof.qp_status, prof.qp_iters);

    // 5. Validation
    uart_puts("\n--- Validation ---\n");

    KScalar max_err = 0;
    bool has_nan = false;
    for (int i = 0; i < KN; i++) {
        KScalar got = prof.u_optimal[i];
        KScalar exp = REF_U_OPTIMAL[i];
        if (got != got) {
            has_nan = true;
            uart_printf("  u[%d]: NaN (exp %d x1e6)\n", i,
                        static_cast<int>(exp * 1e6f));
            continue;
        }
        KScalar err = got - exp;
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        uart_printf("  u[%d]: got=%d exp=%d err=%d (x1e6)\n", i,
                    static_cast<int>(got * 1e6f),
                    static_cast<int>(exp * 1e6f),
                    static_cast<int>(err * 1e6f));
    }

    const KScalar tol = 1e-3f;
    bool pass = !has_nan && (max_err < tol);
    uart_printf("\n  Max error: %d (tol: %d) (x1e6)\n",
                static_cast<int>(max_err * 1e6f),
                static_cast<int>(tol * 1e6f));
    uart_printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");

    // Summary
    uart_puts("========================================\n");
    uart_printf("  %s - %lu us/step mean\n",
                pass ? "ALL PASS" : "FAIL",
                static_cast<unsigned long>(mean_us));
    uart_printf("  50Hz: %s (%lu us < 20000 us)\n",
                mean_us < 20000 ? "FEASIBLE" : "NOT FEASIBLE",
                static_cast<unsigned long>(mean_us));
    uart_printf("  P95:  %lu us (tail ratio: %lu.%lux)\n",
                static_cast<unsigned long>(times_us[(int)(N_RUNS * 0.95)]),
                static_cast<unsigned long>(times_us[(int)(N_RUNS * 0.95)] * 10 / mean_us / 10),
                static_cast<unsigned long>((times_us[(int)(N_RUNS * 0.95)] * 10 / mean_us) % 10));
    uart_puts("========================================\n");

    // LED toggle
    while (true) {
        led_toggle();
        for (volatile uint32_t i = 0; i < 4000000; i++) {}
    }

    return 0;
}
