// Firmware main for NUCLEO-L476RG
// Self-contained DCNN-MPC solver validation and timing benchmark.
//
// 1. Init clock (80MHz), UART (115200), DWT, LED
// 2. Print banner and memory usage
// 3. Run solver 100x with hardcoded test case, report timing
// 4. Run once with profiling, report per-iteration breakdown
// 5. Compare against desktop reference, report pass/fail

#include "board.hpp"
#include "reference_test_case.hpp"
#include "stable_neuron_solver/scp_controller.hpp"
#include "network_weights.hpp"

#include <cmath>
#include <cstring>

using namespace stable_neuron;
using namespace board;
using namespace firmware_ref;

// Global solver — too large for stack (~82 KB)
static SCPController g_controller;

// Configure SCP controller from reference test case
static void configure_from_ref(SCPController& ctrl) {
    SCPParams params;
    params.Q = REF_Q;
    params.R = REF_R;
    params.R_delta = REF_R_DELTA;
    params.tube_weight = REF_TUBE_WEIGHT;
    params.beta_0 = REF_BETA_0;
    params.u_min = REF_U_MIN;
    params.u_max = REF_U_MAX;
    params.delta_u_max = REF_DELTA_U_MAX;
    params.delta_J_min = REF_DELTA_J_MIN;
    params.delta_u_tol = Scalar(0.1) * REF_DELTA_U_MAX;  // Skip re-linearization if QP barely moved
    // With P_REG=0.01 conditioning fix, float32 PIQP converges reliably.
    // SCP typically converges in 2-3 iterations.
    params.maxiters = 10;
    ctrl.configure(params, REF_W_BOUNDS);
}

int main() {
    // 1. Hardware init
    system_init();
    uart_init();
    dwt_init();
    led_init();

    // Wait for serial connection (~3s at 80MHz)
    for (volatile uint32_t i = 0; i < 30000000; i++) {}

    // 2. Banner
    uart_puts("\n\n");
    uart_puts("========================================\n");
    uart_puts("  DCNN-MPC Solver - NUCLEO-L476RG\n");
    uart_puts("  STM32L476RG  80MHz  128KB SRAM\n");
    uart_puts("========================================\n\n");

    // Memory sizes
    uart_printf("sizeof(SCPController)  = %u bytes\n",
                static_cast<unsigned>(sizeof(SCPController)));
    uart_printf("sizeof(StableNeuronSolver)= %u bytes\n",
                static_cast<unsigned>(sizeof(StableNeuronSolver)));
    uart_printf("sizeof(Scalar)         = %u bytes (%s)\n",
                static_cast<unsigned>(sizeof(Scalar)),
                sizeof(Scalar) == 4 ? "float" : "double");
    uart_printf("N=%d, N_STATE=%d, N_HIDDEN=%d\n", N, N_STATE, N_HIDDEN);
    uart_printf("MAX_VARS=%d, MAX_EQ=%d, MAX_INEQ=%d\n", MAX_VARS, MAX_EQ, MAX_INEQ);
    uart_printf("Heap before configure: %lu / %lu bytes\n\n",
                static_cast<unsigned long>(heap_used_bytes()),
                static_cast<unsigned long>(heap_capacity_bytes()));

    // Section sizes from linker
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

    configure_from_ref(g_controller);

    constexpr int N_RUNS = 10;
    uint32_t times_us[N_RUNS];
    SCPResult last_result;

    for (int i = 0; i < N_RUNS; i++) {
        // Reset warm-start each iteration for consistent timing
        configure_from_ref(g_controller);

        uart_printf("  run %d...", i);
        uint32_t t0 = get_microseconds();
        last_result = g_controller.solve<false>(REF_Z_K, REF_U_PREV, g_model_weights);
        uint32_t t1 = get_microseconds();
        times_us[i] = t1 - t0;
        uart_printf(" %lu us, iters=%d\n", static_cast<unsigned long>(times_us[i]), last_result.n_iterations);
    }

    // Compute stats
    uint32_t min_us = times_us[0], max_us = times_us[0];
    uint32_t sum_us = 0;
    for (int i = 0; i < N_RUNS; i++) {
        if (times_us[i] < min_us) min_us = times_us[i];
        if (times_us[i] > max_us) max_us = times_us[i];
        sum_us += times_us[i];
    }
    uint32_t mean_us = sum_us / N_RUNS;

    uart_puts("\n");
    uart_printf("  Runs:  %d\n", N_RUNS);
    uart_printf("  Mean:  %lu us\n", static_cast<unsigned long>(mean_us));
    uart_printf("  Min:   %lu us\n", static_cast<unsigned long>(min_us));
    uart_printf("  Max:   %lu us\n", static_cast<unsigned long>(max_us));
    uart_printf("  Budget: %lu us / 20000 us = %lu.%lu%%\n",
                static_cast<unsigned long>(mean_us),
                static_cast<unsigned long>(mean_us * 100 / 20000),
                static_cast<unsigned long>((mean_us * 1000 / 20000) % 10));
    uart_printf("  Converged: %s, iters: %d\n",
                last_result.converged ? "yes" : "no", last_result.n_iterations);
    uart_printf("  Heap high-water: %lu / %lu bytes\n\n",
                static_cast<unsigned long>(heap_high_water_bytes()),
                static_cast<unsigned long>(heap_capacity_bytes()));

    // 4. Profiled solve — per-iteration breakdown
    uart_puts("--- Profiled Solve (per-iteration timing) ---\n");

    configure_from_ref(g_controller);

    SCPResult prof = g_controller.solve<true>(REF_Z_K, REF_U_PREV, g_model_weights);

    uart_printf("  Total:       %lu us\n", static_cast<unsigned long>(prof.total_time_us));
    uart_printf("  Classify:    %lu us\n", static_cast<unsigned long>(prof.classify_time_us));
    uart_printf("    classify:  %lu us\n", static_cast<unsigned long>(prof.classify_only_us));
    uart_printf("    layout:    %lu us\n", static_cast<unsigned long>(prof.layout_us));
    uart_printf("    exprs:     %lu us\n", static_cast<unsigned long>(prof.neuron_exprs_us));
    uart_printf("    static:    %lu us\n", static_cast<unsigned long>(prof.static_constraints_us));
    uart_printf("  Init fwd:    %lu us\n", static_cast<unsigned long>(prof.initial_forward_us));
    uart_printf("  Iterations:  %d\n", prof.n_iterations);
    uart_printf("  Converged:   %s\n\n", prof.converged ? "yes" : "no");

    uart_puts("  iter  jacobian  qp_build  qp_solve  dcnn_fwd  total (us)\n");
    for (int i = 0; i < prof.n_iterations; i++) {
        const SCPIterTiming& t = prof.iter_timing[i];
        uart_printf("  %d     %lu       %lu       %lu       %lu       %lu\n",
                    i,
                    static_cast<unsigned long>(t.jacobian_us),
                    static_cast<unsigned long>(t.qp_build_us),
                    static_cast<unsigned long>(t.qp_solve_us),
                    static_cast<unsigned long>(t.dcnn_forward_us),
                    static_cast<unsigned long>(t.total_us));
    }
    uart_putc('\n');

    // 5a. Debug: show SCP result + IPM diagnostics
    uart_puts("--- SCP Debug ---\n");
    uart_printf("  piqp_status=%d, piqp_iters=%d\n", prof.last_piqp_status, prof.last_piqp_iters);
    uart_printf("  n_vars=%d, n_eq=%d, n_ineq=%d\n", prof.last_n_vars, prof.last_n_eq, prof.last_n_ineq);
    // Print IPM diagnostics with NaN detection via raw bit patterns
    {
        auto f2u = [](float f) -> uint32_t {
            uint32_t u; std::memcpy(&u, &f, 4); return u;
        };
        uart_printf("  IPM iter0: dual=%d eq=%d ineq=%d mu=%d (x1e6)\n",
                    static_cast<int>(prof.ipm_r_dual_0 * 1e6f),
                    static_cast<int>(prof.ipm_r_eq_0 * 1e6f),
                    static_cast<int>(prof.ipm_r_ineq_0 * 1e6f),
                    static_cast<int>(prof.ipm_mu_0 * 1e6f));
        uart_printf("  IPM final bits: dual=0x%08x eq=0x%08x ineq=0x%08x mu=0x%08x\n",
                    f2u(prof.ipm_r_dual_final),
                    f2u(prof.ipm_r_eq_final),
                    f2u(prof.ipm_r_ineq_final),
                    f2u(prof.ipm_mu_final));
    }
    uart_printf("  cost_x1e3=%d\n", static_cast<int>(prof.cost * 1e3f));

    // IPM sub-iteration timing breakdown (from last QP solve)
    uart_puts("\n--- IPM Sub-timing (last QP, accumulated over all IPM iters) ---\n");
    uart_printf("  residuals:   %lu us\n", static_cast<unsigned long>(prof.ipm_residuals_us));
    uart_printf("  build_kkt:   %lu us\n", static_cast<unsigned long>(prof.ipm_build_kkt_us));
    uart_printf("  ldlt_factor: %lu us\n", static_cast<unsigned long>(prof.ipm_ldlt_factor_us));
    uart_printf("  ldlt_solve:  %lu us\n", static_cast<unsigned long>(prof.ipm_ldlt_solve_us));
    uart_printf("  recover:     %lu us\n", static_cast<unsigned long>(prof.ipm_recover_us));
    uart_printf("  linesearch:  %lu us\n", static_cast<unsigned long>(prof.ipm_linesearch_us));
    {
        uint32_t ipm_total = prof.ipm_residuals_us + prof.ipm_build_kkt_us +
            prof.ipm_ldlt_factor_us + prof.ipm_ldlt_solve_us +
            prof.ipm_recover_us + prof.ipm_linesearch_us;
        uart_printf("  sum:         %lu us (vs qp_solve %lu us)\n",
                    static_cast<unsigned long>(ipm_total),
                    static_cast<unsigned long>(prof.iter_timing[prof.n_iterations - 1].qp_solve_us));
    }

    uart_printf("  validation_fail_idx=%d\n", prof.debug_validation_failed_idx);
    {
        auto f2u = [](float f) -> uint32_t {
            uint32_t u; std::memcpy(&u, &f, 4); return u;
        };
        uart_puts("  QP raw u bits:");
        for (int i = 0; i < N; i++) {
            uart_printf(" 0x%08x", f2u(prof.debug_qp_u[i]));
        }
        uart_putc('\n');
    }
    uart_puts("  SCP u_opt (x1e6):");
    for (int i = 0; i < N; i++) {
        uart_printf(" %d", static_cast<int>(prof.u_optimal[i] * 1e6f));
    }
    uart_putc('\n');

    // 5. Validation: compare u_optimal against reference
    // NOTE: use integer-based output (×1e6) to avoid double printf issues on ARM
    uart_puts("--- Validation ---\n");

    Scalar max_err = 0;
    bool has_nan = false;
    for (int i = 0; i < N; i++) {
        Scalar got = prof.u_optimal[i];
        Scalar exp = REF_U_OPTIMAL[i];
        // Explicit NaN check (NaN != NaN)
        if (got != got) {
            has_nan = true;
            uart_printf("  u[%d]: got=NaN exp=%d (x1e6)\n", i,
                        static_cast<int>(exp * 1e6f));
            continue;
        }
        Scalar err = got - exp;
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        int got_i = static_cast<int>(got * 1e6f);
        int exp_i = static_cast<int>(exp * 1e6f);
        int err_i = static_cast<int>(err * 1e6f);
        uart_printf("  u[%d]: got=%d exp=%d err=%d (x1e6)\n", i, got_i, exp_i, err_i);
    }

    // float32 tolerance: 1e-3 (same as desktop EMBEDDED_TARGET test)
    const Scalar tol = 1e-3f;
    bool pass = !has_nan && (max_err < tol);
    int max_err_i = static_cast<int>(max_err * 1e6f);
    int tol_i = static_cast<int>(tol * 1e6f);
    uart_printf("\n  Max error: %d (tol: %d) (x1e6)\n", max_err_i, tol_i);
    uart_printf("  Result: %s\n\n", pass ? "PASS" : "FAIL");

    uart_puts("========================================\n");
    uart_printf("  %s - %lu us/step mean\n",
                pass ? "ALL PASS" : "FAIL",
                static_cast<unsigned long>(mean_us));
    uart_puts("========================================\n");

    // 6. LED: toggle to indicate completion
    while (true) {
        led_toggle();
        // Simple delay (~500ms at 80MHz)
        for (volatile uint32_t i = 0; i < 4000000; i++) {}
    }

    return 0;
}
