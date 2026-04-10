#pragma once

#include "stable_neuron_solver/config.hpp"
#include "stable_neuron_solver/types.hpp"
#include "stable_neuron_solver/weights.hpp"
#include "stable_neuron_solver/solver.hpp"

namespace stable_neuron {

// SCP algorithm parameters
struct SCPParams {
    Scalar Q, R, R_delta;
    Scalar tube_weight;
    Scalar beta_0;
    Scalar u_min, u_max, delta_u_max;
    Scalar delta_J_min;   // Convergence threshold (default 1e-5)
    Scalar delta_u_tol;   // Adaptive SCP: skip re-linearization if |u_opt-u_nom|_inf < tol
                          // Set to 0 to disable (always run maxiters). Typical: 0.1*delta_u_max
    int maxiters;         // Max SCP iterations (default 10)
};

// Per-iteration timing breakdown (matches Python SCPProfiler categories)
struct SCPIterTiming {
    float jacobian_us;       // Jacobian computation (combined forward+jacobian)
    float qp_build_us;       // QP matrix construction
    float qp_solve_us;       // PIQP solve
    float dcnn_forward_us;   // Prediction recompute after QP
    float total_us;          // Total for this iteration
};

// Result of one full MPC timestep (SCP solve)
struct SCPResult {
    Scalar u_optimal[N];
    Scalar cost;
    int n_iterations;
    bool converged;

    // Last QP solver diagnostics
    int last_piqp_status;   // PIQP status code from final QP solve
    int last_piqp_iters;    // PIQP iteration count from final QP solve
    int last_n_vars;        // QP dimensions
    int last_n_eq;
    int last_n_ineq;
    Scalar diag_q0;         // QP matrix diagnostics
    Scalar diag_P00;
    Scalar diag_G00;

    // Timing fields — only populated when solve<true>() is used.
    // When solve<false>(), all timing fields are zero.
    float classify_time_us;
    // Sub-breakdown of classify_and_setup:
    float classify_only_us;     // neuron_classifier.classify()
    float layout_us;            // build_variable_layout()
    float neuron_exprs_us;      // precompute_neuron_exprs()
    float static_constraints_us; // precompute_static_constraints()
    float initial_forward_us;   // Initial predictions (one-time)
    float total_time_us;

    // Per-iteration detail (up to maxiters entries)
    static constexpr int MAX_ITERS = 10;
    SCPIterTiming iter_timing[MAX_ITERS];

    // Convenience: aggregate per-timestep totals
    float forward_time_us;      // All forward passes (initial + recomputes)
    float jacobian_time_us;     // All Jacobian computations
    float qp_time_us;           // All QP solves (build + solve)

    // IPM debug diagnostics (from last QP solve)
    float ipm_r_dual_0;
    float ipm_r_eq_0;
    float ipm_r_ineq_0;
    float ipm_mu_0;
    float ipm_r_dual_final;
    float ipm_r_eq_final;
    float ipm_r_ineq_final;
    float ipm_mu_final;
    // IPM sub-iteration timing (accumulated microseconds, from last QP solve)
    uint32_t ipm_residuals_us;
    uint32_t ipm_build_kkt_us;
    uint32_t ipm_ldlt_factor_us;
    uint32_t ipm_ldlt_solve_us;
    uint32_t ipm_recover_us;
    uint32_t ipm_linesearch_us;

    // Debug: raw QP u_optimal from first SCP iteration (before validation)
    Scalar debug_qp_u[N];
    int debug_validation_failed_idx;  // -1 if passed, else index of first failure
};

// Full DCNN-MPC pipeline: z_k + u_prev -> u_optimal.
//
// Implements Algorithm 1 from CDC25:
//   1. Warm-start u_nominal from previous solution (shift by 1)
//   2. classify_and_setup (once per timestep)
//   3. SCP loop:
//      a. Compute analytical Jacobians df1/du, df2/du (also gets predictions)
//      b. Solve stability-reduced QP
//      c. Check convergence: |u_opt-u_nom|_inf < delta_u_tol (adaptive) OR |delta_J| < delta_J_min
//      d. Update u_nominal, recompute predictions
//   4. Return u_optimal[0..N-1]
//
// Template parameter Profile controls timing instrumentation:
//   solve<false>() — no timing overhead, all timing fields in SCPResult are zero
//   solve<true>()  — full per-iteration timing breakdown
class SCPController {
public:
    // Configure controller parameters and pass W_bounds (N*2 interleaved).
    void configure(const SCPParams& params, const Scalar* W_bounds);

    // Full pipeline: z_k + u_prev -> u_optimal.
    // Profile=false (default): zero timing overhead, for production/trial runs.
    // Profile=true: detailed per-iteration timing, for profiling.
    template<bool Profile = false>
    SCPResult solve(const Scalar* z_k, Scalar u_prev,
                    const ModelWeights& weights);

private:
    StableNeuronSolver qp_solver_;
    SCPParams params_;
    Scalar W_bounds_[N * 2];

    // Warm-start state (persists between timesteps)
    bool has_previous_;
    Scalar prev_u_optimal_[N];

    // SCP workspace (reused across iterations)
    Scalar u_nominal_[N];
    Scalar y_nominal_[N];
    Scalar f1_nominal_[N];
    Scalar f2_nominal_[N];
    Scalar jac_f1_packed_[N * (N + 1) / 2];  // 15 floats
    Scalar jac_f2_packed_[N * (N + 1) / 2];  // 15 floats

    // Compute nominal predictions (forward pass only) for all N steps.
    void compute_predictions(const Scalar* z_k, const ModelWeights& weights);

    // Compute packed Jacobians for all N steps.
    // Also computes f1/f2/y as a side effect (combined forward+jacobian).
    void compute_jacobians_and_predictions(const Scalar* z_k,
                                            const ModelWeights& weights);
};

// Explicit instantiation declarations (defined in scp_controller.cpp)
extern template SCPResult SCPController::solve<false>(const Scalar*, Scalar, const ModelWeights&);
extern template SCPResult SCPController::solve<true>(const Scalar*, Scalar, const ModelWeights&);

}  // namespace stable_neuron
