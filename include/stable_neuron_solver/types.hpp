#pragma once

#include <cstdint>
#include <Eigen/Dense>
#include "stable_neuron_solver/config.hpp"

namespace stable_neuron {

// Precision: float32 on embedded (hardware FPU), double on desktop (match Python)
#ifdef EMBEDDED_TARGET
using Scalar = float;
#else
using Scalar = double;
#endif

// Eigen type aliases
using VectorN = Eigen::Matrix<Scalar, N, 1>;
using VectorState = Eigen::Matrix<Scalar, N_STATE, 1>;
using VectorHidden = Eigen::Matrix<Scalar, N_HIDDEN, 1>;

// Dynamic-size types for variable-dimension QP
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

// Row-major for C storage order (column-major available via Eigen default)
using MatrixXRM = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Neuron classification status
enum class NeuronStatus : int8_t {
    STABLE_INACTIVE = -1,
    UNSTABLE = 0,
    STABLE_ACTIVE = 1,
};

// Classification result for one (step, net, layer)
struct LayerClassification {
    int8_t status[N_HIDDEN];    // NeuronStatus per neuron
    Scalar h_min[N_HIDDEN];     // Post-ReLU lower bounds
    Scalar h_max[N_HIDDEN];     // Post-ReLU upper bounds
};

// y_f variable info
struct YFInfo {
    bool is_var;     // Whether this network output needs a QP variable
    int var_idx;     // Index into QP variable vector (-1 if not a variable)
};

// Solver result
struct SolverResult {
    Scalar u_optimal[N];
    Scalar s_max_optimal[N];
    Scalar s_min_optimal[N];
    Scalar cost;
    bool is_feasible;
    int n_vars;
    int n_eq;
    int n_ineq;
    int n_unstable;
    int piqp_status;   // PIQP status code (1=solved, -1=max_iter, -8=numerics, etc.)
    int piqp_iters;    // PIQP internal iteration count
    Scalar diag_q0;    // q[0] for debugging QP matrix comparison
    Scalar diag_P00;   // P[0,0]
    Scalar diag_G00;   // G[0,0]
    float classify_time_us;
    float build_time_us;
    float solve_time_us;
    float total_time_us;
    // IPM debug diagnostics
    float ipm_r_dual_0;
    float ipm_r_eq_0;
    float ipm_r_ineq_0;
    float ipm_mu_0;
    float ipm_r_dual_final;
    float ipm_r_eq_final;
    float ipm_r_ineq_final;
    float ipm_mu_final;
    // IPM sub-iteration timing (accumulated microseconds)
    uint32_t ipm_residuals_us;
    uint32_t ipm_build_kkt_us;
    uint32_t ipm_ldlt_factor_us;
    uint32_t ipm_ldlt_solve_us;
    uint32_t ipm_recover_us;
    uint32_t ipm_linesearch_us;
};

// QP matrices (views into pre-allocated storage)
struct QPData {
    int n_vars;
    int n_eq;
    int n_ineq;

    // These point to internal storage in QPBuilder
    const Scalar* P;       // (n_vars × n_vars), upper triangular
    const Scalar* q;       // (n_vars)
    const Scalar* A_eq;    // (n_eq × n_vars)
    const Scalar* b_eq;    // (n_eq)
    const Scalar* A_ineq;  // (n_ineq × n_vars)
    const Scalar* h_lo;    // (n_ineq)
    const Scalar* h_hi;    // (n_ineq)
};

}  // namespace stable_neuron
