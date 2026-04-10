#pragma once

// Types and compile-time constants for the Koopman MPC controller.
// Separate from DCNN types to allow independent compilation.

#include "koopman_config.hpp"  // Generated: D_LIFT, HORIZON, N_FEATURES, etc.
#include <cstdint>

namespace stable_neuron {

// Koopman model dimensions (from generated config)
constexpr int KN = koopman_config::HORIZON;
constexpr int KN_STATE = koopman_config::N_STATE;
constexpr int KN_STATE_Y = koopman_config::N_STATE_Y;
constexpr int KN_STATE_U = koopman_config::N_STATE_U;
constexpr int KD_LIFT = koopman_config::D_LIFT;
constexpr int KD_EXTRA = koopman_config::D_EXTRA;
constexpr int KN_FEATURES = koopman_config::N_FEATURES;

// QP dimensions (fixed — no unstable neurons)
constexpr int KN_VARS = 2 * KN;          // [u(N), t(N)]
constexpr int KN_INEQ = 6 * KN;          // tracking + nonneg + box_lo + box_hi + rate_up + rate_dn

// Precision
#ifdef EMBEDDED_TARGET
using KScalar = float;
#else
using KScalar = double;
#endif

// Weight pointers (into flash / const arrays)
struct KoopmanWeights {
    const KScalar* proj_weight;   // (D_EXTRA x N_FEATURES), column-major
    const KScalar* proj_bias;     // (D_EXTRA,)
    const KScalar* CA;            // (N x D_LIFT), column-major
    const KScalar* F;             // (N x N), column-major, lower-triangular
    const KScalar* s_max;         // (N,)
};

// Controller parameters
struct KoopmanParams {
    KScalar Q, R;
    KScalar beta_0;
    KScalar u_min, u_max, delta_u_max;
};

// Result of one Koopman MPC step
struct KoopmanResult {
    KScalar u_optimal[KN];
    KScalar cost;
    bool is_feasible;
    int qp_status;
    int qp_iters;

    // Timing (only populated when solve<true>)
    float encode_time_us;     // Encoder forward (feature computation + projection)
    float predict_time_us;    // e = CA @ psi
    float qp_build_time_us;   // QP matrix update
    float qp_solve_time_us;   // IPM solve
    float total_time_us;

    // IPM diagnostics
    float ipm_r_dual_final;
    float ipm_mu_final;
};

}  // namespace stable_neuron
