#pragma once

// Koopman MPC controller — single QP per timestep, no SCP.
//
// Pipeline per timestep:
//   1. Encode: psi = encoder(z_k)
//   2. Predict: e = CA @ psi  (state-dependent offsets)
//   3. Nominal: y_nom = e + F @ u_nom  (F is precomputed constant)
//   4. Build QP: update tracking rows/RHS
//   5. Solve QP: custom IPM or PIQP
//   6. Return u_optimal
//
// Compared to DCNN SCP controller:
//   - No neuron classification (no interval arithmetic)
//   - No SCP loop (single QP — predictions are affine in u)
//   - No Jacobian computation (F is constant)
//   - Smaller QP: 2N vars, 6N constraints (vs ~24 vars, ~60 constraints)

#include "condensed_solver/koopman_types.hpp"
#include "condensed_solver/koopman_encoder.hpp"

namespace condensed {

class KoopmanController {
public:
    KoopmanController();

    // Configure controller parameters.
    void configure(const KoopmanParams& params);

    // Full pipeline: z_k + u_prev -> u_optimal.
    // Template parameter Profile controls timing instrumentation.
    template<bool Profile = false>
    KoopmanResult solve(const KScalar* z_k, KScalar u_prev,
                        const KoopmanWeights& weights);

    // Reset warm-start state.
    void reset();

private:
    KoopmanEncoder encoder_;
    KoopmanParams params_;
    bool has_previous_;
    KScalar prev_u_optimal_[KN];

    // Pre-allocated QP matrices (fixed structure, updated per step)
    // P: (2N x 2N) diagonal cost — set once in configure()
    // G: (6N x 2N) constraint matrix — tracking rows updated per step
    // h: (6N) constraint RHS — tracking + rate rows updated per step
    KScalar P_[KN_VARS * KN_VARS];
    KScalar q_[KN_VARS];
    KScalar G_[KN_INEQ * KN_VARS];
    KScalar h_lo_[KN_INEQ];
    KScalar h_hi_[KN_INEQ];

    // Workspace
    KScalar psi_[KD_LIFT];
    KScalar e_[KN];
    KScalar u_nominal_[KN];
    KScalar y_nominal_[KN];

    // Build fixed QP structure (called once in configure).
    void build_fixed_qp();

    // Update QP tracking rows and RHS for current step.
    void update_qp(const KScalar* y_nom, const KScalar* F,
                   const KScalar* u_nom, KScalar u_prev,
                   const KScalar* s_max);
};

// Explicit instantiation declarations
extern template KoopmanResult KoopmanController::solve<false>(const KScalar*, KScalar, const KoopmanWeights&);
extern template KoopmanResult KoopmanController::solve<true>(const KScalar*, KScalar, const KoopmanWeights&);

}  // namespace condensed
