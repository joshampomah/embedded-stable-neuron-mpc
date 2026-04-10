#pragma once

#include "condensed_solver/config.hpp"
#include "condensed_solver/types.hpp"
#include "condensed_solver/weights.hpp"
#include "condensed_solver/qp_builder.hpp"

namespace condensed {

// Top-level condensed QP solver: QPBuilder + PIQP dense.
//
// Lifecycle:
//   1. configure() — set Q, R, constraint parameters
//   2. classify_and_setup() — when z_k or u_prev changes (new MPC timestep)
//   3. solve() — per SCP iteration (warm-starts if same timestep)
class CondensedSolver {
public:
    CondensedSolver();
    ~CondensedSolver();

    // Non-copyable (owns PIQP solver)
    CondensedSolver(const CondensedSolver&) = delete;
    CondensedSolver& operator=(const CondensedSolver&) = delete;

    // Configure parameters
    void configure(
        Scalar Q, Scalar R, Scalar R_delta,
        Scalar tube_weight, Scalar beta_0,
        Scalar u_min, Scalar u_max, Scalar delta_u_max,
        Scalar margin = DEFAULT_MARGIN
    );

    // Classify neurons and precompute (call when z_k or u_prev change)
    void classify_and_setup(
        const Scalar* z_k,
        Scalar u_prev,
        const ModelWeights& weights
    );

    // Solve the QP (call per SCP iteration)
    SolverResult solve(
        const Scalar* z_k,
        Scalar u_prev,
        const Scalar* u_nominal,
        const Scalar* y_nominal,
        const Scalar* jacobians_f1,
        const Scalar* jacobians_f2,
        const Scalar* f1_nominal,
        const Scalar* f2_nominal,
        const Scalar* W_bounds,
        const Scalar* linear_jacobians = nullptr
    );

    int n_vars() const { return builder_.n_vars(); }
    int n_amb() const { return builder_.n_amb(); }

    // Access classifier for cached z-dependent constants
    const NeuronClassifier& classifier() const { return builder_.classifier(); }

    // Access classify sub-timing from last classify_and_setup()
    const QPBuilder::ClassifyTiming& builder_classify_timing() const {
        return builder_.classify_timing();
    }

private:
    QPBuilder builder_;

    // PIQP solver (forward-declared implementation)
    struct PiqpImpl;
    PiqpImpl* piqp_;

    // Dimension tracking for PIQP setup/update decision.
    // When dimensions change, PIQP must be deleted and re-created (setup()).
    // When dimensions match, update() is used (preserves warm-start, no allocation).
    int last_nv_, last_ne_, last_ni_;

    // IPM warm-start: previous solution for warm-starting across MPC timesteps.
    // Stored as the full x vector (u + epigraph vars).
    bool has_prev_x_;
    int prev_nv_;
    Scalar prev_x_[MAX_VARS];
};

}  // namespace condensed
