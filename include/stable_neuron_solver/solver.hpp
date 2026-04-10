#pragma once

#include "stable_neuron_solver/config.hpp"
#include "stable_neuron_solver/types.hpp"
#include "stable_neuron_solver/weights.hpp"
#include "stable_neuron_solver/qp_builder.hpp"

namespace stable_neuron {

// Top-level stability-reduced QP solver: QPBuilder + PIQP dense.
//
// Lifecycle:
//   1. configure() — set Q, R, constraint parameters
//   2. classify_and_setup() — when z_k or u_prev changes (new MPC timestep)
//   3. solve() — per SCP iteration (warm-starts if same timestep)
class StableNeuronSolver {
public:
    StableNeuronSolver();
    ~StableNeuronSolver();

    // Non-copyable (owns PIQP solver)
    StableNeuronSolver(const StableNeuronSolver&) = delete;
    StableNeuronSolver& operator=(const StableNeuronSolver&) = delete;

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
    int n_unstable() const { return builder_.n_unstable(); }

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

}  // namespace stable_neuron
