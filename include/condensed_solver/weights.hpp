#pragma once

#include "condensed_solver/config.hpp"
#include "condensed_solver/types.hpp"

namespace condensed {

// Weight arrays for one ICNN network at one prediction step.
// All arrays are stored in column-major order (Eigen default).
// Wx and W_out weights are pre-clamped to non-negative.
struct NetworkWeights {
    // First layer: W0 (N_HIDDEN × n_input), b0 (N_HIDDEN)
    const Scalar* W0;   // N_HIDDEN rows × n_input cols
    const Scalar* b0;   // N_HIDDEN

    // Internal layers (N_INTERNAL of them):
    // Wx (N_HIDDEN × N_HIDDEN), bx (N_HIDDEN),
    // W_skip (N_HIDDEN × n_input), b_skip (N_HIDDEN)
    const Scalar* Wx[N_INTERNAL];       // Each: N_HIDDEN × N_HIDDEN (pre-clamped >= 0)
    const Scalar* bx[N_INTERNAL];       // Each: N_HIDDEN
    const Scalar* W_skip[N_INTERNAL];   // Each: N_HIDDEN × n_input
    const Scalar* b_skip[N_INTERNAL];   // Each: N_HIDDEN

    // Output layer: W_out (1 × N_HIDDEN), b_out (scalar)
    const Scalar* W_out;  // 1 × N_HIDDEN (pre-clamped >= 0)
    Scalar b_out;

    int n_input;  // N_STATE + (step+1) — varies per step
};

// All network weights for the full model
struct ModelWeights {
    // weights[step][net_idx]: step in [0,N), net_idx: 0=f1, 1=f2
    NetworkWeights nets[N][N_NETWORKS];
};

// Declared in generated/network_weights.hpp, defined by export script
extern const ModelWeights g_model_weights;

}  // namespace condensed
