#pragma once

#include "stable_neuron_solver/config.hpp"
#include "stable_neuron_solver/types.hpp"
#include "stable_neuron_solver/weights.hpp"

namespace stable_neuron {

// Compute df/du for one ICNN network (B matrix only, not A = df/dstate).
//
// Ports jacobian.py:compute_jacobian_analytical() — control columns only.
// Result J_u has shape (1, n_u), written to out_J[0..n_u-1].
//
// Args:
//   z_k:   State vector of length N_STATE.
//   u:     Control sequence of length n_u.
//   n_u:   Number of control inputs (step+1).
//   w:     Network weights.
//   out_J: Output buffer of length n_u. Receives df/du as a row vector.
void icnn_jacobian_u(const Scalar* z_k, const Scalar* u, int n_u,
                     const NetworkWeights& w, Scalar* out_J);

// Combined forward pass + Jacobian computation.
// Avoids redundant computation since the Jacobian chain rule already
// computes h and z at each layer (same as forward pass).
//
// Returns: Scalar forward pass value f(z_k, u).
// out_J:  Receives df/du of length n_u.
Scalar icnn_forward_and_jacobian_u(const Scalar* z_k, const Scalar* u, int n_u,
                                    const NetworkWeights& w, Scalar* out_J);

// Cached variant: accepts pre-computed c0 and skip_const to avoid
// redundant 32x30 matvecs with the z portion of weight matrices.
Scalar icnn_forward_and_jacobian_u_cached(
    const Scalar* c0, const Scalar* const* skip_const,
    const Scalar* u, int n_u,
    const NetworkWeights& w, Scalar* out_J);

}  // namespace stable_neuron
