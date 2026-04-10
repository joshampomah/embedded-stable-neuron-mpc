#pragma once

#include "stable_neuron_solver/config.hpp"
#include "stable_neuron_solver/types.hpp"
#include "stable_neuron_solver/weights.hpp"

namespace stable_neuron {

// Compute f(x) for one ICNN network given state z_k and control sequence u.
//
// Ports jacobian.py:forward_from_weights().
// Input is conceptually x = [z_k, u[0:n_u]] of length N_STATE + n_u.
// The weight matrices W0, W_skip are (N_HIDDEN x n_input) stored column-major.
//
// Args:
//   z_k:  State vector of length N_STATE.
//   u:    Control sequence of length n_u (1..N).
//   n_u:  Number of control inputs for this step (step+1).
//   w:    Network weights for this (step, net).
//
// Returns: Scalar output value f(z_k, u).
Scalar icnn_forward(const Scalar* z_k, const Scalar* u, int n_u,
                    const NetworkWeights& w);

// Cached variant: accepts pre-computed c0 (W0_z*z+b0) and skip_const
// (W_skip_z*z+bx+b_skip) to avoid redundant 32x30 matvecs.
// c0: (N_HIDDEN,) cached from NeuronClassifier
// skip_const: (N_INTERNAL * N_HIDDEN,) cached, indexed [layer][N_HIDDEN]
Scalar icnn_forward_cached(const Scalar* c0, const Scalar* const* skip_const,
                           const Scalar* u, int n_u,
                           const NetworkWeights& w);

}  // namespace stable_neuron
