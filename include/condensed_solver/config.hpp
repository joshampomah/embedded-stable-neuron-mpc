#pragma once

// Compile-time constants for the condensed QP solver.
// Model dimensions are sourced from generated/model_config.hpp (from export_weights.py).

#include "model_config.hpp"  // Generated: N_STATE, N_STATE_Y, N_STATE_U, N_HIDDEN, N_LAYERS, HORIZON

namespace condensed {

// Model architecture (from generated model_config.hpp)
constexpr int N = model_config::HORIZON;
constexpr int N_STATE_Y = model_config::N_STATE_Y;
constexpr int N_STATE_U = model_config::N_STATE_U;
constexpr int N_STATE = model_config::N_STATE;
constexpr int N_HIDDEN = model_config::N_HIDDEN;
constexpr int N_LAYERS = model_config::N_LAYERS;
constexpr int N_INTERNAL = N_LAYERS - 1;  // Number of internal (skip) layers
constexpr int N_NETWORKS = 2;     // f1 and f2 per step

// QP variable bounds (tight for embedded 128KB SRAM)
// Variable layout: [u(N), s_max(N), s_min(N), t(N), y_f?(<=2N), h_amb?(...)]
constexpr int N_FIXED_VARS = 4 * N;  // u, s_max, s_min, t = 20

// Typical: ~96% neurons are safe -> ~4 ambiguous out of 320 total
// Observed max at 50Hz: 6, at 10/25Hz cold-start: 14-15.
// ARM: limited to 10 by 32KB SRAM2 heap (excess neurons force-classified).
// Desktop: 16 for full accuracy.
#ifdef EMBEDDED_TARGET
constexpr int MAX_AMB_VARS = 10;
constexpr int MAX_EQ = 4;           // Observed ne<=4 across all rates
#else
constexpr int MAX_AMB_VARS = 16;
constexpr int MAX_EQ = 2 * N;       // One per y_f variable (up to N steps x 2 networks = 10)
#endif

// y_f variables: at most 2 per step, capped by MAX_EQ (each needs one eq constraint).
constexpr int MAX_Y_F_VARS = (2 * N < MAX_EQ) ? 2 * N : MAX_EQ;
constexpr int MAX_VARS = N_FIXED_VARS + MAX_Y_F_VARS + MAX_AMB_VARS;

// Constraint bounds
// Static: u_bounds(2N=10) + rate(2N=10) + tube(3N=15) + epigraph(2*MAX_AMB)
// Dynamic: tracking(2N=10) + DC_bounds(2N=10)
constexpr int MAX_STATIC_INEQ = 7 * N + 2 * MAX_AMB_VARS;
constexpr int MAX_DYNAMIC_INEQ = 4 * N;
constexpr int MAX_INEQ = MAX_STATIC_INEQ + MAX_DYNAMIC_INEQ;

// Weight dimensions per network
// First layer: W0 (N_HIDDEN x n_input), b0 (N_HIDDEN)
// Internal layer: Wx (N_HIDDEN x N_HIDDEN), bx (N_HIDDEN),
//                 W_skip (N_HIDDEN x n_input), b_skip (N_HIDDEN)
// Output layer: W_out (1 x N_HIDDEN), b_out (1)
// n_input varies per step: N_STATE + (step+1) for step in [0, N)
constexpr int MAX_N_INPUT = N_STATE + N;  // Largest input (step=N-1)

// Default controller parameters (from device_params.json)
// These can be overridden at runtime
constexpr float DEFAULT_U_MIN = 0.0f;
constexpr float DEFAULT_U_MAX = 0.030f;
constexpr float DEFAULT_DELTA_U_MAX = 0.0024f;
constexpr float DEFAULT_BETA_0 = 2.3f;
constexpr float DEFAULT_TUBE_WEIGHT = 0.01f;

// Neuron classification margin
constexpr float DEFAULT_MARGIN = 1e-4f;

}  // namespace condensed
