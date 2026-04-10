#pragma once
// Synthetic reference test case — demo values, not patient data

#include "stable_neuron_solver/types.hpp"
#include "stable_neuron_solver/scp_controller.hpp"

namespace firmware_ref {

using stable_neuron::Scalar;

static constexpr Scalar REF_Q = 3.00000000e+00f;
static constexpr Scalar REF_R = 2.00000000e+02f;
static constexpr Scalar REF_R_DELTA = 0.00000000e+00f;
static constexpr Scalar REF_TUBE_WEIGHT = 0.00000000e+00f;
static constexpr Scalar REF_BETA_0 = 2.30000000e+00f;
static constexpr Scalar REF_U_MIN = 0.00000000e+00f;
static constexpr Scalar REF_U_MAX = 3.00000000e-02f;
static constexpr Scalar REF_DELTA_U_MAX = 2.40000000e-03f;
static constexpr Scalar REF_DELTA_J_MIN = 1.00000000e-03f;
static constexpr int REF_MAXITERS = 10;

static const Scalar REF_W_BOUNDS[10] = {
    -2.40000000e-02f, 2.40000000e-02f,
    -5.20000000e-02f, 5.20000000e-02f,
    -7.50000000e-02f, 7.50000000e-02f,
    -9.40000000e-02f, 9.40000000e-02f,
    -1.14000000e-01f, 1.14000000e-01f,
};

static const Scalar REF_Z_K[30] = {
    2.31257302e+00f, 2.28678951e+00f, 2.36404227e+00f, 2.31049001e+00f, 2.24643306e+00f,
    2.33615951e+00f, 2.43040000e+00f, 2.39470810e+00f, 2.22962648e+00f, 2.17345785e+00f,
    2.23767255e+00f, 2.30413260e+00f, 2.06749692e+00f, 2.27812083e+00f, 2.17540891e+00f,
    1.03267010e-02f, 5.83382136e-03f, 2.02644758e-03f, 1.08223251e-02f, 7.88031484e-03f,
    4.65362813e-03f, 7.28753038e-03f, 1.33423175e-02f, 1.40106527e-02f, 5.36692795e-03f,
    8.57294746e-03f, 4.82804087e-03f, 8.91450045e-03f, 5.06866838e-03f, 5.87428501e-03f,
};

static constexpr Scalar REF_U_PREV = 5.87428501e-03f;

static const Scalar REF_U_OPTIMAL[5] = {
    1.36832922e-02f, 5.72589112e-03f, 1.04782457e-02f, 4.00818412e-03f, 1.29917298e-02f,
};

static constexpr Scalar REF_COST = 2.00000000e-01f;
static constexpr int REF_N_SCP_ITERS = 2;

}  // namespace firmware_ref
