#include "condensed_solver/icnn_forward.hpp"
#include "condensed_solver/math_ops.hpp"

#include <algorithm>

namespace condensed {

using namespace math;

Scalar icnn_forward(const Scalar* z_k, const Scalar* u, int n_u,
                    const NetworkWeights& w)
{
    const int n_input = N_STATE + n_u;

    // Build input vector x = [z_k, u]
    Scalar x_buf[MAX_N_INPUT];
    vec_copy(x_buf, z_k, N_STATE);
    vec_copy(x_buf + N_STATE, u, n_u);

    // Workspace: pre-activation z, post-activation h
    Scalar h[N_HIDDEN], z_vec[N_HIDDEN];

    // First layer: z = W0 @ x + b0, h = ReLU(z)
    // Product from 0 then add bias (matches Eigen accumulation order for float32)
    matvec(z_vec, w.W0, x_buf, N_HIDDEN, n_input);
    for (int j = 0; j < N_HIDDEN; ++j) z_vec[j] += w.b0[j];
    for (int j = 0; j < N_HIDDEN; ++j)
        h[j] = std::max(z_vec[j], Scalar(0));

    // Internal layers
    for (int i = 0; i < N_INTERNAL; ++i) {
        // z = Wx[i] * h + bx[i] + W_skip[i] * x + b_skip[i]
        // Products from 0 then add biases (matches Eigen accumulation order)
        Scalar tmp[N_HIDDEN];
        matvec(z_vec, w.Wx[i], h, N_HIDDEN, N_HIDDEN);
        matvec(tmp, w.W_skip[i], x_buf, N_HIDDEN, n_input);
        for (int j = 0; j < N_HIDDEN; ++j)
            z_vec[j] += tmp[j] + w.bx[i][j] + w.b_skip[i][j];
        for (int j = 0; j < N_HIDDEN; ++j)
            h[j] = std::max(z_vec[j], Scalar(0));
    }

    // Output layer: y = W_out . h + b_out
    return vec_dot(w.W_out, h, N_HIDDEN) + w.b_out;
}

Scalar icnn_forward_cached(const Scalar* c0, const Scalar* const* skip_const,
                           const Scalar* u, int n_u,
                           const NetworkWeights& w)
{
    Scalar h[N_HIDDEN], z_vec[N_HIDDEN];

    // W0_u starts at column N_STATE
    const Scalar* W0_u = w.W0 + N_STATE * N_HIDDEN;

    // First layer: z = c0 + W0_u * u, h = ReLU(z)
    // Product from 0 then add cached constant (matches Eigen accumulation order)
    matvec(z_vec, W0_u, u, N_HIDDEN, n_u);
    for (int j = 0; j < N_HIDDEN; ++j) z_vec[j] += c0[j];
    for (int j = 0; j < N_HIDDEN; ++j)
        h[j] = std::max(z_vec[j], Scalar(0));

    // Internal layers
    for (int i = 0; i < N_INTERNAL; ++i) {
        const Scalar* W_skip_u = w.W_skip[i] + N_STATE * N_HIDDEN;

        // z = Wx[i] * h + skip_const[i] + W_skip_u * u
        // Products from 0 then add cached constant (matches Eigen accumulation order)
        Scalar tmp[N_HIDDEN];
        matvec(z_vec, w.Wx[i], h, N_HIDDEN, N_HIDDEN);
        matvec(tmp, W_skip_u, u, N_HIDDEN, n_u);
        for (int j = 0; j < N_HIDDEN; ++j)
            z_vec[j] += tmp[j] + skip_const[i][j];
        for (int j = 0; j < N_HIDDEN; ++j)
            h[j] = std::max(z_vec[j], Scalar(0));
    }

    // Output layer
    return vec_dot(w.W_out, h, N_HIDDEN) + w.b_out;
}

}  // namespace condensed
