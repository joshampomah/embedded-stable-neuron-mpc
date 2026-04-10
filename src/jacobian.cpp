#include "condensed_solver/jacobian.hpp"
#include "condensed_solver/math_ops.hpp"

#include <algorithm>

namespace condensed {

using namespace math;

Scalar icnn_forward_and_jacobian_u(const Scalar* z_k, const Scalar* u, int n_u,
                                    const NetworkWeights& w, Scalar* out_J)
{
    const int n_input = N_STATE + n_u;

    // Build input vector x = [z_k, u]
    Scalar x_buf[MAX_N_INPUT];
    vec_copy(x_buf, z_k, N_STATE);
    vec_copy(x_buf + N_STATE, u, n_u);

    // Workspace
    Scalar h[N_HIDDEN], z_vec[N_HIDDEN], dsigma[N_HIDDEN];

    // B matrix: (N_HIDDEN x n_u) col-major — Jacobian of hidden layer w.r.t. u
    Scalar B_buf[N_HIDDEN * N];

    // ---- First layer ----
    // z = W0 @ x + b0
    // Product from 0 then add bias (matches Eigen accumulation order for float32)
    matvec(z_vec, w.W0, x_buf, N_HIDDEN, n_input);
    for (int j = 0; j < N_HIDDEN; ++j) z_vec[j] += w.b0[j];
    for (int j = 0; j < N_HIDDEN; ++j) {
        h[j] = std::max(z_vec[j], Scalar(0));
        dsigma[j] = z_vec[j] > Scalar(0) ? Scalar(1) : Scalar(0);
    }

    // B = dsigma .* W0_u  (W0_u = columns N_STATE..N_STATE+n_u-1 of W0)
    const Scalar* W0_u = w.W0 + N_STATE * N_HIDDEN;
    for (int m = 0; m < n_u; ++m) {
        const Scalar* w_col = W0_u + m * N_HIDDEN;
        Scalar* b_col = B_buf + m * N_HIDDEN;
        for (int j = 0; j < N_HIDDEN; ++j)
            b_col[j] = dsigma[j] * w_col[j];
    }

    // ---- Internal layers ----
    for (int i = 0; i < N_INTERNAL; ++i) {
        // z = Wx @ h + bx + W_skip @ x + b_skip
        // Products from 0 then add biases (matches Eigen accumulation order)
        Scalar tmp_z[N_HIDDEN];
        matvec(z_vec, w.Wx[i], h, N_HIDDEN, N_HIDDEN);
        matvec(tmp_z, w.W_skip[i], x_buf, N_HIDDEN, n_input);
        for (int j = 0; j < N_HIDDEN; ++j)
            z_vec[j] += tmp_z[j] + w.bx[i][j] + w.b_skip[i][j];
        for (int j = 0; j < N_HIDDEN; ++j) {
            h[j] = std::max(z_vec[j], Scalar(0));
            dsigma[j] = z_vec[j] > Scalar(0) ? Scalar(1) : Scalar(0);
        }

        // B = dsigma .* (Wx @ B + W_skip_u)
        // Step 1: tmp = Wx @ B  (GEMM: N_HIDDEN x N_HIDDEN @ N_HIDDEN x n_u)
        Scalar tmp[N_HIDDEN * N];
        gemm(tmp, w.Wx[i], B_buf, N_HIDDEN, N_HIDDEN, n_u);

        // Step 2: tmp += W_skip_u (columns N_STATE..N_STATE+n_u-1 of W_skip)
        const Scalar* W_skip_u = w.W_skip[i] + N_STATE * N_HIDDEN;
        for (int m = 0; m < n_u; ++m) {
            Scalar* t_col = tmp + m * N_HIDDEN;
            const Scalar* w_col = W_skip_u + m * N_HIDDEN;
            for (int j = 0; j < N_HIDDEN; ++j) t_col[j] += w_col[j];
        }

        // Step 3: B = dsigma .* tmp (row-wise scaling)
        for (int m = 0; m < n_u; ++m) {
            const Scalar* t_col = tmp + m * N_HIDDEN;
            Scalar* b_col = B_buf + m * N_HIDDEN;
            for (int j = 0; j < N_HIDDEN; ++j)
                b_col[j] = dsigma[j] * t_col[j];
        }
    }

    // ---- Output layer ----
    // y = W_out . h + b_out
    Scalar y = vec_dot(w.W_out, h, N_HIDDEN) + w.b_out;

    // J_u = W_out @ B  (1 x n_u)
    // out_J[m] = dot(W_out, B[:,m])
    for (int m = 0; m < n_u; ++m)
        out_J[m] = vec_dot(w.W_out, B_buf + m * N_HIDDEN, N_HIDDEN);

    return y;
}

void icnn_jacobian_u(const Scalar* z_k, const Scalar* u, int n_u,
                     const NetworkWeights& w, Scalar* out_J)
{
    icnn_forward_and_jacobian_u(z_k, u, n_u, w, out_J);
}

Scalar icnn_forward_and_jacobian_u_cached(
    const Scalar* c0, const Scalar* const* skip_const,
    const Scalar* u, int n_u,
    const NetworkWeights& w, Scalar* out_J)
{
    Scalar h[N_HIDDEN], z_vec[N_HIDDEN];
    Scalar B_buf[N_HIDDEN * N];

    const Scalar* W0_u = w.W0 + N_STATE * N_HIDDEN;

    // ---- First layer (using cached c0 = W0_z * z + b0) ----
    // Product from 0 then add cached constant (matches Eigen accumulation order)
    matvec(z_vec, W0_u, u, N_HIDDEN, n_u);
    for (int j = 0; j < N_HIDDEN; ++j) z_vec[j] += c0[j];

    // ReLU + build active index for sparse GEMM in internal layers.
    // Active neurons (dsigma=1) have nonzero B rows; inactive ones (dsigma=0) are
    // skipped in the Wx@B GEMM and Wx@h matvec, saving ~50% of FMAs + flash reads.
    int n_active = 0;
    int active_idx[N_HIDDEN];
    for (int j = 0; j < N_HIDDEN; ++j) {
        if (z_vec[j] > Scalar(0)) {
            h[j] = z_vec[j];
            active_idx[n_active++] = j;
        } else {
            h[j] = Scalar(0);
        }
    }

    // B = dsigma .* W0_u (only active rows are nonzero)
    for (int m = 0; m < n_u; ++m) {
        const Scalar* w_col = W0_u + m * N_HIDDEN;
        Scalar* b_col = B_buf + m * N_HIDDEN;
        vec_zero(b_col, N_HIDDEN);
        for (int a = 0; a < n_active; ++a) {
            int j = active_idx[a];
            b_col[j] = w_col[j];  // dsigma[j]=1 for active
        }
    }

    // ---- Internal layers (using cached skip_const) ----
    for (int i = 0; i < N_INTERNAL; ++i) {
        const Scalar* W_skip_u = w.W_skip[i] + N_STATE * N_HIDDEN;

        // z = Wx * h + skip_const + W_skip_u * u
        // Sparse matvec: only accumulate active columns of Wx (where h != 0)
        vec_zero(z_vec, N_HIDDEN);
        for (int a = 0; a < n_active; ++a) {
            int p = active_idx[a];
            Scalar hp = h[p];
            const Scalar* wx_col = w.Wx[i] + p * N_HIDDEN;
            for (int j = 0; j < N_HIDDEN; ++j)
                z_vec[j] += hp * wx_col[j];
        }
        // Products from 0 then add cached constants (matches Eigen accumulation order)
        Scalar tmp_z[N_HIDDEN];
        matvec(tmp_z, W_skip_u, u, N_HIDDEN, n_u);
        for (int j = 0; j < N_HIDDEN; ++j)
            z_vec[j] += tmp_z[j] + skip_const[i][j];

        // ReLU
        for (int j = 0; j < N_HIDDEN; ++j) {
            h[j] = std::max(z_vec[j], Scalar(0));
        }

        // B = dsigma .* (Wx @ B + W_skip_u)
        // Column-wise sparse GEMM + fused add + scale.
        // Only active rows of B contribute to Wx @ B (inactive rows are zero).
        // Processes one B column at a time: acc = Wx @ B[:,m], then
        // B[:,m] = dsigma .* (acc + W_skip_u[:,m]).
        for (int m = 0; m < n_u; ++m) {
            const Scalar* b_col_in = B_buf + m * N_HIDDEN;
            Scalar* b_col_out = B_buf + m * N_HIDDEN;
            const Scalar* w_skip_col = W_skip_u + m * N_HIDDEN;

            // Sparse GEMM: acc = Wx @ B[:,m] (only active rows of B)
            Scalar acc[N_HIDDEN];
            vec_zero(acc, N_HIDDEN);
            for (int a = 0; a < n_active; ++a) {
                int p = active_idx[a];
                Scalar bp = b_col_in[p];
                const Scalar* wx_col = w.Wx[i] + p * N_HIDDEN;
                for (int j = 0; j < N_HIDDEN; ++j)
                    acc[j] += bp * wx_col[j];
            }

            // Fused add W_skip_u + dsigma scaling
            for (int j = 0; j < N_HIDDEN; ++j)
                b_col_out[j] = (z_vec[j] > Scalar(0))
                    ? (acc[j] + w_skip_col[j])
                    : Scalar(0);
        }

        // Update active_idx for next layer
        n_active = 0;
        for (int j = 0; j < N_HIDDEN; ++j) {
            if (z_vec[j] > Scalar(0))
                active_idx[n_active++] = j;
        }
    }

    // ---- Output layer ----
    Scalar y = vec_dot(w.W_out, h, N_HIDDEN) + w.b_out;

    for (int m = 0; m < n_u; ++m)
        out_J[m] = vec_dot(w.W_out, B_buf + m * N_HIDDEN, N_HIDDEN);

    return y;
}

}  // namespace condensed
