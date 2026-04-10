#include "stable_neuron_solver/qp_builder.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cstring>

// Timer (same as solver.cpp)
#if defined(EMBEDDED_TARGET) && defined(__arm__)
extern "C" uint32_t get_microseconds();
#else
#include <chrono>
static uint32_t get_microseconds() {
    static auto start = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    return static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
}
#endif

namespace stable_neuron {

using Eigen::Map;
using Eigen::Matrix;

// Helper: access packed triangular Jacobian
// Packed format: [J1(1), J2(2), J3(3), J4(4), J5(5)] = 15 floats
// Step k has (k+1) elements starting at offset k*(k+1)/2
static inline const Scalar* packed_jacobian(const Scalar* packed, int step) {
    return packed + step * (step + 1) / 2;
}

void QPBuilder::configure(
    Scalar Q, Scalar R, Scalar R_delta,
    Scalar tube_weight, Scalar beta_0,
    Scalar u_min, Scalar u_max, Scalar delta_u_max,
    Scalar margin)
{
    Q_ = Q; R_ = R; R_delta_ = R_delta;
    tube_weight_ = tube_weight; beta_0_ = beta_0;
    u_min_ = u_min; u_max_ = u_max;
    delta_u_max_ = delta_u_max; margin_ = margin;
}

void QPBuilder::classify_and_setup(
    const Scalar* z_k,
    Scalar u_prev,
    const ModelWeights& weights)
{
    uint32_t t0 = get_microseconds();
    classifier_.classify(z_k, u_prev, u_min_, u_max_, delta_u_max_, margin_, weights);
    uint32_t t1 = get_microseconds();
    build_variable_layout();
    uint32_t t2 = get_microseconds();
    precompute_neuron_exprs(z_k, weights);
    uint32_t t3 = get_microseconds();
    precompute_static_constraints(z_k, u_prev, weights);
    uint32_t t4 = get_microseconds();

    classify_timing_.classify_us = static_cast<float>(t1 - t0);
    classify_timing_.layout_us = static_cast<float>(t2 - t1);
    classify_timing_.neuron_exprs_us = static_cast<float>(t3 - t2);
    classify_timing_.static_constraints_us = static_cast<float>(t4 - t3);
}

// =========================================================================
// Variable layout
// =========================================================================

void QPBuilder::build_variable_layout() {
    int offset = N_FIXED_VARS;  // After u, s_max, s_min, t

    // y_f variables: only if last layer has unstable neurons
    // Cap at MAX_Y_F_VARS to stay within total variable budget.
    for (int step = 0; step < N; ++step) {
        for (int net_idx = 0; net_idx < N_NETWORKS; ++net_idx) {
            int last_layer = N_LAYERS - 1;
            const auto& cls = classifier_.get(step, net_idx, last_layer);
            bool has_amb = false;
            for (int j = 0; j < N_HIDDEN; ++j) {
                if (cls.status[j] == static_cast<int8_t>(NeuronStatus::UNSTABLE)) {
                    has_amb = true;
                    break;
                }
            }
            if (has_amb && offset < N_FIXED_VARS + MAX_Y_F_VARS) {
                y_f_info_[step][net_idx] = {true, offset};
                offset++;
            } else {
                y_f_info_[step][net_idx] = {false, -1};
            }
        }
    }

    // unstable neuron variables.
    // Cap at MAX_UNSTABLE_VARS to ensure epigraph constraints fit within
    // MAX_STATIC_INEQ (each amb var generates 2 epigraph rows).
    // Neurons beyond the budget are treated as stable-active (conservative).
    int n_unstable_assigned = 0;
    int n_unstable_total = 0;  // Total unstable (before capping)
    std::memset(unstable_var_idx_, -1, sizeof(unstable_var_idx_));
    for (int step = 0; step < N; ++step) {
        for (int net_idx = 0; net_idx < N_NETWORKS; ++net_idx) {
            for (int layer = 0; layer < N_LAYERS; ++layer) {
                const auto& cls = classifier_.get(step, net_idx, layer);
                for (int j = 0; j < N_HIDDEN; ++j) {
                    if (cls.status[j] == static_cast<int8_t>(NeuronStatus::UNSTABLE)) {
                        n_unstable_total++;
                        if (n_unstable_assigned < MAX_UNSTABLE_VARS && offset < MAX_VARS) {
                            unstable_var_idx_[step][net_idx][layer][j] = offset;
                            offset++;
                            n_unstable_assigned++;
                        }
                        // else: treated as stable-active (unstable_var_idx_ stays -1)
                    }
                }
            }
        }
    }

    n_vars_ = offset;
    n_unstable_ = n_unstable_assigned;
    n_unstable_total_ = n_unstable_total;
}

// =========================================================================
// Neuron expression precomputation (ping-pong memory)
// =========================================================================

void QPBuilder::precompute_neuron_exprs(
    const Scalar* z_k,
    const ModelWeights& weights)
{
    for (int step = 0; step < N; ++step) {
        const int n_u = step + 1;
        const int nv = n_vars_;

        for (int net_idx = 0; net_idx < N_NETWORKS; ++net_idx) {
            const auto& nw = weights.nets[step][net_idx];

            int cur = 0;   // Current layer writes to buffer 0
            int prev = 1;  // Previous layer reads from buffer 1

            // === Layer 0 ===
            // W0 is column-major (N_HIDDEN × n_input). u columns start at N_STATE.
            const Scalar* W0_u = nw.W0 + N_STATE * N_HIDDEN;
            const Scalar* c0 = classifier_.get_c0(step, net_idx);
            const auto& cls0 = classifier_.get(step, net_idx, 0);

            // layer_coeffs_ is column-major: elem(row j, col c) = [c * N_HIDDEN + j]
            std::memset(layer_const_[cur], 0, sizeof(Scalar) * N_HIDDEN);
            std::memset(layer_coeffs_[cur], 0, sizeof(Scalar) * N_HIDDEN * nv);

            for (int j = 0; j < N_HIDDEN; ++j) {
                if (cls0.status[j] == static_cast<int8_t>(NeuronStatus::STABLE_ACTIVE)) {
                    layer_const_[cur][j] = c0[j];
                    for (int m = 0; m < n_u; ++m) {
                        layer_coeffs_[cur][m * N_HIDDEN + j] = W0_u[m * N_HIDDEN + j];
                    }
                } else if (cls0.status[j] == static_cast<int8_t>(NeuronStatus::UNSTABLE)) {
                    int vi = unstable_var_idx_[step][net_idx][0][j];
                    if (vi >= 0 && vi < nv) {
                        layer_coeffs_[cur][vi * N_HIDDEN + j] = Scalar(1);
                    }
                }
                // STABLE_INACTIVE: h = 0, already zero
            }

            // === Internal layers ===
            for (int layer = 0; layer < N_INTERNAL; ++layer) {
                std::swap(cur, prev);

                // Wx is column-major N_HIDDEN × N_HIDDEN
                const Scalar* Wx = nw.Wx[layer];
                // W_skip_u: u columns of W_skip (column-major, starting at col N_STATE)
                const Scalar* W_skip_u = nw.W_skip[layer] + N_STATE * N_HIDDEN;
                const Scalar* skip_c = classifier_.get_skip_const(step, net_idx, layer);
                const Scalar* prev_c = layer_const_[prev];
                const Scalar* prev_v = layer_coeffs_[prev];  // column-major
                const auto& prev_cls = classifier_.get(step, net_idx, layer);

                // Build pre-activation directly into layer_[cur], then fix up.
                // Initialize const = skip_const, coeffs = 0
                std::memcpy(layer_const_[cur], skip_c, sizeof(Scalar) * N_HIDDEN);
                std::memset(layer_coeffs_[cur], 0, sizeof(Scalar) * N_HIDDEN * nv);

                // Accumulate Wx * prev for STABLE_ACTIVE/unstable previous neurons
                for (int j = 0; j < N_HIDDEN; ++j) {
                    auto s = prev_cls.status[j];
                    if (s == static_cast<int8_t>(NeuronStatus::STABLE_ACTIVE)) {
                        // const += Wx.col(j) * prev_c[j]
                        const Scalar pj = prev_c[j];
                        const Scalar* Wx_j = Wx + j * N_HIDDEN;
                        for (int r = 0; r < N_HIDDEN; ++r) {
                            layer_const_[cur][r] += Wx_j[r] * pj;
                        }
                        // coeffs += Wx.col(j) * prev_v.row(j) (rank-1 outer product)
                        for (int c = 0; c < nv; ++c) {
                            Scalar vjc = prev_v[c * N_HIDDEN + j];
                            if (vjc == Scalar(0)) continue;
                            const Scalar* Wx_j2 = Wx + j * N_HIDDEN;
                            Scalar* dst = layer_coeffs_[cur] + c * N_HIDDEN;
                            for (int r = 0; r < N_HIDDEN; ++r) {
                                dst[r] += Wx_j2[r] * vjc;
                            }
                        }
                    } else if (s == static_cast<int8_t>(NeuronStatus::UNSTABLE)) {
                        int vi = unstable_var_idx_[step][net_idx][layer][j];
                        if (vi >= 0 && vi < nv) {
                            const Scalar* Wx_j = Wx + j * N_HIDDEN;
                            Scalar* dst = layer_coeffs_[cur] + vi * N_HIDDEN;
                            for (int r = 0; r < N_HIDDEN; ++r) {
                                dst[r] += Wx_j[r];
                            }
                        }
                    }
                    // STABLE_INACTIVE: zero row, skip
                }

                // Add skip connection u terms (columns 0..n_u-1)
                for (int m = 0; m < n_u; ++m) {
                    const Scalar* ws_m = W_skip_u + m * N_HIDDEN;
                    Scalar* dst = layer_coeffs_[cur] + m * N_HIDDEN;
                    for (int r = 0; r < N_HIDDEN; ++r) {
                        dst[r] += ws_m[r];
                    }
                }

                // Apply current-layer classification: keep STABLE_ACTIVE, fix others
                int curr_layer = layer + 1;
                const auto& cls = classifier_.get(step, net_idx, curr_layer);

                for (int j = 0; j < N_HIDDEN; ++j) {
                    if (cls.status[j] == static_cast<int8_t>(NeuronStatus::STABLE_ACTIVE)) {
                        // Keep pre-activation values as-is
                    } else if (cls.status[j] == static_cast<int8_t>(NeuronStatus::UNSTABLE)) {
                        int vi = unstable_var_idx_[step][net_idx][curr_layer][j];
                        layer_const_[cur][j] = Scalar(0);
                        for (int c = 0; c < nv; ++c) {
                            layer_coeffs_[cur][c * N_HIDDEN + j] = Scalar(0);
                        }
                        if (vi >= 0 && vi < nv) {
                            layer_coeffs_[cur][vi * N_HIDDEN + j] = Scalar(1);
                        }
                    } else {
                        // STABLE_INACTIVE: zero
                        layer_const_[cur][j] = Scalar(0);
                        for (int c = 0; c < nv; ++c) {
                            layer_coeffs_[cur][c * N_HIDDEN + j] = Scalar(0);
                        }
                    }
                }
            }

            // === Output expression ===
            // W_out is 1 × N_HIDDEN (N_HIDDEN contiguous values, column-major = sequential)
            const Scalar* W_out = nw.W_out;
            Scalar b_out = nw.b_out;
            const Scalar* last_c = layer_const_[cur];
            const Scalar* last_v = layer_coeffs_[cur];  // column-major

            // output_const = b_out + dot(W_out, last_c)
            Scalar dot = Scalar(0);
            for (int j = 0; j < N_HIDDEN; ++j) {
                dot += W_out[j] * last_c[j];
            }
            output_const_[step][net_idx] = b_out + dot;

            // output_coeffs = W_out @ last_v  (1 × nv)
            for (int c = 0; c < nv; ++c) {
                Scalar acc = Scalar(0);
                const Scalar* col = last_v + c * N_HIDDEN;
                for (int j = 0; j < N_HIDDEN; ++j) {
                    acc += W_out[j] * col[j];
                }
                output_coeffs_[step][net_idx][c] = acc;
            }
        }
    }
}

// =========================================================================
// Static constraints
// =========================================================================

void QPBuilder::precompute_static_constraints(
    const Scalar* z_k,
    Scalar u_prev,
    const ModelWeights& weights)
{
    const int nv = n_vars_;

    // Clear P and q
    std::memset(P_, 0, sizeof(Scalar) * nv * nv);
    std::memset(q_static_, 0, sizeof(Scalar) * nv);

    // Helper lambda for P access (row-major upper triangular)
    auto P = [&](int r, int c) -> Scalar& {
        return P_[r * nv + c];
    };

    // === Cost matrix P (upper triangular) ===

    // Q on t variables (index 3*N+i)
    for (int i = 0; i < N; ++i) {
        P(3 * N + i, 3 * N + i) = Scalar(2) * Q_;
    }

    // R on u variables (index i)
    for (int i = 0; i < N; ++i) {
        P(i, i) = Scalar(2) * R_;
    }

    // Tube weight: (s_max - s_min)^2
    for (int i = 0; i < N; ++i) {
        int si = N + i;      // s_max index
        int sj = 2 * N + i;  // s_min index
        P(si, si) += Scalar(2) * tube_weight_;
        P(sj, sj) += Scalar(2) * tube_weight_;
        // Cross term: upper triangle (si < sj always since si=N+i, sj=2N+i)
        P(si, sj) -= Scalar(2) * tube_weight_;
    }

    // R_delta: (u[i] - u[i-1])^2
    if (R_delta_ > 0) {
        for (int i = 1; i < N; ++i) {
            P(i, i) += Scalar(2) * R_delta_;
            P(i - 1, i - 1) += Scalar(2) * R_delta_;
            P(i - 1, i) -= Scalar(2) * R_delta_;  // upper triangle
        }
        // q: -2 * R_delta * u_prev for u[0]
        q_static_[0] -= Scalar(2) * R_delta_ * u_prev;
    }

    // === Equality constraints ===
    n_eq_ = 0;
    std::memset(A_eq_, 0, sizeof(Scalar) * MAX_EQ * nv);
    for (int step = 0; step < N; ++step) {
        for (int net_idx = 0; net_idx < N_NETWORKS; ++net_idx) {
            const auto& info = y_f_info_[step][net_idx];
            if (!info.is_var) continue;

            int y_idx = info.var_idx;
            Scalar out_c = output_const_[step][net_idx];
            const Scalar* out_v = output_coeffs_[step][net_idx];

            // y_f - out_v @ x = out_c  →  row: e_y - out_v, rhs: out_c
            Scalar* row = A_eq_ + n_eq_ * nv;
            for (int v = 0; v < nv; ++v) {
                row[v] = -out_v[v];
            }
            row[y_idx] = Scalar(1);  // Override: y_f coefficient
            b_eq_[n_eq_] = out_c;
            n_eq_++;
        }
    }

    // === Static inequality constraints (built directly into A_ineq_/h_hi_) ===
    n_static_ineq_ = 0;
    std::memset(A_ineq_, 0, sizeof(Scalar) * MAX_INEQ * nv);

    auto add_row = [&](const Scalar* coeffs, Scalar rhs) {
        if (n_static_ineq_ >= MAX_INEQ) return;
        std::memcpy(A_ineq_ + n_static_ineq_ * nv, coeffs, sizeof(Scalar) * nv);
        h_hi_[n_static_ineq_] = rhs;
        n_static_ineq_++;
    };

    Scalar row_buf[MAX_VARS];

    // u bounds: -u[i] <= -u_min, u[i] <= u_max
    for (int i = 0; i < N; ++i) {
        std::memset(row_buf, 0, sizeof(Scalar) * nv);
        row_buf[i] = Scalar(-1);
        add_row(row_buf, -u_min_);

        std::memset(row_buf, 0, sizeof(Scalar) * nv);
        row_buf[i] = Scalar(1);
        add_row(row_buf, u_max_);
    }

    // Rate constraints: |u[0] - u_prev| <= delta_u_max, |u[i] - u[i-1]| <= delta_u_max
    {
        std::memset(row_buf, 0, sizeof(Scalar) * nv);
        row_buf[0] = Scalar(1);
        add_row(row_buf, u_prev + delta_u_max_);

        std::memset(row_buf, 0, sizeof(Scalar) * nv);
        row_buf[0] = Scalar(-1);
        add_row(row_buf, -u_prev + delta_u_max_);
    }
    for (int i = 1; i < N; ++i) {
        std::memset(row_buf, 0, sizeof(Scalar) * nv);
        row_buf[i] = Scalar(1);
        row_buf[i - 1] = Scalar(-1);
        add_row(row_buf, delta_u_max_);

        std::memset(row_buf, 0, sizeof(Scalar) * nv);
        row_buf[i] = Scalar(-1);
        row_buf[i - 1] = Scalar(1);
        add_row(row_buf, delta_u_max_);
    }

    // Tube: s_max >= 0, s_min <= 0, s_min <= s_max
    for (int i = 0; i < N; ++i) {
        std::memset(row_buf, 0, sizeof(Scalar) * nv);
        row_buf[N + i] = Scalar(-1);  // -s_max <= 0
        add_row(row_buf, Scalar(0));

        std::memset(row_buf, 0, sizeof(Scalar) * nv);
        row_buf[2 * N + i] = Scalar(1);  // s_min <= 0
        add_row(row_buf, Scalar(0));
    }
    for (int i = 0; i < N; ++i) {
        std::memset(row_buf, 0, sizeof(Scalar) * nv);
        row_buf[2 * N + i] = Scalar(1);   // s_min
        row_buf[N + i] = Scalar(-1);       // -s_max  → s_min - s_max <= 0
        add_row(row_buf, Scalar(0));
    }

    // ICNN epigraph constraints
    Map<const VectorState> z(z_k);
    for (int step = 0; step < N; ++step) {
        int n_u = step + 1;
        for (int net_idx = 0; net_idx < N_NETWORKS; ++net_idx) {
            add_icnn_epigraph(step, net_idx, weights.nets[step][net_idx],
                              z_k, n_u, n_static_ineq_);
        }
    }

    n_dynamic_ineq_ = 4 * N;
}

void QPBuilder::add_icnn_epigraph(
    int step, int net_idx,
    const NetworkWeights& nw,
    const Scalar* z_k, int n_u,
    int& row_out)
{
    const int nv = n_vars_;
    Map<const VectorState> z(z_k);

    // Layer 0 epigraph
    {
        Map<const Matrix<Scalar, N_HIDDEN, Eigen::Dynamic>> W0(nw.W0, N_HIDDEN, nw.n_input);
        auto W0_u = W0.middleCols(N_STATE, n_u);
        // Use cached c0 from classifier
        Map<const VectorHidden> c0(classifier_.get_c0(step, net_idx));

        const auto& cls = classifier_.get(step, net_idx, 0);

        for (int j = 0; j < N_HIDDEN; ++j) {
            if (cls.status[j] != static_cast<int8_t>(NeuronStatus::UNSTABLE)) continue;
            int h_idx = unstable_var_idx_[step][net_idx][0][j];
            if (h_idx < 0) continue;

            if (row_out >= MAX_INEQ) return;

            // h >= W0_u[j,:] @ u + c0[j]  →  W0_u[j,:] @ u - h <= -c0[j]
            Scalar* row = A_ineq_ + row_out * nv;
            std::memset(row, 0, sizeof(Scalar) * nv);
            row[h_idx] = Scalar(-1);
            for (int m = 0; m < n_u; ++m) {
                row[m] += W0_u(j, m);
            }
            h_hi_[row_out] = -c0(j);
            row_out++;

            if (row_out >= MAX_INEQ) return;

            // h >= 0  →  -h <= 0
            row = A_ineq_ + row_out * nv;
            std::memset(row, 0, sizeof(Scalar) * nv);
            row[h_idx] = Scalar(-1);
            h_hi_[row_out] = Scalar(0);
            row_out++;
        }
    }

    // Internal layers epigraph
    for (int layer = 0; layer < N_INTERNAL; ++layer) {
        int curr_layer = layer + 1;

        // Early-exit: skip entire layer if no unstable neurons
        const auto& cls = classifier_.get(step, net_idx, curr_layer);
        bool has_amb = false;
        for (int j = 0; j < N_HIDDEN; ++j) {
            if (cls.status[j] == static_cast<int8_t>(NeuronStatus::UNSTABLE)) {
                has_amb = true;
                break;
            }
        }
        if (!has_amb) continue;

        Map<const Matrix<Scalar, N_HIDDEN, N_HIDDEN>> Wx(nw.Wx[layer]);
        Map<const Matrix<Scalar, N_HIDDEN, Eigen::Dynamic>> W_skip(
            nw.W_skip[layer], N_HIDDEN, nw.n_input);
        auto W_skip_u = W_skip.middleCols(N_STATE, n_u);

        // Use cached skip_const from classifier
        Map<const VectorHidden> skip_const(
            classifier_.get_skip_const(step, net_idx, layer));

        // Reconstruct previous layer const/coeffs from classification
        const auto& prev_cls = classifier_.get(step, net_idx, layer);

        Scalar prev_c_buf[N_HIDDEN];
        Scalar prev_v_buf[N_HIDDEN * MAX_VARS];
        std::memset(prev_c_buf, 0, sizeof(Scalar) * N_HIDDEN);
        std::memset(prev_v_buf, 0, sizeof(Scalar) * N_HIDDEN * nv);

        if (layer == 0) {
            // Previous is layer 0 — use cached c0
            Map<const Matrix<Scalar, N_HIDDEN, Eigen::Dynamic>> W0(nw.W0, N_HIDDEN, nw.n_input);
            auto W0_u_l = W0.middleCols(N_STATE, n_u);
            Map<const VectorHidden> c0_l(classifier_.get_c0(step, net_idx));

            for (int j = 0; j < N_HIDDEN; ++j) {
                if (prev_cls.status[j] == static_cast<int8_t>(NeuronStatus::STABLE_ACTIVE)) {
                    prev_c_buf[j] = c0_l(j);
                    for (int m = 0; m < n_u; ++m) {
                        prev_v_buf[m * N_HIDDEN + j] = W0_u_l(j, m);  // Column-major
                    }
                } else if (prev_cls.status[j] == static_cast<int8_t>(NeuronStatus::UNSTABLE)) {
                    int vi = unstable_var_idx_[step][net_idx][0][j];
                    if (vi >= 0 && vi < nv)
                        prev_v_buf[vi * N_HIDDEN + j] = Scalar(1);  // Column-major
                }
            }
        }

        Map<VectorHidden> pvc(prev_c_buf);
        Map<Matrix<Scalar, N_HIDDEN, Eigen::Dynamic>> pvv(prev_v_buf, N_HIDDEN, nv);

        VectorHidden pre_const = Wx * pvc + skip_const;
        Matrix<Scalar, N_HIDDEN, Eigen::Dynamic> pre_coeffs = Wx * pvv;

        for (int j = 0; j < N_HIDDEN; ++j) {
            if (cls.status[j] != static_cast<int8_t>(NeuronStatus::UNSTABLE)) continue;
            int h_idx = unstable_var_idx_[step][net_idx][curr_layer][j];
            if (h_idx < 0) continue;

            if (row_out >= MAX_INEQ) return;

            // h >= pre_activation  →  pre_coeffs[j,:] @ x + skip_u[j,:] @ u - h <= -pre_const[j]
            Scalar* row = A_ineq_ + row_out * nv;
            std::memset(row, 0, sizeof(Scalar) * nv);
            row[h_idx] = Scalar(-1);
            for (int v = 0; v < nv; ++v) {
                row[v] += pre_coeffs(j, v);
            }
            for (int m = 0; m < n_u; ++m) {
                row[m] += W_skip_u(j, m);
            }
            h_hi_[row_out] = -pre_const(j);
            row_out++;

            if (row_out >= MAX_INEQ) return;

            // h >= 0  →  -h <= 0
            row = A_ineq_ + row_out * nv;
            std::memset(row, 0, sizeof(Scalar) * nv);
            row[h_idx] = Scalar(-1);
            h_hi_[row_out] = Scalar(0);
            row_out++;
        }
    }
}

// =========================================================================
// Dynamic QP build (per SCP iteration)
// =========================================================================

QPData QPBuilder::build_qp(
    const Scalar* z_k,
    Scalar u_prev,
    const Scalar* u_nominal,
    const Scalar* y_nominal,
    const Scalar* jacobians_f1,
    const Scalar* jacobians_f2,
    const Scalar* f1_nominal,
    const Scalar* f2_nominal,
    const Scalar* W_bounds,
    const Scalar* linear_jacobians)
{
    const int nv = n_vars_;
    const int n_dyn = n_dynamic_ineq_;

    // Static rows are already in A_ineq_/h_hi_ from classify_and_setup().
    // Only append dynamic rows after n_static_ineq_.
    Scalar* A_dyn = A_ineq_ + n_static_ineq_ * nv;
    Scalar* h_dyn = h_hi_ + n_static_ineq_;

    std::memset(A_dyn, 0, sizeof(Scalar) * n_dyn * nv);

    int row = 0;

    // Tracking: s_max - t <= beta_0 - y_nominal
    for (int i = 0; i < N; ++i) {
        Scalar* r = A_dyn + row * nv;
        r[N + i] = Scalar(1);       // s_max
        r[3 * N + i] = Scalar(-1);  // -t
        h_dyn[row] = beta_0_ - y_nominal[i];
        row++;
    }

    // Tracking: -t <= 0
    for (int i = 0; i < N; ++i) {
        Scalar* r = A_dyn + row * nv;
        r[3 * N + i] = Scalar(-1);
        h_dyn[row] = Scalar(0);
        row++;
    }

    // DC upper bound: y_f1 - s_max - J_f2_eff @ u <= rhs
    for (int step = 0; step < N; ++step) {
        int n_u = step + 1;
        const Scalar* J_f2 = packed_jacobian(jacobians_f2, step);
        const Scalar* J_lin = linear_jacobians ? packed_jacobian(linear_jacobians, step) : nullptr;

        // J_f2_eff = J_f2 - J_linear (if linear_jacobians provided)
        Scalar J_f2_eff[N];
        for (int m = 0; m < n_u; ++m) {
            J_f2_eff[m] = J_f2[m] - (J_lin ? J_lin[m] : Scalar(0));
        }

        Scalar w_max = W_bounds[step * 2 + 1];
        Scalar rhs = f1_nominal[step];
        for (int m = 0; m < n_u; ++m) {
            rhs -= J_f2_eff[m] * u_nominal[m];
        }
        rhs -= w_max;

        Scalar* r = A_dyn + row * nv;
        r[N + step] = Scalar(-1);  // -s_max

        // y_f1 term
        const auto& f1_info = y_f_info_[step][0];
        if (f1_info.is_var) {
            r[f1_info.var_idx] = Scalar(1);
        } else {
            rhs -= output_const_[step][0];
            for (int v = 0; v < nv; ++v) {
                r[v] += output_coeffs_[step][0][v];
            }
        }

        // -J_f2_eff @ u
        for (int m = 0; m < n_u; ++m) {
            r[m] -= J_f2_eff[m];
        }

        h_dyn[row] = rhs;
        row++;
    }

    // DC lower bound: s_min + y_f2 - J_f1_eff @ u <= rhs
    for (int step = 0; step < N; ++step) {
        int n_u = step + 1;
        const Scalar* J_f1 = packed_jacobian(jacobians_f1, step);
        const Scalar* J_lin = linear_jacobians ? packed_jacobian(linear_jacobians, step) : nullptr;

        // J_f1_eff = J_f1 + J_linear
        Scalar J_f1_eff[N];
        for (int m = 0; m < n_u; ++m) {
            J_f1_eff[m] = J_f1[m] + (J_lin ? J_lin[m] : Scalar(0));
        }

        Scalar w_min = W_bounds[step * 2];
        Scalar rhs = f2_nominal[step];
        for (int m = 0; m < n_u; ++m) {
            rhs -= J_f1_eff[m] * u_nominal[m];
        }
        rhs += w_min;

        Scalar* r = A_dyn + row * nv;
        r[2 * N + step] = Scalar(1);  // s_min

        // y_f2 term
        const auto& f2_info = y_f_info_[step][1];
        if (f2_info.is_var) {
            r[f2_info.var_idx] = Scalar(1);
        } else {
            rhs -= output_const_[step][1];
            for (int v = 0; v < nv; ++v) {
                r[v] += output_coeffs_[step][1][v];
            }
        }

        // -J_f1_eff @ u
        for (int m = 0; m < n_u; ++m) {
            r[m] -= J_f1_eff[m];
        }

        h_dyn[row] = rhs;
        row++;
    }

    // Set h_lo to -inf
    int n_total = n_static_ineq_ + n_dyn;
    for (int i = 0; i < n_total; ++i) {
        h_lo_[i] = Scalar(-1e30);
    }

    // Copy q_static to q (could add dynamic q terms if needed)
    std::memcpy(q_, q_static_, sizeof(Scalar) * nv);

    QPData data;
    data.n_vars = nv;
    data.n_eq = n_eq_;
    data.n_ineq = n_total;
    data.P = P_;
    data.q = q_;
    data.A_eq = A_eq_;
    data.b_eq = b_eq_;
    data.A_ineq = A_ineq_;
    data.h_lo = h_lo_;
    data.h_hi = h_hi_;
    return data;
}

}  // namespace stable_neuron
