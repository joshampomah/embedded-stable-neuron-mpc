#include "condensed_solver/scp_controller.hpp"
#include "condensed_solver/icnn_forward.hpp"
#include "condensed_solver/jacobian.hpp"

#include <algorithm>
#include <cmath>
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

namespace condensed {

void SCPController::configure(const SCPParams& params, const Scalar* W_bounds) {
    params_ = params;
    std::memcpy(W_bounds_, W_bounds, sizeof(Scalar) * N * 2);
    has_previous_ = false;

    qp_solver_.configure(
        params.Q, params.R, params.R_delta,
        params.tube_weight, params.beta_0,
        params.u_min, params.u_max, params.delta_u_max
    );
}

void SCPController::compute_predictions(const Scalar* z_k,
                                          const ModelWeights& weights) {
    const auto& cls = qp_solver_.classifier();
    for (int step = 0; step < N; ++step) {
        int n_u = step + 1;

        // Gather skip_const pointers for this network
        const Scalar* skip_f1[N_INTERNAL], *skip_f2[N_INTERNAL];
        for (int i = 0; i < N_INTERNAL; ++i) {
            skip_f1[i] = cls.get_skip_const(step, 0, i);
            skip_f2[i] = cls.get_skip_const(step, 1, i);
        }

        Scalar f1 = icnn_forward_cached(
            cls.get_c0(step, 0), skip_f1, u_nominal_, n_u, weights.nets[step][0]);
        Scalar f2 = icnn_forward_cached(
            cls.get_c0(step, 1), skip_f2, u_nominal_, n_u, weights.nets[step][1]);
        f1_nominal_[step] = f1;
        f2_nominal_[step] = f2;
        y_nominal_[step] = f1 - f2;
    }
}

void SCPController::compute_jacobians_and_predictions(
    const Scalar* z_k, const ModelWeights& weights)
{
    const auto& cls = qp_solver_.classifier();
    int offset = 0;
    for (int step = 0; step < N; ++step) {
        int n_u = step + 1;

        // Gather skip_const pointers for this network
        const Scalar* skip_f1[N_INTERNAL], *skip_f2[N_INTERNAL];
        for (int i = 0; i < N_INTERNAL; ++i) {
            skip_f1[i] = cls.get_skip_const(step, 0, i);
            skip_f2[i] = cls.get_skip_const(step, 1, i);
        }

        Scalar f1 = icnn_forward_and_jacobian_u_cached(
            cls.get_c0(step, 0), skip_f1,
            u_nominal_, n_u, weights.nets[step][0],
            jac_f1_packed_ + offset);

        Scalar f2 = icnn_forward_and_jacobian_u_cached(
            cls.get_c0(step, 1), skip_f2,
            u_nominal_, n_u, weights.nets[step][1],
            jac_f2_packed_ + offset);

        f1_nominal_[step] = f1;
        f2_nominal_[step] = f2;
        y_nominal_[step] = f1 - f2;

        offset += n_u;
    }
}

template<bool Profile>
SCPResult SCPController::solve(const Scalar* z_k, Scalar u_prev,
                                const ModelWeights& weights)
{
    SCPResult result;
    std::memset(&result, 0, sizeof(result));
    result.debug_validation_failed_idx = -1;

    [[maybe_unused]] uint32_t t_start = 0, t0 = 0, t1 = 0;
    [[maybe_unused]] float agg_forward_us = 0, agg_jacobian_us = 0, agg_qp_us = 0;

    if constexpr (Profile) {
        t_start = get_microseconds();
    }

    // ---- Warm-start ----
    if (has_previous_) {
        for (int i = 0; i < N - 1; ++i)
            u_nominal_[i] = prev_u_optimal_[i + 1];
        u_nominal_[N - 1] = prev_u_optimal_[N - 1];
    } else {
        Scalar u_init = std::max(params_.u_min, std::min(u_prev, params_.u_max));
        for (int i = 0; i < N; ++i)
            u_nominal_[i] = u_init;
    }

    // Clip to bounds
    for (int i = 0; i < N; ++i)
        u_nominal_[i] = std::max(params_.u_min,
                                  std::min(u_nominal_[i], params_.u_max));

    // ---- Classify neurons (once per timestep) ----
    if constexpr (Profile) { t0 = get_microseconds(); }
    qp_solver_.classify_and_setup(z_k, u_prev, weights);
    if constexpr (Profile) {
        t1 = get_microseconds();
        result.classify_time_us = static_cast<float>(t1 - t0);
        // Sub-breakdown from QPBuilder
        const auto& ct = qp_solver_.builder_classify_timing();
        result.classify_only_us = ct.classify_us;
        result.layout_us = ct.layout_us;
        result.neuron_exprs_us = ct.neuron_exprs_us;
        result.static_constraints_us = ct.static_constraints_us;
    }

    // ---- Initial predictions ----
    if constexpr (Profile) { t0 = get_microseconds(); }
    compute_predictions(z_k, weights);
    if constexpr (Profile) {
        t1 = get_microseconds();
        result.initial_forward_us = static_cast<float>(t1 - t0);
        agg_forward_us += result.initial_forward_us;
    }

    // ---- SCP loop ----
    Scalar J_prev = Scalar(1e30);
    int j = 0;
    bool converged = false;

    for (j = 1; j <= params_.maxiters; ++j) {
        [[maybe_unused]] uint32_t iter_start = 0;
        if constexpr (Profile) {
            iter_start = get_microseconds();
        }

        // Step 3: Compute Jacobians (also recomputes f1/f2/y)
        if constexpr (Profile) { t0 = get_microseconds(); }
        compute_jacobians_and_predictions(z_k, weights);
        if constexpr (Profile) {
            t1 = get_microseconds();
            result.iter_timing[j - 1].jacobian_us = static_cast<float>(t1 - t0);
            agg_jacobian_us += result.iter_timing[j - 1].jacobian_us;
        }

        // Step 4: Solve QP
        SolverResult qp_result = qp_solver_.solve(
            z_k, u_prev,
            u_nominal_, y_nominal_,
            jac_f1_packed_, jac_f2_packed_,
            f1_nominal_, f2_nominal_,
            W_bounds_, nullptr
        );

        if constexpr (Profile) {
            result.iter_timing[j - 1].qp_build_us = qp_result.build_time_us;
            result.iter_timing[j - 1].qp_solve_us = qp_result.solve_time_us;
            agg_qp_us += qp_result.build_time_us + qp_result.solve_time_us;
        }

        // Store last QP diagnostics
        result.last_piqp_status = qp_result.piqp_status;
        result.last_piqp_iters = qp_result.piqp_iters;
        result.last_n_vars = qp_result.n_vars;
        result.last_n_eq = qp_result.n_eq;
        result.last_n_ineq = qp_result.n_ineq;
        result.diag_q0 = qp_result.diag_q0;
        result.diag_P00 = qp_result.diag_P00;
        result.diag_G00 = qp_result.diag_G00;
        result.ipm_r_dual_0 = qp_result.ipm_r_dual_0;
        result.ipm_r_eq_0 = qp_result.ipm_r_eq_0;
        result.ipm_r_ineq_0 = qp_result.ipm_r_ineq_0;
        result.ipm_mu_0 = qp_result.ipm_mu_0;
        result.ipm_r_dual_final = qp_result.ipm_r_dual_final;
        result.ipm_r_eq_final = qp_result.ipm_r_eq_final;
        result.ipm_r_ineq_final = qp_result.ipm_r_ineq_final;
        result.ipm_mu_final = qp_result.ipm_mu_final;
        result.ipm_residuals_us = qp_result.ipm_residuals_us;
        result.ipm_build_kkt_us = qp_result.ipm_build_kkt_us;
        result.ipm_ldlt_factor_us = qp_result.ipm_ldlt_factor_us;
        result.ipm_ldlt_solve_us = qp_result.ipm_ldlt_solve_us;
        result.ipm_recover_us = qp_result.ipm_recover_us;
        result.ipm_linesearch_us = qp_result.ipm_linesearch_us;

        if (!qp_result.is_feasible) {
            if constexpr (Profile) {
                result.iter_timing[j - 1].dcnn_forward_us = 0;
                result.iter_timing[j - 1].total_us =
                    static_cast<float>(get_microseconds() - iter_start);
            }
            result.cost = Scalar(1e30);
            result.n_iterations = j;
            result.converged = false;
            Scalar u_safe = std::max(params_.u_min,
                                      std::min(u_prev, params_.u_max));
            for (int i = 0; i < N; ++i)
                result.u_optimal[i] = u_safe;
            break;
        }

        // Validate QP solution: reject garbage (NaN or out-of-bounds)
        {
            // Store raw QP u for debugging (first SCP iteration only)
            if (j == 1) {
                std::memcpy(result.debug_qp_u, qp_result.u_optimal, sizeof(Scalar) * N);
            }

            bool valid = true;
            constexpr Scalar BOUND_TOL = Scalar(1e-3);
            for (int i = 0; i < N; ++i) {
                Scalar u_i = qp_result.u_optimal[i];
                // NaN check (NaN != NaN) must come first since NaN
                // comparisons always return false
                if (u_i != u_i ||
                    u_i < params_.u_min - BOUND_TOL ||
                    u_i > params_.u_max + BOUND_TOL) {
                    valid = false;
                    if (result.debug_validation_failed_idx < 0)
                        result.debug_validation_failed_idx = i;
                    break;
                }
            }
            if (!valid) {
                // Use previous nominal as final answer
                if constexpr (Profile) {
                    result.iter_timing[j - 1].dcnn_forward_us = 0;
                    result.iter_timing[j - 1].total_us =
                        static_cast<float>(get_microseconds() - iter_start);
                }
                std::memcpy(result.u_optimal, u_nominal_, sizeof(Scalar) * N);
                result.cost = J_prev;
                result.n_iterations = j;
                result.converged = false;
                break;
            }
        }

        // Step 5: Convergence checks
        Scalar J_j = qp_result.cost;

        // (a) Iteration 1 only: adaptive linearization check |u_opt - u_nom|_inf.
        // delta_J can never fire on iteration 1 (J_prev=inf), so we use delta_u
        // instead. If the QP barely moved u from the linearization point, the
        // Jacobians are still accurate and re-linearization is unnecessary.
        // On iteration 2+, delta_J works correctly and is used below.
        if (j == 1 && params_.delta_u_tol > Scalar(0)) {
            Scalar max_delta_u = Scalar(0);
            for (int i = 0; i < N; ++i) {
                Scalar d = std::abs(qp_result.u_optimal[i] - u_nominal_[i]);
                if (d > max_delta_u) max_delta_u = d;
            }
            if (max_delta_u <= params_.delta_u_tol) {
                if constexpr (Profile) {
                    result.iter_timing[j - 1].dcnn_forward_us = 0;
                    result.iter_timing[j - 1].total_us =
                        static_cast<float>(get_microseconds() - iter_start);
                }
                std::memcpy(result.u_optimal, qp_result.u_optimal, sizeof(Scalar) * N);
                result.cost = J_j;
                result.n_iterations = j;
                result.converged = true;
                converged = true;
                break;
            }
        }

        // (b) Cost convergence check (original delta_J criterion)
        Scalar delta_J;
        if (J_prev >= Scalar(1e29)) {
            delta_J = Scalar(1e30);  // First iteration: no previous cost
        } else {
            delta_J = std::abs(J_j - J_prev);
        }

        // Step 6: Update nominal from QP solution
        std::memcpy(u_nominal_, qp_result.u_optimal, sizeof(Scalar) * N);

        if (delta_J <= params_.delta_J_min) {
            if constexpr (Profile) {
                result.iter_timing[j - 1].dcnn_forward_us = 0;
                result.iter_timing[j - 1].total_us =
                    static_cast<float>(get_microseconds() - iter_start);
            }
            std::memcpy(result.u_optimal, qp_result.u_optimal, sizeof(Scalar) * N);
            result.cost = J_j;
            result.n_iterations = j;
            result.converged = true;
            converged = true;
            break;
        }

        J_prev = J_j;

        // Step 6 cont: Recompute predictions with updated nominal
        if constexpr (Profile) { t0 = get_microseconds(); }
        compute_predictions(z_k, weights);
        if constexpr (Profile) {
            t1 = get_microseconds();
            result.iter_timing[j - 1].dcnn_forward_us = static_cast<float>(t1 - t0);
            agg_forward_us += result.iter_timing[j - 1].dcnn_forward_us;
            result.iter_timing[j - 1].total_us =
                static_cast<float>(get_microseconds() - iter_start);
        }
    }

    // If loop exhausted without converging or breaking
    if (!converged && result.cost < Scalar(1e29)) {
        std::memcpy(result.u_optimal, u_nominal_, sizeof(Scalar) * N);
        result.cost = J_prev;
        result.n_iterations = params_.maxiters;
        result.converged = false;
    }

    // Store for warm-start
    std::memcpy(prev_u_optimal_, result.u_optimal, sizeof(Scalar) * N);
    has_previous_ = true;

    if constexpr (Profile) {
        uint32_t t_end = get_microseconds();
        result.forward_time_us = agg_forward_us;
        result.jacobian_time_us = agg_jacobian_us;
        result.qp_time_us = agg_qp_us;
        result.total_time_us = static_cast<float>(t_end - t_start);
    }

    return result;
}

// Explicit instantiations
template SCPResult SCPController::solve<false>(const Scalar*, Scalar, const ModelWeights&);
template SCPResult SCPController::solve<true>(const Scalar*, Scalar, const ModelWeights&);

}  // namespace condensed
