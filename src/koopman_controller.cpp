#include "stable_neuron_solver/koopman_controller.hpp"
#include "stable_neuron_solver/math_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

// Timer (same pattern as scp_controller.cpp)
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

// Use custom IPM directly (same solver as DCNN, reusable for any dense QP)
#ifdef USE_CUSTOM_IPM
#include "stable_neuron_solver/custom_ipm.hpp"
#else
#include <piqp/piqp.hpp>
#endif

namespace stable_neuron {

KoopmanController::KoopmanController() : has_previous_(false) {
    std::memset(prev_u_optimal_, 0, sizeof(prev_u_optimal_));
    std::memset(P_, 0, sizeof(P_));
    std::memset(q_, 0, sizeof(q_));
    std::memset(G_, 0, sizeof(G_));
    std::memset(h_lo_, 0, sizeof(h_lo_));
    std::memset(h_hi_, 0, sizeof(h_hi_));
}

void KoopmanController::configure(const KoopmanParams& params) {
    params_ = params;
    build_fixed_qp();
}

void KoopmanController::reset() {
    has_previous_ = false;
    std::memset(prev_u_optimal_, 0, sizeof(prev_u_optimal_));
}

void KoopmanController::build_fixed_qp() {
    // P: diagonal cost matrix (2N x 2N)
    // P = diag(2R, ..., 2R, 2Q, ..., 2Q)
    std::memset(P_, 0, sizeof(P_));
    for (int i = 0; i < KN; ++i) {
        P_[i * KN_VARS + i] = KScalar(2) * params_.R;         // u block
    }
    for (int i = 0; i < KN; ++i) {
        int idx = KN + i;
        P_[idx * KN_VARS + idx] = KScalar(2) * params_.Q;     // t block
    }

    // q: zero (no linear cost term)
    std::memset(q_, 0, sizeof(q_));

    // G: constraint matrix (6N x 2N), row-major layout for custom_ipm
    // Row layout:
    //   [0, N):     tracking:  J[i,:] @ u - t_i <= rhs  (J rows updated per step)
    //   [N, 2N):    nonneg:    -t_i <= 0
    //   [2N, 3N):   box_lo:    -u_i <= -u_min
    //   [3N, 4N):   box_hi:     u_i <= u_max
    //   [4N, 5N):   rate_up:    u_i - u_{i-1} <= delta_u_max
    //   [5N, 6N):   rate_dn:   -u_i + u_{i-1} <= delta_u_max
    std::memset(G_, 0, sizeof(G_));

    for (int i = 0; i < KN_INEQ; ++i) {
        h_lo_[i] = KScalar(-1e30);
    }
    std::memset(h_hi_, 0, sizeof(h_hi_));

    // Nonneg: -t_i <= 0
    for (int i = 0; i < KN; ++i) {
        G_[(KN + i) * KN_VARS + (KN + i)] = KScalar(-1);
        h_hi_[KN + i] = KScalar(0);
    }

    // Box lower: -u_i <= -u_min
    for (int i = 0; i < KN; ++i) {
        G_[(2 * KN + i) * KN_VARS + i] = KScalar(-1);
        h_hi_[2 * KN + i] = -params_.u_min;
    }

    // Box upper: u_i <= u_max
    for (int i = 0; i < KN; ++i) {
        G_[(3 * KN + i) * KN_VARS + i] = KScalar(1);
        h_hi_[3 * KN + i] = params_.u_max;
    }

    // Rate up: u_i - u_{i-1} <= delta_u_max
    for (int i = 0; i < KN; ++i) {
        G_[(4 * KN + i) * KN_VARS + i] = KScalar(1);
        if (i > 0) {
            G_[(4 * KN + i) * KN_VARS + (i - 1)] = KScalar(-1);
        }
        h_hi_[4 * KN + i] = params_.delta_u_max;
    }

    // Rate down: -u_i + u_{i-1} <= delta_u_max
    for (int i = 0; i < KN; ++i) {
        G_[(5 * KN + i) * KN_VARS + i] = KScalar(-1);
        if (i > 0) {
            G_[(5 * KN + i) * KN_VARS + (i - 1)] = KScalar(1);
        }
        h_hi_[5 * KN + i] = params_.delta_u_max;
    }

    // Tracking rows: G[0:N] has J columns (u block) and -1 for t_i
    // The J part is updated per step; the -t_i part is fixed:
    for (int i = 0; i < KN; ++i) {
        G_[i * KN_VARS + (KN + i)] = KScalar(-1);  // -t_i coefficient
    }
}

void KoopmanController::update_qp(const KScalar* y_nom, const KScalar* F_mat,
                                    const KScalar* u_nom, KScalar u_prev,
                                    const KScalar* s_max) {
    // Update tracking rows (first KN rows of G): J = F (lower-triangular)
    // F is column-major (KN x KN), G is row-major (KN_INEQ x KN_VARS)
    for (int i = 0; i < KN; ++i) {
        // Clear u columns of tracking row i
        for (int j = 0; j < KN; ++j) {
            G_[i * KN_VARS + j] = F_mat[j * KN + i];  // F col-major -> G row-major
        }
        // Note: G[i, KN+i] = -1 is already set in build_fixed_qp
    }

    // Update tracking RHS: J @ u_nom - y_nom - s_max + beta_0
    for (int i = 0; i < KN; ++i) {
        KScalar Ju = 0;
        for (int j = 0; j <= i; ++j) {
            Ju += F_mat[j * KN + i] * u_nom[j];  // F is lower-tri, col-major
        }
        h_hi_[i] = Ju - y_nom[i] - s_max[i] + params_.beta_0;
    }

    // Update rate constraints for u_prev (step 0)
    h_hi_[4 * KN] = params_.delta_u_max + u_prev;
    h_hi_[5 * KN] = params_.delta_u_max - u_prev;
}

template<bool Profile>
KoopmanResult KoopmanController::solve(const KScalar* z_k, KScalar u_prev,
                                        const KoopmanWeights& weights) {
    KoopmanResult result{};
    uint32_t t0 = 0, t1 = 0;

    // --- 1. Encode: psi = encoder(z_k) ---
    if (Profile) t0 = get_microseconds();
    encoder_.encode(z_k, psi_, weights);
    if (Profile) {
        t1 = get_microseconds();
        result.encode_time_us = static_cast<float>(t1 - t0);
    }

    // --- 2. Predict: e = CA @ psi ---
    if (Profile) t0 = get_microseconds();
    // CA: (KN x KD_LIFT), column-major. e = CA @ psi.
    math::matvec(e_, weights.CA, psi_, KN, KD_LIFT);
    if (Profile) {
        t1 = get_microseconds();
        result.predict_time_us = static_cast<float>(t1 - t0);
    }

    // --- 3. Warm-start u_nominal ---
    if (has_previous_) {
        // Shift previous solution left by 1, repeat last
        for (int i = 0; i < KN - 1; ++i) {
            u_nominal_[i] = prev_u_optimal_[i + 1];
        }
        u_nominal_[KN - 1] = prev_u_optimal_[KN - 1];
        // Clip to bounds
        for (int i = 0; i < KN; ++i) {
            u_nominal_[i] = std::max(params_.u_min, std::min(params_.u_max, u_nominal_[i]));
        }
    } else {
        for (int i = 0; i < KN; ++i) {
            u_nominal_[i] = std::max(params_.u_min, std::min(params_.u_max, u_prev));
        }
    }

    // --- 4. Compute y_nominal = e + F @ u_nominal ---
    math::matvec(y_nominal_, weights.F, u_nominal_, KN, KN);
    for (int i = 0; i < KN; ++i) {
        y_nominal_[i] += e_[i];
    }

    // --- 5. Build QP ---
    if (Profile) t0 = get_microseconds();
    update_qp(y_nominal_, weights.F, u_nominal_, u_prev, weights.s_max);
    if (Profile) {
        t1 = get_microseconds();
        result.qp_build_time_us = static_cast<float>(t1 - t0);
    }

    // --- 6. Solve QP ---
    if (Profile) t0 = get_microseconds();

#ifdef USE_CUSTOM_IPM
    // Use the custom IPM — same solver as DCNN, works on any dense QP.
    // G_ is row-major, which is what custom_ipm expects (P_rm, A_rm, G_rm).
    static CustomIPM ipm;
    static bool ipm_configured = false;

    CustomIPM::Settings& s = ipm.settings();
    s.eps_abs = KScalar(1e-4);
    s.max_iter = 50;
    s.iterative_refinement = true;
    s.ruiz_iter = 10;
    s.P_REG = KScalar(0.1);      // Stronger PSD regularization for float32 stability

    // No equality constraints
    static KScalar A_empty[1] = {0};
    static KScalar b_empty[1] = {0};

    // Always cold-start: the G matrix changes structure each step
    // (tracking rows depend on state), and update() warm-start
    // can diverge when the problem changes significantly.
    ipm.setup(KN_VARS, 0, KN_INEQ,
              P_, q_, A_empty, b_empty, G_, h_hi_);
    ipm_configured = true;

    // Warm-start from u_nominal with meaningful t values
    KScalar x0[KN_VARS];
    for (int i = 0; i < KN; ++i) x0[i] = u_nominal_[i];
    // t_i >= max(0, y_nom_i + s_max_i - beta_0), give slack
    for (int i = 0; i < KN; ++i) {
        KScalar violation = y_nominal_[i] + weights.s_max[i] - params_.beta_0;
        x0[KN + i] = (violation > KScalar(0)) ? violation + KScalar(0.01) : KScalar(0.01);
    }
    ipm.warm_start_x(x0);

    int status = ipm.solve();
    const KScalar* x_sol = ipm.x();
    const auto& ipm_result = ipm.result();

    result.is_feasible = (status == 1);
    result.qp_status = status;
    result.qp_iters = ipm_result.iter;
    result.ipm_r_dual_final = static_cast<float>(ipm_result.r_dual_final);
    result.ipm_mu_final = static_cast<float>(ipm_result.mu_final);

    if (result.is_feasible) {
        for (int i = 0; i < KN; ++i) {
            result.u_optimal[i] = x_sol[i];
        }
        result.cost = static_cast<KScalar>(ipm_result.primal_obj);
    } else {
        // Fallback to nominal
        std::memcpy(result.u_optimal, u_nominal_, KN * sizeof(KScalar));
        result.cost = KScalar(1e30);
    }

#else
    // PIQP dense solver
    using PiqpScalar = KScalar;
    using DenseMat = Eigen::Matrix<PiqpScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using DenseVec = Eigen::Matrix<PiqpScalar, Eigen::Dynamic, 1>;

    // Map our row-major arrays into Eigen matrices
    // P is symmetric so row/col major doesn't matter
    Eigen::Map<const DenseMat> P_map(P_, KN_VARS, KN_VARS);
    Eigen::Map<const DenseVec> q_map(q_, KN_VARS);

    // G needs to be transposed from row-major to column-major for Eigen
    DenseMat G_cm(KN_INEQ, KN_VARS);
    for (int i = 0; i < KN_INEQ; ++i)
        for (int j = 0; j < KN_VARS; ++j)
            G_cm(i, j) = G_[i * KN_VARS + j];

    Eigen::Map<const DenseVec> h_lo_map(h_lo_, KN_INEQ);
    Eigen::Map<const DenseVec> h_hi_map(h_hi_, KN_INEQ);

    DenseMat A_eq(0, KN_VARS);
    DenseVec b_eq(0);

    static piqp::DenseSolver<PiqpScalar> solver;
    static bool solver_configured = false;

    solver.settings().verbose = false;
    solver.settings().eps_abs = PiqpScalar(1e-6);
    solver.settings().eps_rel = PiqpScalar(1e-6);
    solver.settings().max_iter = 200;

    if (!solver_configured) {
        solver.setup(P_map, q_map, A_eq, b_eq, G_cm, h_lo_map, h_hi_map);
        solver_configured = false;  // always update since G changes
    }
    solver.setup(P_map, q_map, A_eq, b_eq, G_cm, h_lo_map, h_hi_map);

    auto piqp_status = solver.solve();
    result.is_feasible = (piqp_status == piqp::PIQP_SOLVED);
    result.qp_status = static_cast<int>(piqp_status);
    result.qp_iters = static_cast<int>(solver.result().info.iter);

    if (result.is_feasible) {
        const auto& x_sol = solver.result().x;
        for (int i = 0; i < KN; ++i) {
            result.u_optimal[i] = x_sol[i];
        }
        result.cost = static_cast<KScalar>(solver.result().info.primal_obj);
    } else {
        std::memcpy(result.u_optimal, u_nominal_, KN * sizeof(KScalar));
        result.cost = KScalar(1e30);
    }
#endif

    if (Profile) {
        t1 = get_microseconds();
        result.qp_solve_time_us = static_cast<float>(t1 - t0);
        result.total_time_us = result.encode_time_us + result.predict_time_us
                              + result.qp_build_time_us + result.qp_solve_time_us;
    }

    // Update warm-start state
    std::memcpy(prev_u_optimal_, result.u_optimal, KN * sizeof(KScalar));
    has_previous_ = true;

    return result;
}

// Explicit instantiations
template KoopmanResult KoopmanController::solve<false>(const KScalar*, KScalar, const KoopmanWeights&);
template KoopmanResult KoopmanController::solve<true>(const KScalar*, KScalar, const KoopmanWeights&);

}  // namespace stable_neuron
