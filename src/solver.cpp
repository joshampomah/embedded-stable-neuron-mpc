#include "stable_neuron_solver/solver.hpp"

#ifdef USE_CUSTOM_IPM
#include "stable_neuron_solver/custom_ipm.hpp"
#else
// PIQP headers
#include "piqp/piqp.hpp"
#include <Eigen/Dense>
#endif

#include <cstring>
#include <cstdlib>
#include <cstdio>

// Bump allocator reset (embedded only — avoids heap fragmentation)
#if defined(EMBEDDED_TARGET) && defined(__arm__)
extern "C" void heap_reset_to_baseline();
extern "C" void heap_reset_overflow();
#endif

// Timer
#if defined(EMBEDDED_TARGET) && defined(__arm__)
// On STM32, use DWT cycle counter (provided by board.cpp)
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

#ifdef USE_CUSTOM_IPM

struct StableNeuronSolver::PiqpImpl {
    CustomIPM solver;
};

#else

using Eigen::Map;

struct StableNeuronSolver::PiqpImpl {
    piqp::DenseSolver<Scalar> solver;
};

#endif  // USE_CUSTOM_IPM

StableNeuronSolver::StableNeuronSolver()
    : piqp_(nullptr), last_nv_(-1), last_ne_(-1), last_ni_(-1),
      has_prev_x_(false), prev_nv_(0) {}

StableNeuronSolver::~StableNeuronSolver() {
#if defined(EMBEDDED_TARGET) && defined(__arm__)
    // Bump allocator: free is a no-op, no cleanup needed
#else
    delete piqp_;
#endif
}

void StableNeuronSolver::configure(
    Scalar Q, Scalar R, Scalar R_delta,
    Scalar tube_weight, Scalar beta_0,
    Scalar u_min, Scalar u_max, Scalar delta_u_max,
    Scalar margin)
{
    builder_.configure(Q, R, R_delta, tube_weight, beta_0,
                       u_min, u_max, delta_u_max, margin);
    // Force fresh solver setup on next solve (Q/R/constraints changed)
    last_nv_ = -1;
}

void StableNeuronSolver::classify_and_setup(
    const Scalar* z_k,
    Scalar u_prev,
    const ModelWeights& weights)
{
    builder_.classify_and_setup(z_k, u_prev, weights);
}

SolverResult StableNeuronSolver::solve(
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
    SolverResult result;
    uint32_t t0 = get_microseconds();

    // Build QP
    QPData qp = builder_.build_qp(
        z_k, u_prev, u_nominal, y_nominal,
        jacobians_f1, jacobians_f2, f1_nominal, f2_nominal,
        W_bounds, linear_jacobians
    );

    uint32_t t1 = get_microseconds();
    result.build_time_us = static_cast<float>(t1 - t0);

    const int nv = qp.n_vars;
    const int ne = qp.n_eq;
    const int ni = qp.n_ineq;

    // Diagnostic (desktop only — prints to stderr to avoid protocol corruption)
#ifndef EMBEDDED_TARGET
    static int solve_count_ = 0;
#endif

    // Safety check
    if (nv > MAX_VARS || nv <= 0 || ne < 0 || ne > MAX_EQ ||
        ni < 0 || ni + 2 * ne > QPBuilder::MAX_G_ROWS) {
        result.is_feasible = false;
        result.cost = Scalar(1e30);
        result.n_vars = nv;
        result.n_unstable = builder_.n_unstable();
        std::memset(result.u_optimal, 0, sizeof(Scalar) * N);
        std::memset(result.s_max_optimal, 0, sizeof(Scalar) * N);
        std::memset(result.s_min_optimal, 0, sizeof(Scalar) * N);
        result.total_time_us = static_cast<float>(get_microseconds() - t0);
        return result;
    }

    (void)last_nv_; (void)last_ne_; (void)last_ni_;  // unused: always cold-start

#ifdef USE_CUSTOM_IPM

    {
#if defined(EMBEDDED_TARGET) && defined(__arm__)
        heap_reset_to_baseline();
#else
        delete piqp_;
#endif
        piqp_ = new PiqpImpl();

        auto& s = piqp_->solver.settings();
#ifdef EMBEDDED_TARGET
        s.rho_init = Scalar(1e-4);
        s.delta_init = Scalar(1e-4);
        s.reg_lower_limit = Scalar(1e-6);
        s.eps_abs = Scalar(1e-4);
        s.max_iter = 50;
        s.iterative_refinement = false;
#else
        s.rho_init = Scalar(1e-8);
        s.delta_init = Scalar(1e-6);
        s.reg_lower_limit = Scalar(0);
        s.eps_abs = Scalar(1e-6);
        s.max_iter = 250;
        // Allow override for tolerance sweep testing
        if (const char* env = std::getenv("IPM_EPS_ABS")) {
            s.eps_abs = Scalar(std::atof(env));
        }
#endif
        piqp_->solver.setup(nv, ne, ni,
                            qp.P, qp.q, qp.A_eq, qp.b_eq, qp.A_ineq, qp.h_hi);

        // Warm-start x from previous MPC timestep if dimensions match.
        // The first N entries (u variables) transfer well between timesteps
        // since consecutive MPC problems are similar. Epigraph variables
        // (s_max, s_min, t, h_amb) may differ but the IPM recovers quickly.
        if (has_prev_x_ && prev_nv_ == nv) {
            piqp_->solver.warm_start_x(prev_x_);
        }

        last_nv_ = nv;
        last_ne_ = ne;
        last_ni_ = ni;
    }

    int status = piqp_->solver.solve();

    uint32_t t2 = get_microseconds();
    result.solve_time_us = static_cast<float>(t2 - t1);

    const auto& ipm_result = piqp_->solver.result();
    result.is_feasible = (status == 1 || status == -1);  // SOLVED or MAX_ITER
    result.n_vars = nv;
    result.n_eq = ne;
    result.n_ineq = ni;
    result.n_unstable = builder_.n_unstable();
    result.piqp_status = status;
    result.piqp_iters = ipm_result.iter;
    result.diag_q0 = qp.q[0];
    result.diag_P00 = qp.P[0];
    result.diag_G00 = qp.A_ineq[0];
    result.classify_time_us = 0;
    result.ipm_r_dual_0 = static_cast<float>(ipm_result.r_dual_0);
    result.ipm_r_eq_0 = static_cast<float>(ipm_result.r_eq_0);
    result.ipm_r_ineq_0 = static_cast<float>(ipm_result.r_ineq_0);
    result.ipm_mu_0 = static_cast<float>(ipm_result.mu_0);
    result.ipm_r_dual_final = static_cast<float>(ipm_result.r_dual_final);
    result.ipm_r_eq_final = static_cast<float>(ipm_result.r_eq_final);
    result.ipm_r_ineq_final = static_cast<float>(ipm_result.r_ineq_final);
    result.ipm_mu_final = static_cast<float>(ipm_result.mu_final);
    result.ipm_residuals_us = ipm_result.residuals_us;
    result.ipm_build_kkt_us = ipm_result.build_kkt_us;
    result.ipm_ldlt_factor_us = ipm_result.ldlt_factor_us;
    result.ipm_ldlt_solve_us = ipm_result.ldlt_solve_us;
    result.ipm_recover_us = ipm_result.recover_us;
    result.ipm_linesearch_us = ipm_result.linesearch_us;

    if (result.is_feasible) {
        const Scalar* x = piqp_->solver.x();
        for (int i = 0; i < N; ++i) {
            result.u_optimal[i] = x[i];
            result.s_max_optimal[i] = x[N + i];
            result.s_min_optimal[i] = x[2 * N + i];
        }
        // Recompute cost using original (unregularized) P from QPBuilder.
        // The custom IPM adds P_REG to zero-diagonal entries, which inflates
        // the reported primal_obj for epigraph variables.
        // qp.P is row-major upper triangular: 0.5*x^T*P*x + q^T*x
        Scalar cost = Scalar(0);
        for (int i = 0; i < nv; ++i) cost += qp.q[i] * x[i];
        for (int r = 0; r < nv; ++r) {
            cost += Scalar(0.5) * qp.P[r * nv + r] * x[r] * x[r];
            for (int c = r + 1; c < nv; ++c) {
                cost += qp.P[r * nv + c] * x[r] * x[c];
            }
        }
        result.cost = cost;

        // Save solution for warm-starting next MPC timestep
        std::memcpy(prev_x_, x, sizeof(Scalar) * nv);

#ifndef EMBEDDED_TARGET
        if (solve_count_++ < 3) {
            fprintf(stderr, "[DIAG] nv=%d ne=%d ni=%d n_unstable=%d/%d iters=%d status=%d\n",
                    nv, ne, ni, builder_.n_unstable(), builder_.n_unstable_total(),
                    ipm_result.iter, ipm_result.status);
            fprintf(stderr, "  u=[");
            for (int i = 0; i < N; ++i) fprintf(stderr, "%.6f%s", x[i], i<N-1?",":"");
            fprintf(stderr, "] cost=%.4f\n", cost);
            fprintf(stderr, "  r_dual=%.2e r_ineq=%.2e mu=%.2e\n",
                    ipm_result.r_dual_final, ipm_result.r_ineq_final, ipm_result.mu_final);
        }
#endif
        prev_nv_ = nv;
        has_prev_x_ = true;
    } else {
        std::memcpy(result.u_optimal, u_nominal, sizeof(Scalar) * N);
        std::memset(result.s_max_optimal, 0, sizeof(Scalar) * N);
        std::memset(result.s_min_optimal, 0, sizeof(Scalar) * N);
        result.cost = Scalar(1e30);
    }

#else  // PIQP path

    // --- Convert QP data: row-major → column-major + PD regularization ---
    // Uses QPBuilder's scratch buffers (available after build_qp() returns).

    Scalar* P_cm = builder_.scratch_P();   // capacity: N_HIDDEN * MAX_VARS >= nv*nv
    Scalar* G_cm = builder_.scratch_G();   // capacity: MAX_G_ROWS * MAX_VARS >= ni*nv
    Scalar* A_cm = builder_.scratch_Aeq(); // capacity: N_HIDDEN * MAX_VARS >= ne*nv

    // P: row-major upper triangle → column-major symmetric + PD regularization
    std::memset(P_cm, 0, sizeof(Scalar) * nv * nv);
    for (int c = 0; c < nv; ++c) {
        for (int r = 0; r <= c; ++r) {
            Scalar val = qp.P[r * nv + c];
            P_cm[c * nv + r] = val;
            P_cm[r * nv + c] = val;
        }
    }
    // Regularize PSD → PD (epigraph vars have zero P diagonal)
    constexpr Scalar P_REG = Scalar(0.01);
    constexpr Scalar P_ZERO = Scalar(1e-4);
    for (int i = 0; i < nv; ++i) {
        if (P_cm[i * nv + i] < P_ZERO) {
            P_cm[i * nv + i] = P_REG;
        }
    }

    // G: row-major → column-major (stride = ni)
    for (int c = 0; c < nv; ++c) {
        for (int r = 0; r < ni; ++r) {
            G_cm[c * ni + r] = qp.A_ineq[r * nv + c];
        }
    }

    // A: row-major → column-major (stride = ne)
    for (int c = 0; c < nv; ++c) {
        for (int r = 0; r < ne; ++r) {
            A_cm[c * ne + r] = qp.A_eq[r * nv + c];
        }
    }

    using Eigen::Map;

    // PIQP workaround: ne=0 (no equality constraints) triggers internal Eigen
    // assertion failure in PIQP's preconditioner. Add a trivial dummy row (0=0).
    int ne_piqp = ne;
    Scalar b_eq_dummy[1] = {Scalar(0)};
    const Scalar* b_eq_ptr = qp.b_eq;
    if (ne == 0) {
        ne_piqp = 1;
        std::memset(A_cm, 0, sizeof(Scalar) * nv);  // all-zero row
        b_eq_ptr = b_eq_dummy;
    }

    // Create Eigen Maps
    Map<MatrixX> P_map(P_cm, nv, nv);
    Map<VectorX> q_map(const_cast<Scalar*>(qp.q), nv);
    Map<MatrixX> A_map(A_cm, ne_piqp, nv);
    Map<VectorX> b_map(const_cast<Scalar*>(b_eq_ptr), ne_piqp);
    Map<MatrixX> G_map(G_cm, ni, nv);
    Map<VectorX> h_map(const_cast<Scalar*>(qp.h_hi), ni);

    // Always cold-start: delete and recreate PIQP solver each time.
    // Warm-start via update() accumulates numerical errors and crashes.
    // (Same finding as custom IPM — see MEMORY.md)
#if defined(EMBEDDED_TARGET) && defined(__arm__)
    heap_reset_to_baseline();
#else
    delete piqp_;
#endif
    piqp_ = new PiqpImpl();

    auto& s = piqp_->solver.settings();
    s.verbose = false;
#ifdef EMBEDDED_TARGET
    s.eps_abs = Scalar(1e-4);
    s.eps_rel = Scalar(1e-4);
    s.eps_duality_gap_abs = Scalar(1e-4);
    s.eps_duality_gap_rel = Scalar(1e-4);
    s.reg_lower_limit = Scalar(1e-4);
    s.reg_finetune_lower_limit = Scalar(1e-4);
    s.iterative_refinement_eps_abs = Scalar(1e-5);
    s.iterative_refinement_eps_rel = Scalar(1e-5);
    s.iterative_refinement_static_regularization_eps = Scalar(1e-4);
    s.iterative_refinement_static_regularization_rel = Scalar(1e-8);
    s.max_iter = 50;
    s.preconditioner_iter = 0;
#else
    s.eps_abs = Scalar(1e-6);
    s.eps_rel = Scalar(1e-6);
    s.max_iter = 500;
#endif
    piqp_->solver.setup(P_map, q_map, A_map, b_map, G_map, h_map);
    last_nv_ = nv;
    last_ne_ = ne;
    last_ni_ = ni;

#if defined(EMBEDDED_TARGET) && defined(__arm__)
    // Reclaim overflow before solve — transient temporaries from setup/update
    heap_reset_overflow();
#endif
    auto status = piqp_->solver.solve();

    uint32_t t2 = get_microseconds();
    result.solve_time_us = static_cast<float>(t2 - t1);

    result.is_feasible = (status == piqp::PIQP_SOLVED ||
                          status == piqp::PIQP_MAX_ITER_REACHED);
    result.n_vars = nv;
    result.n_eq = ne;
    result.n_ineq = ni;
    result.n_unstable = builder_.n_unstable();
    result.piqp_status = static_cast<int>(status);
    result.piqp_iters = static_cast<int>(piqp_->solver.result().info.iter);
    result.diag_q0 = qp.q[0];
    result.diag_P00 = qp.P[0];
    result.diag_G00 = qp.A_ineq[0];
    result.classify_time_us = 0;

    if (result.is_feasible) {
        const auto& x = piqp_->solver.result().x;
        for (int i = 0; i < N; ++i) {
            result.u_optimal[i] = x(i);
            result.s_max_optimal[i] = x(N + i);
            result.s_min_optimal[i] = x(2 * N + i);
        }
        // Recompute cost using original (unregularized) P from QPBuilder.
        Scalar cost = Scalar(0);
        for (int i = 0; i < nv; ++i) cost += qp.q[i] * x(i);
        for (int r = 0; r < nv; ++r) {
            cost += Scalar(0.5) * qp.P[r * nv + r] * x(r) * x(r);
            for (int c = r + 1; c < nv; ++c) {
                cost += qp.P[r * nv + c] * x(r) * x(c);
            }
        }
        result.cost = cost;
    } else {
        std::memcpy(result.u_optimal, u_nominal, sizeof(Scalar) * N);
        std::memset(result.s_max_optimal, 0, sizeof(Scalar) * N);
        std::memset(result.s_min_optimal, 0, sizeof(Scalar) * N);
        result.cost = Scalar(1e30);
    }

#endif  // USE_CUSTOM_IPM

    result.total_time_us = static_cast<float>(get_microseconds() - t0);
    return result;
}

}  // namespace stable_neuron
