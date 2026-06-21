#pragma once

// Custom Mehrotra predictor-corrector IPM for small dense QPs.
// No Eigen dependency — all math is plain C-style loops on static arrays.
//
// Problem form:
//   min  0.5 x'Px + q'x
//   s.t. Ax = b       (p <= MAX_EQ equalities, via PMM penalty)
//        Gx + s = h   (m <= MAX_INEQ inequalities, s >= 0)

#include "stable_neuron_solver/config.hpp"
#include "stable_neuron_solver/types.hpp"  // Scalar

namespace stable_neuron {

// Max inequalities for IPM: original + equality pairs from conversion
static constexpr int MAX_INEQ_IPM = MAX_INEQ + 2 * MAX_EQ;  // 93

class CustomIPM {
public:
    struct Settings {
        Scalar eps_abs = Scalar(1e-6);
        int max_iter = 500;
        Scalar tau = Scalar(0.99);      // fraction-to-boundary
        Scalar rho_init = Scalar(1e-6); // initial primal regularization (PMM)
        Scalar delta_init = Scalar(1e-4); // initial dual regularization (PMM)
        Scalar reg_lower_limit = Scalar(1e-10); // floor for rho/delta decay
        Scalar P_REG = Scalar(0.01);    // PSD→PD regularization
        Scalar P_ZERO = Scalar(1e-4);   // threshold for "zero" diagonal
        bool iterative_refinement = true;  // extra solve for float32 accuracy
        int ruiz_iter = 0;                // Ruiz equilibration iterations (0 to disable)
    };

    struct Result {
        int status;       // 1=SOLVED, -1=MAX_ITER, -8=NUMERICS
        int iter;
        Scalar primal_obj;
        // Solver diagnostics
        Scalar r_dual_0;     // dual residual norm (iteration 0)
        Scalar r_eq_0;       // eq residual norm (iteration 0)
        Scalar r_ineq_0;     // ineq residual norm (iteration 0)
        Scalar mu_0;         // initial complementarity
        Scalar r_dual_final; // dual residual norm (final)
        Scalar r_eq_final;   // eq residual norm (final)
        Scalar r_ineq_final; // ineq residual norm (final)
        Scalar mu_final;     // final complementarity
        // Sub-iteration timing (accumulated microseconds across all iterations)
        uint32_t residuals_us;
        uint32_t build_kkt_us;
        uint32_t ldlt_factor_us;
        uint32_t ldlt_solve_us;    // predictor + corrector solves
        uint32_t recover_us;       // RHS builds + G*dx recovery + ds/dz
        uint32_t linesearch_us;    // step sizes + updates + delta scheduling
    };

    CustomIPM();

    Settings& settings() { return settings_; }

    // Setup (cold start): stores problem data, initializes x/s/y/z.
    // P_rm, A_rm, G_rm are row-major from QPBuilder.
    // P_rm is upper-triangular row-major; symmetrized + regularized internally.
    void setup(int n, int p, int m,
               const Scalar* P_rm, const Scalar* q,
               const Scalar* A_rm, const Scalar* b,
               const Scalar* G_rm, const Scalar* h);

    // Update (warm start): updates problem data, keeps x/s/y/z.
    void update(const Scalar* P_rm, const Scalar* q,
                const Scalar* A_rm, const Scalar* b,
                const Scalar* G_rm, const Scalar* h);

    // Warm-start from a previous solution's primal variables.
    // Recomputes slacks s = h - G*x0 (shifted to be positive) and sets
    // z = mu_target / s for consistent complementarity. Must be called
    // after setup() and before solve().
    void warm_start_x(const Scalar* x0);

    int solve();  // returns status code

    const Scalar* x() const { return x_; }
    const Result& result() const { return result_; }

private:
    // Convert row-major to column-major, symmetrize P, apply P_REG.
    // Equalities (p_orig, A_rm, b) converted to paired inequalities internally.
    void store_problem(int p_orig, int m_orig,
                       const Scalar* P_rm, const Scalar* q,
                       const Scalar* A_rm, const Scalar* b,
                       const Scalar* G_rm, const Scalar* h);

    // Mehrotra starting point (cold start)
    void initialize();

    // Shared: compute s from x, shift to positive, set z=1, y=0, rho/delta
    void init_slacks_and_duals();

    // Compute residuals and mu.
    // Returns non-regularized residual norms (for convergence check),
    // then adds proximal terms in-place (for KKT solve).
    void compute_residuals(Scalar& dual_inf_nr, Scalar& ineq_inf_nr);

    // Build KKT matrix: K = P + rho*I + G'*diag(w)*G + (1/delta)*A'A
    void build_kkt();

    // In-place LDLT factorization of K_ (no pivoting)
    bool ldlt_factor();

    // Solve K*dx = rhs using factored LDLT
    void ldlt_solve(const Scalar* rhs, Scalar* dx);

    // Compute K*v from stored components (P, G, w, A, rho) — NOT from factored K_
    // Used for iterative refinement residual computation.
    void kkt_matvec(const Scalar* v, Scalar* out);

    // Solve K*dx = rhs with one step of iterative refinement.
    // Improves accuracy from O(eps*cond) to O(eps^2*cond).
    void ldlt_solve_refined(const Scalar* rhs, Scalar* dx);

    // Fraction-to-boundary step size for z and s
    Scalar step_size(const Scalar* v, const Scalar* dv, int len);

    Settings settings_;
    Result result_;

    int n_, p_, m_;          // vars, equalities, inequalities (after conversion)
    int p_orig_, m_orig_;    // original counts (before eq→ineq conversion)
    bool initialized_;

    // Problem data (column-major for P, G, A)
    Scalar P_[MAX_VARS * MAX_VARS];
    Scalar q_[MAX_VARS];
    Scalar A_[MAX_EQ * MAX_VARS];      // column-major
    Scalar b_[MAX_EQ];
    Scalar G_[MAX_INEQ_IPM * MAX_VARS];    // column-major
    Scalar h_[MAX_INEQ_IPM];

    // Primal-dual variables
    Scalar x_[MAX_VARS];
    Scalar y_[MAX_EQ];           // equality multipliers
    Scalar z_[MAX_INEQ_IPM];    // inequality multipliers (z >= 0)
    Scalar s_[MAX_INEQ_IPM];    // slacks (s >= 0)

    // Search directions
    Scalar dx_[MAX_VARS];
    Scalar dy_[MAX_EQ];
    Scalar dz_[MAX_INEQ_IPM];
    Scalar ds_[MAX_INEQ_IPM];

    // Affine directions (for Mehrotra corrector)
    Scalar dz_aff_[MAX_INEQ_IPM];
    Scalar ds_aff_[MAX_INEQ_IPM];

    // Residuals
    Scalar r_dual_[MAX_VARS];        // Px + q + A'y + G'z
    Scalar r_eq_[MAX_EQ];            // Ax - b
    Scalar r_ineq_[MAX_INEQ_IPM];   // Gx + s - h
    Scalar mu_;                      // s'z / m

    // KKT system
    Scalar K_[MAX_VARS * MAX_VARS];    // column-major
    Scalar D_[MAX_VARS];               // LDLT diagonal
    Scalar inv_D_[MAX_VARS];           // 1/D (precomputed, avoids VDIV in solve)
    Scalar rhs_[MAX_VARS];             // Schur complement RHS

    // Workspace
    Scalar w_[MAX_INEQ_IPM];          // z/s weights for Schur complement
    Scalar tmp_m_[MAX_INEQ_IPM];      // temporary (m,) vector

    // G sparsity structure (computed once per store_problem call).
    // G is typically ~80% zeros (bound/rate/tube rows have 1-2 nonzeros).
    //
    // Arrays are static (BSS) rather than instance members because CustomIPM
    // is heap-allocated and SRAM2 is only 32KB. Only one instance exists at
    // a time (cold-start pattern), so static storage is safe.
    static constexpr int MAX_G_NNZ = 1024;  // total nonzeros budget
    static int g_nnz_total_;

    // CSC index (column → rows): used by sparse matvecs (compute_residuals, recover)
    static int col_nnz_[MAX_VARS];              // nonzeros per G column
    static int col_start_[MAX_VARS + 1];        // CSC col pointers
    static int g_row_idx_[MAX_G_NNZ];           // row indices (flat CSC)

    // CSR index (row → cols): used by build_kkt row-wise outer product
    static int row_start_[MAX_INEQ_IPM + 1];    // CSR row pointers
    static int g_col_idx_[MAX_G_NNZ];           // column indices (flat CSR)

    // Analyze G_ sparsity after store_problem fills it.
    void analyze_sparsity();

    // Ruiz equilibration: iterative row/column scaling of KKT data.
    // Scales P, G, q, h in-place; stores accumulated scalings in d_x_, d_g_.
    // Called once per store_problem; solution x_ is unscaled after solve().
    void ruiz_equilibrate();

    // Ruiz scaling factors (accumulated across iterations).
    // x_original = d_x_ * x_scaled; z_original = (1/d_g_) * z_scaled.
    Scalar d_x_[MAX_VARS];
    Scalar d_g_[MAX_INEQ_IPM];

    // KKT regularization
    Scalar rho_cur_;                  // primal regularization (P + rho*I)
    Scalar delta_cur_;                // dual regularization: w = z/(s + delta*z) ≤ 1/delta
};

}  // namespace stable_neuron
